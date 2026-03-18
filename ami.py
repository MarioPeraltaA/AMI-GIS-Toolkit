"""Dealing with AMI data.

Author::

    Mario R. Peralta A.

For feedback::

    mario.peralta@ieee.org

"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pandas as pd
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import numpy as np
import folium
import json


class DomainComparable:
    """Shared domain comparison utilities for pandas-backed classes."""

    df: pd.DataFrame | None = None

    def _resolve_other_df(
            self,
            other: object,
            domain_label: str
    ) -> pd.DataFrame:
        """Set up the other table regading type of object."""
        if isinstance(other, GIS):
            table_b = other.gdf

            if table_b is None:
                message_log: str = (
                    "Other GIS object does not have a GeoDataFrame loaded."
                )
                raise ValueError(message_log)

            # Special GIS handling: OBJECTID is actually NISE
            if domain_label == "NISE":
                if "NISE" in table_b.columns:
                    return table_b
                elif "OBJECTID" in table_b.columns:
                    return table_b.rename(columns={"OBJECTID": "NISE"}).copy()
                else:
                    message_log: str = (
                        "GIS GeoDataFrame must contain either "
                        "'NISE' or 'OBJECTID' when domain_label='NISE'."
                    )
                    raise ValueError(message_log)

            return table_b.copy()

        elif isinstance(other, AMI):
            table_b = other.df
        elif isinstance(other, InfoClientManager):
            table_b = other.df
        elif isinstance(other, pd.DataFrame):
            table_b = other.copy()
        else:
            raise TypeError(f"Unsupported type for 'other': {type(other)}")

        if table_b is None:
            raise ValueError("Other object does not have a DataFrame loaded.")

        return table_b.copy()

    def test_domain(
        self,
        other: object,
        domain_label: str = "NodeID",
        self_label: str = "self",
        other_label: str = "other",
        status_col: str = "domain_status",
        drop_null_keys: bool = True
    ) -> pd.DataFrame:
        """Compare identifier domain against another dataset.

        Returns one DataFrame with one row per distinct
        key and membership flags.

        Output columns
        --------------
        - ``<domain_label>``: normalized key
        - ``in_<self_label>``: boolean
        - ``in_<other_label>``: boolean
        - ``<status_col>``:
            * in_both
            * only_in_<self_label>
            * only_in_<other_label>

        """
        table_a = self.df
        if table_a is None:
            raise ValueError("self.df is None. Load data first.")

        table_b = self._resolve_other_df(other, domain_label)

        if domain_label not in table_a.columns:
            raise ValueError(
                f"Column '{domain_label}' not found in self.df"
            )

        if domain_label not in table_b.columns:
            raise ValueError(
                f"Column '{domain_label}' not found in other dataset"
            )

        self_flag = f"in_{self_label}"
        other_flag = f"in_{other_label}"

        a_keys = table_a[[domain_label]].copy()
        b_keys = table_b[[domain_label]].copy()

        if drop_null_keys:
            a_keys = a_keys[a_keys[domain_label].notna()]
            b_keys = b_keys[b_keys[domain_label].notna()]

        a_keys = a_keys.drop_duplicates()
        b_keys = b_keys.drop_duplicates()

        # Normalize key dtype to string for safe comparison across sources
        a_keys[domain_label] = a_keys[domain_label].astype("string")
        b_keys[domain_label] = b_keys[domain_label].astype("string")

        a_keys[self_flag] = True
        b_keys[other_flag] = True

        audit_df = a_keys.merge(
            b_keys,
            on=domain_label,
            how="outer"
        )

        audit_df[self_flag] = (
            audit_df[self_flag]
            .astype('boolean')
            .fillna(False)
        )
        audit_df[other_flag] = (
            audit_df[other_flag]
            .astype('boolean')
            .fillna(False)
        )

        audit_df[status_col] = np.where(
            audit_df[self_flag] & audit_df[other_flag],
            "in_both",
            np.where(
                audit_df[self_flag] & ~audit_df[other_flag],
                f"only_in_{self_label}",
                f"only_in_{other_label}"
            )
        )

        return audit_df


@dataclass
class GIS(DomainComparable, ABC):
    """Abstract GIS contract.

    Field ``OBJECTID`` is actually **NISE** code.

    """

    gdf: gpd.GeoDataFrame | None = None
    df: pd.DataFrame | None = None

    @abstractmethod
    def load_gis(self):
        """Read GIS of each meter."""
        ...


@dataclass
class AMI(DomainComparable, ABC):
    """Abstract AMI device.

    Bear in mind ``LOCALIZACION`` is indeed the
    **NISE** and ``LOCALIZACION_REAL`` the actual location code.

    """

    df: pd.DataFrame | None = None

    @abstractmethod
    def load_data(self):
        """Read smart meters data."""
        ...

    @abstractmethod
    def set_df(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Process some datatype and columns name."""
        ...


@dataclass
class InfoClientManager(DomainComparable, ABC):
    """Database of some utility's customers.

    The customer master data so to speak. It is meant
    to be static but updated.

    """

    df: pd.DataFrame | None = None

    @abstractmethod
    def read_customers_data(self):
        """Load information data of utility's customers."""
        ...

    def reconcile_entities(
        self,
        other: AMI,
        entity_keys: pd.Series,
        entity_label: str = "NISE",
        self_attr: str = "DESCSECTOR",
        other_attr: str = "TIPO_SECTOR",
        date_col: str = "FECHA_LECTURA"
    ) -> pd.DataFrame | None:
        """Compare whether entity attributes agree across two data sources.

        Verify and flag whether two systems report
        the same information about the same entity.

        .. Note::

            If the ``other`` object it is a time series
            the latest value is assumed to be the valid one.

        .. Warning::

            This works better if attributes to compare
            to each other are categorical.

        .. Warning::

            Make sure data type of entity keys is
            consistent with the structures under
            comparison.

        """
        self_df = self.df
        if self_df is None:
            raise ValueError("self.df is None.")

        if not isinstance(other, AMI):
            raise TypeError(f"'other' must be AMI, got {type(other)}")

        other_df = other.df
        if other_df is None:
            raise ValueError("other.df is None.")

        ts_customers = other_df[other_df[entity_label].isin(entity_keys)]

        ami_keys = (
            ts_customers.loc[
                ts_customers.groupby(entity_label)[date_col].idxmax()
            ]
            .set_index(entity_label)[other_attr]
        )

        utility_keys = (
            self_df[self_df[entity_label].isin(entity_keys)]
            .set_index(entity_label)[self_attr]
        )

        cross_source_df = (
            utility_keys.to_frame("self_source")
            .join(ami_keys.rename("other_source"), how="inner")
        )

        cross_source_df["match"] = (
            cross_source_df["self_source"].astype("string")
            == cross_source_df["other_source"].astype("string")
        )
        return cross_source_df


@dataclass
class CNFLCustomers(InfoClientManager):
    """Database of CNFL utility."""

    database_path: str = "./Data/Customers/Infoclientes.txt"
    datatype_path: str = "./Data/Customers/datatype.json"
    columns_dtype: dict = field(init=False)
    df: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.df = self.read_customers_data()

    def set_columns_data_type(self) -> dict | None:
        """Map each column to suitable data type."""
        try:
            with open(self.datatype_path, "r", encoding="utf-8") as f:
                dtype_map = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file '{self.datatype_path}' was not found.")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON from the file. Details: {e}")
            return None
        else:
            self.columns_dtype = dtype_map
            return dtype_map

    def process_info_data(
            self,
            info_df: pd.DataFrame
    ):
        """Clean up and normalize customers dataset."""
        date_cols: list[str] = [
            "FEC_INST_NISE",
            "FECHA_LECTURA",
            "FECHA_HASTA",
            "FECHA_CONEXION_RED",
            "FEC_INST_MED",
            "FECHA_INTER",
        ]

        for col in date_cols:
            info_df[col] = pd.to_datetime(
                info_df[col],
                format="%Y%m%d",
                errors="coerce"
            )

        info_df["NISE"] = (
            info_df["NISE"].astype("string").str.strip()
            .replace(r"^(\d+),0+$", r"\1", regex=True)
        )
        info_df["NISE"] = (
            pd.to_numeric(info_df["NISE"], errors="coerce")
            .astype("Int64")
        )

        return info_df

    def read_customers_data(self) -> pd.DataFrame:
        """Load and process CNFL's Infoclientes file."""
        dtype_map = self.set_columns_data_type()
        info_df = pd.read_csv(
            self.database_path,
            sep=";",
            dtype=dtype_map
        )
        info_df = self.process_info_data(info_df)
        return info_df


@dataclass
class GISCircuit(GIS):
    """Handle and cope with GIS (No electrical modeling)."""

    gis_path: str = "./GIS/*.shp"
    layers: dict[
        str, list[gpd.GeoDataFrame | str]
    ] = field(default_factory=dict)
    per_slice: int | None = None
    customers_layer_name: str = "Loads"

    def __post_init__(self):
        self.load_gis()
        self.paint_layers()
        self.gdf = self.layers[self.customers_layer_name][0]

        if (
            "OBJECTID" in self.gdf.columns
            and "NISE" not in self.gdf.columns
        ):
            self.gdf = self.gdf.rename(columns={"OBJECTID": "NISE"})

        # for compatibility with DomainComparable
        self.df = pd.DataFrame(
            self.gdf.drop(columns="geometry", errors="ignore")
        )

    def set_layer(
        self,
        gdf: gpd.GeoDataFrame,
        rename_cols: dict[str, str] | None = None
    ) -> gpd.GeoDataFrame:
        """Customize a tiny bit current layer."""
        if "IDLOCALIZA" in gdf.columns:
            gdf["IDLOCALIZA"] = pd.to_numeric(
                gdf["IDLOCALIZA"], errors="coerce"
            ).astype("Int64")

        if rename_cols:
            gdf.rename(columns=rename_cols, inplace=True)

        return gdf

    def read_gis(
        self,
        path: str,
        epsg: int = 5367,
        to_local: bool = True,
        rename_cols: dict[str, str] | None = {
            "IDLOCALIZA": "LOCALIZACION_REAL",
            "NUMEROMEDI": "MEDIDOR"
        }
    ) -> gpd.GeoDataFrame | None:
        """Read a GIS file and return a GeoDataFrame with a consistent CRS.

        By default, the data is returned in EPSG:4326 (WGS84), which is
        required for Folium compatibility. If ``to_local=True``, the data
        is reprojected to the specified local EPSG code, typically used
        for utility-scale electrical modeling (e.g., EPSG:5367 in Costa Rica).

        Parameters
        ----------
        path : str
            File path to the GIS data.
        epsg : int, optional
            EPSG code of the local projection to use. Default is 5367.
        to_local : bool, optional
            If True, the GeoDataFrame is returned in the local projection.
            If False (default), it is returned in EPSG:4326.

        Returns
        -------
        geopandas.GeoDataFrame or None
            GeoDataFrame in the requested CRS, or None if reading fails.

        """
        try:
            gdf = gpd.read_file(path)
            gdf = self.set_layer(gdf, rename_cols=rename_cols)
        except Exception as e:
            print(f"Error reading GIS file '{path}': {e}")
            return None

        current_epsg = gdf.crs.to_epsg()
        if current_epsg is None:
            print(
                f"Warning: No CRS found in '{path}'. "
                "Returning raw GeoDataFrame."
            )
            return gdf

        target_epsg = epsg if to_local else 4326
        if current_epsg != target_epsg:
            return gdf.to_crs(epsg=target_epsg)
        return gdf

    def load_gis(self):
        """Read all GIS layers.

        If :py:attr:`GISCircuit.per_slice` is ``None``
        it will take the whole data set.

        """
        shapefiles: list[str] = glob.glob(self.gis_path)

        for shp_path in shapefiles:
            layer_name = shp_path.split("/")[-1].split(".")[0]
            try:
                gdf = self.read_gis(shp_path)
                if gdf is None or not gdf.shape[0]:
                    raise ValueError(f"No values in {layer_name}")
            except ValueError as e:
                print(f"EmptyGeoDataFrame: {e}.")
                continue
            else:
                if self.per_slice:
                    n_sample = len(gdf) * self.per_slice // 100
                    self.layers[layer_name] = [gdf[:n_sample]]
                else:
                    self.layers[layer_name] = [gdf]

    def paint_layers(
            self,
            seed: int = 7859
    ) -> dict[str, list[str]]:
        """Assign eye-cathing color to each layer.

        Uses ``rng.shuffle`` instead of ``rng.integers`` to make sure
        all colors are different.

        """
        lib_colors = list(plt.cm.colors.cnames)
        size = len(self.layers)
        rng = np.random.default_rng(seed=seed)
        rnd_ints = np.arange(0, len(lib_colors))
        rng.shuffle(rnd_ints)
        colors = [lib_colors[c] for c in rnd_ints[:size]]

        for i, gdf_list in enumerate(self.layers.values()):
            gdf_list.append(colors[i])

        return self.layers

    def explore_network(self) -> folium.Map:
        """Map of the circuit.

        Go easy on the number of elements.

        """
        ckt_map = folium.Map(
            crs="EPSG3857",
            zoom_start=15,
            control_scale=True,
            tiles="cartodbpositron"
        )

        for name, layer in self.layers.items():
            gdf, color = layer
            gdf.explore(
                m=ckt_map,
                popup=True,
                tooltip=True,
                name=name,
                color=color,
                show=False
            )

        folium.TileLayer(
            "Cartodb dark_matter", show=False
        ).add_to(ckt_map)
        folium.LayerControl().add_to(ckt_map)
        return ckt_map


@dataclass
class ConsumptionData(AMI):
    r"""Data structure for managing and analyzing energy consumption profiles.

    This class handles energy-related measurements obtained from
    AMI (Advanced Metering Infrastructure) systems. It gathers and
    organizes instantaneous power, voltage data, and daily energy
    consumption increments (:math:`\Delta` Energy).

    **Sampling Frequency:**
    The intended sampling frequency is **one measurement per day**.
    However, actual intervals may vary depending on the meter
    configuration and data availability.

    Attributes
    ----------
    data_path : str
        Default directory path for storing or loading energy consumption data.
    gis : GISCircuit | None
        Optional GIS circuit object associated with the dataset, used for
        spatial analysis or network mapping.
    df : pandas.DataFrame | None
        Raw AMI dataset containing power, voltage, and energy measurements.
    ene_data : pandas.DataFrame | None
        ...
    mde_data : pandas.DataFrame | None
        ...
    power_data : pandas.DataFrame | None
        Instantaneous active kW, reactive kVAR and mag. apparent kVA
        power either demanded from the network or injected to it.
    voltage_data : pandas.DataFrame | None
        ...
    current_data : pandas.DataFrame | None
        ...
    pf_data : pandas.DataFrame | None
        ...
    energy_df : pandas.DataFrame | None
        Processed daily energy increments (:math:`\Delta` kWh), suitable for
        time-series analysis and load profiling.
    ami_gdf : geopandas.GeoDataFrame | None
        Geospatial representation of AMI devices and measurements, useful for
        visualization and spatial queries.

    """

    data_path: str = "./Data/Consumption"
    gis: GISCircuit | None = None
    df: pd.DataFrame | None = None
    ene_data: pd.DataFrame | None = None
    mde_data: pd.DataFrame | None = None
    power_data: pd.DataFrame | None = None
    voltage_data: pd.DataFrame | None = None
    current_data: pd.DataFrame | None = None
    pf_data: pd.DataFrame | None = None
    energy_df: pd.DataFrame | None = None
    ami_gdf: gpd.GeoDataFrame | None = None

    def __post_init__(self):
        """Set and put data structures."""
        self.load_data()
        self.set_energy_df()
        if self.gis:
            _ = self.put_geometry()

    def set_df(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """Process datatype and add hour column."""
        df["CONTADOR"] = df["CONTADOR"].astype(str).str.lower()

        df["LOCALIZACION"] = (
            df["LOCALIZACION"].astype("string").str.strip()
            .replace(r"^(\d+),0+$", r"\1", regex=True)
        )
        df["NISE"] = (
            pd.to_numeric(df["LOCALIZACION"], errors="coerce")
            .astype("Int64")
        )
        df.drop(columns=["LOCALIZACION"], inplace=True)

        df["VALOR_LECTURA"] = pd.to_numeric(
            df["VALOR_LECTURA"], errors="coerce"
        )
        df["MEDIDOR"] = (
            pd.to_numeric(df["MEDIDOR"], errors="coerce")
            .astype("Int64")
        )
        df["LOCALIZACION_REAL"] = pd.to_numeric(
            df["LOCALIZACION_REAL"], errors="coerce"
        ).astype("Int64")

        df["FECHA_LECTURA"] = df["FECHA_LECTURA_REAL"]

        seconds_s = (
            df["FECHA_LECTURA"].dt.hour.fillna(0) * 3600.0
            + df["FECHA_LECTURA"].dt.minute.fillna(0) * 60.0
            + df["FECHA_LECTURA"].dt.second.fillna(0)
        )
        df["Hour"] = seconds_s / 3600.0

        return df

    def split_ene_mde(
            self,
            df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Energetic and Max. Power Demand flow structures."""
        ene_mde_df = df[
            (df.OPERACION.isin(["Delivered", "Received"]))
            & (df.CONTADOR == "total")
            & (df.TIPO_CONSUMO != "FPO")
        ]
        ene_df = ene_mde_df[
            ene_mde_df.TIPO_CONSUMO
            .astype(str)
            .str
            .contains("ENE", na=False)
        ]
        mde_df = ene_mde_df[
            ene_mde_df.TIPO_CONSUMO
            .astype(str)
            .str
            .contains("MDE", na=False)
        ]
        return ene_df.reset_index(drop=True), mde_df.reset_index(drop=True)

    def get_power(
        self,
        abc_df: pd.DataFrame
    ) -> pd.DataFrame:
        return abc_df[
            abc_df.TIPO_CONSUMO.isin(
                ["kW", "kVAR", "kVA"]
            )
        ].reset_index(drop=True)

    def get_voltage(
        self,
        abc_df: pd.DataFrame
    ) -> pd.DataFrame:
        return abc_df[
            abc_df.TIPO_CONSUMO.isin(
                ["Voltage", "Voltage Angle"]
            )
        ].reset_index(drop=True)

    def get_current(
        self,
        abc_df: pd.DataFrame
    ) -> pd.DataFrame:
        return abc_df[
            abc_df.TIPO_CONSUMO.isin(
                ["Current", "Current Angle"]
            )
        ].reset_index(drop=True)

    def get_pf(
        self,
        abc_df: pd.DataFrame
    ) -> pd.DataFrame:
        return abc_df[
            abc_df.TIPO_CONSUMO.isin(
                ["Power Factor", "Power Factor Angle"]
            )
        ].reset_index(drop=True)

    def load_data(self):
        """Read and set raw daily dirty data.

        Regarding meter function (SGDA) filter transformer meters out:

            Either None or Bidireccional
        ENE and MDE either Delivered or Received but without FPO:

            CONTADOR (total) OPERACION (Delivered, Received)
            TIPO_CONSUMO (filter FPO out).

        Phase analysis:

            CONTADOR (a, b, c) and OPERACION (A, B, C) gets

                - kW, kVAR, kVA
                - Voltage, Voltage Angle
                - Current, Current Angle
                - Power Factor, Power Factor Angle

        """
        df = pd.read_parquet(self.data_path)
        df = self.set_df(df)
        self.df = df.copy()

        df = df[df.SGDA.isin([None, "Bidireccional"])]

        self.ene_data, self.mde_data = self.split_ene_mde(df)

        phases: set[str] = {"a", "b", "c", "A", "B", "C"}
        abc_df = df[(df.CONTADOR.isin(phases)) & (df.OPERACION.isin(phases))]
        self.power_data = self.get_power(abc_df)
        self.voltage_data = self.get_voltage(abc_df)
        self.current_data = self.get_current(abc_df)
        self.pf_data = self.get_pf(abc_df)

    def set_energy_df(self):
        """Spread and sort by date Energy structure."""
        if self.ene_data is None:
            raise ValueError("ene_data is not set.")

        kwh_df = (
            self.ene_data
            .pivot_table(
                index=["MEDIDOR", "FECHA_LECTURA", "Hour"],
                columns="OPERACION",
                values="VALOR_LECTURA",
                aggfunc="first"
            )
            .reset_index()
        )

        kwh_df = kwh_df.sort_values(by=["MEDIDOR", "FECHA_LECTURA"])
        kwh_df["Period"] = kwh_df.groupby("MEDIDOR")["FECHA_LECTURA"].diff()

        customers = kwh_df.groupby("MEDIDOR")
        kwh_df["Daily Sent"] = customers["Delivered"].diff()
        kwh_df["Daily Gotten"] = customers["Received"].diff()

        self.energy_df = kwh_df

    def n_phases(
            self
    ) -> pd.DataFrame:
        """Count number of phases of each meter."""
        if self.voltage_data is None:
            raise ValueError("voltage_data not loaded.")

        return (
            self.voltage_data.groupby("MEDIDOR")["CONTADOR"]
            .nunique()
            .reset_index(name="n_phases")
        )

    def put_geometry(
        self,
        node_col_label: str = "MEDIDOR",
        geo_of_col: str = "LOCALIZACION_REAL"
    ) -> gpd.GeoDataFrame:
        """Map meter location to actual geometry point.

        As long as ami device location is not None
        and such location exists in GIS data.

        .. Note::

            Drop duplicates between ``MEDIDOR`` and ``NISE``
            so that it is force to map one meter to one
            customer NISE although one customer may have
            multiple meters, this is only for visualization
            purposes.

        """
        if self.gis is None:
            raise ValueError("gis is None.")

        gdf = self.gis.layers["Loads"][0]
        to_geom = dict(zip(gdf[geo_of_col], gdf["geometry"]))

        ami_df = (
            self.df[[node_col_label, geo_of_col]]
            .drop_duplicates()
        )

        ami_df = ami_df[ami_df[geo_of_col].notna()].reset_index(drop=True)
        ami_df["geometry"] = ami_df[geo_of_col].map(to_geom)
        ami_df = ami_df[ami_df["geometry"].notna()].reset_index(drop=True)

        ami_gdf = gpd.GeoDataFrame(
            ami_df, geometry="geometry", crs="EPSG:5367"
        )
        self.ami_gdf = ami_gdf
        return ami_gdf


@dataclass
class VoltageData(AMI):
    """Data structure for managing and analyzing daily voltage profiles.

    This class is designed to handle voltage measurements collected from
    AMI (Advanced Metering Infrastructure) systems. It focuses on daily
    profiles, typically sampled at regular intervals.

    **Expected Sampling Frequency:**
    Voltage measurements are generally expected every 5, 10, or 15 minutes.
    However, only a limited number of key customers may have meters configured
    to record voltage at such high temporal resolution. Further more
    it looks like most of them measure only at one phase.

    Attributes
    ----------
    data_path : str
        Default directory path for storing or loading voltage data files.
    gis : GISCircuit | None
        Optional GIS circuit object associated with the dataset. Used for
        spatial analysis or network mapping.
    df : pandas.DataFrame | None
        Raw AMI dataset containing voltage measurements and metadata.
    voltage_df : pandas.DataFrame | None
        Processed voltage data, typically cleaned
        and time-indexed for analysis.
    ami_gdf : geopandas.GeoDataFrame | None
        Geospatial representation of AMI devices and measurements, useful for
        visualization and spatial queries.

    """

    data_path: str = "./Data/Voltages"
    phase_vals: list[str] = field(
        default_factory=lambda: ["Phase A Average RMS Voltage"]
    )
    gis: GISCircuit | None = None
    df: pd.DataFrame | None = None
    voltage_data: pd.DataFrame | None = None
    voltage_df: pd.DataFrame | None = None
    ami_gdf: gpd.GeoDataFrame | None = None

    def __post_init__(self):
        self.load_data()
        self.set_voltage_df()

    def set_df(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Process datatype."""
        df["LOCALIZACION"] = (
            df["LOCALIZACION"].astype("string").str.strip()
            .replace(r"^(\d+),0+$", r"\1", regex=True)
        )
        df["NISE"] = (
            pd.to_numeric(df["LOCALIZACION"], errors="coerce")
            .astype("Int64")
        )
        df.drop(columns=["LOCALIZACION"], inplace=True)

        df["VALOR_LECTURA"] = (
            pd.to_numeric(df["VALOR_LECTURA"], errors="coerce")
        )
        df["MEDIDOR"] = (
            pd.to_numeric(df["MEDIDOR"], errors="coerce")
            .astype("Int64")
        )
        df["FECHA_LECTURA"] = df["FECHA_LECTURA_REAL"]

        df["UNIDAD"] = df["UNIDAD"].astype(str)
        df = df[df["UNIDAD"].isin(self.phase_vals)]

        df = df.sort_values(
            by=["MEDIDOR", "FECHA_LECTURA"]
        ).reset_index(drop=True)
        return df

    def load_data(self):
        """Read and set raw daily dirty data."""
        df = pd.read_parquet(self.data_path)
        df = self.set_df(df)
        self.df = df.copy()
        self.voltage_data = df[
            ["MEDIDOR", "NISE", "FECHA_LECTURA", "UNIDAD", "VALOR_LECTURA"]
        ]

    def set_voltage_df(self):
        """Reframe dataset for further analysis."""
        if self.voltage_data is None:
            raise ValueError("voltage_data not loaded.")

        v_df = self.voltage_data[
            ["MEDIDOR", "FECHA_LECTURA", "UNIDAD", "VALOR_LECTURA"]
        ].copy()

        v_df.rename(columns={
            "MEDIDOR": "node",
            "FECHA_LECTURA": "ts",
            "UNIDAD": "phase",
            "VALOR_LECTURA": "V"
        }, inplace=True)

        self.voltage_df = v_df


@dataclass
class PowerData(AMI):
    """Data structure for managing and analyzing daily power profiles.

    This class handles active and reactive power measurements collected from
    AMI (Advanced Metering Infrastructure) systems at high temporal resolution
    (typically every 10 or 15 minutes). Energy and power data are
    organized into separate structures for active and
    reactive components, enabling flexible
    analysis of load profiles and network behavior.

    **Attributes**
    ----------
    data_path : str
        Default directory path for storing or loading power data files.
    gis : GISCircuit | None
        Optional GIS circuit object associated with the dataset, used for
        spatial analysis or network mapping.
    df : pandas.DataFrame | None
        Raw AMI dataset containing both active and reactive power measurements.
    kwh_data : pandas.DataFrame | None
        Raw energy data (kWh) representing active energy consumption over time.
    kvarh_data : pandas.DataFrame | None
        Raw energy data (kVArh) representing reactive energy measurements.
    active_df : pandas.DataFrame | None
        Processed active power dataset, typically cleaned and time-indexed for
        load profile analysis.
    reactive_df : pandas.DataFrame | None
        Processed reactive power dataset, typically cleaned and
        time-indexed for network behavior analysis.
    ami_gdf : geopandas.GeoDataFrame | None
        Geospatial representation of AMI devices and measurements, useful for
        visualization and spatial queries.

    """

    data_path: str = "./Data/Power"
    gis: GISCircuit | None = None
    df: pd.DataFrame | None = None
    kwh_data: pd.DataFrame | None = None
    kvarh_data: pd.DataFrame | None = None
    active_df: pd.DataFrame | None = None
    reactive_df: pd.DataFrame | None = None
    ami_gdf: gpd.GeoDataFrame | None = None

    def __post_init__(self):
        self.load_data()
        self.set_active_df()
        self.set_reactive_df()

    def set_df(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Process datatype."""
        df["LOCALIZACION"] = (
            df["LOCALIZACION"].astype("string").str.strip()
            .replace(r"^(\d+),0+$", r"\1", regex=True)
        )
        df["NISE"] = (
            pd.to_numeric(df["LOCALIZACION"], errors="coerce")
            .astype("Int64")
        )
        df.drop(columns=["LOCALIZACION"], inplace=True)

        df["VALOR_LECTURA"] = pd.to_numeric(
            df["VALOR_LECTURA"], errors="coerce"
        )
        df["MEDIDOR"] = (
            pd.to_numeric(df["MEDIDOR"], errors="coerce")
            .astype("Int64")
        )
        df["LOCALIZACION_REAL"] = pd.to_numeric(
            df["LOCALIZACION_REAL"], errors="coerce"
        ).astype("Int64")
        df["FECHA_LECTURA"] = df["FECHA_LECTURA_REAL"]

        df = df.sort_values(
            by=["MEDIDOR", "FECHA_LECTURA"]
        ).reset_index(drop=True)
        return df

    def split_power(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Retrieve kWh and kVARh."""
        df = df[df.OPERACION.isin(["Delivered", "Received"])]
        active_df = df[df.UNIDAD == "kWh"]
        reactive_df = df[df.UNIDAD == "kVARh"]
        return active_df, reactive_df

    def load_data(self):
        """Read and set raw daily dirty data."""
        df = pd.read_parquet(self.data_path)
        df = self.set_df(df)
        self.df = df.copy()

        df = df[df.SGDA.isin([None, "Bidireccional"])]
        kwh_data, kvarh_data = self.split_power(df)

        columns = [
            "MEDIDOR",
            "NISE",
            "LOCALIZACION_REAL",
            "OPERACION",
            "INTERVALO",
            "FECHA_LECTURA",
            "ENERGIA",
            "DEMANDA"
        ]
        self.kwh_data = kwh_data[columns].reset_index(drop=True)
        self.kvarh_data = kvarh_data[columns].reset_index(drop=True)

    def set_active_df(self):
        """Restructure original data set.

        Both ``Delivered`` and ``Received`` fields
        become nested in either ``ENERGIA`` and ``DEMANDA``.
        Sort by date and assume everyone sends and
        receives power but in case they don't then it
        is either zero or filled up with `NaN`.

        .. Note::

            Columns are labeled from load's perspective.

        """
        if self.kwh_data is None:
            raise ValueError("kwh_data not loaded.")

        active_df = (
            self.kwh_data
            .pivot_table(
                index=["MEDIDOR", "FECHA_LECTURA"],
                columns="OPERACION",
                values=["ENERGIA", "DEMANDA"],
                aggfunc="first"
            )
            .reset_index()
        )

        rename_dict = {
            ("MEDIDOR", ""): ("node", ""),
            ("FECHA_LECTURA", ""): ("ts", ""),
            ("DEMANDA", "Delivered"): ("P", "Pdem"),
            ("DEMANDA", "Received"): ("P", "Pgen"),
            ("ENERGIA", "Delivered"): ("E", "Edem"),
            ("ENERGIA", "Received"): ("E", "Egen"),
        }

        active_df.columns = active_df.columns.map(
            lambda c: rename_dict.get(c, c)
        )
        self.active_df = active_df

    def set_reactive_df(self):
        """Restructure original data set.

        Both ``Delivered`` and ``Received`` fields
        become nested in either ``ENERGIA`` and ``DEMANDA``.
        Sort by date and assume everyone sends and
        receives power but in case they don't then it
        is either zero or filled up with `NaN`.

        """
        if self.kvarh_data is None:
            raise ValueError("kvarh_data not loaded.")

        reactive_df = (
            self.kvarh_data
            .pivot_table(
                index=["MEDIDOR", "FECHA_LECTURA"],
                columns="OPERACION",
                values=["ENERGIA", "DEMANDA"],
                aggfunc="first"
            )
            .reset_index()
        )

        rename_dict = {
            ("MEDIDOR", ""): ("node", ""),
            ("FECHA_LECTURA", ""): ("ts", ""),
            ("DEMANDA", "Delivered"): ("Q", "Qdem"),
            ("DEMANDA", "Received"): ("Q", "Qgen"),
            ("ENERGIA", "Delivered"): ("E", "Edem"),
            ("ENERGIA", "Received"): ("E", "Egen"),
        }

        reactive_df.columns = reactive_df.columns.map(
            lambda c: rename_dict.get(c, c)
        )
        self.reactive_df = reactive_df

    def flat_df(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Turn nested structure into flat one.

        By kicking out falsy columns: None, False, ""
        for top level columns and join those with
        inner columns with underscore e.g::

            df[[("node", ""), ("ts", ""), ("P", "Pdem")]]

        Becomes::

            df[["node", "ts", "P_Pdem"]]

        """
        flat_df = df.copy()
        flat_df.columns = [
            "_".join(filter(None, col)) if isinstance(col, tuple) else col
            for col in flat_df.columns
        ]
        return flat_df.copy()


@dataclass
class ScenariosManager(ABC):
    """Use data to make up scenarios.

    Smart City and Microgrid loadshapes.
    Vitual potential realities so to speak, in case of
    a University campus these could be the scenarios:

        - Weekdays of each academic season.
        - Weekdays on vacations.
        - Weekends throughout the year.
        - The event of maximum possible demand.

    """

    df: pd.DataFrame
    pdata: PowerData | None = None
    vdata: VoltageData | None = None
    cdata: ConsumptionData | None = None


@dataclass
class CityScenarios(ScenariosManager):
    """Smart City and Microgrid loadshapes."""

    season_curves: pd.DataFrame | None = None
    weekends_curve: pd.DataFrame | None = None
    sdata: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.sdata = self.city_scenarios(self.df)

    def set_seasons(
        self,
        df: pd.DataFrame,
        seasons_map: dict[int, str] = {
            1: "III",
            2: "III",
            3: "I",
            4: "I",
            5: "I",
            6: "I",
            7: "V",
            8: "II",
            9: "II",
            10: "II",
            11: "II",
            12: "V"
        }
    ) -> pd.DataFrame:
        """Classify demand nature based on seasons.

        Let default seasons be I, II, III, v then::

            seasons = {
                1: "III",
                2: "III",
                3: "I",
                4: "I",
                5: "I",
                6: "I",
                7: "V",
                8: "II",
                9: "II",
                10: "II",
                11: "II",
                12: "V"
            }

        Where each number represents the month respectively.

        .. Note::

            Seasons only consider weekdays. To play around with
            weekends see :py:meth:`ScenariosManager.weekend_curve`

        """
        df = df.copy()
        df["season"] = df["ts"].dt.month.map(seasons_map)
        return df

    def city_scenarios(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Set labels up of an academic Microgrid.

        Retain only those days with 96 samples so
        acrossed average demand is consistent over time.

        """
        df = df.copy()
        df["day"] = df["ts"].dt.weekday.map(
            lambda x: "weekend" if x >= 5 else "weekday"
        )
        df = self.set_seasons(df)
        df["date"] = df["ts"].dt.date

        day_counts = (
            df.groupby(["node", "date"])
            .size()
            .rename("samples")
            .reset_index()
        )

        valid_days = day_counts[day_counts["samples"] == 96]

        df = df.merge(
            valid_days[["node", "date"]],
            on=["node", "date"],
            how="inner"
        )
        return df

    def get_weekends_curve(
        self,
        power: str = "P"
    ) -> pd.DataFrame:
        """Retrieve weekends and compute average throughout time."""
        wend = self.sdata[self.sdata["day"] == "weekend"].copy()
        wend["timestep"] = wend.groupby(["node", "date"]).cumcount()

        wend_avg_shape = (
            wend.groupby(["node", "timestep"])[power]
            .mean()
            .reset_index()
            .sort_values(["node", "timestep"])
        )

        self.weekends_curve = wend_avg_shape
        return wend_avg_shape

    def avg_curves(
        self,
        power: str = "P"
    ) -> pd.DataFrame:
        """Compute average load shape of each weekday season."""
        wday = self.sdata[self.sdata["day"] == "weekday"].copy()
        wday["timestep"] = (
            wday.groupby(["season", "node", "date"])
            .cumcount()
        )

        wday_avg_shape = (
            wday.groupby(["season", "node", "timestep"])[power]
            .mean()
            .reset_index()
            .sort_values(["season", "node", "timestep"])
        )

        self.season_curves = wday_avg_shape
        return wday_avg_shape


if __name__ == "__main__":
    data = ConsumptionData()
    domain_audit = data.test_domain(
        CNFLCustomers(),
        domain_label="NISE",
        self_label="consumption",
        other_label="customers"
    )
    print(domain_audit.head())
