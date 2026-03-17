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
                message_log = (
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
                    raise ValueError(
                        "GIS GeoDataFrame must contain either 'NISE' or 'OBJECTID' "
                        "when domain_label='NISE'."
                    )

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

        Returns one DataFrame with one row per distinct key and membership flags.

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
            raise ValueError(f"Column '{domain_label}' not found in self.df")

        if domain_label not in table_b.columns:
            raise ValueError(f"Column '{domain_label}' not found in other dataset")

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

        audit_df[self_flag] = audit_df[self_flag].fillna(False).astype(bool)
        audit_df[other_flag] = audit_df[other_flag].fillna(False).astype(bool)

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
        """Compare whether entity attributes agree across two data sources."""
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

    def process_info_data(self, info_df: pd.DataFrame):
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
        info_df["NISE"] = pd.to_numeric(info_df["NISE"], errors="coerce").astype("Int64")

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

        if "OBJECTID" in self.gdf.columns and "NISE" not in self.gdf.columns:
            self.gdf = self.gdf.rename(columns={"OBJECTID": "NISE"})

        # for compatibility with DomainComparable
        self.df = pd.DataFrame(self.gdf.drop(columns="geometry", errors="ignore"))

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
        """Read a GIS file and return a GeoDataFrame with a consistent CRS."""
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
        """Read all GIS layers."""
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

    def paint_layers(self, seed: int = 7859) -> dict[str, list[str]]:
        """Assign eye-catching color to each layer."""
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
        """Map of the circuit."""
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

        folium.TileLayer("Cartodb dark_matter", show=False).add_to(ckt_map)
        folium.LayerControl().add_to(ckt_map)
        return ckt_map


@dataclass
class ConsumptionData(AMI):
    """Data structure for managing and analyzing energy consumption profiles."""

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
        df["NISE"] = pd.to_numeric(df["LOCALIZACION"], errors="coerce").astype("Int64")
        df.drop(columns=["LOCALIZACION"], inplace=True)

        df["VALOR_LECTURA"] = pd.to_numeric(df["VALOR_LECTURA"], errors="coerce")
        df["MEDIDOR"] = pd.to_numeric(df["MEDIDOR"], errors="coerce").astype("Int64")
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
        ene_df = ene_mde_df[ene_mde_df.TIPO_CONSUMO.astype(str).str.contains("ENE", na=False)]
        mde_df = ene_mde_df[ene_mde_df.TIPO_CONSUMO.astype(str).str.contains("MDE", na=False)]
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
        """Read and set raw daily dirty data."""
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

    def n_phases(self) -> pd.DataFrame:
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
        """Map meter location to actual geometry point."""
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
    """Data structure for managing and analyzing daily voltage profiles."""

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
        df["NISE"] = pd.to_numeric(df["LOCALIZACION"], errors="coerce").astype("Int64")
        df.drop(columns=["LOCALIZACION"], inplace=True)

        df["VALOR_LECTURA"] = pd.to_numeric(df["VALOR_LECTURA"], errors="coerce")
        df["MEDIDOR"] = pd.to_numeric(df["MEDIDOR"], errors="coerce").astype("Int64")
        df["FECHA_LECTURA"] = df["FECHA_LECTURA_REAL"]

        df["UNIDAD"] = df["UNIDAD"].astype(str)
        df = df[df["UNIDAD"].isin(self.phase_vals)]

        df = df.sort_values(by=["MEDIDOR", "FECHA_LECTURA"]).reset_index(drop=True)
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
    """Data structure for managing and analyzing daily power profiles."""

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
        df["NISE"] = pd.to_numeric(df["LOCALIZACION"], errors="coerce").astype("Int64")
        df.drop(columns=["LOCALIZACION"], inplace=True)

        df["VALOR_LECTURA"] = pd.to_numeric(df["VALOR_LECTURA"], errors="coerce")
        df["MEDIDOR"] = pd.to_numeric(df["MEDIDOR"], errors="coerce").astype("Int64")
        df["LOCALIZACION_REAL"] = pd.to_numeric(
            df["LOCALIZACION_REAL"], errors="coerce"
        ).astype("Int64")
        df["FECHA_LECTURA"] = df["FECHA_LECTURA_REAL"]

        df = df.sort_values(by=["MEDIDOR", "FECHA_LECTURA"]).reset_index(drop=True)
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
        """Restructure original data set."""
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

        active_df.columns = active_df.columns.map(lambda c: rename_dict.get(c, c))
        self.active_df = active_df

    def set_reactive_df(self):
        """Restructure original data set."""
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

        reactive_df.columns = reactive_df.columns.map(lambda c: rename_dict.get(c, c))
        self.reactive_df = reactive_df

    def flat_df(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Turn nested structure into flat one."""
        flat_df = df.copy()
        flat_df.columns = [
            "_".join(filter(None, col)) if isinstance(col, tuple) else col
            for col in flat_df.columns
        ]
        return flat_df.copy()


@dataclass
class ScenariosManager(ABC):
    """Use data to make up scenarios."""

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
        """Classify demand nature based on seasons."""
        df = df.copy()
        df["season"] = df["ts"].dt.month.map(seasons_map)
        return df

    def city_scenarios(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Set labels up of an academic Microgrid."""
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

    def get_weekends_curve(self, power: str = "P") -> pd.DataFrame:
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

    def avg_curves(self, power: str = "P") -> pd.DataFrame:
        """Compute average load shape of each weekday season."""
        wday = self.sdata[self.sdata["day"] == "weekday"].copy()
        wday["timestep"] = wday.groupby(["season", "node", "date"]).cumcount()

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
