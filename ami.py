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


@dataclass
class GIS(ABC):
    """Abstract GIS contract.

    Field ``OBJECTID`` is actually **NISE** code.

    """

    gdf: gpd.GeoDataFrame | None = None

    @abstractmethod
    def load_gis(self):
        """Read *GIS* of each meter."""
        ...

    def test_domain(
            self,
            other: object,
            domain_label: str = "NodeID"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compare data classes.

        Run a diagnosis of inconsistencies
        between two data classes and extract them all.

        +------------+-------------------------------+
        | state      | meaning                       |
        +============+===============================+
        | both       | consistent identifiers        |
        +------------+-------------------------------+
        | left_only  | Table A - only identifiers    |
        +------------+-------------------------------+
        | right_only | Table B - only identifiers    |
        +------------+-------------------------------+

        .. warning::

            Make sure domain field is label the same
            in both structures.

        """
        col = domain_label
        table_a = self.gdf
        if isinstance(other, GIS) and col == "NISE":
            table_b = other.gdf.copy(deep=True)
            table_b.rename(
                columns={"OBJECTID": domain_label},
                inplace=True
            )
        elif isinstance(other, AMI):
            table_b = other.df.copy(deep=True)
        elif isinstance(other, pd.DataFrame):
            table_b = other.copy(deep=True)
        elif isinstance(other, InfoClientManager):
            table_b = other.df.copy(deep=True)

        table_a_keys = table_a[[col]].drop_duplicates()
        table_b_keys = table_b[[col]].drop_duplicates()
        diff = (
            table_a_keys
            .merge(table_b_keys, on=col, how='outer', indicator=True)
        )
        in_a_not_b = diff.loc[diff['_merge'] == 'left_only', col]
        in_b_not_a = diff.loc[diff['_merge'] == 'right_only', col]
        return (
            in_a_not_b,
            in_b_not_a
        )


@dataclass
class AMI(ABC):
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
    def set_df(self):
        """Process some datatype and columns name."""
        ...

    def test_domain(
            self,
            other: object,
            domain_label: str = "NodeID"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compare data classes.

        Run a diagnosis of inconsistencies
        between two data classes and extract them all.

        +------------+-------------------------------+
        | state      | meaning                       |
        +============+===============================+
        | both       | consistent identifiers        |
        +------------+-------------------------------+
        | left_only  | Table A - only identifiers    |
        +------------+-------------------------------+
        | right_only | Table B - only identifiers    |
        +------------+-------------------------------+

        .. warning::

            Make sure domain field is label the same
            in both structures.

        """
        col = domain_label
        table_a = self.df
        if isinstance(other, GIS) and col == "NISE":
            # table_b = other.layers['Loads'][0]
            table_b = other.gdf.copy(deep=True)
            table_b.rename(
                columns={"OBJECTID": domain_label},
                inplace=True
            )
        elif isinstance(other, AMI):
            table_b = other.df.copy(deep=True)
        elif isinstance(other, pd.DataFrame):
            table_b = other.copy(deep=True)
        elif isinstance(other, InfoClientManager):
            table_b = other.df.copy(deep=True)

        table_a_keys = table_a[[col]].drop_duplicates()
        table_b_keys = table_b[[col]].drop_duplicates()
        diff = (
            table_a_keys
            .merge(table_b_keys, on=col, how='outer', indicator=True)
        )
        in_a_not_b = diff.loc[diff['_merge'] == 'left_only', col]
        in_b_not_a = diff.loc[diff['_merge'] == 'right_only', col]
        return (
            in_a_not_b,
            in_b_not_a
        )


@dataclass
class InfoClientManager(ABC):
    """Database of some utility's costumers."""

    df: pd.DataFrame | None = None

    @abstractmethod
    def read_costumers_data(
        self
    ):
        """Load information data of utility's costumers."""
        ...


@dataclass
class CNFLCostumers(InfoClientManager):
    """Database of CNFL utility."""

    database_path: str = "./Data/Costumers/Infoclientes.txt"
    datatype_path: str = "./Data/Costumers/datatype.json"
    columns_dtype: dict = field(init=False)
    df: pd.DataFrame = field(init=False)

    def __post_init__(
            self
    ):
        """Instantiate company's costumers database."""
        self.df = self.read_costumers_data()

    def set_columns_data_type(
            self,
    ) -> dict | None:
        """Map each column to suitable data type."""
        try:
            with open(self.datatype_path, "r", encoding="utf-8") as f:
                dtype_map = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file '{self.datatype_path}' was not found.")
            return
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON from the file. Details: {e}")
            return
        else:
            self.columns_dtype = dtype_map
            return dtype_map

    def process_info_data(
            self,
            info_df: pd.DataFrame
    ):
        """Clean up and normalize costumers dataset."""
        date_cols: list[str] = [
            "FEC_INST_NISE",
            "FECHA_LECTURA",
            # "FECHA_DESDE",
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

        info_df['NISE'] = (
            info_df['NISE'].str.strip()
            .replace(r"^(\d+),0+$", r"\1", regex=True)
            .astype("Int64")
        )
        return info_df

    def read_costumers_data(
            self
    ) -> pd.DataFrame:
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
    layers: dict[str, list[gpd.GeoDataFrame, str]] = field(
        default_factory=dict
    )
    per_slice: int | None = None
    costumers_layer_name: str = "Loads"

    def __post_init__(
            self
    ):
        """Initialize GIS data structure."""
        self.load_gis()
        self.paint_layers()
        self.gdf = self.layers[self.costumers_layer_name][0]
        self.gdf.rename(
            columns={"OBJECTID": "NISE"}, inplace=True
        )

    def set_layer(
            self,
            gdf: gpd.GeoDataFrame,
            rename_cols: dict[str, str] | None = None
    ) -> gpd.GeoDataFrame:
        """Customize a tiny bit current layer."""
        if "IDLOCALIZA" in gdf.columns:
            gdf['IDLOCALIZA'] = gdf['IDLOCALIZA'].astype("Int64")
        if rename_cols:
            gdf.rename(
                columns=rename_cols, inplace=True
            )
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
            logg: str = (
                f"Warning: No CRS found in '{path}'. "
                "Returning raw GeoDataFrame."
            )
            print(logg)
            return gdf

        target_epsg = epsg if to_local else 4326
        if current_epsg != target_epsg:
            return gdf.to_crs(epsg=target_epsg)
        return gdf

    def load_gis(
            self
    ):
        """Read all GIS layers.

        If :py:attr:`GISCircuit.per_slice` is ``None``
        it will take the whole data set.

        """
        shapefiles: list[str] = glob.glob(self.gis_path)

        for shp_path in shapefiles:
            layer_name = shp_path.split("/")[-1].split(".")[0]
            try:
                gdf = self.read_gis(shp_path)
                if not gdf.shape[0]:
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
    ) -> dict[str, list[str, str]]:
        """Assign eye-cathing color to each layer.

        Uses ``rng.shuffle`` instead of ``rng.integers`` to make sure
        all colors are different.

        """
        # Get the list of X11/CSS4 color names
        lib_colors = list(plt.cm.colors.cnames)
        size = len(self.layers)
        # Seed for reproducibility
        rng = np.random.default_rng(seed=seed)
        rnd_ints = np.arange(0, len(lib_colors))
        rng.shuffle(rnd_ints)
        colors = [lib_colors[c] for c in rnd_ints[:size]]
        # Add to dict style in place
        for i, gdf_list in enumerate(self.layers.values()):
            gdf_list.append(colors[i])

    def explore_network(
            self
    ) -> folium.Map:
        """Map of the circuit.

        Easy on the number of elements.

        """
        ckt_map = folium.Map(
            crs="EPSG3857",
            zoom_start=15,
            control_scale=True,
            tiles="cartodbpositron"
        )
        # Pile up layers
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

        # Customize tile
        folium.TileLayer("Cartodb dark_matter", show=False).add_to(ckt_map)
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
    ene_data: pd.DataFrame | None = None         # Energy
    mde_data: pd.DataFrame | None = None         # Max. Demand
    power_data: pd.DataFrame | None = None
    voltage_data: pd.DataFrame | None = None
    current_data: pd.DataFrame | None = None
    pf_data: pd.DataFrame | None = None         # Power Factor
    energy_df: pd.DataFrame | None = None       # Re-frame ENE
    ami_gdf: gpd.GeoDataFrame | None = None     # AMI mapped on GIS

    def __post_init__(
            self
    ):
        """Initiate data type."""
        self.load_data()
        self.set_energy_df()
        if self.gis:
            _ = self.put_geometry()

    def set_df(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """Process datatype and add hour column."""
        df['CONTADOR'] = df['CONTADOR'].str.lower()
        df['LOCALIZACION'] = df['LOCALIZACION'].astype("Int64")  # NISE
        df.rename(columns={"LOCALIZACION": "NISE"}, inplace=True)
        # Number type
        df['VALOR_LECTURA'] = df['VALOR_LECTURA'].astype("float")
        df['MEDIDOR'] = df['MEDIDOR'].astype("Int64")            # ID
        df['LOCALIZACION_REAL'] = df['LOCALIZACION_REAL'].astype("Int64")
        df['FECHA_LECTURA'] = df['FECHA_LECTURA_REAL']    # Switch columns
        # Add "Hour" column
        seconds_s = (
            (df['FECHA_LECTURA'].dt.hour * 3600.0)
            + (df['FECHA_LECTURA'].dt.minute * 60.0)
            + (df['FECHA_LECTURA'].dt.second)
        )
        df['Hour'] = seconds_s / 3600.0
        return df

    def split_ene_mde(
            self,
            df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Energetic and Max. Power Demand flow strutures."""
        ene_mde_df = (
            df[(df.OPERACION.isin(["Delivered", "Received"]))
               & (df.CONTADOR == "total")
               & (df.TIPO_CONSUMO != "FPO")]
        )
        ene_df = (
            ene_mde_df[ene_mde_df.TIPO_CONSUMO.apply(lambda x: "ENE" in x)]
        )
        mde_df = (
            ene_mde_df[ene_mde_df.TIPO_CONSUMO.apply(lambda x: "MDE" in x)]
        )
        return ene_df.reset_index(drop=True), mde_df.reset_index(drop=True)

    def get_power(
        self,
        abc_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Retrieve kW, kVAR, kVA."""
        return abc_df[abc_df.TIPO_CONSUMO.isin([
            "kW", "kVAR", "kVA"
        ])].reset_index(drop=True)

    def get_voltage(
        self,
        abc_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Retrieve Voltage, Voltage Angle."""
        return abc_df[abc_df.TIPO_CONSUMO.isin([
            "Voltage", "Voltage Angle"
        ])].reset_index(drop=True)

    def get_current(
        self,
        abc_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Retrieve Current, Current Angle."""
        return abc_df[abc_df.TIPO_CONSUMO.isin([
            "Current", "Current Angle"
        ])].reset_index(drop=True)

    def get_pf(
        self,
        abc_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Retrieve Power Factor, Power Factor Angle."""
        return abc_df[abc_df.TIPO_CONSUMO.isin([
            "Power Factor", "Power Factor Angle"
        ])].reset_index(drop=True)

    def load_data(
            self
    ):
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
        self.df = df.copy()    # Deep copy
        # Without transformer meters
        df = df[df.SGDA.isin([None, "Bidireccional"])]
        # ENE and MDE
        self.ene_data, self.mde_data = self.split_ene_mde(df)
        # Phase analysis
        phases: set[str] = {"a", "b", "c", "A", "B", "C"}
        abc_df = df[(df.CONTADOR.isin(phases)) & (df.OPERACION.isin(phases))]
        self.power_data = self.get_power(abc_df)
        self.voltage_data = self.get_voltage(abc_df)
        self.current_data = self.get_current(abc_df)
        self.pf_data = self.get_pf(abc_df)

    def set_energy_df(
            self
    ):
        """Spread and sort by date Energy structure."""
        kwh_df = (
            self.ene_data
            .pivot_table(
                index=["MEDIDOR",
                       "FECHA_LECTURA",
                       "Hour"],
                columns="OPERACION",
                values="VALOR_LECTURA",
                aggfunc="first"
            )
            .reset_index()
        )
        # Set delta time
        kwh_df = kwh_df.sort_values(by=["MEDIDOR", "FECHA_LECTURA"])
        kwh_df["Period"] = (
            kwh_df.groupby("MEDIDOR")["FECHA_LECTURA"].diff()
        )
        # Set approx. "daily" sent and gotten
        costumers = kwh_df.groupby("MEDIDOR")
        kwh_df['Daily Sent'] = costumers['Delivered'].diff()
        kwh_df['Daily Gotten'] = costumers['Received'].diff()

        self.energy_df = kwh_df

    def n_phases(
            self
    ) -> pd.DataFrame:
        """Count number of phases of each meter."""
        gr = (
            self.voltage_data.groupby('MEDIDOR')['CONTADOR']
            .nunique()
            .reset_index(name="n_phases")
        )
        return gr

    def put_geometry(
            self,
            node_col_label: str = "MEDIDOR",
            geo_of_col: str = "LOCALIZACION_REAL"
    ) -> gpd.GeoDataFrame:
        """Map meter location to actual geometry point.

        As long as ami device location is not None
        and such location exists in GIS data.

        .. note::

            Drop duplicates between ``MEDIDOR`` and ``NISE``
            so that it is force to map one meter to one
            costumer NISE although one costumer may have
            multiple meters, this is only for visualization
            purposes.

        """
        gdf = self.gis.layers['Loads'][0]   # Retrieve costumers location
        # to_geom = dict(zip(gdf['IDLOCALIZA'], gdf['geometry']))
        to_geom = dict(zip(gdf[geo_of_col], gdf['geometry']))
        # ami_df = (
        #     self.df[['MEDIDOR', 'NISE']]
        #     .drop_duplicates()
        # )
        ami_df = (
            self.df[[node_col_label, geo_of_col]]
            .drop_duplicates()
        )
        # Remove ami with no location id
        ami_df = (
            ami_df[ami_df[geo_of_col].notna()]
            .reset_index(drop=True)
        )
        # Map id to actual geometry (Point)
        ami_df['geometry'] = (
            ami_df[geo_of_col].map(to_geom)
        )
        # Remove ami with no geometry
        ami_df = (
            ami_df[ami_df['geometry'].notna()]
            .reset_index(drop=True)
        )
        # Instantiate GeoDataFrame object
        ami_gdf = gpd.GeoDataFrame(
            ami_df, geometry='geometry', crs="EPSG:5367"
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
        default_factory=["Phase A Average RMS Voltage"]
    )
    gis: GISCircuit | None = None
    df: pd.DataFrame | None = None
    voltage_data: pd.DataFrame | None = None
    voltage_df: pd.DataFrame | None = None
    ami_gdf: gpd.GeoDataFrame | None = None

    def __post_init__(
            self
    ):
        """Initiate data type."""
        self.load_data()
        self.set_voltage_df()

    def set_df(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """Process datatype."""
        df['LOCALIZACION'] = df['LOCALIZACION'].astype("Int64")  # NISE
        df.rename(columns={"LOCALIZACION": "NISE"}, inplace=True)
        # Number type
        df['VALOR_LECTURA'] = df['VALOR_LECTURA'].astype("float")
        df['MEDIDOR'] = df['MEDIDOR'].astype("int64")            # ID
        df['FECHA_LECTURA'] = df['FECHA_LECTURA_REAL']    # Switch columns
        df = df[df.UNIDAD.isin(self.phase_vals)]
        df = (
            df.sort_values(by=["MEDIDOR", "FECHA_LECTURA"])
            .reset_index(drop=True)
        )
        return df

    def load_data(
            self
    ):
        """Read and set raw daily dirty data."""
        # Retain original DataFrame
        df = pd.read_parquet(self.data_path)
        df = self.set_df(df)
        self.df = df.copy()    # Copy
        self.voltage_data = df[[
            'MEDIDOR',
            'NISE',
            'FECHA_LECTURA',
            'UNIDAD',
            'VALOR_LECTURA'
        ]]

    def set_voltage_df(
            self
    ):
        """Reframe dataset for further analysis."""
        columns = [
            "MEDIDOR", "FECHA_LECTURA", "UNIDAD", "VALOR_LECTURA"
        ]
        v_df = self.voltage_data[columns].copy()
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

    def __post_init__(
            self
    ):
        """Load and set the data."""
        self.load_data()
        self.set_active_df()
        self.set_reactive_df()

    def set_df(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """Process datatype."""
        df['LOCALIZACION'] = df['LOCALIZACION'].astype("Int64")  # NISE
        df.rename(columns={"LOCALIZACION": "NISE"}, inplace=True)
        # Number type
        df['VALOR_LECTURA'] = df['VALOR_LECTURA'].astype("float")
        df['MEDIDOR'] = df['MEDIDOR'].astype("int64")            # ID
        df['LOCALIZACION_REAL'] = df['LOCALIZACION_REAL'].astype("Int64")
        df['FECHA_LECTURA'] = df['FECHA_LECTURA_REAL']    # Switch columns
        df = (
            df.sort_values(by=["MEDIDOR", "FECHA_LECTURA"])
            .reset_index(drop=True)
        )
        return df

    def split_power(
            self,
            df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Retrieve kWh and kVARh."""
        df = (
            df[(df.OPERACION.isin(["Delivered", "Received"]))]
        )
        active_df = (
            df[df.UNIDAD == "kWh"]
        )
        reactive_df = (
            df[df.UNIDAD == "kVARh"]
        )
        return (active_df, reactive_df)

    def load_data(
            self
    ):
        """Read and set raw daily dirty data."""
        df = pd.read_parquet(self.data_path)
        df = self.set_df(df)
        self.df = df.copy()    # Copy
        # Without transformer meters
        df = df[df.SGDA.isin([None, "Bidireccional"])]
        # Split active and reactive
        kwh_data, kvarh_data = self.split_power(df)
        # Beef data
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

    def set_active_df(
            self
    ):
        """Restructure original data set.

        Both ``Delivered`` and ``Received`` fields
        become nested in either ``ENERGIA`` and ``DEMANDA``.
        Sort by date and assume everyone sends and
        receives power but in case they don't then it
        is either zero or filled up with `NaN`.

        .. note::

            Columns are labeled from load's perspective.

        """
        active_df = (
            self.kwh_data
            .pivot_table(
                index=[
                    "MEDIDOR",
                    "FECHA_LECTURA"
                ],
                columns="OPERACION",
                values=["ENERGIA", "DEMANDA"],
                aggfunc="first"
            ).reset_index()
        )
        # Rename columns
        rename_dict: dict[tuple[str], tuple[str]] = {
            ("MEDIDOR", ""): ("node", ""),  # Top level
            ("FECHA_LECTURA", ""): ("ts", ""),  # Top level
            ("DEMANDA", "Delivered"): ("P", "Pdem"),
            ("DEMANDA", "Received"): ("P", "Pgen"),
            ("ENERGIA", "Delivered"): ("E", "Edem"),
            ("ENERGIA", "Received"): ("E", "Egen")
        }
        active_df.columns = active_df.columns.map(
            lambda c: rename_dict.get(c, c)
        )
        self.active_df = active_df

    def set_reactive_df(
            self
    ):
        """Restructure original data set.

        Both ``Delivered`` and ``Received`` fields
        become nested in either ``ENERGIA`` and ``DEMANDA``.
        Sort by date and assume everyone sends and
        receives power but in case they don't then it
        is either zero or filled up with `NaN`.

        """
        reactive_df = (
            self.kvarh_data
            .pivot_table(
                index=[
                    "MEDIDOR",
                    "FECHA_LECTURA"
                ],
                columns="OPERACION",
                values=["ENERGIA", "DEMANDA"],
                aggfunc="first"
            ).reset_index()
        )
        # Rename columns
        rename_dict: dict[tuple[str], tuple[str]] = {
            ("MEDIDOR", ""): ("node", ""),  # Top level
            ("FECHA_LECTURA", ""): ("ts", ""),  # Top level
            ("DEMANDA", "Delivered"): ("Q", "Qdem"),
            ("DEMANDA", "Received"): ("Q", "Qgen"),
            ("ENERGIA", "Delivered"): ("E", "Edem"),
            ("ENERGIA", "Received"): ("E", "Egen")
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
            "_".join(filter(None, col)) for col in flat_df.columns
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
    """Smart City and Microgrid loadshapes.

    Vitual potential realities so to speak, in case of
    a University campus these could be the scenarios:

        - Weekdays of each academic season.
        - Weekdays on vacations.
        - Weekends throughout the year.
        - The event of maximum possible demand.

    """

    season_curves: pd.DataFrame | None = None   # Weekdays only
    weekends_curve: pd.DataFrame | None = None
    sdata: pd.DataFrame = field(init=False)

    def __post_init__(self):
        """Set incoming data structure."""
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
        df["day"] = df["ts"].dt.weekday.apply(
            lambda x: "weekend" if x >= 5 else "weekday"
        )
        df = self.set_seasons(df)
        df["date"] = df["ts"].dt.date
        # Keep valid days only: With 96 samples per day
        day_counts = (
            df.groupby(["node", "date"])
            .size()
            .rename("samples")
            .reset_index()
        )
        valid_days = day_counts[day_counts["samples"] == 96]
        # Intercept valid days on big data and update
        df = df.merge(
            valid_days[["node", "date"]], on=["node", "date"], how="inner"
        )
        return df

    def get_weekends_curve(
            self,
            power: str = "P"
    ) -> pd.DataFrame:
        """Retrieve weekends and compute average throughout time."""
        wend = self.sdata[self.sdata['day'] == "weekend"].copy()
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
        """Compute average load shape of each weekday seasons."""
        wday = self.sdata[self.sdata['day'] == "weekday"].copy()
        # Time as if it ware index (0, 1, ..., 95)
        wday["timestep"] = wday.groupby(["season", "node", "date"]).cumcount()
        # Average shape over all valid weekdays for each node
        wday_avg_shape = (
            wday.groupby(["season", "node", "timestep"])[power]
            .mean()
            .reset_index()
            .sort_values(["node", "timestep"])
        )
        self.season_curves = wday_avg_shape
        return wday_avg_shape


if __name__ == "__main__":
    data = ConsumptionData()
    energy_data = data.ene_data

    ax = energy_data['Hour'].hist(
        grid=False,
        figsize=(8.0, 4.0),
        legend=True,
        bins=25,
    )
