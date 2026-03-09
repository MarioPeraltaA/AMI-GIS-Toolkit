
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Tuple, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

import json

spark = SparkSession.getActiveSession()  # in a Databricks notebook / job

# GIS base (kept mostly local / small; heavy joins done in Spark)
# ---------------------------------------------------------------


@dataclass
class GIS(ABC):
    """Abstract GIS contract (Spark + optional GeoPandas for visualization).

    For scalability, GIS tabular attributes should live in Spark tables.
    Geo aspects (geometry) can stay in GeoPandas on the driver if small.
    """

    # Spark DataFrame containing at least an identifier column (e.g., NISE)
    sdf: Optional[DataFrame] = None

    @abstractmethod
    def load_gis(self):
        """Read GIS of each meter into Spark (and optional GeoPandas for maps)."""
        ...

    def test_domain(
        self,
        other: object,
        domain_label: str = "NodeID"
    ) -> Tuple[DataFrame, DataFrame]:
        """Compare identifiers between this GIS and another object.

        Returns:
        - in_a_not_b: ids present only in this sdf
        - in_b_not_a: ids present only in other's df/sdf

        """
        col = domain_label
        table_a = self.sdf

        if table_a is None:
            raise ValueError("self.sdf is None. Load GIS data first.")

        # Resolve other into a Spark DataFrame
        if isinstance(other, GIS) and col == "NISE":
            table_b = other.sdf
        elif isinstance(other, AMI):
            table_b = other.sdf
        elif isinstance(other, InfoClientManager):
            table_b = other.sdf
        elif isinstance(other, DataFrame):
            table_b = other
        else:
            raise TypeError(f"Unsupported type for 'other': {type(other)}")

        if table_b is None:
            raise ValueError("Other object does not have a Spark DataFrame loaded.")

        # Distinct keys
        a_keys = table_a.select(col).distinct().alias("a")
        b_keys = table_b.select(col).distinct().alias("b")

        # Full outer join to detect existence differences
        diff = (
            a_keys.join(b_keys, on=[col], how="outer")
        )

        # In A not B: B side is null
        in_a_not_b = diff.filter(F.col(f"b.{col}").isNull()).select(F.col(f"a.{col}").alias(col))
        # In B not A: A side is null
        in_b_not_a = diff.filter(F.col(f"a.{col}").isNull()).select(F.col(f"b.{col}").alias(col))

        return in_a_not_b, in_b_not_a

# AMI base in Spark
# -----------------


@dataclass
class AMI(ABC):
    """Abstract AMI device backed by Spark DataFrames."""

    sdf: Optional[DataFrame] = None  # main Spark DataFrame

    @abstractmethod
    def load_data(self):
        """Read smart meters data into Spark."""
        ...

    @abstractmethod
    def set_sdf(self, sdf: DataFrame) -> DataFrame:
        """Process datatype and columns names on a Spark DataFrame."""
        ...

    def test_domain(
        self,
        other: object,
        domain_label: str = "NodeID"
    ) -> Tuple[DataFrame, DataFrame]:
        """Compare identifiers between this AMI dataset and another object.

        Same semantics as GIS.test_domain, but for AMI.sdf.

        """
        col = domain_label
        table_a = self.sdf

        if table_a is None:
            raise ValueError("self.sdf is None. Load AMI data first.")

        # Resolve other into a Spark DataFrame
        if isinstance(other, GIS) and col == "NISE":
            table_b = other.sdf
        elif isinstance(other, AMI):
            table_b = other.sdf
        elif isinstance(other, InfoClientManager):
            table_b = other.sdf
        elif isinstance(other, DataFrame):
            table_b = other
        else:
            raise TypeError(f"Unsupported type for 'other': {type(other)}")

        if table_b is None:
            raise ValueError("Other object does not have a Spark DataFrame loaded.")

        a_keys = table_a.select(col).distinct().alias("a")
        b_keys = table_b.select(col).distinct().alias("b")

        diff = (
            a_keys.join(b_keys, on=[col], how="outer")
        )

        in_a_not_b = diff.filter(F.col(f"b.{col}").isNull()).select(F.col(f"a.{col}").alias(col))
        in_b_not_a = diff.filter(F.col(f"a.{col}").isNull()).select(F.col(f"b.{col}").alias(col))

        return in_a_not_b, in_b_not_a

# InfoClientManager base and CNFLCostumers in Spark
# -------------------------------------------------

@dataclass
class InfoClientManager(ABC):
    """Database of some utility's customers."""
    sdf: Optional[DataFrame] = None

    @abstractmethod
    def read_customers_data(self) -> DataFrame:
        """Load information data of utility's customers into Spark."""
        ...


@dataclass
class CNFLCostumers(InfoClientManager):
    """CNFL customers database migrated to Spark.

    Assumes a Delta or parquet table exists; if you still have a text file
    and JSON schema, you can load it once and write to Delta.

    """

    table_name: str = "cnfl.infoclientes"     # Delta table in metastore
    datatype_path: str = "/dbfs/FileStore/datatype.json"  # JSON in DBFS
    columns_dtype: dict = field(init=False)

    def __post_init__(self):
        self.sdf = self.read_customers_data()

    def set_columns_data_type(self) -> dict:
        # If you really need a Python dict of dtypes (for initial ingestion)
        with open(self.datatype_path, "r", encoding="utf-8") as f:
            dtype_map = json.load(f)
        self.columns_dtype = dtype_map
        return dtype_map

    def read_customers_data(self) -> DataFrame:
        # Here we just read from a (Delta) table registered in Databricks
        # If not yet created, you would ingest from raw file in another step.
        return spark.table(self.table_name)

# ConsumptionData in Spark
# ------------------------

@dataclass
class ConsumptionData(AMI):
    """Spark-based ConsumptionData.

    Assumes a Delta or parquet dataset at `table_name`/`path`.

    """

    table_name: Optional[str] = "ami.consumption"  # Spark table (Delta)
    data_path: Optional[str] = None                # alternative: path
    gis: Optional[GIS] = None

    ene_data: Optional[DataFrame] = None
    mde_data: Optional[DataFrame] = None
    power_data: Optional[DataFrame] = None
    voltage_data: Optional[DataFrame] = None
    current_data: Optional[DataFrame] = None
    pf_data: Optional[DataFrame] = None
    energy_df: Optional[DataFrame] = None

    def __post_init__(self):
        self.load_data()
        self.set_energy_df()

    def set_sdf(self, sdf: DataFrame) -> DataFrame:
        """Equivalent to set_df, but using Spark operations."""
        sdf = (
            sdf
            .withColumn("CONTADOR", F.lower(F.col("CONTADOR")))
            .withColumn("NISE", F.col("LOCALIZACION").cast("long"))
            .drop("LOCALIZACION")
            .withColumn("VALOR_LECTURA", F.col("VALOR_LECTURA").cast("double"))
            .withColumn("MEDIDOR", F.col("MEDIDOR").cast("long"))
            .withColumn("LOCALIZACION_REAL", F.col("LOCALIZACION_REAL").cast("long"))
            .withColumn("FECHA_LECTURA", F.col("FECHA_LECTURA_REAL"))
        )

        # Compute Hour as decimal hours from timestamp
        sdf = sdf.withColumn(
            "Hour",
            (
                F.hour("FECHA_LECTURA") * 3600.0 +
                F.minute("FECHA_LECTURA") * 60.0 +
                F.second("FECHA_LECTURA")
            ) / 3600.0
        )
        return sdf

    def split_ene_mde(self, sdf: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Spark equivalent of split_ene_mde."""
        base = (
            sdf.filter(
                (F.col("OPERACION").isin("Delivered", "Received")) &
                (F.col("CONTADOR") == "total") &
                (F.col("TIPO_CONSUMO") != "FPO")
            )
        )
        ene_df = base.filter(F.col("TIPO_CONSUMO").contains("ENE"))
        mde_df = base.filter(F.col("TIPO_CONSUMO").contains("MDE"))
        return ene_df, mde_df

    def get_power(self, abc_df: DataFrame) -> DataFrame:
        return abc_df.filter(F.col("TIPO_CONSUMO").isin("kW", "kVAR", "kVA"))

    def get_voltage(self, abc_df: DataFrame) -> DataFrame:
        return abc_df.filter(F.col("TIPO_CONSUMO").isin("Voltage", "Voltage Angle"))

    def get_current(self, abc_df: DataFrame) -> DataFrame:
        return abc_df.filter(F.col("TIPO_CONSUMO").isin("Current", "Current Angle"))

    def get_pf(self, abc_df: DataFrame) -> DataFrame:
        return abc_df.filter(F.col("TIPO_CONSUMO").isin("Power Factor", "Power Factor Angle"))

    def load_data(self):
        """Read and set raw daily dirty data using Spark."""
        if self.table_name:
            sdf = spark.table(self.table_name)
        elif self.data_path:
            sdf = spark.read.format("delta").load(self.data_path)
        else:
            raise ValueError("Either table_name or data_path must be provided.")

        sdf = self.set_sdf(sdf)
        self.sdf = sdf

        # Filter out transformer meters
        df = sdf.filter(F.col("SGDA").isin([None, "Bidireccional"]))

        # ENE and MDE
        self.ene_data, self.mde_data = self.split_ene_mde(df)

        # Phase analysis
        phases = ["a", "b", "c", "A", "B", "C"]
        abc_df = df.filter(
            F.col("CONTADOR").isin(phases) & F.col("OPERACION").isin(phases)
        )
        self.power_data = self.get_power(abc_df)
        self.voltage_data = self.get_voltage(abc_df)
        self.current_data = self.get_current(abc_df)
        self.pf_data = self.get_pf(abc_df)

    def set_energy_df(self):
        """Replaces the pandas pivot_table + groupby with Spark operations."""
        if self.ene_data is None:
            raise ValueError("ene_data is not set.")

        df = self.ene_data

        # Pivot OPERACION -> Delivered, Received
        # We can use groupBy + agg with conditional aggregation
        pivoted = (
            df.groupBy("MEDIDOR", "FECHA_LECTURA", "Hour")
            .agg(
                F.first(F.when(F.col("OPERACION") == "Delivered", F.col("VALOR_LECTURA")), ignorenulls=True).alias("Delivered"),
                F.first(F.when(F.col("OPERACION") == "Received", F.col("VALOR_LECTURA")), ignorenulls=True).alias("Received"),
            )
        )

        # Sort
        window_by_meter = (
            F.window(F.col("FECHA_LECTURA"), "99999 days")  # dummy; we'll use analytic window
        )
        from pyspark.sql.window import Window
        w = Window.partitionBy("MEDIDOR").orderBy("FECHA_LECTURA")

        # Period = difference in FECHA_LECTURA between consecutive rows
        pivoted = pivoted.withColumn(
            "Period",
            F.col("FECHA_LECTURA").cast("timestamp").cast("long") -
            F.lag("FECHA_LECTURA").over(w).cast("timestamp").cast("long")
        )

        # Daily Sent / Gotten as diffs of Delivered / Received
        pivoted = pivoted.withColumn(
            "Daily Sent",
            F.col("Delivered") - F.lag("Delivered").over(w)
        ).withColumn(
            "Daily Gotten",
            F.col("Received") - F.lag("Received").over(w)
        )

        self.energy_df = pivoted

    def n_phases(self) -> DataFrame:
        """Count number of phases of each meter using Spark."""
        if self.voltage_data is None:
            raise ValueError("voltage_data not loaded.")
        return (
            self.voltage_data
            .groupBy("MEDIDOR")
            .agg(F.countDistinct("CONTADOR").alias("n_phases"))
        )

# VoltageData in Spark
# --------------------


@dataclass
class VoltageData(AMI):
    table_name: Optional[str] = "ami.voltages"
    data_path: Optional[str] = None
    phase_vals: list = field(default_factory=lambda: ["Phase A Average RMS Voltage"])

    voltage_data: Optional[DataFrame] = None
    voltage_df: Optional[DataFrame] = None

    def __post_init__(self):
        self.load_data()
        self.set_voltage_df()

    def set_sdf(self, sdf: DataFrame) -> DataFrame:
        sdf = (
            sdf
            .withColumn("NISE", F.col("LOCALIZACION").cast("long"))
            .drop("LOCALIZACION")
            .withColumn("VALOR_LECTURA", F.col("VALOR_LECTURA").cast("double"))
            .withColumn("MEDIDOR", F.col("MEDIDOR").cast("long"))
            .withColumn("FECHA_LECTURA", F.col("FECHA_LECTURA_REAL"))
            .filter(F.col("UNIDAD").isin(self.phase_vals))
            .orderBy("MEDIDOR", "FECHA_LECTURA")
        )
        return sdf

    def load_data(self):
        if self.table_name:
            sdf = spark.table(self.table_name)
        elif self.data_path:
            sdf = spark.read.format("delta").load(self.data_path)
        else:
            raise ValueError("Either table_name or data_path must be provided.")

        sdf = self.set_sdf(sdf)
        self.sdf = sdf

        self.voltage_data = sdf.select(
            "MEDIDOR",
            "NISE",
            "FECHA_LECTURA",
            "UNIDAD",
            "VALOR_LECTURA"
        )

    def set_voltage_df(self):
        """Reframe dataset: rename columns node, ts, phase, V."""
        if self.voltage_data is None:
            raise ValueError("voltage_data not loaded.")

        v_df = (
            self.voltage_data
            .select(
                F.col("MEDIDOR").alias("node"),
                F.col("FECHA_LECTURA").alias("ts"),
                F.col("UNIDAD").alias("phase"),
                F.col("VALOR_LECTURA").alias("V"),
            )
        )
        self.voltage_df = v_df

# PowerData in Spark
# ------------------

@dataclass
class PowerData(AMI):
    table_name: Optional[str] = "ami.power"
    data_path: Optional[str] = None

    kwh_data: Optional[DataFrame] = None
    kvarh_data: Optional[DataFrame] = None
    active_df: Optional[DataFrame] = None
    reactive_df: Optional[DataFrame] = None

    def __post_init__(self):
        self.load_data()
        self.set_active_df()
        self.set_reactive_df()

    def set_sdf(self, sdf: DataFrame) -> DataFrame:
        sdf = (
            sdf
            .withColumn("NISE", F.col("LOCALIZACION").cast("long"))
            .drop("LOCALIZACION")
            .withColumn("VALOR_LECTURA", F.col("VALOR_LECTURA").cast("double"))
            .withColumn("MEDIDOR", F.col("MEDIDOR").cast("long"))
            .withColumn("LOCALIZACION_REAL", F.col("LOCALIZACION_REAL").cast("long"))
            .withColumn("FECHA_LECTURA", F.col("FECHA_LECTURA_REAL"))
            .orderBy("MEDIDOR", "FECHA_LECTURA")
        )
        return sdf

    def split_power(self, sdf: DataFrame) -> Tuple[DataFrame, DataFrame]:
        base = sdf.filter(F.col("OPERACION").isin("Delivered", "Received"))
        active_df = base.filter(F.col("UNIDAD") == "kWh")
        reactive_df = base.filter(F.col("UNIDAD") == "kVARh")
        return active_df, reactive_df

    def load_data(self):
        if self.table_name:
            sdf = spark.table(self.table_name)
        elif self.data_path:
            sdf = spark.read.format("delta").load(self.data_path)
        else:
            raise ValueError("Either table_name or data_path must be provided.")

        sdf = self.set_sdf(sdf)
        self.sdf = sdf

        df = sdf.filter(F.col("SGDA").isin([None, "Bidireccional"]))
        kwh_data, kvarh_data = self.split_power(df)

        select_cols = [
            "MEDIDOR",
            "NISE",
            "LOCALIZACION_REAL",
            "OPERACION",
            "INTERVALO",
            "FECHA_LECTURA",
            "ENERGIA",
            "DEMANDA",
        ]

        self.kwh_data = kwh_data.select(*select_cols)
        self.kvarh_data = kvarh_data.select(*select_cols)

    def set_active_df(self):
        """Spark equivalent of pivoting Delivered/Received into nested P/E columns."""
        if self.kwh_data is None:
            raise ValueError("kwh_data not loaded.")

        df = self.kwh_data

        # groupBy and conditional aggregations
        active_df = (
            df.groupBy("MEDIDOR", "FECHA_LECTURA")
            .agg(
                F.first(F.when(F.col("OPERACION") == "Delivered", F.col("DEMANDA")), ignorenulls=True).alias("P_Pdem"),
                F.first(F.when(F.col("OPERACION") == "Received", F.col("DEMANDA")), ignorenulls=True).alias("P_Pgen"),
                F.first(F.when(F.col("OPERACION") == "Delivered", F.col("ENERGIA")), ignorenulls=True).alias("E_Edem"),
                F.first(F.when(F.col("OPERACION") == "Received", F.col("ENERGIA")), ignorenulls=True).alias("E_Egen"),
            )
            .select(
                F.col("MEDIDOR").alias("node"),
                F.col("FECHA_LECTURA").alias("ts"),
                "P_Pdem",
                "P_Pgen",
                "E_Edem",
                "E_Egen",
            )
        )

        self.active_df = active_df

    def set_reactive_df(self):
        if self.kvarh_data is None:
            raise ValueError("kvarh_data not loaded.")

        df = self.kvarh_data

        reactive_df = (
            df.groupBy("MEDIDOR", "FECHA_LECTURA")
            .agg(
                F.first(F.when(F.col("OPERACION") == "Delivered", F.col("DEMANDA")), ignorenulls=True).alias("Q_Qdem"),
                F.first(F.when(F.col("OPERACION") == "Received", F.col("DEMANDA")), ignorenulls=True).alias("Q_Qgen"),
                F.first(F.when(F.col("OPERACION") == "Delivered", F.col("ENERGIA")), ignorenulls=True).alias("E_Edem"),
                F.first(F.when(F.col("OPERACION") == "Received", F.col("ENERGIA")), ignorenulls=True).alias("E_Egen"),
            )
            .select(
                F.col("MEDIDOR").alias("node"),
                F.col("FECHA_LECTURA").alias("ts"),
                "Q_Qdem",
                "Q_Qgen",
                "E_Edem",
                "E_Egen",
            )
        )

        self.reactive_df = reactive_df

# ScenariosManager / CityScenarios in Spark
# -----------------------------------------

@dataclass
class ScenariosManager(ABC):
    """Use data to make up scenarios (Spark)."""
    sdf: DataFrame
    pdata: Optional[PowerData] = None
    vdata: Optional[VoltageData] = None
    cdata: Optional[ConsumptionData] = None


@dataclass
class CityScenarios(ScenariosManager):
    season_curves: Optional[DataFrame] = None
    weekends_curve: Optional[DataFrame] = None
    sdata: DataFrame = field(init=False)

    def __post_init__(self):
        self.sdata = self.city_scenarios(self.sdf)

    def set_seasons(
        self,
        sdf: DataFrame,
        seasons_map: dict = {
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
    ) -> DataFrame:
        # Build a map literal: map(1 -> "III", 2 -> "III", ...)
        mapping_expr = F.create_map(
            *[x for kv in seasons_map.items() for x in (F.lit(kv[0]), F.lit(kv[1]))]
        )

        return sdf.withColumn("season", mapping_expr[F.month("ts")])

    def city_scenarios(self, sdf: DataFrame) -> DataFrame:
        """Set labels up of an academic Microgrid using Spark.

        Assumes 'ts' and 'node' columns exist.
        Keep only days with 96 samples per node.

        """
        sdf = sdf.withColumn(
            "day",
            F.when(F.dayofweek("ts").isin(1, 7), F.lit("weekend")).otherwise(F.lit("weekday"))
        )

        sdf = self.set_seasons(sdf)

        sdf = sdf.withColumn("date", F.to_date("ts"))

        # Count samples per node/date
        day_counts = (
            sdf.groupBy("node", "date")
            .agg(F.count("*").alias("samples"))
        )

        valid_days = day_counts.filter(F.col("samples") == 96).select("node", "date")

        # Inner join to keep only valid days
        sdf_valid = sdf.join(valid_days, on=["node", "date"], how="inner")

        return sdf_valid

    def get_weekends_curve(self, power: str = "P") -> DataFrame:
        """Compute average weekend curve per node.

        Assumes column named `power` exists.

        """
        from pyspark.sql.window import Window

        w = Window.partitionBy("node", "date").orderBy("ts")

        wend = self.sdata.filter(F.col("day") == "weekend")
        wend = wend.withColumn("timestep", F.row_number().over(w) - 1)

        wend_avg_shape = (
            wend.groupBy("node", "timestep")
            .agg(F.mean(power).alias(power))
            .orderBy("node", "timestep")
        )

        self.weekends_curve = wend_avg_shape
        return wend_avg_shape

    def avg_curves(self, power: str = "P") -> DataFrame:
        """Compute average weekday load shape of each season."""
        from pyspark.sql.window import Window

        wday = self.sdata.filter(F.col("day") == "weekday")
        w = Window.partitionBy("season", "node", "date").orderBy("ts")
        wday = wday.withColumn("timestep", F.row_number().over(w) - 1)

        wday_avg_shape = (
            wday.groupBy("season", "node", "timestep")
            .agg(F.mean(power).alias(power))
            .orderBy("season", "node", "timestep")
        )

        self.season_curves = wday_avg_shape
        return wday_avg_shape
