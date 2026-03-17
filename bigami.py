
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
        """Compare identifiers between this dataset and another object.

        Returns:
            in_both: ids present in both datasets
            in_a_not_b: ids present only in this sdf
            in_b_not_a: ids present only in other's sdf/df

        Comparison is done on a normalized string representation of `domain_label`
        to avoid type conflicts and implicit casts.

        """
        col = domain_label
        table_a = self.sdf

        if table_a is None:
            raise ValueError("self.sdf is None. Load data first.")

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

        norm_col = "__key"

        a_keys = (
            table_a
            .select(F.col(col).cast("string").alias(col))
            .distinct()
            .withColumn(norm_col, F.col(col))
        )

        b_keys = (
            table_b
            .select(F.col(col).cast("string").alias(col))
            .distinct()
            .withColumn(norm_col, F.col(col))
        )

        diff = (
            a_keys.alias("a")
            .join(
                b_keys.alias("b"),
                on=[F.col(f"a.{norm_col}") == F.col(f"b.{norm_col}")],
                how="outer"
            )
        )

        # Present in both
        in_both = (
            diff
            .filter(F.col(f"a.{norm_col}").isNotNull() & F.col(f"b.{norm_col}").isNotNull())
            .select(F.col(f"a.{col}").alias(col))
        )

        # Present only in A
        in_a_not_b = (
            diff
            .filter(F.col(f"a.{norm_col}").isNotNull() & F.col(f"b.{norm_col}").isNull())
            .select(F.col(f"a.{col}").alias(col))
        )

        # Present only in B
        in_b_not_a = (
            diff
            .filter(F.col(f"b.{norm_col}").isNotNull() & F.col(f"a.{norm_col}").isNull())
            .select(F.col(f"b.{col}").alias(col))
        )

        return in_both, in_a_not_b, in_b_not_a

# AMI base in Spark
# -----------------


@dataclass
class AMI(ABC):
    """Abstract AMI device backed by Spark DataFrames."""

    sdf: Optional[DataFrame] = None  # main Spark DataFrame
    # Optional generic time filtering
    from_ts: Optional[str] = None     # e.g., "YYYY-MM-DD"
    to_ts: Optional[str] = None       # e.g., "YYYY-MM-DD"
    ts_col: str = "FECHA_LECTURA_REAL"  # default raw timestamp column name
    # Stores Spark DataFrames with invalid raw values by audit key
    audit_invalid_values: bool = False
    invalid_value_audit: dict[str, DataFrame] = field(
        default_factory=dict, init=False
    )

    @abstractmethod
    def load_data(self):
        """Read smart meters data into Spark."""
        ...

    @abstractmethod
    def set_sdf(self, sdf: DataFrame) -> DataFrame:
        """Process datatype and columns names on a Spark DataFrame."""
        ...

    def _apply_time_filter(self, sdf: DataFrame) -> DataFrame:
        """Apply [from_ts, to_ts) filter if provided.

        - This is called *before* set_sdf so you scan fewer rows.
        - Assumes ts_col is present in the raw table.

        """
        if self.from_ts is None and self.to_ts is None:
            return sdf

        cond = F.lit(True)
        if self.from_ts is not None:
            cond = cond & (F.col(self.ts_col) >= F.lit(self.from_ts))
        if self.to_ts is not None:
            cond = cond & (F.col(self.ts_col) < F.lit(self.to_ts))
        return sdf.filter(cond)

    def test_domain(
        self,
        other: object,
        domain_label: str = "NodeID"
    ) -> Tuple[DataFrame, DataFrame]:
        """Compare identifiers between this AMI dataset and another object.

        Returns:
            in_both: ids present in both datasets
            in_a_not_b: ids present only in this sdf
            in_b_not_a: ids present only in other's sdf/df

        """
        col = domain_label
        table_a = self.sdf

        if table_a is None:
            raise ValueError("self.sdf is None. Load data first.")

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

        norm_col = "__key"

        a_keys = (
            table_a
            .select(F.col(col).cast("string").alias(col))
            .distinct()
            .withColumn(norm_col, F.col(col))
        )

        b_keys = (
            table_b
            .select(F.col(col).cast("string").alias(col))
            .distinct()
            .withColumn(norm_col, F.col(col))
        )

        diff = (
            a_keys.alias("a")
            .join(
                b_keys.alias("b"),
                on=[F.col(f"a.{norm_col}") == F.col(f"b.{norm_col}")],
                how="outer"
            )
        )

        # Present in both
        in_both = (
            diff
            .filter(F.col(f"a.{norm_col}").isNotNull() & F.col(f"b.{norm_col}").isNotNull())
            .select(F.col(f"a.{col}").alias(col))
        )

        # Present only in A
        in_a_not_b = (
            diff
            .filter(F.col(f"a.{norm_col}").isNotNull() & F.col(f"b.{norm_col}").isNull())
            .select(F.col(f"a.{col}").alias(col))
        )

        # Present only in B
        in_b_not_a = (
            diff
            .filter(F.col(f"b.{norm_col}").isNotNull() & F.col(f"a.{norm_col}").isNull())
            .select(F.col(f"b.{col}").alias(col))
        )

        return in_both, in_a_not_b, in_b_not_a

    def audit_odd_values(
        self,
        sdf: DataFrame,
        col_name: str,
        valid_match_regex: str = r"^[+-]?\d+$",
        audit_key: Optional[str] = None
    ) -> DataFrame:
        """Identify and quantify values that do not follow a regex.

        Build a Spark DataFrame with raw
        values from ``col_name`` that do NOT match a bigint pattern.

        Valid bigint pattern:
            optional leading sign (+/-), followed by one or more digits

        Examples considered valid:
            123
            -45
            +77
            00012

        Examples considered invalid:
            N/A
            X0X07H16
            12.3
            1,000
            45A

        Parameters
        ----------
        sdf : DataFrame
            Input Spark DataFrame.
        col_name : str
            Column to validate/cast.
        valid_match_regex : str
            Regex pattern to match. Default is for integer-like
            values suitable for later bigint cast
            optional sign, then digits only.
        audit_key : str
            Column to be interrogated.

        Returns
        -------
        invalid_df : DataFrame
            Spark DataFrame with columns:
                - invalid_raw_value
                - invalid_count
                - column_name

        Also stores the DataFrame in:
            self.invalid_value_audit[audit_key or col_name]

        """

        audit_key = audit_key or col_name
        raw_as_str = F.col(col_name).cast("string")
        invalid_cond = ~raw_as_str.rlike(valid_match_regex)

        invalid_df = (
            sdf
            .select(raw_as_str.alias("invalid_raw_value"))
            .filter(invalid_cond)
            .groupBy("invalid_raw_value")
            .agg(F.count(F.lit(1)).alias("invalid_count"))
            .withColumn("column_name", F.lit(col_name))
            .select("column_name", "invalid_raw_value", "invalid_count")
            # .orderBy(F.desc("invalid_count"), F.asc("invalid_raw_value"))
        )

        self.invalid_value_audit[audit_key] = invalid_df
        return invalid_df

    def write_invalid_value_audit(
        self,
        table_name: str = "ami.invalid_value_audit",
        mode: str = "overwrite",
        source_name: Optional[str] = None
    ) -> None:
        """Write all invalid value audits into a single Delta table.

        Persist all invalid-value audit results stored in ``self.invalid_value_audit``
        into a single Delta table.

        This method is intended to consolidate the output of previous calls to
        ``audit_odd_values`` into one Spark/Delta table for easier querying,
        monitoring, and long-term storage.

        Each entry of ``self.invalid_value_audit`` is expected to be a Spark DataFrame
        describing invalid raw values detected for a given column. Those individual
        audit DataFrames are enriched with metadata columns and then unioned into a
        single Spark DataFrame, which is finally written as a Delta table.

        The resulting Delta table is useful for:
            - tracking malformed values found in integer-like fields.
            - monitoring source data quality over time.
            - building dashboards or SQL reports on invalid values.
            - comparing invalid-value patterns across different datasets or runs.

        Parameters
        ----------
        table_name : str, default "ami.invalid_value_audit"
            Fully qualified Spark table name where the consolidated audit results
            will be written.

            Examples:
                - "ami.invalid_value_audit"
                - "my_catalog.ami.invalid_value_audit"   (if using Unity Catalog)

            The table will be created automatically if it does not already exist.

        mode : str, default "append"
            Write mode used when saving the Delta table.

            Common values include:
                - "append":
                    Adds the new audit results as new rows in the target Delta table.
                    This is the recommended mode if you want to preserve audit history
                    across multiple runs.
                - "overwrite":
                    Replaces the contents of the target table with the current audit
                    results only.
                - "ignore":
                    Does nothing if the target table already exists.
                - "error" or "errorifexists":
                    Raises an error if the target table already exists.

            In most production audit workflows, "append" is the preferred option.

        source_name : Optional[str], default None
            Optional label identifying the source object, dataset, or pipeline that
            generated the audit.

            If provided, a new column named ``source_name`` is added to the output
            Delta table. This is useful when multiple AMI-derived classes write into
            the same audit table, for example:
                - "ConsumptionData"
                - "VoltageData"
                - "PowerData"

            If ``None``, the ``source_name`` column is not added.

        Returns
        -------
        None
            This method does not return anything. Its effect is to write the
            consolidated audit results to the specified Delta table.

        Raises
        ------
        ValueError
            If ``self.invalid_value_audit`` is empty, meaning no audit results are
            available to write. In that case, you must run ``audit_odd_values`` first
            or instantiate the class with ``audit_invalid_values=True`` and load data.

        Delta Table Structure
        ---------------------
        The generated Delta table contains one row per distinct invalid raw value
        detected in each audited column.

        Base columns coming from ``audit_odd_values``:
            - column_name : string
                Name of the original column that was audited.
            - invalid_raw_value : string
                Raw value that did not satisfy the expected validation rule
                (typically a regex for integer-like values).
            - invalid_count : bigint
                Number of times that invalid raw value appeared in the audited data.

        Additional metadata columns added by this method:
            - audit_key : string
                Key used in ``self.invalid_value_audit`` to identify the audit result.
                This is often the same as the column name, but can be different if
                a custom ``audit_key`` was passed to ``audit_odd_values``.
            - audit_ts : timestamp
                Timestamp at which this consolidated audit record was written.
                Useful for preserving audit history across runs.
            - source_name : string, optional
                User-provided label identifying the source dataset/class/process that
                produced the audit. This column is present only if ``source_name`` is
                passed to the method.

        Example resulting schema
        ------------------------
        If ``source_name`` is provided, the Delta table may look like:

        +-------------------+--------------------+---------------+-------------------+------------------+
        | column_name       |  invalid_raw_value | invalid_count |     audit_key     | source_name      |
        +===================+====================+===============+===================+==================+
        | MEDIDOR           |      X0X07H16      |      25       |      MEDIDOR      | ConsumptionData  |
        +-------------------+--------------------+---------------+-------------------+------------------+
        | MEDIDOR           |        N/A         |      7        |      MEDIDOR      | ConsumptionData  |
        +-------------------+--------------------+---------------+-------------------+------------------+
        | LOCALIZACION_REAL |        12.3        |      3        | LOCALIZACION_REAL | ConsumptionData  |
        +-------------------+--------------------+---------------+-------------------+------------------+
        | LOCALIZACION      |      PLANTA BRA    |      14       |       NISE        | ConsumptionData  |
        +-------------------+--------------------+---------------+-------------------+------------------+

        Notes
        -----
        - This method assumes all DataFrames stored in ``self.invalid_value_audit``
        share the same base schema produced by ``audit_odd_values``.
        - The method uses ``unionByName`` to combine all audit DataFrames safely by
        column name.
        - When ``mode="append"``, repeated executions will accumulate audit history
        in the target Delta table.
        - If you want only the latest audit snapshot, use ``mode="overwrite"``.

        Example
        -------
        >>> cdata = ConsumptionData(audit_invalid_values=True)
        >>> cdata.write_invalid_value_audit(
        >>>     table_name="ami.invalid_value_audit",
        >>>     mode="overwrite",
        >>>     source_name="ConsumptionData"
        >>> )

        """
        if not self.invalid_value_audit:
            message_log: str = (
                "No invalid value audits found. "
                "Run audit_odd_values first."
            )
            raise ValueError(message_log)

        audit_frames = []

        for audit_key, audit_df in self.invalid_value_audit.items():
            enriched_df = audit_df.withColumn("audit_key", F.lit(audit_key))

            if source_name is not None:
                enriched_df = enriched_df.withColumn("source_name", F.lit(source_name))

            audit_frames.append(enriched_df)

        final_df = audit_frames[0]
        for df in audit_frames[1:]:
            final_df = final_df.unionByName(df)

        (
            final_df
            .write
            .format("delta")
            .mode(mode)
            .saveAsTable(table_name)
        )

# InfoClientManager base and CNFLCustomers in Spark
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
class CNFLCustomers(InfoClientManager):
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
    # Optional date filters (inherited from AMI, but override defaults)
    from_ts: Optional[str] = "2026-02-28"
    to_ts: Optional[str] = "2026-03-13"
    ts_col: str = "FECHA_LECTURA_REAL"
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
        """Process datatype and add hour column, robust to malformed values."""
        if self.audit_invalid_values:
            # Audit bad raw values first, without casting
            _ = self.audit_odd_values(
                sdf,
                "LOCALIZACION",
                audit_key="NISE" # This field it is actually the NISE
            )
            _ = self.audit_odd_values(sdf, "MEDIDOR")
            _ = self.audit_odd_values(sdf, "LOCALIZACION_REAL")

        sdf = (
            sdf
            # CONTADOR as lowercase string, safe even if not originally string
            .withColumn("CONTADOR", F.lower(F.col("CONTADOR").cast("string")))

            # LOCALIZACION → NISE, using try_cast semantics:
            #   numeric-like → bigint
            #   malformed (e.g. 'PLANTA BRA') → NULL, no error
            .withColumn(
                "LOCALIZACION_STR",
                F.trim(F.col("LOCALIZACION").cast("string"))
            )
            .withColumn(
                "LOCALIZACION_CLEAN",
                F.regexp_replace("LOCALIZACION_STR", r"^(\d+),0+$", r"$1")
            )
            .withColumn(
                "NISE",
                F.expr("try_cast(LOCALIZACION_CLEAN as bigint)")
            )
            .drop("LOCALIZACION", "LOCALIZACION_STR", "LOCALIZACION_CLEAN")

            # VALOR_LECTURA as double with try_cast behavior
            .withColumn(
                "VALOR_LECTURA",
                F.expr("try_cast(VALOR_LECTURA as double)")
            )

            # MEDIDOR as bigint with try_cast
            .withColumn(
                "MEDIDOR",
                F.expr("try_cast(MEDIDOR as bigint)")
            )

            # LOCALIZACION_REAL as bigint with try_cast
            .withColumn(
                "LOCALIZACION_REAL",
                F.expr("try_cast(LOCALIZACION_REAL as bigint)")
            )

            # FECHA_LECTURA from FECHA_LECTURA_REAL (assuming already timestamp)
            .withColumn("FECHA_LECTURA", F.col("FECHA_LECTURA_REAL"))
        )

        # Compute Hour as decimal hours from timestamp.
        # If FECHA_LECTURA is NULL, hour/minute/second return NULL -> Hour is NULL.
        sdf = sdf.withColumn(
            "Hour",
            (
                F.hour("FECHA_LECTURA") * F.lit(3600.0) +
                F.minute("FECHA_LECTURA") * F.lit(60.0) +
                F.second("FECHA_LECTURA")
            ) / F.lit(3600.0)
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
        """Read and set raw daily dirty data using Spark, with optional time filtering."""
        if self.table_name:
            sdf = spark.table(self.table_name)
        elif self.data_path:
            sdf = spark.read.format("delta").load(self.data_path)
        else:
            raise ValueError("Either table_name or data_path must be provided.")

        # Apply date filter as early as possible (still raw schema)
        sdf = self._apply_time_filter(sdf)

        # Then apply all type cleaning / derived columns
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
    # Optional date filters (inherited from AMI, but override defaults)
    from_ts: Optional[str] = "2026-02-28"
    to_ts: Optional[str] = "2026-03-13"
    ts_col: str = "FECHA_LECTURA_REAL"
    voltage_data: Optional[DataFrame] = None
    voltage_df: Optional[DataFrame] = None

    def __post_init__(self):
        self.load_data()
        self.set_voltage_df()

    def set_sdf(self, sdf: DataFrame) -> DataFrame:
        """Process datatype, robust to malformed values."""
        if self.audit_invalid_values:
            # Audit bad raw values first, without casting
            _ = self.audit_odd_values(
                sdf,
                "LOCALIZACION",
                audit_key="NISE" # This field it is actually the NISE
            )
            _ = self.audit_odd_values(sdf, "MEDIDOR")
            _ = self.audit_odd_values(sdf, "LOCALIZACION_REAL")

        sdf = (
            sdf
            # NISE from LOCALIZACION with try_cast
            .withColumn(
                "LOCALIZACION_STR",
                F.trim(F.col("LOCALIZACION").cast("string"))
            )
            .withColumn(
                "LOCALIZACION_CLEAN",
                F.regexp_replace("LOCALIZACION_STR", r"^(\d+),0+$", r"$1")
            )
            .withColumn(
                "NISE",
                F.expr("try_cast(LOCALIZACION_CLEAN as bigint)")
            )
            .drop("LOCALIZACION", "LOCALIZACION_STR", "LOCALIZACION_CLEAN")

            # VALOR_LECTURA as double
            .withColumn(
                "VALOR_LECTURA",
                F.expr("try_cast(VALOR_LECTURA as double)")
            )

            # MEDIDOR as bigint
            .withColumn(
                "MEDIDOR",
                F.expr("try_cast(MEDIDOR as bigint)")
            )

            # FECHA_LECTURA from FECHA_LECTURA_REAL
            .withColumn("FECHA_LECTURA", F.col("FECHA_LECTURA_REAL"))

            # UNIDAD normalized to string
            .withColumn("UNIDAD", F.col("UNIDAD").cast("string"))

            # Filter by selected phase values
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

        sdf = self._apply_time_filter(sdf)  # <── added
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
    # Optional date filters (inherited from AMI, but override defaults)
    from_ts: Optional[str] = "2026-02-28"
    to_ts: Optional[str] = "2026-03-13"
    ts_col: str = "FECHA_LECTURA_REAL"
    kwh_data: Optional[DataFrame] = None
    kvarh_data: Optional[DataFrame] = None
    active_df: Optional[DataFrame] = None
    reactive_df: Optional[DataFrame] = None

    def __post_init__(self):
        self.load_data()
        self.set_active_df()
        self.set_reactive_df()

    def set_sdf(self, sdf: DataFrame) -> DataFrame:
        """Process datatype, robust to malformed values."""
        if self.audit_invalid_values:
            # Audit bad raw values first, without casting
            _ = self.audit_odd_values(
                sdf,
                "LOCALIZACION",
                audit_key="NISE" # This field it is actually the NISE
            )
            _ = self.audit_odd_values(sdf, "MEDIDOR")
            _ = self.audit_odd_values(sdf, "LOCALIZACION_REAL")

        sdf = (
            sdf
            # NISE from LOCALIZACION
            .withColumn(
                "LOCALIZACION_STR",
                F.trim(F.col("LOCALIZACION").cast("string"))
            )
            .withColumn(
                "LOCALIZACION_CLEAN",
                F.regexp_replace("LOCALIZACION_STR", r"^(\d+),0+$", r"$1")
            )
            .withColumn(
                "NISE",
                F.expr("try_cast(LOCALIZACION_CLEAN as bigint)")
            )
            .drop("LOCALIZACION", "LOCALIZACION_STR", "LOCALIZACION_CLEAN")

            # VALOR_LECTURA as double
            .withColumn(
                "VALOR_LECTURA",
                F.expr("try_cast(VALOR_LECTURA as double)")
            )

            # MEDIDOR as bigint
            .withColumn(
                "MEDIDOR",
                F.expr("try_cast(MEDIDOR as bigint)")
            )

            # LOCALIZACION_REAL as bigint
            .withColumn(
                "LOCALIZACION_REAL",
                F.expr("try_cast(LOCALIZACION_REAL as bigint)")
            )

            # FECHA_LECTURA from FECHA_LECTURA_REAL
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

        sdf = self._apply_time_filter(sdf)  # <── added
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
