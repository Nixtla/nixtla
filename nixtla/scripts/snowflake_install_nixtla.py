"""
This script deploys Nixtla forecasting components to Snowflake:
- External access integrations for API calls
- Python UDTFs for batch forecasting and evaluation
- SQL stored procedures for inference
- Stored procedure for model finetuning
- Example datasets

Usage:
    python -m nixtla.scripts.snowflake_install_nixtla [OPTIONS]

Options:
    --connection_name: Snowflake connection name (default: "default")
    --database: Database name (optional, will prompt)
    --schema: Schema name (optional, will prompt)
    --stage_path: Stage path (optional, will prompt)
    --integration_name: External access integration name (optional, default: "nixtla_access_integration")
    --base_url: Nixtla API base URL (optional, will prompt)
        - https://api.nixtla.io (default, TimeGPT)
        - https://tsmp.nixtla.io (TimeGPT-2, supports all models)
"""

import os
import shutil
import subprocess
import sys

from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Mapping, Optional
from urllib.parse import urlparse

import pandas as pd
from fire import Fire
from rich import print
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt
from snowflake.snowpark import Session
from pathlib import Path

# ============================================================================
# Constants
# ============================================================================

# Python packages required for Snowflake UDTFs
PACKAGES = [
    "annotated-types",
    "anyio",
    "certifi",
    "httpcore",
    "orjson",
    "pandas",
    "tenacity",
    "tqdm",
    "numpy",
    "zstandard",
    "httpx",
    "pydantic",
    "snowflake-snowpark-python",
    "requests",
    "narwhals",
    "rich",
]

# Data column names
BASE_COLUMNS = ["unique_id", "ds", "y"]
FORECAST_COLUMN = "forecast"
ANOMALY_COLUMN = "anomaly"

# Secret identifiers
SECRET_API_KEY = "nixtla_api_key"
SECRET_BASE_URL = "nixtla_base_url"

# Example data parameters
TRAINING_DAYS = 330  # Days of training data
TOTAL_DAYS = 365  # Total days including forecast period
ANOMALY_DAYS = 180  # Days for anomaly detection examples

TEMPLATE_ACCESS_INTEGRATION = """
CREATE OR REPLACE NETWORK RULE {ds_prefix}{network_rule_name}
MODE = EGRESS
TYPE = HOST_PORT
VALUE_LIST = ('{api_host}');

//<br>

CREATE OR REPLACE SECRET {ds_prefix}nixtla_api_key
  TYPE = GENERIC_STRING
  SECRET_STRING = '{nixtla_api_key}';

//<br>

CREATE OR REPLACE SECRET {ds_prefix}nixtla_base_url
  TYPE = GENERIC_STRING
  SECRET_STRING = '{base_url}';

//<br>

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION {integration_name}
  ALLOWED_NETWORK_RULES = ({ds_prefix}{network_rule_name})
  ALLOWED_AUTHENTICATION_SECRETS = ({ds_prefix}nixtla_api_key, {ds_prefix}nixtla_base_url)
  ENABLED = true;
"""

TEMPLATE_SP = """
CREATE OR REPLACE PROCEDURE {ds_prefix}NIXTLA_FORECAST(
    "INPUT_DATA" VARCHAR,
    "PARAMS" OBJECT,
    "MAX_BATCHES" NUMBER(38,0) DEFAULT 1000
)
RETURNS TABLE (
  UNIQUE_ID VARCHAR, DS TIMESTAMP, FORECAST DOUBLE, CONFIDENCE_INTERVALS VARIANT
)
LANGUAGE SQL AS
$$
DECLARE
  res RESULTSET;
BEGIN
  res := (SELECT b.* FROM
    (SELECT
        MOD(HASH(unique_id), :MAX_BATCHES) AS gp,
        unique_id,
        ds,
        OBJECT_CONSTRUCT_KEEP_NULL(*) AS data_obj
      FROM IDENTIFIER(:INPUT_DATA)) a,
    TABLE(nixtla_forecast_batch(:PARAMS, data_obj) OVER (PARTITION BY gp ORDER BY unique_id, ds)) b);
  RETURN TABLE(res);
END;
$$
;

//<br>

CREATE OR REPLACE PROCEDURE {ds_prefix}NIXTLA_EVALUATE(
    "INPUT_DATA" VARCHAR,
    "METRICS" ARRAY DEFAULT ['MAPE', 'MAE'],
    "MAX_BATCHES" NUMBER(38,0) DEFAULT 1000000
)
RETURNS TABLE (
  UNIQUE_ID VARCHAR, FORECASTER VARCHAR, METRIC VARCHAR, VALUE DOUBLE
)
LANGUAGE SQL AS
$$
DECLARE
  res RESULTSET;
BEGIN
  res := (SELECT b.* FROM
    (SELECT MOD(HASH(unique_id), :MAX_BATCHES) AS gp, unique_id, ds, y, OBJECT_CONSTRUCT_KEEP_NULL(*) AS obj FROM TABLE(:INPUT_DATA)) a,
    TABLE(nixtla_evaluate_batch(:METRICS, obj) OVER (PARTITION BY gp ORDER BY unique_id, ds)) b);
  RETURN TABLE(res);
END;
$$
;

//<br>

CREATE OR REPLACE PROCEDURE {ds_prefix}NIXTLA_DETECT_ANOMALIES(
    "INPUT_DATA" VARCHAR,
    "PARAMS" OBJECT DEFAULT OBJECT_CONSTRUCT(),
    "MAX_BATCHES" NUMBER(38,0) DEFAULT 1000
)
RETURNS TABLE (
  UNIQUE_ID VARCHAR, DS TIMESTAMP, Y DOUBLE, TIMEGPT DOUBLE, ANOMALY VARCHAR, TIMEGPT_LO DOUBLE, TIMEGPT_HI DOUBLE
)
LANGUAGE SQL AS
$$
DECLARE
  res RESULTSET;
BEGIN
  res := (SELECT b.* FROM
    (SELECT
        MOD(HASH(unique_id), :MAX_BATCHES) AS gp,
        unique_id,
        ds,
        y
     FROM IDENTIFIER(:INPUT_DATA)) a,
    TABLE(nixtla_detect_anomalies_batch(:PARAMS, unique_id, ds, y) OVER (PARTITION BY gp ORDER BY unique_id, ds)) b);
  RETURN TABLE(res);
END;
$$
;

//<br>

CREATE OR REPLACE PROCEDURE {ds_prefix}NIXTLA_EXPLAIN(
    "INPUT_DATA" VARCHAR,
    "PARAMS" OBJECT,
    "MAX_BATCHES" NUMBER(38,0) DEFAULT 1000
)
RETURNS TABLE (
  UNIQUE_ID VARCHAR, DS TIMESTAMP, FORECAST DOUBLE, FEATURE VARCHAR, CONTRIBUTION DOUBLE
)
LANGUAGE SQL AS
$$
DECLARE
  res RESULTSET;
BEGIN
  res := (SELECT b.* FROM
    (SELECT
        MOD(HASH(unique_id), :MAX_BATCHES) AS gp,
        unique_id,
        ds,
        OBJECT_CONSTRUCT_KEEP_NULL(*) AS data_obj
      FROM IDENTIFIER(:INPUT_DATA)) a,
    TABLE(nixtla_explain_batch(:PARAMS, data_obj) OVER (PARTITION BY gp ORDER BY unique_id, ds)) b);
  RETURN TABLE(res);
END;
$$
;
"""


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DeploymentConfig:
    """Configuration for Nixtla Snowflake deployment."""

    database: str
    schema: str
    stage: str
    integration_name: str = "nixtla_access_integration"
    base_url: str = "https://api.nixtla.io"
    network_rule_name: str = "nixtla_network_rule"

    @property
    def prefix(self) -> str:
        """Return fully qualified prefix for Snowflake objects."""
        return f"{self.database}.{self.schema}."

    @property
    def api_host(self) -> str:
        """Extract hostname from base_url for network rules."""
        return urlparse(self.base_url).netloc

    def get_security_params(self) -> dict:
        """Generate security parameters for UDTFs and stored procedures."""
        return {
            "secrets": {
                SECRET_API_KEY: f"{self.prefix}{SECRET_API_KEY}",
                SECRET_BASE_URL: f"{self.prefix}{SECRET_BASE_URL}",
            },
            "external_access_integrations": [self.integration_name],
        }


@dataclass
class ExampleTestCase:
    """Test case definition for validating Snowflake deployment examples."""

    name: str
    description: str
    sql_query: str
    input_table: str
    nixtla_method: str  # 'forecast', 'detect_anomalies', 'evaluation'
    nixtla_params: dict
    compare_columns: list[str]  # Columns to compare between Snowflake and direct client


# ============================================================================
# Helper Functions
# ============================================================================


def ask_with_defaults(question: str, *defaults, password: bool = False) -> str:
    """
    Ask user a question with multiple default value sources.

    Args:
        question: Question to ask the user
        *defaults: Callables that return default values (tried in order)
        password: Whether to hide input

    Returns:
        User's answer or first available default
    """
    default_value = None
    for get_default in defaults:
        result = get_default()
        if result:
            default_value = result
            break

    if password and default_value:
        question += " (press enter to use the default value)"
    elif password:
        question += " (must provide, default value was not found)"

    while True:
        answer = Prompt.ask(
            question,
            default=default_value,
            password=password,
            show_default=not password and default_value is not None,
        )
        if answer:
            return answer
        print("Please provide a valid answer.")


def show_available_objects(session: Session, object_type: str) -> None:
    """Display available Snowflake objects of given type."""
    try:
        if object_type == "DATABASES":
            objects = [row[1] for row in session.sql("SHOW DATABASES").collect()]
            print(f"[cyan]Available databases: {', '.join(objects)}[/cyan]")
        elif object_type == "SCHEMAS":
            objects = [row[1] for row in session.sql("SHOW SCHEMAS").collect()]
            preview = objects[:10]
            suffix = "..." if len(objects) > 10 else ""
            print(f"[cyan]Available schemas: {', '.join(preview)}{suffix}[/cyan]")
    except Exception:
        pass


def ensure_snowflake_object_exists(
    session: Session,
    object_type: str,
    name: str,
    create_sql: str,
    check_sql: str,
) -> bool:
    """
    Ensure a Snowflake object exists, creating it if necessary.

    Args:
        session: Active Snowflake session
        object_type: Type of object (e.g., "Database", "Schema", "Stage")
        name: Name of the object
        create_sql: SQL to create the object
        check_sql: SQL to check if object exists

    Returns:
        True if object exists or was created successfully, False otherwise
    """
    try:
        session.sql(check_sql).collect()
        print(f"[cyan]{object_type} '{name}' exists.[/cyan]")
        return True
    except Exception:
        print(f"[yellow]{object_type} '{name}' does not exist.[/yellow]")
        if not Confirm.ask(
            f"Do you want to create {object_type.lower()} '{name}'?", default=True
        ):
            return False

        try:
            session.sql(create_sql).collect()
            print(f"[green]{object_type} '{name}' created successfully![/green]")
            return True
        except Exception as create_error:
            print(f"[red]Failed to create {object_type.lower()}: {create_error}[/red]")
            return False


def execute_sql_script(
    session: Session, script: str, separator: str = "//<br>"
) -> None:
    """
    Execute a multi-statement SQL script.

    Args:
        session: Active Snowflake session
        script: SQL script with statements separated by separator
        separator: Statement separator in the script
    """
    for query in script.split(separator):
        query = query.strip().rstrip(";")
        if query:
            session.sql(query).collect()


# ============================================================================
# Session Management
# ============================================================================


def create_snowflake_session(connection_name: str) -> Session:
    """
    Create Snowflake session with MFA support.

    Args:
        connection_name: Name of connection from ~/.snowflake/config.toml

    Returns:
        Active Snowflake session
    """
    session_builder = Session.builder.config("connection_name", connection_name)

    try:
        return session_builder.create()
    except Exception as e:
        if "MFA with TOTP is required" in str(e) or "250001" in str(e):
            print("[yellow]MFA authentication required[/yellow]")
            passcode = Prompt.ask(
                "Enter your MFA code from authenticator app", password=False
            )
            return session_builder.config("passcode", passcode).create()
        raise


def get_config_from_session(session: Session) -> tuple[Optional[str], Optional[str]]:
    """
    Extract database and schema from session configuration.

    Args:
        session: Active Snowflake session

    Returns:
        Tuple of (database, schema) from config file
    """
    try:
        conn_params = session._conn._conn  # Access underlying connection
        config_database = getattr(conn_params, "database", None)
        config_schema = getattr(conn_params, "schema", None)
        return config_database, config_schema
    except Exception:
        return None, None


# ============================================================================
# Infrastructure Setup
# ============================================================================


def setup_database(
    session: Session,
    database_arg: Optional[str],
    config_database: Optional[str],
) -> Optional[str]:
    """
    Ensure database exists and is active.

    Args:
        session: Active Snowflake session
        database_arg: Database name from CLI args
        config_database: Database from config file

    Returns:
        Name of the active database, or None if setup failed
    """
    show_available_objects(session, "DATABASES")

    database = ask_with_defaults(
        "Snowflake database: ",
        lambda: database_arg,
        lambda: config_database,
        lambda: session.get_current_database(),
    )

    # Check if database exists, create if needed
    if not ensure_snowflake_object_exists(
        session,
        "Database",
        database,
        f"CREATE DATABASE IF NOT EXISTS {database}",
        f"DESC DATABASE {database}",
    ):
        return None

    session.use_database(database)
    return database


def setup_schema(
    session: Session,
    database: str,
    schema_arg: Optional[str],
    config_schema: Optional[str],
) -> Optional[str]:
    """
    Ensure schema exists and is active.

    Args:
        session: Active Snowflake session
        database: Database name
        schema_arg: Schema name from CLI args
        config_schema: Schema from config file

    Returns:
        Name of the active schema, or None if setup failed
    """
    show_available_objects(session, "SCHEMAS")

    schema = ask_with_defaults(
        "Snowflake schema: ",
        lambda: schema_arg,
        lambda: config_schema,
        lambda: session.get_current_schema(),
    )

    # Check if schema exists, create if needed
    if not ensure_snowflake_object_exists(
        session,
        "Schema",
        schema,
        f"CREATE SCHEMA IF NOT EXISTS {schema}",
        f"DESC SCHEMA {database}.{schema}",
    ):
        return None

    session.use_schema(schema)
    return schema


def setup_stage(session: Session, stage_path_arg: Optional[str]) -> Optional[str]:
    """
    Ensure stage exists.

    Args:
        session: Active Snowflake session
        stage_path_arg: Stage path from CLI args

    Returns:
        Name of the stage, or None if setup failed
    """
    stage = ask_with_defaults(
        "Stage path (without @) for the artifact",
        lambda: stage_path_arg,
    )

    # Check if stage exists, create if needed
    if not ensure_snowflake_object_exists(
        session,
        "Stage",
        stage,
        f"CREATE STAGE IF NOT EXISTS {stage}",
        f"DESC STAGE {stage}",
    ):
        return None

    return stage


# ============================================================================
# Package Management
# ============================================================================


def detect_package_installer() -> tuple[list[str], bool]:
    """
    Detect available package installer (uv, pip, pip3, pip).

    Returns:
        Tuple of (install_command, is_uv)
        where install_command is the command to run pip install
        and is_uv indicates if uv is being used
    """
    # Option 1: Check if uv is available (UV-managed projects)
    if shutil.which("uv") is not None:
        print("[cyan]Detected UV environment, using 'uv pip install'[/cyan]")
        return ["uv", "pip", "install"], True

    # Option 2: Try standard pip
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            print("[cyan]Using standard pip[/cyan]")
            return [sys.executable, "-m", "pip", "install"], False
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Option 3: Try pip3 command directly
    if shutil.which("pip3") is not None:
        print("[cyan]Using pip3 command[/cyan]")
        return ["pip3", "install"], False

    # Option 4: Try pip command directly
    if shutil.which("pip") is not None:
        print("[cyan]Using pip command[/cyan]")
        return ["pip", "install"], False

    raise RuntimeError(
        "No package installer found. Please install pip, pip3, or uv to continue."
    )


def package_and_upload_nixtla(session: Session, stage: str) -> None:
    """
    Package nixtla client and upload to Snowflake stage.

    Args:
        session: Active Snowflake session
        stage: Stage name to upload to
    """
    with TemporaryDirectory() as tmpdir:
        # Import version from nixtla package
        from nixtla import __version__ as nixtla_version

        # Detect package installer
        pip_cmd, use_uv = detect_package_installer()

        # Install packages with appropriate flags
        # UV uses --target instead of -t
        install_args = pip_cmd + [
            "--target" if use_uv else "-t",
            tmpdir,
            f"nixtla=={nixtla_version}",
            "utilsforecast",
            "httpx",
            "--no-deps",  # Avoid pulling in heavy things like pandas/numpy into the ZIP
        ]

        subprocess.run(install_args, check=True)

        # Create zip archive
        shutil.make_archive(os.path.join(tmpdir, "nixtla"), "zip", tmpdir)
        zip_path = os.path.join(tmpdir, "nixtla.zip")

        # Upload to stage (normalize path for Windows and quote file URI)
        zip_posix = Path(zip_path).resolve().as_posix()
        # On Windows, file URIs conventionally start with file:///C:/...
        file_uri = f"file:///{zip_posix}" if os.name == "nt" else f"file://{zip_posix}"
        result = session.sql(
            f"PUT '{file_uri}' @{stage} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        ).collect()

        if result[0]["status"].lower() != "uploaded":
            raise ValueError("Upload failed " + str(result))

        print("[green]Nixtla client package uploaded[/green]")


# ============================================================================
# Security Integration
# ============================================================================


def create_security_integration(
    session: Session,
    config: DeploymentConfig,
    api_key: Optional[str] = None,
    skip_confirmation: bool = False,
) -> None:
    """
    Create network rules, secrets, and external access integration.

    Args:
        session: Active Snowflake session
        config: Deployment configuration
        api_key: Nixtla API key (if None, will prompt or use env var)
        skip_confirmation: If True, skip interactive confirmation prompts
    """
    # Ensure session is using the correct database and schema context
    session.use_database(config.database)
    session.use_schema(config.schema)

    if api_key is None:
        nixtla_api_key = ask_with_defaults(
            "Nixtla API key: ",
            lambda: os.environ.get("NIXTLA_API_KEY"),
            password=True,
        )
    else:
        nixtla_api_key = api_key

    # Clean and escape the API key
    # Strip whitespace and quotes that might have been accidentally entered
    nixtla_api_key = nixtla_api_key.strip().strip("'\"")

    # Escape single quotes for SQL (replace ' with '')
    nixtla_api_key_escaped = nixtla_api_key.replace("'", "''")

    script = TEMPLATE_ACCESS_INTEGRATION.format(
        ds_prefix=config.prefix,
        nixtla_api_key=nixtla_api_key_escaped,
        integration_name=config.integration_name,
        api_host=config.api_host,
        base_url=config.base_url,
        network_rule_name=config.network_rule_name,
    )

    if not skip_confirmation:
        print(Markdown(f"```sql\n{script}\n```"))

    if skip_confirmation or Confirm.ask(
        "Do you want to run the script now?", default=False
    ):
        execute_sql_script(session, script)
        if not skip_confirmation:
            print(
                f"[green]Access integration '{config.integration_name}' created![/green]"
            )


# ============================================================================
# UDTF Creation
# ============================================================================


def create_udtfs(session: Session, config: DeploymentConfig) -> None:
    """
    Deploy UDTFs for forecasting and evaluation.

    Args:
        session: Active Snowflake session
        config: Deployment configuration with stage and integration name
    """
    # Ensure session is using the correct database and schema context
    session.use_database(config.database)
    session.use_schema(config.schema)

    import pandas as pd
    from snowflake.snowpark.functions import udtf
    from snowflake.snowpark.types import (
        ArrayType,
        DoubleType,
        MapType,
        StringType,
        StructField,
        StructType,
        TimestampType,
        VariantType,
    )

    security_params = config.get_security_params()

    common_params = {
        "session": session,
        "packages": PACKAGES,
        "imports": [f"@{config.stage}/nixtla.zip"],
        "replace": True,
        "immutable": True,
        "is_permanent": True,
        "stage_location": config.stage,
    }

    # Forecast UDTF (with exogenous variables support)
    @udtf(
        input_types=[MapType(), MapType()],
        output_schema=StructType(
            [
                StructField("unique_id", StringType()),
                StructField("ds", TimestampType()),
                StructField("forecast", DoubleType()),
                StructField("confidence_intervals", VariantType()),
            ]
        ),
        name="nixtla_forecast_batch",
        **security_params,
        **common_params,
    )
    class ForecastUDTF:
        def __init__(self):
            import _snowflake
            from nixtla import NixtlaClient

            token = _snowflake.get_generic_secret_string(SECRET_API_KEY)
            base_url = _snowflake.get_generic_secret_string(SECRET_BASE_URL)
            self.client = NixtlaClient(api_key=token, base_url=base_url)

        @staticmethod
        def _find_forecast_column(forecast_df: pd.DataFrame) -> str:
            """
            Find the main forecast column from the forecast result.

            Prefers custom model names over 'TimeGPT', excluding interval columns.

            Args:
                forecast_df: DataFrame returned from forecast API

            Returns:
                Name of the forecast column

            Raises:
                ValueError: If no valid forecast column is found
            """
            base_cols = set(BASE_COLUMNS)

            # First, try to find custom model column (not TimeGPT, not intervals)
            for col in forecast_df.columns:
                col_lower = col.lower()
                if col_lower not in base_cols and not col.startswith("TimeGPT"):
                    return col

            # Fall back to TimeGPT main forecast column (not intervals)
            for col in forecast_df.columns:
                if (
                    col.startswith("TimeGPT")
                    and "-lo-" not in col
                    and "-hi-" not in col
                ):
                    return col

            raise ValueError("No valid forecast column found in output")

        @staticmethod
        def _extract_confidence_intervals(forecast_df: pd.DataFrame) -> pd.Series:
            """
            Extract confidence intervals from forecast result into VARIANT format.

            Looks for columns matching pattern: *-lo-<level> and *-hi-<level>
            Returns nested dict structure: {"80": {"lo": val, "hi": val}, "95": {...}}

            Args:
                forecast_df: DataFrame returned from forecast API

            Returns:
                Series containing dict/None for each row's confidence intervals
            """
            import re

            # Find all interval columns using regex
            interval_pattern = re.compile(r"^.*-(lo|hi)-(\d+)$", re.IGNORECASE)
            interval_cols = {}  # Maps (level, bound) -> column_name

            for col in forecast_df.columns:
                match = interval_pattern.match(col)
                if match:
                    bound_type = match.group(1).lower()  # "lo" or "hi"
                    level = match.group(2)  # "80", "95", etc.
                    interval_cols[(level, bound_type)] = col

            if not interval_cols:
                # No confidence intervals found
                return pd.Series([None] * len(forecast_df))

            # Extract unique levels and sort them
            levels = sorted(set(level for level, _ in interval_cols.keys()))

            def build_interval_dict(row):
                """Build nested dict structure for a single row."""
                intervals = {}
                for level in levels:
                    level_data = {}

                    # Extract lo and hi values for this level
                    lo_col = interval_cols.get((level, "lo"))
                    hi_col = interval_cols.get((level, "hi"))

                    if lo_col and lo_col in row.index:
                        val = row[lo_col]
                        level_data["lo"] = float(val) if pd.notna(val) else None
                    if hi_col and hi_col in row.index:
                        val = row[hi_col]
                        level_data["hi"] = float(val) if pd.notna(val) else None

                    if level_data:
                        intervals[level] = level_data

                # Return as Python dict - Snowflake VARIANT will handle serialization
                return intervals if intervals else None

            return forecast_df.apply(build_interval_dict, axis=1)

        def end_partition(self, df: pd.DataFrame) -> pd.DataFrame:
            """Execute forecast for a partition of data."""
            # The vectorized UDTF receives df with 2 columns matching input_types:
            #   Column 0 (MapType): PARAMS config object, same for every row
            #   Column 1 (MapType): data_obj from OBJECT_CONSTRUCT_KEEP_NULL(*),
            #                       each row is a dict of the original table columns
            config = df.iloc[0, 0]
            if config.get("finetune_steps", 0) > 0:
                raise ValueError("Finetuning is not allowed during forecasting")

            h = config.get("h")
            if h is None:
                raise ValueError("Parameter 'h' (forecast horizon) is required")

            # Reconstruct the tabular DataFrame from the per-row dicts in column 1
            data = pd.DataFrame(df.iloc[:, 1].tolist())
            data.columns = data.columns.str.lower()
            data = data.sort_values(by=["unique_id", "ds"]).reset_index(drop=True)
            data["ds"] = pd.to_datetime(data["ds"])

            # Check for future exogenous variables
            futr_exog_list = config.get("futr_exog_list")
            X_df = None

            if futr_exog_list:
                futr_exog_list = [col.lower() for col in futr_exog_list]

                # Separate historical and future exog data based on h parameter
                # For each series, the last h rows are future exog data
                future_mask = data.groupby("unique_id").cumcount(ascending=False) < h
                hist_data = data[~future_mask].copy()
                future_exog_data = data[future_mask].copy()

                if not future_exog_data.empty:
                    # Prepare X_df with unique_id, ds, and future exog columns
                    X_df = future_exog_data[["unique_id", "ds"] + futr_exog_list].copy()
                    print(
                        f"Using future exogenous variables: {futr_exog_list} "
                        f"({len(X_df)} rows for {X_df['unique_id'].nunique()} series)"
                    )

                data = hist_data
            else:
                # No future exog list - filter out any rows where y is null
                data = data[data["y"].notna()].copy()

            # Handle historical exogenous variables
            extra_cols = [col for col in data.columns if col not in BASE_COLUMNS]
            hist_exog_list = config.get("hist_exog_list")

            forecast_params = dict(config)
            # Remove futr_exog_list from params - we've already processed it to create X_df
            forecast_params.pop("futr_exog_list", None)

            if hist_exog_list:
                # Normalize and validate declared exogenous variables
                # Store original names for error messages before lowercasing
                original_hist_exog = list(hist_exog_list)
                hist_exog_list = [col.lower() for col in hist_exog_list]
                missing_cols = [
                    col for col in hist_exog_list if col not in data.columns
                ]

                if missing_cols:
                    # Show both original (user-specified) and normalized names for clarity
                    raise ValueError(
                        f"Columns specified in hist_exog_list not found in data: {missing_cols}. "
                        f"Original specification: {original_hist_exog}. "
                        f"Available columns (lowercase): {list(data.columns)}"
                    )

                forecast_params["hist_exog_list"] = hist_exog_list

            elif extra_cols:
                # Warn about unused columns
                exog_example = ", ".join(repr(c) for c in extra_cols)
                print(
                    f"Warning: Found extra columns {extra_cols} in input data. "
                    f"To use them as exogenous variables, add 'hist_exog_list' to PARAMS: "
                    f"PARAMS => OBJECT_CONSTRUCT('h', <value>, 'hist_exog_list', ARRAY_CONSTRUCT({exog_example}))"
                )

            # Add future exogenous data if available
            if X_df is not None:
                forecast_params["X_df"] = X_df

            # Execute forecast
            forecast = self.client.forecast(df=data, **forecast_params)

            # Extract forecast column and build result
            forecast_col = self._find_forecast_column(forecast)
            result = forecast[["unique_id", "ds", forecast_col]].copy()
            result = result.rename(columns={forecast_col: FORECAST_COLUMN})

            # Extract confidence intervals
            result["confidence_intervals"] = self._extract_confidence_intervals(
                forecast
            )

            return result[["unique_id", "ds", "forecast", "confidence_intervals"]]

        end_partition._sf_vectorized_input = pd.DataFrame  # type: ignore

    # Evaluate UDTF
    @udtf(
        input_types=[ArrayType(), MapType()],
        output_schema=StructType(
            [
                StructField("unique_id", StringType()),
                StructField("forecaster", StringType()),
                StructField("metric", StringType()),
                StructField("value", DoubleType()),
            ]
        ),
        name="nixtla_evaluate_batch",
        **common_params,
    )
    class EvaluateUDTF:
        @staticmethod
        def _prepare_input_data(df: pd.DataFrame) -> pd.DataFrame:
            """Extract and prepare evaluation data from UDTF input."""
            data = pd.DataFrame(df.iloc[:, 1].tolist())
            data.columns = data.columns.str.lower()
            data = data.sort_values(by=["unique_id", "ds"])
            data["ds"] = pd.to_datetime(data["ds"])
            return data

        @staticmethod
        def _find_forecaster_columns(data: pd.DataFrame) -> list[str]:
            """Find all forecaster columns (non-base columns)."""
            return [col for col in data.columns if col.lower() not in BASE_COLUMNS]

        @staticmethod
        def _format_evaluation_results(result: pd.DataFrame) -> pd.DataFrame:
            """Melt evaluation results to match output schema."""
            melted = pd.melt(
                result,
                id_vars=["unique_id", "metric"],
                var_name="forecaster",
                value_name="value",
            )
            return melted[["unique_id", "forecaster", "metric", "value"]]

        def end_partition(self, df: pd.DataFrame) -> pd.DataFrame:
            """Execute evaluation for a partition of data."""
            from utilsforecast.evaluation import evaluate

            # Extract config and prepare data
            metrics = list(self._parse_metrics(df.iloc[0, 0]))
            data = self._prepare_input_data(df)
            forecasters = self._find_forecaster_columns(data)

            # Evaluate and format results
            result = evaluate(data, metrics=metrics, models=forecasters)
            return self._format_evaluation_results(result)

        def _parse_metrics(self, metrics):
            """Parse metric names to metric functions."""
            from utilsforecast import losses

            metric_map = {
                "mape": losses.mape,
                "mae": losses.mae,
                "mse": losses.mse,
            }

            for metric_name in metrics:
                metric_key = metric_name.lower()
                if metric_key in metric_map:
                    yield metric_map[metric_key]
                else:
                    raise NotImplementedError(f"Unsupported metric {metric_name}")

        end_partition._sf_vectorized_input = pd.DataFrame  # type: ignore

    # Anomaly Detection UDTF
    @udtf(
        input_types=[MapType(), StringType(), TimestampType(), DoubleType()],
        output_schema=StructType(
            [
                StructField("unique_id", StringType()),
                StructField("ds", TimestampType()),
                StructField("y", DoubleType()),
                StructField("TimeGPT", DoubleType()),
                StructField("anomaly", StringType()),
                StructField("TimeGPT_lo", DoubleType()),
                StructField("TimeGPT_hi", DoubleType()),
            ]
        ),
        name="nixtla_detect_anomalies_batch",
        **security_params,
        **common_params,
    )
    class AnomalyDetectionUDTF:
        def __init__(self):
            import _snowflake
            from nixtla import NixtlaClient

            token = _snowflake.get_generic_secret_string(SECRET_API_KEY)
            base_url = _snowflake.get_generic_secret_string(SECRET_BASE_URL)
            self.client = NixtlaClient(api_key=token, base_url=base_url)

        @staticmethod
        def _prepare_input_data(df: pd.DataFrame) -> pd.DataFrame:
            """Extract anomaly detection input from UDTF DataFrame."""
            return pd.DataFrame(
                {
                    "unique_id": df.iloc[:, 1],
                    "ds": df.iloc[:, 2],
                    "y": df.iloc[:, 3],
                }
            )

        @staticmethod
        def _extract_confidence_bounds(
            anomalies: pd.DataFrame, level: int
        ) -> pd.DataFrame:
            """Extract and rename confidence interval columns."""
            lo_col = f"TimeGPT-lo-{level}"
            hi_col = f"TimeGPT-hi-{level}"

            result_cols = ["unique_id", "ds", "y", "TimeGPT", "anomaly"]

            if lo_col in anomalies.columns and hi_col in anomalies.columns:
                result = anomalies[result_cols + [lo_col, hi_col]].copy()
            else:
                result = anomalies[result_cols].copy()
                result[lo_col] = None
                result[hi_col] = None

            # Rename to match output schema
            result.columns = [
                "unique_id",
                "ds",
                "y",
                "TimeGPT",
                "anomaly",
                "TimeGPT_lo",
                "TimeGPT_hi",
            ]
            return result

        def end_partition(self, df: pd.DataFrame) -> pd.DataFrame:
            """Execute anomaly detection for a partition."""
            # Prepare inputs
            input_df = self._prepare_input_data(df)
            config = df.iloc[0, 0]
            level = config.get("level", 99)

            # Detect anomalies
            anomalies = self.client.detect_anomalies(df=input_df, **config)

            # Extract bounds and format output
            result = self._extract_confidence_bounds(anomalies, level)
            result[ANOMALY_COLUMN] = result[ANOMALY_COLUMN].astype(str)

            return result

        end_partition._sf_vectorized_input = pd.DataFrame  # type: ignore

    # Explain UDTF (feature contributions / SHAP values)
    @udtf(
        input_types=[MapType(), MapType()],
        output_schema=StructType(
            [
                StructField("unique_id", StringType()),
                StructField("ds", TimestampType()),
                StructField("forecast", DoubleType()),
                StructField("feature", StringType()),
                StructField("contribution", DoubleType()),
            ]
        ),
        name="nixtla_explain_batch",
        **security_params,
        **common_params,
    )
    class ExplainUDTF:
        """UDTF for computing feature contributions (SHAP values) in long format."""

        def __init__(self):
            import _snowflake
            from nixtla import NixtlaClient

            token = _snowflake.get_generic_secret_string(SECRET_API_KEY)
            base_url = _snowflake.get_generic_secret_string(SECRET_BASE_URL)
            self.client = NixtlaClient(api_key=token, base_url=base_url)

        def end_partition(self, df: pd.DataFrame) -> pd.DataFrame:
            """Execute forecast with feature contributions and return in long format."""
            # The vectorized UDTF receives df with 2 columns matching input_types:
            #   Column 0 (MapType): PARAMS config object, same for every row
            #   Column 1 (MapType): data_obj from OBJECT_CONSTRUCT_KEEP_NULL(*),
            #                       each row is a dict of the original table columns
            config = dict(df.iloc[0, 0])
            if config.get("finetune_steps", 0) > 0:
                raise ValueError("Finetuning is not allowed during explain")

            h = config.get("h")
            if h is None:
                raise ValueError("Parameter 'h' (forecast horizon) is required")

            # Reconstruct the tabular DataFrame from the per-row dicts in column 1
            data = pd.DataFrame(df.iloc[:, 1].tolist())
            data.columns = data.columns.str.lower()
            data = data.sort_values(by=["unique_id", "ds"]).reset_index(drop=True)
            data["ds"] = pd.to_datetime(data["ds"])

            # Handle future exogenous variables
            futr_exog_list = config.get("futr_exog_list")
            X_df = None

            if futr_exog_list:
                futr_exog_list = [col.lower() for col in futr_exog_list]
                future_mask = data.groupby("unique_id").cumcount(ascending=False) < h
                hist_data = data[~future_mask].copy()
                future_exog_data = data[future_mask].copy()

                if not future_exog_data.empty:
                    X_df = future_exog_data[["unique_id", "ds"] + futr_exog_list].copy()

                data = hist_data
            else:
                data = data[data["y"].notna()].copy()

            # Handle historical exogenous variables
            hist_exog_list = config.get("hist_exog_list")

            forecast_params = dict(config)
            forecast_params.pop("futr_exog_list", None)

            if hist_exog_list:
                # Store original names for error messages before lowercasing
                original_hist_exog = list(hist_exog_list)
                hist_exog_list = [col.lower() for col in hist_exog_list]
                missing_cols = [
                    col for col in hist_exog_list if col not in data.columns
                ]
                if missing_cols:
                    # Show both original (user-specified) and normalized names for clarity
                    raise ValueError(
                        f"Columns specified in hist_exog_list not found in data: {missing_cols}. "
                        f"Original specification: {original_hist_exog}. "
                        f"Available columns (lowercase): {list(data.columns)}"
                    )
                forecast_params["hist_exog_list"] = hist_exog_list

            if X_df is not None:
                forecast_params["X_df"] = X_df

            # Force feature_contributions to True
            forecast_params["feature_contributions"] = True

            # Execute forecast
            _ = self.client.forecast(df=data, **forecast_params)

            # Check if feature_contributions were computed
            if (
                not hasattr(self.client, "feature_contributions")
                or self.client.feature_contributions is None
            ):
                raise ValueError(
                    "Feature contributions were not computed. "
                    "Ensure exogenous variables are provided via hist_exog_list or futr_exog_list."
                )

            contrib_df = self.client.feature_contributions

            # Identify feature columns (everything except unique_id, ds, TimeGPT)
            id_cols = ["unique_id", "ds"]
            forecast_col = "TimeGPT"
            feature_cols = [
                col
                for col in contrib_df.columns
                if col not in id_cols and col != forecast_col
            ]

            if not feature_cols:
                raise ValueError(
                    "No feature contribution columns found. "
                    "Ensure exogenous variables are provided."
                )

            # Melt to long format
            result = pd.melt(
                contrib_df,
                id_vars=id_cols + [forecast_col],
                value_vars=feature_cols,
                var_name="feature",
                value_name="contribution",
            )

            # Rename forecast column to match output schema
            result = result.rename(columns={forecast_col: "forecast"})

            # Reorder columns to match schema
            result = result[["unique_id", "ds", "forecast", "feature", "contribution"]]

            return result

        end_partition._sf_vectorized_input = pd.DataFrame  # type: ignore

    print("[green]UDTFs created![/green]")


# ============================================================================
# Stored Procedures
# ============================================================================


def create_stored_procedures(
    session: Session, config: DeploymentConfig, skip_confirmation: bool = False
) -> None:
    """
    Deploy SQL stored procedures for inference and evaluation.

    Args:
        session: Active Snowflake session
        config: Deployment configuration
        skip_confirmation: If True, skip interactive confirmation prompts
    """
    # Ensure session is using the correct database and schema context
    session.use_database(config.database)
    session.use_schema(config.schema)

    script = TEMPLATE_SP.format(ds_prefix=config.prefix)

    if not skip_confirmation:
        print(Markdown(f"```sql\n{script}\n```"))

    if skip_confirmation or Confirm.ask(
        "Do you want to run the script now?", default=False
    ):
        execute_sql_script(session, script)
        if not skip_confirmation:
            print("[green]Stored procedures created![/green]")


def create_finetune_sproc(session: Session, config: DeploymentConfig) -> None:
    """
    Deploy stored procedure for finetuning.

    Args:
        session: Active Snowflake session
        config: Deployment configuration with stage and integration name
    """
    # Ensure session is using the correct database and schema context
    session.use_database(config.database)
    session.use_schema(config.schema)

    from snowflake.snowpark.functions import sproc

    security_params = config.get_security_params()

    @sproc(
        session=session,
        name="nixtla_finetune",
        packages=PACKAGES,
        imports=[f"@{config.stage}/nixtla.zip"],
        replace=True,
        is_permanent=True,
        stage_location=config.stage,
        **security_params,
    )
    def nixtla_finetune(
        session: Session,
        input_data: str,
        params: Optional[dict] = None,
        max_series: int = 1000,
    ) -> str:
        import _snowflake
        from snowflake.snowpark import functions as F
        from nixtla import NixtlaClient

        if params is None:
            params = {}

        token = _snowflake.get_generic_secret_string("nixtla_api_key")
        base_url = _snowflake.get_generic_secret_string("nixtla_base_url")
        client = NixtlaClient(api_key=token, base_url=base_url)

        input_table = session.table(input_data)
        ids = (
            input_table.select("unique_id", F.hash("unique_id").alias("_od"))
            .distinct()
            .order_by("_od")
            .limit(max_series)
        )

        train_data = (
            input_table.join(ids, on="unique_id", how="inner", rsuffix="_")
            .select("unique_id", "ds", "y")
            .order_by("unique_id", "ds")
            .to_pandas()
        )

        train_data.columns = train_data.columns.str.lower()

        if "finetune_steps" not in params:
            params["finetune_steps"] = train_data["unique_id"].nunique()

        return client.finetune(train_data, **params)

    print("[green]Finetune stored procedure created![/green]")


# ============================================================================
# Example Data
# ============================================================================


def show_example_usage_scripts(config: DeploymentConfig) -> None:
    """
    Display sample SQL scripts showing how to use the example datasets.

    Uses test case definitions from get_example_test_cases() to ensure
    displayed examples match what's tested.

    Args:
        config: Deployment configuration containing database, schema, etc.
    """
    # Get test cases using the deployment config
    test_cases = get_example_test_cases(config)

    # Map of test case names to display in the UI (filtering and ordering)
    display_cases = [
        "basic_forecast",
        "evaluation_metrics",
        "anomaly_detection",
        "forecast_with_future_exog",
        "explain_forecast",
    ]

    # Number emojis
    numbers = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]

    # Header
    print(
        "\n╔══════════════════════════════════════════════════════════════════════════════╗"
    )
    print(
        "║                        📚 SAMPLE USAGE SCRIPTS                               ║"
    )
    print(
        "╚══════════════════════════════════════════════════════════════════════════════╝\n"
    )
    print("Copy and paste these SQL scripts to test your Nixtla deployment!\n")

    # Display each test case
    for idx, case_name in enumerate(display_cases):
        test_case = next((tc for tc in test_cases if tc.name == case_name), None)
        if not test_case:
            continue

        print(
            "┌──────────────────────────────────────────────────────────────────────────────┐"
        )
        print(f"│ {numbers[idx]}  {test_case.description.upper():<74} │")
        print(
            "└──────────────────────────────────────────────────────────────────────────────┘\n"
        )
        print(test_case.sql_query)
        print("\n")

    # Footer
    print(
        "╔══════════════════════════════════════════════════════════════════════════════╗"
    )
    print(
        "║                    🎉 Ready to start forecasting!                            ║"
    )
    print(
        "╚══════════════════════════════════════════════════════════════════════════════╝"
    )


def get_example_test_cases(config: DeploymentConfig) -> list[ExampleTestCase]:
    """
    Get structured test case definitions for validating Snowflake deployment.

    These test cases can be used to:
    1. Execute SQL queries via Snowflake UDTFs/stored procedures
    2. Execute equivalent operations with NixtlaClient on DataFrames
    3. Compare results to validate deployment correctness

    Args:
        config: Deployment configuration with database, schema, etc.

    Returns:
        List of ExampleTestCase objects defining test scenarios
    """
    prefix = config.prefix

    return [
        ExampleTestCase(
            name="basic_forecast",
            description="Forecast 14 days with confidence intervals (80%, 95%)",
            sql_query=f"""CALL {prefix}NIXTLA_FORECAST(
    INPUT_DATA => '{prefix}EXAMPLE_TRAIN',
    PARAMS => OBJECT_CONSTRUCT(
        'h', 14,
        'freq', 'D',
        'level', ARRAY_CONSTRUCT(80, 95)
    )
)""",
            input_table="train",
            nixtla_method="forecast",
            nixtla_params={"h": 14, "freq": "D", "level": [80, 95]},
            compare_columns=["unique_id", "ds", "TimeGPT"],
        ),
        ExampleTestCase(
            name="forecast_without_intervals",
            description="Forecast 7 days without confidence intervals",
            sql_query=f"""CALL {prefix}NIXTLA_FORECAST(
    INPUT_DATA => '{prefix}EXAMPLE_TRAIN',
    PARAMS => OBJECT_CONSTRUCT(
        'h', 7,
        'freq', 'D'
    )
)""",
            input_table="train",
            nixtla_method="forecast",
            nixtla_params={"h": 7, "freq": "D"},
            compare_columns=["unique_id", "ds", "TimeGPT"],
        ),
        ExampleTestCase(
            name="anomaly_detection",
            description="Detect anomalies with 95% confidence level",
            sql_query=f"""CALL {prefix}NIXTLA_DETECT_ANOMALIES(
    INPUT_DATA => '{prefix}EXAMPLE_ANOMALY_DATA',
    PARAMS => OBJECT_CONSTRUCT(
        'level', 95,
        'freq', 'D'
    )
)""",
            input_table="anomaly",
            nixtla_method="detect_anomalies",
            nixtla_params={"level": 95, "freq": "D"},
            compare_columns=["unique_id", "ds", "y", "anomaly"],
        ),
        ExampleTestCase(
            name="forecast_with_future_exog",
            description="Forecast with historical and future exogenous variables",
            sql_query=f"""CALL {prefix}NIXTLA_FORECAST(
    INPUT_DATA => '{prefix}EXAMPLE_TRAIN',
    PARAMS => OBJECT_CONSTRUCT(
        'h', 14,
        'freq', 'D',
        'hist_exog_list', ARRAY_CONSTRUCT('temperature'),
        'futr_exog_list', ARRAY_CONSTRUCT('is_weekend', 'promotion')
    ),
    MAX_BATCHES => 1000
)""",
            input_table="train",
            nixtla_method="forecast",
            nixtla_params={
                "h": 14,
                "freq": "D",
                "X_df": None,  # Will be set dynamically in tests based on data
            },
            compare_columns=["unique_id", "ds", "TimeGPT"],
        ),
        ExampleTestCase(
            name="evaluation_metrics",
            description="Evaluation metrics (MAPE, MAE, MSE)",
            sql_query=f"""CALL {prefix}NIXTLA_EVALUATE(
    INPUT_DATA => '{prefix}EXAMPLE_ALL_DATA',
    METRICS => ARRAY_CONSTRUCT('MAPE', 'MAE', 'MSE')
)""",
            input_table="all_data",
            nixtla_method="evaluation",
            nixtla_params={
                # Note: The stored procedure evaluates existing predictions
                # in the data (e.g., the 'timegpt' column in EXAMPLE_ALL_DATA)
            },
            compare_columns=["unique_id", "forecaster", "metric", "value"],
        ),
        ExampleTestCase(
            name="explain_forecast",
            description="Explain forecast with historical and future exogenous variables",
            sql_query=f"""CALL {prefix}NIXTLA_EXPLAIN(
    INPUT_DATA => '{prefix}EXAMPLE_TRAIN',
    PARAMS => OBJECT_CONSTRUCT(
        'h', 14,
        'freq', 'D',
        'hist_exog_list', ARRAY_CONSTRUCT('temperature'),
        'futr_exog_list', ARRAY_CONSTRUCT('is_weekend', 'promotion')
    )
)""",
            input_table="train",
            nixtla_method="forecast",
            nixtla_params={
                "h": 14,
                "freq": "D",
                "feature_contributions": True,
                "X_df": None,  # Will be set dynamically in tests based on data
            },
            compare_columns=["unique_id", "ds", "forecast", "feature", "contribution"],
        ),
    ]


def _generate_time_series(
    series_id: str,
    dates: pd.DatetimeIndex,
    include_exog: bool = False,
    anomalies: Optional[Mapping[int, float]] = None,
    forecast_start_idx: Optional[int] = None,
    trend_rate: float = 0.5,
    seasonal_amplitude: float = 10.0,
    seasonal_type: str = "sin",
    base_value: float = 100.0,
    noise_std: float = 2.0,
) -> list[dict]:
    """
    Generate a single time series with optional exogenous variables and anomalies.

    Args:
        series_id: Unique identifier for the series
        dates: Date range for the series
        include_exog: Whether to include exogenous variables (temperature, is_weekend, promotion)
        anomalies: Dict mapping day_index -> anomaly_value to inject into the series
        forecast_start_idx: If provided, add a 'TimeGPT' forecast column for indices >= this value
        trend_rate: Rate of linear trend growth per time step (default: 0.5)
        seasonal_amplitude: Amplitude of seasonal component (default: 10.0)
        seasonal_type: Type of seasonality - 'sin' or 'cos' (default: 'sin')
        base_value: Base value for the series (default: 100.0)
        noise_std: Standard deviation of random noise (default: 2.0)

    Returns:
        List of row dicts with columns: unique_id, ds, y, [exog columns], [TimeGPT]
    """
    import numpy as np

    data = []
    for i, date in enumerate(dates):
        # Base pattern: trend + seasonality + noise
        trend = i * trend_rate
        if seasonal_type == "cos":
            seasonal = seasonal_amplitude * np.cos(2 * np.pi * i / 7)
        else:  # default to sin
            seasonal = seasonal_amplitude * np.sin(2 * np.pi * i / 7)
        noise = np.random.normal(0, noise_std)
        value = base_value + trend + seasonal + noise

        row = {
            "unique_id": series_id,
            "ds": date,
            "y": value,
        }

        # Add exogenous variables if requested
        if include_exog:
            temperature = 20 + 10 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 2)
            is_weekend = 1 if date.dayofweek >= 5 else 0
            promotion = 1 if i % 14 == 0 else 0  # Promotion every 2 weeks

            # Add exogenous effects to value
            temp_effect = (temperature - 20) * 0.5
            weekend_effect = 5 if is_weekend else 0
            promo_effect = 8 if promotion else 0
            row["y"] += temp_effect + weekend_effect + promo_effect

            row.update(
                {
                    "temperature": temperature,
                    "is_weekend": is_weekend,
                    "promotion": promotion,
                }
            )
        else:
            row.update(
                {
                    "temperature": None,
                    "is_weekend": None,
                    "promotion": None,
                }
            )

        # Inject anomalies if specified
        if anomalies and i in anomalies:
            row["y"] += anomalies[i]

        # Add forecast column if requested (for evaluation examples)
        if forecast_start_idx is not None:
            if i >= forecast_start_idx:
                # Simulated forecast with slightly more noise
                forecast_value = value + np.random.normal(0, 3)
                row["TimeGPT"] = forecast_value
            else:
                row["TimeGPT"] = None

        data.append(row)

    return data


def load_example_datasets(
    session: Session,
    config: DeploymentConfig,
    return_dataframes: bool = False,
) -> Optional[dict[str, pd.DataFrame]]:
    """
    Load example datasets for testing.

    Args:
        session: Active Snowflake session
        config: Deployment configuration (used for displaying example scripts)
        return_dataframes: If True, return the generated DataFrames before upload

    Returns:
        If return_dataframes=True, dict with keys: 'train', 'all_data', 'anomaly'
    """
    # Generate sample training data with exogenous variables
    print("[cyan]Generating sample training data with exogenous variables...[/cyan]")

    dates = pd.date_range(start="2020-01-01", periods=TOTAL_DAYS, freq="D")
    training_dates = dates[:TRAINING_DAYS]

    # Generate three series using helper function
    series_1 = _generate_time_series("series_1", training_dates, include_exog=False)
    series_2 = _generate_time_series("series_2", training_dates, include_exog=True)
    series_3 = _generate_time_series("series_3", training_dates, include_exog=False)

    # Combine all series
    train = pd.DataFrame(series_1 + series_2 + series_3)

    # Generate and append future exogenous data
    import numpy as np

    future_dates = dates[TRAINING_DAYS : TRAINING_DAYS + 14]
    future_exog_data = []
    for i, date in enumerate(future_dates):
        # Calculate day index relative to original series
        day_idx = TRAINING_DAYS + i

        temperature = (
            20 + 10 * np.sin(2 * np.pi * day_idx / 365) + np.random.normal(0, 2)
        )
        is_weekend = 1 if date.dayofweek >= 5 else 0
        promotion = 1 if day_idx % 14 == 0 else 0

        future_exog_data.append(
            {
                "unique_id": "series_2",
                "ds": date,
                "y": None,
                "temperature": temperature,
                "is_weekend": is_weekend,
                "promotion": promotion,
            }
        )
    future_exog_df = pd.DataFrame(future_exog_data)
    train = pd.concat([train, future_exog_df]).reset_index(drop=True)

    train.columns = train.columns.str.upper()
    session.write_pandas(
        train,
        "EXAMPLE_TRAIN",
        auto_create_table=True,
        overwrite=True,
        use_logical_type=True,
    )

    # Generate all_data (training + some forecasts for evaluation)
    # Include forecast column for days after training period
    all_series_1 = _generate_time_series(
        "series_1", dates, include_exog=False, forecast_start_idx=TRAINING_DAYS
    )
    all_series_2 = _generate_time_series(
        "series_2", dates, include_exog=False, forecast_start_idx=TRAINING_DAYS
    )
    all_series_3 = _generate_time_series(
        "series_3", dates, include_exog=False, forecast_start_idx=TRAINING_DAYS
    )

    all_data_df = pd.DataFrame(all_series_1 + all_series_2 + all_series_3)
    all_data_df.columns = all_data_df.columns.str.upper()
    session.write_pandas(
        all_data_df,
        "EXAMPLE_ALL_DATA",
        auto_create_table=True,
        overwrite=True,
        use_logical_type=True,
    )

    # Generate anomaly detection example data with injected anomalies
    print("[cyan]Generating anomaly detection example data...[/cyan]")

    anomaly_dates = pd.date_range(start="2020-01-01", periods=ANOMALY_DAYS, freq="D")

    # Series 1: Few extreme spikes and drops
    sensor_1_anomalies = {
        30: 50,  # Day 30: Large spike
        60: -40,  # Day 60: Large drop
        120: 45,  # Day 120: Another spike
    }
    sensor_1 = _generate_time_series(
        series_id="sensor_1",
        dates=anomaly_dates,
        anomalies=sensor_1_anomalies,
        trend_rate=0.3,
        seasonal_amplitude=15.0,
        seasonal_type="sin",
        base_value=150.0,
        noise_std=2.0,
    )

    # Series 2: Cluster of anomalies
    sensor_2_anomalies = {
        **{i: 35 for i in range(90, 96)},  # Days 90-95: Consecutive spikes
        150: -30,  # Day 150: Single drop
    }
    sensor_2 = _generate_time_series(
        series_id="sensor_2",
        dates=anomaly_dates,
        anomalies=sensor_2_anomalies,
        trend_rate=0.2,
        seasonal_amplitude=10.0,
        seasonal_type="cos",
        base_value=200.0,
        noise_std=1.5,
    )

    # Series 3: Subtle anomalies
    sensor_3_anomalies = {
        45: 25,  # Day 45: Moderate spike
        100: -20,  # Day 100: Moderate drop
    }
    sensor_3 = _generate_time_series(
        series_id="sensor_3",
        dates=anomaly_dates,
        anomalies=sensor_3_anomalies,
        trend_rate=0.4,
        seasonal_amplitude=12.0,
        seasonal_type="sin",
        base_value=180.0,
        noise_std=3.0,
    )

    anomaly_df = pd.DataFrame(sensor_1 + sensor_2 + sensor_3)
    anomaly_df.columns = anomaly_df.columns.str.upper()
    session.write_pandas(
        anomaly_df,
        "EXAMPLE_ANOMALY_DATA",
        auto_create_table=True,
        overwrite=True,
        use_logical_type=True,
    )

    print("[green]Example datasets created successfully![/green]")
    print(
        "[cyan]  - EXAMPLE_TRAIN: 3 series × 330 days + 14 future points = 990 + 14 rows[/cyan]"
    )
    print(
        "[cyan]  - EXAMPLE_ALL_DATA: 3 series × 365 days = 1095 rows (for evaluation)[/cyan]"
    )
    print(
        "[cyan]  - EXAMPLE_ANOMALY_DATA: 3 series × 180 days = 540 rows (with injected anomalies)[/cyan]"
    )
    print("\n[yellow]Exogenous variables in EXAMPLE_TRAIN:[/yellow]")
    print(
        "[yellow]  - series_1: No exogenous variables (for basic forecasting examples)[/yellow]"
    )
    print(
        "[yellow]  - series_2: temperature, is_weekend, promotion (for exogenous examples)[/yellow]"
    )
    print("[yellow]  - series_3: No exogenous variables (for comparison)[/yellow]")
    print("\n[yellow]Injected anomalies in EXAMPLE_ANOMALY_DATA:[/yellow]")
    print("[yellow]  - sensor_1: Spikes on days 30, 120; Drop on day 60[/yellow]")
    print(
        "[yellow]  - sensor_2: Cluster of spikes on days 90-95; Drop on day 150[/yellow]"
    )
    print(
        "[yellow]  - sensor_3: Moderate spike on day 45; Moderate drop on day 100[/yellow]"
    )

    # Show sample usage scripts
    show_example_usage_scripts(config)

    # Return DataFrames if requested (before they were uppercased for Snowflake)
    if return_dataframes:
        # Need to return lowercase versions for NixtlaClient compatibility
        train_df = train.copy()
        train_df.columns = train_df.columns.str.lower()

        all_data_copy = all_data_df.copy()
        all_data_copy.columns = all_data_copy.columns.str.lower()

        anomaly_copy = anomaly_df.copy()
        anomaly_copy.columns = anomaly_copy.columns.str.lower()

        return {
            "train": train_df,
            "all_data": all_data_copy,
            "anomaly": anomaly_copy,
        }

    return None


# ============================================================================
# Main Deployment Flow
# ============================================================================


def deploy_snowflake_core(
    session: Session,
    config: DeploymentConfig,
    api_key: str,
    deploy_security: bool = True,
    deploy_package: bool = True,
    deploy_udtfs: bool = True,
    deploy_procedures: bool = True,
    deploy_finetune: bool = True,
    deploy_examples: bool = True,
) -> DeploymentConfig:
    """
    Core deployment logic without user interaction.

    This function contains the pure deployment logic and can be called
    programmatically or from tests. It does not perform any interactive
    prompts or confirmations.

    Args:
        session: Active Snowflake session
        config: Deployment configuration with database, schema, stage, etc.
        api_key: Nixtla API key for authentication
        deploy_security: Whether to deploy security integration (network rules, secrets)
        deploy_package: Whether to package and upload Nixtla client
        deploy_udtfs: Whether to create UDTFs
        deploy_procedures: Whether to create stored procedures
        deploy_finetune: Whether to create finetune stored procedure
        deploy_examples: Whether to load example datasets

    Returns:
        DeploymentConfig that was used for deployment
    """
    # Set API key in environment for the deployment functions
    os.environ["NIXTLA_API_KEY"] = api_key

    # Deploy components based on flags
    if deploy_security:
        create_security_integration(session, config, api_key, skip_confirmation=True)

    if deploy_package:
        package_and_upload_nixtla(session, config.stage)

    if deploy_udtfs:
        create_udtfs(session, config)

    if deploy_procedures:
        create_stored_procedures(session, config, skip_confirmation=True)

    if deploy_finetune:
        create_finetune_sproc(session, config)

    if deploy_examples:
        load_example_datasets(session, config)

    return config


def deploy_snowflake(
    connection_name: str = "default",
    database: Optional[str] = None,
    schema: Optional[str] = None,
    stage_path: Optional[str] = None,
    integration_name: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """
    Deploy Nixtla to Snowflake with interactive prompts.

    Args:
        connection_name: Name of connection from ~/.snowflake/config.toml
        database: Database name (optional, will prompt if not provided)
        schema: Schema name (optional, will prompt if not provided)
        stage_path: Stage path (optional, will prompt if not provided)
        integration_name: External access integration name (optional, default: nixtla_access_integration)
        base_url: Nixtla API base URL (optional, will prompt if not provided)
            - https://api.nixtla.io (default, TimeGPT)
            - https://tsmp.nixtla.io (TimeGPT-2, supports all models)
    """
    # Create session
    session = create_snowflake_session(connection_name)

    with session:
        # Get config from session
        config_database, config_schema = get_config_from_session(session)

        # Setup infrastructure
        db = setup_database(session, database, config_database)
        if not db:
            return

        sch = setup_schema(session, db, schema, config_schema)
        if not sch:
            return

        stage = setup_stage(session, stage_path)
        if not stage:
            return

        # Ask for integration name if not provided
        _integration_name = ask_with_defaults(
            "External access integration name: ",
            lambda: integration_name,
            lambda: "nixtla_access_integration",
        )

        # Ask for base URL if not provided
        base_url_options = {
            "1": "https://api.nixtla.io",
            "2": "https://tsmp.nixtla.io",
        }
        print("\n[cyan]Available Nixtla API endpoints:[/cyan]")
        print("  [1] https://api.nixtla.io (default, TimeGPT)")
        print("  [2] https://tsmp.nixtla.io (TimeGPT-2, supports all models)")
        _base_url_input = ask_with_defaults(
            "Nixtla API base URL (enter 1, 2, or full URL): ",
            lambda: base_url,
            lambda: "1",
        )
        # Map shorthand to full URL
        _base_url = base_url_options.get(_base_url_input, _base_url_input)

        # Create config object
        config = DeploymentConfig(
            database=db,
            schema=sch,
            stage=stage,
            integration_name=_integration_name,
            base_url=_base_url,
        )

        print(f"[cyan]Using integration: {config.integration_name}[/cyan]")
        print(f"[cyan]Using API endpoint: {config.base_url}[/cyan]")

        # Ask user which components to deploy
        deploy_security = Confirm.ask(
            "Do you want to generate the security script?", default=False
        )
        deploy_package = Confirm.ask(
            "Do you want to (re)package the Nixtla client?", default=False
        )
        deploy_udtfs = Confirm.ask(
            "Do you want to (re)create the UDTFs?", default=False
        )
        deploy_procedures = Confirm.ask(
            "Do you want to generate the stored procedures script for inference and evaluation?",
            default=False,
        )
        deploy_finetune = Confirm.ask(
            "Do you want to (re)create the stored procedure for finetuning?",
            default=False,
        )
        deploy_examples = Confirm.ask(
            "Do you want to (re)create the example datasets?",
            default=False,
        )

        # Get API key
        nixtla_api_key = ask_with_defaults(
            "Nixtla API key: ",
            lambda: os.environ.get("NIXTLA_API_KEY"),
            password=True,
        )

        # Call core deployment function
        deploy_snowflake_core(
            session=session,
            config=config,
            api_key=nixtla_api_key,
            deploy_security=deploy_security,
            deploy_package=deploy_package,
            deploy_udtfs=deploy_udtfs,
            deploy_procedures=deploy_procedures,
            deploy_finetune=deploy_finetune,
            deploy_examples=deploy_examples,
        )


if __name__ == "__main__":
    Fire(deploy_snowflake)
