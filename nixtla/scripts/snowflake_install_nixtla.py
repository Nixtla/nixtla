"""
Refactored version of snowflake_install_nixtla.py with improved readability and modularity.

This script deploys Nixtla forecasting components to Snowflake:
- External access integrations for API calls
- Python UDTFs for batch forecasting and evaluation
- SQL stored procedures for inference
- Stored procedure for model finetuning
- Example datasets

Usage:
    python -m nixtla.scripts.snowflake_deploy_refactored [OPTIONS]

Options:
    --connection_name: Snowflake connection name (default: "default")
    --database: Database name (optional, will prompt)
    --schema: Schema name (optional, will prompt)
    --stage_path: Stage path (optional, will prompt)
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Optional

import pandas as pd
from fire import Fire
from rich import print
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt
from snowflake.snowpark import Session

# ============================================================================
# Constants
# ============================================================================

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
]

TEMPLATE_ACCESS_INTEGRATION = """
CREATE OR REPLACE NETWORK RULE {ds_prefix}nixtla_network_rule
MODE = EGRESS
TYPE = HOST_PORT
VALUE_LIST = ('api.nixtla.io');

//<br>

CREATE OR REPLACE SECRET {ds_prefix}nixtla_api_key
  TYPE = GENERIC_STRING
  SECRET_STRING = '{nixtla_api_key}';

//<br>

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION {integration_name}
  ALLOWED_NETWORK_RULES = ({ds_prefix}nixtla_network_rule)
  ALLOWED_AUTHENTICATION_SECRETS = ({ds_prefix}nixtla_api_key)
  ENABLED = true;
"""

TEMPLATE_SP = """
CREATE OR REPLACE PROCEDURE {ds_prefix}NIXTLA_FORECAST(
    "INPUT_DATA" VARCHAR,
    "PARAMS" OBJECT,
    "MAX_BATCHES" NUMBER(38,0) DEFAULT 1000
)
RETURNS TABLE (
  UNIQUE_ID VARCHAR, DS TIMESTAMP, FORECAST DOUBLE
)
LANGUAGE SQL AS
$$
DECLARE
  res RESULTSET;
BEGIN
  res := (SELECT b.* FROM
    (SELECT
        MOD(HASH($1), :MAX_BATCHES) AS gp,
        TO_VARCHAR($1) AS unique_id,
        $2::TIMESTAMP_NTZ AS ds,
        TO_DOUBLE($3) AS y
     FROM TABLE(:INPUT_DATA)) a,
    TABLE(nixtla_forecast_batch(:PARAMS, unique_id, ds, y) OVER (PARTITION BY gp ORDER BY unique_id, ds)) b);
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
    (SELECT MOD(HASH(unique_id), :MAX_BATCHES) AS gp, unique_id, ds, y, object_construct(*) AS obj FROM TABLE(:INPUT_DATA)) a,
    TABLE(nixtla_evaluate_batch(:METRICS, obj) OVER (PARTITION BY gp ORDER BY unique_id, ds)) b);
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

    @property
    def prefix(self) -> str:
        """Return fully qualified prefix for Snowflake objects."""
        return f"{self.database}.{self.schema}."

    def get_security_params(self) -> dict:
        """Generate security parameters for UDTFs and stored procedures."""
        return {
            "secrets": {"nixtla_api_key": "nixtla_api_key"},
            "external_access_integrations": [self.integration_name],
        }


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
        if not Confirm.ask(f"Do you want to create {object_type.lower()} '{name}'?", default=True):
            return False

        try:
            session.sql(create_sql).collect()
            print(f"[green]{object_type} '{name}' created successfully![/green]")
            return True
        except Exception as create_error:
            print(f"[red]Failed to create {object_type.lower()}: {create_error}[/red]")
            return False


def execute_sql_script(session: Session, script: str, separator: str = "//<br>") -> None:
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
            passcode = Prompt.ask("Enter your MFA code from authenticator app", password=False)
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
        "Stage path (without @) for the artifact: ",
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
            "--no-deps",
        ]

        subprocess.run(install_args, check=True)

        # Create zip archive
        shutil.make_archive(os.path.join(tmpdir, "nixtla"), "zip", tmpdir)
        zip_path = os.path.join(tmpdir, "nixtla.zip")

        # Upload to stage
        result = session.sql(
            f"PUT file://{zip_path} @{stage} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        ).collect()

        if result[0]["status"].lower() != "uploaded":
            raise ValueError("Upload failed " + str(result))

        print("[green]Nixtla client package uploaded[/green]")


# ============================================================================
# Security Integration
# ============================================================================


def create_security_integration(session: Session, config: DeploymentConfig) -> None:
    """
    Create network rules, secrets, and external access integration.

    Args:
        session: Active Snowflake session
        config: Deployment configuration
    """
    nixtla_api_key = ask_with_defaults(
        "Nixtla API key: ",
        lambda: os.environ.get("NIXTLA_API_KEY"),
        password=True,
    )

    # Clean and escape the API key
    # Strip whitespace and quotes that might have been accidentally entered
    nixtla_api_key = nixtla_api_key.strip().strip("'\"")

    # Escape single quotes for SQL (replace ' with '')
    nixtla_api_key_escaped = nixtla_api_key.replace("'", "''")

    script = TEMPLATE_ACCESS_INTEGRATION.format(
        ds_prefix=config.prefix,
        nixtla_api_key=nixtla_api_key_escaped,
        integration_name=config.integration_name,
    )

    print(Markdown(f"```sql\n{script}\n```"))

    if Confirm.ask("Do you want to run the script now?", default=False):
        execute_sql_script(session, script)
        print(f"[green]Access integration '{config.integration_name}' created![/green]")


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

    # Forecast UDTF
    @udtf(
        input_types=[MapType(), StringType(), TimestampType(), DoubleType()],
        output_schema=StructType([
            StructField("unique_id", StringType()),
            StructField("ds", TimestampType()),
            StructField("forecast", DoubleType()),
        ]),
        name="nixtla_forecast_batch",
        **security_params,
        **common_params,
    )
    class ForecastUDTF:
        def __init__(self):
            import _snowflake
            from nixtla import NixtlaClient

            token = _snowflake.get_generic_secret_string("nixtla_api_key")
            self.client = NixtlaClient(api_key=token)

        def end_partition(self, df: pd.DataFrame) -> pd.DataFrame:
            input_df = pd.DataFrame({
                "unique_id": df.iloc[:, 1],
                "ds": df.iloc[:, 2],
                "y": df.iloc[:, 3],
            })

            config = df.iloc[0, 0]
            if config.get("finetune_steps", 0) > 0:
                raise ValueError("Finetuning is not allowed during forecasting")

            forecast = self.client.forecast(df=input_df, **config)

            # Find forecast column
            forecast_col = None
            for col in forecast.columns:
                if col.lower() not in ["unique_id", "ds", "y"]:
                    forecast_col = col
                    break

            if forecast_col is None:
                raise ValueError("No valid column found for forecast")

            return forecast[["unique_id", "ds", forecast_col]].rename(
                columns={forecast_col: "forecast"}
            )

        end_partition._sf_vectorized_input = pd.DataFrame  # type: ignore

    # Evaluate UDTF
    @udtf(
        input_types=[ArrayType(), MapType()],
        output_schema=StructType([
            StructField("unique_id", StringType()),
            StructField("forecaster", StringType()),
            StructField("metric", StringType()),
            StructField("value", DoubleType()),
        ]),
        name="nixtla_evaluate_batch",
        **common_params,
    )
    class EvaluateUDTF:
        def end_partition(self, df: pd.DataFrame) -> pd.DataFrame:
            from utilsforecast.evaluation import evaluate

            metrics = list(self._parse_metrics(df.iloc[0, 0]))
            data = pd.DataFrame(df.iloc[:, 1].tolist())
            data.columns = data.columns.str.lower()
            data = data.sort_values(by=["unique_id", "ds"])
            data["ds"] = pd.to_datetime(data["ds"])

            forecasters = [
                col for col in data.columns
                if col.lower() not in ["unique_id", "ds", "y"]
            ]

            result = evaluate(data, metrics=metrics, models=forecasters, train_df=data)
            result = pd.melt(
                result,
                id_vars=["unique_id", "metric"],
                var_name="forecaster",
                value_name="value",
            )

            return result[["unique_id", "forecaster", "metric", "value"]]

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

    print("[green]UDTFs created![/green]")


# ============================================================================
# Stored Procedures
# ============================================================================


def create_stored_procedures(session: Session, config: DeploymentConfig) -> None:
    """
    Deploy SQL stored procedures for inference and evaluation.

    Args:
        session: Active Snowflake session
        config: Deployment configuration
    """
    script = TEMPLATE_SP.format(ds_prefix=config.prefix)
    print(Markdown(f"```sql\n{script}\n```"))

    if Confirm.ask("Do you want to run the script now?", default=False):
        execute_sql_script(session, script)
        print("[green]Stored procedures created![/green]")


def create_finetune_sproc(session: Session, config: DeploymentConfig) -> None:
    """
    Deploy stored procedure for finetuning.

    Args:
        session: Active Snowflake session
        config: Deployment configuration with stage and integration name
    """
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
        client = NixtlaClient(api_key=token)

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


def load_example_datasets(session: Session) -> None:
    """
    Load example datasets for testing.

    Args:
        session: Active Snowflake session
    """
    # Load and upload all_data
    all_data = pd.read_parquet("all_data.parquet")
    all_data.columns = all_data.columns.str.upper()
    session.write_pandas(
        all_data,
        "EXAMPLE_ALL_DATA",
        auto_create_table=True,
        overwrite=True,
        use_logical_type=True,
    )

    # Load and upload train
    train = pd.read_parquet("train.parquet")
    train.columns = train.columns.str.upper()
    session.write_pandas(
        train,
        "EXAMPLE_TRAIN",
        auto_create_table=True,
        overwrite=True,
        use_logical_type=True,
    )

    print("[green]Example datasets (EXAMPLE_ALL_DATA and EXAMPLE_TRAIN) created![/green]")


# ============================================================================
# Main Deployment Flow
# ============================================================================


def deploy_snowflake(
    connection_name: str = "default",
    database: Optional[str] = None,
    schema: Optional[str] = None,
    stage_path: Optional[str] = None,
    integration_name: Optional[str] = None,
) -> None:
    """
    Deploy Nixtla to Snowflake with interactive prompts.

    Args:
        connection_name: Name of connection from ~/.snowflake/config.toml
        database: Database name (optional, will prompt if not provided)
        schema: Schema name (optional, will prompt if not provided)
        stage_path: Stage path (optional, will prompt if not provided)
        integration_name: External access integration name (optional, default: nixtla_access_integration)
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

        # Create config object
        config = DeploymentConfig(
            database=db,
            schema=sch,
            stage=stage,
            integration_name=_integration_name,
        )

        print(f"[cyan]Using integration: {config.integration_name}[/cyan]")

        # Deploy components (each step is optional)
        if Confirm.ask("Do you want to generate the security script?", default=False):
            create_security_integration(session, config)

        if Confirm.ask("Do you want to (re)package the Nixtla client?", default=False):
            package_and_upload_nixtla(session, config.stage)

        if Confirm.ask("Do you want to (re)create the UDTFs?", default=False):
            create_udtfs(session, config)

        if Confirm.ask(
            "Do you want to generate the stored procedures script for inference and evaluation?",
            default=False,
        ):
            create_stored_procedures(session, config)

        if Confirm.ask(
            "Do you want to (re)create the stored procedure for finetuning?",
            default=False,
        ):
            create_finetune_sproc(session, config)

        if Confirm.ask(
            "Do you want to (re)create the example datasets?",
            default=False,
        ):
            load_example_datasets(session)


def main():
    Fire(deploy_snowflake)


if __name__ == "__main__":
    main()
