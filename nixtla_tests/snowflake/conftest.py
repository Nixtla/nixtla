"""
Pytest fixtures for Snowflake integration tests.

Required environment variables:
    SF_ACCOUNT: Snowflake account identifier (e.g., "myorg-account123")
    SF_USER: Snowflake username
    SF_PASSWORD: Snowflake password
    SF_WAREHOUSE: Snowflake warehouse (optional)
    SF_ROLE: Snowflake role (optional)
    NIXTLA_API_KEY_FOR_SF: Nixtla API key for authentication

Optional test resource configuration:
    SF_TEST_DATABASE: Base database name (default: "NIXTLA_TESTDB")
    SF_TEST_SCHEMA: Base schema name (default: "NIXTLA_SCHEMA")
    SF_TEST_STAGE: Base stage name (default: "NIXTLA_STAGE")

All Snowflake assets (database, schema, stage, network rule, integration)
are namespaced with a short UUID-based suffix to avoid conflicts when
tests run concurrently across different environments.

Note: Tests will be skipped if required environment variables are not set.
"""

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

# Resolve the project root (where pyproject.toml lives) so tests install
# the local package instead of fetching an (unreleased) version from PyPI.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
from dotenv import load_dotenv
from snowflake.snowpark import Session

from nixtla.scripts.snowflake_install_nixtla import (
    DeploymentConfig,
    deploy_snowflake_core,
)

load_dotenv(override=True)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class SnowflakeTestConfig:
    """Configuration for test resources."""

    database: str
    schema: str
    stage: str
    api_key: str
    namespace: str


# ============================================================================
# Session-Scoped Fixtures (Connection & Configuration)
# ============================================================================


@pytest.fixture(scope="session")
def test_config() -> SnowflakeTestConfig:
    """
    Load test configuration from environment variables.

    Returns:
        SnowflakeTestConfig with database, schema, stage, and API key
    """
    api_key = os.getenv("NIXTLA_API_KEY_FOR_SF")

    if not api_key:
        pytest.skip("NIXTLA_API_KEY_FOR_SF not set, skipping Snowflake tests")

    assert isinstance(api_key, str) and len(api_key) > 0, (
        "NIXTLA_API_KEY_FOR_SF must be a non-empty string"
    )
    ns = uuid.uuid4().hex[:8].upper()

    # Base names from env (if provided), then namespace them
    base_db = os.getenv("SF_TEST_DATABASE", "NIXTLA_TESTDB").upper()
    base_schema = os.getenv("SF_TEST_SCHEMA", "NIXTLA_SCHEMA").upper()
    base_stage = os.getenv("SF_TEST_STAGE", "NIXTLA_STAGE").upper()

    return SnowflakeTestConfig(
        database=f"{base_db}_{ns}",
        schema=f"{base_schema}_{ns}",
        stage=f"{base_stage}_{ns}",
        api_key=api_key,
        namespace=ns,
    )


@pytest.fixture(scope="session")
def snowflake_session(
    test_config: SnowflakeTestConfig,
) -> Generator[Session, None, None]:
    """
    Create a Snowflake session for testing.

    Requires environment variables:
    - SF_ACCOUNT: Snowflake account identifier
    - SF_USER: Snowflake username
    - SF_PASSWORD: Snowflake password
    - SF_WAREHOUSE: Snowflake warehouse (optional)
    - SF_ROLE: Snowflake role (optional)

    Yields:
        Active Snowflake session

    Cleanup:
        Closes session and drops test database at end of session
    """
    # Check required environment variables
    account = os.getenv("SF_ACCOUNT")
    user = os.getenv("SF_USER")
    password = os.getenv("SF_PASSWORD")
    warehouse = os.getenv("SF_WAREHOUSE")
    role = os.getenv("SF_ROLE")

    # Check which env vars are missing
    missing_vars = []
    if not account:
        missing_vars.append("SF_ACCOUNT")
    if not user:
        missing_vars.append("SF_USER")
    if not password:
        missing_vars.append("SF_PASSWORD")

    if missing_vars:
        pytest.skip(
            f"Snowflake credentials not configured. "
            f"Missing environment variables: {', '.join(missing_vars)}"
        )

    # Build connection from env vars
    connection_params = {
        "account": account,
        "user": user,
        "password": password,
    }
    if warehouse:
        connection_params["warehouse"] = warehouse
    if role:
        connection_params["role"] = role

    print(f"Connecting to Snowflake with account: {account}")
    session = Session.builder.configs(connection_params).create()

    try:
        yield session
    finally:
        # Cleanup: Drop test database
        if session:
            try:
                print(f"Cleaning up: Dropping database {test_config.database}")
                session.sql(f"DROP DATABASE IF EXISTS {test_config.database}").collect()
            except Exception as e:
                print(f"Warning: Failed to drop test database: {e}")

            try:
                session.close()
            except Exception as e:
                print(f"Warning: Failed to close session: {e}")


# ============================================================================
# Resource Setup Fixtures (Database, Schema, Stage)
# ============================================================================


@pytest.fixture(scope="session")
def ensure_test_database(
    snowflake_session: Session, test_config: SnowflakeTestConfig
) -> Generator[str, None, None]:
    """
    Ensure test database exists.

    Yields:
        Database name

    Cleanup:
        Handled by snowflake_session fixture
    """
    database_name = test_config.database

    # Create database if it doesn't exist
    snowflake_session.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}").collect()
    snowflake_session.use_database(database_name)

    yield database_name

    # Cleanup handled by snowflake_session


@pytest.fixture(scope="session")
def ensure_test_schema(
    snowflake_session: Session,
    test_config: SnowflakeTestConfig,
    ensure_test_database: str,
) -> Generator[str, None, None]:
    """
    Ensure test schema exists.

    Yields:
        Schema name

    Cleanup:
        Handled by database cleanup
    """
    schema_name = test_config.schema

    # Create schema if it doesn't exist
    snowflake_session.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}").collect()
    snowflake_session.use_schema(schema_name)

    yield schema_name

    # Cleanup handled by database cleanup


@pytest.fixture(scope="session")
def ensure_test_stage(
    snowflake_session: Session,
    test_config: SnowflakeTestConfig,
    ensure_test_schema: str,
) -> Generator[str, None, None]:
    """
    Ensure test stage exists.

    Yields:
        Stage name

    Cleanup:
        Handled by database cleanup
    """
    stage_name = test_config.stage

    # Create stage if it doesn't exist
    snowflake_session.sql(f"CREATE STAGE IF NOT EXISTS {stage_name}").collect()

    yield stage_name

    # Cleanup handled by database cleanup


# ============================================================================
# Deployment Config Fixtures (Module-Scoped)
# ============================================================================


@pytest.fixture(scope="module")
def deployment_config_api_nixtla(
    test_config: SnowflakeTestConfig,
    ensure_test_database: str,
    ensure_test_schema: str,
    ensure_test_stage: str,
) -> DeploymentConfig:
    """
    Create deployment config for api.nixtla.io endpoint.

    Returns:
        DeploymentConfig configured for api.nixtla.io
    """
    return DeploymentConfig(
        database=test_config.database,
        schema=test_config.schema,
        stage=test_config.stage,
        integration_name=f"nixtla_test_integration_api_{test_config.namespace}",
        base_url="https://api.nixtla.io",
        network_rule_name=f"nixtla_network_rule_api_{test_config.namespace}",
    )


@pytest.fixture(scope="module")
def deployment_config_tsmp_nixtla(
    test_config: SnowflakeTestConfig,
    ensure_test_database: str,
    ensure_test_schema: str,
    ensure_test_stage: str,
) -> DeploymentConfig:
    """
    Create deployment config for tsmp.nixtla.io endpoint.

    Returns:
        DeploymentConfig configured for tsmp.nixtla.io
    """
    return DeploymentConfig(
        database=test_config.database,
        schema=test_config.schema,
        stage=test_config.stage,
        integration_name=f"nixtla_test_integration_tsmp_{test_config.namespace}",
        base_url="https://tsmp.nixtla.io",
        network_rule_name=f"nixtla_network_rule_tsmp_{test_config.namespace}",
    )


# ============================================================================
# Deployment Fixtures (Module-Scoped - Expensive Operations)
# ============================================================================


@pytest.fixture(scope="module")
def deployed_with_api_endpoint(
    snowflake_session: Session,
    deployment_config_api_nixtla: DeploymentConfig,
    test_config: SnowflakeTestConfig,
) -> Generator[DeploymentConfig, None, None]:
    """
    Deploy Nixtla components with api.nixtla.io endpoint.

    This is a module-scoped fixture because deployment is expensive.
    The deployment is shared across all tests in the module.

    Yields:
        DeploymentConfig for the deployed components

    Cleanup:
        Drops integration, secrets, and network rules
    """
    config = deployment_config_api_nixtla

    # Ensure session is using the correct database and schema context
    snowflake_session.use_database(config.database)
    snowflake_session.use_schema(config.schema)

    # Deploy all components using core function
    # Note: deploy_examples=False because we'll load examples separately
    # to capture the DataFrames
    deploy_snowflake_core(
        session=snowflake_session,
        config=config,
        api_key=test_config.api_key,
        deploy_security=True,
        deploy_package=True,
        deploy_udtfs=True,
        deploy_procedures=True,
        deploy_finetune=False,  # Skip finetune to speed up tests
        deploy_examples=False,  # Load examples separately to get DataFrames
        fallback_package_source=_PROJECT_ROOT,
    )

    yield config

    # Cleanup: Drop integration and secrets
    try:
        snowflake_session.sql(
            f"DROP INTEGRATION IF EXISTS {config.integration_name}"
        ).collect()
        snowflake_session.sql(
            f"DROP SECRET IF EXISTS {config.prefix}nixtla_api_key"
        ).collect()
        snowflake_session.sql(
            f"DROP SECRET IF EXISTS {config.prefix}nixtla_base_url"
        ).collect()
        snowflake_session.sql(
            f"DROP NETWORK RULE IF EXISTS {config.prefix}{config.network_rule_name}"
        ).collect()
    except Exception as e:
        print(f"Warning: Failed to cleanup integration resources: {e}")


@pytest.fixture(scope="module")
def deployed_with_tsmp_endpoint(
    snowflake_session: Session,
    deployment_config_tsmp_nixtla: DeploymentConfig,
    test_config: SnowflakeTestConfig,
) -> Generator[DeploymentConfig, None, None]:
    """
    Deploy Nixtla components with tsmp.nixtla.io endpoint.

    This is a module-scoped fixture because deployment is expensive.
    The deployment is shared across all tests in the module.

    Yields:
        DeploymentConfig for the deployed components

    Cleanup:
        Drops integration, secrets, and network rules
    """
    config = deployment_config_tsmp_nixtla

    # Ensure session is using the correct database and schema context
    snowflake_session.use_database(config.database)
    snowflake_session.use_schema(config.schema)

    # Deploy all components using core function
    # Note: deploy_examples=False because example loading is handled separately
    deploy_snowflake_core(
        session=snowflake_session,
        config=config,
        api_key=test_config.api_key,
        deploy_security=True,
        deploy_package=True,
        deploy_udtfs=True,
        deploy_procedures=True,
        deploy_finetune=False,  # Skip finetune to speed up tests
        deploy_examples=False,  # Load examples separately if needed
        fallback_package_source=_PROJECT_ROOT,
    )

    yield config

    # Cleanup: Drop integration and secrets
    try:
        snowflake_session.sql(
            f"DROP INTEGRATION IF EXISTS {config.integration_name}"
        ).collect()
        snowflake_session.sql(
            f"DROP SECRET IF EXISTS {config.prefix}nixtla_api_key"
        ).collect()
        snowflake_session.sql(
            f"DROP SECRET IF EXISTS {config.prefix}nixtla_base_url"
        ).collect()
        snowflake_session.sql(
            f"DROP NETWORK RULE IF EXISTS {config.prefix}{config.network_rule_name}"
        ).collect()
    except Exception as e:
        print(f"Warning: Failed to cleanup integration resources: {e}")


# ============================================================================
# Test Data Fixtures (Function-Scoped - Cheap to Create)
# ============================================================================


@pytest.fixture
def minimal_test_data() -> pd.DataFrame:
    """
    Create minimal test data for forecasting.

    Returns:
        DataFrame with 2 series, 30 days each
    """
    dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
    data = []

    for series_id in ["test_series_1", "test_series_2"]:
        for i, date in enumerate(dates):
            data.append(
                {
                    "unique_id": series_id,
                    "ds": date,
                    "y": 100 + i + (10 if series_id == "test_series_2" else 0),
                }
            )

    df = pd.DataFrame(data)
    df.columns = df.columns.str.upper()  # Snowflake expects uppercase
    return df


# ============================================================================
# Helper Functions (Verification Utilities)
# ============================================================================


def verify_network_rule_exists(session: Session, config: DeploymentConfig) -> bool:
    """
    Verify that network rule exists with correct host.

    Args:
        session: Snowflake session
        config: Deployment configuration

    Returns:
        True if network rule exists and has correct host
    """
    try:
        rule_name = f"{config.prefix}{config.network_rule_name}"
        rule_details = session.sql(f"DESC NETWORK RULE {rule_name}").collect()
        value_list = [row["value_list"] for row in rule_details]
        return bool(value_list) and config.api_host in value_list[0]
    except Exception:
        return False


def verify_secrets_exist(session: Session, config: DeploymentConfig) -> bool:
    """
    Verify that both API key and base URL secrets exist.

    Args:
        session: Snowflake session
        config: Deployment configuration

    Returns:
        True if both secrets exist
    """
    try:
        # Check API key secret
        session.sql(f"DESC SECRET {config.prefix}nixtla_api_key").collect()
        # Check base URL secret
        session.sql(f"DESC SECRET {config.prefix}nixtla_base_url").collect()
        return True
    except Exception:
        return False


def verify_integration_exists(session: Session, config: DeploymentConfig) -> bool:
    """
    Verify that external access integration exists.

    Args:
        session: Snowflake session
        config: Deployment configuration

    Returns:
        True if integration exists
    """
    try:
        session.sql(f"DESC INTEGRATION {config.integration_name}").collect()
        return True
    except Exception:
        return False


def verify_udtfs_exist(session: Session, config: DeploymentConfig) -> bool:
    """
    Verify that UDTFs (forecast, evaluate, detect_anomalies) exist.

    Args:
        session: Snowflake session
        config: Deployment configuration

    Returns:
        True if all UDTFs exist
    """
    # Each UDTF has a specific signature
    udtfs = [
        ("nixtla_forecast_batch", "OBJECT, OBJECT"),
        ("nixtla_evaluate_batch", "ARRAY, OBJECT"),
        ("nixtla_detect_anomalies_batch", "OBJECT, STRING, TIMESTAMP_NTZ, FLOAT"),
        ("nixtla_explain_batch", "OBJECT, OBJECT"),
    ]

    try:
        for udtf_name, signature in udtfs:
            full_name = f"{config.prefix}{udtf_name}"
            session.sql(f"DESC FUNCTION {full_name}({signature})").collect()
        return True
    except Exception as e:
        print(f"UDTF verification failed: {e}")
        return False


def verify_procedures_exist(session: Session, config: DeploymentConfig) -> bool:
    """
    Verify that stored procedures exist.

    Args:
        session: Snowflake session
        config: Deployment configuration

    Returns:
        True if all procedures exist
    """
    # Each procedure has specific signatures with default parameters
    procedures = [
        ("NIXTLA_FORECAST", "VARCHAR, OBJECT, NUMBER"),
        ("NIXTLA_EVALUATE", "VARCHAR, ARRAY, NUMBER"),
        ("NIXTLA_DETECT_ANOMALIES", "VARCHAR, OBJECT, NUMBER"),
        ("NIXTLA_EXPLAIN", "VARCHAR, OBJECT, NUMBER"),
    ]

    try:
        for proc_name, signature in procedures:
            full_name = f"{config.prefix}{proc_name}"
            # Try to describe the procedure with full signature
            session.sql(f"DESC PROCEDURE {full_name}({signature})").collect()
        return True
    except Exception as e:
        print(f"Procedure verification failed: {e}")
        return False
