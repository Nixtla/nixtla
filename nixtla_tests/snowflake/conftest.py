"""
Pytest fixtures for Snowflake integration tests.

Environment variables for connection (Option 1 - Recommended for CI/CD):
    SF_ACCOUNT: Snowflake account identifier (e.g., "myorg-account123")
    SF_USER: Snowflake username
    SF_PASSWORD: Snowflake password
    SF_WAREHOUSE: Snowflake warehouse (optional)
    SF_ROLE: Snowflake role (optional)

Environment variables for connection (Option 2 - Using config file):
    SNOWFLAKE_CONNECTION_NAME: Connection name from ~/.snowflake/config.toml (default: "default")

Required for both options:
    NIXTLA_API_KEY: Nixtla API key for authentication

Optional test resource configuration:
    SF_TEST_DATABASE: Test database name (default: "NIXTLA_TESTDB")
    SF_TEST_SCHEMA: Test schema name (default: "NIXTLA_SCHEMA")
    SF_TEST_STAGE: Test stage name (default: "nixtla_stage")

Connection priority: Environment variables (Option 1) take precedence over config file (Option 2).
"""

import os
from dataclasses import dataclass
from typing import Generator

import pandas as pd
import pytest
from dotenv import load_dotenv
from snowflake.snowpark import Session

from nixtla.scripts.snowflake_install_nixtla import (
    DeploymentConfig,
    create_snowflake_session,
    deploy_snowflake_core,
)

load_dotenv(override=True)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TestConfig:
    """Configuration for test resources."""

    database: str
    schema: str
    stage: str
    api_key: str


# ============================================================================
# Session-Scoped Fixtures (Connection & Configuration)
# ============================================================================


@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """
    Load test configuration from environment variables.

    Returns:
        TestConfig with database, schema, stage, and API key
    """
    api_key = os.getenv("NIXTLA_API_KEY")

    if not api_key:
        pytest.skip("NIXTLA_API_KEY not set, skipping Snowflake tests")

    assert isinstance(api_key, str) and len(api_key) > 0, "NIXTLA_API_KEY must be a non-empty string"
    return TestConfig(
        database=os.getenv("SF_TEST_DATABASE", "NIXTLA_TESTDB"),
        schema=os.getenv("SF_TEST_SCHEMA", "NIXTLA_SCHEMA"),
        stage=os.getenv("SF_TEST_STAGE", "nixtla_stage"),
        api_key=api_key,
    )


@pytest.fixture(scope="session")
def snowflake_session(test_config: TestConfig) -> Generator[Session, None, None]:
    """
    Create a Snowflake session for testing.

    Supports both config file and environment variables:
    - If SF_ACCOUNT, SF_USER, SF_PASSWORD env vars are set, use them directly
    - Otherwise, fall back to SNOWFLAKE_CONNECTION_NAME from config file

    Yields:
        Active Snowflake session

    Cleanup:
        Closes session and drops test database at end of session
    """
    # Try environment variables first (for CI/CD)
    account = os.getenv("SF_ACCOUNT")
    user = os.getenv("SF_USER")
    password = os.getenv("SF_PASSWORD")
    warehouse = os.getenv("SF_WAREHOUSE")
    role = os.getenv("SF_ROLE")

    session = None
    try:
        if account and user and password:
            # Use direct connection from env vars
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
        else:
            # Fall back to config file
            connection_name = os.getenv("SNOWFLAKE_CONNECTION_NAME", "default")
            print(f"Connecting to Snowflake with connection: {connection_name}")
            session = create_snowflake_session(connection_name)

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
    snowflake_session: Session, test_config: TestConfig
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
    test_config: TestConfig,
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
    test_config: TestConfig,
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
    test_config: TestConfig,
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
        integration_name="nixtla_test_integration_api",
        base_url="https://api.nixtla.io",
    )


@pytest.fixture(scope="module")
def deployment_config_tsmp_nixtla(
    test_config: TestConfig,
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
        integration_name="nixtla_test_integration_tsmp",
        base_url="https://tsmp.nixtla.io",
    )


# ============================================================================
# Deployment Fixtures (Module-Scoped - Expensive Operations)
# ============================================================================


@pytest.fixture(scope="module")
def deployed_with_api_endpoint(
    snowflake_session: Session,
    deployment_config_api_nixtla: DeploymentConfig,
    test_config: TestConfig,
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
            f"DROP NETWORK RULE IF EXISTS {config.prefix}nixtla_network_rule"
        ).collect()
    except Exception as e:
        print(f"Warning: Failed to cleanup integration resources: {e}")


@pytest.fixture(scope="module")
def deployed_with_tsmp_endpoint(
    snowflake_session: Session,
    deployment_config_tsmp_nixtla: DeploymentConfig,
    test_config: TestConfig,
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
            f"DROP NETWORK RULE IF EXISTS {config.prefix}nixtla_network_rule"
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
        rule_name = f"{config.prefix}nixtla_network_rule"
        rule_details = session.sql(f"DESC NETWORK RULE {rule_name}").collect()
        value_list = next(
            (row["value"] for row in rule_details if row["name"] == "VALUE_LIST"), None
        )
        return value_list is not None and config.api_host in value_list
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
    udtfs = [
        "nixtla_forecast_batch",
        "nixtla_evaluate_batch",
        "nixtla_detect_anomalies_batch",
    ]

    try:
        for udtf_name in udtfs:
            full_name = f"{config.prefix}{udtf_name}"
            session.sql(f"DESC FUNCTION {full_name}(OBJECT, OBJECT)").collect()
        return True
    except Exception:
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
    procedures = ["NIXTLA_FORECAST", "NIXTLA_EVALUATE", "NIXTLA_DETECT_ANOMALIES"]

    try:
        for proc_name in procedures:
            full_name = f"{config.prefix}{proc_name}"
            # Try to describe the procedure
            session.sql(f"DESC PROCEDURE {full_name}(VARCHAR, OBJECT)").collect()
        return True
    except Exception:
        return False
