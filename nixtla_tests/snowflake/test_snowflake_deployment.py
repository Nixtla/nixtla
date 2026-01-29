"""
Integration tests for Snowflake deployment.

Tests the deployment of Nixtla components with different API endpoints.
"""

from typing import Callable

import pandas as pd
import pytest
from nixtla import NixtlaClient
from snowflake.snowpark import Session

from nixtla.scripts.snowflake_install_nixtla import (
    DeploymentConfig,
    get_example_test_cases,
    load_example_datasets,
)
from nixtla_tests.snowflake.conftest import (
    SnowflakeTestConfig,
    verify_integration_exists,
    verify_network_rule_exists,
    verify_procedures_exist,
    verify_secrets_exist,
    verify_udtfs_exist,
)


def _verify_full_deployment(session: Session, config: DeploymentConfig) -> None:
    """Helper to verify all deployment components exist."""
    # Ensure session is using the correct database and schema context
    session.use_database(config.database)
    session.use_schema(config.schema)

    assert verify_network_rule_exists(session, config), (
        f"Network rule not created with {config.api_host}"
    )
    assert verify_secrets_exist(session, config), "Secrets not created"
    assert verify_integration_exists(session, config), "Integration not created"
    assert verify_udtfs_exist(session, config), "UDTFs not created"
    assert verify_procedures_exist(session, config), "Procedures not created"


@pytest.mark.snowflake
class TestSnowflakeDeployment:
    """Test Snowflake deployment with different endpoints."""

    def test_deploy_with_api_nixtla_io(
        self, snowflake_session: Session, deployed_with_api_endpoint: DeploymentConfig
    ):
        """Test full deployment with api.nixtla.io endpoint."""
        _verify_full_deployment(snowflake_session, deployed_with_api_endpoint)

    def test_deploy_with_tsmp_nixtla_io(
        self, snowflake_session: Session, deployed_with_tsmp_endpoint: DeploymentConfig
    ):
        """Test full deployment with tsmp.nixtla.io endpoint."""
        _verify_full_deployment(snowflake_session, deployed_with_tsmp_endpoint)

    def test_example_datasets_loaded(
        self,
        snowflake_session: Session,
        deployed_with_api_endpoint: DeploymentConfig,
        example_dataframes: dict[str, pd.DataFrame],
    ):
        """Test that example datasets were loaded with correct data."""
        # example_dataframes fixture ensures data is loaded
        config = deployed_with_api_endpoint

        # Ensure session is using the correct database and schema context
        snowflake_session.use_database(config.database)
        snowflake_session.use_schema(config.schema)

        example_tables = [
            "EXAMPLE_TRAIN",
            "EXAMPLE_ALL_DATA",
            "EXAMPLE_ANOMALY_DATA",
        ]

        for table in example_tables:
            result = snowflake_session.sql(
                f"SELECT COUNT(*) as cnt FROM {config.prefix}{table}"
            ).collect()
            assert result[0]["CNT"] > 0, f"{table} is empty"


@pytest.mark.snowflake
class TestDeploymentConfig:
    """Test DeploymentConfig properties and methods."""

    @pytest.mark.parametrize(
        "base_url,expected_host",
        [
            ("https://api.nixtla.io", "api.nixtla.io"),
            ("https://tsmp.nixtla.io", "tsmp.nixtla.io"),
        ],
    )
    def test_api_host_extraction(self, base_url: str, expected_host: str):
        """Test that api_host property correctly extracts hostname from various URLs."""
        config = DeploymentConfig(
            database="TEST_DB",
            schema="TEST_SCHEMA",
            stage="test_stage",
            base_url=base_url,
        )
        assert config.api_host == expected_host
        assert config.base_url == base_url

    def test_security_params_include_both_secrets(self):
        """Test that get_security_params includes both secrets."""
        config = DeploymentConfig(
            database="TEST_DB",
            schema="TEST_SCHEMA",
            stage="test_stage",
            base_url="https://api.nixtla.io",
        )
        params = config.get_security_params()

        assert "secrets" in params
        assert "nixtla_api_key" in params["secrets"]
        assert "nixtla_base_url" in params["secrets"]
        assert "external_access_integrations" in params


# ============================================================================
# Fixtures for Example Script Tests
# ============================================================================


@pytest.fixture(scope="module")
def example_dataframes(
    snowflake_session: Session,
    deployed_with_api_endpoint: DeploymentConfig,
) -> dict[str, pd.DataFrame]:
    """
    Load example datasets to Snowflake and return DataFrames for client comparison.

    This is the ONLY place where example data is loaded. The deployment fixture
    (deployed_with_api_endpoint) does NOT load examples to avoid generating
    different random data twice.

    This ensures the SAME data is used for both:
    1. Snowflake SQL script execution
    2. Direct NixtlaClient comparison

    Returns:
        Dict with keys: 'train', 'all_data', 'anomaly'
    """
    dataframes = load_example_datasets(
        snowflake_session,
        deployed_with_api_endpoint,
        return_dataframes=True,
    )
    assert dataframes is not None, "Failed to load example datasets"
    return dataframes


@pytest.fixture(scope="module")
def nixtla_client(test_config: SnowflakeTestConfig) -> NixtlaClient:
    """Create NixtlaClient for comparison tests."""
    return NixtlaClient(
        api_key=test_config.api_key,
        base_url="https://api.nixtla.io",
    )


# ============================================================================
# Example Scripts Tests
# ============================================================================


@pytest.mark.snowflake
class TestExampleScripts:
    """
    Verify example SQL scripts shown to users work correctly.

    Each test:
    1. Executes the exact SQL script shown to users
    2. Executes equivalent operation with NixtlaClient
    3. Compares results to verify correctness
    """

    def _execute_and_compare(
        self,
        session: Session,
        config: DeploymentConfig,
        example_data: dict[str, pd.DataFrame],
        case_name: str,
        comparison_fn: Callable,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Helper to execute SQL script and compare with client result.

        Args:
            session: Snowflake session
            config: Deployment config
            example_data: Dict of example DataFrames
            case_name: Test case name
            comparison_fn: Function to execute client call and compare

        Returns:
            (snowflake_df, client_df)
        """
        # Ensure session is using the correct database and schema context
        session.use_database(config.database)
        session.use_schema(config.schema)

        # Get test case
        test_cases = get_example_test_cases(config)
        test_case = next(tc for tc in test_cases if tc.name == case_name)

        # 1. Execute SQL (verify it works)
        sf_result = session.sql(test_case.sql_query).collect()
        assert len(sf_result) > 0, (
            f"SQL script '{test_case.description}' returned no results"
        )

        sf_df = pd.DataFrame(sf_result)
        sf_df.columns = sf_df.columns.str.lower()

        # 2. Execute with client and compare (verify correctness)
        client_df = comparison_fn(test_case, example_data[test_case.input_table])

        return sf_df, client_df

    def test_basic_forecast_script(
        self,
        snowflake_session: Session,
        deployed_with_api_endpoint: DeploymentConfig,
        nixtla_client: NixtlaClient,
        example_dataframes: dict[str, pd.DataFrame],
    ):
        """Test: Forecast 14 days with confidence intervals (80%, 95%)."""

        def compare_forecast(test_case, data):
            # Filter out future exog rows (where y is null) for basic forecast
            hist_data = data[data["y"].notna()].copy()
            client_result = nixtla_client.forecast(
                df=hist_data, **test_case.nixtla_params
            )
            return client_result

        sf_df, client_df = self._execute_and_compare(
            snowflake_session,
            deployed_with_api_endpoint,
            example_dataframes,
            "basic_forecast",
            compare_forecast,
        )

        # Validate structure matches
        assert len(sf_df) == len(client_df), "Row count mismatch"
        assert set(sf_df["unique_id"]) == set(client_df["unique_id"]), (
            "Unique IDs mismatch"
        )

        # Snowflake should have confidence_intervals VARIANT column
        assert "confidence_intervals" in sf_df.columns, (
            "Missing confidence_intervals column in Snowflake"
        )

        # Extract confidence intervals from VARIANT column into flat columns
        # The VARIANT column contains: {"80": {"lo": value, "hi": value}, "95": {...}}
        def extract_intervals(row):
            """Extract confidence intervals from VARIANT structure."""
            import json

            intervals = row["confidence_intervals"]
            if intervals is None or (
                isinstance(intervals, float) and pd.isna(intervals)
            ):
                return pd.Series({})

            # Handle potential double-serialization from VARIANT
            # Try to parse as JSON if it's a string
            if isinstance(intervals, str):
                try:
                    intervals = json.loads(intervals)
                    # If still a string after first parse, parse again
                    if isinstance(intervals, str):
                        intervals = json.loads(intervals)
                except (json.JSONDecodeError, TypeError):
                    return pd.Series({})

            # Now intervals should be a dict
            if not isinstance(intervals, dict):
                return pd.Series({})

            result = {}
            for level, bounds in intervals.items():
                if isinstance(bounds, dict):
                    if "lo" in bounds:
                        result[f"timegpt-lo-{level}"] = bounds["lo"]
                    if "hi" in bounds:
                        result[f"timegpt-hi-{level}"] = bounds["hi"]
            return pd.Series(result)

        # Apply extraction to create flat columns
        sf_intervals = sf_df.apply(extract_intervals, axis=1)
        sf_df = pd.concat([sf_df, sf_intervals], axis=1)

        # Normalize column names to lowercase for consistent comparison
        sf_df.columns = sf_df.columns.str.lower()
        client_df.columns = client_df.columns.str.lower()

        # Both should have confidence intervals
        expected_interval_cols = [
            "timegpt-lo-80",
            "timegpt-hi-80",
            "timegpt-lo-95",
            "timegpt-hi-95",
        ]
        for col in expected_interval_cols:
            assert col in sf_df.columns, f"Missing {col} in Snowflake"
            assert col in client_df.columns, f"Missing {col} in client"

        # Sort both dataframes for alignment
        sf_sorted = sf_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        client_sorted = client_df.sort_values(["unique_id", "ds"]).reset_index(
            drop=True
        )

        # Helper to compare columns with tolerance
        def compare_columns(sf_col: str, client_col: str, description: str):
            """Compare two columns with numerical tolerance."""
            pd.testing.assert_series_equal(
                sf_sorted[sf_col],
                client_sorted[client_col],
                check_names=False,
                rtol=1e-3,
                atol=1e-3,
                obj=description,
            )

        # Compare forecast values
        compare_columns("forecast", "timegpt", "Forecast values")

        # Compare confidence intervals
        for col in expected_interval_cols:
            compare_columns(col, col, f"Confidence interval {col}")

    def test_evaluation_metrics_script(
        self,
        snowflake_session: Session,
        deployed_with_api_endpoint: DeploymentConfig,
        nixtla_client: NixtlaClient,
        example_dataframes: dict[str, pd.DataFrame],
    ):
        """Test: Evaluation metrics (MAPE, MAE, MSE).

        Tests the EVALUATE stored procedure which computes forecast accuracy metrics
        on existing predictions.
        """

        def compare_evaluate(test_case, data):
            # The stored procedure evaluates predictions that are already in the data
            # The all_data table has a 'timegpt' column with pre-computed predictions
            from utilsforecast.evaluation import evaluate
            from utilsforecast.losses import mae, mape, mse

            # The data should already contain predictions in the 'timegpt' column
            # Evaluate those predictions using the same metrics (as functions)
            metrics = [mape, mae, mse]

            # Identify forecaster columns (any column not in base cols)
            forecasters = [
                col
                for col in data.columns
                if col.lower() not in ["unique_id", "ds", "y"]
            ]

            result = evaluate(data, metrics=metrics, models=forecasters)

            # Melt to match Snowflake format: unique_id, forecaster, metric, value
            result = pd.melt(
                result,
                id_vars=["unique_id", "metric"],
                var_name="forecaster",
                value_name="value",
            )

            # Convert value column to float64 to match Snowflake dtype
            result["value"] = result["value"].astype(float)

            return result[["unique_id", "forecaster", "metric", "value"]]

        sf_df, client_df = self._execute_and_compare(
            snowflake_session,
            deployed_with_api_endpoint,
            example_dataframes,
            "evaluation_metrics",
            compare_evaluate,
        )

        # Normalize column names to lowercase
        sf_df.columns = sf_df.columns.str.lower()
        client_df.columns = client_df.columns.str.lower()

        # Validate structure - should have evaluation metrics format
        assert len(sf_df) > 0, "Evaluation returned no results"
        assert len(client_df) > 0, "Client evaluation returned no results"

        # Check required columns for evaluation results
        required_cols = ["unique_id", "forecaster", "metric", "value"]
        for col in required_cols:
            assert col in sf_df.columns, f"Missing {col} in Snowflake"
            assert col in client_df.columns, f"Missing {col} in client"

        # Sort and compare evaluation metrics
        sf_sorted = sf_df.sort_values(
            ["unique_id", "metric", "forecaster"]
        ).reset_index(drop=True)
        client_sorted = client_df.sort_values(
            ["unique_id", "metric", "forecaster"]
        ).reset_index(drop=True)

        # Metrics should be very close
        pd.testing.assert_series_equal(
            sf_sorted["value"],
            client_sorted["value"],
            check_names=False,
            rtol=0.1,  # 10% tolerance for metrics
            atol=0.1,
            obj="Evaluation metrics",
        )

    def test_anomaly_detection_script(
        self,
        snowflake_session: Session,
        deployed_with_api_endpoint: DeploymentConfig,
        nixtla_client: NixtlaClient,
        example_dataframes: dict[str, pd.DataFrame],
    ):
        """Test: Detect anomalies with 95% confidence level."""

        def compare_anomaly(test_case, data):
            client_result = nixtla_client.detect_anomalies(
                df=data, **test_case.nixtla_params
            )
            return client_result

        sf_df, client_df = self._execute_and_compare(
            snowflake_session,
            deployed_with_api_endpoint,
            example_dataframes,
            "anomaly_detection",
            compare_anomaly,
        )

        # Normalize column names to lowercase
        sf_df.columns = sf_df.columns.str.lower()
        client_df.columns = client_df.columns.str.lower()

        # Validate structure
        assert len(sf_df) == len(client_df), "Row count mismatch"
        assert "anomaly" in sf_df.columns, "Missing anomaly column in Snowflake"
        assert "anomaly" in client_df.columns, "Missing anomaly column in client"

        # Convert anomaly column to boolean (Snowflake returns string "True"/"False")
        if sf_df["anomaly"].dtype == object:
            sf_df["anomaly"] = (
                sf_df["anomaly"].map({"True": True, "False": False}).fillna(False)
            )
        if client_df["anomaly"].dtype == object:
            client_df["anomaly"] = client_df["anomaly"].astype(bool)

        # Should detect similar number of anomalies
        sf_anomalies = sf_df["anomaly"].sum()
        client_anomalies = client_df["anomaly"].sum()
        assert sf_anomalies > 0, "Snowflake detected no anomalies"
        assert client_anomalies > 0, "Client detected no anomalies"

        # Sort and compare anomaly flags
        sf_sorted = sf_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        client_sorted = client_df.sort_values(["unique_id", "ds"]).reset_index(
            drop=True
        )

        pd.testing.assert_series_equal(
            sf_sorted["anomaly"],
            client_sorted["anomaly"],
            check_names=False,
            obj="Anomaly flags",
        )

    def test_forecast_with_future_exog_script(
        self,
        snowflake_session: Session,
        deployed_with_api_endpoint: DeploymentConfig,
        nixtla_client: NixtlaClient,
        example_dataframes: dict[str, pd.DataFrame],
    ):
        """Test: Forecast with historical and future exogenous variables."""

        def compare_forecast_future_exog(test_case, data):
            # For future exogenous forecasting, we need series_2 which has exog vars
            data_series2 = data[data["unique_id"] == "series_2"].copy()

            hist_data = data_series2[data_series2["y"].notna()]
            futr_data = data_series2[data_series2["y"].isna()]

            # Forecast using both hist_exog_list and X_df (future exog)
            hist_exog_cols = ["temperature"]
            futr_exog_cols = ["is_weekend", "promotion"]

            # X_df should only contain unique_id, ds, and the future exog columns
            # This matches what the Snowflake UDTF does internally
            X_df = futr_data[["unique_id", "ds"] + futr_exog_cols].copy()

            client_result = nixtla_client.forecast(
                df=hist_data,
                h=14,
                freq="D",
                hist_exog_list=hist_exog_cols,  # Historical exogenous
                X_df=X_df,  # Future exogenous (only required columns)
            )
            return client_result

        sf_df, client_df = self._execute_and_compare(
            snowflake_session,
            deployed_with_api_endpoint,
            example_dataframes,
            "forecast_with_future_exog",
            compare_forecast_future_exog,
        )

        # Normalize column names to lowercase
        sf_df.columns = sf_df.columns.str.lower()
        client_df.columns = client_df.columns.str.lower()

        # Validate structure
        assert len(sf_df) > 0, "Forecast with future exog returned no results"
        assert len(client_df) > 0, (
            "Client forecast with future exog returned no results"
        )

        # Check required columns
        for col in ["timegpt", "forecast"]:
            if col in sf_df.columns or col in client_df.columns:
                break
        else:
            assert False, "Missing forecast column in results"

        # Determine forecast column name (normalize to 'forecast')
        sf_forecast_col = "forecast" if "forecast" in sf_df.columns else "timegpt"
        client_forecast_col = (
            "forecast" if "forecast" in client_df.columns else "timegpt"
        )

        # Filter to series_2 for comparison (only series_2 has future exog)
        sf_series2 = sf_df[sf_df["unique_id"] == "series_2"].copy()
        client_series2 = client_df[client_df["unique_id"] == "series_2"].copy()

        # Validate that we have results for series_2
        assert len(sf_series2) > 0, "No results for series_2 from Snowflake"
        assert len(client_series2) > 0, "No results for series_2 from client"
        assert len(sf_series2) == len(client_series2), (
            f"Row count mismatch for series_2: Snowflake={len(sf_series2)}, Client={len(client_series2)}"
        )

        # Sort and compare
        sf_sorted = sf_series2.sort_values(["ds"]).reset_index(drop=True)
        client_sorted = client_series2.sort_values(["ds"]).reset_index(drop=True)

        # Compare forecast values
        pd.testing.assert_series_equal(
            sf_sorted[sf_forecast_col],
            client_sorted[client_forecast_col],
            check_names=False,
            rtol=1e-3,
            atol=1e-3,
            obj="Forecast with future exogenous variables",
        )

    def test_explain_forecast_script(
        self,
        snowflake_session: Session,
        deployed_with_api_endpoint: DeploymentConfig,
        nixtla_client: NixtlaClient,
        example_dataframes: dict[str, pd.DataFrame],
    ):
        """Test: Explain forecast with historical and future exogenous variables (SHAP values)."""

        def compare_explain_future_exog(_test_case, data):
            # For future exogenous forecasting with explain, we need series_2 which has exog vars
            data_series2 = data[data["unique_id"] == "series_2"].copy()

            hist_data = data_series2[data_series2["y"].notna()]
            futr_data = data_series2[data_series2["y"].isna()]

            # Explain using both hist_exog_list and X_df (future exog)
            hist_exog_cols = ["temperature"]
            futr_exog_cols = ["is_weekend", "promotion"]

            # X_df should only contain unique_id, ds, and the future exog columns
            # This matches what the Snowflake UDTF does internally
            x_df = futr_data[["unique_id", "ds"] + futr_exog_cols].copy()

            nixtla_client.forecast(
                df=hist_data,
                h=14,
                freq="D",
                hist_exog_list=hist_exog_cols,  # Historical exogenous
                X_df=x_df,
                feature_contributions=True,
            )

            # Access feature_contributions from the client (wide format)
            contrib_df = nixtla_client.feature_contributions

            # Melt to long format to match Snowflake output
            id_cols = ["unique_id", "ds"]
            forecast_col = "TimeGPT"
            feature_cols = [
                c for c in contrib_df.columns if c not in id_cols and c != forecast_col
            ]

            result = pd.melt(
                contrib_df,
                id_vars=id_cols + [forecast_col],
                value_vars=feature_cols,
                var_name="feature",
                value_name="contribution",
            )
            result = result.rename(columns={forecast_col: "forecast"})
            return result[["unique_id", "ds", "forecast", "feature", "contribution"]]

        sf_df, client_df = self._execute_and_compare(
            snowflake_session,
            deployed_with_api_endpoint,
            example_dataframes,
            "explain_forecast",
            compare_explain_future_exog,
        )

        # Normalize column names to lowercase
        sf_df.columns = sf_df.columns.str.lower()
        client_df.columns = client_df.columns.str.lower()

        # Validate structure
        assert len(sf_df) > 0, "Explain with future exog returned no results"
        assert len(client_df) > 0, "Client explain with future exog returned no results"

        # Check required columns
        required_cols = ["unique_id", "ds", "forecast", "feature", "contribution"]
        for col in required_cols:
            assert col in sf_df.columns, f"Missing {col} in Snowflake"
            assert col in client_df.columns, f"Missing {col} in client"

        # Filter to series_2 for comparison (only series_2 has future exog)
        # Snowflake processes all series from EXAMPLE_TRAIN, but
        # the client comparison only computes for series_2.
        sf_series2 = sf_df[sf_df["unique_id"] == "series_2"].copy()
        client_series2 = client_df[client_df["unique_id"] == "series_2"].copy()

        # Validate that we have results for series_2
        assert len(sf_series2) > 0, "No explain results for series_2 from Snowflake"
        assert len(client_series2) > 0, "No explain results for series_2 from client"

        # Both should have the same features
        sf_features = sorted(sf_series2["feature"].unique())
        client_features = sorted(client_series2["feature"].unique())
        assert sf_features == client_features, (
            f"Feature mismatch: Snowflake={sf_features}, Client={client_features}"
        )

        # Validate row counts match (same number of rows per feature)
        assert len(sf_series2) == len(client_series2), (
            f"Row count mismatch for series_2: Snowflake={len(sf_series2)}, Client={len(client_series2)}"
        )

        # Sort and compare contributions
        sort_cols = ["unique_id", "ds", "feature"]
        sf_sorted = sf_series2.sort_values(sort_cols).reset_index(drop=True)
        client_sorted = client_series2.sort_values(sort_cols).reset_index(drop=True)

        # Compare forecast values
        pd.testing.assert_series_equal(
            sf_sorted["forecast"],
            client_sorted["forecast"],
            check_names=False,
            rtol=1e-3,
            atol=1e-3,
            obj="Explain forecast values with future exog",
        )

        # Compare contribution values
        pd.testing.assert_series_equal(
            sf_sorted["contribution"],
            client_sorted["contribution"],
            check_names=False,
            rtol=1e-3,
            atol=1e-3,
            obj="Feature contributions with future exog",
        )
