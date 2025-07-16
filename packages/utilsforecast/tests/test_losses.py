import warnings

import numpy as np
import pandas as pd
import pytest

from utilsforecast.compat import POLARS_INSTALLED
from utilsforecast.data import generate_series
from utilsforecast.losses import (
    bias,
    calibration,
    coverage,
    mae,
    mape,
    mase,
    mqloss,
    mse,
    msse,
    quantile_loss,
    rmae,
    rmse,
    rmsse,
    scaled_crps,
    scaled_mqloss,
    scaled_quantile_loss,
    smape,
    tweedie_deviance,
)

if POLARS_INSTALLED:
    import polars as pl

warnings.filterwarnings("ignore", message="Unknown section References")


@pytest.fixture
def setup_series():
    models = ["model0", "model1"]
    series = generate_series(10, n_models=2, level=[80])
    series_pl = generate_series(10, n_models=2, level=[80], engine="polars")
    return series, series_pl, models


@pytest.fixture
def quantile_test_data():
    df = pd.DataFrame(
        {
            "unique_id": [0, 1, 2],
            "y": [1.0, 2.0, 3.0],
            "overestimation": [2.0, 3.0, 4.0],  # y + 1
            "underestimation": [0.0, 1.0, 2.0],  # y - 1
        }
    )
    df["unique_id"] = df["unique_id"].astype("category")
    df = pd.concat([df, df.assign(unique_id=2)]).reset_index(drop=True)

    ql_models_test = ["overestimation", "underestimation"]
    quantiles = np.array([0.1, 0.9])

    return df, ql_models_test, quantiles


@pytest.fixture
def quantile_models():
    return {
        0.1: {
            "model0": "model0-lo-80",
            "model1": "model1-lo-80",
        },
        0.9: {
            "model0": "model0-hi-80",
            "model1": "model1-hi-80",
        },
    }


@pytest.fixture
def multi_quantile_models():
    return {
        "model0": ["model0-lo-80", "model0-hi-80"],
        "model1": ["model1-lo-80", "model1-hi-80"],
    }


def pd_vs_pl(pd_df, pl_df, models):
    np.testing.assert_allclose(
        pd_df[models].to_numpy(),
        pl_df.sort("unique_id").select(models).to_numpy(),
    )


class TestBasicMetrics:
    def test_mae(self, setup_series):
        series, series_pl, models = setup_series
        pd_vs_pl(
            mae(series, models),
            mae(series_pl, models),
            models,
        )

    def test_mse(self, setup_series):
        series, series_pl, models = setup_series
        pd_vs_pl(
            mse(series, models),
            mse(series_pl, models),
            models,
        )

    def test_rmse(self, setup_series):
        series, series_pl, models = setup_series
        pd_vs_pl(
            rmse(series, models),
            rmse(series_pl, models),
            models,
        )

    def test_bias(self, setup_series):
        series, series_pl, models = setup_series
        pd_vs_pl(
            bias(series, models),
            bias(series_pl, models),
            models,
        )


class TestPercentageErrors:
    def test_mape(self, setup_series):
        series, series_pl, models = setup_series
        pd_vs_pl(
            mape(series, models),
            mape(series_pl, models),
            models,
        )

    def test_smape(self, setup_series):
        series, series_pl, models = setup_series
        pd_vs_pl(
            smape(series, models),
            smape(series_pl, models),
            models,
        )


class TestScaleIndependentErrors:
    def test_mase(self, setup_series):
        series, series_pl, models = setup_series
        pd_vs_pl(
            mase(series, models, 7, series),
            mase(series_pl, models, 7, series_pl),
            models,
        )

    def test_rmae(self, setup_series):
        series, series_pl, models = setup_series
        pd_vs_pl(
            rmae(series, models, models[0]),
            rmae(series_pl, models, models[0]),
            models,
        )

    def test_msse(self, setup_series):
        series, series_pl, models = setup_series
        pd_vs_pl(
            msse(series, models, 7, series),
            msse(series_pl, models, 7, series_pl),
            models,
        )

    def test_rmsse(self, setup_series):
        series, series_pl, models = setup_series
        pd_vs_pl(
            rmsse(series, models, 7, series),
            rmsse(series_pl, models, 7, series_pl),
            models,
        )


class TestQuantileLoss:
    def test_quantile_loss_basic(self, quantile_test_data):
        df, ql_models_test, quantiles = quantile_test_data

        for q in quantiles:
            ql_df = quantile_loss(
                df, models=dict(zip(ql_models_test, ql_models_test)), q=q
            )
            # For overestimation, delta_y = y - y_hat = -1 so ql = max(-q, -(q-1))
            expected_over = max(-q, -(q - 1))
            assert all(expected_over == ql for ql in ql_df["overestimation"])
            # For underestimation, delta_y = y - y_hat = 1, so ql = max(q, q-1)
            expected_under = max(q, q - 1)
            assert all(expected_under == ql for ql in ql_df["underestimation"])

    def test_quantile_loss_pandas_polars(self, setup_series, quantile_models):
        series, series_pl, models = setup_series
        quantiles = np.array([0.1, 0.9])

        for q in quantiles:
            pd_vs_pl(
                quantile_loss(series, quantile_models[q], q=q),
                quantile_loss(series_pl, quantile_models[q], q=q),
                models,
            )

    def test_scaled_quantile_loss(self, setup_series, quantile_models):
        series, series_pl, models = setup_series
        quantiles = np.array([0.1, 0.9])

        for q in quantiles:
            pd_vs_pl(
                scaled_quantile_loss(
                    series, quantile_models[q], seasonality=1, train_df=series, q=q
                ),
                scaled_quantile_loss(
                    series_pl,
                    quantile_models[q],
                    seasonality=1,
                    train_df=series_pl,
                    q=q,
                ),
                models,
            )


class TestMultiQuantileLoss:
    def test_multi_quantile_loss_calculation(
        self, setup_series, multi_quantile_models, quantile_models
    ):
        series, _, models = setup_series
        quantiles = np.array([0.1, 0.9])

        expected = (
            pd.concat(
                [
                    quantile_loss(series, models=quantile_models[q], q=q)
                    for q in quantiles
                ]
            )
            .groupby("unique_id", observed=True, as_index=False)
            .mean()
        )

        actual = mqloss(series, models=multi_quantile_models, quantiles=quantiles)
        pd.testing.assert_frame_equal(actual, expected)

    def test_multi_quantile_loss_pandas_polars(
        self, setup_series, multi_quantile_models
    ):
        series, series_pl, models = setup_series
        quantiles = np.array([0.1, 0.9])

        pd_vs_pl(
            mqloss(series, multi_quantile_models, quantiles=quantiles),
            mqloss(series_pl, multi_quantile_models, quantiles=quantiles),
            models,
        )

    def test_multi_quantile_loss_output_shape(
        self, setup_series, multi_quantile_models
    ):
        series, series_pl, models = setup_series
        quantiles = np.array([0.1, 0.9])

        for series_df in [series, series_pl]:
            if isinstance(series_df, pd.DataFrame):
                df_test = series_df.assign(
                    unique_id=lambda df: df["unique_id"].astype(str)
                )
            else:
                df_test = series_df.with_columns(pl.col("unique_id").cast(pl.Utf8))

            mql_df = mqloss(df_test, multi_quantile_models, quantiles=quantiles)

            # Check shape
            expected_shape = (series["unique_id"].nunique(), 1 + len(models))
            assert mql_df.shape == expected_shape

            # Check for null values
            if isinstance(mql_df, pd.DataFrame):
                null_vals = mql_df.isna().sum().sum()
            else:
                null_vals = mql_df.select(pl.all().is_null().sum()).sum_horizontal()
            assert null_vals.item() == 0

    def test_scaled_multi_quantile_loss(self, setup_series, multi_quantile_models):
        series, series_pl, models = setup_series
        quantiles = np.array([0.1, 0.9])

        pd_vs_pl(
            scaled_mqloss(
                series,
                multi_quantile_models,
                quantiles=quantiles,
                seasonality=1,
                train_df=series,
            ),
            scaled_mqloss(
                series_pl,
                multi_quantile_models,
                quantiles=quantiles,
                seasonality=1,
                train_df=series_pl,
            ),
            models,
        )


class TestProbabilisticMetrics:
    def test_coverage(self, setup_series):
        series, series_pl, models = setup_series
        pd_vs_pl(
            coverage(series, models, 80),
            coverage(series_pl, models, 80),
            models,
        )

    def test_calibration(self, setup_series, quantile_models):
        series, series_pl, models = setup_series
        pd_vs_pl(
            calibration(series, quantile_models[0.1]),
            calibration(series_pl, quantile_models[0.1]),
            models,
        )

    def test_scaled_crps(self, setup_series, multi_quantile_models):
        series, series_pl, models = setup_series
        quantiles = np.array([0.1, 0.9])

        pd_vs_pl(
            scaled_crps(series, multi_quantile_models, quantiles),
            scaled_crps(series_pl, multi_quantile_models, quantiles),
            models,
        )

class TestTweedieDeviance:
    
    @pytest.mark.parametrize("power", [0, 1, 1.5, 2, 3])
    def test_non_zero_handling(self, setup_series, power):
        series, series_pl, models = setup_series
    # for power in [0, 1, 1.5, 2, 3]:
        # Test Pandas vs Polars
        td_pd = tweedie_deviance(series,   models, target_col="y", power=power)
        td_pl = tweedie_deviance(series_pl, models, target_col="y", power=power)
        pd_vs_pl(
            td_pd,
            td_pl,
            models,
        )
        # Test for NaNs
        assert not td_pd[models].isna().any().any(), f"NaNs found in pd DataFrame for power {power}"
        assert not td_pl.select(pl.col(models).is_null().any()).sum_horizontal().item(), f"NaNs found in pl DataFrame for power {power}"
        # Test for infinites
        is_infinite = td_pd[models].isin([np.inf, -np.inf]).any().any()
        assert not is_infinite, f"Infinities found in pd DataFrame for power {power}"
        is_infinite_pl = td_pl.select(pl.col(models).is_infinite().any()).sum_horizontal().item()
        assert not is_infinite_pl, f"Infinities found in pl DataFrame for power {power}"

    @pytest.mark.parametrize("power", [0, 1, 1.5])
    def test_zero_handling(self, setup_series, power):
        series, series_pl, models = setup_series
        # Test zero handling (skip power >=2 since it requires all y > 0)
        series.loc[0, 'y'] = 0.0  # Set a zero value to test the zero handling
        series.loc[49, 'y'] = 0.0  # Set another zero value to test the zero handling
        series_pl[0, 'y'] = 0.0  # Set a zero value to test the zero handling
        series_pl[49, 'y'] = 0.0  # Set another zero value to test the zero handling
        
        # Test Pandas vs Polars
        td_pd = tweedie_deviance(series,   models, target_col="y", power=power)
        td_pl = tweedie_deviance(series_pl, models, target_col="y", power=power)
        pd_vs_pl(
            td_pd,
            td_pl,
            models,
        )
        # Test for NaNs
        assert not td_pd[models].isna().any().any(), f"NaNs found in pd DataFrame for power {power}"
        assert not td_pl.select(pl.col(models).is_null().any()).sum_horizontal().item(), f"NaNs found in pl DataFrame for power {power}"
        # Test for infinites
        is_infinite = td_pd[models].isin([np.inf, -np.inf]).any().any()
        assert not is_infinite, f"Infinities found in pd DataFrame for power {power}"
        is_infinite_pl = td_pl.select(pl.col(models).is_infinite().any()).sum_horizontal().item()
        assert not is_infinite_pl, f"Infinities found in pl DataFrame for power {power}"