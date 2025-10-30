import inspect
import math
import warnings
from functools import partial

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest

import utilsforecast.losses as ufl
from utilsforecast.data import generate_series


warnings.filterwarnings("ignore", message="Unknown section References")


def pd_vs_pl(pd_df, pl_df, models):
    """Compare pandas and polars results for narwhals-based loss functions."""
    pd.testing.assert_frame_equal(
        pd_df[models].reset_index(drop=True),
        pl_df[models].to_pandas().reset_index(drop=True)
    )


# @pytest.fixture(scope="module")
def setup_series(engine):
    models = ["model0", "model1"]
    series = generate_series(10, n_models=2, level=[80], engine=engine)
    return series, models


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


def manual_loop(df, models, loss_fn, seasonality=None, baseline=None):
    df = nw.from_native(df)
    results = []
    requires_train = "y_train" in inspect.signature(loss_fn).parameters
    for uid, uid_df in df.group_by("unique_id"):
        uid_res = {"unique_id": uid}
        y_true = uid_df["y"].to_numpy()
        kwargs = {}
        if requires_train:
            kwargs["y_train"] = y_true
        if seasonality is not None:
            kwargs["seasonality"] = seasonality
        if baseline is not None:
            kwargs["y_pred_baseline"] = uid_df[baseline].to_numpy()
        for model in models:
            y_pred = uid_df[model].to_numpy()
            uid_res[model] = loss_fn(y_true, y_pred, **kwargs)
        results.append(uid_res)
    return pd.DataFrame(results)


def mae_single(y_true, y_pred, **kwargs):
    return np.abs(y_true - y_pred).mean()


def mse_single(y_true, y_pred, **kwargs):
    return np.square(y_true - y_pred).mean()


def rmse_single(y_true, y_pred, **kwargs):
    return np.sqrt(np.square(y_true - y_pred).mean())


def bias_single(y_true, y_pred, **kwargs):
    return np.mean(y_pred - y_true)


def cfe_single(y_true, y_pred, **kwargs):
    return np.sum(y_pred - y_true)


def pis_single(y_true, y_pred, **kwargs):
    return np.sum(np.abs(y_pred - y_true))


def spis_single(y_true, y_pred, **kwargs):
    return np.sum(np.abs(y_pred - y_true)) / np.mean(y_true)


def mape_single(y_true, y_pred, **kwargs):
    return np.mean(np.abs(y_true - y_pred) / y_true)


def smape_single(y_true, y_pred, **kwargs):
    return np.mean(np.abs(y_true - y_pred) / (y_true + y_pred))


def mase_single(y_true, y_pred, y_train, seasonality, **kwargs):
    scale = np.abs(y_train[:-seasonality] - y_train[seasonality:]).mean()
    return np.abs(y_true - y_pred).mean() / scale


def rmae_single(y_true, y_pred, baseline, **kwargs):
    num = np.abs(y_true - y_pred).mean()
    den = np.abs(y_true - baseline).mean()
    return num / den


def msse_single(y_true, y_pred, seasonality, y_train, **kwargs):
    num = np.square(y_true - y_pred).mean()
    den = np.square(y_train[:-seasonality] - y_train[seasonality:]).mean()
    return num / den


def rmsse_single(y_true, y_pred, seasonality, y_train, **kwargs):
    num = np.square(y_true - y_pred).mean()
    den = np.square(y_train[:-seasonality] - y_train[seasonality:]).mean()
    return math.sqrt(num / den)


def nd_single(y_true, y_pred, **kwargs):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


def coverage_single(y_true, y_pred_lo, y_pred_hi, **kwargs):
    return np.mean((y_true >= y_pred_lo) & (y_true <= y_pred_hi))


def tweedie_deviance_single(y_true, y_pred, power, **kwargs):
    if power == 0:
        return np.mean((y_true - y_pred) ** 2)
    elif power == 1:
        return np.mean(2 * (y_true * np.log(y_true / y_pred) - (y_true - y_pred)))
    elif power == 2:
        return np.mean(2 * (np.log(y_pred) - np.log(y_true)) + y_true / y_pred - 1)
    else:
        return np.mean(
            2
            * (
                (y_true ** (2 - power)) / ((1 - power) * (2 - power))
                - (y_true * (y_pred ** (1 - power))) / (1 - power)
                + (y_pred ** (2 - power)) / (2 - power)
            )
        )


def linex_single(y_true, y_pred, a=1.0, **kwargs):
    error = y_pred - y_true
    return np.mean(np.exp(a * error) - a * error - 1)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "utils_fn,single_fn",
    [
        (ufl.mae, mae_single),
        (ufl.mse, mse_single),
        (ufl.rmse, rmse_single),
        (ufl.bias, bias_single),
        (ufl.cfe, cfe_single),
        (ufl.pis, pis_single),
        (ufl.spis, spis_single),
        (ufl.mape, mape_single),
        (ufl.smape, smape_single),
        (ufl.mase, mase_single),
        (ufl.rmae, rmae_single),
        (ufl.msse, msse_single),
        (ufl.rmsse, rmsse_single),
        (ufl.nd, nd_single),
        (ufl.coverage, coverage_single),
        (ufl.linex, linex_single),
        (
            partial(ufl.linex, a=2.0),
            partial(linex_single, a=2.0),
        ),
        (
            partial(ufl.linex, a=-1.5),
            partial(linex_single, a=-1.5),
        ),
        (
            partial(ufl.tweedie_deviance, power=0),
            partial(tweedie_deviance_single, power=0),
        ),
        (
            partial(ufl.tweedie_deviance, power=1),
            partial(tweedie_deviance_single, power=1),
        ),
        (
            partial(ufl.tweedie_deviance, power=2),
            partial(tweedie_deviance_single, power=2),
        ),
    ],
)
def test_loss(engine, utils_fn, single_fn):
    series, models = setup_series(engine)
    seasonality = 7
    level = 80
    baseline = models[0]
    loss_params = inspect.signature(utils_fn).parameters
    kwargs = {}
    if "seasonality" in loss_params:
        kwargs["seasonality"] = seasonality
    if "train_df" in loss_params:
        kwargs["train_df"] = series
    if "baseline" in loss_params:
        kwargs["baseline"] = baseline
    if "level" in loss_params:
        kwargs["level"] = level
    if "a" in loss_params:
        # Get the default value or use 1.0
        kwargs["a"] = loss_params["a"].default if loss_params["a"].default != inspect.Parameter.empty else 1.0
    actual = utils_fn(series, models, **kwargs)

    df = nw.from_native(series)
    results = []
    for (uid,), uid_df in df.group_by("unique_id"):
        uid_res = {"unique_id": uid}
        y_true = uid_df["y"].to_numpy()
        for model in models:
            single_kwargs = {}
            if "level" in loss_params:
                single_kwargs["y_pred_lo"] = uid_df[f"{model}-lo-{level}"].to_numpy()
                single_kwargs["y_pred_hi"] = uid_df[f"{model}-hi-{level}"].to_numpy()
            y_pred = uid_df[model].to_numpy()
            uid_res[model] = single_fn(
                y_true,
                y_pred=y_pred,
                y_train=y_true,
                seasonality=seasonality,
                baseline=uid_df[baseline].to_numpy(),
                **single_kwargs,
            )
        results.append(uid_res)

    expected = nw.from_native(type(series)(results)).sort("unique_id")
    actual = nw.from_native(actual).sort("unique_id")
    np.testing.assert_allclose(actual[models], expected[models])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_linex_loss_numerical(engine):
    """Test linex loss with known numerical values."""
    a = 0.2

    # Create test data using narwhals-agnostic approach
    data = {
        "unique_id": [0, 0, 0],
        "y": [1.0, 2.0, 3.0],
        "model": [1.0, 2.5, 2.0],
    }

    if engine == "pandas":
        df = pd.DataFrame(data)
    else:
        import polars as pl
        df = pl.DataFrame(data)

    # Calculate expected value
    # errors: model - y = [0.0, 0.5, -1.0]
    # linex: exp(a*e) - a*e - 1
    errors = np.array([0.0, 0.5, -1.0])
    expected_values = np.exp(a * errors) - a * errors - 1
    expected_mean = expected_values.mean()

    # Calculate actual using narwhals
    result = ufl.linex(df, ["model"], a=a)
    actual_value = nw.from_native(result)["model"].item()

    # Assert
    np.testing.assert_allclose(actual_value, expected_mean, rtol=1e-10)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_spis(engine):
    """Test scaled PIS (sPIS)."""
    series, models = setup_series(engine)
    result = ufl.spis(df=series, train_df=series, models=models)

    # Check that result has correct shape
    assert result.shape[0] > 0
    df_nw = nw.from_native(result)

    # Check that all model columns exist
    for col in models:
        assert col in df_nw.columns
        # Check no nulls - narwhals series uses null_count()
        assert df_nw[col].null_count() == 0


def quantile_loss_single(y_true, y_pred, q, **kwargs):
    delta_y = y_true - y_pred
    return np.maximum(q * delta_y, (q - 1) * delta_y).mean()


def scaled_quantile_loss_single(y_true, y_pred, q, y_train, seasonality, **kwargs):
    qloss = quantile_loss_single(y_true, y_pred, q)
    scale = mae_single(y_train[seasonality:], y_train[:-seasonality])
    return qloss / scale


def calibration_single(y_true, y_pred, **kwargs):
    return np.mean(y_true < y_pred)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("q", [0.1, 0.9])
@pytest.mark.parametrize(
    "utils_fn,single_fn",
    [
        (ufl.quantile_loss, quantile_loss_single),
        (ufl.scaled_quantile_loss, scaled_quantile_loss_single),
        (ufl.calibration, calibration_single),
    ],
)
def test_single_quantile_losses(engine, utils_fn, single_fn, q):
    series, models = setup_series(engine)
    q_models = {
        0.1: {
            "model0": "model0-lo-80",
            "model1": "model1-lo-80",
        },
        0.9: {
            "model0": "model0-hi-80",
            "model1": "model1-hi-80",
        },
    }
    seasonality = 7
    loss_params = inspect.signature(utils_fn).parameters
    kwargs = {}
    if "seasonality" in loss_params:
        kwargs["seasonality"] = seasonality
    if "train_df" in loss_params:
        kwargs["train_df"] = series
    if "q" in loss_params:
        kwargs["q"] = q
    actual = utils_fn(series, q_models[q], **kwargs)

    df = nw.from_native(series)
    results = []
    for (uid,), uid_df in df.group_by("unique_id"):
        uid_res = {"unique_id": uid}
        y_true = uid_df["y"].to_numpy()
        for model, pred_col in q_models[q].items():
            y_pred = uid_df[pred_col].to_numpy()
            uid_res[model] = single_fn(
                y_true,
                y_pred,
                y_train=y_true,
                seasonality=seasonality,
                q=q,
            )
        results.append(uid_res)

    expected = nw.from_native(type(series)(results)).sort("unique_id")
    actual = nw.from_native(actual).sort("unique_id")
    np.testing.assert_allclose(actual[models], expected[models])


class TestQuantileLoss:
    def test_quantile_loss_basic(self, quantile_test_data):
        df, ql_models_test, quantiles = quantile_test_data

        for q in quantiles:
            ql_df = ufl.quantile_loss(
                df, models=dict(zip(ql_models_test, ql_models_test)), q=q
            )
            # For overestimation, delta_y = y - y_hat = -1 so ql = max(-q, -(q-1))
            expected_over = max(-q, -(q - 1))
            assert all(expected_over == ql for ql in ql_df["overestimation"])
            # For underestimation, delta_y = y - y_hat = 1, so ql = max(q, q-1)
            expected_under = max(q, q - 1)
            assert all(expected_under == ql for ql in ql_df["underestimation"])

    @pytest.mark.parametrize("engine", ["pandas", "polars"])
    @pytest.mark.parametrize("q", [0.1, 0.9])
    def test_quantile_loss_engines(self, engine, q, quantile_models):
        series, models = setup_series(engine)
        result = ufl.quantile_loss(series, quantile_models[q], q=q)
        assert result.shape[0] > 0  # Has results
        df_nw = nw.from_native(result)
        for col in models:
            assert col in df_nw.columns

    @pytest.mark.parametrize("engine", ["pandas", "polars"])
    @pytest.mark.parametrize("q", [0.1, 0.9])
    def test_scaled_quantile_loss_engines(self, engine, q, quantile_models):
        series, models = setup_series(engine)
        result = ufl.scaled_quantile_loss(
            series, quantile_models[q], seasonality=1, train_df=series, q=q
        )
        assert result.shape[0] > 0  # Has results
        df_nw = nw.from_native(result)
        for col in models:
            assert col in df_nw.columns


class TestMultiQuantileLoss:
    def test_multi_quantile_loss_calculation(
        self, multi_quantile_models, quantile_models
    ):
        series, models = setup_series("pandas")
        quantiles = np.array([0.1, 0.9])

        expected = (
            pd.concat(
                [
                    ufl.quantile_loss(series, models=quantile_models[q], q=q)
                    for q in quantiles
                ]
            )
            .groupby("unique_id", observed=True, as_index=False)
            .mean()
        )

        actual = ufl.mqloss(series, models=multi_quantile_models, quantiles=quantiles)
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize("engine", ["pandas", "polars"])
    def test_multi_quantile_loss_engines(self, engine, multi_quantile_models):
        series, models = setup_series(engine)
        quantiles = np.array([0.1, 0.9])

        result = ufl.mqloss(series, multi_quantile_models, quantiles=quantiles)
        assert result.shape[0] > 0
        df_nw = nw.from_native(result)
        for col in models:
            assert col in df_nw.columns

    @pytest.mark.parametrize("engine", ["pandas", "polars"])
    def test_multi_quantile_loss_output_shape(self, engine, multi_quantile_models):
        series, models = setup_series(engine)
        quantiles = np.array([0.1, 0.9])

        df_nw = nw.from_native(series)
        df_test = df_nw.with_columns(nw.col("unique_id").cast(nw.String))

        mql_df = ufl.mqloss(df_test.to_native(), multi_quantile_models, quantiles=quantiles)

        # Check shape
        expected_shape = (df_nw["unique_id"].n_unique(), 1 + len(models))
        assert mql_df.shape == expected_shape

        # Check for null values
        mql_nw = nw.from_native(mql_df)
        for col in models:
            assert mql_nw[col].null_count() == 0

    @pytest.mark.parametrize("engine", ["pandas", "polars"])
    def test_scaled_multi_quantile_loss_engines(self, engine, multi_quantile_models):
        series, models = setup_series(engine)
        quantiles = np.array([0.1, 0.9])

        result = ufl.scaled_mqloss(
            series,
            multi_quantile_models,
            quantiles=quantiles,
            seasonality=1,
            train_df=series,
        )
        assert result.shape[0] > 0
        df_nw = nw.from_native(result)
        for col in models:
            assert col in df_nw.columns


class TestProbabilisticMetrics:
    @pytest.mark.parametrize("engine", ["pandas", "polars"])
    def test_coverage(self, engine):
        series, models = setup_series(engine)
        result = ufl.coverage(series, models, 80)
        assert result.shape[0] > 0
        df_nw = nw.from_native(result)
        for col in models:
            assert col in df_nw.columns

    @pytest.mark.parametrize("engine", ["pandas", "polars"])
    def test_calibration(self, engine, quantile_models):
        series, models = setup_series(engine)
        result = ufl.calibration(series, quantile_models[0.1])
        assert result.shape[0] > 0
        df_nw = nw.from_native(result)
        for col in models:
            assert col in df_nw.columns

    @pytest.mark.parametrize("engine", ["pandas", "polars"])
    def test_scaled_crps(self, engine, multi_quantile_models):
        series, models = setup_series(engine)
        quantiles = np.array([0.1, 0.9])

        result = ufl.scaled_crps(series, multi_quantile_models, quantiles)
        assert result.shape[0] > 0
        df_nw = nw.from_native(result)
        for col in models:
            assert col in df_nw.columns


class TestTweedieDeviance:
    @pytest.mark.parametrize("engine", ["pandas", "polars"])
    @pytest.mark.parametrize("power", [0, 1, 1.5, 2, 3])
    def test_non_zero_handling(self, engine, power):
        series, models = setup_series(engine)
        # Test Tweedie deviance
        td = ufl.tweedie_deviance(series, models, target_col="y", power=power)
        td_nw = nw.from_native(td)

        # Test for NaNs and infinites
        for col in models:
            assert td_nw[col].null_count() == 0, (
                f"NaNs found in {engine} DataFrame for power {power}"
            )

    @pytest.mark.parametrize("engine", ["pandas", "polars"])
    @pytest.mark.parametrize("power", [0, 1, 1.5])
    def test_zero_handling(self, engine, power):
        series, models = setup_series(engine)
        # Test zero handling (skip power >=2 since it requires all y > 0)
        # Set first unique_id's y values to 0 using narwhals
        series_nw = nw.from_native(series)
        first_id = series_nw["unique_id"].head(1).item()
        series_nw = series_nw.with_columns(
            nw.when(nw.col("unique_id") == first_id)
            .then(0.0)
            .otherwise(nw.col("y"))
            .alias("y")
        )
        series = series_nw.to_native()

        # Test Tweedie deviance
        td = ufl.tweedie_deviance(series, models, target_col="y", power=power)
        td_nw = nw.from_native(td)

        # Test for NaNs
        for col in models:
            assert td_nw[col].null_count() == 0, (
                f"NaNs found in {engine} DataFrame for power {power}"
            )
