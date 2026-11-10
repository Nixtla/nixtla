import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest

import utilsforecast.processing as ufp
from utilsforecast.data import generate_series
from utilsforecast.rectify import (
    align_rectify_features,
    compute_rectify_residuals,
    rectify,
)


def _make_actuals_and_forecasts(engine="pandas", n_series=3, h=5, seed=42):
    series = generate_series(
        n_series=n_series,
        freq="D" if engine == "pandas" else "1d",
        min_length=50,
        max_length=50,
        equal_ends=True,
        engine=engine,
        seed=seed,
    )
    series_nw = nw.from_native(series)
    rng = np.random.RandomState(seed)

    actuals_list = []
    forecasts_list = []
    for uid in sorted(series_nw["unique_id"].unique().to_list()):
        grp = series_nw.filter(nw.col("unique_id") == uid).tail(h)
        y = grp["y"].to_numpy()
        bias = np.arange(1, h + 1) * 0.5 + rng.randn(h) * 0.1
        actuals_nw = grp.select(["unique_id", "ds", "y"])
        actuals_list.append(nw.to_native(actuals_nw))
        forecasts_native = nw.to_native(grp.select(["unique_id", "ds"]))
        forecasts_native = ufp.assign_columns(forecasts_native, "model0", y + bias)
        forecasts_native = ufp.assign_columns(forecasts_native, "model1", y - bias * 0.8)
        forecasts_list.append(forecasts_native)

    actuals_df = ufp.vertical_concat(actuals_list, match_categories=False)
    forecasts_df = ufp.vertical_concat(forecasts_list, match_categories=False)
    return actuals_df, forecasts_df


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_residual_values(engine):
    actuals_df, forecasts_df = _make_actuals_and_forecasts(engine=engine)
    models = ["model0", "model1"]

    result = compute_rectify_residuals(
        df=actuals_df,
        forecasts_df=forecasts_df,
        models=models,
    )
    result_nw = nw.from_native(result)
    actuals_nw = nw.from_native(actuals_df)
    forecasts_nw = nw.from_native(forecasts_df)

    assert "horizon" in result_nw.columns
    assert result_nw["horizon"].min() == 1
    assert result_nw["horizon"].max() == 5
    assert result_nw.shape[0] == actuals_nw.shape[0]

    actuals_sorted = actuals_nw.sort("unique_id", "ds")
    forecasts_sorted = forecasts_nw.sort("unique_id", "ds")
    result_sorted = result_nw.sort("unique_id", "ds")
    for model in models:
        expected = actuals_sorted["y"].to_numpy() - forecasts_sorted[model].to_numpy()
        actual = result_sorted[model].to_numpy()
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_single_model(engine):
    actuals_df, forecasts_df = _make_actuals_and_forecasts(engine=engine)

    result = compute_rectify_residuals(
        df=actuals_df,
        forecasts_df=forecasts_df,
        models=["model0"],
    )
    result_nw = nw.from_native(result)
    assert "model0" in result_nw.columns
    assert "model1" not in result_nw.columns


def test_custom_column_names():
    actuals_df = pd.DataFrame({
        "series_id": [0, 0, 0, 1, 1, 1],
        "timestamp": pd.date_range("2020-01-01", periods=3).tolist() * 2,
        "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    forecasts_df = pd.DataFrame({
        "series_id": [0, 0, 0, 1, 1, 1],
        "timestamp": pd.date_range("2020-01-01", periods=3).tolist() * 2,
        "my_model": [1.1, 2.2, 3.3, 4.1, 5.2, 6.3],
    })
    result = compute_rectify_residuals(
        df=actuals_df,
        forecasts_df=forecasts_df,
        models=["my_model"],
        id_col="series_id",
        time_col="timestamp",
        target_col="target",
    )
    result_nw = nw.from_native(result)
    assert set(result_nw.columns) == {"series_id", "timestamp", "horizon", "my_model"}


def test_residuals_misaligned_rows_warns():
    actuals_df = pd.DataFrame({
        "unique_id": [0, 0, 0, 1, 1, 1],
        "ds": pd.date_range("2020-01-01", periods=3).tolist() * 2,
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    # Drop one row so actuals and forecasts don't align perfectly
    forecasts_df = pd.DataFrame({
        "unique_id": [0, 0, 0, 1, 1],
        "ds": pd.date_range("2020-01-01", periods=3).tolist() + pd.date_range("2020-01-01", periods=2).tolist(),
        "model0": [1.1, 2.2, 3.3, 4.1, 5.2],
    })
    with pytest.warns(UserWarning, match="1 row\\(s\\) in actuals had no matching entry"):
        compute_rectify_residuals(
            df=actuals_df, forecasts_df=forecasts_df, models=["model0"]
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_residual_values_with_cutoff_col(engine):
    actuals_df, forecasts_df = _make_actuals_and_forecasts(
        engine=engine, n_series=2, h=3
    )
    actuals = nw.from_native(actuals_df)
    forecasts = nw.from_native(forecasts_df)
    actuals_folds = []
    forecasts_folds = []
    for cutoff in range(2):
        actuals_folds.append(
            nw.to_native(actuals.with_columns(nw.lit(cutoff).alias("cutoff")))
        )
        forecasts_folds.append(
            nw.to_native(
                forecasts.with_columns(
                    nw.lit(cutoff).alias("cutoff"),
                    (nw.col("model0") + cutoff).alias("model0"),
                    (nw.col("model1") - cutoff).alias("model1"),
                )
            )
        )

    actuals_cv = ufp.vertical_concat(actuals_folds, match_categories=False)
    forecasts_cv = ufp.vertical_concat(forecasts_folds, match_categories=False)
    result = compute_rectify_residuals(
        df=actuals_cv,
        forecasts_df=forecasts_cv,
        models=["model0", "model1"],
        cutoff_col="cutoff",
    )

    result_sorted = nw.from_native(result).sort("unique_id", "cutoff", "ds")
    actuals_sorted = nw.from_native(actuals_cv).sort("unique_id", "cutoff", "ds")
    forecasts_sorted = nw.from_native(forecasts_cv).sort("unique_id", "cutoff", "ds")
    assert set(result_sorted.columns) == {
        "unique_id",
        "ds",
        "cutoff",
        "horizon",
        "model0",
        "model1",
    }
    for uid in result_sorted["unique_id"].unique().to_list():
        for cutoff in range(2):
            fold = result_sorted.filter(
                (nw.col("unique_id") == uid) & (nw.col("cutoff") == cutoff)
            )
            np.testing.assert_array_equal(
                fold["horizon"].to_numpy(), np.arange(1, 4)
            )

    for model in ["model0", "model1"]:
        expected = actuals_sorted["y"].to_numpy() - forecasts_sorted[model].to_numpy()
        actual = result_sorted[model].to_numpy()
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_align_per_horizon(engine):
    actuals_df, forecasts_df = _make_actuals_and_forecasts(engine=engine, h=5)
    models = ["model0", "model1"]
    residuals_df = compute_rectify_residuals(
        df=actuals_df, forecasts_df=forecasts_df, models=models,
    )
    features = np.random.RandomState(0).randn(len(nw.from_native(actuals_df)), 3)

    result = align_rectify_features(residuals_df=residuals_df, features=features)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(range(1, 6))
    for h in range(1, 6):
        X, y_dict = result[h]
        assert X.ndim == 2
        assert X.shape[1] == 3
        assert set(y_dict.keys()) == set(models)
        for model in models:
            assert len(y_dict[model]) == X.shape[0]



@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_align_horizon_aware(engine):
    actuals_df, forecasts_df = _make_actuals_and_forecasts(engine=engine, h=5)
    residuals_df = compute_rectify_residuals(
        df=actuals_df, forecasts_df=forecasts_df, models=["model0"],
    )
    n_rows = nw.from_native(actuals_df).shape[0]
    features = np.random.RandomState(0).randn(n_rows, 3)

    X, y_dict = align_rectify_features(
        residuals_df=residuals_df, features=features, mode="horizon_aware",
    )
    assert X.shape == (n_rows, 4)


def test_align_invalid_mode():
    actuals_df, forecasts_df = _make_actuals_and_forecasts(engine="pandas", h=2)
    residuals_df = compute_rectify_residuals(
        df=actuals_df, forecasts_df=forecasts_df, models=["model0"],
    )
    features = np.zeros((nw.from_native(residuals_df).shape[0], 2))

    with pytest.raises(ValueError, match="mode must be"):
        align_rectify_features(
            residuals_df=residuals_df,
            features=features,
            mode="wrong_mode",
        )


def test_align_feature_row_mismatch():
    actuals_df, forecasts_df = _make_actuals_and_forecasts(engine="pandas", h=2)
    residuals_df = compute_rectify_residuals(
        df=actuals_df, forecasts_df=forecasts_df, models=["model0"],
    )

    with pytest.raises(ValueError, match="same number of rows"):
        align_rectify_features(
            residuals_df=residuals_df,
            features=np.zeros((1, 2)),
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_rectify_zero_correction(engine):
    _, forecasts_df = _make_actuals_and_forecasts(engine=engine, h=5)
    models = ["model0", "model1"]
    forecasts_nw = nw.from_native(forecasts_df)

    class ZeroCorrector:
        def predict(self, X):
            return np.zeros(X.shape[0])

    correction_models = {
        h: {m: ZeroCorrector() for m in models}
        for h in range(1, 6)
    }
    features = np.zeros((forecasts_nw.shape[0], 2))

    result = rectify(
        df=forecasts_df,
        models=models,
        correction_models=correction_models,
        features=features,
    )
    result_nw = nw.from_native(result)
    for model in models:
        np.testing.assert_allclose(
            result_nw[model].to_numpy(),
            forecasts_nw[model].to_numpy(),
        )



@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_rectify_constant_correction(engine):
    _, forecasts_df = _make_actuals_and_forecasts(engine=engine, h=5)
    forecasts_nw = nw.from_native(forecasts_df)

    class ConstantCorrector:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(X.shape[0], self.value)

    correction_models = {
        h: {"model0": ConstantCorrector(value=h * 0.1)}
        for h in range(1, 6)
    }
    features = np.random.RandomState(0).randn(forecasts_nw.shape[0], 3)

    result = rectify(
        df=forecasts_df,
        models=["model0"],
        correction_models=correction_models,
        features=features,
    )
    result_nw = nw.from_native(result).sort("unique_id", "ds")
    base_nw = forecasts_nw.sort("unique_id", "ds")

    for uid in base_nw["unique_id"].unique().to_list():
        base_vals = base_nw.filter(nw.col("unique_id") == uid)["model0"].to_numpy()
        corrected = result_nw.filter(nw.col("unique_id") == uid)["model0"].to_numpy()
        for h_idx in range(5):
            np.testing.assert_allclose(
                corrected[h_idx], base_vals[h_idx] + (h_idx + 1) * 0.1,
            )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_rectify_horizon_aware(engine):
    _, forecasts_df = _make_actuals_and_forecasts(engine=engine, h=5)
    forecasts_nw = nw.from_native(forecasts_df)

    class HorizonAwareCorrector:
        def __init__(self, scale):
            self.scale = scale

        def predict(self, X):
            return X[:, -1] * self.scale

    result = rectify(
        df=forecasts_df,
        models=["model0"],
        correction_models={"model0": HorizonAwareCorrector(scale=0.2)},
        features=np.zeros((forecasts_nw.shape[0], 2)),
        mode="horizon_aware",
    )
    result_nw = nw.from_native(result).sort("unique_id", "ds")
    base_nw = forecasts_nw.sort("unique_id", "ds")

    for uid in base_nw["unique_id"].unique().to_list():
        base_vals = base_nw.filter(nw.col("unique_id") == uid)["model0"].to_numpy()
        corrected = result_nw.filter(nw.col("unique_id") == uid)["model0"].to_numpy()
        expected = base_vals + np.arange(1, 6) * 0.2
        np.testing.assert_allclose(corrected, expected)


def test_rectify_feature_row_mismatch():
    _, forecasts_df = _make_actuals_and_forecasts(engine="pandas", h=2)

    with pytest.raises(ValueError, match="same number of rows"):
        rectify(
            df=forecasts_df,
            models=["model0"],
            correction_models={},
            features=np.zeros((1, 2)),
        )


def test_rectify_invalid_correction_model_shape():
    _, forecasts_df = _make_actuals_and_forecasts(engine="pandas", h=2)
    forecasts_nw = nw.from_native(forecasts_df)

    class ZeroCorrector:
        def predict(self, X):
            return np.zeros(X.shape[0])

    with pytest.raises(ValueError, match="missing horizons"):
        rectify(
            df=forecasts_df,
            models=["model0"],
            correction_models={"model0": ZeroCorrector()},
            features=np.zeros((forecasts_nw.shape[0], 2)),
        )


def test_rectify_invalid_per_horizon_correction_model_shape():
    _, forecasts_df = _make_actuals_and_forecasts(engine="pandas", h=2)
    forecasts_nw = nw.from_native(forecasts_df)

    class ZeroCorrector:
        def predict(self, X):
            return np.zeros(X.shape[0])

    with pytest.raises(ValueError, match="map each horizon"):
        rectify(
            df=forecasts_df,
            models=["model0"],
            correction_models={h: ZeroCorrector() for h in range(1, 3)},
            features=np.zeros((forecasts_nw.shape[0], 2)),
        )


def test_residuals_missing_target_col():
    df = pd.DataFrame({
        "unique_id": [0, 0],
        "ds": pd.date_range("2020-01-01", periods=2),
    })
    forecasts_df = pd.DataFrame({
        "unique_id": [0, 0],
        "ds": pd.date_range("2020-01-01", periods=2),
        "model0": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="missing"):
        compute_rectify_residuals(df=df, forecasts_df=forecasts_df, models=["model0"])


def test_residuals_missing_model_col():
    df = pd.DataFrame({
        "unique_id": [0, 0],
        "ds": pd.date_range("2020-01-01", periods=2),
        "y": [1.0, 2.0],
    })
    forecasts_df = pd.DataFrame({
        "unique_id": [0, 0],
        "ds": pd.date_range("2020-01-01", periods=2),
    })
    with pytest.raises(ValueError, match="missing model columns"):
        compute_rectify_residuals(df=df, forecasts_df=forecasts_df, models=["model0"])


def test_rectify_missing_model_col():
    df = pd.DataFrame({
        "unique_id": [0, 0],
        "ds": pd.date_range("2020-01-01", periods=2),
    })
    with pytest.raises(ValueError, match="missing model columns"):
        rectify(
            df=df, models=["model0"],
            correction_models={}, features=np.zeros((2, 1)),
        )
