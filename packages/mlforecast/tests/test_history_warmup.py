import datetime
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast
from mlforecast.conformal_prediction import PredictionIntervals
from mlforecast.lag_transforms import (
    ExpandingMean,
    ExponentiallyWeightedMean,
    RollingMean,
)
from mlforecast.target_transforms import Differences, LocalStandardScaler
from mlforecast.utils import generate_daily_series

H = 4


def _gen(engine, n_static_features=0):
    return generate_daily_series(
        10,
        min_length=60,
        max_length=80,
        equal_ends=True,
        n_static_features=n_static_features,
        static_as_categorical=False,
        engine=engine,
    )


def _split_last(df, engine, days):
    if engine == "polars":
        cutoff = df["ds"].max() - datetime.timedelta(days=days)
        return df.filter(pl.col("ds") <= cutoff), df.filter(pl.col("ds") > cutoff)
    cutoff = df["ds"].max() - datetime.timedelta(days=days)
    return df[df["ds"] <= cutoff], df[df["ds"] > cutoff]


def _config(kind, engine="pandas"):
    if kind == "local":
        return dict(
            lags=[1, 7],
            lag_transforms={
                1: [
                    RollingMean(7),
                    ExpandingMean(),
                    ExponentiallyWeightedMean(alpha=0.5),
                ]
            },
        )
    if kind == "pooled":
        return dict(
            lags=[1, 7],
            lag_transforms={
                1: [
                    RollingMean(7),
                    RollingMean(7, global_=True),
                    RollingMean(7, groupby=["static_0"]),
                    ExpandingMean(global_=True),
                ]
            },
        )
    assert kind == "target_tfms"
    return dict(
        lags=[1, 7],
        lag_transforms={1: [RollingMean(7), ExpandingMean()]},
        target_transforms=[Differences([1]), LocalStandardScaler()],
        date_features=["weekday" if engine == "polars" else "dayofweek"],
    )


def _new_fcst(kind, engine="pandas"):
    freq = "1d" if engine == "polars" else "D"
    return MLForecast(models=[LinearRegression()], freq=freq, **_config(kind, engine))


def _assert_preds_equal(p1, p2):
    np.testing.assert_allclose(
        np.asarray(p1["LinearRegression"]),
        np.asarray(p2["LinearRegression"]),
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["local", "pooled", "target_tfms"])
def test_history_warmup_predict_parity(engine, kind):
    n_statics = 1 if kind == "pooled" else 0
    fitted = _new_fcst(kind, engine)
    fitted.fit(_gen(engine, n_statics))
    expected = fitted.predict(H)

    warmed = _new_fcst(kind, engine)
    warmed.models_ = fitted.models_
    assert warmed.history_warmup(_gen(engine, n_statics)) is warmed
    _assert_preds_equal(expected, warmed.predict(H))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["local", "pooled", "target_tfms"])
def test_history_warmup_update_predict_parity(engine, kind):
    n_statics = 1 if kind == "pooled" else 0
    train, tail = _split_last(_gen(engine, n_statics), engine, days=3)

    fitted = _new_fcst(kind, engine)
    fitted.fit(train)
    fitted.update(tail)
    expected = fitted.predict(H)

    train, tail = _split_last(_gen(engine, n_statics), engine, days=3)
    warmed = _new_fcst(kind, engine)
    warmed.models_ = fitted.models_
    warmed.history_warmup(train)
    warmed.update(tail)
    _assert_preds_equal(expected, warmed.predict(H))


def test_history_warmup_partition_by():
    series = _gen("pandas")
    # alternate promo daily so every 3-wide window has samples in both buckets
    series["promo"] = series["ds"].dt.dayofweek % 2

    def new_fcst():
        return MLForecast(
            models=[LinearRegression()],
            freq="D",
            lags=[1],
            lag_transforms={1: [RollingMean(3, partition_by=["promo"])]},
        )

    fitted = new_fcst()
    fitted.fit(series.copy(), static_features=[])
    last = series["ds"].max()
    x_df = series[series["ds"] > last - datetime.timedelta(days=H)][
        ["unique_id", "ds"]
    ].copy()
    x_df["ds"] += datetime.timedelta(days=H)
    x_df["promo"] = x_df["ds"].dt.dayofweek % 2
    expected = fitted.predict(H, X_df=x_df)

    warmed = new_fcst()
    warmed.models_ = fitted.models_
    warmed.history_warmup(series.copy(), static_features=[])
    _assert_preds_equal(expected, warmed.predict(H, X_df=x_df))


def test_history_warmup_trims_like_fit():
    fitted = _new_fcst("pooled")
    fitted.fit(_gen("pandas", 1))

    warmed = _new_fcst("pooled")
    warmed.history_warmup(_gen("pandas", 1))

    # keep_last_n inferred like in fit and per-series arrays trimmed to it
    assert warmed.ts.keep_last_n == fitted.ts.keep_last_n
    np.testing.assert_array_equal(warmed.ts.ga.indptr, fitted.ts.ga.indptr)
    # pooled states trimmed the same way (finite-window ones trim, the state
    # containing ExpandingMean keeps full history)
    for key, state in warmed.ts._pooled_states.items():
        fitted_state = fitted.ts._pooled_states[key]
        np.testing.assert_array_equal(state.time_index, fitted_state.time_index)

    explicit = _new_fcst("local")
    explicit.history_warmup(_gen("pandas"), keep_last_n=30)
    assert explicit.ts.keep_last_n == 30
    assert np.diff(explicit.ts.ga.indptr).max() == 30


def test_history_warmup_multioutput_models():
    series = _gen("pandas")[["unique_id", "ds", "y"]]

    fitted = MLForecast(models=[LinearRegression()], freq="D", lags=[1, 7])
    fitted.fit(series.copy(), max_horizon=H)
    expected = fitted.predict(H)

    warmed = MLForecast(models=[LinearRegression()], freq="D", lags=[1, 7])
    warmed.models_ = fitted.models_
    warmed.history_warmup(series.copy(), max_horizon=H)
    _assert_preds_equal(expected, warmed.predict(H))

    # without max_horizon the existing model-shape guard fires
    unshaped = MLForecast(models=[LinearRegression()], freq="D", lags=[1, 7])
    unshaped.models_ = fitted.models_
    unshaped.history_warmup(series.copy())
    with pytest.raises(ValueError, match="one model per horizon"):
        unshaped.predict(H)


def test_history_warmup_preserves_prototype_metadata():
    """Re-warming an already fitted instance keeps its model-shape metadata."""
    series = _gen("pandas")[["unique_id", "ds", "y"]]
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1, 7])
    fcst.fit(series.copy(), max_horizon=H)
    expected = fcst.predict(H)

    fcst.history_warmup(series.copy())
    assert fcst.ts.max_horizon == H
    _assert_preds_equal(expected, fcst.predict(H))


def test_history_warmup_requires_models_for_predict():
    fcst = _new_fcst("local")
    fcst.history_warmup(_gen("pandas"))
    with pytest.raises(ValueError, match="No fitted models"):
        fcst.predict(H)


def test_history_warmup_preserves_horizon_features_shape():
    """Re-warming an already-fitted instance with `horizon_features` but no
    explicit `max_horizon`/`horizons` must resolve against the preserved
    shape instead of raising ("only supported when using max_horizon or
    horizons")."""
    df = _gen("pandas")[["unique_id", "ds", "y"]]
    df["exog_h1"] = df["y"] * 0.9
    df["exog_h2"] = df["y"] * 0.7
    horizon_features = {1: ["exog_h1"], 2: ["exog_h2"]}

    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    fcst.fit(
        df.copy(),
        static_features=[],
        max_horizon=H,
        horizon_features=horizon_features,
    )
    expected = fcst.horizon_features_

    fcst.history_warmup(
        df.copy(), static_features=[], horizon_features=horizon_features
    )
    assert fcst.horizon_features_ == expected
    assert fcst.ts.max_horizon == H


def test_history_warmup_preserved_horizon_features_missing_cols():
    """Re-warming a `horizon_features` instance without re-passing them
    preserves the mapping (`_fit` rebuilds `features_order_` from the new
    frame). If that frame drops the referenced exog columns the preserved
    mapping is stale, and warming must raise a clear error instead of letting
    a raw `KeyError` surface later inside `predict`."""
    df = _gen("pandas")[["unique_id", "ds", "y"]]
    df["exog_h1"] = df["y"] * 0.9
    df["exog_h2"] = df["y"] * 0.7
    horizon_features = {1: ["exog_h1"], 2: ["exog_h2"]}

    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    fcst.fit(
        df.copy(),
        static_features=[],
        max_horizon=H,
        horizon_features=horizon_features,
    )

    # re-warm with the exog columns dropped and without re-passing the mapping
    without_exog = df[["unique_id", "ds", "y"]].copy()
    with pytest.raises(ValueError, match="exog_h1.*exog_h2"):
        fcst.history_warmup(without_exog, static_features=[])

    # still fine when the columns are present (mapping stays valid)
    fcst.history_warmup(df.copy(), static_features=[])
    assert fcst.horizon_features_ == horizon_features


def test_history_warmup_calibration_state_defaults():
    """A fresh instance with externally supplied models has no calibration
    state until warmed. `predict(level=...)` should warn like an
    un-configured `fit` does (not raise `AttributeError`), and `save()`
    should omit the interval artifact instead of crashing. Re-warming an
    already-calibrated instance must leave its state untouched."""
    df = _gen("pandas")[["unique_id", "ds", "y"]]

    source = _new_fcst("local")
    source.fit(df.copy())

    warmed = _new_fcst("local")
    warmed.models_ = source.models_
    warmed.history_warmup(df.copy())

    with pytest.warns(UserWarning, match="prediction intervals"):
        warmed.predict(H, level=[90])

    with tempfile.TemporaryDirectory() as tmpdir:
        savedir = Path(tmpdir) / "fcst"
        savedir.mkdir()
        warmed.save(savedir)  # must not raise despite no calibration state

    calibrated = _new_fcst("local")
    calibrated.fit(df.copy(), prediction_intervals=PredictionIntervals(n_windows=2))
    cs_df_before = calibrated._cs_df
    calibrated.history_warmup(df.copy())
    assert calibrated._cs_df is cs_df_before


def test_history_warmup_as_numpy():
    """A fresh instance can opt into numpy model input via `as_numpy`; when
    omitted, re-warming an already-configured instance keeps its existing
    setting."""

    class RecordingModel(LinearRegression):
        last_X_type = None

        def predict(self, X):
            RecordingModel.last_X_type = type(X)
            return super().predict(np.asarray(X))

    df = _gen("pandas")[["unique_id", "ds", "y"]]

    source = MLForecast(models=[RecordingModel()], freq="D", lags=[1])
    source.fit(df.copy(), as_numpy=True)

    fresh = MLForecast(models=[RecordingModel()], freq="D", lags=[1])
    fresh.models_ = source.models_
    fresh.history_warmup(df.copy(), as_numpy=True)
    fresh.predict(H)
    assert RecordingModel.last_X_type is np.ndarray

    source.history_warmup(df.copy())
    assert source.ts.as_numpy is True


def test_history_warmup_validation():
    series = _gen("pandas")
    duplicated = pd.concat([series, series.head(1)])
    fcst = _new_fcst("local")
    with pytest.raises(ValueError):
        fcst.history_warmup(duplicated)

    pooled = _new_fcst("pooled")
    with pytest.warns(UserWarning, match="gap-free"):
        pooled.history_warmup(_gen("pandas", 1), validate_data=False)
