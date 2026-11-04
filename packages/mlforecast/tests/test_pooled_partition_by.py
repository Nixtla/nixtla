import operator
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast
from mlforecast.core import TimeSeries
from mlforecast.lag_transforms import (
    Combine,
    ExpandingMean,
    ExponentiallyWeightedMean,
    LookupLag,
    Offset,
    RollingMean,
)


def _panel(engine="pandas", n_series=4, n_times=12, seed=0):
    rng = np.random.default_rng(seed)
    data = {
        "unique_id": np.repeat([f"s{i}" for i in range(n_series)], n_times),
        "ds": np.tile(np.arange(n_times), n_series),
        "y": rng.normal(10, 1, n_series * n_times),
        "brand": np.repeat(np.arange(n_series) % 2, n_times),
        "promo": rng.integers(0, 2, n_series * n_times),
    }
    if engine == "polars":
        return pl.DataFrame(data)
    return pd.DataFrame(data)


def _future(df, h, seed=1):
    rng = np.random.default_rng(seed)
    ids = np.repeat(sorted(df["unique_id"].unique()), h)
    last = df["ds"].max()
    return pd.DataFrame(
        {
            "unique_id": ids,
            "ds": np.tile(np.arange(last + 1, last + 1 + h), len(ids) // h),
            "promo": rng.integers(0, 2, len(ids)),
        }
    )


def _fit_ts(df, tfms, **kwargs):
    ts = TimeSeries(freq=1, lag_transforms={1: tfms})
    res = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y", dropna=False, **kwargs
    )
    return ts, res


@pytest.mark.parametrize("kwarg", ["groupby", "global"])
def test_removed_scope_kwargs_raise(kwarg):
    with pytest.raises(TypeError, match=kwarg):
        RollingMean(3, **{kwarg: True})


def test_global_and_partition_by_exclusive():
    with pytest.raises(ValueError, match="can't be used together"):
        RollingMean(3, global_=True, partition_by=["brand"])


@pytest.mark.parametrize(
    "make_tfm",
    [
        lambda partition_by: RollingMean(3, partition_by=partition_by),
        lambda partition_by: ExpandingMean(partition_by=partition_by),
        lambda partition_by: ExponentiallyWeightedMean(0.5, partition_by=partition_by),
    ],
)
@pytest.mark.parametrize("partition_by", [[], ()])
def test_empty_partition_by_raises(make_tfm, partition_by):
    with pytest.raises(ValueError, match="use `global_=True`"):
        make_tfm(partition_by)


def test_time_agg_requires_pooled_scope():
    with pytest.raises(ValueError, match="pooled scope"):
        RollingMean(3, time_agg="sum")
    # EWM's default time_agg is allowed per series
    ExponentiallyWeightedMean(0.5)


@pytest.mark.parametrize(
    "tfm, name",
    [
        (RollingMean(3, global_=True), "global_rolling_mean_lag1_window_size3"),
        (
            RollingMean(3, partition_by=["brand"]),
            "partby_brand_rolling_mean_lag1_window_size3",
        ),
        (
            RollingMean(3, partition_by=["unique_id", "promo"]),
            "partby_unique_id__promo_rolling_mean_lag1_window_size3",
        ),
        (
            RollingMean(3, partition_by=["promo", "unique_id"]),
            "partby_promo__unique_id_rolling_mean_lag1_window_size3",
        ),
        (
            LookupLag(partition_by=["unique_id", "promo"]),
            "partby_unique_id__promo_lookup_lag1",
        ),
    ],
)
def test_feature_names(tfm, name):
    assert tfm._get_name(1) == name


@pytest.mark.parametrize(
    "tfm, match",
    [
        (
            RollingMean(3, partition_by=["unique_id"]),
            "with min_samples=1 and without partition_by instead",
        ),
        (
            RollingMean(3, partition_by=["unique_id", "brand"]),
            "with min_samples=1 and without partition_by instead",
        ),
        (
            RollingMean(3, min_samples=2, partition_by=["unique_id", "brand"]),
            "use the transform without partition_by instead",
        ),
        (
            ExpandingMean(partition_by=["unique_id", "brand"]),
            "use the transform without partition_by instead",
        ),
        (LookupLag(partition_by=["promo"]), "LookupLag's partition_by must include"),
        (LookupLag(partition_by=["unique_id"]), "use `Lag` instead"),
        (
            RollingMean(3, partition_by=["unique_id", "promo"], time_agg="sum"),
            "time_agg requires a partition_by without",
        ),
        (
            Offset(RollingMean(3, partition_by=["unique_id"]), 1),
            "only static",
        ),
    ],
)
def test_fit_rejects_keys(tfm, match):
    with pytest.raises(ValueError, match=match):
        _fit_ts(_panel(), [tfm], static_features=["brand"])


def test_id_and_dynamic_key_allowed():
    # brand isn't listed, so the key is dynamic and re-bucketed every step
    _fit_ts(_panel(), [RollingMean(3, partition_by=["unique_id", "brand"])])
    _fit_ts(
        _panel(),
        [ExponentiallyWeightedMean(0.5, partition_by=["unique_id", "promo"])],
    )


def test_combine_needs_same_key():
    pooled = RollingMean(3, partition_by=["brand"])
    with pytest.raises(ValueError, match="different global_/partition_by"):
        Combine(pooled, RollingMean(3), operator.truediv)
    with pytest.raises(ValueError, match="different global_/partition_by"):
        Combine(pooled, RollingMean(3, global_=True), operator.truediv)
    # same key in a different order
    Combine(
        RollingMean(3, partition_by=["brand", "promo"]),
        RollingMean(5, partition_by=["promo", "brand"]),
        operator.truediv,
    )


def test_key_order_shares_state():
    df = _panel()
    with pytest.warns(UserWarning, match="only differ in column order") as record:
        ts, res = _fit_ts(
            df,
            [
                RollingMean(3, partition_by=["promo", "unique_id"]),
                RollingMean(3, partition_by=["unique_id", "promo"]),
                LookupLag(partition_by=["unique_id", "promo"]),
            ],
        )
    assert sum("column order" in str(w.message) for w in record) == 1
    assert list(ts._pooled_states) == [("promo", "unique_id")]
    np.testing.assert_allclose(
        res["partby_promo__unique_id_rolling_mean_lag1_window_size3"],
        res["partby_unique_id__promo_rolling_mean_lag1_window_size3"],
    )


def test_min_samples_default_with_id_in_key():
    ts, _ = _fit_ts(
        _panel(),
        [
            RollingMean(3, partition_by=["unique_id", "promo"]),
            RollingMean(3, partition_by=["promo"]),
        ],
    )
    kernels = {
        key: leaves[0]._pooled_kernel for key, leaves in ts._get_pooled_tfms().items()
    }
    assert kernels[("promo", "unique_id")].min_samples() == 1
    assert kernels[("promo",)].min_samples() == 3


@pytest.mark.parametrize("static", [True, False])
def test_ewm_warning_only_with_dynamic_key(static):
    kwargs = {"static_features": ["brand"]} if static else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tfm = ExponentiallyWeightedMean(0.5, partition_by=["brand"])
        _fit_ts(_panel(), [tfm], **kwargs)
    msgs = [str(w.message) for w in caught if "Partitioned EWM" in str(w.message)]
    assert bool(msgs) is not static


def test_dynamic_key_requires_x_df_and_updates():
    df = _panel()
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        lags=[1],
        lag_transforms={1: [ExpandingMean(partition_by=["promo"])]},
    )
    fcst.fit(df.drop(columns="brand"))
    with pytest.raises(ValueError, match="X_df is required.*promo"):
        fcst.predict(2)
    new = _future(df, 2).assign(y=1.0)
    with pytest.raises(ValueError, match=r"\['promo'\] must be provided"):
        fcst.update(new.drop(columns="promo"))
    fcst.update(new)
    preds = fcst.predict(2, X_df=_future(pd.concat([df, new]), 2, seed=2))
    assert preds.shape[0] == 8


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_predict_ids_with_id_in_key(engine):
    df = _panel(engine)
    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=5)],
        freq=1,
        lags=[1],
        lag_transforms={
            1: [
                RollingMean(3, min_samples=1, partition_by=["unique_id", "promo"]),
                LookupLag(partition_by=["unique_id", "promo"]),
            ]
        },
    )
    fcst.fit(df, static_features=["brand"], dropna=False)
    X_df = _future(df.to_pandas() if engine == "polars" else df, 3)
    if engine == "polars":
        X_df = pl.from_pandas(X_df)
    full = fcst.predict(3, X_df=X_df)
    sub = fcst.predict(3, X_df=X_df, ids=["s1", "s3"])
    if engine == "polars":
        full, sub = full.to_pandas(), sub.to_pandas()
    expected = full[full["unique_id"].isin(["s1", "s3"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(sub, expected)


def test_predict_ids_rejected_when_pooling_across_series():
    df = _panel()
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        lags=[1],
        lag_transforms={1: [RollingMean(3, partition_by=["brand"])]},
    )
    fcst.fit(df.drop(columns="promo"), static_features=["brand"])
    with pytest.raises(ValueError, match="Cannot use `ids`"):
        fcst.predict(2, ids=["s1"])
