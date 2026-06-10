import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlforecast.core import TimeSeries
from mlforecast.lag_transforms import (
    ExpandingMax,
    ExpandingMean,
    ExpandingMin,
    ExpandingStd,
    ExponentiallyWeightedMean,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingStd,
)

_LAGS = [1, 3]


def _make_df(engine, rows, categorical_cols=None):
    if engine == "polars":
        df = pl.DataFrame(rows)
        for col in categorical_cols or []:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))
    else:
        df = pd.DataFrame(rows)
        for col in categorical_cols or []:
            df[col] = pd.Categorical(df[col])
    return df


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_new_series_new_group_update_then_predict(engine, lag):
    """Regression: new series in a new group must get correct bucket ID
    and produce valid predictions after update()."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "b", "b", "b"],
        "ds": [1, 2, 3, 1, 2, 3],
        "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        "brand": ["x", "x", "x", "y", "y", "y"],
    })
    tfm = RollingMean(2, groupby=["brand"])
    ts = TimeSeries(freq=1, lag_transforms={lag: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["brand"],
    )
    assert ts._pooled_groups[("brand",)] is not None
    state = ts._pooled_groups[("brand",)]
    assert len(np.unique(state.bucket_id)) == 2

    update_df = _make_df(engine, {
        "unique_id": ["a", "b", "c"],
        "ds": [4, 4, 4],
        "y": [40.0, 400.0, 1000.0],
        "brand": ["x", "y", "z"],
    })
    ts.update(update_df)

    state = ts._pooled_groups[("brand",)]
    assert state.series_bucket_id is not None
    assert len(state.series_bucket_id) == 3
    assert len(np.unique(state.series_bucket_id)) == 3
    assert len(state.bucket_df) == len(df) + len(update_df)
    n_buckets = len(np.unique(state.bucket_id))
    assert n_buckets == 3

    statics = ts.static_features_
    if engine == "pandas":
        uid_to_brand = dict(zip(statics["unique_id"], statics["brand"]))
    else:
        uid_to_brand = dict(zip(
            statics["unique_id"].to_list(), statics["brand"].to_list()
        ))
    assert uid_to_brand["a"] == "x"
    assert uid_to_brand["b"] == "y"
    assert uid_to_brand["c"] == "z"

    ts._predict_setup()
    features = ts._update_features()
    col = tfm._get_name(lag)
    # brand x: a=[10,20,30,40], brand y: b=[100,200,300,400]
    # At ds=5, rolling(2) over brand x looks at window [5-lag-1, 5-lag]
    #   lag=1: [3,4] → mean(30,40)=35; lag=3: [1,2] → mean(10,20)=15
    # brand z: c has only 1 obs → NaN
    expected_x = {1: 35.0, 3: 15.0}
    expected_y = {1: 350.0, 3: 150.0}
    expected = np.array([expected_x[lag], expected_y[lag], np.nan])
    np.testing.assert_allclose(features[col].to_numpy(), expected, equal_nan=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_global_update_preserves_bucket_df(engine, lag):
    """After update(), bucket_df should include both old and new observations."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "b", "b"],
        "ds": [1, 2, 1, 2],
        "y": [1.0, 2.0, 10.0, 20.0],
    })
    ts = TimeSeries(freq=1, lag_transforms={lag: [RollingMean(2, global_=True)]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False,
    )
    assert ts._pooled_global is not None
    orig_len = len(ts._pooled_global.bucket_df)

    update_df = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [3, 3],
        "y": [3.0, 30.0],
    })
    ts.update(update_df)
    new_len = len(ts._pooled_global.bucket_df)
    assert new_len == orig_len + 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_group_update_preserves_bucket_df(engine, lag):
    """After update(), group bucket_df should include new observations."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "b", "b"],
        "ds": [1, 2, 1, 2],
        "y": [1.0, 2.0, 10.0, 20.0],
        "brand": ["x", "x", "x", "x"],
    })
    tfm = RollingMean(2, groupby=["brand"])
    ts = TimeSeries(freq=1, lag_transforms={lag: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["brand"],
    )
    orig_len = len(ts._pooled_groups[("brand",)].bucket_df)

    update_df = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [3, 3],
        "y": [3.0, 30.0],
        "brand": ["x", "x"],
    })
    ts.update(update_df)
    new_len = len(ts._pooled_groups[("brand",)].bucket_df)
    assert new_len == orig_len + 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_global_sequential_updates(engine, lag):
    """Sequential update() calls correctly increment time_index."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "b", "b"],
        "ds": [1, 2, 1, 2],
        "y": [1.0, 2.0, 10.0, 20.0],
    })
    ts = TimeSeries(freq=1, lag_transforms={lag: [RollingMean(2, global_=True)]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False,
    )
    update1 = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [3, 3],
        "y": [3.0, 30.0],
    })
    ts.update(update1)
    state = ts._pooled_global
    assert state.next_time_index_by_bucket[0] == 3

    update2 = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [4, 4],
        "y": [4.0, 40.0],
    })
    ts.update(update2)
    state = ts._pooled_global
    assert state.next_time_index_by_bucket[0] == 4
    unique_idx = np.unique(state.time_index)
    assert len(unique_idx) == 4


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_staggered_series_start(engine, lag):
    """Series starting at different timestamps don't inject zeros."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "b", "b"],
        "ds": [1, 2, 3, 2, 3],
        "y": [1.0, 2.0, 3.0, 20.0, 30.0],
    })
    ts = TimeSeries(freq=1, lag_transforms={lag: [RollingMean(2, global_=True)]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False,
    )
    state = ts._pooled_global
    assert len(state.y) == 5
    assert 0.0 not in state.y


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_categorical_groupby_update_with_new_group(engine, lag):
    """Update with a new categorical group value works correctly."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": [1, 2, 1, 2],
            "y": [1.0, 2.0, 10.0, 20.0],
            "brand": ["b", "b", "b", "b"],
        },
        categorical_cols=["brand"],
    )
    tfm = RollingMean(2, groupby=["brand"])
    ts = TimeSeries(freq=1, lag_transforms={lag: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["brand"],
    )
    state = ts._pooled_groups[("brand",)]
    assert len(np.unique(state.bucket_id)) == 1

    update_df = _make_df(
        engine,
        {
            "unique_id": ["a", "b", "c"],
            "ds": [3, 3, 3],
            "y": [3.0, 30.0, 100.0],
            "brand": ["b", "b", "a"],
        },
        categorical_cols=["brand"],
    )
    ts.update(update_df)

    state = ts._pooled_groups[("brand",)]
    assert len(np.unique(state.bucket_id)) == 2
    assert len(state.series_bucket_id) == 3
    assert len(np.unique(state.series_bucket_id)) == 2
    assert len(state.bucket_df) == len(df) + len(update_df)

    ts._predict_setup()
    features = ts._update_features()
    col = tfm._get_name(lag)
    # brand b: [1,2,10,20,3,30] at ds 1,2,1,2,3,3
    # At ds=4, rolling(2) window is [4-lag-1, 4-lag]
    #   lag=1: [2,3] → mean(2,20,3,30)=13.75; lag=3: [0,1] → mean(1,10)=5.5
    # brand a: c has only 1 obs → NaN
    expected_val = {1: 13.75, 3: 5.5}
    expected = np.array([expected_val[lag], expected_val[lag], np.nan])
    np.testing.assert_allclose(features[col].to_numpy(), expected, equal_nan=True)


def test_compute_pooled_features_raises_for_unsupported():
    """Transforms returning None from _compute_bucket_feature raise NotImplementedError."""
    from mlforecast.pooled import PooledState, compute_pooled_features
    from mlforecast.lag_transforms import _BaseLagTransform
    from mlforecast.grouped_array import GroupedArray

    class DummyTransform(_BaseLagTransform):
        pass

    ga = GroupedArray(np.array([1.0, 2.0]), np.array([0, 2], dtype=np.int32))
    state = PooledState(
        ga=ga,
        bucket_df=pd.DataFrame({"uid": ["a", "a"], "ds": [1, 2], "_bucket_pos": [0, 1]}),
        groups=None,
        group_cols=None,
        series_bucket_id=np.array([0]),
        bucket_id=np.zeros(2, dtype=np.int64),
        time=np.array([1, 2]),
        time_index=np.array([0, 1], dtype=np.int64),
        y=np.array([1.0, 2.0]),
        next_time_index_by_bucket={0: 2},
        join_cols=["uid", "ds"],
    )
    with pytest.raises(NotImplementedError, match="does not support pooled"):
        compute_pooled_features(state, {"dummy": DummyTransform()})


class TestValidateDataWarning:
    """Warning when validate_data=False with pooled transforms."""

    def _make_fcst(self, transforms):
        from sklearn.linear_model import LinearRegression
        from mlforecast.forecast import MLForecast

        return MLForecast(
            models=[LinearRegression()],
            freq=1,
            lags=[1],
            lag_transforms=transforms,
        )

    def _make_simple_df(self):
        return pd.DataFrame(
            {
                "unique_id": ["a"] * 4 + ["b"] * 4,
                "ds": [1, 2, 3, 4] * 2,
                "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                "brand": ["x"] * 4 + ["x"] * 4,
            }
        )

    def test_warns_global(self):
        fcst = self._make_fcst({1: [RollingMean(window_size=2, global_=True)]})
        df = self._make_simple_df()
        with pytest.warns(UserWarning, match="Pooled.*validate_data"):
            fcst.preprocess(df, static_features=["brand"], validate_data=False)

    def test_warns_groupby(self):
        fcst = self._make_fcst(
            {1: [RollingMean(window_size=2, groupby=["brand"])]}
        )
        df = self._make_simple_df()
        with pytest.warns(UserWarning, match="Pooled.*validate_data"):
            fcst.preprocess(df, static_features=["brand"], validate_data=False)

    def test_no_warning_when_validated(self):
        fcst = self._make_fcst({1: [RollingMean(window_size=2, global_=True)]})
        df = self._make_simple_df()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fcst.preprocess(df, static_features=["brand"], validate_data=True)

    def test_no_warning_without_pooled(self):
        fcst = self._make_fcst({1: [RollingMean(window_size=2)]})
        df = self._make_simple_df()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fcst.preprocess(df, validate_data=False)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_ewm_lag_semantics(engine):
    """EWM with lag > 1 must only consume timestamps up to k-lag, not k-1.

    Regression test for the two-pointer fix in _ewm_from_agg.
    """
    df = _make_df(engine, {
        "unique_id": ["a"] * 4 + ["b"] * 4,
        "ds": list(range(4)) * 2,
        "y": [6.0, 7.0, 8.0, 9.0, 6.0, 7.0, 8.0, 9.0],
    })
    tfm = ExponentiallyWeightedMean(alpha=0.5, global_=True)
    ts = TimeSeries(freq=1, lag_transforms={2: [tfm]})
    result = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y", dropna=False,
    )
    col = tfm._get_name(2)
    if engine == "pandas":
        vals_a = result.loc[result["unique_id"] == "a", col].values
    else:
        vals_a = result.filter(pl.col("unique_id") == "a")[col].to_numpy()
    # Per-timestamp global means: t0=6, t1=7, t2=8, t3=9
    # lag=2, alpha=0.5:
    #   k=0: consume up to ts -2 → nothing → NaN
    #   k=1: consume up to ts -1 → nothing → NaN
    #   k=2: consume up to ts  0 → ewm=6.0
    #   k=3: consume up to ts  1 → ewm=0.5*7 + 0.5*6 = 6.5
    np.testing.assert_allclose(vals_a, [np.nan, np.nan, 6.0, 6.5], equal_nan=True)

    # Prediction: at t=4, consume up to ts 2 → ewm(t0,t1,t2) = 7.25
    ts._predict_setup()
    features = ts._update_features()
    pred_vals = features[col].to_numpy()
    np.testing.assert_allclose(pred_vals, [7.25, 7.25])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_ewm_lag_semantics_groupby(engine):
    """EWM lag semantics hold in groupby mode too."""
    df = _make_df(engine, {
        "unique_id": ["a"] * 4 + ["b"] * 4,
        "ds": list(range(4)) * 2,
        "y": [6.0, 7.0, 8.0, 9.0, 6.0, 7.0, 8.0, 9.0],
        "grp": ["X"] * 8,
    })
    tfm = ExponentiallyWeightedMean(alpha=0.5, groupby=["grp"])
    ts = TimeSeries(freq=1, lag_transforms={2: [tfm]})
    result = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["grp"],
    )
    col = tfm._get_name(2)
    if engine == "pandas":
        vals_a = result.loc[result["unique_id"] == "a", col].values
    else:
        vals_a = result.filter(pl.col("unique_id") == "a")[col].to_numpy()
    np.testing.assert_allclose(vals_a, [np.nan, np.nan, 6.0, 6.5], equal_nan=True)

    ts._predict_setup()
    features = ts._update_features()
    np.testing.assert_allclose(features[col].to_numpy(), [7.25, 7.25])


def _fit_and_collect(engine, lag, tfms, y_a, y_b, n_times, grp=None):
    """Helper: fit global or groupby transforms, return per-series-a preprocess
    values and prediction values."""
    data = {
        "unique_id": ["a"] * n_times + ["b"] * n_times,
        "ds": list(range(n_times)) * 2,
        "y": y_a + y_b,
    }
    if grp is not None:
        data["grp"] = [grp] * (2 * n_times)
    df = _make_df(engine, data)

    ts = TimeSeries(freq=1, lag_transforms={lag: tfms})
    static = ["grp"] if grp else None
    result = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=static,
    )
    ts._predict_setup()
    features = ts._update_features()
    out = {}
    for tfm in tfms:
        col = tfm._get_name(lag)
        if engine == "pandas":
            vals_a = result.loc[result["unique_id"] == "a", col].values
        else:
            vals_a = result.filter(pl.col("unique_id") == "a")[col].to_numpy()
        pred = features[col].to_numpy()
        out[col] = {"preprocess": vals_a, "predict": pred}
    return out


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_pooled_transforms_lag3_global(engine):
    """All decomposable transforms produce correct values with lag=3 in global mode."""
    y_a = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    y_b = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    # Per-ts global means: t0=1.5, t1=3.5, t2=5.5, t3=7.5, t4=9.5, t5=11.5
    lag = 3
    tfms = [
        RollingMean(window_size=3, global_=True),
        RollingStd(window_size=3, global_=True),
        RollingMin(window_size=3, global_=True),
        RollingMax(window_size=3, global_=True),
        ExpandingMean(global_=True),
        ExpandingStd(global_=True),
        ExpandingMin(global_=True),
        ExpandingMax(global_=True),
    ]
    out = _fit_and_collect(engine, lag, tfms, y_a, y_b, 6)

    nan = np.nan
    # Preprocess (series a, 6 timestamps) — lag=3 means feature at k uses obs up to k-3
    # Rolling window=3: uses obs in [k-3-2, k-3] = [k-5, k-3]
    #   k=0..3: not enough → NaN; k=4: t0,t1 → count=4≥3 → mean=2.5; k=5: t0,t1,t2 → count=6≥3 → mean=3.5
    expected = {
        "global_rolling_mean_lag3_window_size3":     ([nan, nan, nan, nan, 2.5, 3.5],       [5.5, 5.5]),
        "global_rolling_std_lag3_window_size3":       ([nan, nan, nan, nan, 1.290994, 1.870829], [1.870829, 1.870829]),
        "global_rolling_min_lag3_window_size3":       ([nan, nan, nan, nan, 1.0, 1.0],       [3.0, 3.0]),
        "global_rolling_max_lag3_window_size3":       ([nan, nan, nan, nan, 4.0, 6.0],       [8.0, 8.0]),
        "global_expanding_mean_lag3":                 ([nan, nan, nan, 1.5, 2.5, 3.5],       [4.5, 4.5]),
        "global_expanding_std_lag3":                  ([nan, nan, nan, 0.707107, 1.290994, 1.870829], [2.449490, 2.449490]),
        "global_expanding_min_lag3":                  ([nan, nan, nan, 1.0, 1.0, 1.0],       [1.0, 1.0]),
        "global_expanding_max_lag3":                  ([nan, nan, nan, 2.0, 4.0, 6.0],       [8.0, 8.0]),
    }
    for col, (exp_pre, exp_pred) in expected.items():
        np.testing.assert_allclose(
            out[col]["preprocess"], exp_pre, atol=1e-5, equal_nan=True,
            err_msg=f"preprocess mismatch for {col}",
        )
        np.testing.assert_allclose(
            out[col]["predict"], exp_pred, atol=1e-5,
            err_msg=f"predict mismatch for {col}",
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_pooled_transforms_lag2_groupby(engine):
    """Decomposable transforms produce correct values with lag=2 in groupby mode."""
    # Group X: series a=[1,2,3,4,5], b=[10,20,30,40,50]
    # Per-ts global means in group X: t0=5.5, t1=11, t2=16.5, t3=22, t4=27.5
    y_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_b = [10.0, 20.0, 30.0, 40.0, 50.0]
    lag = 2
    tfms = [
        RollingMean(window_size=3, groupby=["grp"]),
        RollingMin(window_size=3, groupby=["grp"]),
        RollingMax(window_size=3, groupby=["grp"]),
        ExpandingMean(groupby=["grp"]),
        ExpandingMin(groupby=["grp"]),
        ExpandingMax(groupby=["grp"]),
        ExponentiallyWeightedMean(alpha=0.5, groupby=["grp"]),
    ]
    out = _fit_and_collect(engine, lag, tfms, y_a, y_b, 5, grp="X")

    nan = np.nan
    expected = {
        "groupby_grp_rolling_mean_lag2_window_size3":  [nan, nan, nan, 8.25, 11.0],
        "groupby_grp_rolling_min_lag2_window_size3":   [nan, nan, nan, 1.0, 1.0],
        "groupby_grp_rolling_max_lag2_window_size3":   [nan, nan, nan, 20.0, 30.0],
        "groupby_grp_expanding_mean_lag2":             [nan, nan, 5.5, 8.25, 11.0],
        "groupby_grp_expanding_min_lag2":              [nan, nan, 1.0, 1.0, 1.0],
        "groupby_grp_expanding_max_lag2":              [nan, nan, 10.0, 20.0, 30.0],
        "groupby_grp_exponentially_weighted_mean_lag2_alpha0.5": [nan, nan, 5.5, 8.25, 12.375],
    }
    for col, exp_pre in expected.items():
        np.testing.assert_allclose(
            out[col]["preprocess"], exp_pre, atol=1e-5, equal_nan=True,
            err_msg=f"preprocess mismatch for {col}",
        )


@pytest.mark.parametrize("tfm_factory", [
    lambda m: RollingMean(window_size=4, **m),
    lambda m: RollingStd(window_size=4, **m),
    lambda m: RollingMin(window_size=4, **m),
    lambda m: RollingMax(window_size=4, **m),
    lambda m: ExpandingMean(**m),
    lambda m: ExpandingStd(**m),
    lambda m: ExpandingMin(**m),
    lambda m: ExpandingMax(**m),
    lambda m: ExponentiallyWeightedMean(alpha=0.3, **m),
], ids=[
    "RollingMean", "RollingStd", "RollingMin", "RollingMax",
    "ExpandingMean", "ExpandingStd", "ExpandingMin", "ExpandingMax", "EWM",
])
@pytest.mark.parametrize("lag", _LAGS)
def test_fast_vs_slow_equivalence(tfm_factory, lag):
    """Fast path (aggregate-based) matches slow path (row-level) at every lag.

    Exercises all three code paths: fit (_compute_from_aggregates via
    compute_pooled_features), preprocess (_compute_ts_level_from_aggs),
    and predict (_compute_latest_from_aggs).
    """
    from mlforecast.pooled import compute_pooled_features

    rng = np.random.default_rng(42)
    n_series, n_times = 8, 12
    ids = np.repeat([f"s{i}" for i in range(n_series)], n_times)
    times = np.tile(range(n_times), n_series)
    y = rng.standard_normal(n_series * n_times)
    grps = np.repeat(["A"] * (n_series // 2) + ["B"] * (n_series // 2), n_times)
    df = pd.DataFrame({"unique_id": ids, "ds": times, "y": y, "grp": grps})

    # --- fit path: fast vs slow via compute_pooled_features ---
    tfm_g = tfm_factory({"global_": True})
    ts = TimeSeries(freq=1, lag_transforms={lag: [tfm_g]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["grp"],
    )
    state = ts._pooled_global
    col = tfm_g._get_name(lag)
    fitted_tfm = ts.transforms[col]

    fast = compute_pooled_features(state, {col: fitted_tfm})
    saved_aggs = state._ts_aggs
    state._ts_aggs = {}
    slow = compute_pooled_features(state, {col: fitted_tfm})
    state._ts_aggs = saved_aggs

    np.testing.assert_allclose(
        fast[col], slow[col], atol=1e-10, equal_nan=True,
        err_msg=f"fit fast vs slow mismatch for {col}",
    )

    # --- preprocess path: global _compute_ts_level_from_aggs ---
    result_fast = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["grp"],
    )
    fast_pre = result_fast[col].values

    ts_slow = TimeSeries(
        freq=1,
        lag_transforms={lag: [tfm_factory({"global_": True})]},
    )
    ts_slow._fit(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=["grp"],
    )
    ts_slow._pooled_global._ts_aggs = {}
    result_slow = ts_slow._transform(df=df, dropna=False)
    slow_pre = result_slow[col].values
    np.testing.assert_allclose(
        fast_pre, slow_pre, atol=1e-10, equal_nan=True,
        err_msg=f"preprocess global fast vs slow for {col}",
    )

    # --- preprocess path: groupby _compute_ts_level_from_aggs ---
    tfm_grp = tfm_factory({"groupby": ["grp"]})
    ts_grp = TimeSeries(freq=1, lag_transforms={lag: [tfm_grp]})
    result_grp = ts_grp.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["grp"],
    )
    col_grp = tfm_grp._get_name(lag)
    fast_grp_pre = result_grp[col_grp].values

    ts_grp_slow = TimeSeries(
        freq=1,
        lag_transforms={lag: [tfm_factory({"groupby": ["grp"]})]},
    )
    ts_grp_slow._fit(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=["grp"],
    )
    for st in ts_grp_slow._pooled_groups.values():
        st._ts_aggs = {}
        st._idsorted_to_bucket_pos = None
    result_grp_slow = ts_grp_slow._transform(df=df, dropna=False)
    slow_grp_pre = result_grp_slow[col_grp].values
    np.testing.assert_allclose(
        fast_grp_pre, slow_grp_pre, atol=1e-10, equal_nan=True,
        err_msg=f"preprocess groupby fast vs slow for {col_grp}",
    )

    # --- predict path: _compute_latest_from_aggs ---
    ts2 = TimeSeries(freq=1, lag_transforms={lag: [tfm_factory({"global_": True})]})
    ts2.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["grp"],
    )
    ts2._predict_setup()
    features = ts2._update_features()
    pred_col = list(ts2.transforms.keys())[0]
    pred_vals = features[pred_col].to_numpy()
    assert not np.all(np.isnan(pred_vals)), f"predict returned all NaN for {pred_col}"


# ---------------------------------------------------------------------------
# Null/NaN groupby key support.
#
# A missing (null/NaN/None) value in a groupby key column must collapse all
# missing values into a single bucket (SQL PARTITION BY semantics), identically
# across pandas/polars and key dtypes, without crashing fit/predict/update.
# ---------------------------------------------------------------------------
from mlforecast.pooled import (  # noqa: E402
    add_bucket_id,
    lookup_bucket_ids,
    _attach_bucket_id,
    _extend_groups,
)


def _bids(df):
    return np.asarray(df["_bucket_id"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("key_kind", ["numeric", "string", "categorical"])
def test_add_bucket_id_collapses_missing(engine, key_kind):
    if key_kind == "numeric":
        vals = [0.0, None, 0.0, 1.0, None]
    else:
        vals = ["a", None, "a", "b", None]
    cat = ["g"] if key_kind == "categorical" else None
    df = _make_df(engine, {"g": vals, "y": [1, 2, 3, 4, 5]}, categorical_cols=cat)
    merged, groups = add_bucket_id(df, ["g"])
    bids = _bids(merged)
    # rows 1 & 4 (missing) share a bucket; rows 0 & 2 (repeated value) share one;
    # row 3 (distinct value) is its own.
    assert bids[1] == bids[4]
    assert bids[0] == bids[2]
    assert len({int(bids[0]), int(bids[1]), int(bids[3])}) == 3
    assert len(groups) == 3
    assert np.all(bids >= 0)  # no -9.2e18 / NaN garbage


def test_polars_null_and_nan_collapse_to_one_bucket():
    # Engine-origin behavior the SQLite oracle can't reach: a polars float column
    # holding BOTH null and NaN must collapse them into a single missing bucket.
    df = pl.DataFrame({
        "g": [0.0, float("nan"), None, 1.0, float("nan"), None],
        "y": list(range(6)),
    })
    merged, groups = add_bucket_id(df, ["g"])
    bids = _bids(merged)
    assert len({int(bids[1]), int(bids[2]), int(bids[4]), int(bids[5])}) == 1
    assert bids[0] != bids[1] and bids[3] != bids[1]
    assert len(groups) == 3  # 0.0, missing, 1.0


def test_null_nan_parity_across_engines():
    # pandas NaN and polars null/NaN must produce the same bucket structure.
    pdf = pd.DataFrame({"g": [0.0, np.nan, 0.0, 1.0], "y": [1, 2, 3, 4]})
    plf = pl.DataFrame({"g": [0.0, float("nan"), 0.0, 1.0], "y": [1, 2, 3, 4]})
    _, gp = add_bucket_id(pdf, ["g"])
    _, gl = add_bucket_id(plf, ["g"])
    assert len(gp) == len(gl) == 3


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_lookup_mixed_int_float(engine):
    # Fit data contaminated to float by a NaN; later clean integer keys must
    # still match the same buckets (no string "0" vs "0.0" mismatch), and the
    # int<->float reconcile must not raise on the missing value.
    fit = _make_df(engine, {"g": [0.0, 1.0, float("nan")], "y": [1, 2, 3]})
    _, groups = add_bucket_id(fit, ["g"])
    if engine == "polars":
        data = pl.DataFrame({"g": pl.Series([0, 1], dtype=pl.Int64)})
        nan_data = pl.DataFrame({"g": pl.Series([float("nan")], dtype=pl.Float64)})
    else:
        data = pd.DataFrame({"g": pd.Series([0, 1], dtype="int64")})
        nan_data = pd.DataFrame({"g": pd.Series([np.nan], dtype="float64")})
    bids = lookup_bucket_ids(data, groups, ["g"])
    assert bids[0] == 0 and bids[1] == 1
    assert np.all(bids >= 0)
    # a missing key in lookup data finds the existing missing bucket
    nan_bid = lookup_bucket_ids(nan_data, groups, ["g"])
    assert nan_bid[0] == 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fractional_float_does_not_collide_with_int_bucket(engine):
    # The int<->float reconcile must only int-encode *integral* floats: a
    # genuinely fractional key 1.5 must NOT match an existing integer bucket 1.
    if engine == "polars":
        fit = pl.DataFrame({"g": pl.Series([1, 2], dtype=pl.Int64), "y": [1, 2]})
        q = pl.DataFrame({"g": pl.Series([1.5, 1.0, 2.0], dtype=pl.Float64)})
    else:
        fit = pd.DataFrame({"g": pd.Series([1, 2], dtype="int64"), "y": [1, 2]})
        q = pd.DataFrame({"g": pd.Series([1.5, 1.0, 2.0], dtype="float64")})
    _, groups = add_bucket_id(fit, ["g"])
    res = lookup_bucket_ids(q, groups, ["g"])
    assert np.isnan(res[0])      # 1.5 unmatched, not bucket 1
    assert res[1] == 0           # 1.0 matches integer bucket for 1
    assert res[2] == 1           # 2.0 matches integer bucket for 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_large_int_keys_stay_distinct(engine):
    # The string encoding must never route int keys through float: two distinct
    # integers above 2**53 (not exactly representable as float64) must stay in
    # distinct buckets through both creation and lookup. A large-int value cannot
    # be carried on a float-typed column without precision loss, so this pins the
    # int-side encoding; the mixed int/float reconcile branch itself is covered by
    # test_lookup_mixed_int_float and test_fractional_float_does_not_collide_with_int_bucket.
    a, b = 2 ** 53 + 1, 2 ** 53 + 2
    if engine == "polars":
        df = pl.DataFrame({"g": pl.Series([a, b, a], dtype=pl.Int64), "y": [1, 2, 3]})
    else:
        df = pd.DataFrame({"g": pd.Series([a, b, a], dtype="int64"), "y": [1, 2, 3]})
    merged, groups = add_bucket_id(df, ["g"])
    bids = _bids(merged)
    assert bids[0] == bids[2] and bids[0] != bids[1]
    assert len(groups) == 2
    look = lookup_bucket_ids(df, groups, ["g"])
    assert look[0] != look[1]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_mixed_schema_reconcile_does_not_widen_int_side(engine):
    # Drives the int<->float reconcile branch (float lookup vs integer groups)
    # and asserts the integer side is never widened to Float64: an integral float
    # lookup matches the integer bucket, while distinct large integer buckets
    # built from integer-dtype groups remain distinct (no precision collapse).
    a, b = 2 ** 53 + 1, 2 ** 53 + 2
    if engine == "polars":
        groups_src = pl.DataFrame({"g": pl.Series([a, b, 1], dtype=pl.Int64), "y": [1, 2, 3]})
        q = pl.DataFrame({"g": pl.Series([1.0], dtype=pl.Float64)})
    else:
        groups_src = pd.DataFrame({"g": pd.Series([a, b, 1], dtype="int64"), "y": [1, 2, 3]})
        q = pd.DataFrame({"g": pd.Series([1.0], dtype="float64")})
    _, groups = add_bucket_id(groups_src, ["g"])
    assert len(groups) == 3  # large ints not collapsed by any float widening
    # float 1.0 reconciles to the integer-1 bucket via the to_int path.
    look = lookup_bucket_ids(q, groups, ["g"])
    assert look[0] == 2 and not np.isnan(look[0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_extend_does_not_duplicate_existing_missing_bucket(engine):
    fit = _make_df(engine, {"g": ["x", None, "y"], "y": [1, 2, 3]})
    _, groups = add_bucket_id(fit, ["g"])
    n0 = len(groups)
    upd = _make_df(engine, {"g": [None, "z"], "y": [9, 10]})
    att = _attach_bucket_id(upd, groups, ["g"])
    ext, groups2 = _extend_groups(att, groups, ["g"])
    bids = _bids(ext)
    assert np.all(bids >= 0)
    # the pre-existing missing bucket is reused; only the new value "z" is added.
    assert len(groups2) == n0 + 1


def test_one_missing_column_keeps_distinct_buckets():
    # Multi-column key: (X, None) and (X, "n") must be distinct buckets, while
    # two (X, None) rows collapse together.
    df = pd.DataFrame({
        "b": ["X", "X", "X", "Y"],
        "r": ["n", None, None, "s"],
        "y": [1, 2, 3, 4],
    })
    merged, groups = add_bucket_id(df, ["b", "r"])
    bids = _bids(merged)
    assert bids[1] == bids[2]  # both (X, None)
    assert bids[0] != bids[1]  # (X, "n") distinct from (X, None)
    assert len(groups) == 3


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_groupby_null_key_fit_predict_update(engine):
    from sklearn.linear_model import LinearRegression
    from mlforecast.forecast import MLForecast

    df = _make_df(engine, {
        "unique_id": ["a"] * 6 + ["b"] * 6 + ["c"] * 6,
        "ds": list(range(6)) * 3,
        "y": [float(i) for i in range(18)],
        "brand": ["x"] * 6 + [None] * 6 + [None] * 6,
    })
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        lags=[1],
        lag_transforms={1: [RollingMean(window_size=2, groupby=["brand"])]},
    )
    fcst.fit(df, static_features=["brand"])
    preds = fcst.predict(2)
    pvals = np.asarray(preds["LinearRegression"])
    assert np.all(np.isfinite(pvals))  # no IndexError / garbage bucket
    # update including the null-brand series must not crash
    upd = _make_df(engine, {
        "unique_id": ["a", "b", "c"],
        "ds": [6, 6, 6],
        "y": [6.0, 7.0, 8.0],
        "brand": ["x", None, None],
    })
    fcst.update(upd)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["string-null", "numeric-nan"])
def test_null_groupby_key_no_static_change_error(engine, kind):
    # A series whose entire (static) groupby key is missing must NOT raise
    # "values change over time" (NaN != NaN). Covers pandas object-None,
    # pandas float-NaN, polars null, and polars float-NaN.
    if kind == "string-null":
        brand = ["x", "x", "x", None, None, None]
    else:
        brand = [1.0, 1.0, 1.0, float("nan"), float("nan"), float("nan")]
    df = _make_df(engine, {
        "unique_id": ["a"] * 3 + ["b"] * 3,
        "ds": [0, 1, 2] * 2,
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "brand": brand,
    })
    ts = TimeSeries(
        freq=1, lag_transforms={1: [RollingMean(window_size=2, groupby=["brand"])]}
    )
    # must not raise
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["brand"],
    )
