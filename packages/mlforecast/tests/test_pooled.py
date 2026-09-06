import warnings

import copy
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
    LookupLag,
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


def _next_ordinal_by_bucket(state):
    """Next calendar position, per bucket.

    `main` carried a separate calendar per bucket; the channel engine keeps one
    dense calendar shared by every bucket, so they advance in lockstep by
    construction. Expressed in the old vocabulary so these assertions keep their
    meaning.
    """
    return {b: state.n_ordinals for b in range(state.n_buckets)}


def _set_bucket_ids(state, context_df, id_col="unique_id"):
    """Re-bucket every series from a per-series context frame."""
    from mlforecast.pooled import lookup

    arrays = []
    if state.mode == "local":
        arrays.append(np.asarray(context_df[id_col].to_numpy()))
    arrays += [
        np.asarray(context_df[c].to_numpy())
        for c in state.group_cols + state.partition_cols
    ]
    state.set_series_bucket_id(lookup(arrays, state.bucket_uniques))
    return state.series_bucket_id


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_new_series_new_group_update_then_predict(engine, lag):
    """Regression: new series in a new group must get correct bucket ID
    and produce valid predictions after update()."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "a", "b", "b", "b"],
            "ds": [1, 2, 3, 1, 2, 3],
            "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "brand": ["x", "x", "x", "y", "y", "y"],
        },
    )
    tfm = RollingMean(2, groupby=["brand"])
    ts = TimeSeries(freq=1, lag_transforms={lag: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=["brand"],
        keep_last_n=10_000,
    )
    assert ts._pooled_states[("groupby", ("brand",), ())] is not None
    state = ts._pooled_states[("groupby", ("brand",), ())]
    assert state.n_buckets == 2

    update_df = _make_df(
        engine,
        {
            "unique_id": ["a", "b", "c"],
            "ds": [4, 4, 4],
            "y": [40.0, 400.0, 1000.0],
            "brand": ["x", "y", "z"],
        },
    )
    ts.update(update_df)

    state = ts._pooled_states[("groupby", ("brand",), ())]
    assert state.series_bucket_id is not None
    assert len(state.series_bucket_id) == 3
    assert len(np.unique(state.series_bucket_id)) == 3
    assert state.base["count"].sum() == len(df) + len(update_df)
    n_buckets = state.n_buckets
    assert n_buckets == 3

    statics = ts.static_features_
    if engine == "pandas":
        uid_to_brand = dict(zip(statics["unique_id"], statics["brand"]))
    else:
        uid_to_brand = dict(
            zip(statics["unique_id"].to_list(), statics["brand"].to_list())
        )
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
    """After update(), the cell store must hold both old and new observations."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": [1, 2, 1, 2],
            "y": [1.0, 2.0, 10.0, 20.0],
        },
    )
    ts = TimeSeries(freq=1, lag_transforms={lag: [RollingMean(2, global_=True)]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
    )
    assert ("global", (), ()) in ts._pooled_states
    orig_len = ts._pooled_states[("global", (), ())].base["count"].sum()

    update_df = _make_df(
        engine,
        {
            "unique_id": ["a", "b"],
            "ds": [3, 3],
            "y": [3.0, 30.0],
        },
    )
    ts.update(update_df)
    new_len = ts._pooled_states[("global", (), ())].base["count"].sum()
    assert new_len == orig_len + 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_group_update_preserves_bucket_df(engine, lag):
    """After update(), the group cell store must hold the new observations."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": [1, 2, 1, 2],
            "y": [1.0, 2.0, 10.0, 20.0],
            "brand": ["x", "x", "x", "x"],
        },
    )
    tfm = RollingMean(2, groupby=["brand"])
    ts = TimeSeries(freq=1, lag_transforms={lag: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=["brand"],
    )
    orig_len = ts._pooled_states[("groupby", ("brand",), ())].base["count"].sum()

    update_df = _make_df(
        engine,
        {
            "unique_id": ["a", "b"],
            "ds": [3, 3],
            "y": [3.0, 30.0],
            "brand": ["x", "x"],
        },
    )
    ts.update(update_df)
    new_len = ts._pooled_states[("groupby", ("brand",), ())].base["count"].sum()
    assert new_len == orig_len + 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_global_sequential_updates(engine, lag):
    """Sequential update() calls correctly increment time_index."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": [1, 2, 1, 2],
            "y": [1.0, 2.0, 10.0, 20.0],
        },
    )
    ts = TimeSeries(freq=1, lag_transforms={lag: [RollingMean(2, global_=True)]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
    )
    update1 = _make_df(
        engine,
        {
            "unique_id": ["a", "b"],
            "ds": [3, 3],
            "y": [3.0, 30.0],
        },
    )
    ts.update(update1)
    state = ts._pooled_states[("global", (), ())]
    assert _next_ordinal_by_bucket(state)[0] == 3

    update2 = _make_df(
        engine,
        {
            "unique_id": ["a", "b"],
            "ds": [4, 4],
            "y": [4.0, 40.0],
        },
    )
    ts.update(update2)
    state = ts._pooled_states[("global", (), ())]
    assert _next_ordinal_by_bucket(state)[0] == 4
    assert state.base["count"].shape[1] == 4  # calendar spans four timestamps


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_staggered_series_start(engine, lag):
    """Series starting at different timestamps don't inject zeros."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "a", "b", "b"],
            "ds": [1, 2, 3, 2, 3],
            "y": [1.0, 2.0, 3.0, 20.0, 30.0],
        },
    )
    ts = TimeSeries(freq=1, lag_transforms={lag: [RollingMean(2, global_=True)]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        keep_last_n=10_000,  # full-history check: disable pooled trim
        dropna=False,
    )
    state = ts._pooled_states[("global", (), ())]
    # no phantom zeros: ds=1 has only series "a", ds=2 and ds=3 have both
    np.testing.assert_array_equal(state.base["count"][0], [1, 2, 2])
    assert state.base["count"].sum() == 5


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
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=["brand"],
    )
    state = ts._pooled_states[("groupby", ("brand",), ())]
    assert state.n_buckets == 1

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

    state = ts._pooled_states[("groupby", ("brand",), ())]
    assert state.n_buckets == 2
    assert len(state.series_bucket_id) == 3
    assert len(np.unique(state.series_bucket_id)) == 2
    assert state.base["count"].sum() == len(df) + len(update_df)

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
    """A transform with no pooled kernel must fail loudly, not silently."""
    from mlforecast.pooled import get_kernel
    from mlforecast.lag_transforms import _BaseLagTransform

    class DummyTransform(_BaseLagTransform):
        pass

    with pytest.raises(NotImplementedError, match="does not support pooled"):
        get_kernel(DummyTransform())


# === partition_by tests ===


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_local_fit_transform(engine):
    """partition_by with local mode (no global_/groupby) creates correct buckets."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "ds": [1, 2, 3, 4, 1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "promo": [0, 0, 1, 1, 0, 1, 0, 1],
        },
    )
    tfm = RollingMean(2, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    result = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
    )
    # partition_by creates a partition state
    part_key = ("local", (), ("promo",))
    assert part_key in ts._pooled_states
    state = ts._pooled_states[part_key]
    assert state.mode == "local"
    assert state.partition_cols == ["promo"]
    # The feature column should exist
    col = tfm._get_name(1)
    assert "partby_promo" in col
    assert col in result.columns if engine == "polars" else col in result.columns


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_global_fit_transform(engine):
    """partition_by with global_ creates global+partition buckets."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "ds": [1, 2, 3, 4, 1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "promo": [0, 0, 1, 1, 0, 1, 0, 1],
        },
    )
    tfm = RollingMean(2, global_=True, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
    )
    part_key = ("nonlocal", (), ("promo",))
    assert part_key in ts._pooled_states
    state = ts._pooled_states[part_key]
    assert state.mode == "nonlocal"
    assert state.partition_cols == ["promo"]
    col = tfm._get_name(1)
    assert "global_" in col
    assert "partby_promo" in col


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_groupby_fit_transform(engine):
    """partition_by with groupby creates group+partition buckets."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "ds": [1, 2, 3, 4, 1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "brand": ["x", "x", "x", "x", "y", "y", "y", "y"],
            "promo": [0, 0, 1, 1, 0, 1, 0, 1],
        },
    )
    tfm = RollingMean(2, groupby=["brand"], partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=["brand"],
    )
    part_key = ("nonlocal", ("brand",), ("promo",))
    assert part_key in ts._pooled_states
    state = ts._pooled_states[part_key]
    assert state.mode == "nonlocal"
    # key_cols should include both brand and promo
    assert "brand" in state.group_cols
    assert "promo" in state.partition_cols
    col = tfm._get_name(1)
    assert "groupby_brand" in col
    assert "partby_promo" in col


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_local_predict(engine):
    """partition_by local: predict produces features without errors."""
    from mlforecast.forecast import MLForecast
    from sklearn.linear_model import LinearRegression

    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 12 + ["b"] * 12,
            "ds": list(range(1, 13)) + list(range(1, 13)),
            "y": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                90.0,
                100.0,
                110.0,
                120.0,
            ],
            "promo": [
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
            ],
        },
    )
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[],
    )
    future_df = _make_df(
        engine,
        {
            "unique_id": ["a", "b"],
            "ds": [13, 13],
            "promo": [1, 0],
        },
    )
    preds = fcst.predict(h=1, X_df=future_df)
    assert len(preds) == 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_not_in_local_tfms(engine):
    """Transforms with partition_by should not appear in local transforms."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "a", "b", "b", "b"],
            "ds": [1, 2, 3, 1, 2, 3],
            "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "promo": [0, 0, 1, 0, 1, 0],
        },
    )
    from mlforecast.lag_transforms import Lag

    tfm_local = Lag(1)
    tfm_part = RollingMean(2, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm_local, tfm_part]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
    )
    local_tfms = ts._get_local_tfms(ts.transforms)
    for t in local_tfms.values():
        assert not getattr(t, "partition_by", None)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_update(engine):
    """update() with partition_by states works correctly."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "a", "b", "b", "b"],
            "ds": [1, 2, 3, 1, 2, 3],
            "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "promo": [0, 0, 1, 0, 1, 0],
        },
    )
    tfm = RollingMean(2, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
    )
    part_key = ("local", (), ("promo",))
    state = ts._pooled_states[part_key]
    orig_len = state.base["count"].sum()

    update_df = _make_df(
        engine,
        {
            "unique_id": ["a", "b"],
            "ds": [4, 4],
            "y": [4.0, 40.0],
            "promo": [1, 0],
        },
    )
    ts.update(update_df)
    state = ts._pooled_states[part_key]
    assert state.base["count"].sum() == orig_len + 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_local_numeric_values(engine):
    """Verify rolling mean per (id, promo) bucket matches hand-computed values."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 6 + ["b"] * 6,
            "ds": list(range(1, 7)) * 2,
            "y": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
            ],
            "promo": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
        },
    )
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    result = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
    )
    col = tfm._get_name(1)
    vals = result[col].to_numpy()
    expected = np.array(
        [
            np.nan,
            10.0,
            15.0,
            np.nan,
            30.0,
            40.0,  # series a
            np.nan,
            100.0,
            np.nan,
            300.0,
            np.nan,
            400.0,  # series b
        ]
    )
    np.testing.assert_allclose(vals, expected, equal_nan=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_global_numeric_values(engine):
    """Verify rolling mean per (promo) bucket with global parent calendar."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 3 + ["b"] * 3,
            "ds": [1, 2, 3, 1, 2, 3],
            "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "promo": [0, 1, 0, 1, 0, 1],
        },
    )
    tfm = RollingMean(2, min_samples=1, global_=True, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    result = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
    )
    col = tfm._get_name(1)
    vals = result[col].to_numpy()
    expected = np.array([np.nan, 100.0, 105.0, np.nan, 10.0, 60.0])
    np.testing.assert_allclose(vals, expected, equal_nan=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_ordinals_have_parent_gaps(engine):
    """Verify ordinals are [0,2,4] not [0,1,2] when partition has gaps."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 5,
            "ds": [1, 2, 3, 4, 5],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "promo": [0, 1, 0, 1, 0],
        },
    )
    tfm = RollingMean(2, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        keep_last_n=10_000,  # full-history check: disable pooled trim
        dropna=False,
        static_features=[],
    )
    state = ts._pooled_states[("local", (), ("promo",))]
    # Buckets sit on the shared calendar, not a compacted per-bucket one:
    # bucket (a,0) is observed at ds=[1,3,5] -> ordinals [0,2,4] (NOT [0,1,2])
    # bucket (a,1) is observed at ds=[2,4]   -> ordinals [1,3]   (NOT [0,1])
    observed = [
        np.flatnonzero(state.base["count"][b] > 0).tolist()
        for b in range(state.n_buckets)
    ]
    assert sorted(observed) == sorted([[0, 2, 4], [1, 3]])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_range_semantics_with_gaps(engine):
    """RANGE (not ROWS) windowing on a partition bucket with timestamp gaps.

    Single series, parent calendar [1,2,3,4,5]. The promo=1 bucket is observed
    only at ts [1,3,5] -> parent ordinals [0,2,4]. With RollingMean(window_size=2)
    at lag 1, the window at ts=5 (ordinal 4) spans parent ordinals [2,3]; only
    ordinal 2 (ts=3) is observed, so the mean is y[ts=3]=30. ROWS semantics would
    instead average the two preceding *observations* (ts=1 and ts=3) -> 20, which
    is what this guards against.
    """
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 5,
            "ds": [1, 2, 3, 4, 5],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "promo": [1, 0, 1, 0, 1],
        },
    )
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    out = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
    )
    col = tfm._get_name(1)
    if engine == "polars":
        out = out.to_pandas()
    vals = out[out["promo"] == 1].sort_values("ds")[col].to_numpy()
    # ts=1 -> empty window (NaN); ts=3 -> only ts=1 (10); ts=5 -> only ts=3 (RANGE=30, not ROWS=20)
    np.testing.assert_array_equal(np.isnan(vals), [True, False, False])
    np.testing.assert_allclose(vals[1:], [10.0, 30.0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_expanding_with_parent_gaps(engine):
    """ExpandingMean over a gapped partition bucket accumulates by parent ordinal.

    Same gap setup as the RANGE test. With lag 1 the expanding mean at each
    observation averages all bucket observations at parent ordinals strictly
    below the current one: ts=3 -> {ts=1}=10; ts=5 -> {ts=1,ts=3}=20. The lag
    offset is applied in parent-ordinal space, so the gap at ordinal 3 (ts=4,
    unobserved) does not change which observations fall inside the window.
    """
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 5,
            "ds": [1, 2, 3, 4, 5],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "promo": [1, 0, 1, 0, 1],
        },
    )
    tfm = ExpandingMean(partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    out = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
    )
    col = tfm._get_name(1)
    if engine == "polars":
        out = out.to_pandas()
    vals = out[out["promo"] == 1].sort_values("ds")[col].to_numpy()
    np.testing.assert_array_equal(np.isnan(vals), [True, False, False])
    np.testing.assert_allclose(vals[1:], [10.0, 20.0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_dynamic_keys_multistep(engine):
    """Multi-step prediction with changing promo values in X_df."""
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 10 + ["b"] * 10,
            "ds": list(range(1, 11)) * 2,
            "y": [float(i) for i in range(1, 11)]
            + [float(i * 10) for i in range(1, 11)],
            "promo": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        },
    )
    tfm = RollingMean(3, min_samples=1, partition_by=["promo"])
    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[],
    )
    future_df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": [11, 12, 11, 12],
            "promo": [1, 0, 0, 1],
        },
    )
    preds = fcst.predict(h=2, X_df=future_df)
    assert len(preds) == 4
    if engine == "pandas":
        pred_vals = preds.iloc[:, -1].values
    else:
        pred_vals = preds[:, -1].to_numpy()
    assert not np.any(np.isnan(pred_vals))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_predict_ids_with_nonlocal_partition_raises(engine):
    """predict(ids=...) must be blocked for nonlocal partition transforms."""
    from mlforecast.forecast import MLForecast
    from sklearn.linear_model import LinearRegression

    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 4 + ["b"] * 4,
            "ds": [1, 2, 3, 4, 1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "promo": [0, 0, 1, 1, 0, 1, 0, 1],
        },
    )
    tfm = RollingMean(2, min_samples=1, global_=True, partition_by=["promo"])
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[],
    )
    future_df = _make_df(
        engine,
        {
            "unique_id": ["a"],
            "ds": [5],
            "promo": [0],
        },
    )
    with pytest.raises(ValueError, match="Cannot use `ids`"):
        fcst.predict(h=1, X_df=future_df, ids=["a"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_default_static_features_with_partition_cols(engine):
    """static_features=None should auto-exclude partition_by columns."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 4 + ["b"] * 4,
            "ds": [1, 2, 3, 4, 1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "promo": [0, 0, 1, 1, 0, 1, 0, 1],
        },
    )
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    result = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
    )
    assert "promo" not in ts.static_features_.columns
    assert tfm._get_name(1) in result.columns


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_backup_restore(engine):
    """_backup() correctly restores partition_by state."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "a", "b", "b", "b"],
            "ds": [1, 2, 3, 1, 2, 3],
            "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "promo": [0, 0, 1, 0, 1, 0],
        },
    )
    tfm = RollingMean(2, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
    )
    part_key = ("local", (), ("promo",))
    orig_y_len = ts._pooled_states[part_key].base["count"].sum()

    with ts._backup():
        # Simulate some mutation
        ts._predict_setup()
        ts._update_y(np.array([99.0, 99.0]))

    # After backup restore, state should be back to original
    assert ts._pooled_states[part_key].base["count"].sum() == orig_y_len


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_local_partition_prediction_advances_sibling_calendar(engine):
    """Multi-step predict: sibling buckets under same parent advance together.

    Uses RollingMean(2, min_samples=1) partition transform plus a regular
    lag so that fit doesn't drop the series. Captures features at each step
    via before_predict_callback and asserts exact partition feature values.
    """
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    # Series "a": promo alternates 0,1,0,1,...
    # Bucket (a,0): ds=[1,3,5,7,9] y=[10,30,50,70,90] parent ordinals [0,2,4,6,8]
    # Bucket (a,1): ds=[2,4,6,8,10] y=[20,40,60,80,100] parent ordinals [1,3,5,7,9]
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 10,
            "ds": list(range(1, 11)),
            "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "promo": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        },
    )
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    col = tfm._get_name(1)
    captured = []

    def save_features(x):
        if hasattr(x, "to_numpy"):
            captured.append(x[col].to_numpy().copy())
        else:
            captured.append(x.copy())
        return x

    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lags=[1],
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[],
    )
    # Predict 2 steps: promo=[1, 0]
    # h=0 (ds=11, promo=1): RollingMean(2,min_samples=1) of bucket(a,1)
    #   at parent ordinal 10. lag=1, window=2 → ordinals [8,9].
    #   bucket(a,1) has y=80 at ord 7, y=100 at ord 9. Ord 8 has no obs.
    #   Only ord 9 (y=100) is in [8,9]. → mean([100]) = 100.0
    # h=1 (ds=12, promo=0): bucket(a,0) at parent ordinal 11.
    #   lag=1, window=2 → ordinals [9,10]. bucket(a,0) has last obs at ord 8.
    #   Neither ord 9 nor 10 has an obs for (a,0). → NaN
    #   Key check: parent calendar advanced to length 11 for BOTH buckets.
    future_df = _make_df(
        engine,
        {
            "unique_id": ["a", "a"],
            "ds": [11, 12],
            "promo": [1, 0],
        },
    )
    preds = fcst.predict(h=2, X_df=future_df, before_predict_callback=save_features)
    assert len(preds) == 2
    assert len(captured) == 2
    np.testing.assert_allclose(captured[0][0], 100.0)
    assert np.isnan(captured[1][0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_local_partition_update_advances_sibling_calendar(engine):
    """update() advances parent calendar so sibling bucket sees new timestamp.

    After updating with promo=1 at ds=6, verify:
    - ALL sibling buckets have next_time_index = 6 (not just the updated one)
    - Feature values are correct when querying after update
    """
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 5,
            "ds": [1, 2, 3, 4, 5],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "promo": [0, 1, 0, 1, 0],
        },
    )
    # RollingMean(1, min_samples=1): returns lag-1 value within the bucket
    tfm = RollingMean(1, min_samples=1, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        keep_last_n=10_000,
        dropna=False,
        static_features=[],
    )
    part_key = ("local", (), ("promo",))
    state = ts._pooled_states[part_key]
    for bid in _next_ordinal_by_bucket(state):
        assert _next_ordinal_by_bucket(state)[bid] == 5

    update_df = _make_df(
        engine,
        {
            "unique_id": ["a"],
            "ds": [6],
            "y": [60.0],
            "promo": [1],
        },
    )
    ts.update(update_df)
    state = ts._pooled_states[part_key]
    # After update at ds=6 with promo=1, parent calendar = [1,2,3,4,5,6]
    # ALL sibling buckets should now have next_time_index = 6
    for bid in _next_ordinal_by_bucket(state):
        assert _next_ordinal_by_bucket(state)[bid] == 6

    # Verify feature computation doesn't crash and uses correct ordinals.
    # After update at ds=6 with promo=1:
    #   Bucket (a,0): observations at parent ordinals [0,2,4] (ds=[1,3,5])
    #   Bucket (a,1): observations at parent ordinals [1,3,5] (ds=[2,4,6])
    # Both buckets now at next_time_index=6. A prediction at ds=7 (ordinal 6)
    # with RollingMean(1) lag=1 for bucket(a,1) looks at ordinal 5 → y=60.
    # For bucket(a,0) it looks at ordinal 5 → no observation → NaN.
    ts._predict_setup()
    features = ts._update_features()
    col = tfm._get_name(1)
    feat_val = features[col].to_numpy()[0]
    # After update() a series sits in the bucket it was last *observed* in --
    # here (a,1), via the promo=1 row at ds=6 -- so the lag-1 lookup at ordinal 5
    # finds y=60. Real prediction always refreshes the assignment from X_df
    # first; this calls _update_features directly to inspect the state.
    np.testing.assert_allclose(feat_val, 60.0)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_new_partition_bucket_uses_existing_parent_calendar(engine):
    """A partition value unseen at fit has no bucket, so its feature is NaN.

    `main` allocated a bucket on the shared calendar for it; the channel engine
    leaves it out of the vocabulary. Either way it has no history, so the feature
    is NaN -- and the fitted state must be left untouched.
    """
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 4,
            "ds": [1, 2, 3, 4],
            "y": [10.0, 20.0, 30.0, 40.0],
            "promo": [0, 0, 0, 0],
        },
    )
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        keep_last_n=10_000,  # full-history check: disable pooled trim
        dropna=False,
        static_features=[],
    )
    part_key = ("local", (), ("promo",))
    state = ts._pooled_states[part_key]
    n_buckets_before, n_ordinals_before = state.n_buckets, state.n_ordinals

    ts._predict_setup()
    ctx = _make_df(engine, {"unique_id": ["a"], "promo": [1]})
    _set_bucket_ids(state, ctx)

    assert int(state.series_bucket_id[0]) == -1  # promo=1 never seen at fit
    assert state.n_buckets == n_buckets_before
    assert state.n_ordinals == n_ordinals_before
    # an unassigned series gets no pooled value
    assert np.isnan(state.broadcast(np.zeros(state.n_buckets))[0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_global_partition_update_advances_sibling_calendar(engine):
    """Global+partition: update advances all sibling bucket ordinals."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "a", "b", "b", "b"],
            "ds": [1, 2, 3, 1, 2, 3],
            "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "promo": [0, 1, 0, 1, 0, 1],
        },
    )
    tfm = RollingMean(2, min_samples=1, global_=True, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        keep_last_n=10_000,  # full-history check: disable pooled trim
        dropna=False,
        static_features=[],
    )
    part_key = ("nonlocal", (), ("promo",))
    state = ts._pooled_states[part_key]
    # Global parent calendar = [1,2,3], length 3
    for bid in _next_ordinal_by_bucket(state):
        assert _next_ordinal_by_bucket(state)[bid] == 3

    update_df = _make_df(
        engine,
        {
            "unique_id": ["a", "b"],
            "ds": [4, 4],
            "y": [40.0, 400.0],
            "promo": [1, 1],
        },
    )
    ts.update(update_df)
    state = ts._pooled_states[part_key]
    # Parent calendar now [1,2,3,4], ALL buckets should be at 4
    for bid in _next_ordinal_by_bucket(state):
        assert _next_ordinal_by_bucket(state)[bid] == 4


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_groupby_partition_update_advances_sibling_calendar(engine):
    """Groupby+partition: update advances sibling buckets within each group."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "a", "b", "b", "b"],
            "ds": [1, 2, 3, 1, 2, 3],
            "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "brand": ["x", "x", "x", "y", "y", "y"],
            "promo": [0, 1, 0, 1, 0, 1],
        },
    )
    tfm = RollingMean(2, min_samples=1, groupby=["brand"], partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        keep_last_n=10_000,  # full-history check: disable pooled trim
        dropna=False,
        static_features=["brand"],
    )
    part_key = ("nonlocal", ("brand",), ("promo",))
    state = ts._pooled_states[part_key]
    # Each brand group has parent calendar [1,2,3], length 3
    for bid in _next_ordinal_by_bucket(state):
        assert _next_ordinal_by_bucket(state)[bid] == 3

    update_df = _make_df(
        engine,
        {
            "unique_id": ["a", "b"],
            "ds": [4, 4],
            "y": [40.0, 400.0],
            "brand": ["x", "y"],
            "promo": [1, 0],
        },
    )
    ts.update(update_df)
    state = ts._pooled_states[part_key]
    for bid in _next_ordinal_by_bucket(state):
        assert _next_ordinal_by_bucket(state)[bid] == 4


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_assignment_missing_key_error(engine):
    """Missing partition key in X_df and static_features raises ValueError."""
    from mlforecast.forecast import MLForecast
    from sklearn.linear_model import LinearRegression

    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 4,
            "ds": [1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0],
            "promo": [0, 1, 0, 1],
        },
    )
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[],
    )
    # X_df missing "promo" column but has another exogenous feature
    future_df = _make_df(
        engine,
        {
            "unique_id": ["a"],
            "ds": [5],
            "other_feature": [1.0],
        },
    )
    with pytest.raises(ValueError, match="X_df is missing future values"):
        fcst.predict(h=1, X_df=future_df)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_predict_requires_x_df(engine):
    """predict(h) without X_df must error when partition_by is configured."""
    from mlforecast.forecast import MLForecast
    from sklearn.linear_model import LinearRegression

    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 4,
            "ds": [1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0],
            "promo": [0, 1, 0, 1],
        },
    )
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[],
    )
    with pytest.raises(ValueError, match="X_df is required for prediction"):
        fcst.predict(h=1)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_local_unseen_partition_predict(engine):
    """Fit with promo=0 only, predict h=1 with unseen promo=1."""
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 4,
            "ds": [1, 2, 3, 4],
            "y": [10.0, 20.0, 30.0, 40.0],
            "promo": [0, 0, 0, 0],
        },
    )
    tfm = RollingMean(1, min_samples=1, partition_by=["promo"])
    col = tfm._get_name(1)
    captured = []

    def save_features(x):
        if hasattr(x, "to_numpy"):
            captured.append(x[col].to_numpy().copy())
        else:
            captured.append(x.copy())
        return x

    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[],
    )
    future_df = _make_df(
        engine,
        {
            "unique_id": ["a"],
            "ds": [5],
            "promo": [1],
        },
    )
    preds = fcst.predict(h=1, X_df=future_df, before_predict_callback=save_features)
    assert len(preds) == 1
    # Unseen bucket promo=1 has no historical data → feature is NaN
    assert np.isnan(captured[0][0])
    # Prediction should still be finite (model handles NaN)
    if engine == "pandas":
        assert not np.isnan(preds.iloc[0, -1])
    else:
        assert not np.isnan(preds[:, -1].to_numpy()[0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_global_partition_unseen_bucket_predict(engine):
    """Regression test for Bug 1: global+partition with unseen bucket at predict.

    Previously raised IndexError because append_predictions passed
    new_groups=zeros for buckets that didn't exist in the GA yet.
    """
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "ds": [1, 2, 3, 4, 1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "promo": [0, 0, 0, 0, 0, 0, 0, 0],
        },
    )
    tfm = RollingMean(1, min_samples=1, global_=True, partition_by=["promo"])
    col = tfm._get_name(1)
    captured = []

    def save_features(x):
        if hasattr(x, "to_numpy"):
            captured.append(x[col].to_numpy().copy())
        else:
            captured.append(x.copy())
        return x

    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[],
    )
    future_df = _make_df(
        engine,
        {
            "unique_id": ["a", "b"],
            "ds": [5, 5],
            "promo": [1, 1],
        },
    )
    preds = fcst.predict(h=1, X_df=future_df, before_predict_callback=save_features)
    assert len(preds) == 2
    # Unseen bucket promo=1 has no historical data → feature is NaN
    assert np.all(np.isnan(captured[0]))
    # Predictions should be finite
    if engine == "pandas":
        pred_vals = preds.iloc[:, -1].values
    else:
        pred_vals = preds[:, -1].to_numpy()
    assert not np.any(np.isnan(pred_vals))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_global_partition_new_bucket_inherits_parent_calendar(engine):
    """Same, for a global+partition bucket: unseen partition value -> no bucket."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": [1, 2, 1, 2],
            "y": [1.0, 2.0, 10.0, 20.0],
            "promo": [0, 0, 0, 0],
        },
    )
    tfm = RollingMean(2, min_samples=1, global_=True, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        keep_last_n=10_000,
        dropna=False,
        static_features=[],
    )
    state = ts._pooled_states[("nonlocal", (), ("promo",))]
    n_buckets_before, n_ordinals_before = state.n_buckets, state.n_ordinals

    ts._predict_setup()
    ctx = _make_df(engine, {"unique_id": ["a", "b"], "promo": [1, 1]})
    _set_bucket_ids(state, ctx)

    assert set(state.series_bucket_id.tolist()) == {-1}
    assert state.n_buckets == n_buckets_before
    assert state.n_ordinals == n_ordinals_before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_datetime_update_new_bucket(engine):
    """Regression test for Bug 3: datetime dtype mismatch on new parent grid.

    _resolve_parent_for_bucket used to create np.array([], dtype=float)
    which raises DTypePromotionError with datetime timestamps.
    """
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 4,
            "ds": dates,
            "y": [10.0, 20.0, 30.0, 40.0],
            "promo": [0, 0, 0, 0],
        },
    )
    tfm = RollingMean(1, min_samples=1, partition_by=["promo"])
    freq = "1d" if engine == "polars" else "D"
    ts = TimeSeries(freq=freq, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        keep_last_n=10_000,  # full-history check: disable pooled trim
        dropna=False,
        static_features=[],
    )
    # Update with a new partition value — this triggers _resolve_parent_for_bucket
    # which creates a new parent grid. With the dtype fix, the grid dtype
    # matches the existing datetime64 dtype.
    update_df = _make_df(
        engine,
        {
            "unique_id": ["a"],
            "ds": pd.to_datetime(["2020-01-05"]),
            "y": [50.0],
            "promo": [1],
        },
    )
    ts.update(update_df)
    part_key = ("local", (), ("promo",))
    state = ts._pooled_states[part_key]
    # Parent calendar = [2020-01-01..05], length 5
    for bid in _next_ordinal_by_bucket(state):
        assert _next_ordinal_by_bucket(state)[bid] == 5


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_backup_restore_with_dynamic_buckets(engine):
    """Predicting with an unseen partition value must leave the fitted state intact.

    The dynamic bucket is created inside `_backup()`, so the bucket vocabulary,
    bucket count and calendar length must all come back unchanged.
    """
    from mlforecast.forecast import MLForecast
    from sklearn.base import BaseEstimator

    class _NanTolerant(BaseEstimator):
        """The unseen partition value produces a NaN feature by design."""

        def fit(self, _X, _y=None):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "a", "a"],
            "ds": [1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0],
            "promo": [0, 0, 0, 0],
        },
    )
    fcst = MLForecast(
        models=[_NanTolerant()],
        freq=1,
        lags=[1],
        lag_transforms={1: [RollingMean(2, min_samples=1, partition_by=["promo"])]},
    )
    fcst.fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[],
        dropna=True,
    )
    part_key = ("local", (), ("promo",))
    before = fcst.ts._pooled_states[part_key]
    n_buckets_before = before.n_buckets
    uniques_before = (
        None if before.bucket_uniques is None else before.bucket_uniques.copy()
    )
    n_ordinals_before = before.n_ordinals

    # Predict with unseen promo=1 -- creates a dynamic bucket inside _backup()
    future_df = _make_df(
        engine,
        {"unique_id": ["a"], "ds": [5], "promo": [1]},
    )
    fcst.predict(h=1, X_df=future_df)

    after = fcst.ts._pooled_states[part_key]
    assert after.n_buckets == n_buckets_before
    assert after.n_ordinals == n_ordinals_before
    if uniques_before is None:
        assert after.bucket_uniques is None
    else:
        np.testing.assert_array_equal(after.bucket_uniques, uniques_before)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_predict_multi_horizon_multiple_unseen(engine):
    """h=4 prediction where X_df introduces several never-seen partition values.

    Each new promo value spawns a fresh bucket on the fly; the shared global
    parent calendar keeps every bucket aligned, so the brand-new buckets start
    empty (NaN feature) yet predictions stay finite across all horizons.
    """
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 5 + ["b"] * 5,
            "ds": list(range(1, 6)) * 2,
            "y": [float(i) for i in range(1, 6)] + [float(i * 10) for i in range(1, 6)],
            "promo": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        },
    )
    tfm = RollingMean(2, min_samples=1, global_=True, partition_by=["promo"])
    col = tfm._get_name(1)
    captured = []

    def save_features(x):
        captured.append(x[col].to_numpy().copy())
        return x

    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(df, id_col="unique_id", time_col="ds", target_col="y", static_features=[])
    # promo values 2 and 3 are never seen during fit; alternate across 4 horizons
    future_df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 4 + ["b"] * 4,
            "ds": [6, 7, 8, 9] * 2,
            "promo": [2, 3, 2, 3, 2, 3, 2, 3],
        },
    )
    preds = fcst.predict(h=4, X_df=future_df, before_predict_callback=save_features)
    assert len(preds) == 8
    # step 0 hits the brand-new promo=2 bucket with no history -> NaN feature
    assert np.all(np.isnan(captured[0]))
    if engine == "pandas":
        pred_vals = preds.iloc[:, -1].to_numpy()
    else:
        pred_vals = preds[:, -1].to_numpy()
    assert not np.any(np.isnan(pred_vals))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_local_partition_recursive_h3_consistency(engine):
    """Local partition recursive predict slides the rolling window correctly.

    Single series, constant promo=1, RollingMean(window_size=2, lag=1). Each
    horizon's feature must equal the mean of the two most recent values (real,
    then predicted), so the window advances over the appended predictions
    rather than resetting or jumping.
    """
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 5,
            "ds": [1, 2, 3, 4, 5],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "promo": [1, 1, 1, 1, 1],
        },
    )
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    col = tfm._get_name(1)
    feats = []

    def save_features(x):
        feats.append(float(x[col].to_numpy()[0]))
        return x

    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(df, id_col="unique_id", time_col="ds", target_col="y", static_features=[])
    future_df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 3,
            "ds": [6, 7, 8],
            "promo": [1, 1, 1],
        },
    )
    preds = fcst.predict(h=3, X_df=future_df, before_predict_callback=save_features)
    if engine == "pandas":
        p = preds.iloc[:, -1].to_numpy()
    else:
        p = preds[:, -1].to_numpy()
    # h1 uses real y[ds4],y[ds5]; h2 uses y[ds5],pred[ds6]; h3 uses pred[ds6],pred[ds7]
    np.testing.assert_allclose(feats[0], (40.0 + 50.0) / 2)
    np.testing.assert_allclose(feats[1], (50.0 + p[0]) / 2)
    np.testing.assert_allclose(feats[2], (p[0] + p[1]) / 2)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_update_batch_multiple_ids_new_buckets(engine):
    """A single update() carrying several series, each with an unseen partition
    value, registers one new bucket per value and advances every parent calendar.
    """
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "ds": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "promo": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
    )
    tfm = RollingMean(2, min_samples=1, global_=True, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        keep_last_n=10_000,  # full-history check: disable pooled trim
        dropna=False,
        static_features=[],
    )
    key = ("nonlocal", (), ("promo",))
    state = ts._pooled_states[key]
    assert len(state.bucket_uniques) == 1  # only promo=0 seen at fit

    # one batch at ds=4: three series, three never-seen promo values
    update_df = _make_df(
        engine,
        {
            "unique_id": ["a", "b", "c"],
            "ds": [4, 4, 4],
            "y": [4.0, 40.0, 400.0],
            "promo": [1, 2, 3],
        },
    )
    ts.update(update_df)
    state = ts._pooled_states[key]
    assert len(state.bucket_uniques) == 4  # promo {0, 1, 2, 3}
    # parent calendar advanced to [1,2,3,4]; every sibling bucket sees length 4
    for bid in _next_ordinal_by_bucket(state):
        assert _next_ordinal_by_bucket(state)[bid] == 4


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_update_sparse_then_dense(engine):
    """Fit on sparse partition transitions, then apply dense per-step updates;
    the resulting state must match a from-scratch fit on the combined data.
    """

    def _aggs_by_key(state):
        """Per-bucket observed ordinals and aggregates, keyed by bucket key."""
        out = {}
        for bid in range(state.n_buckets):
            counts = state.base["count"][bid]
            obs = np.flatnonzero(counts > 0)
            out[state.bucket_uniques[bid]] = (
                obs.tolist(),
                np.round(state.base["sum"][bid][obs], 6).tolist(),
                counts[obs].tolist(),
            )
        return out

    def _build():
        return TimeSeries(
            freq=1,
            lag_transforms={
                1: [RollingMean(2, min_samples=1, global_=True, partition_by=["promo"])]
            },
        )

    base = {
        "unique_id": ["a", "a", "a", "b", "b", "b"],
        "ds": [1, 2, 3, 1, 2, 3],
        "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        "promo": [0, 0, 0, 0, 0, 0],
    }
    dense_steps = [
        {"ds": 4, "promo": [1, 0], "y": [4.0, 40.0]},
        {"ds": 5, "promo": [0, 1], "y": [5.0, 50.0]},
        {"ds": 6, "promo": [1, 1], "y": [6.0, 60.0]},
    ]

    ts_incr = _build()
    ts_incr.fit_transform(
        _make_df(engine, base),
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        keep_last_n=10_000,  # full-history check: disable pooled trim
        dropna=False,
        static_features=[],
    )
    rows = [base]
    for step in dense_steps:
        u = {
            "unique_id": ["a", "b"],
            "ds": [step["ds"], step["ds"]],
            "y": step["y"],
            "promo": step["promo"],
        }
        ts_incr.update(_make_df(engine, u))
        rows.append(u)

    combined = {k: sum((r[k] for r in rows), []) for k in base}
    ts_scratch = _build()
    ts_scratch.fit_transform(
        _make_df(engine, combined),
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
        keep_last_n=10_000,  # full-history check: disable pooled trim
    )
    key = ("nonlocal", (), ("promo",))
    assert _aggs_by_key(ts_incr._pooled_states[key]) == _aggs_by_key(
        ts_scratch._pooled_states[key]
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_static_features_explicit_with_partition_cols(engine):
    """Explicit static_features are honored while partition columns are excluded.

    With static_features=["brand","region"] and partition_by=["promo"], the
    fitted static set keeps brand/region but drops promo (it is dynamic and
    re-supplied via X_df at predict), and promo never enters features_order_.
    """
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 4 + ["b"] * 4,
            "ds": [1, 2, 3, 4] * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "brand": ["x"] * 4 + ["y"] * 4,
            "region": ["N"] * 4 + ["S"] * 4,
            "promo": [0, 1, 0, 1, 1, 0, 1, 0],
        },
    )
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=["brand", "region"],
    )
    static_cols = set(ts.static_features_.columns)
    assert {"brand", "region"} <= static_cols
    assert "promo" not in static_cols
    assert "promo" not in ts.features_order_


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_rolling_min_samples_boundary(engine):
    """min_samples is the exact coverage threshold inside a partition bucket.

    RollingMean(window_size=3, min_samples=2) at lag 1 over a single promo=1
    bucket: at ds=2 the window holds 1 observation (< min_samples) -> NaN; at
    ds=3 it holds exactly 2 (== min_samples) -> the mean (10+20)/2 = 15.
    """
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 5,
            "ds": [1, 2, 3, 4, 5],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "promo": [1, 1, 1, 1, 1],
        },
    )
    tfm = RollingMean(3, min_samples=2, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    out = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
    )
    col = tfm._get_name(1)
    if engine == "polars":
        out = out.to_pandas()
    vals = out.sort_values("ds")[col].to_numpy()
    # ds=1,2 below threshold -> NaN; ds=3 hits exactly min_samples -> value
    assert np.isnan(vals[0]) and np.isnan(vals[1])
    np.testing.assert_allclose(vals[2], 15.0)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_cv_fold_independent(engine):
    """cross_validation runs across folds with partition_by and does not leak
    dynamic buckets between folds.
    """
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    n_times = 12
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * n_times + ["b"] * n_times,
            "ds": list(range(1, n_times + 1)) * 2,
            "y": [float(i) for i in range(1, n_times + 1)]
            + [float(i * 10) for i in range(1, n_times + 1)],
            "promo": ([0, 1] * (n_times // 2)) * 2,
        },
    )
    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=5)],
        freq=1,
        lag_transforms={
            1: [RollingMean(2, min_samples=1, global_=True, partition_by=["promo"])]
        },
    )
    cv = fcst.cross_validation(df, n_windows=2, h=2, static_features=[])
    assert len(cv) == 2 * 2 * 2  # n_windows * h * n_series
    pred_vals = cv["HistGradientBoostingRegressor"].to_numpy()
    assert not np.any(np.isnan(pred_vals))
    # no bucket bleed across folds: only promo {0, 1} ever exists
    state = fcst.ts._pooled_states[("nonlocal", (), ("promo",))]
    assert len(state.bucket_uniques) == 2


# === Tests ported from feature/groupby_with_range_semantics ===


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
        fcst = self._make_fcst({1: [RollingMean(window_size=2, groupby=["brand"])]})
        df = self._make_simple_df()
        with pytest.warns(UserWarning, match="Pooled.*validate_data"):
            fcst.preprocess(df, static_features=["brand"], validate_data=False)

    def test_warns_partition_by(self):
        fcst = self._make_fcst(
            {1: [RollingMean(window_size=2, global_=True, partition_by=["brand"])]}
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
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 4 + ["b"] * 4,
            "ds": list(range(4)) * 2,
            "y": [6.0, 7.0, 8.0, 9.0, 6.0, 7.0, 8.0, 9.0],
        },
    )
    tfm = ExponentiallyWeightedMean(alpha=0.5, global_=True)
    ts = TimeSeries(freq=1, lag_transforms={2: [tfm]})
    result = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
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
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 4 + ["b"] * 4,
            "ds": list(range(4)) * 2,
            "y": [6.0, 7.0, 8.0, 9.0, 6.0, 7.0, 8.0, 9.0],
            "grp": ["X"] * 8,
        },
    )
    tfm = ExponentiallyWeightedMean(alpha=0.5, groupby=["grp"])
    ts = TimeSeries(freq=1, lag_transforms={2: [tfm]})
    result = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=["grp"],
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
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=static,
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
        "global_rolling_mean_lag3_window_size3": (
            [nan, nan, nan, nan, 2.5, 3.5],
            [5.5, 5.5],
        ),
        "global_rolling_std_lag3_window_size3": (
            [nan, nan, nan, nan, 1.290994, 1.870829],
            [1.870829, 1.870829],
        ),
        "global_rolling_min_lag3_window_size3": (
            [nan, nan, nan, nan, 1.0, 1.0],
            [3.0, 3.0],
        ),
        "global_rolling_max_lag3_window_size3": (
            [nan, nan, nan, nan, 4.0, 6.0],
            [8.0, 8.0],
        ),
        "global_expanding_mean_lag3": ([nan, nan, nan, 1.5, 2.5, 3.5], [4.5, 4.5]),
        "global_expanding_std_lag3": (
            [nan, nan, nan, 0.707107, 1.290994, 1.870829],
            [2.449490, 2.449490],
        ),
        "global_expanding_min_lag3": ([nan, nan, nan, 1.0, 1.0, 1.0], [1.0, 1.0]),
        "global_expanding_max_lag3": ([nan, nan, nan, 2.0, 4.0, 6.0], [8.0, 8.0]),
    }
    for col, (exp_pre, exp_pred) in expected.items():
        np.testing.assert_allclose(
            out[col]["preprocess"],
            exp_pre,
            atol=1e-5,
            equal_nan=True,
            err_msg=f"preprocess mismatch for {col}",
        )
        np.testing.assert_allclose(
            out[col]["predict"],
            exp_pred,
            atol=1e-5,
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
        "groupby_grp_rolling_mean_lag2_window_size3": [nan, nan, nan, 8.25, 11.0],
        "groupby_grp_rolling_min_lag2_window_size3": [nan, nan, nan, 1.0, 1.0],
        "groupby_grp_rolling_max_lag2_window_size3": [nan, nan, nan, 20.0, 30.0],
        "groupby_grp_expanding_mean_lag2": [nan, nan, 5.5, 8.25, 11.0],
        "groupby_grp_expanding_min_lag2": [nan, nan, 1.0, 1.0, 1.0],
        "groupby_grp_expanding_max_lag2": [nan, nan, 10.0, 20.0, 30.0],
        "groupby_grp_exponentially_weighted_mean_lag2_alpha0.5": [
            nan,
            nan,
            5.5,
            8.25,
            12.375,
        ],
    }
    for col, exp_pre in expected.items():
        np.testing.assert_allclose(
            out[col]["preprocess"],
            exp_pre,
            atol=1e-5,
            equal_nan=True,
            err_msg=f"preprocess mismatch for {col}",
        )


def _view_test_frame():
    rng = np.random.default_rng(42)
    n_series, n_times = 8, 12
    ids = np.repeat([f"s{i}" for i in range(n_series)], n_times)
    times = np.tile(range(n_times), n_series)
    grps = np.repeat(["A"] * (n_series // 2) + ["B"] * (n_series // 2), n_times)
    return pd.DataFrame(
        {
            "unique_id": ids,
            "ds": times,
            "y": rng.standard_normal(n_series * n_times),
            "grp": grps,
            "promo": rng.integers(0, 2, n_series * n_times),
        }
    )


def _assert_view_matches_direct(tfm_factory, lag, mode, time_agg):
    """Compute a pooled feature, then again from a state holding only that view."""
    from mlforecast.pooled import _collapse

    kwargs = dict(mode)
    if time_agg is not None:
        kwargs["time_agg"] = time_agg
    tfm = tfm_factory(kwargs)
    df = _view_test_frame()
    ts = TimeSeries(freq=1, lag_transforms={lag: [tfm]})
    out = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=["grp"],
    )
    col = tfm._get_name(lag)
    got = out[col].to_numpy()

    # the view the kernel read must equal collapsing the base store directly
    key = next(iter(ts._pooled_states))
    state = ts._pooled_states[key]
    agg = tfm.time_agg
    view = state.channels(agg)
    if agg is None:
        direct = state.base
    else:
        direct = _collapse(*_view_value(state.base, agg), state.cell_shift(agg))
    for name in view:
        np.testing.assert_allclose(
            view[name], direct[name], equal_nan=True, err_msg=f"{col} {name}"
        )
    assert np.isfinite(got[~np.isnan(got)]).all()


@pytest.mark.parametrize(
    "tfm_factory",
    [
        lambda m: RollingMean(window_size=4, **m),
        lambda m: RollingStd(window_size=4, **m),
        lambda m: RollingMin(window_size=4, **m),
        lambda m: RollingMax(window_size=4, **m),
        lambda m: ExpandingMean(**m),
        lambda m: ExpandingStd(**m),
        lambda m: ExpandingMin(**m),
        lambda m: ExpandingMax(**m),
        lambda m: ExponentiallyWeightedMean(alpha=0.3, **m),
    ],
    ids=[
        "RollingMean",
        "RollingStd",
        "RollingMin",
        "RollingMax",
        "ExpandingMean",
        "ExpandingStd",
        "ExpandingMin",
        "ExpandingMax",
        "EWM",
    ],
)
@pytest.mark.parametrize("lag", _LAGS)
def test_fast_vs_slow_equivalence(tfm_factory, lag):
    """A cached collapsed view equals a state built directly for that collapse.

    `PooledState` keys on the bucket definition alone and derives each `time_agg`
    as a cached view over one base store. This pins that invariant: the view must
    be indistinguishable from aggregating the panel for that `time_agg` directly.
    """
    _assert_view_matches_direct(tfm_factory, lag, {"global_": True}, None)
    _assert_view_matches_direct(tfm_factory, lag, {"groupby": ["grp"]}, None)


@pytest.mark.parametrize(
    "tfm_factory",
    [
        lambda m: RollingMean(window_size=4, **m),
        lambda m: RollingStd(window_size=4, **m),
        lambda m: RollingMin(window_size=4, **m),
        lambda m: RollingMax(window_size=4, **m),
        lambda m: ExpandingMean(**m),
        lambda m: ExpandingStd(**m),
        lambda m: ExpandingMin(**m),
        lambda m: ExpandingMax(**m),
        lambda m: ExponentiallyWeightedMean(alpha=0.3, **m),
    ],
    ids=[
        "RollingMean",
        "RollingStd",
        "RollingMin",
        "RollingMax",
        "ExpandingMean",
        "ExpandingStd",
        "ExpandingMin",
        "ExpandingMax",
        "EWM",
    ],
)
@pytest.mark.parametrize("lag", _LAGS)
def test_fast_vs_slow_partition(tfm_factory, lag):
    """Same view/direct invariant with partition_by buckets."""
    _assert_view_matches_direct(
        tfm_factory, lag, {"global_": True, "partition_by": ["promo"]}, None
    )
    _assert_view_matches_direct(
        tfm_factory, lag, {"groupby": ["grp"], "partition_by": ["promo"]}, None
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_fast_vs_slow_local_partition_with_nan(engine, lag):
    """Local partition_by with a missing partition value: the slow-path join
    (forced by clearing the aggregate cache and the idsorted permutation) keys on
    (id, time), so missing-partition rows are matched, not dropped — matching the
    fast path. Guards the local ``join_cols`` fix on both engine join paths.

    The missing value is ``None`` (-> polars *null*, pandas NaN): a raw polars join
    on a null key does NOT match (raw pandas merge and polars NaN both do), so the
    polars case is what actually exercises the fix."""
    n_series, n_times = 4, 10
    ids = np.repeat([f"s{i}" for i in range(n_series)], n_times)
    times = np.tile(range(n_times), n_series)
    y = np.random.default_rng(11).standard_normal(n_series * n_times)
    # contiguous missing run so the missing-partition bucket has dense observations
    # and (with min_samples=1) produces non-NaN values — otherwise the fast vs slow
    # comparison is vacuously all-NaN and would not catch a dropped-row join.
    promo = [None, None, None, None, None, 0.0, 0.0, 1.0, 1.0, 0.0] * n_series
    df = _make_df(
        engine,
        {
            "unique_id": ids.tolist(),
            "ds": times.tolist(),
            "y": y.tolist(),
            "promo": promo,
        },
    )

    tfm = RollingMean(window_size=2, min_samples=1, partition_by=["promo"])
    col = tfm._get_name(lag)
    ts = TimeSeries(freq=1, lag_transforms={lag: [tfm]})
    fast = np.asarray(
        ts.fit_transform(
            df,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            dropna=False,
            static_features=[],
        )[col]
    )

    ts_slow = TimeSeries(
        freq=1,
        lag_transforms={
            lag: [RollingMean(window_size=2, min_samples=1, partition_by=["promo"])]
        },
    )
    ts_slow._fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[],
    )
    for st in ts_slow._pooled_states.values():
        st._ts_aggs = {}
        st._idsorted_to_bucket_pos = None
    slow = np.asarray(ts_slow._transform(df=df, dropna=False)[col])

    np.testing.assert_allclose(
        fast,
        slow,
        atol=1e-10,
        equal_nan=True,
        err_msg=f"local+partition NaN fast vs slow for {col}",
    )
    # the NaN-partition rows must receive values from the slow-path join, not be
    # dropped (which would leave them NaN where the fast path has a value).
    assert not np.all(np.isnan(slow))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_predict_x_df_partition_column_has_nan(engine):
    """predict with an X_df whose partition column is NaN routes to the existing
    missing bucket and yields finite predictions instead of crashing."""
    from sklearn.linear_model import LinearRegression
    from mlforecast.forecast import MLForecast

    n_series, n_times = 4, 8
    ids = np.repeat([f"s{i}" for i in range(n_series)], n_times)
    times = np.tile(range(n_times), n_series)
    y = np.random.default_rng(3).standard_normal(n_series * n_times)
    promo = np.tile([0.0, np.nan, 1.0, np.nan, 0.0, 1.0, 0.0, np.nan], n_series)
    df = _make_df(
        engine,
        {
            "unique_id": ids.tolist(),
            "ds": times.tolist(),
            "y": y.tolist(),
            "promo": promo.tolist(),
        },
    )
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        lags=[1],
        lag_transforms={
            1: [
                RollingMean(
                    window_size=2, min_samples=1, global_=True, partition_by=["promo"]
                )
            ]
        },
    )
    fcst.fit(df, static_features=[])

    # All future partition values are NaN: every step routes to the existing
    # missing bucket (created at fit), which stays populated as predictions feed
    # back in — exercising null-partition routing through update_series_bucket_id
    # at predict time without an IndexError / garbage bucket.
    h = 2
    fut_ids = np.repeat([f"s{i}" for i in range(n_series)], h)
    fut_ds = np.tile([n_times, n_times + 1], n_series)
    fut_promo = np.full(n_series * h, np.nan)
    X_df = _make_df(
        engine,
        {
            "unique_id": fut_ids.tolist(),
            "ds": fut_ds.tolist(),
            "promo": fut_promo.tolist(),
        },
    )
    preds = fcst.predict(h, X_df=X_df)
    pvals = np.asarray(preds["LinearRegression"])
    assert np.all(np.isfinite(pvals))  # no IndexError / dropped NaN-partition rows


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_groupby_partition_null_scope_resolves_at_predict(engine):
    """NaN group keys must share one scope, at fit and at predict.

    ``discount`` is genuinely numeric with NaN, so without a sentinel encoding
    raw ``NaN != NaN`` would spawn a fresh scope per row. The encoded key must
    put every NaN-discount bucket in the same group scope.
    """
    from mlforecast.pooled import encode_keys

    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 6 + ["b"] * 6 + ["c"] * 6,
            "ds": list(range(6)) * 3,
            "y": [float(i) for i in range(18)],
            "discount": [0.25] * 6 + [float("nan")] * 6 + [float("nan")] * 6,
            "promo": [0, 1, 0, 1, 0, 1] * 3,
        },
    )
    tfm = RollingMean(
        window_size=2, min_samples=1, groupby=["discount"], partition_by=["promo"]
    )
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=["discount"],
    )
    state = ts._pooled_states[("nonlocal", ("discount",), ("promo",))]

    # every NaN-discount bucket shares one group scope: the encoded key splits
    # into (discount, promo), and the discount half is identical across them
    nan_scope = encode_keys([np.array([float("nan")])])[0]
    scopes = {k.split("\x1f")[0] for k in state.bucket_uniques}
    null_buckets = [k for k in state.bucket_uniques if k.split("\x1f")[0] == nan_scope]
    assert len(null_buckets) >= 2  # promo 0 and 1 within the NaN scope
    assert len(scopes) == 2  # 0.25 and NaN, not one scope per NaN row

    # predict-time context: NaN-discount series b takes an UNSEEN promo (9)
    ts._predict_setup()
    ctx = _make_df(
        engine,
        {
            "unique_id": ["a", "b", "c"],
            "discount": [0.25, float("nan"), float("nan")],
            "promo": [0, 9, 1],
        },
    )
    _set_bucket_ids(state, ctx)  # must not raise

    bids = state.series_bucket_id
    assert bids[1] == -1  # (NaN, 9) was never seen -> no bucket
    # (NaN, 1) still resolves to its existing bucket rather than a fresh scope
    assert bids[2] >= 0
    assert state.bucket_uniques[bids[2]].split("\x1f")[0] == nan_scope


def test_fractional_float_partition_feature_parity_across_engines():
    """fit_transform with a fractional-float partition key yields identical feature
    outputs on pandas and polars. The SQLite oracle is pandas-only, so this guards
    the polars float->string encoding path against divergence."""
    n_series, n_times = 4, 8
    ids = np.repeat([f"s{i}" for i in range(n_series)], n_times)
    times = np.tile(range(n_times), n_series)
    y = np.random.default_rng(5).standard_normal(n_series * n_times)
    discount = [0.1, 0.1, 0.25, 0.5, 0.1, 0.25, 0.1, 0.5] * n_series
    rows = {
        "unique_id": ids.tolist(),
        "ds": times.tolist(),
        "y": y.tolist(),
        "discount": discount,
    }
    outs = []
    for engine in ["pandas", "polars"]:
        tfm = RollingMean(
            window_size=2, min_samples=1, global_=True, partition_by=["discount"]
        )
        col = tfm._get_name(1)
        ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
        res = ts.fit_transform(
            _make_df(engine, rows),
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            dropna=False,
            static_features=[],
        )
        outs.append(np.asarray(res[col]))
    np.testing.assert_allclose(outs[0], outs[1], atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_prediction_fast_path_partition(engine):
    """Multi-step predict with fast path + partition_by produces finite values."""
    n_series, n_times = 4, 8
    ids = np.repeat([f"s{i}" for i in range(n_series)], n_times)
    times = np.tile(range(n_times), n_series)
    y = np.random.default_rng(7).standard_normal(n_series * n_times)
    promo = np.tile([0, 0, 1, 1, 0, 1, 0, 1][:n_times], n_series)
    rows = {
        "unique_id": ids.tolist(),
        "ds": times.tolist(),
        "y": y.tolist(),
        "promo": promo.tolist(),
    }
    df = _make_df(engine, rows)

    tfm = RollingMean(window_size=3, global_=True, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[],
    )
    ts._predict_setup()
    for step in range(3):
        features = ts._update_features()
        col = tfm._get_name(1)
        vals = features[col].to_numpy()
        assert not np.all(np.isnan(vals)), f"step {step}: all NaN for {col}"
        ts._update_y(vals)


def test_partition_ewm_skips_missing_parent_ordinals():
    """EWM on a partition bucket with gapped parent ordinals [0,1,4,5]
    decays only across observed bucket timestamps, not across missing
    parent ordinals 2 and 3."""
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 8 + ["b"] * 8,
            "ds": list(range(8)) * 2,
            "y": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                12.0,
                22.0,
                32.0,
                42.0,
                52.0,
                62.0,
                72.0,
                82.0,
            ],
            "promo": [0, 0, 1, 1, 0, 0, 1, 1] * 2,
        }
    )

    alpha = 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tfm = ExponentiallyWeightedMean(
            alpha=alpha, global_=True, partition_by=["promo"]
        )
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    result = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
    )
    col = tfm._get_name(1)
    vals = result[col].values

    # promo=0 bucket: parent ordinals [0,1,4,5], aggregate means [11,21,51,61]
    # Two-pointer EWM (lag=1, alpha=0.5):
    #   ord 0: upper=-1 → NaN
    #   ord 1: consume ord 0 (mean=11) → 11.0
    #   ord 4: consume ord 1 (mean=21) → 0.5*21 + 0.5*11 = 16.0
    #          ords 2,3 are missing — NOT consumed, no extra decay
    #   ord 5: consume ord 4 (mean=51) → 0.5*51 + 0.5*16 = 33.5
    expected_p0 = [np.nan, 11.0, 16.0, 33.5]

    # promo=1 bucket: parent ordinals [2,3,6,7], aggregate means [31,41,71,81]
    #   ord 2: upper=1, nothing observed ≤ 1 in this bucket → NaN
    #   ord 3: consume ord 2 (mean=31) → 31.0
    #   ord 6: consume ord 3 (mean=41) → 0.5*41 + 0.5*31 = 36.0
    #   ord 7: consume ord 6 (mean=71) → 0.5*71 + 0.5*36 = 53.5
    expected_p1 = [np.nan, 31.0, 36.0, 53.5]

    promo = df["promo"].values
    for start in range(0, len(df), 8):
        chunk = vals[start : start + 8]
        p = promo[start : start + 8]
        np.testing.assert_allclose(
            chunk[p == 0],
            expected_p0,
            atol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            chunk[p == 1],
            expected_p1,
            atol=1e-10,
            equal_nan=True,
        )


def test_global_partition_ewm_uses_timestamp_mean_once():
    """Multiple series in the same partition bucket at the same timestamp
    contribute their aggregate mean once to EWM, not once per row."""
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 5 + ["b"] * 5 + ["c"] * 5,
            "ds": list(range(5)) * 3,
            "y": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                12.0,
                22.0,
                32.0,
                42.0,
                52.0,
                14.0,
                24.0,
                34.0,
                44.0,
                54.0,
            ],
            "promo": [0] * 15,
        }
    )

    alpha = 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tfm = ExponentiallyWeightedMean(
            alpha=alpha, global_=True, partition_by=["promo"]
        )
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    result = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
    )
    col = tfm._get_name(1)
    vals = result[col].values

    # Per-timestamp means: [12, 22, 32, 42, 52]
    # EWM (lag=1, alpha=0.5):
    #   ord 0: NaN
    #   ord 1: 12.0
    #   ord 2: 0.5*22 + 0.5*12 = 17.0
    #   ord 3: 0.5*32 + 0.5*17 = 24.5
    #   ord 4: 0.5*42 + 0.5*24.5 = 33.25
    expected = [np.nan, 12.0, 17.0, 24.5, 33.25]

    # If each row contributed individually (3 rows at each timestamp):
    # ord 0 would consume 10→12→14 with three EWM steps, giving a different
    # final ewm at ord 0 that propagates differently. The expected values
    # above only hold when each timestamp contributes its mean once.
    for i in range(3):
        np.testing.assert_allclose(
            vals[i * 5 : (i + 1) * 5],
            expected,
            atol=1e-10,
            equal_nan=True,
        )


def test_partition_ewm_warning():
    """ExponentiallyWeightedMean emits a warning when partition_by is set,
    and does not warn without partition_by."""
    with pytest.warns(UserWarning, match="Partitioned EWM"):
        ExponentiallyWeightedMean(alpha=0.3, partition_by=["promo"])
    with pytest.warns(UserWarning, match="Partitioned EWM"):
        ExponentiallyWeightedMean(alpha=0.3, global_=True, partition_by=["promo"])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ExponentiallyWeightedMean(alpha=0.3)
        ExponentiallyWeightedMean(alpha=0.3, global_=True)
        ExponentiallyWeightedMean(alpha=0.3, groupby=["grp"])


# ---------------------------------------------------------------------------
# Null/NaN groupby key support.
#
# A missing (null/NaN/None) value in a groupby key column must collapse all
# missing values into a single bucket (SQL PARTITION BY semantics), identically
# across pandas/polars and key dtypes, without crashing fit/predict/update.
# ---------------------------------------------------------------------------
from mlforecast.pooled import (  # noqa: E402
    encode_keys,
    factorize,
    lookup,
)


def _keys(df, cols):
    """Key columns as numpy arrays, the form the pooled engine consumes."""
    return [np.asarray(df[c].to_numpy()) for c in cols]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("key_kind", ["numeric", "string", "categorical"])
def test_add_bucket_id_collapses_missing(engine, key_kind):
    if key_kind == "numeric":
        vals = [0.0, None, 0.0, 1.0, None]
    else:
        vals = ["a", None, "a", "b", None]
    cat = ["g"] if key_kind == "categorical" else None
    df = _make_df(engine, {"g": vals, "y": [1, 2, 3, 4, 5]}, categorical_cols=cat)
    bids, uniques = factorize(_keys(df, ["g"]))
    # rows 1 & 4 (missing) share a bucket; rows 0 & 2 (repeated value) share one;
    # row 3 (distinct value) is its own.
    assert bids[1] == bids[4]
    assert bids[0] == bids[2]
    assert len({int(bids[0]), int(bids[1]), int(bids[3])}) == 3
    assert len(uniques) == 3
    assert np.all(bids >= 0)


def test_polars_null_and_nan_collapse_to_one_bucket():
    # A polars float column holding BOTH null and NaN must collapse them into a
    # single missing bucket.
    df = pl.DataFrame(
        {
            "g": [0.0, float("nan"), None, 1.0, float("nan"), None],
            "y": list(range(6)),
        }
    )
    bids, uniques = factorize(_keys(df, ["g"]))
    assert len({int(bids[1]), int(bids[2]), int(bids[4]), int(bids[5])}) == 1
    assert bids[0] != bids[1] and bids[3] != bids[1]
    assert len(uniques) == 3  # 0.0, missing, 1.0


def test_null_nan_parity_across_engines():
    # pandas NaN and polars null/NaN must produce the same bucket structure.
    pdf = pd.DataFrame({"g": [0.0, np.nan, 0.0, 1.0], "y": [1, 2, 3, 4]})
    plf = pl.DataFrame({"g": [0.0, float("nan"), 0.0, 1.0], "y": [1, 2, 3, 4]})
    _, gp = factorize(_keys(pdf, ["g"]))
    _, gl = factorize(_keys(plf, ["g"]))
    assert len(gp) == len(gl) == 3


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_lookup_mixed_int_float(engine):
    # Fit data contaminated to float by a NaN; later clean integer keys must
    # still match the same buckets (no "0" vs "0.0" mismatch).
    fit = _make_df(engine, {"g": [0.0, 1.0, float("nan")], "y": [1, 2, 3]})
    _, uniques = factorize(_keys(fit, ["g"]))
    if engine == "polars":
        data = pl.DataFrame({"g": pl.Series([0, 1], dtype=pl.Int64)})
        nan_data = pl.DataFrame({"g": pl.Series([float("nan")], dtype=pl.Float64)})
    else:
        data = pd.DataFrame({"g": pd.Series([0, 1], dtype="int64")})
        nan_data = pd.DataFrame({"g": pd.Series([np.nan], dtype="float64")})
    bids = lookup(_keys(data, ["g"]), uniques)
    assert np.all(bids >= 0)
    assert bids[0] != bids[1]
    # a missing key in lookup data finds the existing missing bucket
    nan_bid = lookup(_keys(nan_data, ["g"]), uniques)
    assert nan_bid[0] >= 0
    assert nan_bid[0] not in set(bids.tolist())


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fractional_float_does_not_collide_with_int_bucket(engine):
    # Only *integral* floats reconcile to int buckets: a genuinely fractional
    # key 1.5 must NOT match an existing integer bucket 1.
    if engine == "polars":
        fit = pl.DataFrame({"g": pl.Series([1, 2], dtype=pl.Int64), "y": [1, 2]})
        q = pl.DataFrame({"g": pl.Series([1.5, 1.0, 2.0], dtype=pl.Float64)})
    else:
        fit = pd.DataFrame({"g": pd.Series([1, 2], dtype="int64"), "y": [1, 2]})
        q = pd.DataFrame({"g": pd.Series([1.5, 1.0, 2.0], dtype="float64")})
    _, uniques = factorize(_keys(fit, ["g"]))
    res = lookup(_keys(q, ["g"]), uniques)
    assert res[0] == -1  # 1.5 unmatched, not bucket for 1
    assert res[1] == 0  # 1.0 matches integer bucket for 1
    assert res[2] == 1  # 2.0 matches integer bucket for 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_large_int_keys_stay_distinct(engine):
    # Int keys must never be routed through float: two distinct integers above
    # 2**53 (not exactly representable as float64) stay in distinct buckets
    # through both creation and lookup.
    a, b = 2**53 + 1, 2**53 + 2
    if engine == "polars":
        df = pl.DataFrame({"g": pl.Series([a, b, a], dtype=pl.Int64), "y": [1, 2, 3]})
    else:
        df = pd.DataFrame({"g": pd.Series([a, b, a], dtype="int64"), "y": [1, 2, 3]})
    bids, uniques = factorize(_keys(df, ["g"]))
    assert bids[0] == bids[2] and bids[0] != bids[1]
    assert len(uniques) == 2
    look = lookup(_keys(df, ["g"]), uniques)
    assert look[0] != look[1]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_mixed_schema_reconcile_does_not_widen_int_side(engine):
    # An integral float lookup matches its integer bucket, while distinct large
    # integer buckets stay distinct (no precision collapse from float widening).
    a, b = 2**53 + 1, 2**53 + 2
    if engine == "polars":
        groups_src = pl.DataFrame(
            {"g": pl.Series([a, b, 1], dtype=pl.Int64), "y": [1, 2, 3]}
        )
        q = pl.DataFrame({"g": pl.Series([1.0], dtype=pl.Float64)})
    else:
        groups_src = pd.DataFrame(
            {"g": pd.Series([a, b, 1], dtype="int64"), "y": [1, 2, 3]}
        )
        q = pd.DataFrame({"g": pd.Series([1.0], dtype="float64")})
    bids, uniques = factorize(_keys(groups_src, ["g"]))
    assert len(uniques) == 3  # large ints not collapsed by any float widening
    look = lookup(_keys(q, ["g"]), uniques)
    assert look[0] == bids[2] and look[0] >= 0


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fractional_float_keys_stay_distinct(engine):
    # Close-but-distinct fractional floats each get their own bucket, repeated
    # values collapse, and a lookup recovers the same ids on both engines.
    vals = [1.0000000001, 1.0000000002, 0.1, 0.2, 0.25, 0.1, 1.0000000001]
    df = _make_df(engine, {"g": vals, "y": list(range(len(vals)))})
    bids, uniques = factorize(_keys(df, ["g"]))
    assert np.all(bids >= 0)
    assert len(uniques) == 5  # 5 distinct floats; the two repeats collapse
    assert bids[5] == bids[2]  # 0.1 repeat
    assert bids[6] == bids[0]  # 1.0000000001 repeat
    assert (
        len({int(bids[0]), int(bids[1]), int(bids[2]), int(bids[3]), int(bids[4])}) == 5
    )
    look = lookup(_keys(df, ["g"]), uniques)
    np.testing.assert_array_equal(look, bids)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_extend_does_not_duplicate_existing_missing_bucket(engine):
    # Growing the vocabulary reuses the pre-existing missing bucket and only
    # adds the genuinely new value.
    fit = _make_df(engine, {"g": ["x", None, "y"], "y": [1, 2, 3]})
    _, uniques = factorize(_keys(fit, ["g"]))
    upd = _make_df(engine, {"g": [None, "z"], "y": [9, 10]})
    grown = np.union1d(uniques, encode_keys(_keys(upd, ["g"])))
    assert len(grown) == len(uniques) + 1
    bids = lookup(_keys(upd, ["g"]), grown)
    assert np.all(bids >= 0)


def test_one_missing_column_keeps_distinct_buckets():
    # Multi-column key: (X, None) and (X, "n") must be distinct buckets, while
    # two (X, None) rows collapse together.
    df = pd.DataFrame(
        {
            "b": ["X", "X", "X", "Y"],
            "r": ["n", None, None, "s"],
            "y": [1, 2, 3, 4],
        }
    )
    bids, uniques = factorize(_keys(df, ["b", "r"]))
    assert bids[1] == bids[2]  # both (X, None)
    assert bids[0] != bids[1]  # (X, "n") distinct from (X, None)
    assert len(uniques) == 3


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_groupby_null_key_fit_predict_update(engine):
    from sklearn.linear_model import LinearRegression
    from mlforecast.forecast import MLForecast

    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 6 + ["b"] * 6 + ["c"] * 6,
            "ds": list(range(6)) * 3,
            "y": [float(i) for i in range(18)],
            "brand": ["x"] * 6 + [None] * 6 + [None] * 6,
        },
    )
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
    upd = _make_df(
        engine,
        {
            "unique_id": ["a", "b", "c"],
            "ds": [6, 6, 6],
            "y": [6.0, 7.0, 8.0],
            "brand": ["x", None, None],
        },
    )
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
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 3 + ["b"] * 3,
            "ds": [0, 1, 2] * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "brand": brand,
        },
    )
    ts = TimeSeries(
        freq=1, lag_transforms={1: [RollingMean(window_size=2, groupby=["brand"])]}
    )
    # must not raise
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=["brand"],
    )


def test_polars_join_preserves_row_order():
    """Bucket ids must line up positionally with the rows they came from.

    The engine consumes bucket ids as numpy arrays indexed against the caller's
    row order, so `factorize`/`lookup` must be strictly positional -- no sorting
    or joining that could permute rows.
    """
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(200)
    base = pl.DataFrame({"k": shuffled.tolist()})
    bids, uniques = factorize(_keys(base, ["k"]))
    # positional: each row's id must map back to that row's own key
    np.testing.assert_array_equal(uniques[bids], encode_keys(_keys(base, ["k"])))
    # and id i is the i-th smallest encoded key
    assert uniques.tolist() == sorted(uniques.tolist())
    key_to_bid = dict(zip(shuffled.tolist(), bids.tolist()))
    reversed_df = pl.DataFrame({"k": shuffled[::-1].tolist()})
    look = lookup(_keys(reversed_df, ["k"]), uniques)
    assert look.tolist() == [key_to_bid[k] for k in shuffled[::-1].tolist()]


def test_polars_shuffled_rows_feature_parity_with_pandas():
    """groupby+partition_by features on a shuffled-row polars frame match the
    pandas-engine run, at fit and at the first prediction step."""
    n_series, n_times = 40, 8
    rng = np.random.default_rng(1)
    ids = np.repeat([f"s{i:02d}" for i in range(n_series)], n_times)
    times = np.tile(range(n_times), n_series)
    y = rng.standard_normal(n_series * n_times)
    brand = np.repeat([f"b{i % 5}" for i in range(n_series)], n_times)
    promo = rng.integers(0, 2, n_series * n_times)
    order = rng.permutation(n_series * n_times)
    rows = {
        "unique_id": ids[order].tolist(),
        "ds": times[order].tolist(),
        "y": y[order].tolist(),
        "brand": brand[order].tolist(),
        "promo": promo[order].tolist(),
    }
    fit_outs, step_outs = [], []
    for engine in ["pandas", "polars"]:
        tfm = RollingMean(2, min_samples=1, groupby=["brand"], partition_by=["promo"])
        col = tfm._get_name(1)
        ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
        res = ts.fit_transform(
            _make_df(engine, rows),
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            dropna=False,
            static_features=[],
        )
        fit_outs.append(np.asarray(res[col], dtype=float))
        ts._predict_setup()
        features = ts._update_features()
        step_outs.append(np.asarray(features[col], dtype=float))
    np.testing.assert_allclose(fit_outs[0], fit_outs[1], atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(step_outs[0], step_outs[1], atol=1e-12, equal_nan=True)


def test_polars_shuffled_rows_slow_path_parity_with_pandas():
    """The bucket-feature join path (slow-path transforms go through
    TimeSeries._join_bucket_features) preserves row order on polars: a
    shuffled-row polars frame matches pandas positionally."""
    from mlforecast.lag_transforms import RollingQuantile

    n_series, n_times = 6, 10
    rng = np.random.default_rng(2)
    ids = np.repeat([f"s{i}" for i in range(n_series)], n_times)
    times = np.tile(range(n_times), n_series)
    y = rng.standard_normal(n_series * n_times)
    order = rng.permutation(n_series * n_times)
    rows = {
        "unique_id": ids[order].tolist(),
        "ds": times[order].tolist(),
        "y": y[order].tolist(),
    }
    outs = []
    for engine in ["pandas", "polars"]:
        tfm = RollingQuantile(0.5, 3, min_samples=1, global_=True)
        col = tfm._get_name(1)
        ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
        res = ts.fit_transform(
            _make_df(engine, rows),
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            dropna=False,
            static_features=[],
        )
        outs.append(np.asarray(res[col], dtype=float))
    np.testing.assert_allclose(outs[0], outs[1], atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_append_predictions_preserves_time_dtype(engine):
    """Recursive prediction must keep the shared calendar and channels intact.

    `main` stored real timestamps per bucket, so it could degrade datetime64 to
    object; the channel engine stores integer ordinals on one shared calendar,
    so the equivalent risk is the calendar or the float channels drifting.
    """
    n_series, n_times = 4, 8
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=n_times, freq="D")
    rows = {
        "unique_id": np.repeat([f"s{i}" for i in range(n_series)], n_times).tolist(),
        "ds": list(dates) * n_series,
        "y": rng.standard_normal(n_series * n_times).tolist(),
        "brand": np.repeat(["x", "x", "z", "z"], n_times).tolist(),
        "promo": np.tile([0, 0, 1, 1, 0, 1, 0, 1], n_series).tolist(),
    }
    tfms = [
        RollingMean(2, min_samples=1, global_=True),
        RollingMean(2, min_samples=1, groupby=["brand"], partition_by=["promo"]),
    ]
    freq = "1d" if engine == "polars" else "D"
    ts = TimeSeries(freq=freq, lag_transforms={1: tfms})
    ts.fit_transform(
        _make_df(engine, rows),
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
    )
    ordinals_before = {k: st.n_ordinals for k, st in ts._pooled_states.items()}
    ts._predict_setup()
    n_steps = 2
    for step in range(n_steps):
        features = ts._update_features()
        for tfm in tfms:
            vals = np.asarray(features[tfm._get_name(1)], dtype=float)
            assert not np.all(np.isnan(vals)), f"step {step}: all NaN"
        ts._update_y(np.ones(n_series))
    for key, state in ts._pooled_states.items():
        assert state.n_ordinals == ordinals_before[key] + n_steps, key
        for name, arr in state.base.items():
            assert arr.dtype == np.float64, (key, name)
            assert arr.shape[0] == state.n_buckets, (key, name)


def _diffed_range_mean_oracle(
    ids, times, diffs, promos, qid, qt, qpromo, scope, lag=1, window=2
):
    """Expected RANGE rolling mean over a differenced target.

    Fixtures use a contiguous integer calendar shared by all series, so parent
    ordinals equal the timestamps for both the global and the per-series
    (local) parent calendars."""
    lo, hi = qt - lag - window + 1, qt - lag
    mask = (times >= lo) & (times <= hi)
    if scope == "global":
        pass  # one bucket: every series
    else:  # local + partition_by: bucket is (series, promo value)
        mask &= (ids == qid) & (promos == qpromo)
    vals = diffs[mask]
    vals = vals[~np.isnan(vals)]
    return float(np.mean(vals)) if len(vals) else np.nan


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_target_transforms_with_pooled_preprocess(engine):
    """Pooled states must be built on the *transformed* target when
    target_transforms are configured (the df_for_pooled plumbing in
    TimeSeries._fit)."""
    from mlforecast import MLForecast
    from mlforecast.target_transforms import Differences
    from sklearn.linear_model import LinearRegression

    n_series, n_times = 2, 12
    rng = np.random.default_rng(8)
    ids = np.repeat(["a", "b"], n_times)
    times = np.tile(np.arange(n_times), n_series)
    y = rng.standard_normal(n_series * n_times).cumsum()
    promos = np.tile([0, 1], n_series * n_times // 2)
    df = _make_df(
        engine,
        {
            "unique_id": ids.tolist(),
            "ds": times.tolist(),
            "y": y.tolist(),
            "promo": promos.tolist(),
        },
    )
    g_tfm = RollingMean(2, min_samples=1, global_=True)
    p_tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        target_transforms=[Differences([1])],
        lag_transforms={1: [g_tfm, p_tfm]},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prep = fcst.preprocess(df, static_features=[], dropna=False, keep_last_n=10_000)
    if engine == "polars":
        prep = prep.to_pandas()

    diffs = pd.Series(y).groupby(ids).diff().to_numpy()
    for scope, tfm in [("global", g_tfm), ("local", p_tfm)]:
        col = tfm._get_name(1)
        expected = np.array(
            [
                _diffed_range_mean_oracle(
                    ids,
                    times,
                    diffs,
                    promos,
                    row.unique_id,
                    row.ds,
                    row.promo,
                    scope,
                )
                for row in prep.itertuples()
            ]
        )
        np.testing.assert_allclose(
            prep[col].to_numpy(),
            expected,
            atol=1e-12,
            equal_nan=True,
        )

    # the pooled states' target must be the differenced values
    for state in fcst.ts._pooled_states.values():
        # the cell store aggregates the transformed target, so its totals must
        # match the differenced values that survived the transform
        finite = diffs[~np.isnan(diffs)]
        assert state.base["count"].sum() == len(finite)
        np.testing.assert_allclose(
            state.base["sum"].sum(),
            finite.sum(),
            atol=1e-12,
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_target_transforms_with_pooled_predict(engine):
    """Recursive predict with Differences + pooled transforms: a linear target
    has a constant differenced series, so predictions have a closed form
    y(T+k) = y(T) + k*slope."""
    from mlforecast import MLForecast
    from mlforecast.target_transforms import Differences
    from sklearn.linear_model import LinearRegression

    n_times, slope = 12, 3.0
    ids = np.repeat(["a", "b"], n_times)
    times = np.tile(np.arange(n_times), 2)
    y = np.concatenate(
        [10.0 + slope * np.arange(n_times), 100.0 + slope * np.arange(n_times)]
    )
    promos = np.tile([0, 1], n_times)
    df = _make_df(
        engine,
        {
            "unique_id": ids.tolist(),
            "ds": times.tolist(),
            "y": y.tolist(),
            "promo": promos.tolist(),
        },
    )
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        target_transforms=[Differences([1])],
        lag_transforms={
            1: [
                RollingMean(2, min_samples=1, global_=True),
                RollingMean(2, min_samples=1, partition_by=["promo"]),
            ]
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fcst.fit(df, static_features=[], dropna=True)
        # promo parity matches the training pattern so each step's partition
        # window is non-empty
        X_df = _make_df(
            engine,
            {
                "unique_id": ["a", "a", "b", "b"],
                "ds": [12, 13, 12, 13],
                "promo": [0, 1, 0, 1],
            },
        )
        preds = fcst.predict(h=2, X_df=X_df)
    if engine == "polars":
        preds = preds.to_pandas()
    preds = preds.sort_values(["unique_id", "ds"])
    expected = np.array(
        [
            y[n_times - 1] + slope * np.array([1, 2]),
            y[2 * n_times - 1] + slope * np.array([1, 2]),
        ]
    ).ravel()
    np.testing.assert_allclose(
        preds["LinearRegression"].to_numpy(),
        expected,
        rtol=1e-6,
    )


def _range_quantile_oracle(
    hist, qid, qt, qpromo, mode, p=0.5, lag=1, window=3, min_samples=1
):
    """Expected RANGE rolling quantile per bucket. ``hist`` is a list of
    (id, t, y, promo) rows on a contiguous integer calendar (ordinals == t)."""
    lo, hi = qt - lag - window + 1, qt - lag
    if mode == "global":  # bucket key is (promo,)
        vals = [r[2] for r in hist if lo <= r[1] <= hi and r[3] == qpromo]
    else:  # local: bucket key is (id, promo)
        vals = [
            r[2] for r in hist if lo <= r[1] <= hi and r[0] == qid and r[3] == qpromo
        ]
    vals = [v for v in vals if not np.isnan(v)]
    if len(vals) < max(min_samples, 1):
        return np.nan
    return float(np.quantile(vals, p))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("mode", ["local", "global"])
def test_slow_path_quantile_with_partition_by(engine, mode):
    """RollingQuantile has no aggregate fast path, so partition_by routes it
    through the row-level slow path at fit and through build_query_arrays at
    predict. Pin both against a RANGE-window oracle."""
    from mlforecast.lag_transforms import RollingQuantile

    n_series, n_times = 3, 10
    rng = np.random.default_rng(11)
    ids = np.repeat([f"s{i}" for i in range(n_series)], n_times)
    times = np.tile(np.arange(n_times), n_series)
    y = rng.standard_normal(n_series * n_times)
    promos = np.tile([0, 1, 0, 0, 1, 1, 0, 1, 0, 1], n_series)
    df = _make_df(
        engine,
        {
            "unique_id": ids.tolist(),
            "ds": times.tolist(),
            "y": y.tolist(),
            "promo": promos.tolist(),
        },
    )
    tfm = RollingQuantile(
        0.5,
        3,
        min_samples=1,
        global_=(mode == "global"),
        partition_by=["promo"],
    )
    col = tfm._get_name(1)
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    res = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
    )
    hist = list(zip(ids, times, y, promos))
    expected_fit = np.array(
        [_range_quantile_oracle(hist, i, t, pr, mode) for i, t, _, pr in hist]
    )
    np.testing.assert_allclose(
        np.asarray(res[col], dtype=float),
        expected_fit,
        atol=1e-12,
        equal_nan=True,
    )

    # two recursive steps through the build_query_arrays slow predict path;
    # predict() hands _predict_recursive an X_df sorted by (id, time) with the
    # id/time columns dropped, so mimic that shape here
    uid_order = [f"s{i}" for i in range(n_series)]
    step_promos = {10: [0, 0, 1], 11: [1, 0, 1]}
    X_df = _make_df(
        engine,
        {
            "promo": [
                v
                for i in range(n_series)
                for v in (step_promos[10][i], step_promos[11][i])
            ],
        },
    )
    ts._predict_setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for step, t_query in enumerate([10, 11]):
            new_x = ts._get_features_for_next_step(X_df)
            vals = np.asarray(new_x[col], dtype=float)
            expected = np.array(
                [
                    _range_quantile_oracle(
                        hist,
                        uid,
                        t_query,
                        step_promos[t_query][i],
                        mode,
                    )
                    for i, uid in enumerate(uid_order)
                ]
            )
            np.testing.assert_allclose(
                vals,
                expected,
                atol=1e-12,
                equal_nan=True,
            )
            fake_preds = np.arange(n_series, dtype=float) + 10 * (step + 1)
            ts._update_y(fake_preds)
            hist.extend(
                (uid, t_query, fake_preds[i], step_promos[t_query][i])
                for i, uid in enumerate(uid_order)
            )


# ---------------------------------------------------------------------------
# time_agg: pre-aggregate rows sharing a timestamp within each bucket, then
# apply the transform over that per-timestamp series.
# ---------------------------------------------------------------------------
import operator as _operator  # noqa: E402

from sklearn.base import clone as _sk_clone  # noqa: E402

from mlforecast.lag_transforms import (  # noqa: E402
    Combine,
    ExpandingQuantile,
    Offset,
    RollingQuantile,
    SeasonalRollingMean,
)
from mlforecast.pooled import (  # noqa: E402
    _build_cells,
    _collapse,
    _view_value,
)

_BASE = ("count", "sum", "sumsq", "min", "max")


def _one_bucket_cells():
    # ord 0: [1, 3]; ord 1: [nan] (all-NaN); ord 2: [5]
    bid = np.array([0, 0, 0, 0])
    ordv = np.array([0, 0, 1, 2])
    y = np.array([1.0, 3.0, np.nan, 5.0])
    return bid, ordv, y, _build_cells(bid, ordv, y, 1, 3, _BASE)


def test_reaggregate_ts_aggs_values_and_nan():
    """Collapsing to one value per timestamp, with an all-NaN timestamp present.

    Unobserved cells (``count == 0``) carry a fill the kernels mask out, so only
    the observed cells are asserted on.
    """
    _, _, _, cells = _one_bucket_cells()
    expected = {
        "sum": [4.0, 5.0],
        "count": [2.0, 1.0],
        "mean": [2.0, 5.0],
        "min": [1.0, 5.0],
        "max": [3.0, 5.0],
    }
    for agg, vals in expected.items():
        r = _collapse(*_view_value(cells, agg), shift=0.0)
        obs = r["count"][0] > 0
        if agg == "count":
            # COUNT over an all-NULL group is 0, which is still an observation
            np.testing.assert_allclose(r["sum"][0][[0, 2]], vals)
        else:
            np.testing.assert_array_equal(obs, [True, False, True])
            np.testing.assert_allclose(r["sum"][0][obs], vals)
            np.testing.assert_allclose(r["sumsq"][0][obs], np.square(vals))
            np.testing.assert_allclose(r["min"][0][obs], vals)
            np.testing.assert_allclose(r["max"][0][obs], vals)


def test_reaggregate_ts_aggs_does_not_mutate_input():
    """Collapsed views are derived, so the shared base store must be untouched."""
    _, _, _, cells = _one_bucket_cells()
    before = {k: v.copy() for k, v in cells.items()}
    for agg in ("sum", "count", "mean", "min", "max"):
        _collapse(*_view_value(cells, agg), shift=0.0)
    for k, arr in before.items():
        np.testing.assert_allclose(cells[k], arr, equal_nan=True)


@pytest.mark.parametrize("time_agg", ["sum", "count", "mean", "min", "max"])
def test_collapse_matches_reaggregate(time_agg):
    """Collapse-then-aggregate == aggregate-then-collapse.

    This is the invariant that lets `PooledState` keep one base store per bucket
    key and derive each `time_agg` as a cached view instead of re-aggregating the
    panel per `time_agg`. It covers the ``sumsq`` centre too: both sides take
    it from the bucket's first observed cell.
    """
    rng = np.random.default_rng(0)
    bid = np.repeat([0, 1], 10)
    ordv = np.tile([0, 0, 1, 2, 2, 2, 3, 4, 4, 5], 2)
    y = rng.standard_normal(20)
    y[3] = np.nan  # partial-NaN and all-NaN timestamps
    y[7] = np.nan
    n_buckets, width = 2, 6
    base = _build_cells(bid, ordv, y, n_buckets, width, _BASE)
    v, obs = _view_value(base, time_agg)
    first = v[np.arange(n_buckets), obs.argmax(axis=1)][:, None]
    direct = _collapse(v, obs, first)

    # the same collapse done by hand on the rows, then aggregated
    cb, co = np.nonzero(obs)
    cy = v[obs]
    via_rows = _build_cells(cb, co, cy, n_buckets, width, _BASE)
    for field in ("count", "sum", "sumsq"):
        np.testing.assert_allclose(
            direct[field][obs], via_rows[field][obs], err_msg=f"{time_agg} {field}"
        )
    np.testing.assert_allclose(direct["min"][obs], via_rows["min"][obs])
    np.testing.assert_allclose(direct["max"][obs], via_rows["max"][obs])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_time_agg_sum_literal(engine):
    """Rolling mean of daily sums: hand-computed preprocess + one-step predict."""
    y_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y_b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    # daily sums s = [11, 22, 33, 44, 55, 66]; window=3, lag=1, min_samples=3
    tfms = [RollingMean(window_size=3, groupby=["grp"], time_agg="sum")]
    out = _fit_and_collect(engine, 1, tfms, y_a, y_b, 6, grp="X")
    col = "groupby_grp_rolling_mean_lag1_window_size3_time_aggsum"
    np.testing.assert_allclose(
        out[col]["preprocess"],
        [np.nan, np.nan, np.nan, 22.0, 33.0, 44.0],
        equal_nan=True,
    )
    # predict returns one value per series; both a and b are in group X -> 55
    np.testing.assert_allclose(out[col]["predict"], [55.0, 55.0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_time_agg_mean_differs_from_sum_and_rowpooled(engine):
    y_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y_b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    tfms = [
        RollingMean(window_size=3, groupby=["grp"], time_agg="mean"),
        RollingMean(window_size=3, groupby=["grp"]),  # row-pooled reference
    ]
    out = _fit_and_collect(engine, 1, tfms, y_a, y_b, 6, grp="X")
    mean_col = "groupby_grp_rolling_mean_lag1_window_size3_time_aggmean"
    row_col = "groupby_grp_rolling_mean_lag1_window_size3"
    # daily means = [5.5, 11, 16.5, 22, 27.5, 33]
    np.testing.assert_allclose(
        out[mean_col]["preprocess"],
        [np.nan, np.nan, np.nan, 11.0, 16.5, 22.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(out[mean_col]["predict"], [27.5, 27.5])
    # time_agg='mean' differs from row-pooled: min_samples counts timestamps
    # (2 < 3 at k=2 -> NaN), whereas row-pooling counts rows (4 >= 3 -> 8.25).
    row_pre = out[row_col]["preprocess"]
    mean_pre = out[mean_col]["preprocess"]
    assert np.isnan(mean_pre[2]) and np.isfinite(row_pre[2])
    # where both are defined (k>=3) the values agree for this balanced bucket
    np.testing.assert_allclose(row_pre[3:], mean_pre[3:])


_TIME_AGG_FACTORIES = [
    (lambda m: RollingMean(window_size=4, **m), "RollingMean"),
    (lambda m: RollingStd(window_size=4, **m), "RollingStd"),
    (lambda m: RollingMin(window_size=4, **m), "RollingMin"),
    (lambda m: RollingMax(window_size=4, **m), "RollingMax"),
    (lambda m: ExpandingMean(**m), "ExpandingMean"),
    (lambda m: ExpandingStd(**m), "ExpandingStd"),
    (lambda m: ExpandingMin(**m), "ExpandingMin"),
    (lambda m: ExpandingMax(**m), "ExpandingMax"),
    (lambda m: ExponentiallyWeightedMean(alpha=0.3, **m), "EWM"),
]


@pytest.mark.parametrize(
    "tfm_factory",
    [f[0] for f in _TIME_AGG_FACTORIES],
    ids=[f[1] for f in _TIME_AGG_FACTORIES],
)
@pytest.mark.parametrize("time_agg", ["sum", "count", "mean", "min", "max"])
@pytest.mark.parametrize("lag", _LAGS)
def test_fast_vs_slow_time_agg(tfm_factory, time_agg, lag):
    """Same invariant across every time_agg -- the case the shared store exists for."""
    _assert_view_matches_direct(tfm_factory, lag, {"global_": True}, time_agg)
    _assert_view_matches_direct(tfm_factory, lag, {"groupby": ["grp"]}, time_agg)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_time_agg_multistep_predict(engine):
    """Multi-step recursive predict stays finite and reproducible with time_agg."""
    from sklearn.linear_model import LinearRegression
    from mlforecast import MLForecast

    rng = np.random.default_rng(3)
    rows = []
    for uid in ["a", "b", "c"]:
        grp = "X" if uid != "c" else "Y"
        for t in range(15):
            rows.append((uid, t, float(rng.standard_normal() + 5), grp))
    df = _make_df(
        engine,
        {
            "unique_id": [r[0] for r in rows],
            "ds": [r[1] for r in rows],
            "y": [r[2] for r in rows],
            "grp": [r[3] for r in rows],
        },
    )
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        lags=[1],
        lag_transforms={
            1: [RollingMean(window_size=3, groupby=["grp"], time_agg="sum")]
        },
    )
    fcst.fit(df, static_features=["grp"])
    p1 = fcst.predict(4)
    p2 = fcst.predict(4)
    v1 = (
        p1["LinearRegression"].to_numpy()
        if engine == "polars"
        else p1["LinearRegression"].values
    )
    v2 = (
        p2["LinearRegression"].to_numpy()
        if engine == "polars"
        else p2["LinearRegression"].values
    )
    assert np.all(np.isfinite(v1))
    np.testing.assert_allclose(v1, v2)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_ewm_time_agg_mean_is_noop(engine):
    """Pooled EWM uses bucket means by default, so explicit time_agg='mean'
    must match the default contract exactly."""
    y_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y_b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    out_explicit = _fit_and_collect(
        engine,
        1,
        [ExponentiallyWeightedMean(alpha=0.5, groupby=["grp"], time_agg="mean")],
        y_a,
        y_b,
        6,
        grp="X",
    )
    out_default = _fit_and_collect(
        engine,
        1,
        [ExponentiallyWeightedMean(alpha=0.5, groupby=["grp"])],
        y_a,
        y_b,
        6,
        grp="X",
    )
    col = "groupby_grp_exponentially_weighted_mean_lag1_alpha0.5"
    np.testing.assert_allclose(
        out_explicit[col]["preprocess"],
        out_default[col]["preprocess"],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        out_explicit[col]["predict"], out_default[col]["predict"]
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_time_agg_quantile_slow_path_literal(engine):
    """RollingQuantile has no fast path: time_agg goes through row-collapse."""
    y_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y_b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    # daily sums [11,22,33,44,55,66]; median over window=3 == middle value
    tfms = [RollingQuantile(p=0.5, window_size=3, groupby=["grp"], time_agg="sum")]
    out = _fit_and_collect(engine, 1, tfms, y_a, y_b, 6, grp="X")
    col = "groupby_grp_rolling_quantile_lag1_p0.5_window_size3_time_aggsum"
    np.testing.assert_allclose(
        out[col]["preprocess"],
        [np.nan, np.nan, np.nan, 22.0, 33.0, 44.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(out[col]["predict"], [55.0, 55.0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_time_agg_seasonal_slow_path_literal(engine):
    y_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y_b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    # daily sums [11,22,33,44,55,66]; season_length=2, window=2, lag=1
    tfms = [
        SeasonalRollingMean(
            season_length=2, window_size=2, groupby=["grp"], time_agg="sum"
        )
    ]
    out = _fit_and_collect(engine, 1, tfms, y_a, y_b, 6, grp="X")
    col = (
        "groupby_grp_seasonal_rolling_mean_lag1_season_length2_window_size2_time_aggsum"
    )
    np.testing.assert_allclose(
        out[col]["preprocess"],
        [np.nan, np.nan, np.nan, 22.0, 33.0, 44.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(out[col]["predict"], [55.0, 55.0])


def test_time_agg_min_samples_counts_timestamps():
    """min_samples counts observed timestamps, not rows, when time_agg is set."""
    y_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y_b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    # window covers 3 timestamps; each timestamp has 2 rows (row count would be 6).
    # min_samples=4 > 3 timestamps -> all NaN; without time_agg row-count 6 >= 4.
    tfms = [RollingMean(window_size=3, min_samples=4, global_=True, time_agg="sum")]
    out = _fit_and_collect("pandas", 1, tfms, y_a, y_b, 6)
    col = "global_rolling_mean_lag1_window_size3_min_samples4_time_aggsum"
    assert np.all(np.isnan(out[col]["preprocess"]))
    # min_samples=3 == 3 timestamps -> produces values
    tfms2 = [RollingMean(window_size=3, min_samples=3, global_=True, time_agg="sum")]
    out2 = _fit_and_collect("pandas", 1, tfms2, y_a, y_b, 6)
    col2 = "global_rolling_mean_lag1_window_size3_min_samples3_time_aggsum"
    assert np.any(~np.isnan(out2[col2]["preprocess"]))


@pytest.mark.parametrize(
    "factory",
    [
        lambda **k: RollingMean(window_size=3, **k),
        lambda **k: ExpandingMean(**k),
        lambda **k: ExponentiallyWeightedMean(alpha=0.5, **k),
        lambda **k: SeasonalRollingMean(season_length=2, window_size=2, **k),
        lambda **k: RollingQuantile(p=0.5, window_size=3, **k),
        lambda **k: ExpandingQuantile(p=0.5, **k),
    ],
)
def test_time_agg_validation_errors(factory):
    # bad agg name
    with pytest.raises(ValueError, match="time_agg must be one of"):
        factory(global_=True, time_agg="median")
    # requires a pooled scope
    with pytest.raises(ValueError, match="time_agg requires"):
        factory(time_agg="sum")
    # partition_by alone is still local -> rejected
    with pytest.raises(ValueError, match="time_agg requires"):
        factory(partition_by=["promo"], time_agg="sum")
    # accepted combinations (ignore the unrelated partitioned-EWM warning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        factory(global_=True, time_agg="sum")
        factory(groupby=["g"], time_agg="sum")
        factory(global_=True, partition_by=["promo"], time_agg="sum")
        factory(groupby=["g"], partition_by=["promo"], time_agg="sum")


def test_ewm_time_agg_none_rejected():
    with pytest.raises(ValueError, match="does not accept time_agg=None"):
        ExponentiallyWeightedMean(alpha=0.5, time_agg=None)


def test_time_agg_min_samples_zero_warning_still_fires():
    with pytest.warns(UserWarning, match="min_samples=0"):
        RollingMean(window_size=3, min_samples=0, global_=True, time_agg="sum")


def test_time_agg_feature_name():
    assert "time_aggsum" in RollingMean(
        window_size=3, groupby=["cat"], time_agg="sum"
    )._get_name(7)
    # default None keeps names unchanged
    assert "time_agg" not in RollingMean(window_size=3, groupby=["cat"])._get_name(7)
    assert "time_agg" not in ExponentiallyWeightedMean(
        alpha=0.5, groupby=["cat"]
    )._get_name(7)
    assert "time_agg" not in ExponentiallyWeightedMean(
        alpha=0.5, groupby=["cat"], time_agg="mean"
    )._get_name(7)
    assert "time_aggsum" in ExponentiallyWeightedMean(
        alpha=0.5, groupby=["cat"], time_agg="sum"
    )._get_name(7)


def test_time_agg_offset_delegates_to_inner():
    inner = RollingMean(window_size=3, global_=True, time_agg="sum")
    off = Offset(inner, 1)
    # the wrapper doesn't mirror time_agg (its hooks delegate to the inner
    # transform, which applies its own re-aggregation), but the feature name
    # still carries it
    assert off.time_agg is None
    assert off.tfm.time_agg == "sum"
    assert "time_aggsum" in off._get_name(1)


def test_offset_effective_lag_must_be_positive():
    with pytest.raises(ValueError, match="effective lag"):
        Offset(RollingMean(window_size=2), -1)._set_core_tfm(1)
    with pytest.raises(ValueError, match="effective lag"):
        Offset(
            RollingMean(window_size=2, global_=True, time_agg="count"), -2
        )._set_core_tfm(2)
    # a negative shift is fine while the effective lag stays >= 1
    off = Offset(RollingMean(window_size=2), -1)._set_core_tfm(3)
    assert off._core_tfm.lag == 2


def test_ewm_time_agg_mean_skips_reaggregation():
    """Collapsed views are cached, and the uncollapsed one is the base store itself.

    EWM's native rule is time_agg="mean", so it reads a derived view on every
    step; that view must be built once and reused rather than recomputed.
    """
    from mlforecast.pooled import _build_cells, PooledState

    bid = np.array([0, 0, 0])
    ordv = np.array([0, 1, 2])
    y = np.array([1.0, 2.0, 3.0])
    state = PooledState(
        mode="global",
        group_cols=[],
        partition_cols=[],
        n_buckets=1,
        n_ordinals=3,
        base=_build_cells(bid, ordv, y, 1, 3, ("count", "sum", "sumsq", "min", "max")),
        series_bucket_id=np.zeros(1, dtype=np.int64),
    )
    assert ExponentiallyWeightedMean(alpha=0.5, groupby=["grp"]).time_agg == "mean"
    # no collapse -> the base store itself, not a copy
    assert state.channels(None) is state.base
    # a collapse is derived once and cached
    assert state.channels("mean") is state.channels("mean")
    assert state.channels("mean") is not state.base
    assert state.channels("sum") is not state.channels("mean")


def test_time_agg_combine_mixed():
    t1 = RollingMean(window_size=3, groupby=["grp"], time_agg="sum")
    t2 = RollingMean(window_size=3, groupby=["grp"], time_agg="mean")
    c = Combine(t1, t2, _operator.truediv)
    name = c._get_name(1)
    assert "time_aggsum" in name and "time_aggmean" in name

    y_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y_b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    out = _fit_and_collect("pandas", 1, [c], y_a, y_b, 6, grp="X")
    # sum / mean over a balanced 2-series bucket == 2 everywhere it's defined
    pre = out[name]["preprocess"]
    finite = pre[~np.isnan(pre)]
    np.testing.assert_allclose(finite, np.full(len(finite), 2.0))


def test_time_agg_sklearn_clone_roundtrip():
    tfm = RollingMean(window_size=3, global_=True, time_agg="sum")
    assert tfm.get_params()["time_agg"] == "sum"
    cloned = _sk_clone(tfm)
    assert cloned.time_agg == "sum"
    assert cloned._get_name(1) == tfm._get_name(1)


# === min_samples default resolution ===
# In local partition mode min_samples=None defaults to 1 (SQL RANGE semantics);
# every other mode keeps the window_size default.

from mlforecast.pooled import get_kernel as _get_kernel  # noqa: E402


def _resolve_min_samples(tfm):
    """Effective min_samples, as the pooled kernel resolves it."""
    return _get_kernel(copy.deepcopy(tfm)._set_core_tfm(1)).min_samples()


def test_min_samples_default_resolution():
    assert _resolve_min_samples(RollingMean(7)) == 7
    assert _resolve_min_samples(RollingMean(7, global_=True)) == 7
    assert _resolve_min_samples(RollingMean(7, groupby=["brand"])) == 7
    assert _resolve_min_samples(RollingMean(7, partition_by=["promo"])) == 1
    assert (
        _resolve_min_samples(RollingMean(7, global_=True, partition_by=["promo"])) == 7
    )
    assert (
        _resolve_min_samples(RollingMean(7, groupby=["brand"], partition_by=["promo"]))
        == 7
    )
    # explicit values are never overridden
    assert (
        _resolve_min_samples(RollingMean(7, min_samples=3, partition_by=["promo"])) == 3
    )
    assert (
        _resolve_min_samples(
            SeasonalRollingMean(season_length=7, window_size=4, partition_by=["promo"])
        )
        == 1
    )
    assert (
        _resolve_min_samples(
            SeasonalRollingMean(season_length=7, window_size=4, global_=True)
        )
        == 4
    )


def _fit_transform_values(engine, df, tfm):
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    out = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=[],
    )
    if engine == "polars":
        out = out.to_pandas()
    return out[tfm._get_name(1)].to_numpy(dtype=float)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_local_default_min_samples_is_one(engine):
    """On a dense panel with interleaved promo days, a 7-step window almost
    never holds 7 same-promo observations (the window spans calendar steps
    while only same-promo rows count), so the window_size default would make
    the feature ~100% NaN. Local partition mode defaults to 1 instead."""
    n = 30
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * n,
            "ds": list(range(1, n + 1)),
            "y": [float(i) for i in range(1, n + 1)],
            "promo": [0, 1] * (n // 2),
        },
    )
    default_vals = _fit_transform_values(
        engine, df, RollingMean(7, partition_by=["promo"])
    )
    explicit_one = _fit_transform_values(
        engine, df, RollingMean(7, min_samples=1, partition_by=["promo"])
    )
    np.testing.assert_array_equal(default_vals, explicit_one)
    # usable feature: only the empty-lookback rows at the start are NaN
    assert np.isnan(default_vals).mean() < 0.2
    # the local-mode default (window_size) would have produced all NaN here
    old_default = _fit_transform_values(
        engine, df, RollingMean(7, min_samples=7, partition_by=["promo"])
    )
    assert np.isnan(old_default).all()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_nonlocal_default_min_samples_unchanged(engine):
    """global_ + partition_by keeps the window_size default (counts sum
    across series in the (partition) bucket)."""
    n = 12
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * n + ["b"] * n,
            "ds": list(range(1, n + 1)) * 2,
            "y": [float(i) for i in range(1, 2 * n + 1)],
            "promo": [0, 1] * (n // 2) * 2,
        },
    )
    default_vals = _fit_transform_values(
        engine, df, RollingMean(4, global_=True, partition_by=["promo"])
    )
    explicit_ws = _fit_transform_values(
        engine,
        df,
        RollingMean(4, min_samples=4, global_=True, partition_by=["promo"]),
    )
    explicit_one = _fit_transform_values(
        engine,
        df,
        RollingMean(4, min_samples=1, global_=True, partition_by=["promo"]),
    )
    np.testing.assert_array_equal(default_vals, explicit_ws)
    # the guard still bites on partially-filled windows, unlike min_samples=1
    assert np.isnan(default_vals).sum() > np.isnan(explicit_one).sum()


def test_lookup_lag_requires_partition_by():
    with pytest.raises(ValueError, match="LookupLag requires `partition_by`"):
        LookupLag()
    with pytest.raises(ValueError, match="LookupLag requires `partition_by`"):
        LookupLag(partition_by=None)
    with pytest.raises(ValueError, match="LookupLag requires `partition_by`"):
        LookupLag(partition_by=[])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_lookup_lag_predict_uses_previous_occurrence(engine):
    """Predict looks up the previous matching-occurrence target via X_df."""
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 10,
            "ds": list(range(1, 11)),
            "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "promo": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # promo=1 at ds 3,6,9 (y 30,60,90)
        },
    )
    tfm = LookupLag(partition_by=["promo"])
    col = tfm._get_name(1)
    captured = []

    def save_features(x):
        captured.append(x[col].to_numpy().copy())
        return x

    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lags=[1],
        lag_transforms={1: [tfm]},
    )
    fcst.fit(df, id_col="unique_id", time_col="ds", target_col="y", static_features=[])
    future_df = _make_df(engine, {"unique_id": ["a"], "ds": [11], "promo": [1]})
    preds = fcst.predict(h=1, X_df=future_df, before_predict_callback=save_features)
    assert len(preds) == 1
    # future step is promo=1; the previous promo occurrence is ds=9 (y=90)
    np.testing.assert_allclose(captured[0][0], 90.0)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_lookup_lag_predict_multistep_transformed(engine):
    """Multi-step predict with dynamic keys stays on the transformed scale."""
    from mlforecast.forecast import MLForecast
    from mlforecast.target_transforms import LocalStandardScaler
    from sklearn.ensemble import HistGradientBoostingRegressor

    rows = {
        "unique_id": ["a"] * 10,
        "ds": list(range(1, 11)),
        "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "promo": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    }
    tfm = LookupLag(partition_by=["promo"])
    col = tfm._get_name(1)

    # Expected transformed-target values via a separate preprocess pass.
    fexp = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lags=[1],
        lag_transforms={1: [LookupLag(partition_by=["promo"])]},
        target_transforms=[LocalStandardScaler()],
    )
    prep = fexp.preprocess(_make_df(engine, rows), static_features=[], dropna=False)
    prep_pd = prep.to_pandas() if engine == "polars" else prep
    exp_promo1 = prep_pd[prep_pd["promo"] == 1]["y"].to_numpy()[-1]  # ds=9 transformed
    exp_promo0 = prep_pd[prep_pd["promo"] == 0]["y"].to_numpy()[-1]  # ds=10 transformed

    captured = []

    def save_features(x):
        captured.append(x[col].to_numpy().copy())
        return x

    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lags=[1],
        lag_transforms={1: [tfm]},
        target_transforms=[LocalStandardScaler()],
    )
    fcst.fit(
        _make_df(engine, rows),
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[],
    )
    future_df = _make_df(
        engine,
        {
            "unique_id": ["a", "a"],
            "ds": [11, 12],
            "promo": [1, 0],
        },
    )
    preds = fcst.predict(h=2, X_df=future_df, before_predict_callback=save_features)
    assert len(preds) == 2
    # step 0 (ds=11, promo=1): previous promo=1 occurrence, transformed scale
    np.testing.assert_allclose(captured[0][0], exp_promo1)
    # step 1 (ds=12, promo=0): bucket (a,0) unchanged by ds=11 (which was promo=1)
    np.testing.assert_allclose(captured[1][0], exp_promo0)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_lookup_lag_predict_survives_keep_last_n_trim(engine):
    """LookupLag reaches a far-back occurrence even under an aggressive
    keep_last_n (its pooled state is not finite-window, so it is not trimmed)."""
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * 12,
            "ds": list(range(1, 13)),
            "y": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                90.0,
                100.0,
                110.0,
                120.0,
            ],
            "promo": [
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # single promo far back (ds=3, y=30)
        },
    )
    tfm = LookupLag(partition_by=["promo"])
    col = tfm._get_name(1)
    captured = []

    def save_features(x):
        captured.append(x[col].to_numpy().copy())
        return x

    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lags=[1],
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[],
        keep_last_n=2,
    )
    future_df = _make_df(engine, {"unique_id": ["a"], "ds": [13], "promo": [1]})
    preds = fcst.predict(h=1, X_df=future_df, before_predict_callback=save_features)
    assert len(preds) == 1
    # future promo step reaches the far-back promo occurrence (ds=3, y=30), not NaN
    np.testing.assert_allclose(captured[0][0], 30.0)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", [1, 2, 7, 52])
def test_lookup_lag_predict_various_lags(engine, lag):
    """Predict looks up the correct occurrence across a range of lag values
    (exercises the _compute_latest_from_aggs fast path and its [-lag] indexing)."""
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    n = 120
    df = _make_df(
        engine,
        {
            "unique_id": ["a"] * n,
            "ds": list(range(n)),
            "y": [float(i) for i in range(n)],  # y == ds index
            "promo": [1] * n,  # one partition value → bucket is the whole series
        },
    )
    tfm = LookupLag(partition_by=["promo"])
    col = tfm._get_name(lag)
    captured = []

    def save_features(x):
        captured.append(x[col].to_numpy().copy())
        return x

    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=5)],
        freq=1,
        lags=[1],
        lag_transforms={lag: [tfm]},
    )
    fcst.fit(df, id_col="unique_id", time_col="ds", target_col="y", static_features=[])
    future_df = _make_df(engine, {"unique_id": ["a"], "ds": [n], "promo": [1]})
    preds = fcst.predict(h=1, X_df=future_df, before_predict_callback=save_features)
    assert len(preds) == 1
    # future step looks up the occurrence `lag` back; since y == ds index, that is y[n - lag]
    np.testing.assert_allclose(captured[0][0], float(n - lag))


def test_lookup_lag_compute_latest_from_aggs_nan_and_empty():
    """LookupLag at predict: NaN when a bucket has fewer than `lag` occurrences,
    or when the looked-up occurrence carries no valid observation."""
    from mlforecast.pooled import _RowStore, get_kernel

    tfm = LookupLag(partition_by=["x"])
    tfm._set_core_tfm(2)  # lag = 2
    kernel = get_kernel(tfm)

    # bucket 0: 3 valid occurrences, so the one 2 back from ordinal 3 is 20.0;
    # bucket 1: a single occurrence has nothing 2 back;
    # bucket 2: the occurrence 2 back carries NaN, so the lookup is NaN
    rows = _RowStore.from_rows(
        bucket_id=np.array([0, 0, 0, 1, 2, 2, 2]),
        ordinal=np.array([0, 1, 2, 0, 0, 1, 2]),
        y=np.array([10.0, 20.0, 30.0, 99.0, 5.0, np.nan, 7.0]),
        n_buckets=3,
    )
    got = kernel.values_at(rows, np.array([0, 1, 2]), np.array([3, 1, 3]))
    np.testing.assert_allclose(got[0], 20.0)
    assert np.isnan(got[1]) and np.isnan(got[2])


# %% row kernels: a NaN target is not an observation
from mlforecast.lag_transforms import SeasonalRollingQuantile  # noqa: E402


def _nan_rows():
    """One bucket, two rows per ordinal, NaN in a few of them."""
    y = np.array(
        [np.nan, np.nan, 1.0, np.nan, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0, np.nan, np.nan]
    )
    return np.zeros(y.size, dtype=np.int64), np.repeat(np.arange(6), 2), y


def _median_reference(
    ordinal, y, targets, window=None, lag=1, season_length=1, min_samples=1
):
    """Median of the non-NaN rows in each target's window; ``None`` is expanding."""
    out = np.full(len(targets), np.nan)
    for i, t in enumerate(targets):
        if window is None:
            wanted = ordinal <= t - lag
        else:
            wanted = np.isin(ordinal, t - lag - np.arange(window) * season_length)
        vals = y[wanted]
        vals = vals[~np.isnan(vals)]
        if vals.size >= min_samples:
            out[i] = np.median(vals)
    return out


@pytest.mark.parametrize(
    "make_tfm, reference",
    [
        (
            lambda: RollingQuantile(p=0.5, window_size=3, min_samples=2, global_=True),
            dict(window=3, min_samples=2),
        ),
        (
            lambda: SeasonalRollingQuantile(
                p=0.5, season_length=2, window_size=2, min_samples=1, global_=True
            ),
            dict(window=2, season_length=2),
        ),
        (lambda: ExpandingQuantile(p=0.5, global_=True), dict()),
    ],
    ids=["rolling", "seasonal", "expanding"],
)
def test_row_kernels_skip_nan_targets(make_tfm, reference):
    """A NaN target neither enters a quantile nor counts toward ``min_samples``.

    The channels drop it at aggregation; the row store keeps every row, so the
    gather has to leave it out itself -- one NaN in a window would otherwise
    poison the whole quantile, and the expanding one for good.
    """
    from mlforecast.pooled import _RowStore, get_kernel

    bucket_id, ordinal, y = _nan_rows()
    kernel = get_kernel(make_tfm()._set_core_tfm(1))
    rows = _RowStore.from_rows(bucket_id, ordinal, y, n_buckets=1)
    targets = np.arange(1, 8)
    got = kernel.values_at(kernel._view(rows), np.zeros(7, dtype=np.int64), targets)
    want = _median_reference(ordinal, y, targets, **reference)
    assert np.isfinite(want).sum() >= 4  # the case is not vacuous
    np.testing.assert_allclose(got, want, equal_nan=True)


def test_lookup_lag_keeps_nan_occurrences():
    """The odd one out: a NaN row is still an occurrence, so the lookup lands on it."""
    from mlforecast.pooled import _RowStore, get_kernel

    kernel = get_kernel(LookupLag(partition_by=["x"])._set_core_tfm(1))
    rows = _RowStore.from_rows(
        bucket_id=np.zeros(3, dtype=np.int64),
        ordinal=np.arange(3),
        y=np.array([1.0, np.nan, 3.0]),
        n_buckets=1,
    )
    got = kernel.values_at(
        kernel._view(rows), np.zeros(2, dtype=np.int64), np.array([2, 3])
    )
    assert np.isnan(got[0])
    np.testing.assert_allclose(got[1], 3.0)


def test_nan_targets_left_by_a_target_transform_are_skipped():
    """End to end: the NaN head `Differences` leaves is skipped, and doesn't count.

    Values are ``1.5 * t``, so the differenced target is 3 everywhere it exists.
    At lag 2 with a window of 4 the first timestamp whose window holds four
    real observations is ``ds == 5``; before it the row count alone would
    already satisfy ``min_samples``.
    """
    from mlforecast.target_transforms import Differences

    df = pd.DataFrame(
        {
            "unique_id": np.repeat(["a", "b"], 12),
            "ds": np.tile(np.arange(12), 2),
            "y": np.tile(1.5 * np.arange(12), 2),
        }
    )
    tfm = RollingQuantile(p=0.5, window_size=4, global_=True)
    ts = TimeSeries(
        freq=1, lag_transforms={2: [tfm]}, target_transforms=[Differences([2])]
    )
    out = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y", dropna=False
    )
    got = out[tfm._get_name(2)].to_numpy()
    want = np.where(out["ds"].to_numpy() >= 5, 3.0, np.nan)
    np.testing.assert_allclose(got, want, equal_nan=True)


# %% the row keys bound the bucket count, and windows never cross a bucket
def test_row_store_bucket_count_is_bounded():
    """Past the bound the keys wrap and every search silently misplaces its window."""
    from mlforecast.pooled import _MAX_BUCKETS, _RowStore

    args = (np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), np.ones(1))
    with pytest.raises(ValueError, match="buckets"):
        _RowStore.from_rows(*args, n_buckets=_MAX_BUCKETS + 1)
    rows = _RowStore.from_rows(*args, n_buckets=1)
    with pytest.raises(ValueError, match="buckets"):
        rows.grow(np.zeros(1, dtype=np.int64), _MAX_BUCKETS + 1)


def test_row_store_search_stays_inside_the_bucket():
    """A window reaching back before the calendar must not pick up another bucket's rows."""
    from mlforecast.pooled import _ROW_STRIDE, _RowStore

    rows = _RowStore.from_rows(
        bucket_id=np.array([0, 0, 1, 1]),
        ordinal=np.array([0, 1, 0, 1]),
        y=np.arange(4.0),
        n_buckets=2,
    )
    bucket = np.array([1])
    far_back = np.array([-4 * _ROW_STRIDE])
    for side in ("left", "right"):
        assert rows.search(bucket, far_back, side)[0] == 2
    # in-calendar targets are unaffected
    assert rows.search(bucket, np.array([0]), "left")[0] == 2
    assert rows.search(bucket, np.array([1]), "right")[0] == 4


# %% bucket growth must remap per-kernel inner state
def _stateful_inner_rows(ts):
    """Row count of every accumulator-carrying inner, per pooled leaf."""
    sizes = []
    for leaves in ts._get_pooled_tfms().values():
        for leaf in leaves:
            for obj in leaf._pooled_inner.values():
                stats = getattr(obj, "stats_", None)
                if stats is not None:
                    sizes.append(stats.shape[0])
                elif isinstance(obj, dict) and obj.get("s") is not None:
                    sizes.append(obj["s"].shape[0])
    return sizes


def _grow_setup(engine, tfm, lag, ys=(1.0, 10.0, 100.0), new_y=1000.0, new_brand="a0"):
    """Fit on brands b1/b2, then update adding a series in `new_brand`.

    `new_brand` sorts before the existing keys, so absorbing it renumbers the
    existing buckets -- a genuine permutation, not just an append.
    """
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "b", "b", "c", "c"],
            "ds": [1, 2, 1, 2, 1, 2],
            "y": [ys[0], ys[0] + 1, ys[1], ys[1] + 1, ys[2], ys[2] + 1],
            "brand": ["b1", "b1", "b1", "b1", "b2", "b2"],
        },
    )
    ts = TimeSeries(freq=1, lag_transforms={lag: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=["brand"],
    )
    update = _make_df(
        engine,
        {
            "unique_id": ["a", "b", "c", "d"],
            "ds": [3, 3, 3, 3],
            "y": [ys[0] + 2, ys[1] + 2, ys[2] + 2, new_y],
            "brand": ["b1", "b1", "b2", new_brand],
        },
    )
    ts.update(update)
    return ts


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
@pytest.mark.parametrize(
    "tfm_factory",
    [
        lambda: ExpandingMean(groupby=["brand"]),
        lambda: ExpandingMin(groupby=["brand"]),
        lambda: ExpandingMax(groupby=["brand"]),
        lambda: ExpandingStd(groupby=["brand"]),
        lambda: ExponentiallyWeightedMean(alpha=0.5, groupby=["brand"]),
    ],
    ids=["exp_mean", "exp_min", "exp_max", "exp_std", "ewm"],
)
def test_bucket_growth_remaps_stateful_inner_state(engine, lag, tfm_factory):
    """Growing the vocabulary must permute/extend accumulator-carrying inners.

    Without the remap the inner state keeps its pre-growth row count and the
    next `update` raises a broadcast error.
    """
    ts = _grow_setup(engine, tfm_factory(), lag)
    state = ts._pooled_states[("groupby", ("brand",), ())]
    assert state.n_buckets == 3
    assert list(state.bucket_uniques) == ["a0", "b1", "b2"]
    assert all(n == state.n_buckets for n in _stateful_inner_rows(ts))

    ts._predict_setup()
    ts._update_features()  # would raise pre-fix


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_bucket_growth_leaves_existing_buckets_unchanged(engine, lag):
    """The permutation must not shuffle the surviving buckets' state.

    A remap applied in the wrong direction still produces finite values, so this
    compares against a control whose vocabulary never grew.
    """
    tfm = ExpandingMean(groupby=["brand"])
    grown = _grow_setup(engine, tfm, lag)

    control = TimeSeries(
        freq=1, lag_transforms={lag: [ExpandingMean(groupby=["brand"])]}
    )
    control.fit_transform(
        _make_df(
            engine,
            {
                "unique_id": ["a", "a", "b", "b", "c", "c"],
                "ds": [1, 2, 1, 2, 1, 2],
                "y": [1.0, 2.0, 10.0, 11.0, 100.0, 101.0],
                "brand": ["b1", "b1", "b1", "b1", "b2", "b2"],
            },
        ),
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=["brand"],
    )
    control.update(
        _make_df(
            engine,
            {
                "unique_id": ["a", "b", "c"],
                "ds": [3, 3, 3],
                "y": [3.0, 12.0, 102.0],
                "brand": ["b1", "b1", "b2"],
            },
        )
    )

    col = tfm._get_name(lag)
    grown._predict_setup()
    control._predict_setup()
    got = dict(zip(grown.uids, grown._update_features()[col]))
    want = dict(zip(control.uids, control._update_features()[col]))
    for uid in want:
        np.testing.assert_allclose(got[uid], want[uid], rtol=1e-12)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_bucket_growth_pads_new_expanding_extremes_with_infinity(engine):
    """A new bucket's running extreme must ignore the cells it never saw.

    Padding the accumulator with 0.0 instead of +/-inf would make the extreme
    collapse onto the pad, so the values here are chosen to expose that.
    """
    tfm_min = ExpandingMin(groupby=["brand"])
    ts = _grow_setup(engine, tfm_min, 1, ys=(100.0, 200.0, 300.0), new_y=400.0)
    ts._predict_setup()
    vals = dict(zip(ts.uids, ts._update_features()[tfm_min._get_name(1)]))
    assert vals["d"] == 400.0  # its only observation, not the 0.0 pad

    tfm_max = ExpandingMax(groupby=["brand"])
    ts = _grow_setup(engine, tfm_max, 1, ys=(-100.0, -200.0, -300.0), new_y=-400.0)
    ts._predict_setup()
    vals = dict(zip(ts.uids, ts._update_features()[tfm_max._get_name(1)]))
    assert vals["d"] == -400.0


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_bucket_growth_keeps_accumulator_cell_count_uniform(engine):
    """Every bucket must have consumed the same number of calendar cells.

    `_expanding_fill` reads the new bucket's cell count off an existing row, so a
    change that breaks the lockstep would silently corrupt new buckets.
    """
    ts = _grow_setup(engine, ExpandingMean(groupby=["brand"]), 1)
    leaf = next(iter(ts._get_pooled_tfms().values()))[0]
    for obj in leaf._pooled_inner.values():
        assert np.ptp(obj.stats_[:, 0]) == 0


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_bucket_growth_survives_backup_restore(engine):
    """`_backup` rolls the state back per model; inner rows must still match."""
    ts = _grow_setup(engine, ExpandingMean(groupby=["brand"]), 1)
    state = ts._pooled_states[("groupby", ("brand",), ())]
    for _ in range(2):  # one round per model in a multi-model predict
        with ts._backup():
            ts._predict_setup()
            ts._update_features()
            ts._update_y(np.array([1.0, 2.0, 3.0, 4.0]))
            ts._update_features()
        assert all(n == state.n_buckets for n in _stateful_inner_rows(ts))


# %% update() must advance the accumulator kernels, one appended timestamp at a time
def _split_panel(n_appended, T=8):
    """Three series in two brands, split into a fitted head and an appended tail.

    The extremes sit exactly where a skipped fold would miss them: the last
    fitted timestamp and the first appended one.
    """
    brands = {"a": "b1", "b": "b1", "c": "b2"}
    rows = [
        {"unique_id": sid, "ds": t, "y": 10.0 + (3 * t + 7 * i) % 5, "brand": brand}
        for i, (sid, brand) in enumerate(brands.items())
        for t in range(1, T + n_appended + 1)
    ]
    df = pd.DataFrame(rows)
    df.loc[(df.unique_id == "a") & (df.ds == T), "y"] = 0.5
    df.loc[(df.unique_id == "c") & (df.ds == T + 1), "y"] = 100.0
    return df[df.ds <= T], df[df.ds > T], df


def _fit_ts(df, tfm, lag, keep_last_n=None):
    ts = TimeSeries(freq=1, lag_transforms={lag: [tfm]})
    ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=["brand"],
        keep_last_n=keep_last_n,
    )
    return ts


@pytest.mark.parametrize("keep_last_n", [None, 1], ids=["untrimmed", "trimmed"])
@pytest.mark.parametrize("n_appended", [1, 2])
@pytest.mark.parametrize("lag", _LAGS)
@pytest.mark.parametrize(
    "tfm_factory",
    [
        lambda: ExpandingMean(global_=True),
        lambda: ExpandingMean(groupby=["brand"]),
        lambda: ExpandingStd(groupby=["brand"]),
        lambda: ExpandingMin(global_=True),
        lambda: ExpandingMax(global_=True),
        lambda: ExponentiallyWeightedMean(alpha=0.5, global_=True),
    ],
    ids=[
        "exp_mean",
        "exp_mean_groupby",
        "exp_std_groupby",
        "exp_min",
        "exp_max",
        "ewm",
    ],
)
def test_update_advances_pooled_accumulators(tfm_factory, lag, n_appended, keep_last_n):
    """`update` must leave the accumulators as a fit over everything would.

    The inners fold one source cell per `PooledState.update`, which predict
    calls once per timestamp. An update that only appended columns left them
    behind by that many, so the next predict folded the newest source and
    skipped the ones in between -- the last fitted timestamp among them.
    """
    head, tail, full = _split_panel(n_appended)
    updated = _fit_ts(head, tfm_factory(), lag, keep_last_n)
    updated.update(tail)
    control = _fit_ts(full, tfm_factory(), lag, keep_last_n)
    col = tfm_factory()._get_name(lag)
    for ts in (updated, control):
        ts._predict_setup()
    got = updated._update_features()[col].to_numpy()
    want = control._update_features()[col].to_numpy()
    np.testing.assert_allclose(got, want, rtol=1e-12)
    # and the inner state itself, not just the value read off it
    (leaf,) = next(iter(updated._get_pooled_tfms().values()))
    (ref,) = next(iter(control._get_pooled_tfms().values()))
    for name, inner in leaf._pooled_inner.items():
        stats = getattr(inner, "stats_", None)
        if stats is not None:
            np.testing.assert_allclose(
                stats, ref._pooled_inner[name].stats_, rtol=1e-12
            )


# %% the sumsq channel is centred per bucket, so std survives an offset
from mlforecast.lag_transforms import SeasonalRollingStd  # noqa: E402


def _offset_panel(T=40, seed=0):
    """Two brands at levels 1e6 and 1, two series each, unit-variance noise."""
    rng = np.random.default_rng(seed)
    series = {"a": ("b1", 1e6), "b": ("b1", 1e6), "c": ("b2", 1.0), "d": ("b2", 1.0)}
    rows = [
        {"unique_id": sid, "ds": t, "y": level + rng.standard_normal(), "brand": brand}
        for sid, (brand, level) in series.items()
        for t in range(1, T + 1)
    ]
    return pd.DataFrame(rows)


def _std_reference(df, groups, lag, window=None, season_length=1, time_agg=None):
    """Sample std over each row's window, in extended precision.

    ``groups`` maps a series to its bucket; ``window=None`` is expanding.
    """
    ids, ords = df["unique_id"].to_numpy(), df["ds"].to_numpy()
    ys = df["y"].to_numpy().astype(np.longdouble)
    bucket = np.array([groups[s] for s in ids])
    out = np.full(len(df), np.nan, dtype=np.longdouble)
    for i in range(len(df)):
        if window is None:
            in_window = ords <= ords[i] - lag
        else:
            in_window = np.isin(ords, ords[i] - lag - np.arange(window) * season_length)
        mask = (bucket == bucket[i]) & in_window
        vals = ys[mask]
        if time_agg == "mean":
            vals = np.array(
                [ys[mask & (ords == o)].mean() for o in np.unique(ords[mask])]
            )
        if vals.size >= 2:
            out[i] = vals.std(ddof=1)
    return out.astype(float)


def _offset_groups(df, mode):
    if "groupby" in mode:
        return dict(zip(df["unique_id"], df["brand"]))
    return {sid: 0 for sid in df["unique_id"]}


_STD_CASES = [
    (lambda **m: RollingStd(window_size=5, **m), dict(window=5)),
    (
        lambda **m: SeasonalRollingStd(season_length=3, window_size=3, **m),
        dict(window=3, season_length=3),
    ),
    (lambda **m: ExpandingStd(**m), dict()),
    (
        lambda **m: RollingStd(window_size=5, time_agg="mean", **m),
        dict(window=5, time_agg="mean"),
    ),
    (lambda **m: ExpandingStd(time_agg="mean", **m), dict(time_agg="mean")),
]
_STD_IDS = [
    "rolling",
    "seasonal",
    "expanding",
    "rolling_time_agg",
    "expanding_time_agg",
]


@pytest.mark.parametrize(
    "mode", [dict(global_=True), dict(groupby=["brand"])], ids=["global", "groupby"]
)
@pytest.mark.parametrize("make_tfm, reference", _STD_CASES, ids=_STD_IDS)
def test_pooled_std_is_precise_on_offset_data(make_tfm, reference, mode):
    """``sum(x**2) - sum(x)**2 / n`` keeps three figures at ``y ~ 1e6 +- 1``.

    Centred on each bucket's first observed cell it keeps them all, and one
    bucket at 1e6 next to one at 1 shows the centre has to be per bucket.
    """
    df = _offset_panel()
    lag = 1
    ts = TimeSeries(freq=1, lag_transforms={lag: [make_tfm(**mode)]})
    out = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=["brand"],
    )
    got = out[make_tfm(**mode)._get_name(lag)].to_numpy()
    want = _std_reference(df, _offset_groups(df, mode), lag, **reference)
    ok = ~np.isnan(got)
    assert ok.sum() > len(df) // 2
    np.testing.assert_allclose(got[ok], want[ok], rtol=1e-9)


class _LevelModel:
    """Predicts a fixed level per series, so the recursion feeds known values back."""

    def __init__(self, levels):
        self.levels = levels

    def predict(self, X):  # noqa: ARG002
        return self.levels


@pytest.mark.parametrize("keep_last_n", [None, 1], ids=["untrimmed", "trimmed"])
@pytest.mark.parametrize("time_agg", [None, "mean"], ids=["rows", "time_agg"])
@pytest.mark.parametrize(
    "make_tfm, reference",
    [
        (lambda **m: RollingStd(window_size=4, **m), dict(window=4)),
        (lambda **m: ExpandingStd(**m), dict()),
    ],
    ids=["rolling", "expanding"],
)
def test_pooled_std_stays_centred_through_update_and_predict(
    make_tfm, reference, time_agg, keep_last_n
):
    """Cells appended by `update` and by the predict loop centre where the fit did.

    Trimmed, the block no longer holds the cell the centre came from, so this is
    where a centre re-derived from what is left would silently drift.
    """
    T, h = 20, 2
    df = _offset_panel(T=T + 2)
    head, tail = df[df.ds <= T], df[df.ds > T]
    mode = dict(groupby=["brand"], time_agg=time_agg)
    lag = 1
    ts = _fit_ts(head, make_tfm(**mode), lag, keep_last_n)
    ts.update(tail)
    col = make_tfm(**mode)._get_name(lag)
    levels = np.array([1e6, 1e6, 1.0, 1.0])
    seen = []

    def capture(X):
        seen.append(X[col].to_numpy())
        return X

    ts.predict({"m": _LevelModel(levels)}, horizon=h, before_predict_callback=capture)
    # the reference sees the predictions as rows; the last row of each series
    # only serves as a target
    future = pd.DataFrame(
        {
            "unique_id": np.tile(["a", "b", "c", "d"], h + 1),
            "ds": np.repeat(np.arange(T + 3, T + 4 + h), 4),
            "y": np.tile(levels, h + 1),
            "brand": np.tile(["b1", "b1", "b2", "b2"], h + 1),
        }
    )
    ref_df = pd.concat([df, future], ignore_index=True)
    want = _std_reference(
        ref_df, _offset_groups(ref_df, mode), lag, time_agg=time_agg, **reference
    )
    for step in range(h):
        at = (ref_df["ds"] == T + 3 + step).to_numpy()
        np.testing.assert_allclose(seen[step], want[at], rtol=1e-9)


def test_pooled_std_centre_survives_bucket_growth():
    """A brand appearing at update renumbers the buckets; the centres move with them."""
    T = 12
    df = _offset_panel(T=T)
    head, tail = df[df.ds <= T - 2], df[df.ds > T - 2]
    new = pd.DataFrame(
        {"unique_id": "e", "ds": [T - 1, T], "y": [50.0, 51.0], "brand": "a0"}
    )
    lag = 1
    tfm = RollingStd(window_size=4, groupby=["brand"])
    grown = _fit_ts(head, tfm, lag)
    grown.update(pd.concat([tail, new]))
    state = grown._pooled_states[("groupby", ("brand",), ())]
    assert list(state.bucket_uniques) == ["a0", "b1", "b2"]
    control = _fit_ts(
        pd.concat([df, new]), RollingStd(window_size=4, groupby=["brand"]), lag
    )
    col = tfm._get_name(lag)
    for ts in (grown, control):
        ts._predict_setup()
    got = grown._update_features()[col].to_numpy()
    want = control._update_features()[col].to_numpy()
    np.testing.assert_allclose(got, want, rtol=1e-9)


# %% collapsed views are maintained incrementally, and only for what's read
def _view_state(n_buckets=3, width=4, channels=("count", "sum", "sumsq", "min", "max")):
    from mlforecast.pooled import PooledState

    rng = np.random.default_rng(0)
    base = {}
    for name in channels:
        arr = rng.normal(size=(n_buckets, width))
        if name == "count":
            arr = np.abs(np.floor(arr * 2)) + 1.0
        base[name] = np.ascontiguousarray(arr)
    return PooledState(
        mode="global",
        group_cols=[],
        partition_cols=[],
        n_buckets=n_buckets,
        n_ordinals=width,
        base=base,
        series_bucket_id=np.arange(n_buckets),
    )


_TIME_AGGS_ALL = ["sum", "count", "mean", "min", "max"]


def _assert_views_fresh(state, ctx=""):
    """Every cached view must equal a collapse of the current base."""
    from mlforecast.pooled import _collapse, _view_value

    for time_agg, view in state._views.items():
        direct = _collapse(
            *_view_value(state.base, time_agg), state.cell_shift(time_agg), list(view)
        )
        for name in view:
            np.testing.assert_array_equal(
                view[name], direct[name], err_msg=f"{ctx}:{time_agg}:{name}"
            )


@pytest.mark.parametrize("time_agg", _TIME_AGGS_ALL)
def test_appended_views_match_a_full_recollapse(time_agg):
    """`append` extends cached views instead of dropping them.

    `_collapse` is elementwise per cell, so the extension must be bit-identical
    to recollapsing the whole block -- asserted exactly, not approximately.
    """
    state = _view_state()
    state.channels(time_agg)
    for step in range(5):
        state.append(np.array([1.0, 2.0, 3.0]), bucket_ids=np.arange(3))
        _assert_views_fresh(state, ctx=f"step{step}")
    assert state.width == 9


@pytest.mark.parametrize("time_agg", _TIME_AGGS_ALL)
def test_appended_views_handle_buckets_with_no_rows(time_agg):
    """A bucket absent from a step collapses to the empty-cell fills.

    Extending with the raw column instead of a collapsed one passes the mean
    case and fails here, so this is the guard that matters.
    """
    state = _view_state()
    view = state.channels(time_agg)
    # only bucket 0 gets a row this step
    state.append(np.array([5.0]), bucket_ids=np.array([0]))
    _assert_views_fresh(state, ctx="sparse")
    assert view["count"][1, -1] == 0.0
    assert view["sum"][1, -1] == 0.0
    assert view["sumsq"][1, -1] == 0.0
    assert view["min"][1, -1] == np.inf
    assert view["max"][1, -1] == -np.inf
    # the divide is pre-zeroed, so an unobserved cell is 0.0, never NaN
    assert not np.isnan(view["sum"]).any()


def test_views_are_dropped_on_grow_trim_and_restore():
    """The rare mutations still invalidate rather than maintain the views."""
    state = _view_state(width=6)
    state.bucket_uniques = np.array(["a", "b", "c"], dtype=object)

    state.channels("sum")
    state.grow_buckets(np.array(["a", "b", "c", "d"], dtype=object))
    assert state._views == {}
    _assert_views_fresh(state, ctx="grow")

    state.channels("sum")
    state.trim_to_last(3)
    assert state._views == {}
    _assert_views_fresh(state, ctx="trim")

    state.channels("sum")
    snap = state.snapshot()
    state.append(np.array([1.0, 2.0, 3.0, 4.0]), bucket_ids=np.arange(4))
    state.restore(snap)
    assert state._views == {}
    _assert_views_fresh(state, ctx="restore")


def test_views_hold_only_the_channels_a_kernel_reads():
    """Views are cached, so building all five would keep all five alive."""
    state = _view_state()
    view = state.channels("sum", ("sum", "count"))
    assert set(view) == {"sum", "count"}

    # a second kernel needing more widens the same view in place
    widened = state.channels("sum", ("min", "count"))
    assert widened is view
    assert set(view) == {"sum", "count", "min"}
    _assert_views_fresh(state, ctx="widened")

    # and the widened channels keep tracking appends
    state.append(np.array([1.0, 2.0, 3.0]), bucket_ids=np.arange(3))
    _assert_views_fresh(state, ctx="widened+append")


def test_channels_without_time_agg_is_the_base_store():
    """No collapse to cache, so nothing to maintain or invalidate."""
    state = _view_state()
    assert state.channels(None) is state.base
    assert state._views == {}


@pytest.mark.parametrize("time_agg", _TIME_AGGS_ALL)
def test_appended_views_work_from_a_minimal_base(time_agg):
    """A real `time_agg` state stores only `{count, <source>}`, not all five.

    Every collapsed channel is derived from that pair, so the incremental
    extension has to work off the same minimal base the state actually keeps.
    """
    from mlforecast.pooled import _TIME_AGG_SOURCE, base_channels

    needed = base_channels(("sum", "count"), time_agg)
    assert needed == {"count", _TIME_AGG_SOURCE[time_agg]}
    state = _view_state(channels=tuple(sorted(needed)))
    state.channels(time_agg)
    state.append(np.array([4.0, 5.0]), bucket_ids=np.array([0, 2]))
    _assert_views_fresh(state, ctx="minimal-base")
    assert set(state.base) == needed


# ---------------------------------------------------------------------------
# G5: the sample gate on long sparse windows.
#
# The gate counts observations recovered from a channel *mean*: `k * (S / k)`,
# which is not S in float64 (`49 * (1 / 49)` is 0.9999999999999999). A bucket
# whose window spans many more cells than it holds rows would then fail
# `n_obs >= min_samples` and blank out a real value.
# ---------------------------------------------------------------------------

#: smallest cell count whose round trip loses a ULP at one observation
_GATE_CELLS = 49


def _gate_df(engine, n_times, promo_ordinals):
    """One series over ``n_times`` steps, in partition ``1`` only where asked."""
    return _make_df(
        engine,
        {
            "unique_id": ["a"] * n_times,
            "ds": list(range(n_times)),
            "y": [float(t + 1) for t in range(n_times)],
            "promo": [1 if t in promo_ordinals else 0 for t in range(n_times)],
        },
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "tfm_factory,promo_ordinals,expected",
    [
        (lambda: ExpandingMean(partition_by=["promo"]), (0, _GATE_CELLS), 1.0),
        (lambda: ExpandingMin(partition_by=["promo"]), (0, _GATE_CELLS), 1.0),
        (lambda: ExpandingMax(partition_by=["promo"]), (0, _GATE_CELLS), 1.0),
        (
            lambda: RollingMean(_GATE_CELLS, partition_by=["promo"]),
            (0, _GATE_CELLS),
            1.0,
        ),
        # std needs two observations to clear its own `n > 1` gate
        (lambda: ExpandingStd(partition_by=["promo"]), (0, 1, _GATE_CELLS), 2**-0.5),
    ],
    ids=[
        "ExpandingMean",
        "ExpandingMin",
        "ExpandingMax",
        "RollingMean",
        "ExpandingStd",
    ],
)
def test_g5_1_sparse_window_gate_at_fit(engine, tfm_factory, promo_ordinals, expected):
    """A window of 49 cells holding 1-2 rows still clears `min_samples`."""
    df = _gate_df(engine, _GATE_CELLS + 1, promo_ordinals)
    tfm = tfm_factory()
    col = tfm._get_name(1)
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    out = ts.fit_transform(df, "unique_id", "ds", "y", dropna=False, static_features=[])
    values = np.asarray(out[col].to_numpy(), dtype=np.float64)
    np.testing.assert_allclose(values[_GATE_CELLS], expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "tfm_factory,expected",
    [
        (lambda: ExpandingMean(partition_by=["promo"]), 1.0),
        (lambda: ExpandingMin(partition_by=["promo"]), 1.0),
        (lambda: RollingMean(_GATE_CELLS, partition_by=["promo"]), 1.0),
    ],
    ids=["ExpandingMean", "ExpandingMin", "RollingMean"],
)
def test_g5_2_sparse_window_gate_at_predict(engine, tfm_factory, expected):
    """`update` recovers the same count from the same round trip."""
    from mlforecast.forecast import MLForecast
    from sklearn.dummy import DummyRegressor

    # the first predicted step lands at ordinal 49, so its window spans 49 cells
    df = _gate_df(engine, _GATE_CELLS, promo_ordinals=(0,))
    tfm = tfm_factory()
    col = tfm._get_name(1)
    captured = []

    def save_features(x):
        captured.append(np.asarray(x[col].to_numpy(), dtype=np.float64))
        return x

    fcst = MLForecast(
        models=[DummyRegressor()], freq=1, lags=[1], lag_transforms={1: [tfm]}
    )
    fcst.fit(df, id_col="unique_id", time_col="ds", target_col="y", static_features=[])
    future = _make_df(engine, {"unique_id": ["a"], "ds": [_GATE_CELLS], "promo": [1]})
    fcst.predict(h=1, X_df=future, before_predict_callback=save_features)
    np.testing.assert_allclose(captured[0][0], expected)


# %% kernel dispatch follows the class hierarchy
def test_get_kernel_resolves_subclasses():
    """A user subclass of a supported transform is pooled like its parent."""
    from mlforecast.lag_transforms import _BaseLagTransform
    from mlforecast.pooled import RollingMeanK, get_kernel

    class MyRollingMean(RollingMean):
        pass

    tfm = MyRollingMean(3, global_=True)._set_core_tfm(1)
    assert isinstance(get_kernel(tfm), RollingMeanK)

    class NotPooled(_BaseLagTransform):
        pass

    with pytest.raises(NotImplementedError, match="NotPooled"):
        get_kernel(NotPooled())
