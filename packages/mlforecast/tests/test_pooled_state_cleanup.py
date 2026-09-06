"""Guards for the pooled-state cleanup + cheaper-_backup PR.

G1 (no-trim byte-identical): a deterministic multi-mode, multi-model,
multi-horizon forecast must reproduce golden predictions captured on the
pre-cleanup baseline. Because two models are fitted, ``TimeSeries._backup``
runs between them, so this also guards that the cheaper snapshot/restore
backup is behavior-identical to the original deepcopy.

The fit spans all five pooled modes (global / groupby / local-partition /
global+partition / groupby+partition) and every aggregate field
(mean/std/min/max/expanding/EWM), so a regression in any pooled append or
aggregate path moves a prediction and fails here.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlforecast.forecast import MLForecast
from mlforecast.lag_transforms import (
    ExpandingMean,
    ExponentiallyWeightedMean,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingStd,
)

# Predictions captured on baseline c64a1d5 (pre-cleanup). 12 rows x 2 models,
# row-major (ravel) after sorting by (unique_id, ds). Must stay identical
# through Part A (cleanup) and Part B (cheaper _backup).
_GOLDEN = np.array(
    [
        42.93514960996524,
        46.64935233901883,
        35.51788717586902,
        51.493699673639185,
        48.510361576704526,
        65.11025040404411,
        44.064850390032774,
        47.77905311908637,
        34.194088764839364,
        50.16990126260953,
        48.21717186661735,
        64.81706069395693,
        50.69786936779505,
        54.41207209684865,
        42.872199398702776,
        58.848011896472826,
        55.0232447832594,
        71.62313361059955,
        50.30213063220296,
        54.01633336125667,
        40.83977654200561,
        56.81558903977566,
        55.70428866006259,
        72.30417748740263,
    ]
)


def _make_panel():
    ids, ds, y, brand, promo = [], [], [], [], []
    series = {"a": ("x", 1.0), "b": ("x", 3.0), "c": ("y", 7.0), "d": ("y", 11.0)}
    for sid, (br, base) in series.items():
        for t in range(1, 17):
            ids.append(sid)
            ds.append(t)
            y.append(base + 2.0 * t + 5.0 * ((t * (1 if sid in ("a", "c") else 2)) % 4))
            brand.append(br)
            promo.append(t % 2)
    return pd.DataFrame(
        {"unique_id": ids, "ds": ds, "y": y, "brand": brand, "promo": promo}
    )


def _build_fcst():
    return MLForecast(
        models=[LinearRegression(), LinearRegression(fit_intercept=False)],
        freq=1,
        lags=[1],
        lag_transforms={
            1: [
                RollingMean(2, global_=True),
                RollingMean(2, groupby=["brand"]),
                RollingMean(2, min_samples=1, partition_by=["promo"]),
                RollingMean(2, min_samples=1, global_=True, partition_by=["promo"]),
                RollingMean(
                    2, min_samples=1, groupby=["brand"], partition_by=["promo"]
                ),
                RollingStd(3, min_samples=2, global_=True),
                RollingMin(3, global_=True),
                RollingMax(3, global_=True),
                ExpandingMean(global_=True),
                ExponentiallyWeightedMean(alpha=0.5, global_=True),
            ],
        },
    )


def _agg_arrays(agg):
    return [agg.unique_times, agg.sums, agg.counts, agg.sum_sq, agg.mins, agg.maxs]


def _assert_state_equal(got, ref):
    """Field-for-field equality of two PooledStates' mutable state.

    The mutable state is what `snapshot`/`restore` round-trips: the aggregate
    channels, the shared calendar length, the bucket vocabulary and the current
    series assignment.
    """
    assert got.n_buckets == ref.n_buckets
    assert got.n_ordinals == ref.n_ordinals
    np.testing.assert_array_equal(
        got.series_bucket_id,
        ref.series_bucket_id,
    )
    if ref.bucket_uniques is None:
        assert got.bucket_uniques is None
    else:
        np.testing.assert_array_equal(
            got.bucket_uniques,
            ref.bucket_uniques,
        )
    assert got.base.keys() == ref.base.keys()
    for name in ref.base:
        np.testing.assert_array_equal(
            got.base[name],
            ref.base[name],
        )
    if ref._rows is None:
        assert got._rows is None
    else:
        got_rows, ref_rows = got._rows.merged(), ref._rows.merged()
        for attr in ("ordinal", "y", "indptr"):
            np.testing.assert_array_equal(
                getattr(got_rows, attr),
                getattr(ref_rows, attr),
            )


def test_backup_snapshot_restores_pooled_state_like_deepcopy():
    """_backup's cheap snapshot/restore must leave every pooled state identical
    to a deepcopy taken before predict (predict mutates the states in place and
    _backup rolls them back per model)."""
    import copy

    df = _make_panel()
    fcst = _build_fcst()
    fcst.fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=["brand"],
    )
    ref = {k: copy.deepcopy(s) for k, s in fcst.ts._pooled_states.items()}
    assert ref  # there are pooled states to check
    future = []
    for sid in ["a", "b", "c", "d"]:
        for t in range(17, 20):
            future.append({"unique_id": sid, "ds": t, "promo": t % 2})
    fcst.predict(h=3, X_df=pd.DataFrame(future))
    for key, ref_state in ref.items():
        _assert_state_equal(fcst.ts._pooled_states[key], ref_state)


def test_snapshot_restore_after_dynamic_new_bucket():
    """Growing the bucket vocabulary mutates every per-bucket structure.

    `grow_buckets` re-sorts the vocabulary and permutes the channels, the row
    store and the series assignment, so a snapshot taken before must restore all
    of it."""
    import copy

    df = _make_panel()
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        lags=[1],
        lag_transforms={
            1: [
                RollingMean(2, min_samples=1, groupby=["brand"], partition_by=["promo"])
            ]
        },
    )
    fcst.fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=["brand"],
    )
    key = ("nonlocal", ("brand",), ("promo",))
    state = fcst.ts._pooled_states[key]
    ref = copy.deepcopy(state)
    snap = state.snapshot()
    # introduce brand-new (brand, promo) combos -> new buckets created in place
    from mlforecast.pooled import encode_keys

    new_keys = encode_keys([np.array(["x", "y"], dtype=object), np.array([9, 9])])
    state.grow_buckets(new_keys)
    assert state.n_buckets > ref.n_buckets  # new buckets really appeared
    state.restore(snap)
    _assert_state_equal(state, ref)


def test_g1_pooled_predictions_byte_identical():
    df = _make_panel()
    fcst = _build_fcst()
    fcst.fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=["brand"],
    )
    h = 3
    future = []
    for sid in ["a", "b", "c", "d"]:
        for t in range(17, 17 + h):
            future.append({"unique_id": sid, "ds": t, "promo": t % 2})
    X_df = pd.DataFrame(future)
    preds = fcst.predict(h=h, X_df=X_df)
    preds = preds.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    model_cols = [c for c in preds.columns if c not in ("unique_id", "ds")]
    got = preds[model_cols].to_numpy().ravel()
    # The cleanup must not change any prediction. The tolerance only absorbs
    # cross-platform floating-point noise: OpenBLAS DYNAMIC_ARCH dispatches a
    # different CPU kernel on the CI ubuntu-3.12 / Windows runners than on the
    # reference machine, so the recursive aggregate-append steps and the
    # LinearRegression predict accumulate ~1e-10 absolute / ~1e-12 relative
    # differences by horizon 3 (the failure is concentrated there, on the
    # intercept model). rtol/atol=1e-9 sits ~300x above that noise yet ~1e5x
    # below any genuine regression (a missed state restore or wrong aggregate
    # moves a prediction by >=1e-4 relative). The byte-identical guarantee for
    # the cleanup itself is enforced by the sibling state tests above, which
    # compare every PooledState field with assert_array_equal (platform
    # independent, since they diff two computations from the same run).
    np.testing.assert_allclose(got, _GOLDEN, rtol=1e-9, atol=1e-9)
