"""G4 guards for computing the fit features on the cell store.

At fit the engine used to materialise a ``(n_buckets, width)`` block per
channel and several same-shaped temporaries inside ``combine``, then read
``n_rows`` values out of the result. The channel kernels now compute on the
``_CellStore`` of occupied cells instead. A bounded window (``Rolling*``,
``SeasonalRolling*``, ``Lag``) is located by two ``searchsorted`` calls over
every cell at once and holds at most ``window_size`` cells, which are gathered
and reduced in bounded blocks; an accumulator kernel (``Expanding*``, ``EWM``)
runs its inner transforms over each bucket's occupied cells, which are exactly
the ragged series its recurrence walks, and primes its predict-time state off
the same pass. Nothing scales with the calendar width or the bucket count
beyond the store itself, which is ``O(n_rows)``. Only the row kernels still
take the dense block.

That path is not bit-identical to the dense one for the averaging kernels:
coreforecast's rolling mean carries a running accumulator, while the store sums
each window's cells outright, so the two differ by float-associativity noise --
the same class G2.1 documents for trimming. What *is* exact is the part
anything downstream depends on:

* **G4.1 three-tier equivalence** against the dense block -- the NaN mask is
  exact for every kernel; ``Min``/``Max``/``Lag`` are bit-identical outright
  (order-independent reductions); only the mean/std *values* move, by ~1e-14
  relative.
* **G4.2 dropna row identity** -- ``_transform``'s ``keep_rows`` is driven by
  ``np.isnan``, so an exact mask means the store path cannot change which rows
  survive ``dropna``, which series get dropped, or what the model trains on.
  This is the guarantee that makes the value drift acceptable.
* **G4.3 primes_state is honest** -- the dense ``transform`` leaves inner state
  behind iff the kernel says it does, and priming from the store leaves the
  same state. ``_initialize_lag_transform_states`` trusts the flag, so a
  coreforecast release that started caching inside a rolling ``transform`` has
  to fail here rather than silently unprime.
* **G4.4 who takes the store path** -- every channel kernel does; the row
  kernels return ``None`` and take the dense block, on a shared key too.
* **G4.5 boundaries** -- a window longer than the calendar, a lag past its end,
  buckets with one row and with none, and seasonal strides on every phase.
* **G4.6 the transient is bounded** -- asserted against the block size, not
  merely reported, so a regression to the full-width path fails here.
* **G4.7 predict continues from a store-primed state** -- the features of every
  horizon step match those after a dense-primed fit, so trimming to the
  retention and updating from there is unaffected by where the state came
  from.
"""

import operator
import pickle
import tracemalloc
from contextlib import ExitStack, contextmanager, nullcontext
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

import mlforecast.lag_transforms as L
import mlforecast.pooled as pooled_mod
from mlforecast import MLForecast
from mlforecast.callbacks import SaveFeatures
from mlforecast.pooled import PooledState, base_channels, get_kernel

ID, TIME, TARGET = "unique_id", "ds", "y"
_D = pd.Timestamp("2020-01-01")
#: the kernel classes that compute on the store
_STORE_KERNELS = (
    pooled_mod._RollingMixin,
    pooled_mod._SeasonalMixin,
    pooled_mod._ExpandingMixin,
    pooled_mod.EwmK,
)


class _Constant(BaseEstimator):
    """A model that ignores its features, so predict exercises the engine only."""

    def fit(self, X, y=None):  # noqa: ARG002
        return self

    def predict(self, X):
        return np.zeros(len(X))


@contextmanager
def _dense_path():
    """Send every kernel down the dense ``transform`` path, as the reference."""
    with ExitStack() as stack:
        for cls in _STORE_KERNELS:
            stack.enter_context(
                mock.patch.object(cls, "fit_from_store", lambda *_: None)
            )
        yield


def _panel(n_series=24, length=60, seed=0):
    """Ragged starts, interior gaps, aligned ends -- what the engine allows.

    Aligned ends are required (`_check_aligned_ends`); the gaps are what make
    empty cells, and so the ``count`` channel, actually exercised.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_series):
        start = int(rng.integers(0, 8))
        for t in range(start, length):
            if t < length - 1 and rng.random() < 0.08:
                continue
            rows.append(
                (
                    f"s{i}",
                    _D + pd.Timedelta(days=int(t)),
                    float(rng.normal(10, 3)),
                    f"g{i % 4}",
                    str(rng.integers(0, 3)),
                )
            )
    return pd.DataFrame(rows, columns=[ID, TIME, TARGET, "brand", "promo"])


def _feature(df, transforms, dense=False, dropna=False, lag=1):
    """The fit-time feature columns, on the store or on the dense block."""
    if not isinstance(transforms, list):
        transforms = [transforms]
    fcst = MLForecast(
        freq="D", models=[LinearRegression()], lag_transforms={lag: transforms}
    )
    with _dense_path() if dense else nullcontext():
        out = fcst.preprocess(
            df,
            id_col=ID,
            time_col=TIME,
            target_col=TARGET,
            static_features=["brand"],
            dropna=dropna,
            validate_data=False,
        )
    known = {ID, TIME, TARGET, "brand", "promo"}
    cols = [c for c in out.columns if c not in known]
    return out, cols if len(cols) > 1 else cols[0]


def _channel_kernels():
    """The store-path kernels, as ``(id, factory)`` taking the pooling mode."""
    return [
        ("rolling_mean", lambda **kw: L.RollingMean(window_size=5, **kw)),
        ("rolling_std", lambda **kw: L.RollingStd(window_size=5, **kw)),
        ("rolling_min", lambda **kw: L.RollingMin(window_size=5, **kw)),
        ("rolling_max", lambda **kw: L.RollingMax(window_size=5, **kw)),
        (
            "seasonal_rolling_mean",
            lambda **kw: L.SeasonalRollingMean(season_length=7, window_size=3, **kw),
        ),
        (
            "seasonal_rolling_std",
            lambda **kw: L.SeasonalRollingStd(season_length=7, window_size=3, **kw),
        ),
        (
            "seasonal_rolling_min",
            lambda **kw: L.SeasonalRollingMin(season_length=7, window_size=3, **kw),
        ),
        ("expanding_mean", lambda **kw: L.ExpandingMean(**kw)),
        ("expanding_std", lambda **kw: L.ExpandingStd(**kw)),
        ("expanding_min", lambda **kw: L.ExpandingMin(**kw)),
        ("expanding_max", lambda **kw: L.ExpandingMax(**kw)),
        ("ewm", lambda **kw: L.ExponentiallyWeightedMean(alpha=0.3, **kw)),
    ]


_KERNEL_IDS = [n for n, _ in _channel_kernels()]
_ACCUMULATORS = [
    (n, m) for n, m in _channel_kernels() if n.startswith(("expanding", "ewm"))
]
#: kernels reading order-independent channels, so the store path cannot move them
_EXACT_VALUE_KERNELS = ("min", "max", "lag")


def _assert_equivalent(a, b, name):
    """Mask exact always; values exact for min/max/lag; float noise for the rest.

    ``std`` needs an absolute tolerance as well: it is computed as
    ``sqrt((s2 - s1**2/n) / (n-1))``, so on a bucket whose values are nearly
    identical the subtraction cancels and a last-ulp move in either moment is a
    large *relative* move in a near-zero result. The absolute error stays at the
    same ~1e-11 as everywhere else, which is what the guarantee is about.
    """
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=f"{name}:mask")
    if any(k in name for k in _EXACT_VALUE_KERNELS):
        np.testing.assert_array_equal(a, b, err_msg=f"{name}:values")
    else:
        atol = 1e-9 if "std" in name else 0.0
        np.testing.assert_allclose(
            a, b, rtol=1e-12, atol=atol, equal_nan=True, err_msg=name
        )


_MODES = [
    {"global_": True},
    {"groupby": ["brand"]},
    {"partition_by": ["promo"]},
    {"global_": True, "partition_by": ["promo"]},
    {"groupby": ["brand"], "partition_by": ["promo"]},
]
_MODE_IDS = [
    "global",
    "groupby",
    "local_partition",
    "global_partition",
    "groupby_partition",
]


# --------------------------------------------------------------------------- #
# G4.1 -- mask exact always, values exact for min/max, ~1e-14 for mean/std.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", _MODES, ids=_MODE_IDS)
@pytest.mark.parametrize("name,make", _channel_kernels(), ids=_KERNEL_IDS)
def test_g4_1_store_matches_dense(name, make, mode):
    """Asserted in tiers, strongest first, so a regression says which broke."""
    df = _panel()
    ref, col = _feature(df, make(**mode), dense=True)
    got, _ = _feature(df, make(**mode))
    _assert_equivalent(ref[col].to_numpy(), got[col].to_numpy(), name)


@pytest.mark.parametrize("time_agg", [None, "sum", "mean", "min", "count"])
def test_g4_1_time_agg_views_collapse_per_cell(time_agg):
    """`_collapse` is elementwise, so collapsing the occupied cells must equal
    the occupied cells of the collapsed dense view."""
    df = _panel()
    tfm = L.RollingMean(window_size=5, groupby=["brand"], time_agg=time_agg)
    ref, col = _feature(df, tfm, dense=True)
    tfm = L.RollingMean(window_size=5, groupby=["brand"], time_agg=time_agg)
    got, _ = _feature(df, tfm)
    _assert_equivalent(ref[col].to_numpy(), got[col].to_numpy(), "rolling_mean")


@pytest.mark.parametrize("min_samples", [None, 1, 8])
def test_g4_1_min_samples_gating_is_unaffected(min_samples):
    """The window count is an exact integer sum on the store, so the gate must
    agree with the dense path's `k * mean(count)` at every cell."""
    df = _panel()
    kw = dict(window_size=5, groupby=["brand"], min_samples=min_samples)
    ref, col = _feature(df, L.RollingMean(**kw), dense=True)
    got, _ = _feature(df, L.RollingMean(**kw))
    np.testing.assert_array_equal(
        np.isnan(ref[col].to_numpy()), np.isnan(got[col].to_numpy())
    )


def test_g4_1_wrappers_follow_their_leaves():
    """`Offset`/`Combine` resolve to pooled leaves and must follow them."""
    df = _panel()
    for tfm_factory in (
        lambda: L.Offset(L.RollingMean(window_size=5, groupby=["brand"]), 2),
        lambda: L.Combine(
            L.RollingMean(window_size=5, groupby=["brand"]),
            L.RollingMean(window_size=3, groupby=["brand"]),
            operator.truediv,
        ),
    ):
        ref, col = _feature(df, tfm_factory(), dense=True)
        got, _ = _feature(df, tfm_factory())
        _assert_equivalent(ref[col].to_numpy(), got[col].to_numpy(), "wrapper")


# --------------------------------------------------------------------------- #
# G4.2 -- an exact mask means dropna keeps exactly the same rows.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,make", _channel_kernels(), ids=_KERNEL_IDS)
def test_g4_2_dropna_keeps_identical_rows(name, make):
    """The concrete downstream consequence of the mask being exact.

    If the store path could move a value across the null boundary it would
    change the training set, which no tolerance on the values would catch.
    """
    df = _panel()
    ref, _ = _feature(df, make(groupby=["brand"]), dense=True, dropna=True)
    got, _ = _feature(df, make(groupby=["brand"]), dropna=True)
    assert len(ref) == len(got), name
    np.testing.assert_array_equal(ref[ID].to_numpy(), got[ID].to_numpy())
    np.testing.assert_array_equal(ref[TIME].to_numpy(), got[TIME].to_numpy())


# --------------------------------------------------------------------------- #
# G4.3 / G4.4 -- the kernel metadata the fit path rests on.
# --------------------------------------------------------------------------- #
def _all_kernel_specs():
    return [
        ("RollingMean", L.RollingMean(window_size=4, global_=True)),
        ("RollingStd", L.RollingStd(window_size=4, global_=True)),
        ("RollingMin", L.RollingMin(window_size=4, global_=True)),
        ("RollingMax", L.RollingMax(window_size=4, global_=True)),
        (
            "SeasonalRollingMean",
            L.SeasonalRollingMean(season_length=3, window_size=2, global_=True),
        ),
        (
            "SeasonalRollingStd",
            L.SeasonalRollingStd(season_length=3, window_size=2, global_=True),
        ),
        (
            "SeasonalRollingMin",
            L.SeasonalRollingMin(season_length=3, window_size=2, global_=True),
        ),
        (
            "SeasonalRollingMax",
            L.SeasonalRollingMax(season_length=3, window_size=2, global_=True),
        ),
        ("ExpandingMean", L.ExpandingMean(global_=True)),
        ("ExpandingStd", L.ExpandingStd(global_=True)),
        ("ExpandingMin", L.ExpandingMin(global_=True)),
        ("ExpandingMax", L.ExpandingMax(global_=True)),
        ("EWM", L.ExponentiallyWeightedMean(alpha=0.5, global_=True)),
        ("RollingQuantile", L.RollingQuantile(p=0.5, window_size=4, global_=True)),
        ("ExpandingQuantile", L.ExpandingQuantile(p=0.5, global_=True)),
        (
            "SeasonalRollingQuantile",
            L.SeasonalRollingQuantile(
                p=0.5, season_length=3, window_size=2, global_=True
            ),
        ),
        ("LookupLag", L.LookupLag(partition_by=["promo"])),
    ]


_SPEC_IDS = [n for n, _ in _all_kernel_specs()]


def _state_for(kernel, tfm, n_buckets=5, width=16):
    rng = np.random.default_rng(0)
    n = 200
    return PooledState.build(
        mode="global",
        group_cols=[],
        partition_cols=[],
        bucket_id_by_row=rng.integers(0, n_buckets, n),
        ordinal_by_row=rng.integers(0, width, n),
        y=rng.normal(10, 1, n),
        n_buckets=n_buckets,
        n_ordinals=width,
        series_bucket_id=np.zeros(n_buckets, dtype=np.int64),
        needed=base_channels(kernel.channels, tfm.time_agg),
        needs_rows=getattr(kernel, "needs_rows", False),
    )


@pytest.mark.parametrize("name,tfm", _all_kernel_specs(), ids=_SPEC_IDS)
def test_g4_3_primes_state_matches_reality(name, tfm):
    """`transform` must leave inner state behind exactly when it claims to.

    The priming skip in `_initialize_lag_transform_states` trusts this flag; if
    a coreforecast release started caching inside a rolling `transform`, the
    skip would silently leave it unprimed and this is where that shows up.
    """
    tfm = tfm._set_core_tfm(2) if hasattr(tfm, "_set_core_tfm") else tfm
    kernel = get_kernel(tfm)
    inner = kernel.make_inner()
    state = _state_for(kernel, tfm)
    before = pickle.dumps(inner)
    state.transform(kernel, inner)
    changed = pickle.dumps(inner) != before
    assert changed == kernel.primes_state, name


def _inner_state(kernel, inner):
    """What `update` reads back, as arrays: coreforecast's `stats_` or EWM's own."""
    if isinstance(kernel, pooled_mod.EwmK):
        st = inner["_state"]
        return {"s": st["s"], "started": st["started"], "next_src": st["next_src"]}
    return {name: np.asarray(tfm.stats_) for name, tfm in inner.items()}


def _accumulator_specs():
    """The specs whose kernel primes state, with their inner lag set."""
    out = []
    for name, tfm in _all_kernel_specs():
        tfm = tfm._set_core_tfm(2) if hasattr(tfm, "_set_core_tfm") else tfm
        if get_kernel(tfm).primes_state:
            out.append((name, tfm))
    return out


def _state_with_late_bucket(kernel, tfm, width=8):
    """Bucket 0 observed throughout; bucket 1 only at the last ordinal.

    So with any lag bucket 1 has no cell before the fit's cutoff -- the case
    where an empty coreforecast group would borrow its neighbour's state.
    """
    bid = np.array([0] * width + [1])
    ordi = np.array(list(range(width)) + [width - 1])
    return PooledState.build(
        mode="global",
        group_cols=[],
        partition_cols=[],
        bucket_id_by_row=bid,
        ordinal_by_row=ordi,
        y=np.arange(1.0, bid.size + 1),
        n_buckets=2,
        n_ordinals=width,
        series_bucket_id=np.zeros(2, dtype=np.int64),
        needed=base_channels(kernel.channels, tfm.time_agg),
    )


@pytest.mark.parametrize("shape", ["random", "late_bucket", "lag_past_calendar"])
@pytest.mark.parametrize(
    "name,tfm", _accumulator_specs(), ids=[n for n, _ in _accumulator_specs()]
)
def test_g4_3_store_priming_matches_dense_priming(name, tfm, shape):
    """`prime` must leave the inner as the dense `transform` would.

    The accumulator holds the same values minus the zeros of the empty cells,
    so it agrees to within an ulp; running extremes, the EWM flags and the
    cursor are exact, and the cell count is the calendar cells the dense pass
    consumed, not the occupied ones. A bucket with nothing before the cutoff,
    and a lag past the whole calendar, must prime to the empty-bucket fill
    rather than to a neighbour's state or an error.
    """
    kernel = get_kernel(tfm)
    dense_inner, store_inner = kernel.make_inner(), kernel.make_inner()
    if shape == "random":
        state = _state_for(kernel, tfm)
    elif shape == "late_bucket":
        state = _state_with_late_bucket(kernel, tfm)
    else:
        state = _state_with_late_bucket(kernel, tfm, width=kernel.lag - 1)
    state.prime(kernel, store_inner)
    state.transform(kernel, dense_inner)
    dense, store = _inner_state(kernel, dense_inner), _inner_state(kernel, store_inner)
    if shape == "lag_past_calendar" and not isinstance(kernel, pooled_mod.EwmK):
        # coreforecast's dense pass leaves `stats_` uninitialised when the lag
        # outruns the block, so the reference is the empty-bucket fill itself
        dense = {
            name: np.zeros((state.n_buckets, 2))
            if isinstance(inner, pooled_mod.core_tfms.ExpandingMean)
            else np.full(state.n_buckets, pooled_mod._expanding_fill(inner, None))
            for name, inner in dense_inner.items()
        }
    assert dense.keys() == store.keys()
    for key in dense:
        a, b = np.asarray(dense[key]), np.asarray(store[key])
        if a.dtype.kind == "f":
            np.testing.assert_allclose(a, b, rtol=1e-14, err_msg=f"{name}:{key}")
        else:
            np.testing.assert_array_equal(a, b, err_msg=f"{name}:{key}")


@pytest.mark.parametrize("name,tfm", _all_kernel_specs(), ids=_SPEC_IDS)
def test_g4_4_store_path_membership(name, tfm):
    """Exactly the channel kernels compute on the store, and cell for cell
    they agree with the dense block."""
    tfm = tfm._set_core_tfm(2) if hasattr(tfm, "_set_core_tfm") else tfm
    kernel = get_kernel(tfm)
    state = _state_for(kernel, tfm)
    values = kernel.fit_from_store(state._store)
    assert (values is not None) == isinstance(kernel, _STORE_KERNELS), name
    if values is None:
        return
    dense = state._store.gather(state.transform(kernel, kernel.make_inner()), 0)
    _assert_equivalent(dense, values, name.lower())


def test_g4_4_shared_key_splits_by_leaf():
    """Two leaves share one state; the path is chosen per leaf, not per state,
    and both must agree with the dense block."""
    df = _panel()
    make = lambda: [  # noqa: E731
        L.RollingMean(window_size=5, groupby=["brand"]),
        L.ExpandingMean(groupby=["brand"]),
    ]
    ref, cols = _feature(df, make(), dense=True)
    got, _ = _feature(df, make())
    for col in cols:
        kind = "expanding_mean" if "expanding" in col else "rolling_mean"
        _assert_equivalent(ref[col].to_numpy(), got[col].to_numpy(), kind)


# --------------------------------------------------------------------------- #
# G4.5 -- boundaries.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("window_size", [1, 2, 40, 80])
def test_g4_5_window_around_and_beyond_the_calendar(window_size):
    """A window reaching before the calendar starts is simply shorter."""
    df = _panel(length=40)
    kw = dict(window_size=window_size, groupby=["brand"], min_samples=1)
    ref, col = _feature(df, L.RollingMean(**kw), dense=True)
    got, _ = _feature(df, L.RollingMean(**kw))
    _assert_equivalent(ref[col].to_numpy(), got[col].to_numpy(), "rolling_mean")


@pytest.mark.parametrize("lag", [1, 39, 45])
def test_g4_5_lag_up_to_and_past_the_end(lag):
    """A source ordinal before the calendar gives an empty window, never a
    cell from the previous bucket."""
    df = _panel(length=40)
    kw = dict(window_size=5, partition_by=["promo"], min_samples=1)
    ref, col = _feature(df, L.RollingMean(**kw), dense=True, lag=lag)
    got, _ = _feature(df, L.RollingMean(**kw), lag=lag)
    _assert_equivalent(ref[col].to_numpy(), got[col].to_numpy(), "rolling_mean")
    if lag > 40:
        assert np.isnan(got[col].to_numpy()).all()


@pytest.mark.parametrize("season_length", [5, 7])
@pytest.mark.parametrize("window_size", [2, 3])
@pytest.mark.parametrize("lag", [1, 3])
def test_g4_5_seasonal_windows_on_every_phase(season_length, window_size, lag):
    """The phase-major layout must give the same window for every phase and
    lag, including the phases whose first cell is the phase itself."""
    df = _panel(length=50)
    kw = dict(
        season_length=season_length,
        window_size=window_size,
        groupby=["brand"],
        min_samples=1,
    )
    ref, col = _feature(df, L.SeasonalRollingMean(**kw), dense=True, lag=lag)
    got, _ = _feature(df, L.SeasonalRollingMean(**kw), lag=lag)
    _assert_equivalent(ref[col].to_numpy(), got[col].to_numpy(), "rolling_mean")


def test_g4_5_sparse_buckets():
    """A bucket with one observation, and a long stretch nobody observed.

    An empty stretch of calendar must not pull cells from before it into the
    windows that follow.
    """
    rows = [("s0", _D, 1.0, "g0", "0"), ("s1", _D, 2.0, "g1", "0")]
    for t in (1, 2, 3, 20, 21):
        rows += [
            ("s0", _D + pd.Timedelta(days=t), float(t), "g0", "0"),
            ("s1", _D + pd.Timedelta(days=t), float(t * 2), "g1", "0"),
        ]
    df = pd.DataFrame(rows, columns=[ID, TIME, TARGET, "brand", "promo"])
    kw = dict(window_size=3, groupby=["brand"], min_samples=1)
    ref, col = _feature(df, L.RollingMean(**kw), dense=True)
    got, _ = _feature(df, L.RollingMean(**kw))
    _assert_equivalent(ref[col].to_numpy(), got[col].to_numpy(), "rolling_mean")


# --------------------------------------------------------------------------- #
# G4.6 -- the transient is bounded, not merely smaller.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kernel_name", ["rolling_mean", "rolling_std", "expanding_mean", "ewm"]
)
def test_g4_6_fit_transient_is_bounded_by_the_store(kernel_name):
    """Asserted against the block size, so a regression to full width fails.

    The dense path peaks at several multiples of the *whole* block; the store
    path must stay within the store plus one bounded gather.
    """
    # deliberately sparse -- many buckets over a wide calendar with far fewer
    # rows, which is the partitioned shape the block is oversized for and the
    # only one where the O(n_rows) floor does not swamp the measurement
    n_buckets, width, window, n = 1500, 500, 5, 40_000
    tfm = {
        "rolling_mean": lambda: L.RollingMean(window_size=window, global_=True),
        "rolling_std": lambda: L.RollingStd(window_size=window, global_=True),
        "expanding_mean": lambda: L.ExpandingMean(global_=True),
        "ewm": lambda: L.ExponentiallyWeightedMean(alpha=0.3, global_=True),
    }[kernel_name]()
    tfm = tfm._set_core_tfm(1)
    kernel = get_kernel(tfm)
    rng = np.random.default_rng(0)
    bid = rng.integers(0, n_buckets, n)
    ordi = rng.integers(0, width, n)
    y = rng.normal(10, 1, n)

    def build():
        return PooledState.build(
            mode="global",
            group_cols=[],
            partition_cols=[],
            bucket_id_by_row=bid,
            ordinal_by_row=ordi,
            y=y,
            n_buckets=n_buckets,
            n_ordinals=width,
            series_bucket_id=np.zeros(n_buckets, dtype=np.int64),
            needed=base_channels(kernel.channels, tfm.time_agg),
        )

    def peak_of(dense):
        # a fresh state per run: once the dense path has derived the block it
        # stays cached on the state
        state = build()
        with _dense_path() if dense else nullcontext():
            tracemalloc.start()
            values = state.fit_values(kernel, kernel.make_inner())
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        return peak, values, state

    dense_peak, reference, _ = peak_of(True)
    store_peak, values, state = peak_of(False)

    _assert_equivalent(reference, values, kernel_name)

    # the O(n_rows) cell store and the output vector are unavoidable either way;
    # what must fall is the part that scales with the calendar
    floor = reference.nbytes + state._store.nbytes
    block_bytes = n_buckets * width * 8
    assert store_peak - floor < block_bytes, (
        f"store transient {(store_peak - floor) / 1e6:.1f}MB is not below one "
        f"full block ({block_bytes / 1e6:.1f}MB)"
    )
    assert store_peak < 0.6 * dense_peak, (
        f"store peak {store_peak / 1e6:.1f}MB vs dense {dense_peak / 1e6:.1f}MB"
    )


# --------------------------------------------------------------------------- #
# G4.7 -- predict continues from a store-primed state exactly as from a dense one.
# --------------------------------------------------------------------------- #
def _predict_features(df, tfm, dense, h=3):
    """The pooled feature at each horizon step, via `SaveFeatures`."""
    uids = np.sort(df[ID].unique())
    future = pd.DataFrame(
        {
            ID: np.repeat(uids, h),
            TIME: np.tile(
                pd.date_range(df[TIME].max() + pd.Timedelta(days=1), periods=h),
                len(uids),
            ),
            "promo": np.tile(["0", "1", "2"], len(uids)),
        }
    )
    with _dense_path() if dense else nullcontext():
        fcst = MLForecast(
            freq="D", models=[_Constant()], lags=[1], lag_transforms={1: [tfm]}
        )
        fcst.fit(df, static_features=["brand"], dropna=False, validate_data=False)
        callback = SaveFeatures()
        fcst.predict(h=h, X_df=future, before_predict_callback=callback)
    col = [c for c in fcst.ts.features_order_ if c not in {"lag1", "brand", "promo"}][0]
    return callback.get_features()[col].to_numpy()


@pytest.mark.parametrize("mode", _MODES, ids=_MODE_IDS)
@pytest.mark.parametrize("name,make", _ACCUMULATORS, ids=[n for n, _ in _ACCUMULATORS])
def test_g4_7_predict_from_store_primed_state(name, make, mode):
    """Fit on the store, trim, predict three steps: the same features as a
    dense-primed fit gives, which pins the priming end to end."""
    df = _panel()
    ref = _predict_features(df, make(**mode), dense=True)
    got = _predict_features(df, make(**mode), dense=False)
    _assert_equivalent(ref, got, name)
