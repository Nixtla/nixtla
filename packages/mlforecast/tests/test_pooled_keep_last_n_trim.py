"""G2 guards for trimming pooled states under ``keep_last_n`` (PR 2).

Once ``keep_last_n`` is resolved, a pooled state drops its unused history
prefix, in parity with the ``TimeSeries.ga`` trim, keeping the last
``max(keep_last_n, R_state)`` ordinals -- where ``R_state`` is the largest
``_pooled_retention`` among its leaves, the trailing columns their ``update``
still reads once priming has run. A single byte-identical state hash cannot
survive a trim (it deliberately drops aggregate prefixes), so these guards
assert the *contract* instead:

* **G2.1 prediction-equality** -- predictions from a trimmed model match an
  untrimmed model (the dropped prefix never enters a finite window, so it
  cannot move a forecast). Covers the retention floor (``keep_last_n`` smaller
  than a window). Compared with a tolerance, not byte-for-byte: the rolling
  fast path cumsums the *whole* aggregate vector, so a shorter (trimmed) vector
  accumulates the window sum through different-magnitude partials -- genuine
  float associativity noise (~1e-13), not a regression. The byte-identical
  guarantee lives at the state level (G2.2/G2.3).
* **G2.2 trim == fit-on-slice** -- for a state whose every leaf has a
  *stateless* inner (Rolling*/SeasonalRolling*/Lag/row kernels), a trimmed
  state is byte-identical to a fresh state fit on only the retained tail of
  the input, including after a follow-up ``update()``. This does **not**
  generalise to accumulator-carrying leaves (Expanding*/EWM): the stored block
  still matches (G2.3), but the primed inner does not -- and that difference is
  exactly what makes trimming them sound. See G2.2b.
* **G2.3 suffix invariant** -- each retained aggregate vector equals the tail
  of the untrimmed vector and the retained calendar length equals
  ``max(keep_last_n, R_state)``, pinned per transform.
* **G2.4 retention assertion** -- a state's block is trimmable iff every
  channel leaf declares a finite ``_pooled_retention``, and its raw rows iff
  every row leaf does. Expanding*/EWM *are* trimmable (they carry an
  accumulator, so the dropped prefix is already folded in); ``ExpandingQuantile``
  and ``LookupLag`` keep every row, because they re-gather from ordinal 0, but
  they read nothing off the block, so a finite leaf sharing their key still
  gets its block trimmed.
* **G2.5 prime-then-trim** -- ``PooledState.transform`` refuses to run on a
  trimmed state. It reads the window factor off relative column positions, so
  re-priming from a truncated prefix would be silently wrong; every legitimate
  caller runs before ``_apply_keep_last_n``. Only the accumulator-carrying
  kernels are primed at all: a stateless inner declares ``primes_state = False``
  and is skipped, so it never reaches the guard in the first place.
* **G2.6 EWM fails loudly** -- ``EwmK.run_update`` raises when a cell it needs
  existed and was trimmed away, rather than skipping it and under-decaying.
  Source ordinals that predate the calendar are still skipped, which is the
  legitimate case the two used to share.
* **G2.7 row-gather reach** -- the bounded row kernels (``RollingQuantile``,
  ``SeasonalRollingQuantile``) trim to exactly the oldest ordinal their gather
  reaches; the raw row store is dropped on the same absolute-ordinal cutoff as
  the channels, so one column short would silently narrow the quantile.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast.forecast import MLForecast
from mlforecast.lag_transforms import (
    Combine,
    ExpandingMax,
    ExpandingMean,
    ExpandingMin,
    ExpandingQuantile,
    ExponentiallyWeightedMean,
    Offset,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingQuantile,
    RollingStd,
    SeasonalRollingMean,
    SeasonalRollingQuantile,
)

ID, TIME, TARGET = "unique_id", "ds", "y"


def _make_panel(T=20):
    """Balanced panel: every series spans ds=1..T, so the last ``R`` distinct
    timestamps are the last ``R`` ordinals of every (global / group / parent)
    calendar -- which makes ``df[df.ds > T - R]`` an exact ``last-R-ordinals``
    slice across all modes (used by the fit-on-slice contract test)."""
    ids, ds, y, brand, promo = [], [], [], [], []
    series = {"a": ("x", 1.0), "b": ("x", 3.0), "c": ("y", 7.0), "d": ("y", 11.0)}
    for sid, (br, base) in series.items():
        for t in range(1, T + 1):
            ids.append(sid)
            ds.append(t)
            y.append(base + 2.0 * t + 5.0 * ((t * (1 if sid in ("a", "c") else 2)) % 4))
            brand.append(br)
            promo.append(t % 2)
    return pd.DataFrame({ID: ids, TIME: ds, TARGET: y, "brand": brand, "promo": promo})


# Finite-window transforms across every pooled mode. All trimmable, so a fit
# with a small keep_last_n trims every pooled state.
def _finite_lag_transforms():
    return {
        1: [
            RollingMean(4, global_=True),
            RollingMean(4, groupby=["brand"]),
            RollingMean(3, min_samples=1, partition_by=["promo"]),
            RollingMean(3, min_samples=1, global_=True, partition_by=["promo"]),
            RollingMean(3, min_samples=1, groupby=["brand"], partition_by=["promo"]),
            RollingStd(4, min_samples=2, global_=True),
            RollingMin(4, global_=True),
            RollingMax(4, global_=True),
        ]
    }


def _build_fcst(lag_transforms, lags=(1,)):
    return MLForecast(
        models=[LinearRegression()],
        freq=1,
        lags=list(lags),
        lag_transforms=lag_transforms,
    )


def _future_X(h, T=20):
    rows = []
    for sid in ["a", "b", "c", "d"]:
        for t in range(T + 1, T + 1 + h):
            rows.append({ID: sid, TIME: t, "promo": t % 2})
    return pd.DataFrame(rows)


def _sorted_preds(preds):
    preds = preds.sort_values([ID, TIME]).reset_index(drop=True)
    model_cols = [c for c in preds.columns if c not in (ID, TIME)]
    return preds[model_cols].to_numpy().ravel()


_AGG_FIELDS = ("unique_times", "sums", "counts", "sum_sq", "mins", "maxs")


def _assert_state_byte_identical(got, ref, ctx=""):
    """Field-for-field equality of two PooledStates' mutable state.

    The mutable state is what `snapshot`/`restore` round-trips: the aggregate
    channels, the shared calendar length, the bucket vocabulary and the current
    series assignment.
    """
    assert got.n_buckets == ref.n_buckets, f"{ctx}:n_buckets"
    # `n_ordinals` is the absolute calendar position and keeps counting through a
    # trim, so a trimmed state and a fresh fit on the tail differ there by design;
    # what must match is the stored extent and its contents.
    assert got.width == ref.width, f"{ctx}:width"
    np.testing.assert_array_equal(
        got.series_bucket_id, ref.series_bucket_id, err_msg=f"{ctx}:series_bucket_id"
    )
    if ref.bucket_uniques is None:
        assert got.bucket_uniques is None, f"{ctx}:uniques-none"
    else:
        np.testing.assert_array_equal(
            got.bucket_uniques, ref.bucket_uniques, err_msg=f"{ctx}:uniques"
        )
    assert got.base.keys() == ref.base.keys(), f"{ctx}:channels"
    for name in ref.base:
        np.testing.assert_array_equal(
            got.base[name], ref.base[name], err_msg=f"{ctx}:base[{name}]"
        )
    if ref._rows is None:
        assert got._rows is None, f"{ctx}:rows-none"
    else:
        got_rows, ref_rows = got._rows.merged(), ref._rows.merged()
        for attr in ("ordinal", "y", "indptr"):
            np.testing.assert_array_equal(
                getattr(got_rows, attr),
                getattr(ref_rows, attr),
                err_msg=f"{ctx}:rows.{attr}",
            )


def _preprocess_states(df, keep_last_n, lag_transforms, lags=(1,)):
    fcst = _build_fcst(lag_transforms, lags=lags)
    fcst.preprocess(
        df,
        id_col=ID,
        time_col=TIME,
        target_col=TARGET,
        keep_last_n=keep_last_n,
        static_features=["brand"],
        dropna=False,
    )
    return fcst.ts._pooled_states


# Accumulator-carrying transforms across every pooled mode. Trimmable only
# because the inner keeps its running state, so the dropped prefix stays folded
# in -- these are the ones G2.4c asserts actually shrink.
def _accumulator_lag_transforms():
    return {
        1: [
            ExpandingMean(global_=True),
            ExpandingMean(groupby=["brand"]),
            ExpandingMin(global_=True),
            ExpandingMax(global_=True),
            ExponentiallyWeightedMean(alpha=0.5, global_=True),
            ExponentiallyWeightedMean(alpha=0.5, groupby=["brand"]),
            SeasonalRollingMean(season_length=3, window_size=2, global_=True),
        ]
    }


# A keep_last_n comfortably above every transform window (so the retention
# floor is a no-op and R == keep_last_n), but well below T so a trim happens.
_R = 8
_NO_TRIM = 10_000  # >= calendar length -> trim_to_last is a no-op everywhere


# --------------------------------------------------------------------------- #
# G2.1 -- prediction equality (incl. the retention floor).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("keep_last_n", [2, 6])
@pytest.mark.parametrize(
    "transforms",
    [_finite_lag_transforms, _accumulator_lag_transforms],
    ids=["finite", "accumulator"],
)
def test_g2_1_trimmed_predictions_match_untrimmed(keep_last_n, transforms):
    df = _make_panel()
    h = 4
    X_df = _future_X(h)

    fcst_trim = _build_fcst(transforms())
    fcst_trim.fit(
        df,
        id_col=ID,
        time_col=TIME,
        target_col=TARGET,
        static_features=["brand"],
        keep_last_n=keep_last_n,
    )
    trimmed = _sorted_preds(fcst_trim.predict(h=h, X_df=X_df))

    fcst_full = _build_fcst(transforms())
    fcst_full.fit(
        df,
        id_col=ID,
        time_col=TIME,
        target_col=TARGET,
        static_features=["brand"],
        keep_last_n=_NO_TRIM,
    )
    full = _sorted_preds(fcst_full.predict(h=h, X_df=X_df))

    # keep_last_n=2 is below the widest window; equality here proves the
    # max(keep_last_n, R_state) retention floor kept enough history -- and, for
    # the accumulator set, that trimming their prefix is prediction-neutral.
    np.testing.assert_allclose(trimmed, full, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# G2.2 -- trim == fresh fit on the retained tail, before and after update().
# --------------------------------------------------------------------------- #
# All RollingMean: every leaf here has a stateless inner, which is the scope
# where "trimmed == fresh fit on the tail" holds. G2.2b covers the other case.
_MODES = {
    "global": {1: [RollingMean(4, global_=True)]},
    "groupby": {1: [RollingMean(4, groupby=["brand"])]},
    "global+partition": {
        1: [RollingMean(3, min_samples=1, global_=True, partition_by=["promo"])]
    },
    "groupby+partition": {
        1: [RollingMean(3, min_samples=1, groupby=["brand"], partition_by=["promo"])]
    },
    "local+partition": {1: [RollingMean(3, min_samples=1, partition_by=["promo"])]},
}


def _engine(df, engine):
    if engine == "pandas":
        return df
    import polars as pl

    return pl.from_pandas(df)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("name", list(_MODES))
def test_g2_2_trim_equals_fit_on_truncated_slice(name, engine):
    # both engines so the engine-specific bucket_df trim (filter_with_mask) and
    # the narwhals _idsorted_to_bucket_pos rebuild are exercised under each.
    lag_transforms = _MODES[name]
    T = 20
    df = _make_panel(T)
    df_slice = df[df[TIME] > T - _R].reset_index(drop=True)

    trimmed = _preprocess_states(_engine(df, engine), _R, lag_transforms)
    fresh = _preprocess_states(_engine(df_slice, engine), _NO_TRIM, lag_transforms)

    assert trimmed.keys() == fresh.keys()
    for key in trimmed:
        _assert_state_byte_identical(trimmed[key], fresh[key], ctx=f"{name}:{key}")


def _preprocessed_ts(df, keep_last_n, lag_transforms):
    fcst = _build_fcst(lag_transforms)
    fcst.preprocess(
        df,
        id_col=ID,
        time_col=TIME,
        target_col=TARGET,
        keep_last_n=keep_last_n,
        static_features=["brand"],
        dropna=False,
    )
    return fcst.ts


@pytest.mark.parametrize("name", list(_MODES))
def test_g2_2_trim_then_update_matches_fresh_then_update(name):
    """The next update() must extend a trimmed state exactly as it would extend
    a state freshly fit on the retained tail (exercises the rebuilt _ts_aggs and
    both append conventions). Uses preprocess + TimeSeries.update so no model is
    trained -- the contract is purely about state, and a bare static string
    feature would otherwise trip the regressor in the global/local-only modes."""
    lag_transforms = _MODES[name]
    T = 20
    df = _make_panel(T)
    df_slice = df[df[TIME] > T - _R].reset_index(drop=True)

    ts_trim = _preprocessed_ts(df, _R, lag_transforms)
    ts_fresh = _preprocessed_ts(df_slice, _NO_TRIM, lag_transforms)

    # one new timestamp for every series (update requires all series each step)
    new_rows = pd.DataFrame(
        {
            ID: ["a", "b", "c", "d"],
            TIME: [T + 1] * 4,
            TARGET: [101.0, 103.0, 107.0, 111.0],
            "brand": ["x", "x", "y", "y"],
            "promo": [(T + 1) % 2] * 4,
        }
    )
    ts_trim.update(new_rows)
    ts_fresh.update(new_rows)

    assert ts_trim._pooled_states.keys() == ts_fresh._pooled_states.keys()
    for key in ts_trim._pooled_states:
        _assert_state_byte_identical(
            ts_trim._pooled_states[key],
            ts_fresh._pooled_states[key],
            ctx=f"{name}:{key}",
        )


def _leaf_and_states(df, keep_last_n, lag_transforms):
    fcst = _build_fcst(lag_transforms)
    fcst.preprocess(
        df,
        id_col=ID,
        time_col=TIME,
        target_col=TARGET,
        keep_last_n=keep_last_n,
        static_features=["brand"],
        dropna=False,
    )
    leaf = next(iter(fcst.ts._get_pooled_tfms().values()))[0]
    return leaf, fcst.ts._pooled_states


def test_g2_2b_accumulator_state_differs_from_fit_on_slice():
    """Trimming an Expanding* state is sound *because* the inner disagrees.

    The stored block is still a pure suffix (G2.3), so the state comparator
    passes -- but the primed accumulator carries the dropped prefix, which a
    fresh fit on the tail has never seen. That is the mechanism, so pin it:
    the trimmed model must predict like the full-history one, not like the
    fit-on-slice one.
    """
    T = 20
    df = _make_panel(T)
    df_slice = df[df[TIME] > T - _R].reset_index(drop=True)
    # groupby rather than global_ so `brand` is auto-dropped as an auxiliary
    # column and never reaches the model as a string feature
    lag_transforms = {1: [ExpandingMean(groupby=["brand"])]}
    key = ("groupby", ("brand",), ())

    trim_leaf, trim_states = _leaf_and_states(df, _R, lag_transforms)
    fresh_leaf, fresh_states = _leaf_and_states(df_slice, _NO_TRIM, lag_transforms)

    # (i) the stored block still matches a fresh fit on the tail
    _assert_state_byte_identical(trim_states[key], fresh_states[key], ctx="g2.2b")

    # (ii) ... but the accumulator does not: it counts the whole calendar
    trim_cells = trim_leaf._pooled_inner["count"].stats_[:, 0]
    fresh_cells = fresh_leaf._pooled_inner["count"].stats_[:, 0]
    assert trim_cells[0] > fresh_cells[0]
    assert trim_cells[0] == T - 1  # lag=1, so priming consumed T-1 cells

    # (iii) and that is what makes the trim prediction-neutral
    h = 4
    X_df = _future_X(h)
    args = dict(id_col=ID, time_col=TIME, target_col=TARGET, static_features=["brand"])
    preds = {}
    for tag, data, kln in [
        ("trimmed", df, _R),
        ("full", df, _NO_TRIM),
        ("slice", df_slice, _NO_TRIM),
    ]:
        fcst = _build_fcst({1: [ExpandingMean(groupby=["brand"])]})
        fcst.fit(data, keep_last_n=kln, **args)
        preds[tag] = _sorted_preds(fcst.predict(h=h, X_df=X_df))
    np.testing.assert_allclose(preds["trimmed"], preds["full"], rtol=1e-9, atol=1e-9)
    assert not np.allclose(preds["trimmed"], preds["slice"], rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# G2.3 -- suffix invariant: trimming only drops a prefix; length == retention.
# --------------------------------------------------------------------------- #
def test_g2_3_suffix_invariant_global():
    T = 20
    df = _make_panel(T)
    lag_transforms = {1: [RollingMean(4, global_=True)]}

    untrimmed = _preprocess_states(df, _NO_TRIM, lag_transforms)
    trimmed = _preprocess_states(df, _R, lag_transforms)

    key = ("global", (), ())
    u_base = untrimmed[key].base
    t_base = trimmed[key].base

    full_len = untrimmed[key].width
    assert full_len == T  # global calendar is the 20 distinct timestamps
    # W_state (= lag+window) <= _R, so retention == keep_last_n == _R
    assert trimmed[key].width == _R
    cutoff = full_len - _R

    for name in u_base:
        np.testing.assert_array_equal(
            t_base[name], u_base[name][:, cutoff:], err_msg=name
        )
    # the stored block is exactly the suffix of the calendar, while the absolute
    # ordinal counter keeps running so future windows still line up
    assert trimmed[key].width == _R
    assert trimmed[key].n_ordinals == T
    assert trimmed[key].ordinal_offset == full_len - _R
    assert untrimmed[key].ordinal_offset == 0


@pytest.mark.parametrize(
    "make_tfm, retention",
    [
        (lambda: ExpandingMean(global_=True), 1),
        (lambda: ExponentiallyWeightedMean(alpha=0.5, global_=True), 1),
        (lambda: SeasonalRollingMean(season_length=3, window_size=2, global_=True), 4),
        (lambda: RollingMean(4, global_=True), 4),
    ],
    ids=["expanding", "ewm", "seasonal", "rolling"],
)
def test_g2_3_retention_is_the_declared_tail(make_tfm, retention):
    """Retained width is exactly ``max(keep_last_n, _pooled_retention)``.

    Pinned per transform so a retention that silently widens (back to full
    history) or narrows (below what ``update`` reads) fails here rather than in
    a prediction comparison.
    """
    T = 20
    df = _make_panel(T)
    key = ("global", (), ())
    states = _preprocess_states(df, 1, {1: [make_tfm()]})  # keep_last_n under floor
    assert states[key].width == retention
    assert states[key].n_ordinals == T


# --------------------------------------------------------------------------- #
# G2.4 -- a state is trimmable iff every leaf declares a finite retention.
# --------------------------------------------------------------------------- #
def test_g2_4a_row_gathering_states_keep_full_history():
    """Only the leaves that re-gather from ordinal 0 block the trim.

    ``ExpandingQuantile`` and ``LookupLag`` rebuild from the raw row store with
    no accumulator to carry the prefix, so nothing may be dropped.
    """
    T = 20
    df = _make_panel(T)
    states = _preprocess_states(df, 3, {1: [ExpandingQuantile(p=0.5, global_=True)]})
    state = states[("global", (), ())]
    assert state.width == T
    assert state._rows.ordinal.min() == 0


def test_g2_4b_mixed_key_trims_the_block_and_keeps_the_rows():
    """A finite and an unbounded transform sharing one mode key produce ONE
    state. The unbounded leaf reads only the raw rows, so those stay whole,
    while the channel block the finite leaf reads is trimmed to its retention
    -- and predictions are unchanged."""
    T = 20
    df = _make_panel(T)
    lag_transforms = {
        1: [
            RollingMean(3, global_=True),  # finite: retention 3
            ExpandingQuantile(p=0.5, global_=True),  # unbounded: keeps every row
        ]
    }
    states = _preprocess_states(df, 3, lag_transforms)
    state = states[("global", (), ())]  # both transforms share this key
    assert state.width == 3
    assert state.base["count"].shape[1] == 3
    assert state._rows.ordinal.min() == 0

    def preds(keep_last_n):
        fcst = _build_fcst(lag_transforms)
        fcst.fit(
            df.drop(columns=["brand"]),
            id_col=ID,
            time_col=TIME,
            target_col=TARGET,
            static_features=[],
            keep_last_n=keep_last_n,
        )
        return _sorted_preds(fcst.predict(3, X_df=_future_X(3, T)))

    np.testing.assert_allclose(preds(3), preds(None), rtol=1e-12)


def test_g2_4c_accumulator_states_are_trimmed():
    """Expanding*/EWM states shrink to their declared tail.

    They used to be excluded from the trim outright, which also pinned every
    finite leaf sharing their key at full history. ``nbytes`` is asserted so a
    silent regression to full retention fails here.
    """
    T = 20
    df = _make_panel(T)
    lag_transforms = {
        1: [
            ExpandingMean(global_=True),
            ExponentiallyWeightedMean(alpha=0.5, groupby=["brand"]),
        ]
    }
    full = _preprocess_states(df, _NO_TRIM, lag_transforms)
    states = _preprocess_states(df, 3, lag_transforms)  # tiny keep_last_n

    for key in [("global", (), ()), ("groupby", ("brand",), ())]:
        assert full[key].width == T
        assert states[key].width == 3  # max(keep_last_n=3, retention=1)
        assert states[key].n_ordinals == T
        assert states[key].base["sum"].nbytes < full[key].base["sum"].nbytes


def test_g2_4c_one_accumulator_leaf_no_longer_pins_the_shared_state():
    """A finite leaf sharing a key with an Expanding* leaf still gets trimmed."""
    T = 20
    df = _make_panel(T)
    lag_transforms = {1: [RollingMean(3, global_=True), ExpandingMean(global_=True)]}
    state = _preprocess_states(df, 1, lag_transforms)[("global", (), ())]
    assert state.width == 3  # RollingMean's retention (lag-1 + window), not T


def test_g2_4_offset_and_combine_respect_inner_transform():
    """Offset/Combine delegate retention to their operands: a finite inner
    keeps the state trimmable, an unbounded row-reading inner keeps the rows
    whole while the block its sibling reads is still trimmed."""
    T = 20
    df = _make_panel(T)

    finite = {1: [Offset(RollingMean(3, global_=True), 1)]}
    state = _preprocess_states(df, _R, finite)[("global", (), ())]
    assert state.width == _R  # trimmed

    unbounded = {
        1: [
            Combine(
                RollingMean(3, global_=True),
                ExpandingQuantile(p=0.5, global_=True),
                np.add,
            )
        ]
    }
    state = _preprocess_states(df, 3, unbounded)[("global", (), ())]
    assert state.width == 3  # the block, for the rolling leaf
    assert state._rows.ordinal.min() == 0  # the rows, for the quantile leaf


def test_g2_4_wrapper_retention_delegates_without_double_counting():
    """``Offset`` must not add its shift on top of the inner's baked-in lag.

    ``_set_core_tfm`` already primes the inner at ``lag + n``, so adding ``n``
    again would over-retain (and disagree with ``update_samples``, which does).
    """
    inner = RollingMean(3, global_=True)
    Offset(inner, 1)._set_core_tfm(1)
    offset = Offset(RollingMean(3, global_=True), 1)._set_core_tfm(1)
    assert offset._pooled_retention == offset.tfm._pooled_retention
    assert offset.tfm._core_tfm.lag == 2  # 1 + n
    assert offset._pooled_retention == 1 + 3  # (lag-1) + window_size

    both_finite = Combine(
        RollingMean(3, global_=True), ExpandingMean(global_=True), np.add
    )._set_core_tfm(1)
    assert both_finite._pooled_retention == 3  # max(3, 1)
    with_unbounded = Combine(
        RollingMean(3, global_=True), ExpandingQuantile(p=0.5, global_=True), np.add
    )._set_core_tfm(1)
    assert with_unbounded._pooled_retention is None


# --------------------------------------------------------------------------- #
# G2.5 -- prime-then-trim is enforced, not assumed.
# --------------------------------------------------------------------------- #
def test_g2_5_transform_on_trimmed_state_raises():
    """`transform` reads the window factor off relative column positions.

    On a trimmed block that silently mis-primes the inner transforms, so it has
    to fail instead. Every legitimate caller runs before `_apply_keep_last_n`.
    """
    df = _make_panel(20)
    leaf, states = _leaf_and_states(df, _R, {1: [RollingMean(4, global_=True)]})
    state = states[("global", (), ())]
    assert state.ordinal_offset > 0
    with pytest.raises(RuntimeError, match="trimmed state"):
        state.transform(leaf._pooled_kernel, leaf._pooled_inner)


def _preprocessed_then_trimmed(lag_transforms):
    df = _make_panel(20)
    fcst = _build_fcst(lag_transforms)
    fcst.preprocess(
        df,
        id_col=ID,
        time_col=TIME,
        target_col=TARGET,
        keep_last_n=_R,
        static_features=["brand"],
        dropna=False,
    )
    return fcst


def test_g2_5_reinitializing_states_after_a_trim_raises():
    """The integration flavour: re-priming a trimmed instance is caught.

    Uses an accumulator leaf because only those are primed at all -- a stateless
    inner is skipped outright, which is the stronger version of the same
    guarantee and is pinned just below.
    """
    fcst = _preprocessed_then_trimmed({1: [ExpandingMean(global_=True)]})
    with pytest.raises(RuntimeError, match="trimmed state"):
        fcst.ts._initialize_lag_transform_states()


def test_g2_5_stateless_leaves_are_not_reprimed_at_all():
    """`Rolling*`/`SeasonalRolling*`/`Lag` carry nothing between calls.

    Their `transform` assigns no inner state and their `update` re-derives from
    the stored block, so priming them would build a full-width block only to
    discard it. Skipping it also means a trimmed state is never handed to
    `transform` on this path, rather than being handed one and rejecting it.
    """
    fcst = _preprocessed_then_trimmed({1: [RollingMean(4, global_=True)]})
    state = fcst.ts._pooled_states[("global", (), ())]
    assert state.ordinal_offset > 0
    fcst.ts._initialize_lag_transform_states()  # must not raise


# --------------------------------------------------------------------------- #
# G2.6 -- EWM fails loudly when its column was trimmed away.
# --------------------------------------------------------------------------- #
def test_g2_6_ewm_below_retention_fails_loudly():
    """Skipping the missing cell would under-decay silently, so it must raise.

    `run_update` also legitimately skips source ordinals that predate the
    calendar; only a cell that existed and was dropped is an error.
    """
    df = _make_panel(20)
    leaf, states = _leaf_and_states(
        df, _NO_TRIM, {3: [ExponentiallyWeightedMean(alpha=0.5, global_=True)]}
    )
    state = states[("global", (), ())]
    state.trim_to_last(1)  # lag=3 needs 3
    with pytest.raises(RuntimeError, match="trimmed below its retention"):
        state.update(leaf._pooled_kernel, leaf._pooled_inner)


# --------------------------------------------------------------------------- #
# G2.7 -- row-gathering kernels are trimmed on the same reach they read.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "make_tfm, reach",
    [
        # groupby, not global_, so `brand` is auto-dropped as auxiliary rather
        # than reaching the model as a string feature
        (lambda: RollingQuantile(p=0.5, window_size=4, groupby=["brand"]), 4),
        (
            lambda: SeasonalRollingQuantile(
                p=0.5, season_length=3, window_size=2, groupby=["brand"]
            ),
            4,
        ),
    ],
    ids=["rolling_quantile", "seasonal_rolling_quantile"],
)
def test_g2_7_bounded_row_kernels_trim_to_their_gather_reach(make_tfm, reach):
    """These re-gather raw rows, but over a bounded window, so they do trim.

    `trim_to_last` drops rows on the same absolute-ordinal cutoff as the
    channels, so the retention has to cover exactly the oldest ordinal the
    gather reaches -- one short and the quantile would silently be taken over
    fewer observations.
    """
    T = 20
    df = _make_panel(T)
    h = 4
    X_df = _future_X(h)

    assert make_tfm()._set_core_tfm(1)._pooled_retention == reach
    state = _preprocess_states(df, 1, {1: [make_tfm()]})[("groupby", ("brand",), ())]
    assert state.width == reach  # keep_last_n=1 is below it; the floor holds

    preds = {}
    for tag, kln in [("trimmed", 1), ("full", _NO_TRIM)]:
        fcst = _build_fcst({1: [make_tfm()]})
        fcst.fit(
            df,
            id_col=ID,
            time_col=TIME,
            target_col=TARGET,
            static_features=["brand"],
            keep_last_n=kln,
        )
        preds[tag] = _sorted_preds(fcst.predict(h=h, X_df=X_df))
    np.testing.assert_allclose(preds["trimmed"], preds["full"], rtol=1e-9, atol=1e-9)


def test_g2_1_trimmed_accumulator_survives_update_then_predict():
    """The composite path: trim, then feed new observations, then forecast.

    `update` appends to a block that no longer holds the prefix while the inner
    accumulator still carries it, so this is where a retention that is right for
    `predict` alone would still come apart.
    """
    T = 20
    df = _make_panel(T)
    head = df[df[TIME] < T]
    tail = df[df[TIME] == T]
    h = 4
    X_df = _future_X(h)

    preds = {}
    for tag, kln in [("trimmed", 1), ("full", _NO_TRIM)]:
        fcst = _build_fcst(_accumulator_lag_transforms())
        fcst.fit(
            head,
            id_col=ID,
            time_col=TIME,
            target_col=TARGET,
            static_features=["brand"],
            keep_last_n=kln,
        )
        fcst.ts.update(tail)
        preds[tag] = _sorted_preds(fcst.predict(h=h, X_df=X_df))
    np.testing.assert_allclose(preds["trimmed"], preds["full"], rtol=1e-9, atol=1e-9)
