"""Equivalence tests: vectorized conformal code vs the frozen reference.

The reference is tests/_conformal_reference.py, a verbatim snapshot of
mlforecast/conformal_prediction.py taken before the performance rewrite.
Every test here must pass both BEFORE and AFTER the rewrite (the rewrite
is tolerance-equal, rtol=1e-9).
"""
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

import tests._conformal_reference as ref
from mlforecast import conformal_prediction as cp

warnings.simplefilter("ignore", UserWarning)

RTOL = 1e-9


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def make_series_df(n_series=30, seed=0):
    """Long-format series df including length-1, length-2 and flat edge cases."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_series):
        n = int(rng.integers(3, 60))
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": f"id_{i:03d}",
                    "ds": np.arange(n),
                    "y": rng.normal(size=n).cumsum(),
                }
            )
        )
    frames.append(pd.DataFrame({"unique_id": "single", "ds": [0], "y": [3.5]}))
    frames.append(pd.DataFrame({"unique_id": "pair", "ds": [0, 1], "y": [1.0, 4.0]}))
    frames.append(
        pd.DataFrame({"unique_id": "flat", "ds": np.arange(10), "y": 2.0})
    )
    return pd.concat(frames, ignore_index=True)


def make_cs_df(n_windows=2, n_series=8, cs_h=6, models=("m1", "m2"), seed=1,
               backend="pandas"):
    """Calibration-scores df whose row order matches reshape(n_windows, n_series, cs_h)."""
    rng = np.random.default_rng(seed)
    uids = [f"id_{s:03d}" for s in range(n_series)]
    data = {
        "unique_id": np.tile(np.repeat(uids, cs_h), n_windows),
        "ds": np.tile(np.arange(cs_h), n_windows * n_series),
        "cutoff": np.repeat(np.arange(n_windows), n_series * cs_h),
    }
    for m in models:
        data[m] = rng.normal(size=n_windows * n_series * cs_h)
    df = pd.DataFrame(data)
    return pl.from_pandas(df) if backend == "polars" else df


def make_fcst_df(n_target=4, horizon=5, models=("m1", "m2"), seed=3,
                 backend="pandas"):
    rng = np.random.default_rng(seed)
    uids = [f"tgt_{s}" for s in range(n_target)]
    df = pd.DataFrame(
        {
            "unique_id": np.repeat(uids, horizon),
            "ds": np.tile(np.arange(horizon), n_target),
        }
    )
    for m in models:
        df[m] = rng.normal(size=n_target * horizon)
    return pl.from_pandas(df) if backend == "polars" else df


def assert_frames_close(actual, expected, skip=("unique_id", "ds", "cutoff")):
    assert list(actual.columns) == list(expected.columns)
    for col in expected.columns:
        if col in skip:
            continue
        np.testing.assert_allclose(
            actual[col].to_numpy().astype(float),
            expected[col].to_numpy().astype(float),
            rtol=RTOL,
            err_msg=col,
        )


# ---------------------------------------------------------------------------
# 1. _compute_series_scales
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["mad", "std"])
def test_compute_series_scales_equiv(backend, method):
    df = make_series_df()
    if backend == "polars":
        df = pl.from_pandas(df)
    expected = ref._compute_series_scales(df, "unique_id", "ds", "y", method=method)
    actual = cp._compute_series_scales(df, "unique_id", "ds", "y", method=method)
    assert set(actual) == set(expected)
    for uid in expected:
        np.testing.assert_allclose(
            actual[uid], expected[uid], rtol=RTOL, err_msg=str(uid)
        )


# ---------------------------------------------------------------------------
# 2. _apply_scale_alignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_apply_scale_alignment_equiv(backend):
    cs_df = make_cs_df(backend=backend)
    scales = {f"id_{s:03d}": 0.5 + s for s in range(8)}
    expected = ref._apply_scale_alignment(cs_df, ["m1", "m2"], "unique_id", scales)
    actual = cp._apply_scale_alignment(cs_df, ["m1", "m2"], "unique_id", scales)
    for m in ("m1", "m2"):
        np.testing.assert_allclose(
            actual[m].to_numpy(), expected[m].to_numpy(), rtol=RTOL
        )
    for col in ("unique_id", "ds", "cutoff"):
        np.testing.assert_array_equal(
            actual[col].to_numpy(), expected[col].to_numpy()
        )


# ---------------------------------------------------------------------------
# 3. weighted quantiles (vectorized vs scalar reference)
# ---------------------------------------------------------------------------

def test_weighted_quantiles_matches_scalar_reference():
    rng = np.random.default_rng(2)
    values = rng.normal(size=200)
    values[::7] = values[0]  # ties exercise searchsorted side='left' tie-breaking
    weights = rng.uniform(0.1, 3.0, size=200)
    alphas = np.array([0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975])
    got = cp._weighted_quantiles(values, weights, alphas, w_test=1.3)
    expected = np.array(
        [ref._weighted_quantile(values, weights, a, w_test=1.3) for a in alphas]
    )
    np.testing.assert_array_equal(got, expected)


def test_weighted_quantile_wrapper_unchanged():
    rng = np.random.default_rng(7)
    values = rng.normal(size=50)
    weights = rng.uniform(0.1, 2.0, size=50)
    for alpha in (0.05, 0.5, 0.95):
        assert cp._weighted_quantile(values, weights, alpha, 0.7) == ref._weighted_quantile(
            values, weights, alpha, 0.7
        )


# ---------------------------------------------------------------------------
# 4. interval functions (all four, transfer and non-transfer)
# ---------------------------------------------------------------------------

WEIGHTED_FNS = [
    "_add_weighted_conformal_error_intervals",
    "_add_weighted_conformal_distribution_intervals",
]
UNWEIGHTED_FNS = [
    "_add_conformal_error_intervals",
    "_add_conformal_distribution_intervals",
]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("fn_name", WEIGHTED_FNS)
@pytest.mark.parametrize("is_transfer", [False, True])
@pytest.mark.parametrize("use_weights", [False, True])
@pytest.mark.parametrize("horizon", [5, 6])  # truncated (h < cs_h) and full (h == cs_h)
def test_weighted_interval_equiv(backend, fn_name, is_transfer, use_weights, horizon):
    n_windows, n_series, cs_h = 2, 8, 6
    models = ["m1", "m2"]
    level = [80, 95]
    cs_df = make_cs_df(n_windows, n_series, cs_h, models, backend=backend)
    n_target = 4 if is_transfer else n_series
    fcst_df = make_fcst_df(n_target, horizon, models, backend=backend)
    rng = np.random.default_rng(4)
    weights = (
        rng.uniform(0.1, 2.0, size=n_windows * n_series * cs_h)
        if use_weights
        else None
    )
    target_weights = (
        rng.uniform(0.1, 2.0, size=n_target * horizon)
        if (use_weights and is_transfer)
        else None
    )
    kwargs = dict(
        model_names=models,
        level=level,
        cs_n_windows=n_windows,
        cs_h=cs_h,
        n_series=n_series,
        horizon=horizon,
        weights=weights,
        is_transfer=is_transfer,
        target_weights=target_weights,
    )
    expected = getattr(ref, fn_name)(fcst_df, cs_df, **kwargs)
    actual = getattr(cp, fn_name)(fcst_df, cs_df, **kwargs)
    assert_frames_close(actual, expected)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("fn_name", UNWEIGHTED_FNS)
@pytest.mark.parametrize("is_transfer", [False, True])
@pytest.mark.parametrize("horizon", [5, 6])  # truncated (h < cs_h) and full (h == cs_h)
def test_unweighted_interval_equiv(backend, fn_name, is_transfer, horizon):
    n_windows, n_series, cs_h = 2, 8, 6
    models = ["m1", "m2"]
    level = [80, 95]
    cs_df = make_cs_df(n_windows, n_series, cs_h, models, backend=backend)
    n_target = 4 if is_transfer else n_series
    fcst_df = make_fcst_df(n_target, horizon, models, backend=backend)
    kwargs = dict(
        model_names=models,
        level=level,
        cs_n_windows=n_windows,
        cs_h=cs_h,
        n_series=n_series,
        horizon=horizon,
        is_transfer=is_transfer,
    )
    expected = getattr(ref, fn_name)(fcst_df, cs_df, **kwargs)
    actual = getattr(cp, fn_name)(fcst_df, cs_df, **kwargs)
    assert_frames_close(actual, expected)


# ---------------------------------------------------------------------------
# 5. interval rescaling helper (replaces forecast.py triple loop)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_rescale_interval_columns_equiv(backend):
    import utilsforecast.processing as ufp

    rng = np.random.default_rng(5)
    models = ["m1", "m2"]
    level = [80, 95]
    fcst_df = make_fcst_df(n_target=6, horizon=4, models=models, backend="pandas")
    for m in models:
        for lv in level:
            fcst_df[f"{m}-lo-{lv}"] = fcst_df[m] - rng.uniform(0.5, 2, len(fcst_df))
            fcst_df[f"{m}-hi-{lv}"] = fcst_df[m] + rng.uniform(0.5, 2, len(fcst_df))
    sigma_tgt = rng.uniform(0.5, 3.0, size=len(fcst_df))

    # Reference: the current triple loop from MLForecast.predict()
    expected = fcst_df.copy()
    for model_name in models:
        mean_arr = expected[model_name].to_numpy().astype(float)
        for lv in level:
            for direction in ("lo", "hi"):
                col = f"{model_name}-{direction}-{lv}"
                offset = expected[col].to_numpy().astype(float) - mean_arr
                expected = ufp.assign_columns(
                    expected, col, mean_arr + offset * sigma_tgt
                )

    native = pl.from_pandas(fcst_df) if backend == "polars" else fcst_df
    actual = cp._rescale_interval_columns(native, models, level, sigma_tgt)
    for col in [c for c in expected.columns if "-lo-" in c or "-hi-" in c]:
        np.testing.assert_allclose(
            actual[col].to_numpy(), expected[col].to_numpy(), rtol=RTOL, err_msg=col
        )
