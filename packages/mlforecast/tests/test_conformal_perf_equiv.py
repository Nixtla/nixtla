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
# 1. _raw_series_scales + _floor_series_scales
# ---------------------------------------------------------------------------

def make_scales_df(backend, id_dtype, shuffle):
    df = make_series_df()
    if id_dtype == "int":
        df["unique_id"] = pd.factorize(df["unique_id"])[0]
    if shuffle:
        df = df.sample(frac=1.0, random_state=0)
    if backend == "polars":
        df = pl.from_pandas(df)
        if id_dtype == "categorical":
            df = df.with_columns(pl.col("unique_id").cast(pl.Categorical))
    elif id_dtype == "categorical":
        df["unique_id"] = df["unique_id"].astype("category")
    return df


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("id_dtype", ["str", "categorical", "int"])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("method", ["mad", "std"])
def test_series_scales_equiv(backend, id_dtype, shuffle, method):
    df = make_scales_df(backend, id_dtype, shuffle)
    expected = ref._compute_series_scales(df, "unique_id", "ds", "y", method=method)
    raw = cp._raw_series_scales(df, "unique_id", "ds", "y", (method,))
    actual = cp._floor_series_scales(raw[method], floor_factor=1e-3)
    assert set(actual) == set(expected)
    for uid in expected:
        np.testing.assert_allclose(
            actual[uid], expected[uid], rtol=RTOL, err_msg=str(uid)
        )


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("id_dtype", ["str", "categorical", "int"])
@pytest.mark.parametrize("shuffle", [False, True])
def test_raw_series_scales(backend, id_dtype, shuffle):
    df = make_scales_df(backend, id_dtype, shuffle)
    raw = cp._raw_series_scales(df, "unique_id", "ds", "y", ("mad", "std"))
    pdf = df.to_pandas() if backend == "polars" else df
    expected = {"mad": {}, "std": {}}
    for uid, g in pdf.groupby("unique_id", observed=True):
        y = g.sort_values("ds")["y"].to_numpy()
        dy = np.diff(y)
        if dy.size == 0:
            expected["mad"][uid] = expected["std"][uid] = abs(y[0])
            continue
        expected["mad"][uid] = np.median(np.abs(dy - np.median(dy)))
        expected["std"][uid] = abs(dy[0]) if dy.size == 1 else np.std(dy, ddof=1)
    assert raw.keys() == expected.keys()
    for method in expected:
        assert raw[method].keys() == expected[method].keys()
        for uid, value in expected[method].items():
            np.testing.assert_allclose(
                raw[method][uid], value, rtol=RTOL, atol=1e-12, err_msg=str(uid)
            )
    single, pair, flat = make_scales_df("pandas", id_dtype, False)["unique_id"].unique()[-3:]
    assert raw["mad"][single] == raw["std"][single] == 3.5
    assert raw["std"][pair] == 3.0
    assert raw["mad"][flat] == raw["std"][flat] == 0.0


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_raw_series_scales_large_int_target(backend):
    rng = np.random.default_rng(0)
    y = 30_000_000 + rng.integers(0, 7, size=200).cumsum()
    df = pd.DataFrame({"unique_id": "a", "ds": np.arange(y.size), "y": y})
    if backend == "polars":
        df = pl.from_pandas(df)
    raw = cp._raw_series_scales(df, "unique_id", "ds", "y", ("std",))
    np.testing.assert_allclose(raw["std"]["a"], np.std(np.diff(y), ddof=1), rtol=RTOL)


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
# 3. interval functions (transfer and non-transfer)
# ---------------------------------------------------------------------------

INTERVAL_FNS = [
    "_add_conformal_error_intervals",
    "_add_conformal_distribution_intervals",
]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("fn_name", INTERVAL_FNS)
@pytest.mark.parametrize("is_transfer", [False, True])
@pytest.mark.parametrize("horizon", [5, 6])  # truncated (h < cs_h) and full (h == cs_h)
def test_interval_equiv(backend, fn_name, is_transfer, horizon):
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
    )
    expected = getattr(ref, fn_name)(fcst_df, cs_df, is_transfer=is_transfer, **kwargs)
    if is_transfer:
        actual = cp._add_transfer_intervals(
            fcst_df,
            cs_df,
            model_names=models,
            level=level,
            horizon=horizon,
            cs_h=cs_h,
            method=fn_name.removeprefix("_add_").removesuffix("_intervals"),
        )
    else:
        actual = getattr(cp, fn_name)(fcst_df, cs_df, **kwargs)
    assert_frames_close(actual, expected)


# ---------------------------------------------------------------------------
# 4. interval rescaling helper (replaces forecast.py triple loop)
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
