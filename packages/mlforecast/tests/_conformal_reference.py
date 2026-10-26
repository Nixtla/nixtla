# Frozen snapshot of mlforecast/conformal_prediction.py taken before the
# 2026-06-11 performance vectorization (see docs/superpowers/specs/
# 2026-06-11-conformal-perf-vectorization-design.md).
# Used as the equivalence oracle by tests/test_conformal_perf_equiv.py
# and tmp/bench_conformal.py. DO NOT EDIT.

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import utilsforecast.processing as ufp
from utilsforecast.compat import DFType


def _compute_series_scales(
    df: DFType,
    id_col: str,
    time_col: str,
    target_col: str,
    method: str,
    floor_factor: float = 1e-3,
) -> Dict:
    """Compute per-series scale estimates on first differences for trend invariance.

    Uses MAD(Δy) or std(Δy). A data-relative floor (floor_factor × global median)
    with an absolute backstop of 1e-8 prevents zero-scale collapse for flat or very
    short series.
    """
    raw: Dict = {}
    unique_ids = np.unique(df[id_col].to_numpy())
    for uid in unique_ids:
        mask = ufp.is_in(df[id_col], [uid])
        sub = ufp.filter_with_mask(df, mask)
        t_arr = sub[time_col].to_numpy()
        y_arr = sub[target_col].to_numpy().astype(float)
        sort_idx = np.argsort(t_arr, kind="stable")
        y = y_arr[sort_idx]
        if len(y) < 2:
            raw[uid] = float(np.fabs(y).mean()) if len(y) > 0 else 1.0
            continue
        dy = np.diff(y)
        if method == "mad":
            raw[uid] = float(np.median(np.abs(dy - np.median(dy))))
        else:  # "std"
            raw[uid] = float(np.std(dy, ddof=1) if len(dy) > 1 else np.fabs(dy[0]))

    vals = np.array(list(raw.values()), dtype=float)
    global_median = float(np.median(vals)) if len(vals) > 0 else 1.0
    floor = max(floor_factor * global_median, 1e-8)
    floored: Dict = {}
    for uid, v in raw.items():
        effective = max(v, floor)
        if effective > v:
            warnings.warn(
                f"Series '{uid}' has near-zero scale estimate ({v:.2e}); "
                f"applying floor {floor:.2e}. Check for flat or constant series.",
                UserWarning,
                stacklevel=4,
            )
        floored[uid] = effective
    return floored


def _apply_scale_alignment(
    cs_df: DFType,
    model_names: List[str],
    id_col: str,
    source_scales: Dict,
) -> DFType:
    """Normalize source residuals by per-series source scale.

    For each source row: normalized_residual = residual / σ̂_src(uid).
    The per-target-series σ̂_tgt multiplication is applied post-hoc in
    ``predict()`` after the quantile step.  Returns a copy; does not mutate.
    """
    cs_df = ufp.copy_if_pandas(cs_df, deep=False)
    uid_arr = cs_df[id_col].to_numpy()
    norm_scales = np.vectorize(lambda uid: 1.0 / source_scales[uid])(uid_arr)
    for model in model_names:
        vals = cs_df[model].to_numpy().astype(float) * norm_scales
        cs_df = ufp.assign_columns(cs_df, model, vals)
    return cs_df


def _add_conformal_distribution_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
    weights: Optional[np.ndarray] = None,  # noqa: ARG001
    is_transfer: bool = False,
) -> DFType:
    """
    Adds conformal intervals to a `fcst_df` based on conformal scores `cs_df`.
    `level` should be already sorted. This strategy creates forecasts paths
    based on errors and calculate quantiles using those paths.
    """
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    alphas = [100 - lv for lv in level]
    cuts = [alpha / 200 for alpha in reversed(alphas)]
    cuts.extend(1 - alpha / 200 for alpha in alphas)
    for model in model_names:
        scores = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
        # restrict scores to horizon
        scores = scores[:, :, :horizon]
        mean_flat = fcst_df[model].to_numpy().ravel()
        n_target = len(mean_flat) // horizon
        if is_transfer:
            # Transfer scenario: pool all source calibration points globally.
            # quantile(mean_t + {-s_i, +s_i}) = mean_t + quantile({-s_i, +s_i})
            scores_pooled = scores.reshape(cs_n_windows * n_series, horizon)
            sym_scores = np.vstack([-scores_pooled, scores_pooled])
            global_q = np.quantile(sym_scores, cuts, axis=0)  # (n_cuts, horizon)
            mean_2d = mean_flat.reshape(n_target, horizon)
            quantiles = (global_q[:, np.newaxis, :] + mean_2d[np.newaxis, :, :]).reshape(
                len(cuts), -1
            ).T  # (n_target * horizon, n_cuts)
        else:
            mean = mean_flat.reshape(1, n_series, -1)
            paths = np.vstack([mean - scores, mean + scores])
            quantiles = np.quantile(paths, cuts, axis=0)
            quantiles = quantiles.reshape(len(cuts), -1).T
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        out_cols = lo_cols + hi_cols
        fcst_df = ufp.assign_columns(fcst_df, out_cols, quantiles)
    return fcst_df


def _add_conformal_error_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
    weights: Optional[np.ndarray] = None,  # noqa: ARG001
    is_transfer: bool = False,
) -> DFType:
    """
    Adds conformal intervals to a `fcst_df` based on conformal scores `cs_df`.
    `level` should be already sorted. This startegy creates prediction intervals
    based on the absolute errors.
    """
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    cuts = [lv / 100 for lv in level]
    for model in model_names:
        mean = fcst_df[model].to_numpy().ravel()
        scores = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
        # restrict scores to horizon
        scores = scores[:, :, :horizon]
        n_target = len(mean) // horizon
        if is_transfer:
            # Transfer scenario: pool all source calibration points globally.
            # Compute a single quantile per horizon step, then tile to all target series.
            scores_pooled = scores.reshape(cs_n_windows * n_series, horizon)
            quantiles = np.quantile(scores_pooled, cuts, axis=0)  # (n_levels, horizon)
            quantiles = np.tile(quantiles, (1, n_target))  # (n_levels, n_target * horizon)
        else:
            quantiles = np.quantile(scores, cuts, axis=0)
            quantiles = quantiles.reshape(len(cuts), -1)
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        quantiles = np.vstack([mean - quantiles[::-1], mean + quantiles]).T
        columns = lo_cols + hi_cols
        fcst_df = ufp.assign_columns(fcst_df, columns, quantiles)
    return fcst_df

