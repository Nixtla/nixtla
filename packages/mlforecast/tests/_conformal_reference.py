# Frozen snapshot of mlforecast/conformal_prediction.py taken before the
# 2026-06-11 performance vectorization (see docs/superpowers/specs/
# 2026-06-11-conformal-perf-vectorization-design.md).
# Used as the equivalence oracle by tests/test_conformal_perf_equiv.py
# and tmp/bench_conformal.py. DO NOT EDIT.
__all__ = ["PredictionIntervals", "TransferConformal", "estimate_density_ratio"]

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import utilsforecast.processing as ufp
from utilsforecast.compat import DFType


class PredictionIntervals:
    """Class for storing prediction intervals metadata information."""

    def __init__(
        self,
        n_windows: int = 2,
        h: int = 1,
        method: str = "conformal_distribution",
        scale_estimator: Optional[Literal["mad", "std"]] = None,
    ):
        if n_windows < 2:
            raise ValueError(
                "You need at least two windows to compute conformal intervals"
            )
        allowed_methods = [
            "conformal_error",
            "conformal_distribution",
            "weighted_conformal_error",
            "weighted_conformal_distribution",
        ]
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")
        if scale_estimator is not None and scale_estimator not in ("mad", "std"):
            raise ValueError(
                f"scale_estimator must be 'mad', 'std', or None, got '{scale_estimator}'"
            )
        self.n_windows = n_windows
        self.h = h
        self.method = method
        self.scale_estimator = scale_estimator

    def __repr__(self):
        return (
            f"PredictionIntervals(n_windows={self.n_windows}, h={self.h}, "
            f"method='{self.method}', scale_estimator={self.scale_estimator!r})"
        )


_VALID_TRANSFER_METHODS = (
    "recalibrate",
    "weighted_conformal",
    "scale_aligned",
    "scale_aligned_weighted",
    "error_scaled",
)


@dataclass
class TransferConformal:
    """Predict-time configuration for transfer conformal prediction.

    Pass to ``MLForecast.predict(transfer_conformal=...)`` instead of the
    removed flat kwargs ``transfer_conformal_method``, ``covariate_shift_weights``,
    and ``dre_estimator``.  A plain string is shorthand for
    ``TransferConformal(method=<str>)``.
    """

    method: str = "recalibrate"
    dre_estimator: str = "logistic"
    weights: Optional[Union[np.ndarray, Callable]] = None
    n_windows: Optional[int] = None
    step_size: Optional[int] = None
    cv: int = 5
    clip_quantile: Optional[float] = 0.99

    def __post_init__(self) -> None:
        if self.method not in _VALID_TRANSFER_METHODS:
            raise ValueError(
                f"TransferConformal.method must be one of {_VALID_TRANSFER_METHODS}, "
                f"got '{self.method}'"
            )
        if self.dre_estimator not in ("logistic", "gradient_boosting"):
            raise ValueError(
                f"TransferConformal.dre_estimator must be 'logistic' or "
                f"'gradient_boosting', got '{self.dre_estimator}'"
            )
        if self.weights is not None and self.method not in (
            "weighted_conformal",
            "scale_aligned_weighted",
        ):
            raise ValueError(
                f"TransferConformal.weights is only valid with method='weighted_conformal' "
                f"or 'scale_aligned_weighted', got method='{self.method}'"
            )
        if self.n_windows is not None and self.n_windows < 1:
            raise ValueError(
                f"TransferConformal.n_windows must be >= 1, got {self.n_windows}"
            )
        if self.step_size is not None and self.step_size < 1:
            raise ValueError(
                f"TransferConformal.step_size must be >= 1, got {self.step_size}"
            )

    def validate(self, pi: PredictionIntervals) -> None:
        """Cross-validate against the fitted PredictionIntervals config."""
        if self.method in ("scale_aligned", "scale_aligned_weighted"):
            if pi.scale_estimator is None:
                raise ValueError(
                    f"TransferConformal(method='{self.method}') requires the source model "
                    "to have been fit with PredictionIntervals(scale_estimator='mad' or 'std')."
                )
        if self.method in ("weighted_conformal", "scale_aligned_weighted"):
            if not pi.method.startswith("weighted_conformal"):
                raise ValueError(
                    f"TransferConformal(method='{self.method}') requires the source model "
                    "to have been fit with PredictionIntervals(method='weighted_conformal_error') "
                    "or 'weighted_conformal_distribution'."
                )


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


def _add_signed_transfer_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    horizon: int,
    cs_h: int = 1,
) -> DFType:
    """Add asymmetric prediction intervals from signed conformity scores (y − pred).

    Unlike the symmetric conformal methods, uses separate lower and upper quantiles
    so that a systematically biased transferred model shifts the interval rather than
    merely widening it.

    When ``cs_h == 1`` but ``horizon > 1`` (calibration was done with 1-step ahead
    cross-validation), the same constant-width interval is broadcast across all
    horizon steps, matching the behaviour of the regular conformal methods in that
    case.

    Emits UserWarning when the interval lies entirely above or below the point forecast
    (both quantiles have the same sign), indicating severe source-model bias on the
    target domain.
    """
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    # Determine the effective horizon stored in cs_df. If cs_h=1 but horizon>1,
    # the calibration set only has 1-step scores; broadcast constant intervals.
    effective_cs_h = cs_h if cs_h > 1 else 1
    use_flat_pool = effective_cs_h == 1 and horizon > 1
    n_cal_flat = len(cs_df[model_names[0]].to_numpy())
    n_cal_series = n_cal_flat // effective_cs_h  # n_windows * n_target_series

    for model in model_names:
        mean = fcst_df[model].to_numpy().ravel()
        scores = cs_df[model].to_numpy().astype(float)

        # level is pre-sorted ascending (e.g. [80, 90, 95])
        # lo cols: reversed levels (widest first) → [lo-95, lo-90, lo-80]
        # hi cols: ascending levels (narrowest first) → [hi-80, hi-90, hi-95]
        lo_cuts = [((100 - lv) / 100) / 2 for lv in reversed(level)]
        hi_cuts = [1 - ((100 - lv) / 100) / 2 for lv in level]
        all_cuts = lo_cuts + hi_cuts

        if use_flat_pool:
            # cs_h=1: compute a single quantile over all scores, broadcast to all steps.
            # Shape: (n_cuts,) -> (n_cuts, horizon) via broadcasting.
            q_flat = np.quantile(scores, all_cuts)  # (n_cuts,)
            q_per_horizon = np.tile(q_flat[:, np.newaxis], (1, horizon))  # (n_cuts, horizon)
        else:
            scores_2d = scores.reshape(n_cal_series, effective_cs_h)
            # q_per_horizon: (n_cuts, horizon) — per-step quantiles
            q_per_horizon = np.quantile(scores_2d, all_cuts, axis=0)

        # Bias warnings
        n_lo = len(lo_cuts)
        q_hi_vals = q_per_horizon[n_lo:]   # (n_hi_cuts, horizon)
        q_lo_vals = q_per_horizon[:n_lo]   # (n_lo_cuts, horizon)
        for i, lv in enumerate(level):
            q_hi_lv = q_hi_vals[i]         # upper quantile for this level
            q_lo_lv = q_lo_vals[-(i + 1)]  # matching lower quantile
            if np.all(q_hi_lv < 0):
                warnings.warn(
                    f"Transfer recalibrate level={lv}: upper quantile is negative across "
                    "all horizon steps — interval lies entirely below point forecasts. "
                    "The transferred model systematically over-predicts on the target domain.",
                    UserWarning,
                    stacklevel=3,
                )
            elif np.all(q_lo_lv > 0):
                warnings.warn(
                    f"Transfer recalibrate level={lv}: lower quantile is positive across "
                    "all horizon steps — interval lies entirely above point forecasts. "
                    "The transferred model systematically under-predicts on the target domain.",
                    UserWarning,
                    stacklevel=3,
                )

        # Apply: lo = pred + q_lo, hi = pred + q_hi, broadcast per horizon step
        n_target = len(mean) // horizon
        mean_2d = mean.reshape(n_target, horizon)
        quantiles = (
            q_per_horizon[:, np.newaxis, :] + mean_2d[np.newaxis, :, :]
        ).reshape(len(all_cuts), -1).T  # (n_target * horizon, n_cuts)

        out_cols = (
            [f"{model}-lo-{lv}" for lv in reversed(level)]
            + [f"{model}-hi-{lv}" for lv in level]
        )
        fcst_df = ufp.assign_columns(fcst_df, out_cols, quantiles)

    return fcst_df


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    w_test: float = 1.0,
) -> float:
    """Tibshirani et al. (2019) Eq. (1): weighted (1-alpha) quantile.

    Appends a virtual ∞ residual with weight w_test, then returns the
    leftmost residual value whose cumulative weight meets (1-alpha).
    """
    total = weights.sum() + w_test
    sort_idx = np.argsort(values)
    sorted_vals = np.append(values[sort_idx], np.inf)
    sorted_w = np.append(weights[sort_idx] / total, w_test / total)
    cum_w = np.cumsum(sorted_w)
    idx = np.searchsorted(cum_w, 1.0 - alpha, side="left")
    return float(sorted_vals[idx])


def _add_weighted_conformal_error_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
    weights: Optional[np.ndarray] = None,
    is_transfer: bool = False,
    target_weights: Optional[np.ndarray] = None,
) -> DFType:
    """Weighted conformal error intervals per Tibshirani et al. (2019).

    When weights is None falls back to np.quantile (identical to unweighted).
    When weights are provided computes a global weighted quantile per horizon
    step across all n_windows * n_series calibration points, applied uniformly
    to all target-domain series.
    """
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    cuts = [lv / 100 for lv in level]
    for model in model_names:
        mean = fcst_df[model].to_numpy().ravel()
        scores = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
        scores = scores[:, :, :horizon]
        n_target = len(mean) // horizon
        if is_transfer:
            # Transfer scenario: pool all source calibration points globally.
            scores_pooled = scores.reshape(cs_n_windows * n_series, horizon)
            if weights is None:
                quantiles = np.quantile(scores_pooled, cuts, axis=0)  # (n_levels, horizon)
                quantiles = np.tile(quantiles, (1, n_target))  # (n_levels, n_target * horizon)
            else:
                w_pooled = weights.reshape(cs_n_windows * n_series, cs_h)[:, :horizon]
                w_test = (
                    float(target_weights.mean())
                    if target_weights is not None
                    else float(w_pooled.mean())
                )
                quantiles = np.empty((len(cuts), n_target * horizon))
                for h_i in range(horizon):
                    s_h = scores_pooled[:, h_i]
                    w_h = w_pooled[:, h_i]
                    for li, alpha in enumerate([1 - c for c in cuts]):
                        q = _weighted_quantile(s_h, w_h, alpha, w_test)
                        quantiles[li, h_i::horizon] = q
        elif weights is None:
            quantiles = np.quantile(scores, cuts, axis=0)
            quantiles = quantiles.reshape(len(cuts), -1)
        else:
            w = weights.reshape(cs_n_windows, n_series, cs_h)[:, :, :horizon]
            w_test = float(w.mean())
            quantiles = np.empty((len(cuts), n_series * horizon))
            for h_i in range(horizon):
                s_h = scores[:, :, h_i].ravel()
                w_h = w[:, :, h_i].ravel()
                for li, alpha in enumerate([1 - c for c in cuts]):
                    q = _weighted_quantile(s_h, w_h, alpha, w_test)
                    quantiles[li, h_i::horizon] = q
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        quantiles_out = np.vstack([mean - quantiles[::-1], mean + quantiles]).T
        fcst_df = ufp.assign_columns(fcst_df, lo_cols + hi_cols, quantiles_out)
    return fcst_df


def _add_weighted_conformal_distribution_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
    weights: Optional[np.ndarray] = None,
    is_transfer: bool = False,
    target_weights: Optional[np.ndarray] = None,
) -> DFType:
    """Weighted conformal distribution intervals per Tibshirani et al. (2019).

    When weights is None falls back to np.quantile (identical to unweighted).
    When weights are provided applies _weighted_quantile over synthetic paths.
    """
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    alphas = [100 - lv for lv in level]
    cuts = [alpha / 200 for alpha in reversed(alphas)]
    cuts.extend(1 - alpha / 200 for alpha in alphas)
    for model in model_names:
        scores = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
        scores = scores[:, :, :horizon]
        mean_flat = fcst_df[model].to_numpy().ravel()
        n_target = len(mean_flat) // horizon
        if is_transfer:
            # Transfer scenario: pool all source calibration points globally.
            # quantile(mean_t + {-s_i, +s_i}) = mean_t + quantile({-s_i, +s_i})
            scores_pooled = scores.reshape(cs_n_windows * n_series, horizon)
            mean_2d = mean_flat.reshape(n_target, horizon)
            if weights is None:
                sym_scores = np.vstack([-scores_pooled, scores_pooled])
                global_q = np.quantile(sym_scores, cuts, axis=0)  # (n_cuts, horizon)
                quantiles = (global_q[:, np.newaxis, :] + mean_2d[np.newaxis, :, :]).reshape(
                    len(cuts), -1
                ).T  # (n_target * horizon, n_cuts)
            else:
                w_pooled = weights.reshape(cs_n_windows * n_series, cs_h)[:, :horizon]
                w_double = np.vstack([w_pooled, w_pooled])
                w_test = (
                    float(target_weights.mean())
                    if target_weights is not None
                    else float(w_pooled.mean())
                )
                # Offsets: {-s_i, +s_i} for each horizon step
                sym_scores = np.vstack([-scores_pooled, scores_pooled])
                quantiles = np.empty((n_target * horizon, len(cuts)))
                for h_i in range(horizon):
                    p_h = sym_scores[:, h_i]
                    w_h = w_double[:, h_i]
                    for li, cut in enumerate(cuts):
                        q_offset = _weighted_quantile(p_h, w_h, 1.0 - cut, w_test)
                        quantiles[h_i::horizon, li] = mean_2d[:, h_i] + q_offset
        else:
            mean = mean_flat.reshape(1, n_series, -1)
            paths = np.vstack([mean - scores, mean + scores])  # (2*n_windows, n_series, horizon)
            if weights is None:
                quantiles = np.quantile(paths, cuts, axis=0)
                quantiles = quantiles.reshape(len(cuts), -1).T
            else:
                w = weights.reshape(cs_n_windows, n_series, cs_h)[:, :, :horizon]
                w_double = np.vstack([w, w])  # replicate for both path directions
                w_test = float(w.mean())
                quantiles = np.empty((n_series * horizon, len(cuts)))
                for h_i in range(horizon):
                    p_h = paths[:, :, h_i].ravel()
                    w_h = w_double[:, :, h_i].ravel()
                    for li, cut in enumerate(cuts):
                        q = _weighted_quantile(p_h, w_h, 1.0 - cut, w_test)
                        quantiles[h_i::horizon, li] = q
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        fcst_df = ufp.assign_columns(fcst_df, lo_cols + hi_cols, quantiles)
    return fcst_df


def _build_clf(estimator: str):
    """Build an sklearn classifier for density-ratio estimation."""
    if estimator == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000)
    elif estimator == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier()
    else:
        raise ValueError(
            f"estimator must be 'logistic' or 'gradient_boosting', got '{estimator}'"
        )


def estimate_density_ratio(
    source_features: np.ndarray,
    target_features: np.ndarray,
    estimator: str = "logistic",
    cv: int = 5,
    clip_quantile: Optional[float] = 0.99,
    return_target_weights: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Estimate w(x) = p_target(x) / p_source(x) for source domain points.

    Trains a binary classifier (source=0, target=1) on StandardScaler-
    normalised features and returns the odds ratio p(1|x) / p(0|x) for
    each source point.

    Args:
        source_features: Feature matrix for source-domain calibration points,
            shape (n_source, n_features).
        target_features: Feature matrix for target-domain points,
            shape (n_target, n_features).
        estimator: ``"logistic"`` (default) or ``"gradient_boosting"``.
        cv: Number of stratified K-fold splits for cross-fitting (``cv >= 2``).
            Source weights are computed from out-of-fold predictions, reducing
            overfitting from in-sample scoring. ``cv=0`` or ``cv=1`` uses the
            original in-sample behavior. Defaults to 5.
        clip_quantile: Clip source weights above this quantile of the computed
            weights to prevent extreme values. ``None`` disables clipping.
            Defaults to 0.99.
        return_target_weights: If ``True``, also return per-target-row weights
            (averaged across fold models when ``cv >= 2``). Defaults to ``False``.

    Returns:
        np.ndarray of shape (n_source,) if ``return_target_weights=False``, else
        a tuple ``(source_weights, target_weights)`` where target_weights has
        shape (n_target,).
    """
    from sklearn.preprocessing import StandardScaler

    n_src = len(source_features)
    n_tgt = len(target_features)
    if cv >= 2 and (n_src < cv or n_tgt < cv):
        raise ValueError(
            f"cv={cv} requires at least {cv} samples in both source ({n_src}) "
            f"and target ({n_tgt}). Reduce cv or provide more data."
        )
    X = np.vstack([source_features, target_features])
    y = np.array([0] * n_src + [1] * n_tgt)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if cv >= 2:
        from sklearn.model_selection import StratifiedKFold

        source_weights = np.zeros(n_src)
        fold_probas_tgt: List[np.ndarray] = []
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
        for train_idx, val_idx in skf.split(X_scaled, y):
            clf = _build_clf(estimator)
            clf.fit(X_scaled[train_idx], y[train_idx])
            src_val_mask = val_idx < n_src
            val_src_idx = val_idx[src_val_mask]
            if len(val_src_idx):
                proba = clf.predict_proba(X_scaled[val_src_idx])
                source_weights[val_src_idx] = proba[:, 1] / np.clip(proba[:, 0], 1e-10, None)
            if return_target_weights:
                proba_tgt = clf.predict_proba(X_scaled[n_src:])
                fold_probas_tgt.append(proba_tgt)
        source_weights = np.clip(source_weights, 1e-10, None)
        if return_target_weights:
            avg_tgt = np.mean(fold_probas_tgt, axis=0)
            target_weights = avg_tgt[:, 1] / np.clip(avg_tgt[:, 0], 1e-10, None)
    else:
        clf = _build_clf(estimator)
        clf.fit(X_scaled, y)
        proba = clf.predict_proba(X_scaled[:n_src])
        source_weights = np.clip(proba[:, 1] / np.clip(proba[:, 0], 1e-10, None), 1e-10, None)
        if return_target_weights:
            proba_tgt = clf.predict_proba(X_scaled[n_src:])
            target_weights = proba_tgt[:, 1] / np.clip(proba_tgt[:, 0], 1e-10, None)

    if clip_quantile is not None:
        clip_val = float(np.quantile(source_weights, clip_quantile))
        source_weights = np.minimum(source_weights, clip_val)
        if return_target_weights:
            target_weights = np.minimum(target_weights, clip_val)

    if return_target_weights:
        return source_weights, target_weights
    return source_weights


_INTERVAL_METHODS: Dict[str, Callable[..., Any]] = {
    "conformal_distribution": _add_conformal_distribution_intervals,
    "conformal_error": _add_conformal_error_intervals,
    "weighted_conformal_distribution": _add_weighted_conformal_distribution_intervals,
    "weighted_conformal_error": _add_weighted_conformal_error_intervals,
}


def get_conformal_method(method: str) -> Callable[..., Any]:
    if method not in _INTERVAL_METHODS:
        raise ValueError(
            f"prediction intervals method {method} not supported "
            f"please choose one of {', '.join(_INTERVAL_METHODS.keys())}"
        )
    return _INTERVAL_METHODS[method]


def compute_conformity_scores(
    cv_results: DFType,
    model_names: List[str],
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    signed: bool = False,
) -> DFType:
    """Compute conformity scores over CV folds.

    Args:
        cv_results: Cross-validation output with actual and predicted columns.
        model_names: Names of model columns to convert to errors.
        target_col: Name of the target column (dropped from output).
        feature_cols: Optional list of extra feature columns to keep alongside
            the error columns. Used by the weighted conformal DRE logic.
        signed: If True, compute signed errors (y - pred). If False (default),
            compute absolute errors |pred - target|.
    """
    for model in model_names:
        if signed:
            err = cv_results[target_col] - cv_results[model]
        else:
            err = abs(cv_results[model] - cv_results[target_col])
        cv_results = ufp.assign_columns(cv_results, model, err)
    result = ufp.drop_columns(cv_results, target_col)
    if feature_cols is not None:
        keep = [c for c in result.columns if c not in feature_cols] + feature_cols
        result = result[keep]
    return result


@dataclass
class _TransferMethodSpec:
    """Capability flags for a transfer conformal method."""

    fn: Callable
    needs_preprocess: bool = False
    needs_source_cs: bool = False
    runs_target_cv: bool = False


@dataclass
class TransferResult:
    """Return value from transfer conformal functions.

    Replaces the side-channel pattern where transfer functions mutated
    ``PredictionIntervals`` attributes.  ``predict()`` consumes this as a
    local variable — no instance state, no ``finally`` reset needed.
    """

    cs_df: DFType
    weights: Optional[np.ndarray] = None
    target_scales: Optional[Dict[str, float]] = None  # uid -> sigma_tgt
    target_weights: Optional[np.ndarray] = None
    signed: bool = False



def _robust_scale_ratio(src: np.ndarray, tgt: np.ndarray) -> float:
    """IQR(|tgt_errors|) / IQR(|src_errors|) with std and constant fallbacks."""
    src_abs = np.abs(src)
    tgt_abs = np.abs(tgt)
    iqr_src = float(np.percentile(src_abs, 75) - np.percentile(src_abs, 25))
    iqr_tgt = float(np.percentile(tgt_abs, 75) - np.percentile(tgt_abs, 25))
    if iqr_src >= 1e-10 and iqr_tgt >= 1e-10:
        return iqr_tgt / iqr_src
    # Fallback 1: std ratio
    std_src = float(np.std(src)) if len(src) > 1 else 0.0
    std_tgt = float(np.std(tgt)) if len(tgt) > 1 else 0.0
    if std_src >= 1e-10:
        warnings.warn(
            "IQR of residuals near zero; falling back to std ratio for scale estimation.",
            UserWarning,
            stacklevel=4,
        )
        return std_tgt / max(std_src, 1e-10)
    # Fallback 2: constant
    warnings.warn(
        "Both IQR and std of residuals near zero; scale ratio defaulting to 1.0.",
        UserWarning,
        stacklevel=4,
    )
    return 1.0


def _recalibrate_transfer(
    new_df: DFType,  # noqa: ARG001
    prediction_intervals: PredictionIntervals,
    tc: TransferConformal,
    model_names: List[str],
    target_col: str,
    backtest_results: Optional[DFType] = None,
    id_col: str = "unique_id",  # noqa: ARG001
    time_col: str = "ds",  # noqa: ARG001
    preprocess_fn: Optional[Callable] = None,  # noqa: ARG001
    source_cs_df: Optional[DFType] = None,  # noqa: ARG001
    source_scales: Optional[Dict] = None,  # noqa: ARG001
) -> TransferResult:
    """Recompute conformity scores from frozen-model backtest on new_df.

    Uses signed residuals (y − pred) so systematic bias shifts the interval
    instead of merely widening it.
    """
    effective_n = tc.n_windows if tc.n_windows is not None else prediction_intervals.n_windows
    if effective_n < 2:
        raise ValueError(
            f"transfer method 'recalibrate' requires at least 2 window(s), "
            f"got n_windows={effective_n}."
        )
    return TransferResult(
        cs_df=compute_conformity_scores(backtest_results, model_names, target_col, signed=True),
        signed=True,
    )


def _weighted_conformal_transfer(
    new_df: DFType,
    prediction_intervals: PredictionIntervals,  # noqa: ARG001
    tc: TransferConformal,
    model_names: List[str],
    target_col: str,  # noqa: ARG001
    backtest_results: Optional[DFType] = None,  # noqa: ARG001
    id_col: str = "unique_id",
    time_col: str = "ds",
    preprocess_fn: Optional[Callable] = None,
    source_cs_df: Optional[DFType] = None,
    source_scales: Optional[Dict] = None,  # noqa: ARG001
) -> TransferResult:
    """Compute DRE weights for source conformity scores under covariate shift.

    Unlike ``_recalibrate_transfer``, this method does NOT replace the source
    conformity scores. Instead it trains a logistic classifier to distinguish
    source from target features, computes likelihood-ratio weights for each
    source calibration point, and returns them in a ``TransferResult``.  The
    original ``source_cs_df`` is returned unchanged so the caller can continue
    using source residuals with weighted quantiles.

    Requires ``preprocess_fn`` (``MLForecast.preprocess``) and
    ``source_cs_df`` (the existing ``_cs_df``) to be provided.
    """
    if preprocess_fn is None or source_cs_df is None:
        raise ValueError(
            "transfer_conformal_method='weighted_conformal' requires the model "
            "to have been fit with a weighted_conformal method so that source "
            "features are stored, and preprocess_fn must be supplied."
        )

    non_feature_cols = set(list(model_names) + [id_col, time_col, "cutoff"])
    feature_cols = [c for c in source_cs_df.columns if c not in non_feature_cols]

    if not feature_cols:
        raise ValueError(
            "No feature columns found in source conformity scores. "
            "Refit the model with PredictionIntervals(method='weighted_conformal_error') "
            "or 'weighted_conformal_distribution' so that source features are stored."
        )

    tgt_preprocessed = preprocess_fn(new_df, validate_data=False)
    tgt_feature_cols = [c for c in feature_cols if c in tgt_preprocessed.columns]

    dropped = set(feature_cols) - set(tgt_feature_cols)
    if dropped:
        warnings.warn(
            f"Target data is missing {len(dropped)} feature column(s) used during source fit "
            f"({sorted(dropped)}). Density ratio estimation will use the common subset only.",
            UserWarning,
            stacklevel=2,
        )

    src_np = np.column_stack(
        [source_cs_df[c].to_numpy() for c in tgt_feature_cols]
    ).astype(float)
    tgt_np = np.column_stack(
        [tgt_preprocessed[c].to_numpy() for c in tgt_feature_cols]
    ).astype(float)

    weights, target_weights = estimate_density_ratio(
        src_np,
        tgt_np,
        estimator=tc.dre_estimator,
        cv=tc.cv,
        clip_quantile=tc.clip_quantile,
        return_target_weights=True,
    )
    return TransferResult(cs_df=source_cs_df, weights=weights, target_weights=target_weights)


def _scale_aligned_transfer(
    new_df: DFType,
    prediction_intervals: PredictionIntervals,
    tc: TransferConformal,  # noqa: ARG001
    model_names: List[str],  # noqa: ARG001
    target_col: str,
    backtest_results: Optional[DFType] = None,  # noqa: ARG001
    id_col: str = "unique_id",
    time_col: str = "ds",
    preprocess_fn: Optional[Callable] = None,  # noqa: ARG001
    source_cs_df: Optional[DFType] = None,
    source_scales: Optional[Dict] = None,
) -> TransferResult:
    """Zero-shot scale alignment: compute σ̂_target per series from new_df y history.

    Requires the model to have been fit with
    ``PredictionIntervals(scale_estimator='mad' or 'std')``.
    Returns source conformity scores unchanged together with per-series
    target scales in ``TransferResult.target_scales``.
    """
    if (
        source_scales is None
        or prediction_intervals.scale_estimator is None
        or source_cs_df is None
    ):
        raise ValueError(
            "transfer_conformal_method='scale_aligned' requires the model to have "
            "been fit with PredictionIntervals(scale_estimator='mad' or 'std')."
        )
    target_scale_dict = _compute_series_scales(
        new_df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        method=prediction_intervals.scale_estimator,
    )
    return TransferResult(cs_df=source_cs_df, target_scales=target_scale_dict)


def _scale_aligned_weighted_transfer(
    new_df: DFType,
    prediction_intervals: PredictionIntervals,
    tc: TransferConformal,
    model_names: List[str],
    target_col: str,
    backtest_results: Optional[DFType] = None,  # noqa: ARG001
    id_col: str = "unique_id",
    time_col: str = "ds",
    preprocess_fn: Optional[Callable] = None,
    source_cs_df: Optional[DFType] = None,
    source_scales: Optional[Dict] = None,
) -> TransferResult:
    """Compose scale alignment with DRE weighting.

    Requires fitting with ``weighted_conformal_error`` or
    ``weighted_conformal_distribution`` (for feature columns) AND
    ``scale_estimator`` set on ``PredictionIntervals``.
    """
    if source_cs_df is None:
        raise ValueError(
            "transfer_conformal_method='scale_aligned_weighted' requires source_cs_df."
        )
    wc_result = _weighted_conformal_transfer(
        new_df=new_df,
        prediction_intervals=prediction_intervals,
        tc=tc,
        model_names=model_names,
        target_col=target_col,
        id_col=id_col,
        time_col=time_col,
        preprocess_fn=preprocess_fn,
        source_cs_df=source_cs_df,
        source_scales=source_scales,
    )
    sa_result = _scale_aligned_transfer(
        new_df=new_df,
        prediction_intervals=prediction_intervals,
        tc=tc,
        model_names=model_names,
        target_col=target_col,
        id_col=id_col,
        time_col=time_col,
        preprocess_fn=preprocess_fn,
        source_cs_df=source_cs_df,
        source_scales=source_scales,
    )
    return TransferResult(
        cs_df=source_cs_df,
        weights=wc_result.weights,
        target_scales=sa_result.target_scales,
    )


def _error_scaled_transfer(
    new_df: DFType,  # noqa: ARG001
    prediction_intervals: PredictionIntervals,  # noqa: ARG001
    tc: TransferConformal,  # noqa: ARG001
    model_names: List[str],
    target_col: str,
    backtest_results: Optional[DFType] = None,
    id_col: str = "unique_id",  # noqa: ARG001
    time_col: str = "ds",  # noqa: ARG001
    preprocess_fn: Optional[Callable] = None,  # noqa: ARG001
    source_cs_df: Optional[DFType] = None,
    source_scales: Optional[Dict] = None,  # noqa: ARG001
) -> TransferResult:
    """Scale source conformity scores by a single global IQR ratio (target / source).

    The ratio is computed from the pooled residuals of all series combined — not
    per-series. This corrects for domain-level scale differences between source and
    target but does not account for series-level heterogeneity within the target domain.
    For per-series scale correction use ``scale_aligned`` instead.
    """
    if source_cs_df is None:
        raise ValueError(
            "transfer_conformal_method='error_scaled' requires source_cs_df; "
            "ensure the model was fit with prediction_intervals."
        )
    target_cs_df = compute_conformity_scores(backtest_results, model_names, target_col)

    scaled = ufp.copy_if_pandas(source_cs_df, deep=False)
    for model in model_names:
        src = source_cs_df[model].to_numpy().astype(float)
        tgt = target_cs_df[model].to_numpy().astype(float)
        scale = _robust_scale_ratio(src, tgt)
        scaled = ufp.assign_columns(scaled, model, src * scale)

    return TransferResult(cs_df=scaled)


_TRANSFER_METHODS: Dict[str, _TransferMethodSpec] = {
    "recalibrate": _TransferMethodSpec(
        fn=_recalibrate_transfer,
        runs_target_cv=True,
    ),
    "weighted_conformal": _TransferMethodSpec(
        fn=_weighted_conformal_transfer,
        needs_preprocess=True,
        needs_source_cs=True,
    ),
    "scale_aligned": _TransferMethodSpec(
        fn=_scale_aligned_transfer,
        needs_source_cs=True,
    ),
    "scale_aligned_weighted": _TransferMethodSpec(
        fn=_scale_aligned_weighted_transfer,
        needs_preprocess=True,
        needs_source_cs=True,
    ),
    "error_scaled": _TransferMethodSpec(
        fn=_error_scaled_transfer,
        needs_source_cs=True,
        runs_target_cv=True,
    ),
}


def get_transfer_method_spec(method: str) -> _TransferMethodSpec:
    if method not in _TRANSFER_METHODS:
        raise ValueError(
            f"transfer conformal method {method} not supported "
            f"please choose one of {', '.join(_TRANSFER_METHODS.keys())}"
        )
    return _TRANSFER_METHODS[method]

