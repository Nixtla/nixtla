__all__ = [
    "ErrorScaling",
    "PredictionIntervals",
    "Recalibration",
    "ScaleAlignment",
    "TransferMethod",
]

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import coreforecast.scalers as core_scalers
import narwhals as nw
import numpy as np
import pandas as pd
import utilsforecast.processing as ufp
from coreforecast.grouped_array import GroupedArray as CoreGroupedArray
from utilsforecast.compat import DFType

from ._warnings import _user_warning_stacklevel


def _check_min(obj: Any, field: str, minimum: int) -> None:
    value = getattr(obj, field)
    if value is not None and value < minimum:
        raise ValueError(
            f"{type(obj).__name__}.{field} must be >= {minimum}, got {value}"
        )


@dataclass
class Recalibration:
    """Transfer method that computes the conformity scores from a backtest of the trained models on the transferred series.

    Args:
        h (int, optional): Forecast horizon of the backtest windows. Defaults to the ``h`` of ``PredictionIntervals``. Required if the model was fit without prediction intervals.
        n_windows (int, optional): Number of backtest windows, at least 2. Defaults to the ``n_windows`` of ``PredictionIntervals``, or 2 if the model was fit without them.
        step_size (int): Step size between the backtest windows. Defaults to 1.
    """

    h: Optional[int] = None
    n_windows: Optional[int] = None
    step_size: int = 1

    def __post_init__(self) -> None:
        _check_min(self, "h", 1)
        _check_min(self, "n_windows", 2)
        _check_min(self, "step_size", 1)


@dataclass
class ErrorScaling:
    """Transfer method that scales the training conformity scores by the ratio of the backtest errors on the transferred series to the training errors.

    Requires prediction intervals at fit. The backtest uses the ``h`` of ``PredictionIntervals``.

    Args:
        n_windows (int, optional): Number of backtest windows. Defaults to the ``n_windows`` of ``PredictionIntervals``.
        step_size (int): Step size between the backtest windows. Defaults to 1.
    """

    n_windows: Optional[int] = None
    step_size: int = 1

    def __post_init__(self) -> None:
        _check_min(self, "n_windows", 1)
        _check_min(self, "step_size", 1)


@dataclass
class ScaleAlignment:
    """Transfer method that scales the training conformity scores by the ratio of each transferred series' scale to the training series' scale.

    Requires prediction intervals at fit.

    Args:
        scale_estimator (str): Per-series scale estimator, ``'mad'`` or ``'std'``. Defaults to 'mad'.
    """

    scale_estimator: Literal["mad", "std"] = "mad"

    def __post_init__(self) -> None:
        if self.scale_estimator not in ("mad", "std"):
            raise ValueError(
                "ScaleAlignment.scale_estimator must be 'mad' or 'std', "
                f"got '{self.scale_estimator}'"
            )


TransferMethod = Union[Recalibration, ErrorScaling, ScaleAlignment]


class PredictionIntervals:
    """Class for storing prediction intervals metadata information."""

    def __init__(
        self,
        n_windows: int = 2,
        h: int = 1,
        method: str = "conformal_distribution",
    ):
        """
        Args:
            n_windows (int): Number of cross validation windows used to compute the conformity scores. Defaults to 2.
            h (int): Forecast horizon of each cross validation window. Defaults to 1.
            method (str): ``'conformal_distribution'`` or ``'conformal_error'``. Defaults to 'conformal_distribution'.
        """
        if n_windows < 2:
            raise ValueError(
                "You need at least two windows to compute conformal intervals"
            )
        allowed_methods = ["conformal_error", "conformal_distribution"]
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")
        self.n_windows = n_windows
        self.h = h
        self.method = method

    def __repr__(self):
        return (
            f"PredictionIntervals(n_windows={self.n_windows}, h={self.h}, "
            f"method='{self.method}')"
        )


def _raw_series_scales(
    df: DFType,
    id_col: str,
    time_col: str,
    target_col: str,
    methods: Sequence[str],
) -> Dict[str, Dict]:
    """MAD(Δy) and/or std(Δy) per series, before the floor of `_floor_series_scales`."""
    # process_df casts non-float targets to float32
    df = (
        nw.from_native(df, eager_only=True)
        .select(id_col, time_col, nw.col(target_col).cast(nw.Float64))
        .to_native()
    )
    uids, _, data, indptr, _ = ufp.process_df(df, id_col, time_col, target_col)
    y = data[:, 0]
    dy = core_scalers.Difference(1).fit_transform(CoreGroupedArray(y, indptr))
    dy_ga = CoreGroupedArray(dy, indptr)
    n_dy = np.diff(indptr) - 1
    single = np.abs(y[indptr[:-1]])
    raw = {}
    if "mad" in methods:
        mad = core_scalers.LocalRobustScaler("mad").fit(dy_ga).stats_[:, 1]
        raw["mad"] = np.where(n_dy == 0, single, mad)
    if "std" in methods:
        # coreforecast's std has ddof=0
        std = core_scalers.LocalStandardScaler().fit(dy_ga).stats_[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            std = std * np.sqrt(n_dy / (n_dy - 1))
        first_dy = np.abs(dy[np.minimum(indptr[:-1] + 1, y.size - 1)])
        raw["std"] = np.select([n_dy == 0, n_dy == 1], [single, first_dy], std)
    uid_list = uids.to_list()
    return {m: dict(zip(uid_list, v.tolist())) for m, v in raw.items()}


def _floor_series_scales(raw: Dict, floor_factor: float, kind: str = "Series") -> Dict:
    vals = np.array(list(raw.values()), dtype=float)
    global_median = float(np.median(vals)) if len(vals) > 0 else 1.0
    floor = max(floor_factor * global_median, 1e-8)
    floored: Dict = {}
    for uid, v in raw.items():
        effective = max(v, floor)
        if effective > v:
            warnings.warn(
                f"{kind} '{uid}' has near-zero scale estimate ({v:.2e}); "
                f"applying floor {floor:.2e}. Check for flat or constant series.",
                UserWarning,
                stacklevel=_user_warning_stacklevel(),
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
    # hash-based factorization: avoids np.unique's O(n log n) sort on string ids
    codes, uniques = pd.factorize(uid_arr)
    unique_norm = np.array([1.0 / source_scales[uid] for uid in uniques], dtype=float)
    norm_scales = unique_norm[codes]
    for model in model_names:
        vals = cs_df[model].to_numpy().astype(float) * norm_scales
        cs_df = ufp.assign_columns(cs_df, model, vals)
    return cs_df


def _rescale_interval_columns(
    fcst_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    sigma_tgt: np.ndarray,
) -> DFType:
    """Multiply interval half-widths by per-row σ̂_tgt, one assignment per column."""
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    for model_name in model_names:
        mean_arr = fcst_df[model_name].to_numpy().astype(float)
        for lv in level:
            for direction in ("lo", "hi"):
                col = f"{model_name}-{direction}-{lv}"
                offset = fcst_df[col].to_numpy().astype(float) - mean_arr
                fcst_df = ufp.assign_columns(
                    fcst_df, col, mean_arr + offset * sigma_tgt
                )
    return fcst_df


def _add_conformal_distribution_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
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
        mean = fcst_df[model].to_numpy().reshape(1, n_series, -1)
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
        quantiles = np.quantile(scores, cuts, axis=0)
        quantiles = quantiles.reshape(len(cuts), -1)
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        quantiles = np.vstack([mean - quantiles[::-1], mean + quantiles]).T
        columns = lo_cols + hi_cols
        fcst_df = ufp.assign_columns(fcst_df, columns, quantiles)
    return fcst_df


def _add_transfer_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    horizon: int,
    cs_h: int,
    method: str,
) -> DFType:
    """Add prediction intervals from conformity scores pooled over every calibration series.

    `method` is ``'signed'`` for signed scores (y − pred), or the
    ``PredictionIntervals.method`` of absolute scores. Signed scores use separate
    lower and upper quantiles, so that a systematically biased transferred model
    shifts the interval rather than merely widening it.

    When ``cs_h == 1`` but ``horizon > 1`` (calibration was done with 1-step ahead
    cross-validation), the same constant-width interval is broadcast across all
    horizon steps, matching the behaviour of the regular conformal methods in that
    case.

    With signed scores, emits UserWarning when the interval lies entirely above or
    below the point forecast (both quantiles have the same sign), indicating severe
    source-model bias on the target domain.
    """
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    # level is pre-sorted ascending (e.g. [80, 90, 95])
    # lo cols: reversed levels (widest first) → [lo-95, lo-90, lo-80]
    # hi cols: ascending levels (narrowest first) → [hi-80, hi-90, hi-95]
    lo_cuts = [((100 - lv) / 100) / 2 for lv in reversed(level)]
    hi_cuts = [1 - ((100 - lv) / 100) / 2 for lv in level]
    all_cuts = lo_cuts + hi_cuts

    for model in model_names:
        mean = fcst_df[model].to_numpy().ravel()
        scores = cs_df[model].to_numpy().astype(float).reshape(-1, cs_h)[:, :horizon]
        # (n_cuts, horizon), or (n_cuts, 1) broadcast to every step when cs_h=1
        if method == "signed":
            q_per_horizon = np.quantile(scores, all_cuts, axis=0)
            n_lo = len(lo_cuts)
            q_hi_vals = q_per_horizon[n_lo:]  # (n_hi_cuts, horizon)
            q_lo_vals = q_per_horizon[:n_lo]  # (n_lo_cuts, horizon)
            for i, lv in enumerate(level):
                q_hi_lv = q_hi_vals[i]  # upper quantile for this level
                q_lo_lv = q_lo_vals[-(i + 1)]  # matching lower quantile
                if np.all(q_hi_lv < 0):
                    warnings.warn(
                        f"Recalibration level={lv}: upper quantile is negative across "
                        "all horizon steps — interval lies entirely below point forecasts. "
                        "The transferred model systematically over-predicts on the target domain.",
                        UserWarning,
                        stacklevel=_user_warning_stacklevel(),
                    )
                elif np.all(q_lo_lv > 0):
                    warnings.warn(
                        f"Recalibration level={lv}: lower quantile is positive across "
                        "all horizon steps — interval lies entirely above point forecasts. "
                        "The transferred model systematically under-predicts on the target domain.",
                        UserWarning,
                        stacklevel=_user_warning_stacklevel(),
                    )
        elif method == "conformal_distribution":
            q_per_horizon = np.quantile(np.vstack([-scores, scores]), all_cuts, axis=0)
        else:
            q = np.quantile(scores, [lv / 100 for lv in level], axis=0)
            q_per_horizon = np.vstack([-q[::-1], q])

        # Apply: lo = pred + q_lo, hi = pred + q_hi, broadcast per horizon step
        n_target = len(mean) // horizon
        mean_2d = mean.reshape(n_target, horizon)
        quantiles = (
            (q_per_horizon[:, np.newaxis, :] + mean_2d[np.newaxis, :, :])
            .reshape(len(all_cuts), -1)
            .T
        )  # (n_target * horizon, n_cuts)

        out_cols = [f"{model}-lo-{lv}" for lv in reversed(level)] + [
            f"{model}-hi-{lv}" for lv in level
        ]
        fcst_df = ufp.assign_columns(fcst_df, out_cols, quantiles)

    return fcst_df


_INTERVAL_METHODS: Dict[str, Callable[..., Any]] = {
    "conformal_distribution": _add_conformal_distribution_intervals,
    "conformal_error": _add_conformal_error_intervals,
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
    signed: bool = False,
) -> DFType:
    """Compute conformity scores over CV folds.

    Args:
        cv_results: Cross-validation output with actual and predicted columns.
        model_names: Names of model columns to convert to errors.
        target_col: Name of the target column (dropped from output).
        signed: If True, compute signed errors (y - pred). If False (default),
            compute absolute errors |pred - target|.
    """
    for model in model_names:
        if signed:
            err = cv_results[target_col] - cv_results[model]
        else:
            err = abs(cv_results[model] - cv_results[target_col])
        cv_results = ufp.assign_columns(cv_results, model, err)
    return ufp.drop_columns(cv_results, target_col)


@dataclass
class TransferResult:
    """Calibration adapted to the series passed to ``MLForecast.transfer``."""

    cs_df: DFType
    h: int
    signed: bool = False
    target_scales: Optional[Dict[str, float]] = None  # uid -> sigma_tgt


@dataclass
class Calibration:
    """Conformity scores and per-series scales of the training series, computed at fit."""

    scores: DFType
    source_scales: Dict[str, Dict]  # estimator -> uid -> raw scale


@dataclass
class TransferState:
    """Adaptation to `df` of the transfer method chosen in ``MLForecast.transfer``."""

    result: Optional[TransferResult] = None


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
            stacklevel=_user_warning_stacklevel(),
        )
        return std_tgt / max(std_src, 1e-10)
    # Fallback 2: constant
    warnings.warn(
        "Both IQR and std of residuals near zero; scale ratio defaulting to 1.0.",
        UserWarning,
        stacklevel=_user_warning_stacklevel(),
    )
    return 1.0
