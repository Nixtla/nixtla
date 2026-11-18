"""End-to-end evaluation: ``evaluate``, ``evaluate_calibration``, ``Report``."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field

from .._model import ArrayModel
from ..base import CalibrationError, Forecast, Series
from ..methods.base import ConformalForecaster, ConformalMethod
from .coverage import (
    _coverage_indicator,
    _nanmean,
    coverage_by_horizon,
    coverage_by_series,
    coverage_per_cell,
    marginal_coverage,
)
from .method_diagnostics import method_state
from .scoring import mean_interval_width, winkler_score


class Report(ArrayModel):
    """
    Output of :func:`evaluate` and :func:`evaluate_calibration`.

    Carries generic coverage and scoring metrics plus method-specific state,
    along with the raw intervals/truths/points for callers who want to run
    further analysis.

    Notes
    -----
    ``extra="forbid"``, for the same reason as :class:`~conformalforecast.base.CalibrationResult`
    and :class:`~conformalforecast.base.PredictionResult`: a ``Report`` is
    returned to callers rather than constructed by them, and its only writers
    are :func:`evaluate` and :func:`evaluate_calibration` in this module — a
    mistyped keyword there is a bug worth failing on.

    Like those two result types, ``Report.model_json_schema()`` raises
    ``PydanticInvalidForJsonSchema`` rather than generating a schema: a bare
    ``NDArray`` field reaches pydantic as an ``IsInstanceSchema``, which has
    no JSON schema, and a result type is all array payload, so there is
    nothing left to hide behind ``SkipJsonSchema``. For the same reason it is
    excluded from the round-trip table in ``test_pydantic_conversion.py``.
    """

    model_config = ConfigDict(extra="forbid")

    # Identification
    method_name: str
    alpha: float
    n_holdout_samples: int

    # Generic: coverage
    marginal_coverage: float
    coverage_by_horizon: NDArray[np.floating]  # shape (horizon,)
    coverage_by_series: NDArray[np.floating]  # shape (n_series,)
    coverage_per_cell: NDArray[np.floating]  # shape (n_series, horizon)

    # Generic: scoring
    mean_interval_width: float
    mean_winkler_score: float

    # How much of the grid the metrics above were actually computed from.
    # These differ when a cell is not evaluable: a window that does not exist
    # for a series (ragged panel), or one an online method could not produce an
    # interval for (its warm-up). Such cells are excluded from every mean
    # rather than scored as misses.
    #
    # `marginal_coverage`, `coverage_*` and `mean_winkler_score` are averaged
    # over exactly these cells, since all three need an observed target.
    # `mean_interval_width` is not: a width is a property of the interval alone,
    # so it averages over every cell with two finite bounds, which can be a
    # larger set when a truth is missing but the forecast was still made.
    n_evaluated_cells: int = 0
    n_total_cells: int = 0

    # Method-specific
    method_state: dict[str, Any] = Field(default_factory=dict)

    # Raw arrays (for users wanting custom analysis)
    intervals: NDArray[np.floating] = Field(
        default_factory=lambda: np.empty((0, 0, 0, 2), dtype=np.float64)
    )
    truths: NDArray[np.floating] = Field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=np.float64)
    )
    points: NDArray[np.floating] = Field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=np.float64)
    )

    def summary(self) -> str:
        """Human-readable, multi-line summary for notebooks."""
        target = 1.0 - self.alpha
        cov_h = ", ".join(f"{v:.3f}" for v in self.coverage_by_horizon)

        lines = [
            f"Report — {self.method_name}",
            f"  target coverage: {target:.3f}",
            f"  marginal coverage: {self.marginal_coverage:.3f}",
            f"  coverage by horizon: [{cov_h}]",
            f"  mean width: {self.mean_interval_width:.3f}",
            f"  mean Winkler score: {self.mean_winkler_score:.3f}",
            f"  n_holdout_samples: {self.n_holdout_samples}",
        ]

        if self.n_evaluated_cells < self.n_total_cells:
            excluded = self.n_total_cells - self.n_evaluated_cells
            lines.append(
                f"  evaluated cells: {self.n_evaluated_cells} / {self.n_total_cells} "
                f"({excluded} excluded: warm-up or missing window)"
            )

        if "n_observations" in self.method_state:
            lines.append(f"  n_observations: {self.method_state['n_observations']}")
        if "alpha_t_minus_alpha" in self.method_state:
            drift = self.method_state["alpha_t_minus_alpha"]
            lines.append(
                f"  alpha_t drift: max {float(drift.max()):.3f}, min {float(drift.min()):.3f}"
            )
        if "effective_sample_size" in self.method_state:
            ess = self.method_state["effective_sample_size"]
            lines.append(f"  effective sample size: {ess:.1f}")
        if "n_regressors" in self.method_state:
            lines.append(
                f"  n_regressors: {self.method_state['n_regressors']} "
                f"({self.method_state.get('regressor_class', '?')})"
            )

        return "\n".join(lines)


def _build_report(
    *,
    method_name: str,
    alpha: float,
    intervals: NDArray[np.floating],
    truths: NDArray[np.floating],
    points: NDArray[np.floating],
    state: dict[str, Any],
) -> Report:
    """
    Compute the generic metrics on ``(intervals, truths)`` and wrap into a
    Report.

    Shared between :func:`evaluate` (Mode 2) and :func:`evaluate_calibration`
    (Mode 1) — both produce the same Report structure from the same kind of
    inputs; only the data source differs.
    """
    cov_marginal = marginal_coverage(intervals, truths)
    cov_h = coverage_by_horizon(intervals, truths)
    cov_s = coverage_by_series(intervals, truths)
    cov_cell = coverage_per_cell(intervals, truths)

    evaluable = ~np.isnan(_coverage_indicator(intervals, truths))

    width_per_cell = mean_interval_width(intervals)
    mean_width = float(_nanmean(width_per_cell))
    mean_winkler = float(_nanmean(winkler_score(intervals, truths, alpha)))

    return Report(
        method_name=method_name,
        alpha=alpha,
        n_holdout_samples=intervals.shape[1],
        marginal_coverage=cov_marginal,
        coverage_by_horizon=cov_h,
        coverage_by_series=cov_s,
        coverage_per_cell=cov_cell,
        mean_interval_width=mean_width,
        mean_winkler_score=mean_winkler,
        n_evaluated_cells=int(np.count_nonzero(evaluable)),
        n_total_cells=int(evaluable.size),
        method_state=state,
        intervals=intervals,
        truths=truths,
        points=points,
    )


def _require_fitted(fitted: object, fn: str) -> ConformalForecaster:
    """The argument, narrowed to a fitted object; a method is a CalibrationError."""
    if isinstance(fitted, ConformalForecaster):
        return fitted
    if isinstance(fitted, ConformalMethod):
        raise CalibrationError(
            f"{fn}() takes the fitted object a method's calibrate() / calibrate_cv() "
            f"returns, not the {type(fitted).__name__} method itself."
        )
    raise TypeError(f"{fn}() expects a ConformalForecaster, got {type(fitted).__name__}.")


def evaluate(
    fitted: ConformalForecaster,
    holdout_histories: list[Series],
    holdout_truths: Forecast,
    update_online: bool = True,
) -> Report:
    """
    Run a fitted conformal method on a holdout set and compute Mode 2 diagnostics.

    For each ``(history, truth)`` pair, calls ``fitted.predict(history)``. For
    online methods (``IS_ONLINE = True``), also calls
    ``fitted.update(prediction, truth)`` between iterations unless
    ``update_online=False`` -- which mutates ``fitted`` in place, exactly as a
    deployment would. Pass ``fitted.model_copy(deep=True)`` to keep the
    original untouched.

    Parameters
    ----------
    fitted : ConformalForecaster
        The object a method's ``calibrate`` / ``calibrate_cv`` returned.
    holdout_histories : list of Series
        Each ``Series`` has shape ``(n_series, T_i)``. The list length
        determines ``n_holdout_samples``.
    holdout_truths : Forecast, shape (n_series, n_holdout_samples, horizon)
        Ground truth for each history's forecast horizon.
    update_online : bool, default True
        If the method is online, whether to call ``fitted.update()`` after
        each prediction. Default ``True`` simulates real online deployment;
        set ``False`` to evaluate the calibration-time snapshot.

    Returns
    -------
    Report

    Raises
    ------
    CalibrationError
        If given the method rather than a fitted object.
    ValueError
        If ``holdout_truths`` is not 3-D, or if the number of samples does
        not match ``len(holdout_histories)``.
    """
    fitted = _require_fitted(fitted, "evaluate")

    truths_arr = np.asarray(holdout_truths, dtype=np.float64)
    if truths_arr.ndim != 3:
        raise ValueError(
            "holdout_truths must have shape (n_series, n_holdout_samples, horizon); "
            f"got {truths_arr.shape}."
        )
    n_holdout = len(holdout_histories)
    if truths_arr.shape[1] != n_holdout:
        raise ValueError(
            "len(holdout_histories) must equal holdout_truths.shape[1]; "
            f"got {n_holdout} histories and holdout_truths {truths_arr.shape}."
        )

    point_list: list[NDArray[np.floating]] = []
    interval_list: list[NDArray[np.floating]] = []

    is_online = type(fitted.method_).IS_ONLINE

    for i, history in enumerate(holdout_histories):
        result = fitted.predict(history)
        point_list.append(np.asarray(result.point, dtype=np.float64))
        interval_list.append(np.asarray(result.interval, dtype=np.float64))

        if is_online and update_online:
            truth_i = truths_arr[:, i : i + 1, :]
            fitted.update(result.point, truth_i)

    # Concatenate along the sample axis. Each predict() returns
    # (n_series, 1, horizon) and (n_series, 1, horizon, 2).
    points_arr = np.concatenate(point_list, axis=1)
    intervals_arr = np.concatenate(interval_list, axis=1)

    return _build_report(
        method_name=type(fitted.method_).__name__,
        alpha=float(fitted.method_.alpha),
        intervals=intervals_arr,
        truths=truths_arr,
        points=points_arr,
        state=method_state(fitted),
    )


def evaluate_calibration(fitted: ConformalForecaster) -> Report:
    """
    Compute diagnostics using the fitted object's stored calibration data (Mode 1).

    Reconstructs the intervals the method's own calibration implied at each
    window, then runs Mode 2-style diagnostics on them against the calibration
    truths.

    This differs from :func:`evaluate` in two ways:

    1. No holdout is required — uses stored calibration data.
    2. It is **in-sample**. The calibration windows are the ones the method's
       state was derived from, so coverage here is optimistic and is not a
       substitute for :func:`evaluate` on data the method has not seen.

    Each fitted class supplies the sequence via
    :meth:`~conformalforecast.methods.base.ConformalForecaster.calibration_intervals`.
    For methods whose offsets are a constant of the whole calibration set
    (Split, CQR, WeightedSplit, ACI, NexCP, AgACI) that is the fitted state
    applied to every window. SPCI overrides it, because its offsets are
    conditional on the preceding residual window and broadcasting the final one
    would measure nothing.

    Windows a method cannot evaluate — an online warm-up, or a window that does
    not exist for a series — come back with NaN bounds and are **excluded** from
    the averages rather than scored as misses. ``Report.n_evaluated_cells`` and
    ``Report.n_total_cells`` report how many were used.

    Parameters
    ----------
    fitted : ConformalForecaster
        The object a method's ``calibrate`` / ``calibrate_cv`` returned, with
        ``predictions_calibration_`` and ``truths_calibration_`` populated.

    Returns
    -------
    Report
        Same shape as the report from :func:`evaluate`. The ``method_name``
        field is suffixed with ``" (calibration)"`` to make the data source
        clear.

    Raises
    ------
    CalibrationError
        If given the method rather than a fitted object.
    RuntimeError
        If ``predictions_calibration_`` or ``truths_calibration_`` is None —
        an object returned by a bare ``fit`` never populates them.
    """
    fitted = _require_fitted(fitted, "evaluate_calibration")
    name = type(fitted.method_).__name__
    if fitted.predictions_calibration_ is None or fitted.truths_calibration_ is None:
        raise RuntimeError(
            f"This {name} was fitted through the array surface, so it holds no "
            "calibration forecasts to replay. Use calibrate() or calibrate_cv()."
        )

    predictions = np.asarray(fitted.predictions_calibration_, dtype=np.float64)
    truths = np.asarray(fitted.truths_calibration_, dtype=np.float64)

    intervals = np.asarray(fitted._calibration_intervals(), dtype=np.float64)

    return _build_report(
        method_name=f"{name} (calibration)",
        alpha=float(fitted.method_.alpha),
        intervals=intervals,
        truths=truths,
        points=predictions,
        state=method_state(fitted),
    )
