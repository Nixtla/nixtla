"""Conformalized Quantile Regression (Romano et al. 2019) for panel time series."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from .._quantiles import conformal_quantile
from ..base import (
    CalibrationResult,
    Exog,
    Forecast,
    ScoreFunction,
)
from ..capabilities import SupportsQuantiles
from ..nonconformity.quantile import QuantileScore
from .base import ConformalMethod
from .fixed_offset import FixedOffsetForecaster, _as_quantile_predictions


class ConformalizedQuantileRegression(ConformalMethod[FixedOffsetForecaster]):
    """
    Conformalized Quantile Regression (Romano, Patterson, Candès 2019)
    adapted for panel time series.

    Wraps a quantile-producing forecaster and calibrates its quantile interval
    so that empirical coverage matches the target ``1 - alpha``.

    See :class:`~conformalforecast.methods.base.ConformalMethod` for the
    calibrate/predict lifecycle and the ``calibrate`` / ``calibrate_cv`` split.

    Algorithm
    ---------
    1. **fit**: score the calibration quantile pair ``(q_lo, q_hi)`` against
       ``truths`` with :class:`QuantileScore` (``E = max(q_lo - y, y - q_hi)``)
       and store the exact per-(series, horizon) conformal quantile
       (``k = ceil((1 - alpha)(n + 1))`` order statistic, NaN-aware).
    2. **interval** (on the returned
       :class:`~conformalforecast.methods.fixed_offset.FixedOffsetForecaster`):
       widen a new quantile pair by that stored quantile.

    Unlike the other methods, the ``predictions`` that ``fit`` and the fitted
    ``interval`` take are a **quantile cube**, shape
    ``(n_series, n_windows | n_samples, horizon, 2)`` with last axis
    ``(q_lo, q_hi)``, not point forecasts. ``CONSUMES_QUANTILES`` advertises
    this so the orchestration entry points query the forecaster for quantiles.
    Choosing *which* quantile levels to query at is this class's
    responsibility (from ``alpha`` / ``symmetric`` / ``alpha_lo`` /
    ``alpha_hi``), exposed as :attr:`quantile_levels`.

    The bound adapter must implement :class:`SupportsQuantiles`, checked when
    it is bound by ``calibrate(forecaster, histories, truths)`` or
    ``calibrate_cv(forecaster, n_windows)``.

    The score must be a :class:`QuantileScore` **or a subclass** (the default is
    a plain ``QuantileScore``); anything else is a ``pydantic.ValidationError``
    at construction. A point score would fail anyway, on the trailing axis of
    the quantile cube, but with an opaque broadcast error. The gate also
    **narrows** what is accepted: a custom score honouring the 4-D layout must
    subclass ``QuantileScore``.

    Construction is a pydantic model, so it is **keyword-only**
    (``ConformalizedQuantileRegression(alpha=0.1)``) and an invalid
    ``alpha`` / ``symmetric`` / level combination raises
    ``pydantic.ValidationError`` (a subclass of ``ValueError``).

    Parameters
    ----------
    alpha : float
        Target miscoverage in ``(0, 1)``. The target coverage is ``1 - alpha``.
    symmetric : bool, default True
        If True, the forecaster is queried at ``(alpha/2, 1 - alpha/2)``,
        derived from ``alpha`` by :attr:`quantile_levels`, and ``alpha_lo`` /
        ``alpha_hi`` must be None. A matching pair is accepted and stored as
        None, because it restates the quantiles already requested. If False,
        the user must supply both ``alpha_lo`` and ``alpha_hi``.
    alpha_lo : float, optional
        Lower quantile level requested from the forecaster. Required when
        ``symmetric=False``.
    alpha_hi : float, optional
        Upper quantile level requested from the forecaster. Required when
        ``symmetric=False``.

        The levels shape the interval's width and position, not its
        validity. One conformal quantile of the CQR score is added to both
        sides, so the guarantee is joint coverage ``1 - alpha`` (under
        exchangeability) whatever the levels are, and there is no per-tail
        coverage guarantee. The constraints
        ``alpha_lo + (1 - alpha_hi) == alpha`` and
        ``0 < alpha_lo < 0.5 < alpha_hi < 1`` are a convention that keeps the
        requested quantiles consistent with ``alpha``; validity does not need
        them.
    seed : int, default 0
        Inert (kept for a uniform constructor signature); CQR is deterministic.
    score : QuantileScore or None, default None
        Defaults to :class:`QuantileScore`.
    """

    REQUIRED_CAPABILITIES: ClassVar[tuple[type, ...]] = (SupportsQuantiles,)
    CONSUMES_QUANTILES: ClassVar[bool] = True
    SCORE_TYPE: ClassVar[type[ScoreFunction]] = QuantileScore
    IS_ONLINE: ClassVar[bool] = False

    symmetric: bool = Field(
        default=True,
        description=(
            "Whether the two quantile levels requested from the forecaster sit "
            "symmetrically around the median. True (the default) requests "
            "alpha / 2 and 1 - alpha / 2 and requires alpha_lo / alpha_hi to be "
            "omitted, or set within np.isclose of those derived values -- a "
            "matching pair restates the quantiles already requested, so it is "
            "accepted and stored as None, while a contradictory one is refused. "
            "False takes the pair from alpha_lo / alpha_hi, which both become "
            "mandatory and, by convention, must satisfy "
            "0 < alpha_lo < 0.5 < alpha_hi < 1 with alpha_lo + (1 - alpha_hi) "
            "equal to alpha. Either way the guarantee is joint coverage "
            "1 - alpha (under exchangeability), with no per-tail guarantee: one "
            "conformal quantile of the CQR score is added to both sides, so the "
            "levels shape the interval's width and position, not its validity. "
            "True cannot express an off-centre band, so on a skewed target the "
            "conformal step widens both sides, giving intervals wider than "
            "necessary on the well-behaved side; False can place the band "
            "off-centre, but the levels are yours to pick by hand -- nothing in "
            "the method learns them from the data. "
            "Note that False is currently unreachable through any shipped "
            "adapter: all three Nixtla adapters request quantiles through the "
            "integer level API, where level=90 means the 5th and 95th "
            "percentiles inseparably, so an asymmetric pair is rejected "
            "before the model is consulted. It works only behind a custom "
            "SupportsQuantiles adapter, or on the array surface where you "
            "pass the quantile forecasts yourself."
        ),
    )
    alpha_lo: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description=(
            "Lower quantile level to request from the forecaster, as a "
            "probability: 0.02 asks for the 2nd percentile, which becomes the "
            "pre-calibration lower bound. Must be omitted when "
            "symmetric=True, where the lower level is instead derived from "
            "alpha as alpha / 2 and this field stays None -- a value within "
            "np.isclose tolerance of the derived one is accepted and stored "
            "as None, anything further away is refused; must "
            "be given together with alpha_hi when symmetric=False. By "
            "convention the pair satisfies 0 < alpha_lo < 0.5 < alpha_hi < 1 "
            "and alpha_lo + (1 - alpha_hi) == alpha, which keeps the requested "
            "quantiles consistent with alpha but is not needed for validity. "
            "The level shapes the interval's width and position only: the "
            "guarantee is joint coverage 1 - alpha whatever the levels are, "
            "with no per-tail guarantee. It exists so the pre-calibration band "
            "can sit off-centre on a skewed target."
        ),
    )
    alpha_hi: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description=(
            "Upper quantile level to request from the forecaster, as a "
            "probability: 0.92 asks for the 92nd percentile, which becomes "
            "the pre-calibration upper bound. Same conditional rule as "
            "alpha_lo -- when symmetric=True the upper level is derived from "
            "alpha as 1 - alpha / 2 and this field stays None (only a value "
            "within np.isclose tolerance of the derived one is accepted, and is "
            "stored as None; anything further away is refused), and it is "
            "mandatory alongside alpha_lo when symmetric=False. The constraints "
            "0 < alpha_lo < 0.5 < alpha_hi < 1 and "
            "alpha_lo + (1 - alpha_hi) == alpha are a convention, not a validity "
            "requirement: the level shapes the interval's width and position, "
            "and the guarantee is joint coverage 1 - alpha whatever it is."
        ),
    )

    @model_validator(mode="after")
    def _resolve_levels(self) -> ConformalizedQuantileRegression:
        """Enforce the symmetric/asymmetric level rule."""
        alpha_lo, alpha_hi = self.alpha_lo, self.alpha_hi
        if self.symmetric:
            derived_lo, derived_hi = self.quantile_levels
            # Levels that agree with symmetric derivation are accepted rather than
            # refused: the guard exists to catch a caller who set explicit levels
            # believing symmetric=True would honour them, and there is no such
            # confusion when the values are the ones derivation produces anyway.
            consistent = (
                alpha_lo is not None
                and alpha_hi is not None
                and bool(np.isclose(alpha_lo, derived_lo))
                and bool(np.isclose(alpha_hi, derived_hi))
            )
            if not consistent and (alpha_lo is not None or alpha_hi is not None):
                raise ValueError(
                    "alpha_lo and alpha_hi cannot be specified when symmetric=True. "
                    "Either remove them or set symmetric=False."
                )
            # Normalized to None: the pair is derived from alpha by quantile_levels,
            # so storing it would go stale under model_copy(update={"alpha": ...})
            # and would pin a dumped config to the alpha it was dumped at.
            # validate_assignment=False on ArrayModel, so these do not re-enter
            # validation.
            self.alpha_lo = None
            self.alpha_hi = None
        else:
            if alpha_lo is None or alpha_hi is None:
                raise ValueError(
                    "alpha_lo and alpha_hi must both be provided when symmetric=False."
                )
            if not (0.0 < alpha_lo < 0.5 < alpha_hi < 1.0):
                raise ValueError(
                    "alpha_lo and alpha_hi must satisfy "
                    f"0 < alpha_lo < 0.5 < alpha_hi < 1; got alpha_lo={alpha_lo}, "
                    f"alpha_hi={alpha_hi}."
                )
            combined = alpha_lo + (1.0 - alpha_hi)
            if not np.isclose(combined, self.alpha):
                raise ValueError(
                    "alpha_lo + (1 - alpha_hi) must equal alpha. "
                    f"Got alpha_lo={alpha_lo}, alpha_hi={alpha_hi}, "
                    f"sum={combined}, alpha={self.alpha}."
                )
        return self

    @property
    def quantile_levels(self) -> NDArray[np.floating]:
        """
        The quantile levels the forecaster is queried at.

        Returns
        -------
        NDArray, shape (2,)
            ``(alpha / 2, 1 - alpha / 2)`` when ``symmetric=True``, derived from
            ``alpha`` on every access; ``(alpha_lo, alpha_hi)`` otherwise.

        Raises
        ------
        ValueError
            If ``symmetric=False`` but ``alpha_lo`` or ``alpha_hi`` is unset. Only a
            path that skips validation (``model_copy``, attribute assignment,
            ``model_construct``) can produce this, and such paths can leave the
            level fields inconsistent the other way too: switching to
            ``symmetric=True`` keeps a stale ``alpha_lo`` / ``alpha_hi`` that
            ``model_validate(m.model_dump())`` then refuses. Rebuild through the
            constructor or ``model_validate``.
        """
        if self.symmetric:
            return np.array([self.alpha / 2.0, 1.0 - self.alpha / 2.0], dtype=np.float64)
        if self.alpha_lo is None or self.alpha_hi is None:
            raise ValueError(
                "symmetric=False needs alpha_lo and alpha_hi, but this config "
                "skipped validation (e.g. model_copy, attribute assignment, "
                "model_construct). Pass alpha_lo and alpha_hi, or set "
                "symmetric=True, and rebuild it with the constructor or "
                "model_validate."
            )
        return np.array([self.alpha_lo, self.alpha_hi], dtype=np.float64)

    def _default_score(self) -> ScoreFunction:
        return QuantileScore()

    def _calibration_quantiles(self) -> NDArray[np.floating]:
        """
        Quantile levels to query the forecaster at.

        Returns
        -------
        NDArray, shape (2,)
            See ``quantile_levels``.
        """
        return self.quantile_levels

    def fit(
        self,
        predictions: Forecast,
        truths: Forecast,
        *,
        X: Exog | None = None,
    ) -> FixedOffsetForecaster:
        """
        Fit the conformal quantile on calibration quantile forecasts.

        Parameters
        ----------
        predictions : NDArray, shape (n_series, n_windows, horizon, 2)
            Quantile forecasts; last axis is ``(q_lo, q_hi)``. NaN cells mark
            windows that do not exist.
        truths : Forecast, shape (n_series, n_windows, horizon)
            Realized values.
        X : Exog or None, shape (n_series, n_windows, horizon, n_features)
            Ignored by this method.

        Returns
        -------
        FixedOffsetForecaster
            ``score_quantile_`` has shape ``(n_series, horizon)``;
            ``calibration_.diagnostics`` carries ``"score_quantile"``, a
            defensive copy of it.

        Raises
        ------
        ValueError
            If ``predictions`` is malformed or shape-mismatched with ``truths``.
        InsufficientCalibrationSamples
            If any series has fewer than ``ceil(1 / alpha)`` effective windows.
        """
        qp = _as_quantile_predictions(predictions)
        obs = np.asarray(truths)
        if not np.issubdtype(obs.dtype, np.floating):
            obs = obs.astype(np.float64)
        if obs.shape != qp.shape[:-1]:
            raise ValueError(
                f"truths shape {obs.shape} must equal predictions.shape[:-1] {qp.shape[:-1]}"
            )

        score_fn = self._bind_score()
        score_fn.fit(qp, obs)
        # NaN cells propagate through scoring — never imputed.
        scores = score_fn.score(qp, obs)  # (n_series, n_windows, horizon)
        score_quantile = conformal_quantile(scores, self.alpha)
        n_calibration_samples = np.sum(~np.isnan(scores), axis=1).min(axis=1)
        return FixedOffsetForecaster(
            method_=self._snapshot(),
            score_fn_=score_fn,
            n_calibration_samples_=n_calibration_samples,
            calibration_=CalibrationResult(
                n_calibration_samples=n_calibration_samples.copy(),
                diagnostics={"score_quantile": score_quantile.copy()},
            ),
            score_quantile_=score_quantile,
        )
