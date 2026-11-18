"""Split conformal prediction: the reference single-layer method."""

from __future__ import annotations

import warnings
from typing import ClassVar

import numpy as np
from pydantic import Field, model_validator

from .._quantiles import conformal_quantile
from ..base import (
    CalibrationResult,
    Exog,
    Forecast,
    ScoreFunction,
    as_forecast,
)
from ..nonconformity.absolute import AbsoluteResidual
from ..nonconformity.signed import SignedResidual
from .base import ConformalMethod
from .fixed_offset import FixedOffsetForecaster


class SplitConformal(ConformalMethod[FixedOffsetForecaster]):
    """
    Classic split conformal prediction adapted for panel time series.

    See :class:`~conformalforecast.methods.base.ConformalMethod` for the
    calibrate/predict lifecycle and the ``calibrate`` / ``calibrate_cv`` split.

    Algorithm
    ---------
    1. **fit**: score ``(predictions, truths)`` with the nonconformity score
       and store the per-(series, horizon) exact conformal quantile
       (``k = ceil((1 - alpha)(n + 1))`` order statistic, NaN-aware). When
       ``symmetric=False``, the score is signed and two one-sided conformal
       quantiles are taken instead — one of the signed residual, one of its
       negation — so the interval can shift as well as widen.
    2. **interval** (on the returned
       :class:`~conformalforecast.methods.fixed_offset.FixedOffsetForecaster`):
       apply ``score_fn_.invert`` with the stored quantile.

    Any adapter works — no special capabilities required. Bind one with
    ``calibrate(forecaster, histories, truths)`` or
    ``calibrate_cv(forecaster, n_windows)``. Construction is keyword-only:
    ``SplitConformal(alpha=0.1)``.

    Parameters
    ----------
    alpha : float
        Miscoverage level in (0, 1). The target coverage is ``1 - alpha``.
    symmetric : bool, default True
        True (the default) gives an equal-width interval ``ŷ ± q``, with
        ``q`` one conformal quantile of the absolute residual. It targets
        total coverage ``1 - alpha`` and makes no per-tail guarantee: with
        skewed residuals one tail can take nearly all the misses. False
        scores signed residuals and sets each tail explicitly through
        ``alpha_lo`` / ``alpha_hi``, which both become mandatory.
        ``symmetric=False`` with ``alpha_lo=alpha/2, alpha_hi=1 - alpha/2``
        is the equal-tailed interval: ``alpha/2`` per tail, and it can shift
        as well as widen. It needs at least ``ceil(2 / alpha)`` calibration
        windows per series instead of ``ceil(1 / alpha)``, and its upper
        bound on coverage is slightly looser.
    alpha_lo : float, optional
        Lower-tail miscoverage budget, used only when ``symmetric=False``,
        where it is required alongside ``alpha_hi``. Under exchangeability
        the lower tail misses at most ``alpha_lo`` of the time.
    alpha_hi : float, optional
        Upper-tail level, used only when ``symmetric=False``, where it is
        required alongside ``alpha_lo``. Under exchangeability the upper
        tail misses at most ``1 - alpha_hi`` of the time. Both levels must
        be left None when ``symmetric=True``, even when they equal
        ``(alpha/2, 1 - alpha/2)``: a pair specifies per-tail coverage, which
        the equal-width symmetric interval does not provide. This
        deliberately differs from :class:`ConformalizedQuantileRegression`,
        where the pair only names the quantiles requested from the
        forecaster.

        Paths that skip validation (``model_copy(update=...)``, attribute
        assignment, ``model_construct``) can leave a stale ``alpha_lo`` /
        ``alpha_hi`` after switching to ``symmetric=True``. ``fit`` ignores
        them, but ``model_validate(m.model_dump())`` then refuses the
        config; rebuild through the constructor or ``model_validate``.
    seed : int, default 0
        Unused (kept for a uniform constructor signature); split conformal
        is deterministic by construction.
    score : ScoreFunction or None, default None
        Defaults to :class:`AbsoluteResidual` when ``symmetric=True`` and
        :class:`SignedResidual` otherwise. ``symmetric=False`` requires a
        ``SignedResidual`` (or a subclass): the two tails are separate
        one-sided quantiles, so the score must keep the residual's sign.
    """

    REQUIRED_CAPABILITIES: ClassVar[tuple[type, ...]] = ()
    IS_ONLINE: ClassVar[bool] = False

    symmetric: bool = Field(
        default=True,
        description=(
            "True (the default) gives an equal-width interval, prediction ± q, with "
            "q one conformal quantile of the absolute residual. It targets total "
            "coverage 1 - alpha and makes no per-tail guarantee: with skewed "
            "residuals one tail can take nearly all the misses. False scores signed "
            "residuals and sets each tail explicitly via alpha_lo / alpha_hi, which "
            "both become mandatory. symmetric=False with alpha_lo=alpha/2, "
            "alpha_hi=1-alpha/2 is the equal-tailed interval: alpha/2 per tail, and "
            "it can shift as well as widen. It needs at least ceil(2/alpha) "
            "calibration windows per series instead of ceil(1/alpha), and has a "
            "slightly looser upper bound on coverage."
        ),
    )
    alpha_lo: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description=(
            "Lower-tail miscoverage budget, e.g. 0.05: under exchangeability the "
            "lower tail misses at most alpha_lo of the time. Used only when "
            "symmetric=False, where it is mandatory alongside alpha_hi and the pair "
            "must satisfy 0 < alpha_lo < 0.5 < alpha_hi < 1 with "
            "alpha_lo + (1 - alpha_hi) equal to alpha. Must be left None when "
            "symmetric=True, even when it equals alpha/2: a pair specifies per-tail "
            "coverage, which the equal-width symmetric interval does not provide. "
            "This deliberately differs from ConformalizedQuantileRegression, where "
            "the pair only names the quantiles requested from the forecaster."
        ),
    )
    alpha_hi: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description=(
            "Upper-tail level, e.g. 0.95: under exchangeability the upper tail "
            "misses at most 1 - alpha_hi of the time. Bound to alpha_lo by "
            "alpha_lo + (1 - alpha_hi) == alpha. Used only when symmetric=False, "
            "where it is mandatory. Must be left None when symmetric=True, even "
            "when it equals 1 - alpha/2: a pair specifies per-tail coverage, which "
            "the equal-width symmetric interval does not provide. This deliberately "
            "differs from ConformalizedQuantileRegression, where the pair only "
            "names the quantiles requested from the forecaster."
        ),
    )

    @model_validator(mode="after")
    def _resolve_levels(self) -> SplitConformal:
        """Enforce the symmetric/asymmetric level rule."""
        alpha_lo, alpha_hi = self.alpha_lo, self.alpha_hi
        if self.symmetric:
            if alpha_lo is not None or alpha_hi is not None:
                raise ValueError(
                    "alpha_lo and alpha_hi cannot be specified when symmetric=True. "
                    "symmetric=True builds an equal-width interval with no per-tail "
                    "levels. Remove them, or for an equal-tailed interval use "
                    f"symmetric=False, alpha_lo={self.alpha / 2:g}, "
                    f"alpha_hi={1 - self.alpha / 2:g}."
                )
            return self
        if alpha_lo is None or alpha_hi is None:
            raise ValueError("alpha_lo and alpha_hi must both be provided when symmetric=False.")
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

    def _default_score(self) -> ScoreFunction:
        """A signed residual is what carries an asymmetric interval."""
        return AbsoluteResidual() if self.symmetric else SignedResidual()

    @model_validator(mode="after")
    def _require_signed_score_when_asymmetric(self) -> SplitConformal:
        """Reject an unsigned score when ``symmetric=False``."""
        score = self.score
        if not self.symmetric and score is not None and not isinstance(score, SignedResidual):
            raise ValueError(
                "symmetric=False requires a signed score, got "
                f"{type(score).__name__}. The two tails are separate one-sided "
                "quantiles, so the score must keep the sign of the residual for "
                "them to mean anything. Pass SignedResidual() (the default when "
                "symmetric=False) or a subclass of it."
            )
        return self

    def fit(
        self,
        predictions: Forecast,
        truths: Forecast,
        *,
        X: Exog | None = None,
    ) -> FixedOffsetForecaster:
        """
        Fit the conformal quantile on calibration arrays.

        Parameters
        ----------
        predictions : Forecast, shape (n_series, n_windows, horizon)
            NaN cells mark windows that do not exist for a series.
        truths : Forecast, shape (n_series, n_windows, horizon)
        X : Exog or None, shape (n_series, n_windows, horizon, n_features)
            Ignored by this method.

        Returns
        -------
        FixedOffsetForecaster
            ``score_quantile_`` has shape ``(n_series, horizon)``, or
            ``(n_series, horizon, 2)`` — last axis ``(lower_offset,
            upper_offset)`` — when ``symmetric=False``.
            ``calibration_.diagnostics`` contains ``"score_quantile"``, a
            defensive copy of it.

        Raises
        ------
        ValueError
            If array shapes are invalid or mismatched, or if ``symmetric=False``
            but ``alpha_lo`` or ``alpha_hi`` is unset, which only a path that
            skips validation (``model_copy``, attribute assignment,
            ``model_construct``) can produce.
        InsufficientCalibrationSamples
            When ``symmetric=True``, if any series has fewer than
            ``ceil(1 / alpha)`` effective (non-NaN) calibration windows. When
            ``symmetric=False``, the lower and upper offsets are each their
            own one-sided conformal quantile — at ``alpha_lo`` and
            ``1 - alpha_hi`` respectively — so each enforces its own floor,
            and the binding requirement is the stricter
            ``max(ceil(1 / alpha_lo), ceil(1 / (1 - alpha_hi)))`` rather than
            ``ceil(1 / alpha)``.
        """
        preds = as_forecast(predictions, "predictions")
        obs = as_forecast(truths, "truths")
        if preds.shape != obs.shape:
            raise ValueError(
                f"predictions and truths must have the same shape, "
                f"got {preds.shape} and {obs.shape}"
            )

        score_fn = self._bind_score()
        score_fn.fit(preds, obs)
        # NaN cells propagate through scoring — never imputed.
        scores = score_fn.score(preds, obs)

        if self.symmetric:
            score_quantile = conformal_quantile(scores, self.alpha)
        else:
            if self.alpha_lo is None or self.alpha_hi is None:
                raise ValueError(
                    "symmetric=False needs alpha_lo and alpha_hi, but this config "
                    "skipped validation (e.g. model_copy, attribute assignment, "
                    "model_construct). Pass alpha_lo and alpha_hi, or set "
                    "symmetric=True, and rebuild it with the constructor or "
                    "model_validate."
                )
            # Lower offset: the alpha_lo-quantile of signed residuals, obtained
            # as the negated upper conformal quantile of the negated scores.
            # Each call enforces its own ceil(1/level) floor (see fit's Raises
            # docstring), so `context` names the knob a resulting error is
            # really about — and the value the user set, which for the upper
            # tail is not the `alpha` the message would otherwise quote.
            lower = -conformal_quantile(
                -scores, self.alpha_lo, context=f" (from alpha_lo={self.alpha_lo})"
            )
            upper = conformal_quantile(
                scores, 1.0 - self.alpha_hi, context=f" (from alpha_hi={self.alpha_hi})"
            )
            score_quantile = np.stack([lower, upper], axis=-1)
            if np.any(lower > 0.0) or np.any(upper < 0.0):
                warnings.warn(
                    "Asymmetric split conformal produced an interval lying entirely "
                    "on one side of the point forecast. The forecaster is biased on "
                    "this calibration data; the interval is shifting rather than "
                    "widening to compensate.",
                    UserWarning,
                    stacklevel=2,
                )
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
