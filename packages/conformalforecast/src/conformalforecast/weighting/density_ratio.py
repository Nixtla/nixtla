"""Classifier-based density-ratio weights for covariate shift."""

from __future__ import annotations

import warnings
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

from ..base import Exog
from .base import DensityRatioEstimator, WeightFunction


class DensityRatio(WeightFunction):
    """
    Density-ratio weights estimated by a binary classifier.

    Source rows are labelled 0 and target rows 1; the fitted classifier's odds
    ``p / (1 - p)`` estimate ``d_target(x) / d_source(x)``, the importance
    weight for conformal prediction under covariate shift (Tibshirani et al.,
    2019).

    Calibration weights are computed **out-of-fold**: a calibration point's
    weight never comes from a model that trained on it. Test weights come from
    a final model fitted on all data.

    Parameters
    ----------
    estimator : DensityRatioEstimator
        Any object with ``fit(X, y)`` and ``predict_proba(X) -> (n, 2)``.
        Required — this package ships no default, so it gains no dependency.
        A fresh copy is **not** made; supply a distinct instance per
        ``DensityRatio`` if the estimator is stateful.
    cv : int, default 5
        Cross-fitting folds. Values below 2 disable cross-fitting, in which
        case calibration weights come from a model fitted on all data (faster,
        but the weights are optimistically biased).
    clip_quantile : float or None, default 0.99
        Upper quantile at which weights are clipped, so one extreme weight
        cannot dominate the weighted quantile. None disables clipping.
    standardize : bool, default True
        Standardize features to zero mean and unit variance before fitting.
    seed : int, default 0
        Seed for the fold assignment. Never None.

    Raises
    ------
    pydantic.ValidationError
        If ``cv`` is negative, or ``clip_quantile`` is outside (0, 1] — a
        subclass of ``ValueError``.
    """

    CONSUMES_TARGET_X: ClassVar[bool] = True

    estimator: DensityRatioEstimator = Field(
        description=(
            "Binary probabilistic classifier used to estimate the density ratio. "
            "Any object with fit(X, y) and predict_proba(X) -> (n, 2) works: "
            "scikit-learn, XGBoost, LightGBM and CatBoost classifiers all plug in "
            "unwrapped. Required -- this package ships no default, so it gains no "
            "dependency. A fresh copy is not made; supply a distinct instance per "
            "DensityRatio if the estimator is stateful."
        )
    )
    cv: int = Field(
        default=5,
        ge=0,
        description=(
            "Cross-fitting folds for the out-of-fold calibration weights. Values "
            "below 2 disable cross-fitting, in which case the weights come from a "
            "model fitted on all data -- faster, but optimistically biased."
        ),
    )
    clip_quantile: float | None = Field(
        default=0.99,
        gt=0.0,
        le=1.0,
        description=(
            "Upper quantile of the source weight distribution at which weights are "
            "clipped, so one extreme weight cannot dominate the weighted quantile. "
            "None disables clipping."
        ),
    )
    standardize: bool = Field(
        default=True,
        description="Standardize features to zero mean and unit variance before fitting.",
    )
    seed: int = Field(default=0, ge=0, description="Seed for the fold assignment. Never None.")

    mean_: SkipJsonSchema[NDArray[np.floating] | None] = None
    scale_: SkipJsonSchema[NDArray[np.floating] | None] = None
    clip_value_: SkipJsonSchema[float] = float("inf")
    calibration_weights_: SkipJsonSchema[NDArray[np.floating] | None] = None

    @staticmethod
    def _flatten(X: Exog) -> NDArray[np.floating]:
        """Collapse (n_series, n_obs, horizon, n_features) to (n_rows, n_features)."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 4:
            raise ValueError(
                "exogenous features must be 4-D "
                f"(n_series, n_obs, horizon, n_features), got shape {arr.shape}"
            )
        return arr.reshape(-1, arr.shape[-1])

    def _standardize_fit(self, X: NDArray[np.floating]) -> None:
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale > 0.0, scale, 1.0)

    def _standardize_apply(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if not self.standardize:
            return X
        assert self.mean_ is not None and self.scale_ is not None  # set by _standardize_fit
        result: NDArray[np.floating] = (X - self.mean_) / self.scale_
        return result

    def _odds(self, proba: NDArray[np.floating]) -> NDArray[np.floating]:
        p = np.clip(np.asarray(proba, dtype=np.float64)[:, 1], 1e-12, 1 - 1e-12)
        return p / (1.0 - p)

    def _source_rows(self) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
        """
        Source feature rows that a classifier may be fitted on.

        A ragged panel pads the windows a series does not have, so the features
        in those cells are NaN. They are dropped here rather than handed to the
        estimator, which would fail inside the third-party library. NaN on a
        window that *does* exist is bad data, not padding, and is rejected.

        Returns
        -------
        rows : NDArray, shape (n_kept, n_features)
            Finite feature rows, in flattened calibration order.
        keep : NDArray of bool, shape (n_series * n_windows * horizon,)
            Which flattened cells ``rows`` came from.

        Raises
        ------
        ValueError
            If a cell the valid mask marks as a real window has a non-finite
            feature, or if no source row survives.
        """
        assert self.source_X_ is not None  # checked by fit_target
        flat = self._flatten(self.source_X_)
        finite = np.all(np.isfinite(flat), axis=1)
        keep = self._valid_rows()
        bad = np.flatnonzero(keep & ~finite)
        if bad.size:
            # Without a mask every cell counts as a real window, so a perfectly
            # good ragged panel lands here too. Say which of the two it is
            # rather than telling a caller their data is bad when it is not.
            cause = (
                "no valid mask was supplied to fit_source, so every cell counts as a "
                "real window. Pass fit_source(..., valid=~np.isnan(scores)) if these "
                "rows are a ragged panel's left padding"
                if self.valid_ is None
                else "in a ragged panel NaN is only allowed where the window itself "
                "is absent (left padding); interior or trailing NaN is an error"
            )
            raise ValueError(
                f"{bad.size} source exogenous rows are non-finite on calibration "
                f"windows that do exist: {cause}. First offending flat row index: "
                f"{int(bad[0])}."
            )
        keep = keep & finite
        if not keep.any():
            raise ValueError(
                "no source exogenous rows remain after dropping absent calibration "
                "windows, so the density ratio cannot be estimated."
            )
        return flat[keep], keep

    def fit_target(self, X: Exog | None) -> None:
        """
        Fit the density-ratio classifier against target-domain features.

        Parameters
        ----------
        X : Exog, shape (n_series, n_samples, horizon, n_features)

        Raises
        ------
        ValueError
            If source features were not supplied at ``fit_source``; if the
            target feature count disagrees with the source; if a source cell
            the valid mask marks as a real window is non-finite; or if the
            target features contain any non-finite value.
        """
        if X is None:
            raise ValueError("DensityRatio.fit_target requires target exogenous features.")
        if self.source_X_ is None:
            raise ValueError(
                "DensityRatio requires source exogenous features. Call the method's "
                "fit(X=...) or calibrate(X=...) before adapt(weighting=...)."
            )
        src, keep = self._source_rows()
        tgt = self._flatten(X)
        if src.shape[1] != tgt.shape[1]:
            raise ValueError(
                f"source and target n_features disagree: {src.shape[1]} vs {tgt.shape[1]}"
            )
        if not np.all(np.isfinite(tgt)):
            raise ValueError(
                "target exogenous features contain non-finite values. Unlike source "
                "calibration windows, every target sample is a step to be predicted, "
                "so there is no padding for NaN to mean."
            )

        stacked = np.vstack([src, tgt])
        labels = np.concatenate(
            [np.zeros(len(src), dtype=np.int64), np.ones(len(tgt), dtype=np.int64)]
        )
        if self.standardize:
            self._standardize_fit(stacked)
        stacked_s = self._standardize_apply(stacked)

        n_src = len(src)
        # A class with fewer rows than ``cv`` cannot be spread over every fold, so
        # some fold would hold all of it and leave its complement single-class.
        n_folds = min(self.cv, int(np.bincount(labels, minlength=2).min()))
        if n_folds < self.cv:
            warnings.warn(
                f"cv={self.cv} was capped to {n_folds} folds: the rarer of the source "
                "and target row counts is smaller than cv. "
                + (
                    "Cross-fitting is disabled, so the weights come from a model "
                    "fitted on all data and are optimistically biased."
                    if n_folds < 2
                    else "The out-of-fold weights use fewer folds than requested."
                ),
                UserWarning,
                stacklevel=2,
            )
        if n_folds >= 2:
            oof = self._cross_fit(stacked_s, labels, n_folds)[:n_src]
            # Final model on all data, used for test-point weights.
            self.estimator.fit(stacked_s, labels)
        else:
            self.estimator.fit(stacked_s, labels)
            oof = self._odds(self.estimator.predict_proba(stacked_s[:n_src]))

        self.clip_value_ = (
            float(np.quantile(oof, self.clip_quantile))
            if self.clip_quantile is not None
            else np.inf
        )
        n_series, n_windows, horizon = self._require_source()
        # Cells whose window is absent get weight 0. The weighted quantile
        # zeroes the weight of a NaN score anyway, so this only makes the
        # dropped rows explicit rather than leaving them unwritten.
        weights = np.zeros(keep.size, dtype=np.float64)
        weights[keep] = np.clip(oof, 0.0, self.clip_value_)
        self.calibration_weights_ = weights.reshape(n_series, n_windows, horizon)
        self.is_adapted_ = True

    def _cross_fit(
        self, X: NDArray[np.floating], y: NDArray[np.integer], n_folds: int
    ) -> NDArray[np.floating]:
        """
        Out-of-fold odds for every row, via a stratified fold assignment.

        Stratification is done with a plain numpy shuffle per class, so no
        scikit-learn dependency is introduced.

        Parameters
        ----------
        X : NDArray, shape (n_rows, n_features)
            Stacked, standardized source and target features.
        y : NDArray, shape (n_rows,)
            Source/target labels, 0 and 1.
        n_folds : int
            Folds to use, at least 2 and no more than the rarer class count, so
            every training split sees both classes.

        Returns
        -------
        NDArray, shape (n_rows,)
            Out-of-fold odds.
        """
        rng = np.random.default_rng(self.seed)
        folds = np.empty(len(y), dtype=np.int64)
        for label in (0, 1):
            idx = np.flatnonzero(y == label)
            rng.shuffle(idx)
            folds[idx] = np.arange(len(idx)) % n_folds
        out = np.empty(len(y), dtype=np.float64)
        for k in range(n_folds):
            train = folds != k
            test = ~train
            self.estimator.fit(X[train], y[train])
            out[test] = self._odds(self.estimator.predict_proba(X[test]))
        return out

    def calibration_weights(self) -> NDArray[np.floating]:
        """
        Out-of-fold density-ratio weights for the calibration scores.

        Returns
        -------
        NDArray, shape (n_series, n_windows, horizon)
            Uniform ones when ``fit_target`` has not been called, so an
            unadapted DensityRatio recovers unweighted split conformal.
        """
        shape = self._require_source()
        if not self.is_adapted_:
            return np.ones(shape, dtype=np.float64)
        assert self.calibration_weights_ is not None  # set by fit_target when is_adapted_
        result: NDArray[np.floating] = self.calibration_weights_
        return result

    def test_weights(self, X: Exog | None) -> NDArray[np.floating]:
        """
        Density-ratio weights for the test points (the ``w(x_test)`` term).

        Parameters
        ----------
        X : Exog or None, shape (n_series, n_samples, horizon, n_features)

        Returns
        -------
        NDArray, shape (n_series, n_samples, horizon)
        """
        n_series, _, horizon = self._require_source()
        if not self.is_adapted_ or X is None:
            n_samples = 1 if X is None else int(np.asarray(X).shape[1])
            return np.ones((n_series, n_samples, horizon), dtype=np.float64)
        arr = np.asarray(X, dtype=np.float64)
        rows = self._flatten(arr)
        if not np.all(np.isfinite(rows)):
            raise ValueError(
                "test exogenous features contain non-finite values, so no density "
                "ratio can be evaluated for those points."
            )
        flat = self._standardize_apply(rows)
        w: NDArray[np.floating] = np.clip(
            self._odds(self.estimator.predict_proba(flat)), 0.0, self.clip_value_
        )
        return w.reshape(arr.shape[0], arr.shape[1], arr.shape[2])
