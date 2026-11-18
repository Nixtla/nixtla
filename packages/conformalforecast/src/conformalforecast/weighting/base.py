"""Weight functions for covariate-shift-weighted conformal prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, field_validator
from pydantic.json_schema import SkipJsonSchema

from .._model import ArrayModel
from ..base import Exog


@runtime_checkable
class DensityRatioEstimator(Protocol):
    """
    Binary probabilistic classifier used to estimate a density ratio.

    Structural (not nominal) on purpose: any object exposing ``fit`` and
    ``predict_proba`` satisfies it, so scikit-learn, XGBoost, LightGBM and
    CatBoost classifiers all plug in with no wrapper and no subclassing.
    """

    def fit(self, X: NDArray[np.floating], y: NDArray[np.integer]) -> Any:
        """Fit on features ``X`` (n, n_features) and binary labels ``y`` (n,)."""

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return class probabilities, shape ``(n, 2)``, column 1 = P(y=1|x)."""


class WeightFunction(ArrayModel, ABC):
    """
    Supplies per-cell weights for the weighted conformal quantile.

    Calibration and test weights are separate methods because they are computed
    differently: calibration weights must be out-of-fold (a point's weight may
    not come from a model trained on it), while test weights come from a model
    fitted on all available data.

    Lifecycle:
        1. ``fit_source(X, shape, valid=...)`` — the method's ``fit`` supplies
           source exogenous features, the calibration array shape, and which
           cells of it are real windows rather than ragged padding.
        2. ``fit_target(X)`` — the method's ``adapt`` supplies target features.
           Optional; without it the function stays unadapted.
        3. ``calibration_weights()`` / ``test_weights(X)``.
    """

    # extra="forbid" for the same reason ConformalMethod sets it: a typo'd
    # knob must fail at construction rather than be silently ignored.
    model_config = ConfigDict(extra="forbid")

    #: Whether ``fit_target`` actually uses the target features it is handed.
    #: Declared rather than inferred from ``is_adapted_``, which is already
    #: True for weights fixed in advance: a subclass that ignores target
    #: features must make ``adapt(target_X=...)`` an error, not a no-op.
    CONSUMES_TARGET_X: ClassVar[bool] = False

    source_X_: SkipJsonSchema[Exog | None] = None
    shape_: SkipJsonSchema[tuple[int, int, int] | None] = None
    valid_: SkipJsonSchema[NDArray[np.bool_] | None] = None
    is_adapted_: SkipJsonSchema[bool] = False

    def fit_source(
        self,
        X: Exog | None,
        shape: tuple[int, int, int],
        *,
        valid: NDArray[np.bool_] | None = None,
    ) -> None:
        """
        Record the source features, the calibration shape and the valid mask.

        Parameters
        ----------
        X : Exog or None, shape (n_series, n_windows, horizon, n_features)
        shape : tuple of int
            The ``(n_series, n_windows, horizon)`` shape of the calibration
            scores the weights must align with.
        valid : NDArray of bool or None, shape (n_series, n_windows, horizon)
            False marks a cell whose calibration window does not exist for that
            series — a ragged panel's left-NaN padding. Features in those cells
            are NaN by construction, so a weight function that fits a model on
            them must leave them out. None means every cell is a real window.

        Raises
        ------
        ValueError
            If ``valid`` is given and its shape is not ``shape``.
        """
        if valid is not None:
            mask = np.asarray(valid, dtype=bool)
            if mask.shape != shape:
                raise ValueError(
                    f"valid mask shape {mask.shape} does not match the calibration "
                    f"array shape {shape}."
                )
            valid = mask
        self.source_X_ = X
        self.shape_ = shape
        self.valid_ = valid

    def fit_target(self, X: Exog | None) -> None:
        """
        Fit against target-domain features. Default: no-op (stays unadapted).

        Parameters
        ----------
        X : Exog or None, shape (n_series, n_samples, horizon, n_features)
        """
        return None

    def _require_source(self) -> tuple[int, int, int]:
        if self.shape_ is None:
            raise RuntimeError(
                f"{type(self).__name__}.fit_source() must be called before weights are requested."
            )
        return self.shape_

    def _valid_rows(self) -> NDArray[np.bool_]:
        """
        Flat per-row validity for the source feature block.

        Returns
        -------
        NDArray of bool, shape (n_series * n_windows * horizon,)
            Row-major, matching how the 4-D source features flatten.
        """
        shape = self._require_source()
        if self.valid_ is None:
            return np.ones(int(np.prod(shape)), dtype=bool)
        flat: NDArray[np.bool_] = self.valid_.reshape(-1)
        return flat

    @abstractmethod
    def calibration_weights(self) -> NDArray[np.floating]:
        """
        Weights for the calibration scores.

        Returns
        -------
        NDArray, shape (n_series, n_windows, horizon)
        """

    @abstractmethod
    def test_weights(self, X: Exog | None) -> NDArray[np.floating]:
        """
        Weights for the test points (the ``w(x_test)`` term).

        Parameters
        ----------
        X : Exog or None, shape (n_series, n_samples, horizon, n_features)
            When None, one sample is assumed.

        Returns
        -------
        NDArray, shape (n_series, n_samples, horizon)
        """


class UniformWeights(WeightFunction):
    """
    Constant unit weights. Recovers unweighted split conformal exactly.

    This is the default when no weighting is supplied, so an un-weighted
    ``WeightedSplit`` behaves identically to ``SplitConformal``.
    """

    def calibration_weights(self) -> NDArray[np.floating]:
        return np.ones(self._require_source(), dtype=np.float64)

    def test_weights(self, X: Exog | None) -> NDArray[np.floating]:
        n_series, _, horizon = self._require_source()
        n_samples = 1 if X is None else int(np.asarray(X).shape[1])
        return np.ones((n_series, n_samples, horizon), dtype=np.float64)


class PrecomputedWeights(WeightFunction):
    """
    User-supplied calibration weights.

    The escape hatch for weights computed outside this package. Test weights
    are uniform: with weights fixed in advance there is no model from which to
    evaluate ``w(x_test)``.

    Parameters
    ----------
    weights : NDArray, shape (n_series, n_windows, horizon)
        Non-negative, finite weights aligned with the calibration scores.

    Raises
    ------
    ValueError
        If ``weights`` is not 3-D or contains a negative or non-finite entry.
    """

    weights: NDArray[np.floating]

    @field_validator("weights")
    @classmethod
    def _check_weights(cls, value: NDArray[np.floating]) -> NDArray[np.floating]:
        """Non-negative, finite and 3-D, coerced to float64."""
        w = np.asarray(value, dtype=np.float64)
        if w.ndim != 3:
            raise ValueError(
                f"weights must be 3-D (n_series, n_windows, horizon), got shape {w.shape}"
            )
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must be non-negative and finite.")
        return w

    def model_post_init(self, context: Any) -> None:
        # Weights fixed in advance are adapted by definition; there is no
        # target-fitting step for them.
        self.is_adapted_ = True

    def fit_source(
        self,
        X: Exog | None,
        shape: tuple[int, int, int],
        *,
        valid: NDArray[np.bool_] | None = None,
    ) -> None:
        if self.weights.shape != shape:
            raise ValueError(
                f"PrecomputedWeights shape {self.weights.shape} does not match the "
                f"calibration array shape {shape}."
            )
        super().fit_source(X, shape, valid=valid)

    def calibration_weights(self) -> NDArray[np.floating]:
        self._require_source()
        return self.weights

    def test_weights(self, X: Exog | None) -> NDArray[np.floating]:
        n_series, _, horizon = self._require_source()
        n_samples = 1 if X is None else int(np.asarray(X).shape[1])
        return np.ones((n_series, n_samples, horizon), dtype=np.float64)
