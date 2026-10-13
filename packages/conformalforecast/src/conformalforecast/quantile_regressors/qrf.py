"""Quantile Random Forest implementation of :class:`QuantileRegressor`."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, PrivateAttr

from .base import QuantileRegressor


def _load_backend() -> Any:
    """Import ``quantile_forest``'s estimator, or raise with the install hint.

    Deferred to call time on purpose: ``quantile-forest`` pulls in scikit-learn,
    and both live in the optional ``[spci]`` extra. Importing them at module
    scope would make ``import conformalforecast`` drag in the extra.
    """
    try:
        from quantile_forest import (  # type: ignore[import-untyped]
            RandomForestQuantileRegressor,
        )
    except ImportError as err:
        raise ImportError(
            "QRFQuantileRegressor requires the 'quantile-forest' package. "
            "Install it with: pip install conformalforecast[spci]"
        ) from err
    return RandomForestQuantileRegressor


class QRFQuantileRegressor(QuantileRegressor):
    """
    Quantile Random Forest regressor (Meinshausen 2006).

    Default quantile estimator for SPCI per Xu & Xie (2023). Wraps the
    ``quantile-forest`` package, exposing a clean two-method interface.

    Notes
    -----
    The ``quantile-forest`` package is an optional dependency. The import is
    deferred to :meth:`fit` (not construction) so that a
    :class:`QRFQuantileRegressor` -- including one built as
    :class:`~conformalforecast.methods.spci.SequentialPredictiveConformalInference`'s
    default ``regressor`` -- can be constructed, and its schema generated via
    ``model_json_schema()``, without ``quantile-forest`` installed. Calling
    :meth:`fit` without it installed raises :class:`ImportError`.
    """

    # extra="allow" is load-bearing here, unlike the temporary bridges the
    # pydantic migration used while the hand-written __init__s were being
    # removed -- those are all gone. This one stays: **kwargs forwards to
    # RandomForestQuantileRegressor at fit time, and model_extra is what
    # preserves that passthrough as a real (if unvalidated) field bag.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    n_estimators: int = Field(
        default=100,
        ge=1,
        description=(
            "Number of trees in the forest. Each query row lands in one leaf per "
            "tree; predict_quantile pools the training residuals from all "
            "n_estimators leaves into a single empirical distribution and reads "
            "the quantile off that pool (Meinshausen 2006), rather than "
            "predicting a scalar per tree and averaging. No particular value is "
            "special -- there is no n_estimators that recovers a qualitatively "
            "different estimator, only a larger or smaller pooled sample. A "
            "larger value pools more leaves, so the empirical distribution is "
            "denser and the estimated quantiles are smoother and more stable "
            "across refits, at the cost of proportionally more time and memory "
            "to fit and to predict from (every tree's leaf assignments are kept "
            "for prediction, not discarded after fitting). A smaller value fits "
            "and predicts faster but pools fewer residuals per query, so the "
            "quantile estimate is noisier and more sensitive to which rows "
            "happened to land in which leaf. The knob exists to trade that "
            "compute and memory cost against how stable the quantile estimate is."
        ),
    )
    min_samples_leaf: int = Field(
        default=5,
        ge=1,
        description=(
            "Minimum number of training rows allowed in a leaf. Unlike a "
            "regression tree, a leaf here is not collapsed to one predicted "
            "value -- predict_quantile reads the quantile off the empirical "
            "distribution of the training residuals still sitting in the query "
            "row's leaf, so this knob controls how many residuals that "
            "distribution is built from. No leaf size is special; the trade "
            "runs in one direction as it increases. A larger min_samples_leaf "
            "forces bigger leaves, so each leaf's empirical distribution pools "
            "residuals from a wider region of the feature space: the estimated "
            "quantiles are smoother and less sensitive to noise, but also less "
            "locally adapted, since rows with genuinely different conditional "
            "distributions get pooled together. A smaller value keeps leaves "
            "narrow and locally adapted, but each one holds few residuals, so "
            "the quantile read off it is jumpy and can shift a lot between "
            "nearby query points. The knob exists to trade how locally-adapted "
            "the estimate is against how stable it is."
        ),
    )
    random_state: int = Field(
        default=0,
        ge=0,
        description=(
            "Seed for the forest's two sources of randomness: the bootstrap "
            "resample drawn for each tree and the feature subsample considered "
            "at each split. No value is special and there is no direction -- "
            "every seed produces one equally valid random forest fit, not a "
            "better or worse one, so this is not a knob to tune. Defaults to 0 "
            "rather than None so a directly-constructed regressor is "
            "deterministic; SequentialPredictiveConformalInference threads its "
            "own seed through here for the same reason. Fixing it lets the same "
            "config on the same data reproduce the same fit; varying it across "
            "otherwise-identical runs is how you'd measure how much the fit "
            "itself varies under the forest's own randomness."
        ),
    )

    _forest: Any = PrivateAttr(default=None)

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> None:
        """
        Fit the forest on autoregressive residual windows.

        Parameters
        ----------
        X : NDArray, shape (n_samples, n_features)
            Training features. Each row is a window of past residuals,
            optionally concatenated with exogenous features.
        y : NDArray, shape (n_samples,)
            The next residual after each window.

        Raises
        ------
        ImportError
            If ``quantile-forest`` is not installed. Deferred to here (rather
            than construction) so this class stays constructible -- and its
            schema generable -- without the ``[spci]`` extra.
        """
        backend = _load_backend()
        self._forest = backend(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            **(self.model_extra or {}),
        )
        self._forest.fit(X, y)

    def predict_quantile(
        self,
        X: NDArray[np.floating],
        q: float,
    ) -> NDArray[np.floating]:
        """
        Predict the ``q``-quantile of the conditional residual distribution.

        Parameters
        ----------
        X : NDArray, shape (n_samples, n_features)
        q : float in (0, 1)

        Returns
        -------
        NDArray, shape (n_samples,)

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._forest is None:
            raise RuntimeError("QRFQuantileRegressor must be fit before predict_quantile.")
        pred = self._forest.predict(X, quantiles=[q])
        return np.asarray(pred, dtype=np.float64).reshape(-1)

    def predict_quantiles(
        self,
        X: NDArray[np.floating],
        qs: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Predict several quantiles in a single traversal of the forest.

        ``quantile-forest`` accepts a list of levels and answers them from one
        pass over the leaf distributions, which is why this override exists:
        SPCI's beta search asks for ~42 levels per cell, and one call per level
        measured 42x slower on a fitted forest.

        Parameters
        ----------
        X : NDArray, shape (n_samples, n_features)
        qs : NDArray, shape (n_quantiles,)

        Returns
        -------
        NDArray, shape (n_samples, n_quantiles)

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._forest is None:
            raise RuntimeError("QRFQuantileRegressor must be fit before predict_quantiles.")
        levels = np.asarray(qs, dtype=np.float64).reshape(-1)
        pred = np.asarray(self._forest.predict(X, quantiles=list(levels)), dtype=np.float64)
        return pred.reshape(np.asarray(X).shape[0], levels.shape[0])
