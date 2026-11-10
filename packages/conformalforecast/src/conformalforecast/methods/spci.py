"""Sequential Predictive Conformal Inference (Xu & Xie 2023)."""

from __future__ import annotations

import copy
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from .._quantiles import _ceil
from ..base import (
    CalibrationResult,
    Exog,
    Forecast,
    InsufficientCalibrationSamples,
    Interval,
    ScoreFunction,
    as_forecast,
)
from ..nonconformity.signed import SignedResidual
from ..quantile_regressors.qrf import QRFQuantileRegressor
from .base import ConformalForecaster, ConformalMethod


class SequentialPredictiveConformalInference(ConformalMethod["SPCIForecaster"]):
    """
    Sequential Predictive Conformal Inference on panel time series.

    Per ``(series, horizon)`` cell, a quantile regressor predicts the
    conditional distribution of the next signed residual from the previous
    ``window_size`` residuals (and, optionally, exogenous features aligned to
    the target step). The interval is
    ``[point + Q(beta_hat), point + Q(1 - alpha + beta_hat)]`` where ``beta_hat``
    minimizes the width ``Q(1 - alpha + beta) - Q(beta)`` over ``beta`` in
    ``[0, alpha]``.

    Determinism is controlled by ``seed``: the default regressor is
    :class:`QRFQuantileRegressor` with ``random_state=seed``. A user-supplied
    ``regressor`` is left exactly as passed and is responsible for its own
    determinism.

    .. warning::

       ``regressor`` is typed as the **concrete**
       :class:`~conformalforecast.quantile_regressors.qrf.QRFQuantileRegressor`,
       not the abstract
       :class:`~conformalforecast.quantile_regressors.base.QuantileRegressor`,
       so that the QRF knobs (``n_estimators``, ``min_samples_leaf``,
       ``random_state``) appear as nested config in ``model_json_schema()`` —
       an ABC-typed pydantic field produces an *empty* nested schema. The
       deliberate consequence is a narrowing: a ``QRFQuantileRegressor``
       subclass is accepted, but a ``QuantileRegressor`` subclass that does not
       inherit from ``QRFQuantileRegressor`` is rejected with
       ``pydantic.ValidationError``, and so is a duck-typed object implementing
       only ``fit`` + ``predict_quantile``. Both used to work. Subclass
       ``QRFQuantileRegressor`` to plug in a different backend, and override
       **all three** of ``fit``, ``predict_quantile`` and ``predict_quantiles``.
       Overriding only the first two leaves the inherited forest-backed
       ``predict_quantiles``, which checks for a fitted forest that a custom
       ``fit`` never built and fails inside :meth:`interval` with
       ``RuntimeError: QRFQuantileRegressor must be fit before
       predict_quantiles`` — a message that claims the regressor was not fitted
       when it was. If the base-class loop over ``predict_quantile`` is what
       you want, take it explicitly::

           class MyRegressor(QRFQuantileRegressor):
               predict_quantiles = QuantileRegressor.predict_quantiles

               def fit(self, X, y): ...
               def predict_quantile(self, X, quantile): ...

    See :class:`~conformalforecast.methods.base.ConformalMethod` for the
    calibrate/predict/update lifecycle and the ``calibrate`` / ``calibrate_cv``
    split. ``fit`` returns an :class:`SPCIForecaster`, whose ``update`` appends
    residuals and refits in place.

    Any adapter works — no special capabilities required. Bind one with
    ``calibrate(forecaster, histories, truths)`` or
    ``calibrate_cv(forecaster, n_windows)``.

    The score must be a :class:`SignedResidual` **or a subclass** (the default is
    a plain ``SignedResidual``); anything else is a ``pydantic.ValidationError``
    at construction. This is narrower than the array contract: a custom score
    that inverts a two-sided ``(n_series, horizon, 2)`` threshold must subclass
    ``SignedResidual`` to be accepted.

    Parameters
    ----------
    alpha : float
        Miscoverage in (0, 1). Target coverage is ``1 - alpha``.
    window_size : int, default 100
        Number of lagged residuals used as autoregressive features.
    regressor : QRFQuantileRegressor or None
        Prototype regressor, deep-copied once per ``(series, horizon)`` cell at
        fit time. Defaults to ``QRFQuantileRegressor(random_state=seed)``. See
        the warning above for the type narrowing this field carries.
    beta_grid_size : int, default 21
        Number of ``beta`` grid points over ``[0, alpha]``.
    refit_every : int, default 1
        Refit all regressors every this many :meth:`update` calls.
    seed : int, default 0
        Threaded into the default QRF regressor for determinism. Ignored when
        ``regressor`` is given.
    score : SignedResidual or None, default None
        Defaults to :class:`SignedResidual`. Must be a ``SignedResidual`` or a
        subclass: the regressors predict the *signed* residual quantiles the
        interval offsets the point forecast by, so a symmetric or quantile
        score would make the offsets meaningless.
    """

    REQUIRED_CAPABILITIES: ClassVar[tuple[type, ...]] = ()
    SCORE_TYPE: ClassVar[type[ScoreFunction]] = SignedResidual
    IS_ONLINE: ClassVar[bool] = True

    window_size: int = Field(
        default=100,
        ge=1,
        description=(
            "Number of past residuals used as the autoregressive feature row "
            "for the quantile regressor: each training row of cell (series, "
            "horizon) is window_size consecutive past signed residuals, the "
            "target is the residual that followed them, and the query row is "
            "the last window_size residuals seen. No value is special -- "
            "window_size=1 is still a genuine conditional model on the single "
            "previous residual, not a fallback to unweighted split conformal, "
            "which SPCI has no setting that recovers. A longer window lets the "
            "regressor condition on more history, so slowly-varying volatility "
            "is visible to it, but it costs training rows twice over: a cell "
            "with n effective residuals yields only n - window_size rows, and "
            "fit requires at least window_size + 1 residuals per series or "
            "raises InsufficientCalibrationSamples; each row is also wider, so "
            "the regressor has more features to split on and more room to "
            "overfit. A shorter window leaves more and narrower rows -- cheaper "
            "and better estimated -- but the regressor can only see the "
            "immediate past, so it reacts to the volatility regime it has just "
            "entered and to nothing longer. The knob exists to trade how much "
            "history the conditional quantile model may use against how many "
            "rows are left to estimate it from."
        ),
    )
    regressor: QRFQuantileRegressor | None = Field(
        default=None,
        description=(
            "Quantile regressor prototype. One copy.deepcopy of it is fitted "
            "per (series, horizon) cell, so a panel holds n_series * horizon "
            "independently trained regressors; the prototype itself is never "
            "fitted and stays plain config. Defaults to "
            "QRFQuantileRegressor(random_state=seed) -- the estimator of Xu & "
            "Xie (2023), and the only value that recovers the paper's method; "
            "its own knobs (n_estimators, min_samples_leaf, random_state) are "
            "nested config here. A supplied regressor is used exactly as "
            "passed, so `seed` is ignored and that regressor is responsible for "
            "its own determinism. This field is typed as the concrete "
            "QRFQuantileRegressor rather than the abstract QuantileRegressor "
            "precisely so those nested knobs stay visible in this schema: an "
            "ABC-typed field produces an empty nested schema. The deliberate "
            "consequence is a narrowing -- a QRFQuantileRegressor subclass is "
            "accepted, but a QuantileRegressor subclass that does not inherit "
            "from QRFQuantileRegressor is rejected with "
            "pydantic.ValidationError, and so is a duck-typed object "
            "implementing only fit + predict_quantile. Both used to work; "
            "subclass QRFQuantileRegressor to plug in a different backend, "
            "overriding fit, predict_quantile AND predict_quantiles -- leaving "
            "the inherited forest-backed predict_quantiles in place fails at "
            "interval() with a RuntimeError claiming the regressor was not "
            "fit, so take the generic loop explicitly with the class-body line "
            "predict_quantiles = QuantileRegressor.predict_quantiles. The "
            "knob exists because the conditional quantile model is the part of "
            "SPCI you are most likely to want to swap: a bigger forest, or a "
            "different quantile learner entirely."
        ),
    )
    beta_grid_size: int = Field(
        default=21,
        ge=2,
        description=(
            "Number of grid points searched over beta in [0, alpha]. The "
            "interval for a cell is [point + Q(beta), point + Q(1 - alpha + "
            "beta)] and beta_hat is the grid point minimizing the width Q(1 - "
            "alpha + beta) - Q(beta), ties going to the smallest beta. "
            "beta_grid_size=2 is the one special value: the grid collapses to "
            "{0, alpha}, a choice between the fully lower-tailed and the fully "
            "upper-tailed interval, which is the cheapest search that still "
            "allows asymmetry. Nothing recovers the symmetric alpha/2 interval "
            "outright, though alpha/2 lands on the grid for every odd size, the "
            "default 21 included. A finer grid finds a narrower interval on "
            "skewed residual distributions, and costs on two fronts: 2 * "
            "beta_grid_size quantile levels are queried per cell per interval() "
            "call (one batched predict_quantiles each), and minimizing width "
            "over more candidate levels of the same fitted regressor picks the "
            "level that looks narrowest by noise, which biases coverage "
            "slightly low. A coarser grid predicts faster and carries less of "
            "that selection bias, but leaves width on the table when the "
            "residuals really are skewed. The knob exists to trade interval "
            "sharpness on asymmetric residuals against prediction cost and "
            "that bias."
        ),
    )
    refit_every: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of update() calls between refits of the regressors. Every "
            "update() appends the newest signed residual to each cell's "
            "history, but only every refit_every-th call retrains. refit_every"
            "=1 (the default) refits after every step, so every interval comes "
            "from a regressor that has seen the whole history -- the freshest "
            "and most expensive setting. A large value approaches never "
            "refitting: the appended residuals still shift each cell's query "
            "row, so intervals keep moving, but the learned mapping from window "
            "to residual quantiles is frozen at the last refit and goes stale "
            "exactly when the residual process changes -- which is the regime "
            "shift SPCI was chosen for. Cost is why the knob exists: a refit "
            "trains n_series * horizon regressors from scratch (SPCI's "
            "sanctioned per-cell loop), so on a 100-series, 24-step panel one "
            "refit is 2400 fits, and refit_every=24 divides that bill by 24. "
            "Set it to the cadence at which you believe the residual dynamics "
            "actually change, not lower."
        ),
    )

    @model_validator(mode="after")
    def _default_regressor_from_seed(self) -> SequentialPredictiveConformalInference:
        """Fill in the seeded default regressor.

        Not a ``default_factory``: a plain one cannot read the sibling ``seed``
        field, and the data-aware form arrived in pydantic 2.11, above this
        package's ``>=2.7`` floor.
        Constructed here rather than at class definition so that
        ``import conformalforecast`` -- and ``model_json_schema()`` -- do not
        require the ``[spci]`` extra; ``QRFQuantileRegressor`` defers its
        backend import to ``fit``. Idempotent, so
        ``model_validate(model_dump())`` round-trips: a dumped instance already
        carries a regressor and is left alone.
        """
        if self.regressor is None:
            self.regressor = QRFQuantileRegressor(random_state=self.seed)
        return self

    def _default_score(self) -> ScoreFunction:
        return SignedResidual()

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        predictions: Forecast,
        truths: Forecast,
        *,
        X: Exog | None = None,
    ) -> SPCIForecaster:
        """
        Score residuals and fit one quantile regressor per cell.

        Parameters
        ----------
        predictions : Forecast, shape (n_series, n_windows, horizon)
            NaN cells mark windows that do not exist for a series.
        truths : Forecast, shape (n_series, n_windows, horizon)
        X : Exog or None, shape (n_series, n_windows, horizon, n_features)
            Optional exogenous features. When given, each training row is the
            ``window_size`` lagged residuals concatenated with the exog features
            aligned to the target step.

        Returns
        -------
        SPCIForecaster
            ``calibration_.diagnostics`` carries ``"window_size"``,
            ``"beta_grid_size"`` and ``"refit_every"``.

        Raises
        ------
        ValueError
            If array shapes are invalid or mismatched.
        InsufficientCalibrationSamples
            If any series has too few effective residuals (need at least
            ``max(window_size + 1, ceil(1 / alpha))``).
        """
        preds = as_forecast(predictions, "predictions")
        obs = as_forecast(truths, "truths")
        if preds.shape != obs.shape:
            raise ValueError(
                f"predictions and truths must have the same shape, got "
                f"{preds.shape} and {obs.shape}"
            )
        if X is not None:
            X_arr = np.asarray(X)
            if not np.issubdtype(X_arr.dtype, np.floating):
                X_arr = X_arr.astype(np.float64)
            if X_arr.ndim != 4 or X_arr.shape[:3] != preds.shape:
                raise ValueError(
                    "X must have shape (n_series, n_windows, horizon, n_features) "
                    f"matching predictions {preds.shape}, got {X_arr.shape}"
                )
        else:
            X_arr = None

        score_fn = self._bind_score()
        score_fn.fit(preds, obs)
        residuals = score_fn.score(preds, obs)

        n_eff = np.sum(~np.isnan(residuals), axis=1).min(axis=1)  # (n_series,)
        required = max(self.window_size + 1, int(_ceil(1.0 / self.alpha)))
        bad = np.flatnonzero(n_eff < required)
        if bad.size:
            idx = [int(i) for i in bad]
            raise InsufficientCalibrationSamples(
                f"Series {idx} have fewer than {required} effective residuals "
                f"(window_size={self.window_size}, alpha={self.alpha}).",
                series_indices=idx,
            )

        fitted = SPCIForecaster(
            method_=self._snapshot(),
            score_fn_=score_fn,
            n_calibration_samples_=n_eff,
            calibration_=CalibrationResult(
                n_calibration_samples=n_eff.copy(),
                diagnostics={
                    "window_size": self.window_size,
                    "beta_grid_size": self.beta_grid_size,
                    "refit_every": self.refit_every,
                },
            ),
            quantile_regressors_={},
            residuals_=residuals,
            X_=X_arr,
            n_observations_=residuals.shape[1],
            steps_since_refit_=0,
        )
        fitted._fit_all_regressors()
        return fitted


class SPCIForecaster(ConformalForecaster):
    """
    A fitted :class:`SequentialPredictiveConformalInference`. ``update`` mutates it in place.

    Attributes
    ----------
    method_ : SequentialPredictiveConformalInference
    quantile_regressors_ : dict
        One fitted clone of the method's ``regressor`` per ``(series, horizon)``
        cell, keyed by that pair. Each refit replaces the whole dict.
    residuals_ : NDArray, shape (n_series, n_windows, horizon)
        Signed residuals retained for the online ``update`` step and for
        building each cell's query row.
    X_ : NDArray or None, shape (n_series, n_windows, horizon, n_features)
        Exogenous buffer, ``None`` when fit without exog.
    n_observations_ : int
        Number of windows seen so far (grows by one per :meth:`update`).
    steps_since_refit_ : int
        :meth:`update` calls since the regressors were last refit; they refit
        when this reaches the method's ``refit_every``.
    """

    method_: SequentialPredictiveConformalInference
    quantile_regressors_: dict[tuple[int, int], QRFQuantileRegressor]
    residuals_: NDArray[np.floating]
    #: Shape (n_series, n_windows, horizon, n_features); None when fit without exog.
    X_: NDArray[np.floating] | None
    n_observations_: int
    steps_since_refit_: int

    def _valid_cell(
        self, s: int, h: int
    ) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
        """Return the non-NaN residuals (and aligned exog) for cell ``(s, h)``."""
        res = self.residuals_[s, :, h]
        mask = ~np.isnan(res)
        res_v = res[mask]
        exog_v = None if self.X_ is None else self.X_[s, mask, h, :]
        return res_v, exog_v

    def _train_arrays(
        self,
        res_v: NDArray[np.floating],
        exog_v: NDArray[np.floating] | None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """AR feature matrix (optionally exog-augmented) and aligned target."""
        w = self.method_.window_size
        X_ar = np.lib.stride_tricks.sliding_window_view(res_v[:-1], window_shape=w)
        y = res_v[w:]
        X_train = np.concatenate([X_ar, exog_v[w:]], axis=1) if exog_v is not None else X_ar
        return X_train, y

    def _query_row(
        self,
        res_v: NDArray[np.floating],
        exog_v: NDArray[np.floating] | None,
    ) -> NDArray[np.floating]:
        """Query feature row: last ``window_size`` residuals (+ most recent exog)."""
        w = self.method_.window_size
        q = res_v[-w:].reshape(1, w)
        if exog_v is not None:
            q = np.concatenate([q, exog_v[-1:].reshape(1, -1)], axis=1)
        return q

    def _fit_all_regressors(self) -> None:
        """Fit one :class:`QRFQuantileRegressor` per ``(series, horizon)`` cell.

        Each cell gets a distinct ``copy.deepcopy`` of the ``regressor``
        prototype, so training one cell never overwrites another's and the
        prototype itself stays unfitted config.
        """
        regressor = self.method_.regressor
        assert regressor is not None  # filled in by _default_regressor_from_seed
        n_series, _, horizon = self.residuals_.shape
        regressors: dict[tuple[int, int], QRFQuantileRegressor] = {}
        for s in range(n_series):
            for h in range(horizon):
                res_v, exog_v = self._valid_cell(s, h)
                X_train, y_train = self._train_arrays(res_v, exog_v)
                reg = copy.deepcopy(regressor)
                reg.fit(X_train, y_train)
                regressors[(s, h)] = reg
        self.quantile_regressors_ = regressors

    # ------------------------------------------------------------------
    # Interval / update
    # ------------------------------------------------------------------

    def _beta_offsets(
        self,
        regressor: QRFQuantileRegressor,
        X_query: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Width-minimizing ``(lower_offset, upper_offset)`` for each query row.

        Every level on the beta grid is requested in **one**
        :meth:`QuantileRegressor.predict_quantiles` call, which every regressor
        has: it is a concrete method on the ABC, and ``regressor`` is now typed
        as a :class:`QRFQuantileRegressor`, so the pydantic field rejects
        anything that is not a subclass. Its docstring records why batching
        matters. Ties resolve to the lowest beta, as the sequential ``<``
        comparison did.

        Rows are independent, so many are answered in the same call -- which is
        what makes :meth:`calibration_intervals` one call per cell rather than
        one per window.

        Parameters
        ----------
        regressor : QRFQuantileRegressor
        X_query : NDArray, shape (n_rows, n_features)

        Returns
        -------
        NDArray, shape (n_rows, 2)
            Last axis is ``(lower_offset, upper_offset)``.
        """
        beta_grid = np.linspace(0.0, self.method_.alpha, self.method_.beta_grid_size)
        eps = 1e-6
        lo_levels = np.maximum(beta_grid, eps)
        hi_levels = np.minimum(1.0 - self.method_.alpha + beta_grid, 1.0 - eps)

        levels = np.concatenate([lo_levels, hi_levels])
        preds = regressor.predict_quantiles(X_query, levels)
        n_beta = beta_grid.shape[0]
        arr = np.asarray(preds, dtype=np.float64).reshape(X_query.shape[0], 2 * n_beta)
        lowers = arr[:, :n_beta]
        uppers = arr[:, n_beta:]

        best = np.argmin(uppers - lowers, axis=1)[:, np.newaxis]
        return np.concatenate(
            [np.take_along_axis(lowers, best, axis=1), np.take_along_axis(uppers, best, axis=1)],
            axis=1,
        )

    def _score_thresholds(self, dtype: np.dtype) -> NDArray[np.floating]:
        """Per-cell ``(lower_offset, upper_offset)``, shape ``(n_series, horizon, 2)``.

        Offsets are computed in float64 (the regressor's precision) and stored
        in ``dtype`` so the resulting ``Interval`` matches the prediction dtype.
        """
        n_series, _, horizon = self.residuals_.shape
        thresholds = np.empty((n_series, horizon, 2), dtype=dtype)
        for s in range(n_series):
            for h in range(horizon):
                res_v, exog_v = self._valid_cell(s, h)
                X_query = self._query_row(res_v, exog_v)
                offsets = self._beta_offsets(self.quantile_regressors_[(s, h)], X_query)
                thresholds[s, h, :] = offsets[0]
        return thresholds

    def interval(
        self,
        predictions: Forecast,
        *,
        X: Exog | None = None,
    ) -> Interval:
        """
        Produce asymmetric intervals for new predictions.

        Parameters
        ----------
        predictions : Forecast, shape (n_series, n_samples, horizon)
        X : Exog or None
            Ignored at interval time; the query reuses the most recent
            calibration exog row (adapter-side future-exog threading is not
            implemented).

        Returns
        -------
        Interval, shape (n_series, n_samples, horizon, 2)

        Raises
        ------
        ValueError
            If predictions shape (n_series, horizon) does not match fitted state.
        """
        preds = as_forecast(predictions, "predictions")
        n_series, _, horizon = self.residuals_.shape
        if (preds.shape[0], preds.shape[2]) != (n_series, horizon):
            raise ValueError(
                f"predictions shape (n_series, horizon) = "
                f"{(preds.shape[0], preds.shape[2])} does not match fitted "
                f"state {(n_series, horizon)}"
            )
        result: Interval = self.score_fn_.invert(preds, self._score_thresholds(preds.dtype))
        return result

    def calibration_intervals(self, predictions: Forecast, *, X: Exog | None = None) -> Interval:
        """
        Per-window in-sample intervals, each queried with its own residual history.

        SPCI's offsets are conditional on the ``window_size`` residuals
        immediately preceding the step being predicted, so -- unlike every other
        method here -- :meth:`interval` cannot serve as its own in-sample
        replay. Broadcasting the final query's answer over earlier windows
        places each interval against a residual history it was not conditioned
        on; the resulting coverage is an artifact, not a measurement.

        This slides the query instead: window ``t`` is answered from the
        residuals before ``t``. Those query rows are exactly the training design
        matrix :meth:`_train_arrays` already builds, so each cell costs one
        batched regressor call rather than one per window.

        The result is still **in sample with respect to the quantile
        regressors** -- they were fit on these very rows, and on later residuals
        after any :meth:`update` -- so coverage here is optimistic. It is no
        longer anachronistic.

        Parameters
        ----------
        predictions : Forecast, shape (n_series, n_windows, horizon)
            The point forecasts :meth:`fit` was calibrated on.
        X : Exog or None
            Ignored; each window's exogenous row comes from the fitted buffer.

        Returns
        -------
        Interval, shape (n_series, n_windows, horizon, 2)
            ``(NaN, NaN)`` for the first ``window_size`` **valid** windows of
            each cell, which have too little history to query, and for windows
            that do not exist for that series.

        Raises
        ------
        ValueError
            If ``predictions`` does not match the fitted ``(n_series, n_windows,
            horizon)``.
        """
        preds = as_forecast(predictions, "predictions")
        n_series, n_buffered, horizon = self.residuals_.shape
        n_windows = preds.shape[1]
        # The buffer grows past the calibration windows on every update(), so
        # replay the leading n_windows of it rather than requiring an exact
        # match -- the user may well have deployed the method before asking for
        # diagnostics.
        if (preds.shape[0], preds.shape[2]) != (n_series, horizon) or n_windows > n_buffered:
            raise ValueError(
                f"predictions shape {preds.shape} does not fit the calibration "
                f"residuals {self.residuals_.shape}. calibration_intervals replays "
                "the windows fit() was given, so it needs those same predictions."
            )

        w = self.method_.window_size
        offsets = np.full((*preds.shape, 2), np.nan, dtype=preds.dtype)
        for s in range(n_series):
            for h in range(horizon):
                # Compaction drops this cell's missing windows, so `idx` maps
                # each query row back to its absolute window. Cells with
                # different valid counts therefore get different warm-up
                # boundaries with no extra bookkeeping -- ragged panels work by
                # construction.
                res_cell = self.residuals_[s, :n_windows, h]
                mask = ~np.isnan(res_cell)
                idx = np.flatnonzero(mask)
                if idx.size <= w:
                    continue  # no window has a full residual history behind it
                res_v = res_cell[mask]
                exog_v = None if self.X_ is None else self.X_[s, :n_windows][mask, h, :]
                X_rows, _ = self._train_arrays(res_v, exog_v)
                rows = self._beta_offsets(self.quantile_regressors_[(s, h)], X_rows)
                offsets[s, idx[w:], h, :] = rows

        result: Interval = self.score_fn_.invert(preds, offsets)
        return result

    def _update(
        self,
        prediction: Forecast,
        truth: Forecast,
        *,
        X: Exog | None = None,
    ) -> None:
        """
        Append a new residual and refit every ``refit_every`` calls.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, 1, horizon)
        truth : Forecast, shape (n_series, 1, horizon)
        X : Exog or None, shape (n_series, 1, horizon, n_features)
            Appended to the exog buffer when the method was fit with exog.

        Raises
        ------
        ValueError
            If ``prediction``/``truth`` do not have shape ``(n_series, 1, horizon)``.
        """
        n_series, _, horizon = self.residuals_.shape
        p = as_forecast(prediction, "prediction")
        t = as_forecast(truth, "truth")
        expected = (n_series, 1, horizon)
        if p.shape != expected or t.shape != expected:
            raise ValueError(
                f"prediction and truth must have shape {expected}, got {p.shape} and {t.shape}"
            )
        new_residual = self.score_fn_.score(p, t)
        self.residuals_ = np.concatenate([self.residuals_, new_residual], axis=1)
        if self.X_ is not None:
            if X is None:
                raise ValueError("This method was fit with exogenous X; update() requires X.")
            X_step = np.asarray(X)
            if not np.issubdtype(X_step.dtype, np.floating):
                X_step = X_step.astype(np.float64)
            self.X_ = np.concatenate([self.X_, X_step], axis=1)
        self.n_observations_ += 1
        self.steps_since_refit_ += 1
        if self.steps_since_refit_ >= self.method_.refit_every:
            self._fit_all_regressors()
            self.steps_since_refit_ = 0
