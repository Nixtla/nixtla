"""MLForecast adapter for conformalforecast.

Wraps a fitted ``mlforecast.MLForecast`` instance and exposes it through the
conformalforecast adapter contract.

Scope
-----
The adapter only exposes the operations conformal methods consume internally:
panel-shaped point forecasts, refit, and cross-validation. Quantile forecasts,
bootstrap ensembles, and other library-native uncertainty outputs are out of
scope — conformalforecast derives intervals from point forecasts. Methods that
need quantile or bootstrap inputs (e.g. CQR family, EnbPI) accept those
arrays directly from the user; generate them with MLForecast and pass them
to the method.

Limitations (v0.1)
------------------
- **No** ``SupportsQuantiles``, so CQR cannot be constructed on this adapter,
  unlike the StatsForecast and NeuralForecast ones. This is a known asymmetry,
  not an oversight: a quantile-capable MLForecast means one model per quantile
  (e.g. LightGBM with ``objective='quantile', alpha=q``), so the quantile grid
  is fixed when the models are fit rather than requested at predict time, and
  the adapter cannot honour an arbitrary ``quantiles`` array. Until that is
  designed properly, fit the per-quantile models yourself and pass the arrays
  to :class:`~conformalforecast.ConformalizedQuantileRegression` via
  ``calibrate``.
- Static features, sample weights, and natively conformal predictions
  (``prediction_intervals=...``) are rejected at construction time.
- Future exogenous ``X`` is accepted by ``predict``/``predict_batch``, and only
  when the adapter is constructed with ``exog_features=[...]``. ``train_df``
  must then carry those columns: ``MLForecast.predict`` rebuilds its feature
  matrix from the history frame it is handed, so ``predict`` joins the
  regressors' historical values back on. It follows that the history must lie
  within the training frame's dates -- a panel extending past it raises, since
  the regressors there are unknown -- and that ``refit`` raises, since a history
  panel holds target values only. ``cross_validate`` takes no ``X`` and needs
  none: it runs against the stored training frame and MLForecast splits the
  exogenous columns per window itself.
- Static features are still rejected at construction; only *future* exogenous
  columns are supported.
- Business-day frequencies (``'B'``, ``'BM'``, …) are not tested.
- Only ``refit(history)`` is provided; no incremental update.
- ``predict`` dates ``history`` by locating it in the training panel -- as a
  prefix, a recent window, or an extension of it -- so both an expanding and a
  sliding ``predict_batch`` loop date each window correctly. A history that is
  none of those raises rather than being silently mis-dated; to forecast from
  data the adapter has never seen, call ``refit``.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
from numpy.typing import NDArray
from pydantic import Field, PrivateAttr, model_validator
from pydantic.json_schema import SkipJsonSchema

from ..base import Forecast, ForecasterAdapter, Series
from ..capabilities import SupportsCrossValidation, SupportsRefit

try:
    from mlforecast import MLForecast
except ImportError as _err:
    raise ImportError(
        "MLForecastAdapter requires the 'mlforecast' package. "
        "Install it with: pip install conformalforecast[nixtla]"
    ) from _err

from . import _nixtla_common as _nx


class MLForecastAdapter(
    ForecasterAdapter,
    SupportsRefit,
    SupportsCrossValidation,
):
    """
    Adapter for a fitted :class:`mlforecast.MLForecast` instance.

    The adapter wraps **point** forecasters only. Any MLForecast configuration
    that produces intervals natively (``prediction_intervals``) is rejected at
    construction; conformalforecast produces its own intervals from point forecasts.

    ``n_series`` is derived from ``train_df`` rather than accepted; passing it
    is refused at construction.

    Parameters
    ----------
    mlf : MLForecast
        An already-fitted MLForecast instance with point models only.
    train_df : pandas.DataFrame or polars.DataFrame
        The long-format DataFrame used to fit ``mlf``. Kept in its own backend:
        a polars frame means every frame handed to MLForecast is polars.
    horizon : int
        Forecast horizon in time steps.
    freq : str or int
        The freq you passed to MLForecast: a pandas alias (``'D'``, ``'MS'``)
        for pandas frames, a polars interval (``'1d'``, ``'1mo'``) for polars
        frames, or an int for an integer time column.
    model_name : str
        Which model column to read from MLForecast output. Must be an alias
        present in ``mlf.models_``.
    id_col : str
        Column identifying individual series (default ``'unique_id'``).
    time_col : str
        Timestamp column (default ``'ds'``).
    target_col : str
        Target value column (default ``'y'``).

        All three must match the names ``mlf.fit`` recorded: ``MLForecast.predict``
        takes no column-name arguments and reuses them, so a mismatch is
        refused at construction.
    exog_features : list of str or None
        Names of the future exogenous columns ``mlf`` was fitted with. Required
        to pass ``X`` to :meth:`predict`, and ``train_df`` must carry them:
        ``MLForecast.predict`` rebuilds its feature matrix from the history
        frame, so the regressors' past values have to travel with it.
    Notes
    -----
    Not serialisable, deliberately: ``mlf`` is a live library object and
    ``train_df`` a DataFrame, so ``model_validate(model_dump())`` cannot
    rebuild one -- and it would be refused anyway, since a dump carries the
    derived ``n_series``. Those two fields are hidden from
    ``model_json_schema()`` along with the derived ones, so a schema-filled
    call still has to supply ``mlf`` and ``train_df`` programmatically; the
    schema describes the scalar arguments only.
    """

    mlf: SkipJsonSchema[MLForecast] = Field(description="An already-fitted MLForecast.")
    train_df: SkipJsonSchema[_nx.TrainFrame] = Field(
        description="The long-format pandas or polars DataFrame used to fit ``mlf``."
    )
    freq: str | int = Field(
        description=(
            "The freq you passed to MLForecast: a pandas alias ('D', 'MS') for "
            "pandas frames, a polars interval ('1d', '1mo') for polars frames, or an "
            "int for an integer time column."
        )
    )
    model_name: str = Field(description="Which fitted model alias to read from mlf's output.")
    id_col: str = Field(default="unique_id", description="Column identifying individual series.")
    time_col: str = Field(default="ds", description="Timestamp column.")
    target_col: str = Field(default="y", description="Target value column.")
    exog_features: list[str] | None = Field(
        default=None,
        description=(
            "Names of the future exogenous columns ``mlf`` was fitted with, in "
            "the order they appear on X's feature axis. Required to pass X to "
            "predict, and train_df must carry them."
        ),
    )

    # Derived in the before-validator and refused if passed, so it is hidden
    # from the schema a caller fills in rather than advertised as required.
    n_series: SkipJsonSchema[int] = Field(
        ge=1, description="Number of series in train_df, the leading array axis. Derived."
    )

    _series_ids: tuple[Any, ...] = PrivateAttr(default=())
    _series_ends: Any = PrivateAttr(default=None)  # nw.Series aligned with _series_ids
    _train_panel: Any = PrivateAttr(default=None)

    @property
    def _train(self) -> nw.DataFrame[Any]:
        """``train_df`` as a narwhals frame: a view, not a copy."""
        return nw.from_native(self.train_df, eager_only=True)

    @model_validator(mode="before")
    @classmethod
    def _derive_n_series(cls, data: Any) -> Any:
        """Fill the inherited ``n_series`` field from ``train_df``.

        See :meth:`StatsForecastAdapter._derive_n_series` -- same mechanism and
        same reason: ``n_series`` is a required field, so nothing later than a
        ``mode="before"`` validator can supply it, and passing it explicitly is
        refused rather than silently overwritten.
        """
        if not isinstance(data, dict):
            return data
        if "n_series" in data:
            raise ValueError("n_series is derived from train_df and must not be passed.")
        df = data.get("train_df")
        if "mlf" in data:
            cls._validate_fitted(data["mlf"])
        data["n_series"] = _nx.count_series(df, data.get("id_col", "unique_id"))
        return data

    @model_validator(mode="after")
    def _validate_and_build(self) -> MLForecastAdapter:
        """Run the construction-time validation chain and build the panel state."""
        self._validate_model_name(self.mlf, self.model_name)
        self._validate_no_prediction_intervals(self.mlf)
        self._validate_no_static_features(self.mlf)
        self._validate_no_weight_col(self.mlf)
        self._validate_column_names(self.mlf)
        train = self._train
        _nx.validate_columns(
            train, id_col=self.id_col, time_col=self.time_col, target_col=self.target_col
        )
        _nx.validate_exog_columns(train, exog_features=self.exog_features)
        _nx.validate_freq(train, time_col=self.time_col, freq=self.freq)
        _nx.validate_contiguity(train, id_col=self.id_col, time_col=self.time_col, freq=self.freq)
        _nx.validate_no_nan(train, id_col=self.id_col, target_col=self.target_col)
        self._series_ids, self._series_ends = _nx.build_series_bounds(
            train, self.horizon, id_col=self.id_col, time_col=self.time_col
        )
        self._validate_horizon_compatibility(self.mlf, self.horizon)

        # train_df is the single source of truth for the training frame -- refit
        # replaces it -- so copy it rather than aliasing the caller's frame.
        self.train_df = train.clone().to_native()
        self._train_panel = _nx.df_to_ragged_panel(
            self._train,
            series_ids=self._series_ids,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
        )
        return self

    def supports(self, capability: type) -> bool:
        """Capability check, with the exogenous case resolved at runtime.

        ``SupportsRefit`` depends on how the adapter was configured rather than
        on its class: a history panel carries target values only, so refitting
        an exogenous model would retrain it without its regressors. Gating here
        means the failure surfaces where the forecaster is bound, at the top of
        ``calibrate`` / ``calibrate_cv``, not mid-calibration.
        """
        if capability is SupportsRefit:
            return self.exog_features is None
        return isinstance(self, capability)

    # ------------------------------------------------------------------
    # Construction-time validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_fitted(mlf: Any) -> None:
        if not isinstance(mlf, MLForecast):
            raise ValueError(f"Expected an MLForecast instance, got {type(mlf).__name__}.")
        if not (hasattr(mlf, "models_") and bool(mlf.models_)):
            raise ValueError("MLForecast instance is not fitted. Call mlf.fit(df) first.")

    def _validate_model_name(self, mlf: MLForecast, model_name: str) -> None:
        if model_name not in mlf.models_:
            raise ValueError(
                f"model_name '{model_name}' not found among fitted models. "
                f"Available: {list(mlf.models_.keys())}"
            )

    def _validate_column_names(self, mlf: MLForecast) -> None:
        """The adapter's column names must be the ones ``mlf.fit`` recorded.

        ``MLForecast.predict`` takes no ``id_col``/``time_col``/``target_col``:
        it reuses what ``fit`` stored on ``mlf.ts``, so custom names only work
        when they were passed there too. See
        :func:`_nixtla_common.validate_fitted_column_names`.
        """
        ts = getattr(mlf, "ts", None)
        _nx.validate_fitted_column_names(
            {
                "id_col": getattr(ts, "id_col", None),
                "time_col": getattr(ts, "time_col", None),
                "target_col": getattr(ts, "target_col", None),
            },
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
            library_name="MLForecast",
            fit_call="mlf.fit",
        )

    def _validate_no_prediction_intervals(self, mlf: MLForecast) -> None:
        # MLForecast.fit assigns ``_cs_df = None`` and only writes a DataFrame to
        # it when prediction_intervals is passed. Use getattr so the check
        # survives older mlforecast versions where the attribute may be absent.
        if getattr(mlf, "_cs_df", None) is not None:
            raise ValueError(
                "MLForecast was fit with prediction_intervals. conformalforecast "
                "produces its own intervals from point forecasts — refit with "
                "mlf.fit(df) without the prediction_intervals argument."
            )

    def _validate_no_static_features(self, mlf: MLForecast) -> None:
        # Different mlforecast versions stash this either on ``mlf`` directly
        # or on the inner ``mlf.ts`` core. Check both.
        static = getattr(mlf, "static_features_", None)
        if not static:
            ts = getattr(mlf, "ts", None)
            static = getattr(ts, "static_features", None) if ts is not None else None
        if static:
            raise ValueError(
                "Static features are not supported in conformalforecast v0.1. "
                "Refit MLForecast without static_features."
            )

    def _validate_no_weight_col(self, mlf: MLForecast) -> None:
        weight = getattr(mlf, "weight_col", None)
        if weight is None:
            ts = getattr(mlf, "ts", None)
            weight = getattr(ts, "weight_col", None) if ts is not None else None
        if weight is not None:
            raise ValueError("Sample weights are not supported in conformalforecast v0.1.")

    def _validate_horizon_compatibility(self, mlf: MLForecast, horizon: int) -> None:
        max_horizon = getattr(mlf, "max_horizon", None)
        if max_horizon is None:
            ts = getattr(mlf, "ts", None)
            max_horizon = getattr(ts, "max_horizon", None) if ts is not None else None
        if max_horizon is not None and max_horizon < horizon:
            raise ValueError(
                f"MLForecast was fit with max_horizon={max_horizon}, cannot "
                f"forecast {horizon} steps. Refit with max_horizon >= {horizon} "
                "or with the default recursive strategy."
            )

        horizons_ = getattr(mlf, "horizons_", None)
        if horizons_ is None:
            ts = getattr(mlf, "ts", None)
            internal = getattr(ts, "_horizons", None) if ts is not None else None
            # Internal storage in mlforecast is 0-indexed; the public contract
            # documented in fit() is 1-indexed. Convert before comparing.
            if internal is not None:
                horizons_ = [int(h) + 1 for h in internal]
        if horizons_ is not None:
            required = set(range(1, horizon + 1))
            present = {int(h) for h in horizons_}
            missing = sorted(required - present)
            if missing:
                raise ValueError(
                    f"MLForecast was fit with horizons={sorted(present)}, missing "
                    f"required steps {missing} for horizon={horizon}. Refit with "
                    f"horizons covering 1..{horizon} or with the default recursive strategy."
                )

    # ------------------------------------------------------------------
    # Panel ↔ DataFrame conversion (thin shims over _nixtla_common)
    # ------------------------------------------------------------------

    def _df_to_panel(self, df: nw.DataFrame[Any], value_col: str) -> NDArray[np.floating]:
        return _nx.df_to_ragged_panel(
            df,
            series_ids=self._series_ids,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=value_col,
        )

    def _df_to_forecast_panel(self, df: nw.DataFrame[Any], value_col: str) -> NDArray[np.floating]:
        """Forecast frame to ``(n_series, horizon)``, indexed by step not calendar.

        See :func:`_nixtla_common.df_to_step_panel`: series forecast from
        different end dates carry different timestamps, so the union pivot of
        a union pivot would widen the result past ``horizon``.
        """
        return _nx.df_to_step_panel(
            df,
            value_col,
            series_ids=self._series_ids,
            id_col=self.id_col,
            time_col=self.time_col,
            n_steps=self.horizon,
        )

    def _anchor_shift(self, panel: NDArray[np.floating]) -> int:
        """Steps from the training end to ``panel``'s end, located by value.

        See :func:`_nixtla_common.anchor_shift_for_panel`.
        """
        return _nx.anchor_shift_for_panel(panel, train_panel=self._train_panel)

    def _panel_to_df(self, panel: NDArray[np.floating]) -> nw.DataFrame[Any]:
        """History panel to the long frame ``mlf.predict`` consumes as ``new_df``.

        With ``exog_features`` set, the regressors' historical values are joined
        back on from the training frame. MLForecast rebuilds lags and feature
        columns from ``new_df``, so a target-only history is rejected with
        "The feature names should match those that were passed during fit".
        """
        history_df = _nx.panel_to_df_ragged(
            panel,
            series_ids=self._series_ids,
            shift=self._anchor_shift(panel),
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
            freq=self.freq,
            like=self._train,
        )
        if self.exog_features is None:
            return history_df
        return _nx.attach_exog_history(
            history_df,
            train_df=self._train,
            exog_features=self.exog_features,
            id_col=self.id_col,
            time_col=self.time_col,
        )

    # ------------------------------------------------------------------
    # ForecasterAdapter contract
    # ------------------------------------------------------------------

    def predict(self, history: Series, *, X: Forecast | None = None) -> Forecast:
        """
        Produce a point forecast anchored where ``history`` sits in the training data.

        Parameters
        ----------
        history : Series, shape (n_series, T)
        X : Forecast or None, shape (n_series, 1, horizon, n_features)
            Future exogenous features, aligned to the ``horizon`` steps that
            follow ``history``. Requires the adapter to have been constructed
            with ``exog_features=[...]``.

        Returns
        -------
        Forecast, shape (n_series, 1, horizon)

        Raises
        ------
        ValueError
            If ``X`` is not None but the adapter was constructed without
            ``exog_features``.
        """
        history = self._validate_history(np.asarray(history, dtype=np.float64), allow_ragged=True)

        history_df = self._panel_to_df(history)
        X_df = None
        if X is not None:
            if self.exog_features is None:
                raise ValueError(
                    "predict received X but the adapter was constructed without "
                    "exog_features. Pass exog_features=[...] naming the model's "
                    "future exogenous columns."
                )
            X_df = _nx.exog_to_futr_df(
                np.asarray(X, dtype=np.float64),
                series_ids=self._series_ids,
                # Same anchor as the history frame, or the future features land
                # on dates the model is not forecasting.
                shift=self._anchor_shift(history),
                feature_names=self.exog_features,
                id_col=self.id_col,
                time_col=self.time_col,
                freq=self.freq,
                like=self._train,
            ).to_native()
        result_df = self.mlf.predict(h=self.horizon, new_df=history_df.to_native(), X_df=X_df)

        forecast_panel = self._df_to_forecast_panel(
            nw.from_native(result_df, eager_only=True), self.model_name
        )
        return forecast_panel[:, np.newaxis, :]

    # ------------------------------------------------------------------
    # SupportsRefit
    # ------------------------------------------------------------------

    def refit(self, history: Series) -> None:
        """
        Refit the underlying MLForecast model on a new history panel.

        The set of series must be identical to the original training data.
        If the new panel is longer than the previous one, the end timestamp
        shifts forward proportionally.

        Not available when the adapter was constructed with ``exog_features``:
        a ``Series`` panel carries target values only, so there is nowhere for
        the regressors' new values to come from.

        Parameters
        ----------
        history : Series, shape (n_series, T)

        Raises
        ------
        ValueError
            If ``exog_features`` was set, or if the series IDs change.
        """
        if self.exog_features is not None:
            raise ValueError(
                f"refit is not available on an adapter with exog_features "
                f"{self.exog_features}: a history panel carries target values "
                "only, so refitting would retrain the model without its "
                "regressors and silently change what cross_validate measures. "
                "Fit a new MLForecast on the extended frame and build a new "
                "adapter."
            )
        history = self._validate_history(np.asarray(history, dtype=np.float64), allow_ragged=True)

        # Extra panel width = new data appended at the end.
        # Not `_anchor_shift`: a refit panel is new data, so there is nothing to
        # locate it against -- its extra width *is* the shift.
        shift = int(history.shape[1]) - int(self._train_panel.shape[1])
        ends = _nx.grid_stamps(
            self._train,
            np.full((len(self._series_ids), 1), shift),
            series_ids=self._series_ids,
            id_col=self.id_col,
            time_col=self.time_col,
            freq=self.freq,
        )[self.time_col]
        history_df = _nx.panel_to_df_ragged(
            history,
            series_ids=self._series_ids,
            shift=shift,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
            freq=self.freq,
            like=self._train,
        )
        self.mlf.fit(
            history_df.to_native(),
            static_features=[],
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
        )
        self.train_df = history_df.to_native()
        self._train_panel = _nx.df_to_ragged_panel(
            history_df,
            series_ids=self._series_ids,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
        )
        self._series_ends = ends

    # ------------------------------------------------------------------
    # SupportsCrossValidation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        n_windows: int,
        step_size: int,
        refit: bool | int = False,
        *,
        aligned: bool = False,
    ) -> tuple[Forecast, Forecast]:
        """
        Run rolling-origin cross-validation.

        Parameters
        ----------
        n_windows : int
            Number of evaluation windows (>= 1).
        step_size : int
            Steps between successive windows (>= 1).
        refit : bool or int
            Whether (or how often) to refit between windows.
        aligned : bool, default False
            ``False`` returns dense per-series arrays (window axis = k steps
            before each series' own end). ``True`` returns calendar-aligned
            cutoffs with NaN cells for late-starting series.

        Returns
        -------
        predictions : Forecast, shape (n_series, n_windows, horizon)
        truths : Forecast, shape (n_series, n_windows, horizon)
        """
        if n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {n_windows}")
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")

        cv_df = nw.from_native(
            self.mlf.cross_validation(
                df=self.train_df,
                h=self.horizon,
                n_windows=n_windows,
                step_size=step_size,
                refit=refit,
                # The adapter rejects static features at construction, so every
                # non-target column is dynamic. Saying so is required once the frame
                # carries exogenous regressors: mlforecast otherwise infers them to
                # be static and raises when their values change over time.
                static_features=[],
                id_col=self.id_col,
                time_col=self.time_col,
                target_col=self.target_col,
            ),
            eager_only=True,
        )
        # MLForecast.cross_validation *deletes* the ``models_`` attribute on the
        # instance (verified against mlforecast 1.0.31 and 1.1.0) and instructs
        # callers to refit, so this is repair, not a gratuitous retrain: without
        # it every later ``predict`` raises AttributeError. It refits on the
        # stored training frame, which is the data the models were fit on to
        # begin with.
        #
        # Unconditional on purpose. Guarding on ``models_`` being absent would
        # test attribute presence, but what matters is *staleness*: if a future
        # mlforecast left ``models_`` set to the final fold's estimators, the
        # guard would skip the refit and every later ``predict`` would silently
        # return forecasts from a model fit on the training frame minus the last
        # ``n_windows * step_size`` observations. A redundant refit costs time;
        # that failure costs correctness with nothing raised.
        self.mlf.fit(
            self.train_df,
            static_features=[],
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
        )

        predictions, truths = _nx.reshape_cv_many(
            cv_df,
            [self.model_name, self.target_col],
            series_ids=self._series_ids,
            id_col=self.id_col,
            time_col=self.time_col,
            horizon=self.horizon,
            aligned=aligned,
        )
        return predictions, truths
