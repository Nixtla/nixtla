"""StatsForecast adapter for conformalforecast.

Wraps a fitted ``statsforecast.StatsForecast`` instance and exposes it through
the conformalforecast adapter contract.  This is the reference adapter for the
Nixtla family of forecasting libraries.

Scope
-----
The adapter exposes panel-shaped point forecasts, quantile forecasts, refit,
and cross-validation — everything the current conformal methods consume from a
StatsForecast model. Quantile support drives StatsForecast's level-based
interval API (``<model>-lo-<level>`` / ``<model>-hi-<level>``). Bootstrap
ensembles remain out of scope for v0.1.

Limitations (v0.1)
------------------
- Future exogenous ``X`` is accepted by ``predict``/``predict_batch``, and only
  when the adapter is constructed with ``exog_features=[...]``. ``train_df``
  must then carry those columns: ``sf.forecast`` re-derives the regressor count
  from the frame it is handed, so ``predict`` joins their historical values back
  on before forecasting. It follows that the history must lie within the
  training frame's dates -- a panel extending past it raises, since the
  regressors there are unknown.
  ``cross_validate`` and ``cross_validate_quantiles`` take no ``X`` argument and
  do not need one: they run against the stored training frame, and
  StatsForecast splits its exogenous columns per window itself. Because that
  path reads *every* non-key column of ``train_df`` as a regressor while
  ``predict`` reads only the declared ones, the two sets must agree exactly:
  a column ``exog_features`` does not name is rejected at construction rather
  than left to calibrate a model that ``predict`` will not reproduce.
  ``predict_quantiles`` has no ``X`` in its contract, so it *raises* for an
  exogenous adapter rather than forecast without the regressors. ``refit``
  likewise raises: a history panel holds target values only.
- Static and historical (past-only) covariates are not supported.
- Business-day frequencies (``'B'``, ``'BM'``, …) are not tested.
- Only ``refit(history)`` is provided; no ``update_horizon`` or ``add_series``.
- ``predict`` dates ``history`` by locating it in the training panel -- as a
  prefix, a recent window, or an extension of it -- so both an expanding and a
  sliding ``predict_batch`` loop date each window correctly. A history that is
  none of those raises rather than being silently mis-dated; to forecast from
  data the adapter has never seen, call ``refit``.
- Quantile prediction requires symmetric quantile pairs; ``q == 0.5`` is
  rejected because the StatsForecast level API has no median column.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
from numpy.typing import NDArray
from pydantic import Field, PrivateAttr, model_validator
from pydantic.json_schema import SkipJsonSchema

from ..base import Forecast, ForecasterAdapter, Series, UnsupportedCapability
from ..capabilities import (
    SupportsCrossValidation,
    SupportsCrossValidationQuantiles,
    SupportsQuantiles,
    SupportsRefit,
)

try:
    from statsforecast import StatsForecast
except ImportError as _err:
    raise ImportError(
        "StatsForecastAdapter requires the 'statsforecast' package. "
        "Install it with: pip install conformalforecast[nixtla]"
    ) from _err

from . import _nixtla_common as _nx


class StatsForecastAdapter(
    ForecasterAdapter,
    SupportsRefit,
    SupportsCrossValidation,
    SupportsQuantiles,
    SupportsCrossValidationQuantiles,
):
    """
    Adapter for a fitted :class:`statsforecast.StatsForecast` instance.

    ``n_series`` is derived from ``train_df`` rather than accepted; passing it
    is refused at construction.

    Parameters
    ----------
    sf : StatsForecast
        An already-fitted StatsForecast instance.
    train_df : pandas.DataFrame or polars.DataFrame
        The long-format DataFrame used to fit ``sf``. Kept in its own backend:
        a polars frame means every frame handed to StatsForecast is polars.
    horizon : int
        Forecast horizon in time steps.
    freq : str or int
        The freq you passed to StatsForecast: a pandas alias (``'D'``, ``'MS'``)
        for pandas frames, a polars interval (``'1d'``, ``'1mo'``) for polars
        frames, or an int for an integer time column.
    model_name : str
        Which model column to read from StatsForecast output. Must match one
        of the fitted model aliases.
    id_col : str
        Column identifying individual series (default ``'unique_id'``).
    time_col : str
        Timestamp column (default ``'ds'``).
    target_col : str
        Target value column (default ``'y'``).
    exog_features : list of str or None, default None
        Names of the model's future exogenous regressor columns, in the
        order they appear on the exogenous feature axis of ``X``. When
        ``None`` (the default), passing a non-``None`` ``X`` to ``predict``
        raises ``ValueError``.

        Must name *exactly* the columns of ``train_df`` outside
        ``id_col``/``time_col``/``target_col`` -- no more, no less. Anything
        else is refused at construction, because ``cross_validate`` reads its
        regressors from the whole frame while ``predict`` reads only this list,
        and a mismatch calibrates one model to correct another.
    Notes
    -----
    Not serialisable, deliberately: ``sf`` is a live library object and
    ``train_df`` a DataFrame, so ``model_validate(model_dump())`` cannot
    rebuild one -- and it would be refused anyway, since a dump carries the
    derived ``n_series``. Those two fields are hidden from
    ``model_json_schema()`` along with the derived ones, so a schema-filled
    call still has to supply ``sf`` and ``train_df`` programmatically; the
    schema describes the scalar arguments only.
    """

    sf: SkipJsonSchema[StatsForecast] = Field(description="An already-fitted StatsForecast.")
    train_df: SkipJsonSchema[_nx.TrainFrame] = Field(
        description="The long-format pandas or polars DataFrame used to fit ``sf``."
    )
    freq: str | int = Field(
        description=(
            "The freq you passed to StatsForecast: a pandas alias ('D', 'MS') for "
            "pandas frames, a polars interval ('1d', '1mo') for polars frames, or an "
            "int for an integer time column."
        )
    )
    model_name: str = Field(description="Which fitted model alias to read from sf's output.")
    id_col: str = Field(default="unique_id", description="Column identifying individual series.")
    time_col: str = Field(default="ds", description="Timestamp column.")
    target_col: str = Field(default="y", description="Target value column.")
    exog_features: list[str] | None = Field(
        default=None,
        description=(
            "Names of the future exogenous regressor columns, in the order they "
            "appear on X's feature axis. Must name exactly train_df's non-key "
            "columns. None disables exogenous support."
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

        ``n_series`` is required, so a ``mode="after"`` validator is too late:
        validation would fail before it ran. Passing it explicitly is refused
        rather than silently overwritten.

        The count is the only thing derived here; the rest of the validation
        chain stays in :meth:`_validate_and_build`, in its original order. Its
        first step, ``_validate_fitted`` (which names the type it got and whether
        it was fitted), runs here instead, because pydantic type-checks the
        fields between the two validators and would otherwise pre-empt it with a
        less informative message. A non-frame ``train_df`` reaches
        ``TrainFrame``'s validator, which names the problem; until then the
        count falls back to 1, a placeholder that is never used.
        """
        if not isinstance(data, dict):
            return data
        if "n_series" in data:
            raise ValueError("n_series is derived from train_df and must not be passed.")
        df = data.get("train_df")
        if "sf" in data:
            cls._validate_fitted(data["sf"])
        data["n_series"] = _nx.count_series(df, data.get("id_col", "unique_id"))
        return data

    @model_validator(mode="after")
    def _validate_and_build(self) -> StatsForecastAdapter:
        """Run the construction-time validation chain and build the panel state."""
        self._validate_model_name(self.sf, self.model_name)
        train = self._train
        _nx.validate_columns(
            train, id_col=self.id_col, time_col=self.time_col, target_col=self.target_col
        )
        self._validate_exog_columns(train)
        _nx.validate_freq(train, time_col=self.time_col, freq=self.freq)
        _nx.validate_contiguity(train, id_col=self.id_col, time_col=self.time_col, freq=self.freq)
        _nx.validate_no_nan(train, id_col=self.id_col, target_col=self.target_col)
        self._series_ids, self._series_ends = _nx.build_series_bounds(
            train, self.horizon, id_col=self.id_col, time_col=self.time_col
        )

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

    # ------------------------------------------------------------------
    # Construction-time validation helpers
    # ------------------------------------------------------------------

    def supports(self, capability: type) -> bool:
        """Runtime-aware capability check.

        Quantile support here depends on how the adapter was *configured*, not on
        its class: the :class:`SupportsQuantiles` contract has no ``X``
        parameter, so an adapter built with ``exog_features`` cannot honour it --
        there would be no way to supply the future regressors. Reporting that
        through ``supports`` puts the failure **where the forecaster is bound**
        -- the first thing ``calibrate`` / ``calibrate_cv`` do -- which is where
        CLAUDE.md requires capability checks to happen; without it, an exogenous
        adapter calibrates a CQR method happily and only fails at ``predict`` --
        after the user has paid for a full CV calibration.

        Mirrors :meth:`NeuralForecastAdapter.supports`, which gates on the
        fitted loss for the same reason.

        Parameters
        ----------
        capability : type
            A capability mixin class.

        Returns
        -------
        bool
        """
        if capability in (SupportsQuantiles, SupportsCrossValidationQuantiles, SupportsRefit):
            # All three are unavailable once exogenous regressors are in play:
            # the level-based quantile API cannot carry them, and a target-only
            # history panel cannot supply their values to a refit.
            return self.exog_features is None
        return isinstance(self, capability)

    def _validate_exog_columns(self, train_df: nw.DataFrame[Any]) -> None:
        """``exog_features`` and ``train_df``'s regressor columns must agree exactly.

        Both directions matter, and for the same reason: the two paths through
        this adapter read their regressors from different places. ``predict``
        hands ``sf.forecast`` a frame carrying the *declared* features, and
        ``sf.forecast`` re-derives its regressor set from that frame;
        ``cross_validate`` passes ``train_df`` whole, so StatsForecast reads
        *every* non-key column there.

        The missing-feature direction is the shared
        :func:`_nixtla_common.validate_exog_columns` check -- StatsForecast
        otherwise raises from deep inside ``_parse_X_level``, naming neither the
        column nor the cause. The undeclared-column direction is specific to
        this adapter, and worse -- an undeclared column in the frame does not
        break at all. Calibration silently measures a model with more regressors
        than prediction uses, so the conformal quantile belongs to forecasts the
        method will never produce.
        """
        _nx.validate_exog_columns(train_df, exog_features=self.exog_features)

        reserved = {self.id_col, self.time_col, self.target_col}
        declared = list(self.exog_features or [])
        present = [c for c in train_df.columns if c not in reserved]

        undeclared = [c for c in present if c not in declared]
        if undeclared:
            raise ValueError(
                f"train_df carries column(s) {undeclared} that exog_features does not "
                f"name (exog_features={self.exog_features!r}). StatsForecast treats "
                "every non-key column of a frame as a regressor, so cross_validate "
                "would fit with them while predict -- which sees only the declared "
                "features -- would not, and the conformal quantile would be measured "
                "against a different model than the one it corrects. Either pass "
                f"exog_features={sorted(present)!r}, or drop the extra columns from "
                "train_df."
            )

    @staticmethod
    def _validate_fitted(sf: Any) -> None:
        if not isinstance(sf, StatsForecast):
            raise ValueError(f"Expected a StatsForecast instance, got {type(sf).__name__}.")
        if not (hasattr(sf, "fitted_") and sf.fitted_ is not None):
            raise ValueError("StatsForecast instance is not fitted. Call sf.fit(df) first.")

    def _validate_model_name(self, sf: StatsForecast, model_name: str) -> None:
        available = [m.alias if hasattr(m, "alias") else type(m).__name__ for m in sf.models]
        if model_name not in available:
            raise ValueError(
                f"model_name '{model_name}' not found among fitted models. Available: {available}"
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
        """History panel to the long frame ``sf.forecast`` consumes.

        With ``exog_features`` set, the regressors' historical values are joined
        back on from the training frame. ``sf.forecast`` is a stateless
        fit-and-predict: it re-derives the regressor count from the frame it is
        given, so a target-only history contradicts the ``X_df`` that carries the
        future values.
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
        Produce a point forecast anchored at the end of ``history``.

        The anchor follows the panel's width: a history of ``T`` columns against
        a training panel of ``train_t_len`` ends ``train_t_len - T`` steps before
        the training end. A rolling ``predict_batch`` loop therefore stamps each
        window with its own dates, which is what makes seasonal and calendar
        components line up for older windows.

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
        history = self._validate_history(history, allow_ragged=True)
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
        result_df = self.sf.forecast(
            h=self.horizon,
            df=history_df.to_native(),
            X_df=X_df,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
        )

        # Extract point forecasts and reshape
        forecast_panel = self._df_to_forecast_panel(
            nw.from_native(result_df, eager_only=True), self.model_name
        )
        # (n_series, horizon) -> (n_series, 1, horizon)
        return forecast_panel[:, np.newaxis, :]

    # ------------------------------------------------------------------
    # SupportsRefit
    # ------------------------------------------------------------------

    def refit(self, history: Series) -> None:
        """
        Refit the underlying StatsForecast model on a new history panel.

        The set of series must be identical to the original training data.
        If the new panel is longer than the previous one, the end timestamp
        shifts forward proportionally (i.e. extra length = new data appended
        at the end).

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
                "Fit a new StatsForecast on the extended frame and build a new "
                "adapter."
            )
        history = self._validate_history(history, allow_ragged=True)

        # Infer new end: extra steps = new data at the end of the panel.
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
        self.sf.fit(
            history_df.to_native(),
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
            self.sf.cross_validation(
                df=self.train_df,
                h=self.horizon,
                n_windows=n_windows,
                step_size=step_size,
                refit=refit,
                id_col=self.id_col,
                time_col=self.time_col,
                target_col=self.target_col,
            ),
            eager_only=True,
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

    # ------------------------------------------------------------------
    # SupportsQuantiles
    # ------------------------------------------------------------------

    def predict_quantiles(
        self,
        history: Series,
        quantiles: NDArray[np.floating],
    ) -> Forecast:
        """
        Produce quantile forecasts via StatsForecast's level-based interval API.

        Parameters
        ----------
        history : Series, shape (n_series, T)
        quantiles : NDArray, shape (n_quantiles,)
            Values in ``(0, 1)``. Must come in symmetric pairs around 0.5;
            ``q == 0.5`` is rejected.

        Returns
        -------
        Forecast, shape (n_series, n_quantiles, horizon)
            Quantiles are on the middle axis, in the order the user requested.

        Raises
        ------
        ValueError
            If quantiles fail the symmetric-pair / range checks, or if the
            underlying model did not return a requested quantile column.
        UnsupportedCapability
            If the adapter was constructed with ``exog_features``. The
            ``SupportsQuantiles`` contract has no ``X`` parameter, so there is
            no way to supply the future regressors -- and forecasting a model
            fit *with* regressors without them yields intervals that look
            plausible and are not the model's.
        """
        if self.exog_features is not None:
            raise UnsupportedCapability(
                "predict_quantiles cannot be used with an exogenous model: the "
                "SupportsQuantiles contract takes no X, so the future values of "
                f"{self.exog_features} cannot be supplied and the forecast would "
                "silently omit the regressors the model was fit with. Generate the "
                "quantile forecasts yourself with sf.forecast(..., X_df=..., "
                "level=...) and pass the arrays to the method directly."
            )
        history = self._validate_history(history, allow_ragged=True)
        q_arr = np.asarray(quantiles, dtype=np.float64)
        levels = _nx.quantiles_to_levels(q_arr)

        history_df = self._panel_to_df(history)
        result_df = nw.from_native(
            self.sf.forecast(
                h=self.horizon,
                df=history_df.to_native(),
                level=levels,
                id_col=self.id_col,
                time_col=self.time_col,
                target_col=self.target_col,
            ),
            eager_only=True,
        )

        return _nx.stack_quantile_panels(
            result_df,
            model_name=self.model_name,
            q_arr=q_arr,
            resolver=_nx.sf_resolve_quantile_column,
            series_ids=self._series_ids,
            id_col=self.id_col,
            time_col=self.time_col,
            horizon=self.horizon,
        )

    # ------------------------------------------------------------------
    # SupportsCrossValidationQuantiles
    # ------------------------------------------------------------------

    def cross_validate_quantiles(
        self,
        n_windows: int,
        step_size: int,
        quantiles: NDArray[np.floating],
        refit: bool | int = False,
    ) -> tuple[Forecast, Forecast]:
        """
        Run rolling-origin cross-validation producing quantile forecasts.

        Parameters
        ----------
        n_windows : int
            Number of evaluation windows (>= 1).
        step_size : int
            Steps between successive windows (>= 1).
        quantiles : NDArray, shape (n_quantiles,)
            Values in ``(0, 1)``. Symmetric-pair / range rules of
            :func:`_nixtla_common.quantiles_to_levels` apply.
        refit : bool or int
            Whether (or how often) to refit between windows.

        Returns
        -------
        quantile_predictions : Forecast, shape (n_series, n_windows, horizon, n_quantiles)
            Quantiles are on the **last axis**, in the order requested.
        truths : Forecast, shape (n_series, n_windows, horizon)

        Raises
        ------
        ValueError
            If ``n_windows`` / ``step_size`` are invalid, the quantile
            validation fails, or the model did not return a requested
            quantile column.
        RuntimeError
            If the cross-validation output contains NaN.
        """
        if n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {n_windows}")
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")

        q_arr = np.asarray(quantiles, dtype=np.float64)
        levels = _nx.quantiles_to_levels(q_arr)

        cv_df = nw.from_native(
            self.sf.cross_validation(
                df=self.train_df,
                h=self.horizon,
                n_windows=n_windows,
                step_size=step_size,
                refit=refit,
                level=levels,
                id_col=self.id_col,
                time_col=self.time_col,
                target_col=self.target_col,
            ),
            eager_only=True,
        )

        return _nx.reshape_cv_quantiles(
            cv_df,
            model_name=self.model_name,
            target_col=self.target_col,
            q_arr=q_arr,
            resolver=_nx.sf_resolve_quantile_column,
            series_ids=self._series_ids,
            id_col=self.id_col,
            time_col=self.time_col,
            horizon=self.horizon,
            library_name="StatsForecast",
        )
