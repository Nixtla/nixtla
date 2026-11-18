"""NeuralForecast adapter for conformalforecast.

Wraps a fitted ``neuralforecast.NeuralForecast`` instance and exposes it
through the conformalforecast adapter contract.

Scope
-----
The adapter exposes panel-shaped point forecasts, quantile forecasts, refit,
and cross-validation. Quantile output is gated on whether the underlying model
was fit with a probabilistic loss (any of the quantile-loss classes —
``QuantileLoss`` / ``MQLoss`` / ``IQLoss`` and their Huber variants — or any
loss that exposes ``is_distribution_output = True`` such as ``DistributionLoss``,
``PMM``, ``GMM``).

The adapter always declares the :class:`SupportsQuantiles` and
:class:`SupportsCrossValidationQuantiles` mixins, so ``isinstance`` checks
return True regardless of the loss. However, the ``supports()`` method is
overridden to consult the :attr:`_supports_quantiles_runtime` flag (False for
point-loss models), which gates quantile capabilities **where the forecaster is
bound** for capability-checking callers like :class:`ConformalMethod` (e.g.,
CQR) -- that is, at the top of ``calibrate`` / ``calibrate_cv``, before any
forecasting happens. In addition, calling ``predict_quantiles`` or
``cross_validate_quantiles`` on a point-loss adapter raises
:class:`UnsupportedCapability` at call time. This dual-gate pattern is necessary
because the capability depends on a constructor argument (the loss), not the
adapter class itself.

Limitations (v0.1)
------------------
- Future exogenous features are supported: build the model with
  ``futr_exog_list=[...]`` and pass a matching ``train_df``. The adapter reads
  the list off the model rather than taking an ``exog_features`` argument, since
  NeuralForecast *requires* those values at predict time -- unlike the
  StatsForecast and MLForecast adapters, where exogenous use is opt-in.
  ``predict`` then joins the regressors' history onto its frame, so the history
  must lie within the training frame's dates, and ``refit`` raises. Quantile
  methods (CQR) are unavailable on such a model: ``predict_quantiles`` has no
  ``X`` in its contract.
- Static and historical (past-only) covariates are still rejected at
  construction: a history panel has no way to carry them.
- All models in the underlying ``NeuralForecast`` must share the same horizon.
- ``refit`` retrains the neural network on the new history. This is
  significantly slower than the StatsForecast / MLForecast equivalents; for
  online-style usage, batch refits.
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

from ..base import Forecast, ForecasterAdapter, Series, UnsupportedCapability
from ..capabilities import (
    SupportsCrossValidation,
    SupportsCrossValidationQuantiles,
    SupportsQuantiles,
    SupportsRefit,
)

try:
    from neuralforecast import NeuralForecast  # type: ignore[import-untyped]
except ImportError as _err:
    raise ImportError(
        "NeuralForecastAdapter requires the 'neuralforecast' package. "
        "Install it with: pip install conformalforecast[nixtla]"
    ) from _err

from . import _nixtla_common as _nx


def _model_alias(model: Any) -> str:
    """Resolve a NeuralForecast model's effective alias.

    NF models carry an ``alias`` attribute that defaults to ``None``; in that
    case the column name in predict output falls back to the class name.
    """
    alias = getattr(model, "alias", None)
    return alias if alias is not None else type(model).__name__


# NeuralForecast losses that produce quantile output deterministically.
# Distribution losses (DistributionLoss / PMM / GMM / ISQF) advertise themselves
# via the ``is_distribution_output`` attribute and are detected separately.
QUANTILE_LOSS_CLASS_NAMES = frozenset(
    {
        "QuantileLoss",
        "MQLoss",
        "IQLoss",
        "HuberQLoss",
        "HuberMQLoss",
        "HuberIQLoss",
    }
)


def _supports_quantiles(loss: Any) -> bool:
    """Detect whether a NeuralForecast loss can produce quantile output.

    Two categories qualify:

    * Distribution losses (``DistributionLoss``, ``PMM``, ``GMM``, ``ISQF``)
      which set ``is_distribution_output = True`` and produce quantiles via
      sampling.
    * Direct quantile losses (``QuantileLoss``, ``MQLoss``, ``IQLoss``, and
      their ``Huber*`` variants) which produce quantiles deterministically.
    """
    if getattr(loss, "is_distribution_output", False):
        return True
    return type(loss).__name__ in QUANTILE_LOSS_CLASS_NAMES


class NeuralForecastAdapter(
    ForecasterAdapter,
    SupportsRefit,
    SupportsCrossValidation,
    SupportsQuantiles,
    SupportsCrossValidationQuantiles,
):
    """
    Adapter for a fitted :class:`neuralforecast.NeuralForecast` instance.

    The adapter wraps **point** forecasters only. Any NeuralForecast
    configuration that produces intervals natively (``prediction_intervals``)
    is rejected at construction; conformalforecast produces its own intervals
    from point forecasts.

    The horizon is read from the underlying models' ``h`` attribute. All
    models in ``nf.models`` must share the same horizon.

    ``horizon``, ``n_series`` and ``exog_features`` are all derived -- from the
    models' ``h``, from ``train_df``, and from the selected model's
    ``futr_exog_list`` respectively -- rather than accepted; passing any of them
    is refused at construction.

    Parameters
    ----------
    nf : NeuralForecast
        An already-fitted NeuralForecast instance with point models only.
    train_df : pandas.DataFrame or polars.DataFrame
        The long-format DataFrame used to fit ``nf``. Kept in its own backend:
        a polars frame means every frame handed to NeuralForecast is polars.
    freq : str or int
        The freq you passed to NeuralForecast: a pandas alias (``'D'``, ``'MS'``)
        for pandas frames, a polars interval (``'1d'``, ``'1mo'``) for polars
        frames, or an int for an integer time column.
    model_name : str
        Which model column to read from NeuralForecast output. Must be the
        alias of a model in ``nf.models`` (or its class name if no alias was
        set explicitly).
    id_col : str
        Column identifying individual series (default ``'unique_id'``).
    time_col : str
        Timestamp column (default ``'ds'``).
    target_col : str
        Target value column (default ``'y'``).

        All three must match the names ``nf.fit`` recorded: ``NeuralForecast.predict``
        takes no column-name arguments and reuses them, so a mismatch is
        refused at construction.
    Notes
    -----
    Not serialisable, deliberately: ``nf`` is a live library object and
    ``train_df`` a DataFrame, so ``model_validate(model_dump())`` cannot
    rebuild one -- and it would be refused anyway, since a dump carries the
    derived ``horizon``, ``n_series`` and ``exog_features``. Those two fields are hidden from
    ``model_json_schema()`` along with the derived ones, so a schema-filled
    call still has to supply ``nf`` and ``train_df`` programmatically; the
    schema describes the scalar arguments only.
    """

    nf: SkipJsonSchema[NeuralForecast] = Field(
        description="An already-fitted NeuralForecast with point models only."
    )
    train_df: SkipJsonSchema[_nx.TrainFrame] = Field(
        description="The long-format pandas or polars DataFrame used to fit ``nf``."
    )
    freq: str | int = Field(
        description=(
            "The freq you passed to NeuralForecast: a pandas alias ('D', 'MS') for "
            "pandas frames, a polars interval ('1d', '1mo') for polars frames, or an "
            "int for an integer time column."
        )
    )
    model_name: str = Field(description="Which fitted model alias to read from nf's output.")
    id_col: str = Field(default="unique_id", description="Column identifying individual series.")
    time_col: str = Field(default="ds", description="Timestamp column.")
    target_col: str = Field(default="y", description="Target value column.")
    exog_features: SkipJsonSchema[list[str] | None] = Field(
        default=None,
        description=(
            "Derived, not accepted: the selected model's futr_exog_list, or None "
            "if it has none. NeuralForecast models record their own regressors "
            "and predict requires them, so the model is the source of truth."
        ),
    )

    # All three are derived in the before-/after-validators and refused if
    # passed, so they are hidden from the schema a caller fills in rather than
    # advertised as inputs.
    horizon: SkipJsonSchema[int] = Field(ge=1, description="The models' shared h. Derived.")
    n_series: SkipJsonSchema[int] = Field(
        ge=1, description="Number of series in train_df, the leading array axis. Derived."
    )

    _series_ids: tuple[Any, ...] = PrivateAttr(default=())
    _series_ends: Any = PrivateAttr(default=None)  # nw.Series aligned with _series_ids
    _train_panel: Any = PrivateAttr(default=None)
    _loss_class_name: str = PrivateAttr(default="")
    _supports_quantiles_runtime: bool = PrivateAttr(default=False)

    @property
    def _train(self) -> nw.DataFrame[Any]:
        """``train_df`` as a narwhals frame: a view, not a copy."""
        return nw.from_native(self.train_df, eager_only=True)

    @model_validator(mode="before")
    @classmethod
    def _derive_shape(cls, data: Any) -> Any:
        """Fill the inherited ``horizon`` and ``n_series`` fields from ``nf``/``train_df``.

        Both are required fields, so nothing later than a ``mode="before"``
        validator can supply them, and this adapter has never accepted either --
        the horizon is the models' shared ``h`` and the series count is
        ``train_df``'s -- so both are refused if passed rather than silently
        overwritten. ``exog_features`` is refused for the same reason but has a
        default, so it is filled at its original place in
        :meth:`_validate_and_build`'s chain instead.

        Only the two numbers are derived here; the rest of the validation chain
        stays in :meth:`_validate_and_build`, in its original order. Its first
        two steps run here instead, still in that order, because pydantic
        type-checks the fields between the two validators and would otherwise
        pre-empt them with a less informative message. Where a value cannot be
        read the placeholder 1 is injected, which the after-validator is
        guaranteed to reject with the real error.
        """
        if not isinstance(data, dict):
            return data
        for derived, source in (
            ("horizon", "the models' h"),
            ("n_series", "train_df"),
            ("exog_features", "the model's futr_exog_list"),
        ):
            if derived in data:
                raise ValueError(f"{derived} is derived from {source} and must not be passed.")
        df = data.get("train_df")
        if "nf" in data:
            cls._validate_fitted(data["nf"])
        data["n_series"] = _nx.count_series(df, data.get("id_col", "unique_id"))

        horizons = {int(m.h) for m in getattr(data.get("nf"), "models", None) or []}
        data["horizon"] = horizons.pop() if len(horizons) == 1 else 1
        return data

    @model_validator(mode="after")
    def _validate_and_build(self) -> NeuralForecastAdapter:
        """Run the construction-time validation chain and build the panel state."""
        self._validate_model_name(self.nf, self.model_name)
        self._validate_no_prediction_intervals(self.nf)
        self._validate_column_names(self.nf)
        self._validate_horizon_consistency(self.nf)
        self.exog_features = self._validate_supported_exogenous(self.nf, self.model_name)
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

        # Quantile capability is loss-driven, not class-driven. We always
        # declare SupportsQuantiles / SupportsCrossValidationQuantiles at the
        # class level (so isinstance is True), and override supports() so the
        # gate fires where a method binds this forecaster -- at the top of
        # calibrate / calibrate_cv. We also gate the actual methods at call time.
        selected_model = next(m for m in self.nf.models if _model_alias(m) == self.model_name)
        self._loss_class_name = type(selected_model.loss).__name__
        self._supports_quantiles_runtime = _supports_quantiles(selected_model.loss)
        return self

    def supports(self, capability: type) -> bool:
        """Runtime-aware capability check.

        Quantile capabilities depend on the fitted loss, not the adapter class,
        so this override consults ``_supports_quantiles_runtime`` for
        :class:`SupportsQuantiles` / :class:`SupportsCrossValidationQuantiles`
        and defers to the default ``isinstance`` check otherwise. This puts the
        runtime quantile gate at the top of ``calibrate`` / ``calibrate_cv``,
        where the forecaster is bound, rather than surfacing it only at
        ``predict`` time.

        Parameters
        ----------
        capability : type
            A capability mixin class.

        Returns
        -------
        bool
        """
        if capability in (SupportsQuantiles, SupportsCrossValidationQuantiles):
            # predict_quantiles has no X in its contract, so it cannot supply the
            # futr_df an exogenous model demands -- even though NeuralForecast
            # itself would happily return quantiles for one.
            return self._supports_quantiles_runtime and self.exog_features is None
        if capability is SupportsRefit:
            # A history panel carries target values only, so refitting would
            # retrain the network without its regressors.
            return self.exog_features is None
        return isinstance(self, capability)

    # ------------------------------------------------------------------
    # Construction-time validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_fitted(nf: Any) -> None:
        if not isinstance(nf, NeuralForecast):
            raise ValueError(f"Expected a NeuralForecast instance, got {type(nf).__name__}.")
        if not getattr(nf, "_fitted", False):
            raise ValueError("NeuralForecast instance is not fitted. Call nf.fit(df) first.")

    def _validate_model_name(self, nf: NeuralForecast, model_name: str) -> None:
        aliases = [_model_alias(m) for m in nf.models]
        if model_name not in aliases:
            raise ValueError(
                f"model_name '{model_name}' not found among fitted models. Available: {aliases}"
            )

    def _validate_column_names(self, nf: NeuralForecast) -> None:
        """The adapter's column names must be the ones ``nf.fit`` recorded.

        ``NeuralForecast.predict`` takes no ``id_col``/``time_col``/``target_col``:
        it reuses what ``fit`` stored on the instance, so custom names only work
        when they were passed there too. See
        :func:`_nixtla_common.validate_fitted_column_names`.
        """
        _nx.validate_fitted_column_names(
            {
                "id_col": getattr(nf, "id_col", None),
                "time_col": getattr(nf, "time_col", None),
                "target_col": getattr(nf, "target_col", None),
            },
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
            library_name="NeuralForecast",
            fit_call="nf.fit",
        )

    def _validate_no_prediction_intervals(self, nf: NeuralForecast) -> None:
        # NeuralForecast.fit assigns ``_cs_df`` to a calibration DataFrame only
        # when prediction_intervals is passed; otherwise it stays None (or
        # absent on instances that have never gone through fit).
        if getattr(nf, "_cs_df", None) is not None:
            raise ValueError(
                "NeuralForecast was fit with prediction_intervals. conformalforecast "
                "produces its own intervals from point forecasts — refit with "
                "nf.fit(df) without the prediction_intervals argument."
            )

    def _validate_horizon_consistency(self, nf: NeuralForecast) -> int:
        horizons = {_model_alias(m): int(m.h) for m in nf.models}
        unique_h = set(horizons.values())
        if len(unique_h) != 1:
            raise ValueError(
                f"All models in NeuralForecast must share the same horizon h. Got: {horizons}."
            )
        return int(nf.models[0].h)

    def _validate_supported_exogenous(
        self, nf: NeuralForecast, model_name: str
    ) -> list[str] | None:
        """Reject static and historical covariates; return the future regressors.

        Unlike the StatsForecast and MLForecast adapters, this one takes no
        ``exog_features`` argument: a NeuralForecast model records its own
        ``futr_exog_list``, and ``predict`` on such a model *requires* those
        values -- it raises without a ``futr_df``. So the model is the source of
        truth and the feature is not opt-in.

        Returns
        -------
        list of str or None
            The model's future exogenous columns, or None if it has none.

        Raises
        ------
        ValueError
            If the model uses historical or static covariates, which the
            ``Series`` panel has no way to carry.
        """
        target = next(m for m in nf.models if _model_alias(m) == model_name)
        unsupported = {
            "historical exogenous": "hist_exog_list",
            "static": "stat_exog_list",
        }
        for label, attr in unsupported.items():
            features = getattr(target, attr, None)
            if features:
                raise ValueError(
                    f"Model '{model_name}' uses {label} features ({list(features)}). "
                    "Only future exogenous features are supported: a history panel "
                    "carries target values only, so past-only and static covariates "
                    "have nowhere to travel. Future regressors work -- declare them "
                    "as futr_exog_list on the model."
                )
        futr = getattr(target, "futr_exog_list", None)
        return list(futr) if futr else None

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
        """History panel to the long frame ``nf.predict`` consumes as ``df``.

        For a model with ``futr_exog_list``, the regressors' historical values
        are joined back on from the training frame: NeuralForecast rejects a
        target-only history with "future exogenous variables not found in input
        dataset".
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
            Future exogenous features. Not supported by this adapter.

        Returns
        -------
        Forecast, shape (n_series, 1, horizon)

        Raises
        ------
        ValueError
            If ``X`` is not None but the model declares no future exogenous
            features, or if it declares them and ``X`` is None.
        """
        history = self._validate_history(np.asarray(history, dtype=np.float64), allow_ragged=True)

        history_df = self._panel_to_df(history)
        futr_df = None
        if self.exog_features is None:
            if X is not None:
                raise ValueError(
                    "predict received X but model "
                    f"'{self.model_name}' declares no future exogenous features. "
                    "Build the model with futr_exog_list=[...] to use them."
                )
        else:
            if X is None:
                # NeuralForecast raises on its own here, but only after the
                # history frame has been built and validated -- and its message
                # names neither the adapter nor how to supply the values.
                raise ValueError(
                    f"Model '{self.model_name}' requires future exogenous "
                    f"features {self.exog_features}, so predict needs X with "
                    f"shape (n_series, 1, horizon, {len(self.exog_features)})."
                )
            futr_df = _nx.exog_to_futr_df(
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
        result_df = self.nf.predict(df=history_df.to_native(), futr_df=futr_df)

        forecast_panel = self._df_to_forecast_panel(
            nw.from_native(result_df, eager_only=True), self.model_name
        )
        return forecast_panel[:, np.newaxis, :]

    # ------------------------------------------------------------------
    # SupportsRefit
    # ------------------------------------------------------------------

    def refit(self, history: Series) -> None:
        """
        Refit the underlying NeuralForecast model on a new history panel.

        The set of series must be identical to the original training data.
        If the new panel is longer than the previous one, the end timestamp
        shifts forward proportionally.

        Note that NeuralForecast's ``fit`` retrains the neural networks from
        scratch and may take significantly longer than the StatsForecast or
        MLForecast equivalents. For online-style usage, batch the refits.

        Not available when the model declares future exogenous features: a
        ``Series`` panel carries target values only, so there is nowhere for the
        regressors' new values to come from.

        Parameters
        ----------
        history : Series, shape (n_series, T)

        Raises
        ------
        ValueError
            If the model declares future exogenous features.
        """
        if self.exog_features is not None:
            raise ValueError(
                f"refit is not available: model '{self.model_name}' requires "
                f"future exogenous features {self.exog_features}, and a history "
                "panel carries target values only, so refitting would retrain "
                "the network without its regressors. Fit a new NeuralForecast "
                "on the extended frame and build a new adapter."
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
        self.nf.fit(
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
            self.nf.cross_validation(
                df=self.train_df,
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
    # SupportsQuantiles (runtime-gated)
    # ------------------------------------------------------------------

    def _require_runtime_quantile_support(self) -> None:
        if self.exog_features is not None:
            raise UnsupportedCapability(
                f"predict_quantiles has no X parameter, so it cannot supply the "
                f"future exogenous features {self.exog_features} that model "
                f"'{self.model_name}' requires. Use a point-forecast method "
                "(SplitConformal, ACI, NexCP, AgACI, SPCI) with this adapter."
            )
        if not self._supports_quantiles_runtime:
            raise UnsupportedCapability(
                f"NeuralForecast model '{self.model_name}' was fit with "
                f"non-probabilistic loss '{self._loss_class_name}'. Use a "
                "probabilistic loss: any quantile loss (QuantileLoss, MQLoss, "
                "IQLoss, HuberQLoss, HuberMQLoss, HuberIQLoss) or any "
                "DistributionLoss / PMM / GMM."
            )

    def predict_quantiles(
        self,
        history: Series,
        quantiles: NDArray[np.floating],
    ) -> Forecast:
        """
        Produce quantile forecasts via NeuralForecast's level-based API.

        Parameters
        ----------
        history : Series, shape (n_series, T)
        quantiles : NDArray, shape (n_quantiles,)
            Values in ``(0, 1)``. Must come in symmetric pairs around 0.5;
            ``q == 0.5`` is rejected. NeuralForecast's runtime behaviour for
            ``MQLoss`` ignores arbitrary quantile requests in favour of the
            quantiles the loss was trained with — this adapter therefore
            uses ``level=`` (the underlying API both losses honour) and
            requires symmetric inputs.

        Returns
        -------
        Forecast, shape (n_series, n_quantiles, horizon)
            Quantiles are on the middle axis, in the order requested.

        Raises
        ------
        UnsupportedCapability
            If the underlying model was fit with a non-probabilistic loss.
        ValueError
            If the symmetric-pair / range checks fail, or if the model did
            not return a requested quantile column. For ``MQLoss``, the
            requested quantiles must match a symmetric pair the loss was
            trained with.
        """
        self._require_runtime_quantile_support()
        history = self._validate_history(np.asarray(history, dtype=np.float64), allow_ragged=True)
        q_arr = np.asarray(quantiles, dtype=np.float64)
        levels = _nx.quantiles_to_levels(q_arr)

        history_df = self._panel_to_df(history)
        result_df = nw.from_native(
            self.nf.predict(df=history_df.to_native(), level=levels), eager_only=True
        )

        return _nx.stack_quantile_panels(
            result_df,
            model_name=self.model_name,
            q_arr=q_arr,
            resolver=_nx.nf_resolve_quantile_column,
            series_ids=self._series_ids,
            id_col=self.id_col,
            time_col=self.time_col,
            horizon=self.horizon,
        )

    # ------------------------------------------------------------------
    # SupportsCrossValidationQuantiles (runtime-gated)
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
        UnsupportedCapability
            If the underlying model was fit with a non-probabilistic loss.
        ValueError
            If ``n_windows`` / ``step_size`` are invalid, the quantile
            validation fails, or the model did not return a requested
            quantile column.
        RuntimeError
            If the cross-validation output contains NaN.
        """
        self._require_runtime_quantile_support()
        if n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {n_windows}")
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")
        q_arr = np.asarray(quantiles, dtype=np.float64)
        levels = _nx.quantiles_to_levels(q_arr)

        cv_df = nw.from_native(
            self.nf.cross_validation(
                df=self.train_df,
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
            resolver=_nx.nf_resolve_quantile_column,
            series_ids=self._series_ids,
            id_col=self.id_col,
            time_col=self.time_col,
            horizon=self.horizon,
            library_name="NeuralForecast",
        )
