"""Rectify multi-step forecast correction utilities."""

__all__ = ["compute_rectify_residuals", "align_rectify_features", "rectify"]


import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast, overload

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.stable.v2.typing import IntoDataFrameT

from utilsforecast.validation import validate_format

PerHorizonTrainingData = Dict[int, Tuple[np.ndarray, Dict[str, np.ndarray]]]
HorizonAwareTrainingData = Tuple[np.ndarray, Dict[str, np.ndarray]]
PerHorizonCorrectionModels = Dict[int, Dict[str, Any]]
HorizonAwareCorrectionModels = Dict[str, Any]
Mode = Literal["per_horizon", "horizon_aware"]


def _validate_features(features: np.ndarray, n_rows: int, frame_name: str) -> None:
    if features.ndim != 2:
        raise ValueError("features must be a 2-dimensional numpy array")
    if features.shape[0] != n_rows:
        raise ValueError(
            "features must have the same number of rows as "
            f"{frame_name}, got {features.shape[0]} and {n_rows}"
        )


def _validate_mode(mode: str) -> None:
    if mode not in ("per_horizon", "horizon_aware"):
        raise ValueError(
            f"mode must be 'per_horizon' or 'horizon_aware', got '{mode}'"
        )


def _validate_predictor(model: Any, model_name: str) -> None:
    if not hasattr(model, "predict"):
        raise ValueError(
            f"correction model for '{model_name}' must have a predict method"
        )


def compute_rectify_residuals(
    df: IntoDataFrameT,
    forecasts_df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    cutoff_col: Optional[str] = None,
) -> IntoDataFrameT:
    """Compute per-horizon residuals between actuals and base forecasts.

    Args:
        df (pandas or polars DataFrame): Actuals with columns
            [id_col, time_col, target_col].
        forecasts_df (pandas or polars DataFrame): Base forecasts
            with columns [id_col, time_col, *models].
        models (list of str): Columns that identify the models predictions
            in forecasts_df.
        id_col (str, optional): Column that identifies each serie.
            Defaults to 'unique_id'.
        time_col (str, optional): Column that identifies each timestep.
            Defaults to 'ds'.
        target_col (str, optional): Column that contains the target.
            Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the CV fold
            cutoff. When provided, join and horizon computation are
            grouped by (id_col, cutoff_col). Required for cross-validation
            output where (id_col, time_col) pairs repeat across folds.
            Defaults to None.

    Returns:
        pandas or polars DataFrame: DataFrame with columns
            [id_col, time_col, 'horizon', *models] where each model column
            contains the residual (actual - forecast).
    """
    validate_format(df, id_col=id_col, time_col=time_col, target_col=target_col)
    validate_format(forecasts_df, id_col=id_col, time_col=time_col, target_col=None)
    missing_models = sorted(set(models) - set(forecasts_df.columns))
    if missing_models:
        raise ValueError(f"forecasts_df is missing model columns: {missing_models}")
    if cutoff_col is not None:
        if cutoff_col not in df.columns:
            raise ValueError(f"df is missing cutoff column: {cutoff_col}")
        if cutoff_col not in forecasts_df.columns:
            raise ValueError(f"forecasts_df is missing cutoff column: {cutoff_col}")
    actuals = nw.from_native(df)
    forecasts = nw.from_native(forecasts_df)
    join_on = [id_col, time_col]
    actuals_cols = [id_col, time_col, target_col]
    forecasts_cols = [id_col, time_col, *models]
    if cutoff_col is not None:
        join_on = [id_col, time_col, cutoff_col]
        actuals_cols.append(cutoff_col)
        forecasts_cols.append(cutoff_col)
    actuals_selected = actuals.select(actuals_cols)
    merged = actuals_selected.join(
        forecasts.select(forecasts_cols),
        on=join_on,
        how="inner",
    )
    n_dropped = len(actuals_selected) - len(merged)
    if n_dropped > 0:
        warnings.warn(
            f"{n_dropped} row(s) in actuals had no matching entry in forecasts_df "
            f"on {join_on} and were dropped. Check that both DataFrames cover the "
            "same (id, time) pairs.",
            UserWarning,
            stacklevel=2,
        )
    residual_exprs = [
        (nw.col(target_col) - nw.col(model)).alias(model) for model in models
    ]
    partition_by = [id_col] if cutoff_col is None else [id_col, cutoff_col]
    horizon_expr = (
        nw.col(time_col)
        .cum_count()
        .over(*partition_by, order_by=time_col)
        .cast(nw.Int32)
        .alias("horizon")
    )
    meta_cols = [id_col, time_col]
    if cutoff_col is not None:
        meta_cols.append(cutoff_col)
    sort_cols = [id_col, time_col]
    if cutoff_col is not None:
        sort_cols = [id_col, cutoff_col, time_col]
    result = merged.select(
        [*meta_cols, horizon_expr, *residual_exprs]
    ).sort(*sort_cols)
    return nw.to_native(result)


@overload
def align_rectify_features(
    residuals_df: IntoDataFrameT,
    features: np.ndarray,
    models: Optional[List[str]] = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    mode: Literal["per_horizon"] = "per_horizon",
) -> PerHorizonTrainingData:
    ...


@overload
def align_rectify_features(
    residuals_df: IntoDataFrameT,
    features: np.ndarray,
    models: Optional[List[str]] = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    *,
    mode: Literal["horizon_aware"],
) -> HorizonAwareTrainingData:
    ...


def align_rectify_features(
    residuals_df: IntoDataFrameT,
    features: np.ndarray,
    models: Optional[List[str]] = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    mode: Mode = "per_horizon",
) -> Union[PerHorizonTrainingData, HorizonAwareTrainingData]:
    """Align feature matrices with horizon-indexed residuals.

    Args:
        residuals_df (pandas or polars DataFrame): Output of
            compute_rectify_residuals.
        features (numpy ndarray): Feature matrix with shape
            (n_samples, n_features), row-aligned with residuals_df.
        models (list of str, optional): Columns that identify the model
            residuals. If None, all columns except id_col, time_col, and
            'horizon' are used.
        id_col (str, optional): Column that identifies each serie.
            Defaults to 'unique_id'.
        time_col (str, optional): Column that identifies each timestep.
            Defaults to 'ds'.
        mode (str, optional): 'per_horizon' or 'horizon_aware'.
            Defaults to 'per_horizon'.

    Returns:
        dict or tuple: Training data for rectification models.
            In per_horizon mode: {horizon: (X, {model_name: residuals})}.
            In horizon_aware mode: (X_with_horizon_col, {model_name: residuals}).
    """
    _validate_mode(mode)
    validate_format(
        residuals_df, id_col=id_col, time_col=time_col, target_col=None
    )
    residuals = nw.from_native(residuals_df)
    if "horizon" not in residuals.columns:
        raise ValueError("residuals_df is missing horizon column")
    _validate_features(features, residuals.shape[0], "residuals_df")
    if models is not None:
        model_cols = models
    else:
        model_cols = [
            c for c in residuals.columns if c not in (id_col, time_col, "horizon")
        ]
    missing_models = sorted(set(model_cols) - set(residuals.columns))
    if missing_models:
        raise ValueError(f"residuals_df is missing model columns: {missing_models}")
    if not model_cols:
        raise ValueError("models must contain at least one model column")
    horizons = residuals["horizon"].to_numpy()
    if mode == "per_horizon":
        result: PerHorizonTrainingData = {}
        for h in sorted(int(h) for h in set(horizons)):
            mask = horizons == h
            X_h = features[mask]
            y_dict = {m: residuals[m].to_numpy()[mask] for m in model_cols}
            result[h] = (X_h, y_dict)
        return result
    elif mode == "horizon_aware":
        X_aug = np.column_stack([features, horizons])
        y_dict = {m: residuals[m].to_numpy() for m in model_cols}
        return X_aug, y_dict


@overload
def rectify(
    df: IntoDataFrameT,
    models: List[str],
    correction_models: PerHorizonCorrectionModels,
    features: np.ndarray,
    id_col: str = "unique_id",
    time_col: str = "ds",
    mode: Literal["per_horizon"] = "per_horizon",
) -> IntoDataFrameT:
    ...


@overload
def rectify(
    df: IntoDataFrameT,
    models: List[str],
    correction_models: HorizonAwareCorrectionModels,
    features: np.ndarray,
    id_col: str = "unique_id",
    time_col: str = "ds",
    *,
    mode: Literal["horizon_aware"],
) -> IntoDataFrameT:
    ...


def rectify(
    df: IntoDataFrameT,
    models: List[str],
    correction_models: Union[PerHorizonCorrectionModels, HorizonAwareCorrectionModels],
    features: np.ndarray,
    id_col: str = "unique_id",
    time_col: str = "ds",
    mode: Mode = "per_horizon",
) -> IntoDataFrameT:
    """Apply rectification corrections to base forecasts.

    Args:
        df (pandas or polars DataFrame): Base forecasts with
            columns [id_col, time_col, *models].
        models (list of str): Columns that identify the models predictions.
        correction_models (dict): Fitted correction models.
            In per_horizon mode: {horizon: {model_name: fitted_model}}.
            In horizon_aware mode: {model_name: fitted_model}.
        features (numpy ndarray): Feature matrix with shape
            (n_samples, n_features), row-aligned with df.
        id_col (str, optional): Column that identifies each serie.
            Defaults to 'unique_id'.
        time_col (str, optional): Column that identifies each timestep.
            Defaults to 'ds'.
        mode (str, optional): 'per_horizon' or 'horizon_aware'.
            Defaults to 'per_horizon'. The horizon is recomputed from the
            inference dataframe, so cutoff_col is not supported here.

    Returns:
        pandas or polars DataFrame: Corrected forecasts with same schema
            as input df.
    """
    _validate_mode(mode)
    validate_format(df, id_col=id_col, time_col=time_col, target_col=None)
    missing_models = sorted(set(models) - set(df.columns))
    if missing_models:
        raise ValueError(f"df is missing model columns: {missing_models}")
    frame = nw.from_native(df)
    _validate_features(features, frame.shape[0], "df")
    # At inference time there is no CV fold dimension; horizons are assigned
    # within each series in the forecast panel being corrected.
    horizon_expr = (
        nw.col(time_col)
        .cum_count()
        .over(id_col, order_by=time_col)
        .cast(nw.Int32)
        .alias("__rectify_horizon__")
    )
    with_horizon = frame.with_columns(horizon_expr)
    horizons = with_horizon["__rectify_horizon__"].to_numpy()
    corrections = np.zeros((len(frame), len(models)))
    if mode == "per_horizon":
        per_horizon_models = cast(PerHorizonCorrectionModels, correction_models)
        unique_horizons = sorted(int(h) for h in set(horizons))
        missing_horizons = sorted(set(unique_horizons) - set(per_horizon_models))
        if missing_horizons:
            raise ValueError(
                "correction_models is missing horizons for per_horizon "
                f"mode: {missing_horizons}"
            )
        for h in unique_horizons:
            mask = horizons == h
            X_h = features[mask]
            horizon_models = per_horizon_models[h]
            if not isinstance(horizon_models, dict):
                raise ValueError(
                    "per_horizon mode expects correction_models to map each "
                    "horizon to a dict of fitted models"
                )
            missing_horizon_models = sorted(set(models) - set(horizon_models))
            if missing_horizon_models:
                raise ValueError(
                    f"correction_models[{h}] is missing model columns: "
                    f"{missing_horizon_models}"
                )
            for j, model in enumerate(models):
                _validate_predictor(horizon_models[model], model)
                corrections[mask, j] = horizon_models[model].predict(X_h)
    elif mode == "horizon_aware":
        horizon_aware_models = cast(HorizonAwareCorrectionModels, correction_models)
        missing_keys = sorted(set(models) - set(horizon_aware_models))
        if missing_keys:
            raise ValueError(
                "correction_models is missing model columns for "
                f"horizon_aware mode: {missing_keys}"
            )
        X_aug = np.column_stack([features, horizons])
        for j, model in enumerate(models):
            _validate_predictor(horizon_aware_models[model], model)
            corrections[:, j] = horizon_aware_models[model].predict(X_aug)
    corrected = with_horizon.drop("__rectify_horizon__")
    for j, model in enumerate(models):
        corrected = corrected.with_columns(
            (nw.col(model) + corrections[:, j]).alias(model)
        )
    return nw.to_native(corrected)
