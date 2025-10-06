"""Loss functions for model evaluation."""

__all__ = ['mae', 'mse', 'rmse', 'bias', 'cfe', 'pis', 'spis', 'mape', 'smape', 'mase', 'rmae', 'nd', 'msse', 'rmsse',
           'quantile_loss', 'scaled_quantile_loss', 'mqloss', 'scaled_mqloss', 'coverage', 'calibration', 'scaled_crps',
           'tweedie_deviance']


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import utilsforecast.processing as ufp

from .compat import DataFrame, DFType, pl, pl_DataFrame, pl_Expr


def _base_docstring(*args, **kwargs) -> Callable:
    base_docstring = """

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, actual values and predictions.
        models (list of str): Columns that identify the models predictions.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.
    """

    def docstring_decorator(f: Callable):
        if f.__doc__ is not None:
            f.__doc__ += base_docstring
        return f

    return docstring_decorator(*args, **kwargs)


def _pl_agg_expr(
    df: pl_DataFrame,
    models: Union[List[str], List[Tuple[str, str]]],
    id_col: str,
    gen_expr: Callable[[Any], "pl.Expr"],
    agg: str = "mean",
) -> pl_DataFrame:
    exprs = [gen_expr(model) for model in models]
    df = df.select([id_col, *exprs])
    gb = ufp.group_by(df, id_col, maintain_order=True)
    if agg == "mean":
        return gb.mean()
    else:
        return gb.sum()


def _scale_loss(
    loss_df: DFType,
    scale_type: str,
    models: List[str],
    seasonality: int,
    train_df: DFType,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """
    Args:
        loss_df (pandas or polars DataFrame): Input dataframe with id, actuals, predictions and losses results.
        scale_type (str): Type of scaling. Possible values are 'absolute_error' or 'squared_error'.
        models (list of str): Columns that identify the models predictions.
        seasonality (int): Main frequency of the time series;
            Hourly 24, Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.
        train_df (pandas or polars DataFrame): Training dataframe with id and actual values. Must be sorted by time.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://robjhyndman.com/papers/mase.pdf
    """

    if isinstance(train_df, pd.DataFrame):
        loss_df = loss_df.set_index(id_col)
        # assume train_df is sorted
        lagged = train_df.groupby(id_col, observed=True)[target_col].shift(seasonality)
        if scale_type == "absolute_error":
            scale = train_df[target_col].sub(lagged).abs()
        else:
            scale = train_df[target_col].sub(lagged).pow(2)
        scale = scale.groupby(train_df[id_col], observed=True).mean()
        res = loss_df.div(_zero_to_nan(scale), axis=0).fillna(0)
        res.index.name = id_col
        res = res.reset_index()
    else:
        # assume train_df is sorted
        lagged = pl.col(target_col).shift(seasonality).over(id_col)
        if scale_type == "absolute_error":
            scale_expr = pl.col(target_col).sub(lagged).abs().alias("scale")
        else:
            scale_expr = pl.col(target_col).sub(lagged).pow(2).alias("scale")
        scale = train_df.select([id_col, scale_expr])
        scale = ufp.group_by(scale, id_col).mean()
        scale = scale.with_columns(_zero_to_nan(pl.col("scale")))

        def gen_expr(model):
            return pl.col(model).truediv(pl.col("scale")).fill_nan(0).alias(model)

        full_df = loss_df.join(scale, on=id_col, how="left")
        res = _pl_agg_expr(full_df, models, id_col, gen_expr)

    return res


@_base_docstring
def mae(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Mean Absolute Error (MAE)

    MAE measures the relative prediction
    accuracy of a forecasting method by calculating the
    deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series."""
    if isinstance(df, pd.DataFrame):
        res = (
            (df[models].sub(df[target_col], axis=0))
            .abs()
            .groupby(df[id_col], observed=True)
            .mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            return pl.col(target_col).sub(pl.col(model)).abs().alias(model)

        res = _pl_agg_expr(df, models, id_col, gen_expr)
    return res


@_base_docstring
def mse(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Mean Squared Error (MSE)

    MSE measures the relative prediction
    accuracy of a forecasting method by calculating the
    squared deviation of the prediction and the true
    value at a given time, and averages these devations
    over the length of the series."""
    if isinstance(df, pd.DataFrame):
        res = (
            (df[models].sub(df[target_col], axis=0))
            .pow(2)
            .groupby(df[id_col], observed=True)
            .mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            return pl.col(target_col).sub(pl.col(model)).pow(2).alias(model)

        res = _pl_agg_expr(df, models, id_col, gen_expr)
    return res


@_base_docstring
def rmse(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Root Mean Squared Error (RMSE)

    RMSE measures the relative prediction
    accuracy of a forecasting method by calculating the squared deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    Finally the RMSE will be in the same scale
    as the original time series so its comparison with other
    series is possible only if they share a common scale.
    RMSE has a direct connection to the L2 norm."""
    res = mse(df, models, id_col, target_col)
    if isinstance(res, pd.DataFrame):
        res[models] = res[models].pow(0.5)
    else:
        res = res.with_columns(*[pl.col(c).pow(0.5) for c in models])
    return res


@_base_docstring
def bias(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Forecast estimator bias.

    Defined as prediction - actual"""
    if isinstance(df, pd.DataFrame):
        res = (
            (df[models].sub(df[target_col], axis=0))
            .groupby(df[id_col], observed=True)
            .mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            return pl.col(model).sub(pl.col(target_col)).alias(model)

        res = _pl_agg_expr(df, models, id_col, gen_expr)
    return res


@_base_docstring
def cfe(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """
    Cumulative Forecast Error (CFE)

    Total signed forecast error per series. Positive values mean under forecast; negative mean over forecast.
    """
    if isinstance(df, pd.DataFrame):
        res = (
            df[models]
            .sub(df[target_col], axis=0)
            .groupby(df[id_col], observed=True)
            .sum()
        )
        res.index.name = id_col
        return res.reset_index()
    else:

        def gen_expr(model: str) -> pl.Expr:
            return pl.col(model).sub(pl.col(target_col)).alias(model)

        return _pl_agg_expr(df, models, id_col, gen_expr, agg="sum")


@_base_docstring
def pis(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """
    Compute the raw Absolute Periods In Stock (PIS) for one or multiple models.

    The PIS metric sums the absolute forecast errors per series without any scaling,
    yielding a scale-dependent measure of bias.
    """
    if isinstance(df, pd.DataFrame):
        res = (
            df[models]
            .sub(df[target_col], axis=0)
            .abs()
            .groupby(df[id_col], observed=True)
            .sum()
        )
        res.index.name = id_col
        return res.reset_index()
    else:
        return _pl_agg_expr(
            df,
            models,
            id_col,
            lambda m: pl.col(m).sub(pl.col(target_col)).abs().alias(m),
            agg="sum",
        )


@_base_docstring
def spis(
    df: DFType,
    train_df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """
    Compute the scaled Absolute Periods In Stock (sAPIS) for one or multiple models.

    The sPIS metric scales the sum of absolute forecast errors by the mean in-sample demand,
    yielding a scale-independent bias measure that can be aggregated across series.
    """
    if isinstance(df, pd.DataFrame):
        ins_means = train_df.groupby(id_col)[target_col].mean().rename("insample_mean")
        abs_err_sum = (
            (df[models].sub(df[target_col], axis=0))
            .abs()
            .groupby(df[id_col], observed=True)
            .sum()
        )
        res = abs_err_sum.div(ins_means, axis=0)
        res.index.name = id_col
        return res.reset_index()
    else:
        ins_means = train_df.group_by(id_col).agg(
            pl.col(target_col).mean().alias("insample_mean")
        )
        abs_err = _pl_agg_expr(
            df,
            models,
            id_col,
            lambda m: pl.col(m).sub(pl.col(target_col)).abs().alias(m),
            agg="sum",
        )
        res = (
            abs_err.join(ins_means, on=id_col, how="left")
            .with_columns(
                [(pl.col(m) / pl.col("insample_mean")).alias(m) for m in models]
            )
            .drop("insample_mean")
        )
        return res
    
@_base_docstring
def linex(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    a: float = 1.0,
) -> DFType:
    """
    Linex Loss

    The Linex (Linear Exponential) loss penalizes over- and under-forecasting
    asymmetrically depending on the parameter a.

    - If a > 0, under-forecasting (y > y_hat) is penalized more.
    - If a < 0, over-forecasting (y_hat > y) is penalized more.
    - a must not be 0.
    """
    if np.isclose(a, 0.0):
        raise ValueError("Parameter a in Linex loss must be non-zero.")

    if isinstance(df, pd.DataFrame):
        error = df[models].sub(df[target_col], axis=0)
        loss = (
            (np.exp(a * error) - a * error - 1)
            .groupby(df[id_col], observed=True)
            .mean()
        )
        loss.index.name = id_col
        return loss.reset_index()
    else:

        def gen_expr(model):
            err = pl.col(model).sub(pl.col(target_col))
            return (err.mul(a).exp().sub(err.mul(a)).sub(1)).alias(model)

        return _pl_agg_expr(df, models, id_col, gen_expr)


def _zero_to_nan(series: Union[pd.Series, "pl.Expr"]) -> Union[pd.Series, "pl.Expr"]:
    if isinstance(series, pd.Series):
        res = series.replace(0, np.nan)
    else:
        res = pl.when(series == 0).then(float("nan")).otherwise(series.abs())
    return res


@_base_docstring
def mape(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Mean Absolute Percentage Error (MAPE)

    MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the percentual deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    The closer to zero an observed value is, the higher penalty MAPE loss
    assigns to the corresponding error."""
    if isinstance(df, pd.DataFrame):
        res = (
            df[models]
            .sub(df[target_col], axis=0)
            .abs()
            .div(_zero_to_nan(df[target_col].abs()), axis=0)
            .groupby(df[id_col], observed=True)
            .mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            abs_err = pl.col(target_col).sub(pl.col(model)).abs()
            abs_target = _zero_to_nan(pl.col(target_col))
            ratio = abs_err.truediv(abs_target).alias(model)
            return ratio.fill_nan(None)

        res = _pl_agg_expr(df, models, id_col, gen_expr)
    return res


@_base_docstring
def smape(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Symmetric Mean Absolute Percentage Error (SMAPE)

    SMAPE measures the relative prediction
    accuracy of a forecasting method by calculating the relative deviation
    of the prediction and the observed value scaled by the sum of the
    absolute values for the prediction and observed value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 100% which is desirable compared to normal MAPE that
    may be undetermined when the target is zero."""
    if isinstance(df, pd.DataFrame):
        delta_y = df[models].sub(df[target_col], axis=0).abs()
        scale = df[models].abs().add(df[target_col].abs(), axis=0)
        raw = delta_y.div(scale).fillna(0)
        res = raw.groupby(df[id_col], observed=True).mean()
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            abs_err = pl.col(model).sub(pl.col(target_col)).abs()
            denominator = _zero_to_nan(
                pl.col(model).abs().add(pl.col(target_col)).abs()
            )
            ratio = abs_err.truediv(denominator).alias(model)
            return ratio.fill_nan(0)

        res = _pl_agg_expr(df, models, id_col, gen_expr)
    return res


def mase(
    df: DFType,
    models: List[str],
    seasonality: int,
    train_df: DFType,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Mean Absolute Scaled Error (MASE)

    MASE measures the relative prediction
    accuracy of a forecasting method by comparinng the mean absolute errors
    of the prediction and the observed value against the mean
    absolute errors of the seasonal naive model.
    The MASE partially composed the Overall Weighted Average (OWA),
    used in the M4 Competition.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        seasonality (int): Main frequency of the time series;
            Hourly 24, Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.
        train_df (pandas or polars DataFrame): Training dataframe with id and actual values. Must be sorted by time.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://robjhyndman.com/papers/mase.pdf
    """
    mean_abs_err = mae(df, models, id_col, target_col)
    return _scale_loss(
        loss_df=mean_abs_err,
        scale_type="absolute_error",
        models=models,
        seasonality=seasonality,
        train_df=train_df,
        id_col=id_col,
        target_col=target_col,
    )


def rmae(
    df: DFType,
    models: List[str],
    baseline: str,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Relative Mean Absolute Error (RMAE)

    Calculates the RAME between two sets of forecasts (from two different forecasting methods).
    A number smaller than one implies that the forecast in the
    numerator is better than the forecast in the denominator.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        baseline (str): Column that identifies the baseline model predictions.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.
    """
    numerator = mae(df, models, id_col, target_col)
    denominator = mae(df, [baseline], id_col, target_col)
    if ufp.is_nan(denominator[baseline]).any():
        raise ValueError(f"baseline model ({baseline}) contains NaNs.")
    denominator = ufp.rename(denominator, {baseline: f"{baseline}_denominator"})
    res = ufp.join(numerator, denominator, on=id_col)
    if isinstance(numerator, pd.DataFrame):
        for model in models:
            res[model] = (
                res[model].div(_zero_to_nan(res[f"{baseline}_denominator"])).fillna(0)
            )
        res = res[[id_col, *models]]
    else:

        def gen_expr(model, baseline) -> pl_Expr:
            denominator = _zero_to_nan(pl.col(f"{baseline}_denominator"))
            return pl.col(model).truediv(denominator).fill_nan(0).alias(model)

        exprs: List[pl_Expr] = [gen_expr(m, baseline) for m in models]
        res = res.select([id_col, *exprs])
    return res


def nd(
    df: DFType,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Normalized Deviation (ND)

    ND measures the relative prediction
    accuracy of a forecasting method by calculating the
    sum of the absolute deviation of the prediction and the true
    value at a given time and dividing it by the sum of the absolute
    value of the ground truth.

    Args:
        df: Input dataframe with id, times, actuals and predictions.
        models: Columns that identify the models predictions.
        id_col: Column that identifies each serie. Defaults to 'unique_id'.
        target_col: Column that contains the target. Defaults to 'y'.

    Returns:
        Dataframe with one row per id and one column per model.
    """
    if isinstance(df, pd.DataFrame):
        nom = (
            (df[models].sub(df[target_col], axis=0))
            .abs()
            .groupby(df[id_col], observed=True)
            .sum()
        )
        denom = df[target_col].abs().groupby(df[id_col], observed=True).sum()
        res = nom.div(denom, axis=0).fillna(0)
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr_nom(model):
            return pl.col(target_col).sub(pl.col(model)).abs().alias(model)

        nom = _pl_agg_expr(df, models, id_col, gen_expr_nom, "sum")

        def gen_expr_denom(target_col):
            return pl.col(target_col).abs().alias(target_col)

        denom = _pl_agg_expr(df, [target_col], id_col, gen_expr_denom, "sum")
        df = nom.join(denom, on=id_col, how="left")
        res = df.select(
            [
                id_col,
                *[
                    pl.col(model).truediv(pl.col(target_col)).fill_nan(0).alias(model)
                    for model in models
                ],
            ]
        )

    return res


def msse(
    df: DFType,
    models: List[str],
    seasonality: int,
    train_df: DFType,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Mean Squared Scaled Error (MSSE)

    MSSE measures the relative prediction
    accuracy of a forecasting method by comparinng the mean squared errors
    of the prediction and the observed value against the mean
    squared errors of the seasonal naive model.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        seasonality (int): Main frequency of the time series;
            Hourly 24, Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.
        train_df (pandas or polars DataFrame): Training dataframe with id and actual values. Must be sorted by time.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://otexts.com/fpp3/accuracy.html
    """
    mean_sq_err = mse(df=df, models=models, id_col=id_col, target_col=target_col)
    return _scale_loss(
        loss_df=mean_sq_err,
        scale_type="squared_error",
        models=models,
        seasonality=seasonality,
        train_df=train_df,
        id_col=id_col,
        target_col=target_col,
    )


def rmsse(
    df: DFType,
    models: List[str],
    seasonality: int,
    train_df: DFType,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    res = msse(
        df,
        models=models,
        seasonality=seasonality,
        train_df=train_df,
        id_col=id_col,
        target_col=target_col,
    )
    if isinstance(res, pd.DataFrame):
        res[models] = res[models].pow(0.5)
    else:
        res = res.with_columns(pl.col(models).pow(0.5))
    return res


rmsse.__doc__ = msse.__doc__.replace(  # type: ignore[union-attr]
    "Mean Squared Scaled Error (MSSE)", "Root Mean Squared Scaled Error (RMSSE)"
)


def quantile_loss(
    df: DFType,
    models: Dict[str, str],
    q: float = 0.5,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Quantile Loss (QL)

    QL measures the deviation of a quantile forecast.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.
    A common value for q is 0.5 for the deviation from the median.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (dict from str to str): Mapping from model name to the model predictions for the specified quantile.
        q (float, optional): Quantile for the predictions' comparison. Defaults to 0.5.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.
    """
    if isinstance(df, pd.DataFrame):
        res: Optional[pd.DataFrame] = None
        for model_name, pred_col in models.items():
            delta_y = df[target_col].sub(df[pred_col], axis=0)
            model_res = (
                np.maximum(q * delta_y, (q - 1) * delta_y)
                .groupby(df[id_col], observed=True)
                .mean()
                .rename(model_name)
                .reset_index()
            )
            if res is None:
                res = model_res
            else:
                res[model_name] = model_res[model_name]
    else:

        def gen_expr(model):
            model_name, pred_col = model
            delta_y = pl.col(target_col).sub(pl.col(pred_col))
            try:
                col_max = pl.max_horizontal([q * delta_y, (q - 1) * delta_y])
            except AttributeError:
                col_max = pl.max([q * delta_y, (q - 1) * delta_y])
            return col_max.alias(model_name)

        res = _pl_agg_expr(df, list(models.items()), id_col, gen_expr)
    return res


def scaled_quantile_loss(
    df: DFType,
    models: Dict[str, str],
    seasonality: int,
    train_df: DFType,
    q: float = 0.5,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Scaled Quantile Loss (SQL)

    SQL measures the deviation of a quantile forecast scaled by
    the mean absolute errors of the seasonal naive model.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.
    A common value for q is 0.5 for the deviation from the median.
    This was the official measure used in the M5 Uncertainty competition
    with seasonality = 1.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (dict from str to str): Mapping from model name to the model predictions for the specified quantile.
        seasonality (int): Main frequency of the time series;
            Hourly 24, Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.
        train_df (pandas or polars DataFrame): Training dataframe with id and actual values. Must be sorted by time.
        q (float, optional): Quantile for the predictions' comparison. Defaults to 0.5.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.sciencedirect.com/science/article/pii/S0169207021001722
    """
    q_loss = quantile_loss(
        df=df, models=models, q=q, id_col=id_col, target_col=target_col
    )
    return _scale_loss(
        loss_df=q_loss,
        scale_type="absolute_error",
        models=list(models.keys()),
        seasonality=seasonality,
        train_df=train_df,
        id_col=id_col,
        target_col=target_col,
    )


def mqloss(
    df: DFType,
    models: Dict[str, List[str]],
    quantiles: np.ndarray,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Multi-Quantile loss (MQL)

    MQL calculates the average multi-quantile Loss for
    a given set of quantiles, based on the absolute
    difference between predicted quantiles and observed values.

    The limit behavior of MQL allows to measure the accuracy
    of a full predictive distribution with
    the continuous ranked probability score (CRPS). This can be achieved
    through a numerical integration technique, that discretizes the quantiles
    and treats the CRPS integral with a left Riemann approximation, averaging over
    uniformly distanced quantiles.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (dict from str to list of str): Mapping from model name to the model predictions for each quantile.
        quantiles (numpy array): Quantiles to compare against.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.jstor.org/stable/2629907
    """
    res: Optional[DataFrame] = None
    error = np.empty((df.shape[0], quantiles.size))
    for model, predictions in models.items():
        for j, q_preds in enumerate(predictions):
            error[:, j] = (df[target_col] - df[q_preds]).to_numpy()
        loss = np.maximum(error * quantiles, error * (quantiles - 1)).mean(axis=1)
        model_res = type(df)({id_col: df[id_col], model: loss})
        model_res = ufp.group_by_agg(
            model_res, by=id_col, aggs={model: "mean"}, maintain_order=True
        )
        if res is None:
            res = model_res
        else:
            res = ufp.assign_columns(res, model, model_res[model])
    return res


def scaled_mqloss(
    df: DFType,
    models: Dict[str, List[str]],
    quantiles: np.ndarray,
    seasonality: int,
    train_df: DFType,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Scaled Multi-Quantile loss (SMQL)

    SMQL calculates the average multi-quantile Loss for
    a given set of quantiles, based on the absolute
    difference between predicted quantiles and observed values
    scaled by the mean absolute errors of the seasonal naive model.
    The limit behavior of MQL allows to measure the accuracy
    of a full predictive distribution with
    the continuous ranked probability score (CRPS). This can be achieved
    through a numerical integration technique, that discretizes the quantiles
    and treats the CRPS integral with a left Riemann approximation, averaging over
    uniformly distanced quantiles.
    This was the official measure used in the M5 Uncertainty competition
    with seasonality = 1.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (dict from str to list of str): Mapping from model name to the model predictions for each quantile.
        quantiles (numpy array): Quantiles to compare against.
        seasonality (int): Main frequency of the time series;
            Hourly 24, Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.
        train_df (pandas or polars DataFrame): Training dataframe with id and actual values. Must be sorted by time.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.sciencedirect.com/science/article/pii/S0169207021001722
    """
    mq_loss = mqloss(
        df=df, models=models, quantiles=quantiles, id_col=id_col, target_col=target_col
    )
    return _scale_loss(
        loss_df=mq_loss,
        scale_type="absolute_error",
        models=list(models.keys()),
        seasonality=seasonality,
        train_df=train_df,
        id_col=id_col,
        target_col=target_col,
    )


def coverage(
    df: DFType,
    models: List[str],
    level: int,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Coverage of y with y_hat_lo and y_hat_hi.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        level (int): Confidence level used for intervals.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.jstor.org/stable/2629907
    """
    if isinstance(df, pd.DataFrame):
        out = np.empty((df.shape[0], len(models)))
        for j, model in enumerate(models):
            out[:, j] = df[target_col].between(
                df[f"{model}-lo-{level}"], df[f"{model}-hi-{level}"]
            )
        res = (
            pd.DataFrame(out, columns=models, index=df.index)
            .groupby(df[id_col], observed=True)
            .mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            return (
                pl.col(target_col)
                .is_between(
                    pl.col(f"{model}-lo-{level}"), pl.col(f"{model}-hi-{level}")
                )
                .alias(model)
            )

        res = _pl_agg_expr(df, models, id_col, gen_expr)
    return res


def calibration(
    df: DFType,
    models: Dict[str, str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """
    Fraction of y that is lower than the model's predictions.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (dict from str to str): Mapping from model name to the model predictions.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://www.jstor.org/stable/2629907
    """
    if isinstance(df, pd.DataFrame):
        out = np.empty((df.shape[0], len(models)))
        for j, q_preds in enumerate(models.values()):
            out[:, j] = df[target_col].le(df[q_preds])
        res = (
            pd.DataFrame(out, columns=models.keys(), index=df.index)
            .groupby(df[id_col], observed=True)
            .mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            model_name, q_preds = model
            return pl.col(target_col).le(pl.col(q_preds)).alias(model_name)

        res = _pl_agg_expr(df, list(models.items()), id_col, gen_expr)
    return res


def scaled_crps(
    df: DFType,
    models: Dict[str, List[str]],
    quantiles: np.ndarray,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """Scaled Continues Ranked Probability Score

    Calculates a scaled variation of the CRPS, as proposed by Rangapuram (2021),
    to measure the accuracy of predicted quantiles `y_hat` compared to the observation `y`.
    This metric averages percentual weighted absolute deviations as
    defined by the quantile losses.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, times, actuals and predictions.
        models (dict from str to list of str): Mapping from model name to the model predictions for each quantile.
        quantiles (numpy array): Quantiles to compare against.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.

    References:
        [1] https://proceedings.mlr.press/v139/rangapuram21a.html
    """
    eps: np.float64 = np.finfo(np.float64).eps
    quantiles = np.asarray(quantiles)
    loss = mqloss(df, models, quantiles, id_col, target_col)
    sizes = ufp.counts_by_id(df, id_col)
    if isinstance(loss, pd.DataFrame):
        loss = loss.set_index(id_col)
        sizes = sizes.set_index(id_col)
        assert isinstance(df, pd.DataFrame)
        norm = df[target_col].abs().groupby(df[id_col], observed=True).sum()
        res = 2 * loss.mul(sizes["counts"], axis=0).div(norm + eps, axis=0)
        res.index.name = id_col
        res = res.reset_index()
    else:

        def gen_expr(model):
            return (
                2 * pl.col(model) * pl.col("counts") / (pl.col("norm") + eps)
            ).alias(model)

        grouped_df = ufp.group_by(df, id_col, maintain_order=True)
        norm = grouped_df.agg(pl.col(target_col).abs().sum().alias("norm"))
        res = _pl_agg_expr(
            loss.join(sizes, on=id_col).join(norm, on=id_col),
            list(models.keys()),
            id_col,
            gen_expr,
        )
    return res


def tweedie_deviance(
    df: DFType,
    models: List[str],
    power: float = 1.5,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """
    Compute the Tweedie deviance loss for one or multiple models, grouped by an identifier.

    Each group's deviance is calculated using the mean_tweedie_deviance function, which
    measures the deviation between actual and predicted values under the Tweedie distribution.

    The `power` parameter defines the specific compound distribution:
      - 1: Poisson
      - (1, 2): Compound Poisson-Gamma
      - 2: Gamma
      - >2: Inverse Gaussian

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        power (float, optional): Tweedie power parameter. Determines the compound distribution. Defaults to 1.5.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: DataFrame with one row per id and one column per model, containing the mean Tweedie deviance.

    References:
        [1] https://en.wikipedia.org/wiki/Tweedie_distribution

    """
    if power < 0:
        raise ValueError("Power must be non-negative.")
    elif power >= 2 and np.any(df[[target_col]] <= 0):
        raise ValueError(
            f"Power {power} requires all target values to be strictly positive."
        )

    if np.any(df[models] <= 0):
        raise ValueError(
            "All predictions must be strictly positive for Tweedie deviance."
        )

    if isinstance(df, pd.DataFrame):
        y_true = df[target_col]
        y_pred = df[models]
        delta_y = y_pred.sub(y_true, axis=0)
        if power == 0:
            dev = delta_y.pow(2)
        elif power == 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                dev = np.log(y_pred).sub(np.log(y_true), axis=0)
                dev = -2 * dev.mul(y_true, axis=0).sub(delta_y, axis=0)
            zero_mask = y_true == 0
            dev.loc[zero_mask] = 2 * y_pred[zero_mask]
        elif power == 2:
            with np.errstate(divide="ignore", invalid="ignore"):
                dev = np.log(y_pred).sub(np.log(y_true), axis=0)
                dev = 2 * (dev.add((1 / y_pred).mul(y_true, axis=0), axis=0) - 1)
        else:
            dev1 = y_true.clip(lower=0).pow(2 - power).div((1 - power) * (2 - power))
            dev2 = -y_pred.pow(1 - power).div(1 - power).mul(y_true, axis=0)
            dev3 = y_pred.pow(2 - power).div(2 - power)
            dev = 2 * dev2.add(dev1, axis=0).add(dev3, axis=0)
        # Group by id_col and calculate mean deviance for each model
        res = dev.groupby(df[id_col], observed=True).mean()
        res.index.name = id_col
        res = res.reset_index()

    else:
        if power == 0:

            def gen_expr(model):
                return (pl.col(model) - pl.col(target_col)).pow(2).alias(model)

        elif power == 1:

            def gen_expr(model):
                return (
                    pl.when(pl.col(target_col) == 0)
                    .then(2 * pl.col(model))
                    .otherwise(
                        2
                        * (
                            pl.col(target_col)
                            * (pl.col(target_col).log() - pl.col(model).log())
                            - (pl.col(target_col) - pl.col(model))
                        )
                    )
                    .alias(model)
                )

        elif power == 2:

            def gen_expr(model):
                return (
                    2
                    * (pl.col(model).log() - pl.col(target_col).log())
                    .add(pl.col(target_col).truediv(pl.col(model)))
                    .sub(1)
                ).alias(model)

        else:

            def gen_expr(model):
                return (
                    2
                    * (
                        pl.col(target_col)
                        .clip(0)
                        .pow(2 - power)
                        .truediv((1 - power) * (2 - power))
                    )
                    .sub(
                        pl.col(target_col)
                        .mul(pl.col(model).pow(1 - power))
                        .truediv(1 - power)
                    )
                    .add(pl.col(model).pow(2 - power).truediv(2 - power))
                ).alias(model)

        res = _pl_agg_expr(df, models, id_col, gen_expr)

    return res
