"""Model performance evaluation"""

__all__ = ['evaluate']


import inspect
import re
import reprlib
from typing import Callable, Dict, List, Optional, get_origin

import numpy as np
import pandas as pd

import utilsforecast.processing as ufp

from .compat import AnyDFType, DFType, DistributedDFType, pl, pl_DataFrame
from .losses import _get_group_cols


def _function_name(f: Callable):
    if hasattr(f, "func"):
        # partial fn
        name = f.func.__name__
    else:
        name = f.__name__
    return name


def _quantiles_from_levels(level: List[int]) -> np.ndarray:
    """Returns quantiles associated to `level` and the sorte columns of `model_name`"""
    level = sorted(level)
    alphas = [100 - lv for lv in level]
    quantiles = [alpha / 200 for alpha in reversed(alphas)]
    quantiles.extend([1 - alpha / 200 for alpha in alphas])
    return np.array(quantiles)


def _models_from_levels(model_name: str, level: List[int]) -> List[str]:
    level = sorted(level)
    cols = [f"{model_name}-lo-{lv}" for lv in reversed(level)]
    cols.extend([f"{model_name}-hi-{lv}" for lv in level])
    return cols


def _get_model_cols(
    cols: List[str],
    id_col: str,
    time_col: str,
    target_col: str,
    cutoff_col: str,
) -> List[str]:
    return [
        c
        for c in cols
        if c not in [id_col, time_col, target_col, cutoff_col]
        and not re.search(r"-(?:lo|hi)-\d+", c)
    ]


def _evaluate_wrapper(
    df: pd.DataFrame,
    metrics: List[Callable],
    models: Optional[List[str]],
    level: Optional[List[int]],
    id_col: str,
    time_col: str,
    target_col: str,
    cutoff_col: str,
    agg_fn: Optional[str],
) -> pd.DataFrame:
    group_cols = _get_group_cols(df, id_col, cutoff_col)
    if "_in_sample" in df:
        in_sample_mask = df["_in_sample"]
        train_df = df.loc[in_sample_mask, [*group_cols, time_col, target_col]]
        df = df.loc[~in_sample_mask].drop(columns="_in_sample")
    else:
        train_df = None
    return evaluate(
        df=df,
        metrics=metrics,
        models=models,
        train_df=train_df,
        level=level,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
        agg_fn=agg_fn,
    )


def _distributed_evaluate(
    df: DistributedDFType,
    metrics: List[Callable],
    models: Optional[List[str]],
    train_df: Optional[DFType],
    level: Optional[List[int]],
    id_col: str,
    time_col: str,
    target_col: str,
    cutoff_col: str,
    agg_fn: Optional[str],
) -> DistributedDFType:
    import fugue.api as fa

    if agg_fn is not None:
        raise ValueError("`agg_fn` is not supported in distributed")
    df_cols = fa.get_column_names(df)
    group_cols: list[str] = _get_group_cols(df, id_col, cutoff_col)
    if train_df is not None:
        # align columns in order to vstack them
        def assign_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
            return df.assign(**cols)
        train_cols = [*group_cols, time_col, target_col]
        extra_cols = [c for c in df_cols if c not in train_cols]
        train_df = fa.select_columns(train_df, train_cols)
        train_df = fa.transform(
            train_df,
            using=assign_cols,
            schema=(
                "*," + str(fa.get_schema(df).extract(extra_cols)) + ",_in_sample:bool"
            ),
            params={
                "cols": {
                    **{c: float("nan") for c in extra_cols},
                    "_in_sample": True,
                },
            },
        )
        df = fa.transform(
            df,
            using=assign_cols,
            schema="*,_in_sample:bool",
            params={"cols": {"_in_sample": False}},
        )
        df = fa.union(train_df, df)

    if models is None:
        model_cols = _get_model_cols(df_cols, id_col, time_col, target_col, cutoff_col)
    else:
        model_cols = models
    models_schema = ",".join(f"{m}:double" for m in model_cols)
    result_schema = fa.get_schema(df).extract(*group_cols) + "metric:str" + models_schema
    return fa.transform(
        df,
        using=_evaluate_wrapper,
        schema=result_schema,
        params=dict(
            metrics=metrics,
            models=models,
            level=level,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            cutoff_col=cutoff_col,
            agg_fn=agg_fn,
        ),
        partition={"by": group_cols, "algo": "coarse"},
    )


def evaluate(
    df: AnyDFType,
    metrics: List[Callable],
    models: Optional[List[str]] = None,
    train_df: Optional[AnyDFType] = None,
    level: Optional[List[int]] = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    agg_fn: Optional[str] = None,
) -> AnyDFType:
    """Evaluate forecast using different metrics.

    Args:
        df (pandas, polars, dask or spark DataFrame): Forecasts to evaluate.
            Must have `id_col`, `time_col`, `target_col` and models' predictions.
        metrics (list of callable): Functions with arguments `df`, `models`,
            `id_col`, `target_col` and optionally `train_df`.
        models (list of str, optional): Names of the models to evaluate.
            If `None` will use every column in the dataframe after removing
            id, time and target. Defaults to None.
        train_df (pandas, polars, dask or spark DataFrame, optional): Training set.
            Used to evaluate metrics such as `mase`. Defaults to None.
        level (list of int, optional): Prediction interval levels. Used to compute
            losses that rely on quantiles. Defaults to None.
        id_col (str, optional): Column that identifies each serie.
            Defaults to 'unique_id'.
        time_col (str, optional): Column that identifies each timestep, its values
            can be timestamps or integers. Defaults to 'ds'.
        target_col (str, optional): Column that contains the target.
            Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for
            each forecast cross-validation fold. Defaults to 'cutoff'.
        agg_fn (str, optional): Statistic to compute on the scores by id to reduce
            them to a single number. Defaults to None.

    Returns:
        pandas, polars, dask or spark DataFrame: Metrics with one row per
            (id, metric) combination and one column per model. If `agg_fn` is
            not `None`, there is only one row per metric.
    """
    if not isinstance(df, (pd.DataFrame, pl_DataFrame)):
        return _distributed_evaluate(
            df=df,
            metrics=metrics,
            models=models,
            train_df=train_df,
            level=level,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            cutoff_col=cutoff_col,
            agg_fn=agg_fn,
        )
    if models is None:
        model_cols = _get_model_cols(df.columns, id_col, time_col, target_col, cutoff_col)
    else:
        model_cols = models

    # interval cols
    if level is not None:
        expected_cols = {
            f"{m}-{side}-{lvl}"
            for m in model_cols
            for side in ("lo", "hi")
            for lvl in level
        }
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"The following columns are required for level={level} "
                f"and are missing: {missing}"
            )
    else:
        requires_level = [
            m
            for m in metrics
            if get_origin(inspect.signature(m).parameters["models"].annotation) is dict
        ]
        if requires_level:
            raise ValueError(
                f"The following metrics require setting `level`: {requires_level}"
            )

    # y_train
    metric_requires_y_train = {
        _function_name(m): "train_df" in inspect.signature(m).parameters
        for m in metrics
    }
    y_train_metrics = [
        m for m, requires_yt in metric_requires_y_train.items() if requires_yt
    ]
    if y_train_metrics:
        if train_df is None:
            raise ValueError(
                f"The following metrics require y_train: {y_train_metrics}. "
                "Please provide `train_df`."
            )
        train_df = ufp.sort(train_df, by=[id_col, time_col])
        missing_series = set(df[id_col].unique()) - set(train_df[id_col].unique())
        if missing_series:
            raise ValueError(
                f"The following series are missing from the train_df: {reprlib.repr(missing_series)}"
            )

    results_per_metric = []
    for metric in metrics:
        metric_name = _function_name(metric)
        kwargs = dict(df=df, models=model_cols, id_col=id_col, target_col=target_col)
        if metric_requires_y_train[metric_name]:
            kwargs["train_df"] = train_df
            kwargs["cutoff_col"] = cutoff_col
            kwargs["time_col"] = time_col
        metric_params = inspect.signature(metric).parameters
        if "baseline" in metric_params:
            metric_name = f"{metric_name}_{metric_params['baseline'].default}"
        if "q" in metric_params or metric_params["models"].annotation is Dict[str, str]:
            assert level is not None  # we've already made sure of this above
            for lvl in level:
                quantiles = _quantiles_from_levels([lvl])
                for q, side in zip(quantiles, ["lo", "hi"]):
                    kwargs["models"] = {
                        model: f"{model}-{side}-{lvl}" for model in model_cols
                    }
                    if "q" in metric_params:
                        # this is for calibration, since it uses the predictions for q
                        # but doesn't use it
                        kwargs["q"] = q
                    result = metric(**kwargs)
                    result = ufp.assign_columns(result, "metric", f"{metric_name}_q{q}")
                    results_per_metric.append(result)
        elif "quantiles" in metric_params:
            assert level is not None  # we've already made sure of this above
            quantiles = _quantiles_from_levels(level)
            kwargs["quantiles"] = quantiles
            kwargs["models"] = {
                model: _models_from_levels(model, level) for model in model_cols
            }
            result = metric(**kwargs)
            result = ufp.assign_columns(result, "metric", metric_name)
            results_per_metric.append(result)
        elif "level" in metric_params:
            assert level is not None  # we've already made sure of this above
            for lvl in level:
                kwargs["level"] = lvl
                result = metric(**kwargs)
                result = ufp.assign_columns(
                    result, "metric", f"{metric_name}_level{lvl}"
                )
                results_per_metric.append(result)
        else:
            result = metric(**kwargs)
            result = ufp.assign_columns(result, "metric", metric_name)
            results_per_metric.append(result)
    if isinstance(df, pd.DataFrame):
        df = pd.concat(results_per_metric).reset_index(drop=True)
    else:
        df = pl.concat(results_per_metric, how="diagonal")
    
    if cutoff_col in df.columns:
        id_cols = [id_col, cutoff_col, "metric"]
    else:
        id_cols = [id_col, "metric"]

    model_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + model_cols]
    if agg_fn is not None:
        group_cols = id_cols[1:] # exclude id_col
        df = ufp.group_by_agg(
            df,
            by=group_cols,
            aggs={m: agg_fn for m in model_cols},
            maintain_order=True,
        )
    return df
