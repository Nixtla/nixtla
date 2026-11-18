__all__ = ['evaluate']


from collections.abc import Callable

import narwhals.stable.v2 as nw
import numpy as np
import utilsforecast.evaluation as ufe
from narwhals.typing import FrameT


def evaluate(
    df: FrameT,
    metrics: list[Callable],
    tags: dict[str, np.ndarray],
    models: list[str] | None = None,
    train_df: FrameT | None = None,
    level: list[int] | None = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    agg_fn: str | None = "mean",
    benchmark: str | None = None,
) -> FrameT:
    """Evaluate hierarchical forecast using different metrics.

    Args:
        df (pandas, polars, dask or spark DataFrame): Forecasts to evaluate. Must have `id_col`, `time_col`, `target_col` and models' predictions.
        metrics (list of callable): Functions with arguments `df`, `models`, `id_col`, `target_col` and optionally `train_df`.
        tags (dict): Each key is a level in the hierarchy and its value contains tags associated to that level. Each key is a level in the hierarchy and its value contains tags associated to that level.
        models (list of str, optional): Names of the models to evaluate. If `None` will use every column in the dataframe after removing id, time and target.
        train_df (pandas, polars, dask or spark DataFrame, optional): Training set. Used to evaluate metrics such as `mase`.
        level (list of int, optional): Prediction interval levels. Used to compute losses that rely on quantiles.
        id_col (str): Column that identifies each serie.
        time_col (str): Column that identifies each timestep, its values can be timestamps or integers.
        target_col (str): Column that contains the target.
        agg_fn (str, optional): Statistic to compute on the scores by id to reduce them to a single number.
        benchmark (str, optional): If passed, evaluators are scaled by the error of this benchmark model.

    Returns:
        pandas, polars DataFrame: Metrics with one row per (id, metric) combination and one column per model. If `agg_fn` is not `None`, there is only one row per metric.
    """
    # Check benchmark in columns
    if benchmark is not None:
        if benchmark not in df.columns:
            raise ValueError(f"Benchmark model '{benchmark}' not found in df")
        model_cols = None

    df_nw = nw.from_native(df)
    if train_df is not None:
        train_nw = nw.from_native(train_df)
    tag_scores = []
    tags_ = {**tags, "Overall": np.concatenate(list(tags.values()))}

    eps = np.finfo(np.float32).eps

    for tag, tag_ids in tags_.items():
        df_tag = df_nw.filter(nw.col(id_col).is_in(tag_ids)).to_native()
        if train_df is not None:
            train_nw_tag = train_nw.filter(nw.col(id_col).is_in(tag_ids)).to_native()
        else:
            train_nw_tag = None

        df_score = ufe.evaluate(
            df=df_tag,
            metrics=metrics,
            models=models,
            train_df=train_nw_tag,
            level=level,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            agg_fn=agg_fn,
        )
        df_score_nw = nw.from_native(df_score)
        if benchmark is not None:
            if model_cols is None:
                model_cols = [
                    c for c in df_score_nw.columns if c not in [id_col, "metric"]
                ]
            # NB: the below assumes benchmark loss is always positive, which is a reasonable assumption
            df_score_nw = df_score_nw.with_columns(
                nw.col(model_cols) / nw.col(benchmark).clip(eps)
            )
            df_score_nw = df_score_nw.with_columns(nw.col("metric") + "-scaled")

        df_score_nw = df_score_nw.select(nw.lit(tag).alias("level"), nw.all())
        tag_scores.append(df_score_nw)

    tag_scores_nw = nw.concat(tag_scores)
    tag_scores_nw = nw.maybe_reset_index(tag_scores_nw)

    df_eval = tag_scores_nw.to_native()

    return df_eval
