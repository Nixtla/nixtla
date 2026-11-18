from functools import partial
from typing import List

import pandas as pd
import pytest
from utilsforecast.data import generate_series
from utilsforecast.evaluation import evaluate
from utilsforecast.processing import (
    backtest_splits,
    drop_index_if_pandas,
    join,
    maybe_compute_sort_indices,
    take_rows,
    vertical_concat,
)

from xiuhmolpilli.models.utils.forecaster import maybe_convert_col_to_datetime
from xiuhmolpilli.utils.experiment_handler import (
    generate_train_cv_splits,
    mase,
    ExperimentDataset,
)
from .utils import models


def generate_train_cv_splits_from_backtest_splits(
    df: pd.DataFrame,
    n_windows: int,
    h: int,
    freq: str,
    step_size: int = 1,
):
    df = maybe_convert_col_to_datetime(df, "ds")
    # mlforecast cv code
    results = []
    sort_idxs = maybe_compute_sort_indices(df, "unique_id", "ds")
    if sort_idxs is not None:
        df = take_rows(df, sort_idxs)
    splits = backtest_splits(
        df,
        n_windows=n_windows,
        h=h,
        id_col="unique_id",
        time_col="ds",
        freq=pd.tseries.frequencies.to_offset(freq),
        step_size=h if step_size is None else step_size,
    )
    for _, (cutoffs, train, _) in enumerate(splits):
        train_cv = join(train, cutoffs, on="unique_id")
        results.append(train_cv)
    out = vertical_concat(results)
    out = drop_index_if_pandas(out)
    return out


def generate_exp_dataset(
    n_series,
    freq,
    return_df: bool = False,
) -> ExperimentDataset | pd.DataFrame:
    df = generate_series(n_series, freq=freq, min_length=12)
    df["unique_id"] = df["unique_id"].astype(str)
    df["frequency"] = "frequency"
    df["pandas_frequency"] = freq
    df["seasonality"] = 7
    df["horizon"] = 2
    if return_df:
        return df
    return ExperimentDataset.from_df(df)


def evaluate_cv_from_scratch(
    df: pd.DataFrame,
    fcst_df: pd.DataFrame,
    models: List[str],
    seasonality: int,
) -> pd.DataFrame:
    partial_mase = partial(mase, seasonality=seasonality)
    uids = df["unique_id"].unique()
    results = []
    for uid in uids:
        df_ = df.query("unique_id == @uid")
        fcst_df_ = fcst_df.query("unique_id == @uid")
        cutoffs = fcst_df_["cutoff"].unique()
        for cutoff in cutoffs:
            df__ = df_.query("ds <= @cutoff")
            fcst_df__ = fcst_df_.query("cutoff == @cutoff")
            eval_df = evaluate(
                df=fcst_df__,
                train_df=df__,
                metrics=[partial_mase],
                models=models,
            )
            eval_df["cutoff"] = cutoff
            results.append(eval_df)
    out = pd.concat(results)
    out = out[["unique_id", "cutoff", "metric"] + models]
    return out


def sort_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df.sort_values(cols).reset_index(drop=True)


@pytest.mark.parametrize(
    "freq,n_windows,h,step_size",
    [
        ("H", 3, 2, 1),
        ("H", 1, 12, None),
        ("MS", 3, 2, 2),
    ],
)
def test_generate_train_cv_splits(freq, n_windows, h, step_size):
    df = generate_series(n_series=5, freq=freq)
    df["unique_id"] = df["unique_id"].astype(int)
    df_cv = generate_train_cv_splits_from_backtest_splits(
        df=df,
        n_windows=n_windows,
        step_size=step_size,
        h=h,
        freq=freq,
    )
    cutoffs = df_cv[["unique_id", "cutoff"]].drop_duplicates()
    train_cv_splits = generate_train_cv_splits(
        df=df,
        cutoffs=cutoffs,
    )
    p_sort_df = partial(sort_df, cols=["unique_id", "cutoff", "ds"])
    pd.testing.assert_frame_equal(
        p_sort_df(df_cv),
        p_sort_df(train_cv_splits),
    )


@pytest.mark.parametrize("model", models)
def test_eval(model):
    freq = "H"
    exp_dataset = generate_exp_dataset(n_series=5, freq=freq)
    fcst_df = model.cross_validation(
        exp_dataset.df,
        h=exp_dataset.horizon,
        freq=exp_dataset.pandas_frequency,
    )
    eval_df = exp_dataset.evaluate_forecast_df(
        forecast_df=fcst_df,
        models=[model.alias],
    )
    eval_df_from_scratch = evaluate_cv_from_scratch(
        df=exp_dataset.df,
        fcst_df=fcst_df,
        models=[model.alias],
        seasonality=exp_dataset.seasonality,
    )
    p_sort_df = partial(sort_df, cols=["unique_id", "cutoff", "metric"])
    pd.testing.assert_frame_equal(
        p_sort_df(eval_df),
        p_sort_df(eval_df_from_scratch),
    )
