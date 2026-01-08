__all__ = ['mlforecast_objective']


import copy
from typing import Any, Callable, Dict, Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import optuna
import utilsforecast.processing as ufp
from sklearn.base import BaseEstimator, clone
from utilsforecast.compat import DataFrame

from . import MLForecast
from .compat import CatBoostRegressor
from .core import Freq

_TrialToConfig = Callable[[optuna.Trial], Dict[str, Any]]
CVSplit = Tuple[DataFrame, DataFrame, DataFrame]


def mlforecast_objective(
    df: DataFrame,
    config_fn: _TrialToConfig,
    loss: Callable,
    model: BaseEstimator,
    freq: Freq,
    n_windows: int,
    h: int,
    step_size: Optional[int] = None,
    input_size: Optional[int] = None,
    refit: Union[bool, int] = False,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    weight_col: Optional[str] = None,
    cv_splits: Optional[List[CVSplit]] = None
) -> Callable[[optuna.Trial], float]:
    """optuna objective function for the MLForecast class

    Args:
        df (DataFrame): Series data in long format.
        config_fn (callable): Function that takes an optuna trial and produces a configuration with the following keys:
            - model_params
            - mlf_init_params
            - mlf_fit_params
        loss (callable): Function that takes the validation and train dataframes and produces a float.
        model (BaseEstimator): scikit-learn compatible model to be trained
        freq (str or int): pandas' or polars' offset alias or integer denoting the frequency of the series.
        n_windows (int): Number of windows to evaluate.
        h (int): Forecast horizon.
        step_size (int, optional): Step size between each cross validation window. If None it will be equal to `h`.
            Defaults to None.
        input_size (int, optional): Maximum training samples per serie in each window. If None, will use an expanding window.
            Defaults to None.
        refit (bool or int): Retrain model for each cross validation window.
            If False, the models are trained at the beginning and then used to predict each window.
            If positive int, the models are retrained every `refit` windows. Defaults to False.
        id_col (str): Column that identifies each serie. Defaults to 'unique_id'.
        time_col (str): Column that identifies each timestep, its values can be timestamps or integers. Defaults to 'ds'.
        target_col (str): Column that contains the target. Defaults to 'y'.
        weight_col (str): Column that contains sample weights. Defaults to None.
        cv_splits (List[Tuple[DataFrame, DataFrame, DataFrame]] | None): Optional cached CV splits (cutoffs, train, valid) to 
            reuse across trials. If None, backtest splits are generated on each trial.

    Returns:
        (Callable[[optuna.Trial], float]): optuna objective function
    """
    def objective(trial: optuna.Trial) -> float:
        config = config_fn(trial)
        trial.set_user_attr("config", copy.deepcopy(config))
        
        model_copy = clone(model)
        model_params = config["model_params"]
        if config["mlf_fit_params"].get("static_features", []) and isinstance(model, CatBoostRegressor):
            model_params["cat_features"] = config["mlf_fit_params"]["static_features"]
        model_copy.set_params(**model_params)
        mlf = MLForecast(
            models={"model": model_copy}, 
            freq=freq,
            **config["mlf_init_params"],
        )
        splits = cv_splits
        if splits is None:
            splits = ufp.backtest_splits(
                df,
                n_windows=n_windows,
                h=h,
                id_col=id_col,
                time_col=time_col,
                freq=freq,
                step_size=step_size,
                input_size=input_size,
            )
        elif not isinstance(splits, list):
            splits = list(splits)
        metrics = []
        for i, (_, train, valid) in enumerate(splits):
            should_fit = i == 0 or (refit > 0 and i % refit == 0)
            if should_fit:
                mlf.fit(
                    train,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    weight_col=weight_col,
                    **config["mlf_fit_params"],
                )
            static = [c for c in mlf.ts.static_features_.columns if c != id_col]
            if weight_col:
                dynamic = [
                    c
                    for c in valid.columns
                    if c not in static + [id_col, time_col, target_col, weight_col]
                ]
            else:
                dynamic = [
                    c
                    for c in valid.columns
                    if c not in static + [id_col, time_col, target_col]
                ] 
            if dynamic:
                X_df: Optional[DataFrame] = ufp.drop_columns(
                    valid, static + [target_col]
                )
            else:
                X_df = None
            if weight_col:
                if isinstance(train, pd.DataFrame):
                    new_df = None if should_fit else train.drop(columns=[weight_col], errors="ignore")
                else:
                    if should_fit:
                        new_df = None
                    else:
                        if weight_col in train.columns:
                            new_df = train.drop(weight_col)
                        else:
                            new_df = train
            else:
                new_df = None if should_fit else train
            preds = mlf.predict(
                h=h,
                X_df=X_df,
                new_df=new_df,
            )
            result = ufp.join(
                valid[[id_col, time_col, target_col]],
                preds,
                on=[id_col, time_col],
            )
            if result.shape[0] < valid.shape[0]:
                raise ValueError(
                    "Cross validation result produced less results than expected. "
                    "Please verify that the passed frequency (freq) matches your series' "
                    "and that there aren't any missing periods."
                )
            if weight_col:
                metric = loss(result, train_df=train, weight_col=weight_col)
            else: 
                metric = loss(result, train_df=train)
            metrics.append(metric)
            trial.report(metric, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(metrics).item()

    return objective
