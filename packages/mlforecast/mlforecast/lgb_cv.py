__all__ = ["LightGBMCV"]


import copy
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from utilsforecast.processing import backtest_splits

from mlforecast.core import (
    DateFeature,
    Freq,
    Lags,
    LagTransforms,
    TargetTransform,
    TimeSeries,
)
from mlforecast.utils import _resolve_num_threads


def _mape(y_true, y_pred, ids, _dates, weight_series=None):
    abs_pct_err = (y_true - y_pred).abs() / y_true.abs()  # safer
    grouped = abs_pct_err.groupby(ids, observed=True)
    
    if weight_series is None:
        return grouped.mean().mean()

    w_by_id = weight_series.groupby(ids, observed=True).sum()
    mape_by_id = grouped.mean()

    weight_sum = w_by_id.sum()
    if weight_sum <= 0:
        warnings.warn("Total weight is zero, cannot compute weighted MAPE.", UserWarning)
        return float("nan")

    return (mape_by_id * w_by_id).sum() / weight_sum


def _rmse(y_true, y_pred, ids, _dates, weight_series=None):
    sq_err = (y_true - y_pred) ** 2
    mse_by_id = sq_err.groupby(ids, observed=True).mean()

    if weight_series is None:
        return np.sqrt(mse_by_id.mean())

    w_by_id = weight_series.groupby(ids, observed=True).sum()
    weight_sum = w_by_id.sum()

    if weight_sum <= 0:
        warnings.warn(
            "Warning: Total weight is zero, cannot compute weighted RMSE.",
            UserWarning,
        )
        return float("nan")

    weighted_mse = (mse_by_id * w_by_id).sum() / weight_sum
    return np.sqrt(weighted_mse)


_metric2fn = {"mape": _mape, "rmse": _rmse}


def _update(bst, n):
    for _ in range(n):
        bst.update()


def _predict(ts, bst, valid, h, before_predict_callback, after_predict_callback, weight_col):
    static = ts.static_features_.columns.drop(ts.id_col).tolist()
    dynamic = valid.columns.drop(static + [ts.id_col, ts.time_col, ts.target_col, weight_col], 
                errors="ignore") # drops weight_col if present amongst other cols
    if not dynamic.empty:
        X_df = valid.drop(columns=static + [ts.target_col, weight_col],
                errors="ignore") # drops weight_col if present amongst other cols)
    else:
        X_df = None
    preds = ts.predict(
        {"Booster": bst},
        horizon=h,
        before_predict_callback=before_predict_callback,
        after_predict_callback=after_predict_callback,
        X_df=X_df
    )
    return valid.merge(preds, on=[ts.id_col, ts.time_col], how="left")


def _update_and_predict(
    ts, bst, valid, n, h, before_predict_callback, after_predict_callback, weight_col
):
    _update(bst, n)
    return _predict(ts, bst, valid, h, before_predict_callback, after_predict_callback, weight_col)


CVResult = Tuple[int, float]


class LightGBMCV:
    def __init__(
        self,
        freq: Freq,
        lags: Optional[Lags] = None,
        lag_transforms: Optional[LagTransforms] = None,
        date_features: Optional[Iterable[DateFeature]] = None,
        num_threads: int = 1,
        target_transforms: Optional[List[TargetTransform]] = None,
    ):
        """Create LightGBM CV object.

        Args:
            freq (str or int): Pandas offset alias, e.g. 'D', 'W-THU' or integer denoting the frequency of the series.
            lags (list of int, optional): Lags of the target to use as features. Defaults to None.
            lag_transforms (dict of int to list of functions, optional): Mapping of target lags to their transformations. Defaults to None.
            date_features (list of str or callable, optional): Features computed from the dates. Can be pandas date attributes or functions that will take the dates as input.
                Defaults to None.
            num_threads (int): Number of threads to use when computing the features. Use -1 to use all available CPU cores. Defaults to 1.
            target_transforms (list of transformers, optional): Transformations that will be applied to the target before computing the features and restored after the forecasting step.
                Defaults to None.
        """
        num_threads = _resolve_num_threads(num_threads)
        self.num_threads = num_threads
        cpu_count = os.cpu_count()
        if cpu_count is None:
            num_cpus = 1
        else:
            num_cpus = cpu_count
        self.bst_threads = max(num_cpus // num_threads, 1)
        self.ts = TimeSeries(
            freq=freq,
            lags=lags,
            lag_transforms=lag_transforms,
            date_features=date_features,
            num_threads=self.bst_threads,
            target_transforms=target_transforms,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"freq={self.ts.freq}, "
            f"lag_features={list(self.ts.transforms.keys())}, "
            f"date_features={self.ts.date_features}, "
            f"num_threads={self.num_threads}, "
            f"bst_threads={self.bst_threads})"
        )

    def setup(
        self,
        df: pd.DataFrame,
        n_windows: int,
        h: int,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        step_size: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        weights: Optional[Sequence[float]] = None,
        metric: Union[str, Callable] = "mape",
        input_size: Optional[int] = None,
        weight_col: Optional[str] = None,
    ):
        """Initialize internal data structures to iteratively train the boosters. Use this before calling partial_fit.

        Args:
            df (pandas DataFrame): Series data in long format.
            n_windows (int): Number of windows to evaluate.
            h (int): Forecast horizon.
            id_col (str): Column that identifies each serie. Defaults to 'unique_id'.
            time_col (str): Column that identifies each timestep, its values can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            step_size (int, optional): Step size between each cross validation window. If None it will be equal to `h`.
                Defaults to None.
            params (dict, optional): Parameters to be passed to the LightGBM Boosters. Defaults to None.
            static_features (list of str, optional): Names of the features that are static and will be repeated when forecasting.
                Defaults to None.
            dropna (bool): Drop rows with missing values produced by the transformations. Defaults to True.
            keep_last_n (int, optional): Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.
                Defaults to None.
            weights (sequence of float, optional): Weights to multiply the metric of each window. If None, all windows have the same weight.
                Defaults to None.
            metric (str or callable): Metric used to assess the performance of the models and perform early stopping. Defaults to 'mape'.
            input_size (int, optional): Maximum training samples per serie in each window. If None, will use an expanding window.
                Defaults to None.
            weight_col (str, optional):Column containing sample weights. Higher weights increase the influence of those samples during fitting and evaluation.
                
        Returns:
            (LightGBMCV): CV object with internal data structures for partial_fit.
        """
        if weights is None:
            self.weights = np.full(n_windows, 1 / n_windows)
        elif len(weights) != n_windows:
            raise ValueError("Must specify as many weights as the number of windows")
        else:
            self.weights = np.asarray(weights)
        if callable(metric):
            self.metric_fn = metric
            self.metric_name = metric.__name__
        else:
            if metric not in _metric2fn:
                raise ValueError(
                    f"{metric} is not one of the implemented metrics: ({', '.join(_metric2fn.keys())})"
                )
            self.metric_fn = _metric2fn[metric]
            self.metric_name = metric
        self.items = []
        self.h = h
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.params = {} if params is None else params
        splits = backtest_splits(
            df,
            n_windows=n_windows,
            h=h,
            id_col=id_col,
            time_col=time_col,
            freq=self.ts.freq,
            step_size=step_size,
            input_size=input_size,
        )
        
        for _, train, valid in splits:
            ts = copy.deepcopy(self.ts)
            prep = ts.fit_transform(
                data=train,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                static_features=static_features,
                dropna=dropna,
                keep_last_n=keep_last_n,
                weight_col=weight_col,
            )
            assert isinstance(prep, pd.DataFrame)

            if weight_col is not None:
                current_weights = prep[weight_col].values 
            else:
                current_weights = None           
            ds = lgb.Dataset(prep.drop(columns=[id_col, time_col, target_col, weight_col], errors="ignore"), 
            prep[target_col], weight=current_weights).construct()
                
            bst = lgb.Booster({**self.params, "num_threads": self.bst_threads}, ds)
            bst.predict = partial(bst.predict, num_threads=self.bst_threads)
            self.items.append((ts, bst, valid))
        return self

    def _single_threaded_partial_fit(
        self,
        metric_values,
        num_iterations,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        weight_col: Optional[str] = None
    ):
        for j, (ts, bst, valid) in enumerate(self.items):
            preds = _update_and_predict(
                ts=ts,
                bst=bst,
                valid=valid,
                n=num_iterations,
                h=self.h,
                before_predict_callback=before_predict_callback,
                after_predict_callback=after_predict_callback,
                weight_col=weight_col
            )
            if weight_col is not None:
                metric_values[j] = self.metric_fn(
                    preds[self.target_col],
                    preds["Booster"],
                    preds[self.id_col],
                    preds[self.time_col],
                    weight_series=preds[weight_col])
            else:
                metric_values[j] = self.metric_fn(
                    preds[self.target_col],
                    preds["Booster"],
                    preds[self.id_col],
                    preds[self.time_col])

    def _multithreaded_partial_fit(
        self,
        metric_values,
        num_iterations,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        weight_col: Optional[str] = None
    ):
        with ThreadPoolExecutor(self.num_threads) as executor:
            futures = []
            for ts, bst, valid in self.items:
                _update(bst, num_iterations)
                future = executor.submit(
                    _predict,
                    ts=ts,
                    bst=bst,
                    valid=valid,
                    h=self.h,
                    before_predict_callback=before_predict_callback,
                    after_predict_callback=after_predict_callback,
                    weight_col=weight_col
                )
                futures.append(future)
            cv_preds = [f.result() for f in futures]
        if weight_col:
            metric_values[:] = [
                self.metric_fn(
                    preds[self.target_col],
                    preds["Booster"],
                    preds[self.id_col],
                    preds[self.time_col],
                    weight_series=preds[weight_col].values
                )
                for preds in cv_preds
            ]
        else:
            metric_values[:] = [
                self.metric_fn(
                    preds[self.target_col],
                    preds["Booster"],
                    preds[self.id_col],
                    preds[self.time_col]
                )
                for preds in cv_preds
            ]
            

    def partial_fit(
        self,
        num_iterations: int,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        weight_col: Optional[str] = None,
    ) -> float:
        """Train the boosters for some iterations.

        Args:
            num_iterations (int): Number of boosting iterations to run
            before_predict_callback (callable, optional): Function to call on the features before computing the predictions.
                This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.
                The series identifier is on the index. Defaults to None.
            after_predict_callback (callable, optional): Function to call on the predictions before updating the targets.
                This function will take a pandas Series with the predictions and should return another one with the same structure.
                The series identifier is on the index. Defaults to None.

        Returns:
            (float): Weighted metric after training for num_iterations.
        """
        metric_values = np.empty(len(self.items))
        if self.num_threads == 1:
            self._single_threaded_partial_fit(
                metric_values,
                num_iterations,
                before_predict_callback,
                after_predict_callback,
                weight_col
            )
        else:
            self._multithreaded_partial_fit(
                metric_values,
                num_iterations,
                before_predict_callback,
                after_predict_callback,
                weight_col
            )
        return metric_values @ self.weights

    def should_stop(self, hist, early_stopping_evals, early_stopping_pct) -> bool:
        if len(hist) < early_stopping_evals + 1:
            return False
        improvement_pct = 1 - hist[-1][1] / hist[-(early_stopping_evals + 1)][1]
        return improvement_pct < early_stopping_pct

    def find_best_iter(self, hist, early_stopping_evals) -> int:
        best_iter, best_score = hist[-1]
        for r, m in hist[-(early_stopping_evals + 1) : -1]:
            if m < best_score:
                best_score = m
                best_iter = r
        return best_iter

    def fit(
        self,
        df: pd.DataFrame,
        n_windows: int,
        h: int,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        step_size: Optional[int] = None,
        num_iterations: int = 100,
        params: Optional[Dict[str, Any]] = None,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        eval_every: int = 10,
        weights: Optional[Sequence[float]] = None,
        metric: Union[str, Callable] = "mape",
        verbose_eval: bool = True,
        early_stopping_evals: int = 2,
        early_stopping_pct: float = 0.01,
        compute_cv_preds: bool = False,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        input_size: Optional[int] = None,
        weight_col: Optional[str] = None,
    ) -> List[CVResult]:
        """Train boosters simultaneously and assess their performance on the complete forecasting window.

        Args:
            df (pandas DataFrame): Series data in long format.
            n_windows (int): Number of windows to evaluate.
            h (int): Forecast horizon.
            id_col (str): Column that identifies each serie. Defaults to 'unique_id'.
            time_col (str): Column that identifies each timestep, its values can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            step_size (int, optional): Step size between each cross validation window. If None it will be equal to `h`.
                Defaults to None.
            num_iterations (int): Maximum number of boosting iterations to run. Defaults to 100.
            params (dict, optional): Parameters to be passed to the LightGBM Boosters. Defaults to None.
            static_features (list of str, optional): Names of the features that are static and will be repeated when forecasting.
                Defaults to None.
            dropna (bool): Drop rows with missing values produced by the transformations. Defaults to True.
            keep_last_n (int, optional): Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.
                Defaults to None.
            eval_every (int): Number of boosting iterations to train before evaluating on the whole forecast window. Defaults to 10.
            weights (sequence of float, optional): Weights to multiply the metric of each window. If None, all windows have the same weight.
                Defaults to None.
            metric (str or callable): Metric used to assess the performance of the models and perform early stopping. Defaults to 'mape'.
            verbose_eval (bool): Print the metrics of each evaluation.
            early_stopping_evals (int): Maximum number of evaluations to run without improvement. Defaults to 2.
            early_stopping_pct (float): Minimum percentage improvement in metric value in `early_stopping_evals` evaluations. Defaults to 0.01.
            compute_cv_preds (bool): Compute predictions for each window after finding the best iteration. Defaults to False.
            before_predict_callback (callable, optional): Function to call on the features before computing the predictions.
                This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.
                The series identifier is on the index. Defaults to None.
            after_predict_callback (callable, optional): Function to call on the predictions before updating the targets.
                This function will take a pandas Series with the predictions and should return another one with the same structure.
                The series identifier is on the index. Defaults to None.
            input_size (int, optional): Maximum training samples per serie in each window. If None, will use an expanding window.
                Defaults to None.

        Returns:
            (list of tuple): List of (boosting rounds, metric value) tuples.
        """
        self.setup(
            df=df,
            n_windows=n_windows,
            h=h,
            params=params,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            input_size=input_size,
            step_size=step_size,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            weights=weights,
            metric=metric,
            weight_col=weight_col,
        )
        hist = []
        for i in range(0, num_iterations, eval_every):
            metric_value = self.partial_fit(
                eval_every, before_predict_callback, after_predict_callback, weight_col=weight_col
            )
            rounds = eval_every + i
            hist.append((rounds, metric_value))
            if verbose_eval:
                print(f"[{rounds:,d}] {self.metric_name}: {metric_value:,f}")
            if self.should_stop(hist, early_stopping_evals, early_stopping_pct):
                print(f"Early stopping at round {rounds:,}")
                break
        self.best_iteration_ = self.find_best_iter(hist, early_stopping_evals)
        print(f"Using best iteration: {self.best_iteration_:,}")
        hist = hist[: self.best_iteration_ // eval_every]
        for _, bst, _ in self.items:
            bst.best_iteration = self.best_iteration_

        self.cv_models_ = {f"Booster{i}": item[1] for i, item in enumerate(self.items)}
        if compute_cv_preds:
            with ThreadPoolExecutor(self.num_threads) as executor:
                futures = []
                for ts, bst, valid in self.items:
                    future = executor.submit(
                        _predict,
                        ts=ts,
                        bst=bst,
                        valid=valid,
                        h=self.h,
                        before_predict_callback=before_predict_callback,
                        after_predict_callback=after_predict_callback,
                        weight_col=weight_col
                    )
                    futures.append(future)
                self.cv_preds_ = pd.concat(
                    [f.result().assign(window=i) for i, f in enumerate(futures)]
                )
        self.ts._fit(df, id_col, time_col, target_col, static_features, keep_last_n, weight_col)
        self.ts.as_numpy = False
        return hist

    def predict(
        self,
        h: int,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        X_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute predictions with each of the trained boosters.

        Args:
            h (int): Forecast horizon.
            before_predict_callback (callable): Function to call on the features before computing the predictions.
                This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.
                The series identifier is on the index. Defaults to None.
            after_predict_callback (callable): Function to call on the predictions before updating the targets.
                This function will take a pandas Series with the predictions and should return another one with the same structure.
                The series identifier is on the index. Defaults to None.
            X_df (pd.DataFrame): Dataframe with the future exogenous features. Should have the id column and the time column.
                Defaults to None.

        Returns:
            (pd.DataFrame): Predictions for each serie and timestep, with one column per window.
        """
        return self.ts.predict(
            self.cv_models_,
            horizon=h,
            before_predict_callback=before_predict_callback,
            after_predict_callback=after_predict_callback,
            X_df=X_df,
        )
