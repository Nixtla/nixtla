import argparse
import json
import logging
import os
from math import sqrt
from pathlib import Path
from typing import List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastcore.utils import store_attr
from mlforecast.core import TimeSeries
from mlforecast.forecast import Forecast
from window_ops.ewm import ewm_mean
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean, seasonal_rolling_mean


freq2config = {
    'D': dict(
        lags=[7, 28],
        lag_transforms={
            7: [(rolling_mean, 7), (rolling_mean, 28)],
            28: [
                (rolling_mean, 7),
                (rolling_mean, 28),
                (seasonal_rolling_mean, 7, 4),
                (seasonal_rolling_mean, 7, 8),
            ],
        },
        date_features=['year', 'quarter', 'month', 'week', 'day', 'dayofweek'],
    ),
    'W': dict(
        lags=[1, 2, 3, 4],
        lag_transforms={
            1: [(expanding_mean), (ewm_mean, 0.1), (ewm_mean, 0.3)],
        },
        date_features=['year', 'quarter', 'month', 'week']
    ),
}

class TSForecast:
    """Computes forecast at scale."""

    def __init__(self, filename: str,
                 filename_static: str,
                 filename_temporal: str,
                 filename_temporal_future: str,
                 freq: str,
                 unique_id_column: str,
                 ds_column: str, y_column: str,
                 horizon: int, 
                 backtest_windows: int,
                 objective: str, metric: str,
                 learning_rate: int, n_estimators: int,
                 num_leaves: int, min_data_in_leaf: int,
                 bagging_freq: int, bagging_fraction: float) -> 'TSForecast':
        store_attr()
        self.dir = '/opt/ml'
        self.dir_train = '/opt/ml/input/data/train'
        self.df: pd.DataFrame
        self.df_temporal_future: pd.DataFrame
        self.fcst: Forecast
        self.static_features: List[str]

        self.df, self.df_temporal_future, self.static_features = self._read_file()

    def _read_file(self) -> pd.DataFrame:
        logger.info('Reading file...')
        df = pd.read_csv(f'{self.dir_train}/{self.filename}')
        logger.info('File read.')
        renamer = {self.unique_id_column: 'unique_id',
                   self.ds_column: 'ds',
                   self.y_column: 'y'}

        df.rename(columns=renamer, inplace=True)
        df.set_index(['unique_id', 'ds'], inplace=True)

        static_features = None
        if self.filename_static is not None:
            static = pd.read_csv(f'{self.dir_train}/{self.filename_static}')
            static.rename(columns=renamer, inplace=True)
            static.set_index('unique_id', inplace=True)

            static_features = static.select_dtypes('object').columns.to_list()
            static[static_features] = static[static_features].astype('category')
            
            df = df.merge(static, how='left', left_on=['unique_id'], 
                          right_index=True)
        
        df_temporal_future = None
        if self.filename_temporal is not None:
            if self.filename_temporal_future is None:
                raise ValueError(
                    'You should pass `filename_temporal_future` '
                    'when using temporal variables'
                )
            temporal = pd.read_csv(f'{self.dir_train}/{self.filename_temporal}')
            temporal.rename(columns=renamer, inplace=True)

            temporal.set_index(['unique_id', 'ds'], inplace=True)
            df = df.merge(temporal, how='left', left_on=['unique_id', 'ds'],
                          right_index=True)

            df_temporal_future = pd.read_csv(f'{self.dir_train}/{self.filename_temporal_future}')
            df_temporal_future.rename(columns=renamer, inplace=True)
            # Check size of temporal future variables
            n_uids = df_temporal_future['unique_id'].nunique()
            if n_uids * self.horizon < df_temporal_future.shape[0]:
                raise ValueError(
                    f'Each time series should have at least {self.horizon} '
                    'observations (forecast horizon) for temporal future '
                    'variables.'
                )
            df_temporal_future['ds'] = pd.to_datetime(df_temporal_future['ds'])

        df.reset_index('ds', inplace=True)
        df['ds'] = pd.to_datetime(df['ds'])

        return df, df_temporal_future, static_features

    def train(self) -> 'TSForecast':
        """Train LGB model."""
        model = lgb.LGBMRegressor(
            objective=self.objective,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_data_in_leaf=self.min_data_in_leaf,
            bagging_freq=self.bagging_freq,
            bagging_fraction=self.bagging_fraction,
            random_state=0,
            metric=self.metric,
        )

        flow_config = freq2config[self.freq[0]]
        ts = TimeSeries(
            freq=self.freq,
            num_threads=os.cpu_count(),
            **flow_config
        )

        fcst = Forecast(model, ts)

        if self.backtest_windows > 0:
            dynamic_dfs = self.df.reset_index().drop('y', axis=1) if self.df_temporal_future is not None else None
            results = fcst.backtest(
                self.df,
                self.backtest_windows,
                self.horizon,
                static_features=self.static_features,
                dynamic_dfs=[dynamic_dfs],
            )
            rmses = []
            for i, result in enumerate(results):
                result = result.fillna(0)
                sq_errs = (result['y'] - result['y_pred']).pow(2)
                rmse = sqrt(sq_errs.groupby('unique_id').mean().mean())
                rmses.append(rmse)
                
                result.to_csv(f'{self.dir}/output/valid_{i}.csv')

            print(f'RMSE: {np.mean(rmses):.4f}')

        prep_df = fcst.preprocess(self.df,
                                  static_features=self.static_features)
        rng = np.random.RandomState(0)
        train_mask = rng.rand(prep_df.shape[0]) < 0.9
        train_df, valid_df = prep_df[train_mask], prep_df[~train_mask]
        del prep_df
        X_train, y_train = train_df.drop(columns=['ds', 'y']), train_df['y']
        X_valid, y_valid = valid_df.drop(columns=['ds', 'y']), valid_df['y']
        fcst.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_names=['train', 'valid'],
            verbose=20,
        )

        self.fcst = fcst

        return self

    def predict(self) -> None:
        """Gets forecast."""
        preds = self.fcst.predict(
            self.horizon, 
            dynamic_dfs=[self.df_temporal_future] if self.df_temporal_future is not None else None
        )

        logger.info('Writing forecasts...')
        preds.to_csv(f'{self.dir}/output/forecasts.csv')
        logger.info('File written...')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    # data config
    parser.add_argument('--filename', type=str)
    parser.add_argument('--filename-static', type=str, default=None)
    parser.add_argument('--filename-temporal', type=str, default=None)
    parser.add_argument('--filename-temporal-future', type=str, default=None)
    parser.add_argument('--freq', type=str, default='D')
    parser.add_argument('--unique-id-column', type=str, default='unique_id')
    parser.add_argument('--ds-column', type=str, default='ds')
    parser.add_argument('--y-column', type=str, default='y')

    # forecast
    parser.add_argument('--horizon', type=int, default=28)
    parser.add_argument('--backtest-windows', type=int, default=0)

    # hparams
    parser.add_argument('--objective', type=str, default='l2')
    parser.add_argument('--metric', type=str, default='rmse')
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--num-leaves', type=int, default=128)
    parser.add_argument('--min-data-in-leaf', type=int, default=20)
    parser.add_argument('--bagging-freq', type=int, default=0)
    parser.add_argument('--bagging-fraction', type=float, default=1.)

    args = parser.parse_args()


    forecast = TSForecast(**vars(args))
    forecast = forecast.train()
    forecast.predict()
