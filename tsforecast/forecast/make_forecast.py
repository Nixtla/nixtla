#!/usr/bin/env python

import argparse
import json
import logging
import os
import re
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastcore.utils import store_attr
from statsforecast.core import StatsForecast
from statsforecast.models import auto_arima

class TSForecastARIMA:
    """Computes forecast at scale."""

    def __init__(self, 
		 filename: str,
                 dir_train: str,
                 dir_output: str, 	
		 freq: str,
                 seasonality: int,
                 unique_id_column: str,
                 ds_column: str, 
		 y_column: str,
                 horizon: int) -> 'TSForecast':
        store_attr()
        self.df: pd.DataFrame
        self.df_temporal: pd.DataFrame

        self.df, self.df_temporal = self._read_file()

    def _clean_columns(self, df: pd.DataFrame) -> None:
        new_columns = []
        for column in df.columns:
            new_column = re.sub(r'[",:{}[\]]', '', column)
            new_columns.append(new_column)

        df.columns = new_columns

    def _read_file(self) -> pd.DataFrame:
        logger.info('Reading file...')
        df = pd.read_parquet(f'{self.dir_train}/{self.filename}')
        logger.info('File read.')
        renamer = {self.unique_id_column: 'unique_id',
                   self.ds_column: 'ds',
                   self.y_column: 'y'}

        df.rename(columns=renamer, inplace=True)
        df.set_index(['unique_id', 'ds'], inplace=True)
        self._clean_columns(df)
        
        df_temporal = None
        df.reset_index('ds', inplace=True)
        df['ds'] = pd.to_datetime(df['ds'])

        return df, df_temporal

    def fit(self) -> 'TSForecastARIMA':
        """Train arima model."""
        fcst = StatsForecast(self.df, models=[(auto_arima, self.seasonality)],
			     freq=self.freq, n_jobs=os.cpu_count())
        logger.info('Starting training')
        preds = fcst.forecast(self.horizon)
        logger.info('Writing forecasts...')
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        forecast_file = f'{self.dir_output}/forecasts_{now}.csv'
        preds.to_csv(forecast_file)
        logger.info(f'File written... {forecast_file}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    
    # data config
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--freq', type=str, default='D')
    parser.add_argument('--seasonality', type=int, default=1)
    parser.add_argument('--unique-id-column', type=str, default='unique_id')
    parser.add_argument('--ds-column', type=str, default='ds')
    parser.add_argument('--y-column', type=str, default='y')

    # path config
    parser.add_argument('--dir-train', type=str, default='/opt/ml/input/data/train')
    parser.add_argument('--dir-output', type=str, default='/opt/ml/output/data')

    # forecast
    parser.add_argument('--horizon', type=int, default=14)

    args = parser.parse_args()


    forecast = TSForecastARIMA(**vars(args))
    forecast.fit()
