import time
from multiprocessing import cpu_count

import fire
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from statsforecast import StatsForecast
from statsforecast.models import random_walk_with_drift

from src.data import get_data


def auto_arima_pmdarima(x, h, seasonality):
    try:
        mod = auto_arima(x, m=seasonality,
                         with_intercept=False)
        return mod.predict(h)

    except:
        return random_walk_with_drift(y, h)

def main(dataset: str = 'M3', group: str = 'Other') -> None:
    train, horizon, freq, seasonality = get_data('data/', dataset, group)
    train['ds'] = pd.to_datetime(train['ds']) 
    train = train.set_index('unique_id')
    
    models = [
        (auto_arima_pmdarima, seasonality)
    ]

    start = time.time()
    fcst = StatsForecast(train, models=models, freq=freq, n_jobs=cpu_count())
    fcst.last_dates = pd.DatetimeIndex(fcst.last_dates)
    forecasts = fcst.forecast(horizon)
    end = time.time()
    print(end - start)

    forecasts = forecasts.reset_index()
    forecasts.columns = ['unique_id', 'ds', 'auto_arima_pmdarima']
    forecasts.to_csv(f'data/pmdarima-forecasts-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['auto_arima_pmdarima']})
    time_df.to_csv(f'data/pmdarima-time-{dataset}-{group}.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)
