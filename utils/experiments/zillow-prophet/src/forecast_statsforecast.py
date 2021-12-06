import time
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    historic_average,
    naive,
    random_walk_with_drift, 
    seasonal_naive, 
    ses, 
    window_average
)


def main() -> None:
    train = pd.read_csv('data/prepared-data-train.csv')
    print(train.head())
    train['ds'] = pd.to_datetime(train['ds']) 
    print(train['ds'].dt.freq)
    train = train.set_index('unique_id')
    print(train.info())
    print(train.head())
    
    models = [
	historic_average,
	naive,
	random_walk_with_drift,
	(ses, 0.9),
        (seasonal_naive, 12),
        (window_average, 4)
    ]

    fcst = StatsForecast(train, models=models, freq='M', n_jobs=cpu_count())
    fcst.last_dates = pd.DatetimeIndex(fcst.last_dates)
    start = time.time()
    forecasts = fcst.forecast(4)
    end = time.time()

    forecasts = forecasts.reset_index()
    forecasts.to_csv('data/statsforecast-forecasts.csv', index=False)


if __name__ == '__main__':
    main()
