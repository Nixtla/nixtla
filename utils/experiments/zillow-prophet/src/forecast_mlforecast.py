import time
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from mlforecast.core import TimeSeries
from mlforecast.forecast import Forecast
from sklearn.linear_model import LinearRegression
from window_ops.ewm import ewm_mean
from window_ops.expanding import expanding_mean


def main() -> None:
    train = pd.read_csv('data/prepared-data-train.csv')
    train['ds'] = pd.to_datetime(train['ds']) 
    train = train.set_index('unique_id')
    
    ts = TimeSeries(
        freq='M',
        num_threads=cpu_count(),
        lags=list(range(1, 4)),
        lag_transforms={
            i: [(expanding_mean), 
                (ewm_mean, 0.1), 
                (ewm_mean, 0.3),
                (ewm_mean, 0.5),
                (ewm_mean, 0.7),
                (ewm_mean, 0.9)]
            for i in range(1, 4)
        },
        date_features=['year', 'quarter', 'month']
    ) 

    start = time.time()
   
    model = LinearRegression()
    fcst = Forecast(model, ts)
    fcst.fit(train)
    forecasts = fcst.predict(4).rename(columns={'y_pred': 'mlforecast_lr'})
        
    end = time.time()
    print(f'Time: {end - start}')

    forecasts = forecasts.reset_index()
    forecasts.to_csv('data/mlforecast-forecasts.csv', index=False)


if __name__ == '__main__':
    main()
