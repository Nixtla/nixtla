import time
from copy import deepcopy
from functools import partial
from multiprocessing import Pool, cpu_count

import pandas as pd
from prophet import Prophet


def fit_and_predict(index, ts, horizon): 
    model = Prophet(seasonality_mode='multiplicative',
                    weekly_seasonality=False,
                    changepoint_prior_scale=0.5)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model = model.fit(ts.drop('unique_id', 1))
    forecast = model.make_future_dataframe(periods=horizon, include_history=False, freq='M')
    forecast = model.predict(forecast)
    forecast['unique_id'] = index
    forecast = forecast.filter(items=['unique_id', 'ds', 'yhat'])

    return forecast

def main() -> None:
    train = pd.read_csv('data/prepared-data-train.csv')
    train['ds'] = pd.to_datetime(train['ds'])
    print(train.info())
    
    partial_fit_and_predict = partial(fit_and_predict, horizon=4)
    start = time.time()
    print(f'Parallelism on {cpu_count()} CPU')
    with Pool(cpu_count()) as pool:
        forecasts = pool.starmap(partial_fit_and_predict, train.groupby('unique_id'))
    end = time.time()
    print(end - start)

    forecasts = pd.concat(forecasts)
    forecasts.to_csv('data/prophet-forecasts.csv', index=False)


if __name__ == '__main__':
    main()
