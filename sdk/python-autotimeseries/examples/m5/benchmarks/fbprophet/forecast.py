import time
from copy import deepcopy
from functools import partial
from multiprocessing import Pool, cpu_count

import pandas as pd
from prophet import Prophet


def fit_and_predict(index, ts, horizon): 
    model = Prophet(uncertainty_samples=False)
    model = model.fit(ts.drop('unique_id', 1))
    forecast = model.make_future_dataframe(periods=horizon, include_history=False)
    forecast = model.predict(forecast)
    forecast['unique_id'] = index
    forecast = forecast.filter(items=['unique_id', 'ds', 'yhat'])

    return forecast

def main() -> None:
    train = pd.read_parquet('../../data/m5/parquet/train/target.parquet')
    renamer = {'item_id': 'unique_id', 'timestamp': 'ds', 'demand': 'y'}
    train = train.rename(columns=renamer)
    train['unique_id'] = train['unique_id'].astype(object)
    print(train.info())
    
    partial_fit_and_predict = partial(fit_and_predict, horizon=28)
    start = time.time()
    print(f'Parallelism on {cpu_count()} CPU')
    with Pool(cpu_count()) as pool:
        forecasts = pool.starmap(partial_fit_and_predict, train.groupby('unique_id'))
    end = time.time()
    print(end - start)

    forecasts = pd.concat(forecasts)
    forecasts.to_csv('prophet-forecasts-m5.csv', index=False)


if __name__ == '__main__':
    main()
