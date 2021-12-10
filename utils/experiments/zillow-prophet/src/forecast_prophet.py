import time
from copy import deepcopy
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from nixtlats.losses.numpy import mape
from prophet import Prophet
from sklearn.model_selection import ParameterGrid


params_grid = {'seasonality_mode': ['multiplicative','additive'],
               'changepoint_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5],
               'n_changepoints': [100, 150, 200],
               'fourier_order': [3, 5, 7, 9, 11]}
grid = ParameterGrid(params_grid)

def fit_and_predict(index, ts, horizon): 
    failed_hpo = False
    
    df = ts.drop('unique_id', 1)
    df_val = df.tail(horizon)
    df_train = df.drop(df_val.index)
    y_val = df_val['y'].values
    
    if len(df_train) >= horizon:
        val_results = {'losses': [], 'params': []}

        for params in grid:
            model = Prophet(seasonality_mode=params['seasonality_mode'],
                            weekly_seasonality=False,
                            n_changepoints=params['n_changepoints'],
                            changepoint_prior_scale=params['changepoint_prior_scale'])
            model.add_seasonality(name='monthly', 
                                  period=30.5, 
                                  fourier_order=params['fourier_order'])
            model = model.fit(df_train)
            
            forecast = model.make_future_dataframe(periods=horizon, 
                                                   include_history=False, 
                                                   freq='M')
            forecast = model.predict(forecast)
            forecast['unique_id'] = index
            forecast = forecast.filter(items=['unique_id', 'ds', 'yhat'])
            
            loss = mape(y_val, forecast['yhat'].values) 
            
            val_results['losses'].append(loss)
            val_results['params'].append(params)

        idx_params = np.argmin(val_results['losses']) 
        params = val_results['params'][idx_params]


    else:
        failed_hpo = True
        params = {'seasonality_mode': 'multiplicative',
                  'n_changepoints': 150,
                  'changepoint_prior_scale': 0.5,
                  'fourier_order': 5}
    
    model = Prophet(seasonality_mode=params['seasonality_mode'],
                    weekly_seasonality=False,
                    n_changepoints=params['n_changepoints'],
                    changepoint_prior_scale=params['changepoint_prior_scale'])
    model.add_seasonality(name='monthly', 
                          period=30.5, 
                          fourier_order=params['fourier_order'])
    model = model.fit(df)
    
    forecast = model.make_future_dataframe(periods=horizon, 
                                           include_history=False, 
                                           freq='M')
    forecast = model.predict(forecast)
    forecast['unique_id'] = index
    forecast = forecast.filter(items=['unique_id', 'ds', 'yhat'])

    return forecast, failed_hpo

def main() -> None:
    train = pd.read_csv('data/prepared-data-train.csv')
    train['ds'] = pd.to_datetime(train['ds'])
    #uids = train['unique_id'].unique()[:1]
    #train = train.query('unique_id in @uids')
    print(train.info())
    
    partial_fit_and_predict = partial(fit_and_predict, horizon=4)
    start = time.time()
    print(f'Parallelism on {cpu_count()} CPU')
    with Pool(cpu_count()) as pool:
        results = pool.starmap(partial_fit_and_predict, train.groupby('unique_id'))
    end = time.time()
    print(end - start)

    forecasts, failed = zip(*results)

    print(f'% HPO failed: {100 * np.mean(failed)}')

    forecasts = pd.concat(forecasts)
    forecasts.to_csv('data/prophet-forecasts.csv', index=False)


if __name__ == '__main__':
    main()
