import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from nixtlats.losses.numpy import mape, rmse, smape, mae


if __name__ == '__main__':
    test = pd.read_csv('data/prepared-data-test.csv')
    prophet = pd.read_csv('data/prophet-forecasts.csv').rename({'yhat': 'prophet'}, axis=1)
    arima = pd.read_csv('data/arima-forecasts.csv').rename({'yhat': 'auto.arima'}, axis=1)
    arima['ds'] = (pd.to_datetime(arima['ds']) + MonthEnd(1)).dt.strftime('%Y-%m-%d')
    stats = pd.read_csv('data/statsforecast-forecasts.csv')
    mlforecast = pd.read_csv('data/mlforecast-forecasts.csv')#.rename({'y_pred': 'mlforecast'}, axis=1)
    
    dfs = [test, prophet, arima, stats, mlforecast]
    dfs = [df.set_index(['unique_id', 'ds']) for df in dfs]
    df_eval = pd.concat(dfs, axis=1).reset_index()
    print(df_eval.isna().mean())
    
    #fill missing forecasts with mean
    models = df_eval.drop(columns=['unique_id', 'ds', 'y']).columns.to_list()
    print(models)
    for model in models:
        df_eval[model] = df_eval[model].fillna(df_eval['window_average_window_size-4'])

    evals = {}
    y = df_eval['y'].values.reshape(-1, 4)
    for model in models:
        evals[model] = {}
        y_hat = df_eval[model].values.reshape(-1, 4)
        for metric in (mape, rmse, smape, mae):
            evals[model][metric.__name__] = metric(y, y_hat, axis=-1).mean()

    evals = pd.DataFrame(evals)
    evals = evals.T.sort_values('mape')

    evals.to_csv('data/evaluation.csv')
    print(evals.to_markdown())
