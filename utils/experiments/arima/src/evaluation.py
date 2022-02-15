import numpy as np
import pandas as pd
from neuralforecast.losses.numpy import mape, rmse, smape, mae

from src.data import get_data


def evaluate(dataset: str, group: str):
    str_forecasts = f'{dataset}-{group}'
    Y_test, horizon, *_ = get_data('data/', dataset, group, False)
    
    dfs = [pd.read_csv(f'data/{lib}-forecasts-{str_forecasts}.csv') for lib in ['pmdarima', 'statsforecast']]
    dfs = [df.set_index(['unique_id', 'ds']) for df in dfs]
    df_eval = pd.concat(dfs, axis=1).reset_index()
    df_eval['ds'] = pd.to_datetime(df_eval['ds'])
    df_eval = Y_test.merge(df_eval, how='left', on=['unique_id', 'ds'])
    print(df_eval.isna().mean())
    
    #fill missing forecasts with mean
    models = df_eval.drop(columns=['unique_id', 'ds', 'y']).columns.to_list()
    print(models)

    evals = {}
    y = df_eval['y'].values.reshape(-1, horizon)
    for model in models:
        evals[model] = {}
        y_hat = df_eval[model].values.reshape(-1, horizon)
        for metric in (mape, rmse, smape, mae):
            evals[model][metric.__name__] = metric(y, y_hat, axis=1).mean()

    evals = pd.DataFrame(evals)
    evals = evals.T.sort_values('mape')
    evals = evals.rename_axis('model')

    times = [pd.read_csv(f'data/{lib}-time-{str_forecasts}.csv') for lib in ['pmdarima', 'statsforecast']]
    times = [df.set_index('model') for df in times]
    times = pd.concat(times)
    evals = pd.concat([evals, times], axis=1)
    evals = evals.reset_index()
    evals.insert(0, 'dataset', str_forecasts)

    return evals


if __name__ == '__main__':
    evaluation = [evaluate('M3', group) for group in ['Other', 'Monthly', 'Quarterly', 'Yearly']]
    evaluation = pd.concat(evaluation).sort_values(['dataset', 'model']).reset_index(drop=True)

    evaluation.to_csv('data/evaluation.csv')
    print(evaluation.to_markdown())
