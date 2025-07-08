import os
from pathlib import Path
from time import time
os.environ['NUMBA_RELEASE_GIL'] = 'True'
os.environ['NUMBA_CACHE'] = 'True'

import fire
import numpy as np
import pandas as pd
from datasetsforecast.hierarchical import HierarchicalData, HierarchicalInfo
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.evaluation import HierarchicalEvaluation
from hierarchicalforecast.methods import (
    BottomUp, TopDown, MiddleOut, 
    MinTrace, ERM
)
from statsforecast.core import StatsForecast
from statsforecast.models import ets
from statsforecast.utils import AirPassengers as ap


def rmsse(y, y_hat, y_insample):
    errors = np.mean((y - y_hat) ** 2, axis=1)
    scale = np.mean((y - y_insample[:, [-1]]) ** 2, axis=1)
    return np.mean(np.sqrt(errors) / np.sqrt(scale))

def get_str_model(group: str):
    if group == 'Wiki2':
        #avoid issues with seasonal models
        #due to negative and zero values
        return 'ZZA'
    return 'ZZZ'

def get_ERM_lambda(group: str):
    if group == 'Wiki2':
        return 1e6
    elif group == 'TourismSmall':
        return 2e6
    elif group == 'Labour':
        return 100
    raise Exception(f'Unkwon group {group}')

def pipeline(group: str):
    results_group_dir = Path(f'./results/{group}')
    results_group_dir.mkdir(exist_ok=True, parents=True)
    Y_df, S_df, tags = HierarchicalData.load('data', group)
    Y_df['ds'] = pd.to_datetime(Y_df['ds'])
    meta_info_group = HierarchicalInfo[group]
    h = meta_info_group.horizon
    freq = meta_info_group.freq
    sp = meta_info_group.seasonality
    #split train/test sets
    Y_df_test = Y_df.groupby(['unique_id']).tail(h)
    Y_df_train = Y_df.drop(Y_df_test.index)
    Y_df_test = Y_df_test.set_index('unique_id')
    Y_df_train = Y_df_train.set_index('unique_id')
    #forecaster
    str_model = get_str_model(group)
    init_time = time()
    forecaster = StatsForecast(
        df=Y_df_train,
        models=[(ets, sp, str_model)],
        freq=freq,
        n_jobs=-1,
    )
    Y_df_hat = forecaster.forecast(h, fitted=True)
    Y_df_fitted = forecaster.forecast_fitted_values()
    fcst_time = time() - init_time
    #reconciliation methods
    methods = [
        BottomUp(),
        TopDown(method='forecast_proportions'),
        TopDown(method='average_proportions'),
        TopDown(method='proportion_averages'),
        *[
            MiddleOut(level=name, top_down_method='forecast_proportions') \
            for name in list(tags.keys())[1:-1]
        ],
        *[
            MiddleOut(level=name, top_down_method='average_proportions') \
            for name in list(tags.keys())[1:-1]
        ],
        *[
            MiddleOut(level=name, top_down_method='proportion_averages') \
            for name in list(tags.keys())[1:-1]
        ],
        MinTrace(method='ols'),
        MinTrace(method='wls_struct'),
        MinTrace(method='wls_var'),
        MinTrace(method='mint_shrink'),
        ERM(method='closed'),
        ERM(method='reg', lambda_reg=get_ERM_lambda(group)),
        ERM(method='reg_bu', lambda_reg=get_ERM_lambda(group)),
    ]
    hrec = HierarchicalReconciliation(reconcilers=methods)
    Y_df_hat_rec = hrec.reconcile(
        Y_df_hat, 
        Y_df_fitted,
        S_df,
        tags
    )
    eval_tags = {'All': np.concatenate(list(tags.values()))}
    evaluator = HierarchicalEvaluation(evaluators=[rmsse])
    evals = evaluator.evaluate(
        Y_df_hat_rec,
        Y_df_test,
        eval_tags,
        Y_df=Y_df_train
    )
    evals = evals.loc['All'].reset_index()
    evals = pd.melt(
        evals,
        value_vars=evals.columns.to_list(),
        var_name='model',
        value_name=group,
    )
    evals[['model', 'rec_method']] = evals['model'].str.split('/', expand=True, n=1)
    evals['rec_method'] = evals['rec_method'].fillna('Base')
    evals = evals.drop(0)
    evals = evals[['rec_method', group]] 
    fcst_time = pd.DataFrame({'group': group, 'time': fcst_time}, index=[0])
    evals.to_csv(results_group_dir / 'statsforecast.csv', index=False)
    fcst_time.to_csv(results_group_dir / 'statsforecast-time.csv', index=False)


if __name__=="__main__":
    ets(ap.astype(np.float32), 12, season_length=12)
    fire.Fire(pipeline)



