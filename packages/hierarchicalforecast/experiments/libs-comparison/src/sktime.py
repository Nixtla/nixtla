from pathlib import Path
from time import time

import fire
import numpy as np
import pandas as pd
from datasetsforecast.hierarchical import HierarchicalData, HierarchicalInfo
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.reconcile import ReconcilerForecaster
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconcile import Reconciler


def rmsse(y, y_hat, y_insample):
    errors = np.mean((y - y_hat) ** 2, axis=1)
    scale = np.mean((y - y_insample[:, [-1]]) ** 2, axis=1)
    return np.mean(np.sqrt(errors) / np.sqrt(scale))

def hierarchical_cols(group: str):
    if group == 'Wiki2':
        return ['Country', 'Access', 'Agent', 'Topic'], ['Country', 'Access', 'Agent', 'Topic'], '_'
    elif group == 'Labour':
        return ['Employment', 'Gender', 'Region'], ['Region', 'Employment', 'Gender'], ','
    elif group == 'TourismSmall':
        return ['State', 'Purpose', 'CityNonCity'], ['Purpose', 'State', 'CityNonCity'], '-'
    raise Exception(f'Unknown group {group}')

def pipeline(group: str):
    results_group_dir = Path(f'./results/{group}')
    results_group_dir.mkdir(exist_ok=True, parents=True)
    init_cols, hier_cols, sep = hierarchical_cols(group)
    Y_df, S, tags = HierarchicalData.load('data', group)
    n_series = Y_df['unique_id'].nunique()
    meta_info_group = HierarchicalInfo[group]
    h = meta_info_group.horizon
    freq = meta_info_group.freq
    sp = meta_info_group.seasonality
    #Get only bottom series
    #to contruct full dataset using sktime
    Y_df = Y_df.query('unique_id in @S.columns')
    Y_df[init_cols] = Y_df['unique_id'].str.split(sep, expand=True)
    if group == 'Labour':
        freq = 'M'
    Y_df['ds'] = pd.PeriodIndex(Y_df['ds'], freq=freq)
    Y_df = Y_df.set_index(hier_cols+['ds'])[['y']]
    #Aggregation
    agg = Aggregator(flatten_single_levels=False)
    Y_df = agg.fit_transform(Y_df)
    Y_df = Y_df.reset_index()
    n_agg_series = len(Y_df[hier_cols].drop_duplicates())
    if n_agg_series != n_series:
        raise Exception('mismatch n_series original and sktime')
    #split train/test sets
    Y_df_test = Y_df.groupby(hier_cols).tail(h)
    Y_df_train = Y_df.drop(Y_df_test.index)
    Y_df_test = Y_df_test.set_index(hier_cols+['ds'])
    Y_df_train = Y_df_train.set_index(hier_cols+['ds'])
    #forecaster
    seasonal = 'Additive' if group == 'Wiki2' else None #prevent negative and zero values
    init_time = time()
    forecaster = AutoETS(auto=True, sp=sp, seasonal=seasonal, n_jobs=-1)
    forecaster.fit(Y_df_train)
    prds = forecaster.predict(fh=np.arange(1, h + 1)).rename(columns={'y': 'base'})
    fcst_time = time() - init_time
    #reconciliation methods
    methods = ['bu', 'ols', 'wls_str', 'td_fcst']
    for method in methods:
        reconciler = Reconciler(method=method)
        prds_recon = reconciler.fit_transform(prds[['base']]).rename(columns={'base': method})
        prds = prds.merge(prds_recon, how='left', left_index=True, right_index=True)
    #methods based on residuals
    methods_res = ['wls_var', 'mint_shrink']
    for method in methods_res:
        reconciler = ReconcilerForecaster(forecaster, method=method)
        reconciler.fit(Y_df_train)
        prds_recon = reconciler.predict(fh=np.arange(1, h + 1)).rename(columns={'y': method})
        prds = prds.merge(prds_recon, how='left', left_index=True, right_index=True)
    #adding y_test for evaluation
    prds = prds.merge(Y_df_test, how='left', left_index=True, right_index=True)
    #evaluation
    y_test = prds['y'].values.reshape(-1, h)
    y_insample = Y_df_train['y'].values.reshape(n_series, -1)
    evals = {}
    for method in ['base'] + methods + methods_res:
        y_hat = prds[method].values.reshape(-1, h)
        evals[method] = rmsse(y_test, y_hat, y_insample)
    evals = pd.DataFrame(evals, index=[group])
    fcst_time = pd.DataFrame({'group': group, 'time': fcst_time}, index=[0])
    evals.to_csv(results_group_dir / 'sktime.csv', index=False)
    fcst_time.to_csv(results_group_dir / 'sktime-time.csv', index=False)


if __name__=="__main__":
    fire.Fire(pipeline)



