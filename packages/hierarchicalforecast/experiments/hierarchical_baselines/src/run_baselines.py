import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA, Naive

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.evaluation import HierarchicalEvaluation
from hierarchicalforecast.methods import BottomUp, TopDown, MinTrace, ERM

from hierarchicalforecast.utils import is_strictly_hierarchical
from hierarchicalforecast.utils import HierarchicalPlot, CodeTimer
from hierarchicalforecast.evaluation import scaled_crps, msse, energy_score

from datasetsforecast.hierarchical import HierarchicalData, HierarchicalInfo

import warnings
# Avoid pandas fragmentation warning and positive definite warning
warnings.filterwarnings("ignore") 


class HierarchicalDataset(object):
    # Class with loading, processing and
    # prediction evaluation methods for hierarchical data

    available_datasets = ['Labour','Traffic',
                          'TourismSmall','TourismLarge','Wiki2',
                          'OldTraffic', 'OldTourismLarge']

    @staticmethod
    def _get_hierarchical_scrps(hier_idxs, Y, Yq_hat, q_to_pred):
        # We use the indexes obtained from the aggregation tags
        # to compute scaled CRPS across the hierarchy levels 
        scrps_list = []
        for idxs in hier_idxs:
            y      = Y[idxs, :]
            yq_hat = Yq_hat[idxs, :, :]
            scrps  = scaled_crps(y, yq_hat, q_to_pred)
            scrps_list.append(scrps)
        return scrps_list

    @staticmethod
    def _get_hierarchical_msse(hier_idxs, Y, Y_hat, Y_train):
        # We use the indexes obtained from the aggregation tags
        # to compute scaled CRPS across the hierarchy levels         
        msse_list = []
        for idxs in hier_idxs:
            y       = Y[idxs, :]
            y_hat   = Y_hat[idxs, :]
            y_train = Y_train[idxs, :]
            crps    = msse(y, y_hat, y_train)
            msse_list.append(crps)
        return msse_list    

    @staticmethod
    def _sort_hier_df(Y_df, S_df):
        # NeuralForecast core, sorts unique_id lexicographically
        # deviating from S_df, this class matches S_df and Y_hat_df order.
        Y_df.unique_id = Y_df.unique_id.astype('category')
        Y_df.unique_id = Y_df.unique_id.cat.set_categories(S_df.index)
        Y_df = Y_df.sort_values(by=['unique_id', 'ds'])
        return Y_df

    @staticmethod
    def _nonzero_indexes_by_row(M):
        return [np.nonzero(M[row,:])[0] for row in range(len(M))]    

    @staticmethod
    def load_process_data(dataset, directory='./data'):
        # Load data
        data_info = HierarchicalInfo[dataset]
        Y_df, S_df, tags = HierarchicalData.load(directory=directory,
                                                 group=dataset)

        # Parse and augment data
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        Y_df = HierarchicalDataset._sort_hier_df(Y_df=Y_df, S_df=S_df)

        # Obtain indexes for plots and evaluation
        hier_levels = ['Overall'] + list(tags.keys())
        hier_idxs = [np.arange(len(S_df))] +\
            [S_df.index.get_indexer(tags[level]) for level in list(tags.keys())]
        hier_linked_idxs = HierarchicalDataset._nonzero_indexes_by_row(S_df.values.T)

        # Final output
        data = dict(Y_df=Y_df, S_df=S_df, tags=tags,
                    # Hierarchical idxs
                    hier_idxs=hier_idxs,
                    hier_levels=hier_levels,
                    hier_linked_idxs=hier_linked_idxs,
                    # Dataset Properties
                    horizon=data_info.papers_horizon,
                    freq=data_info.freq,
                    seasonality=data_info.seasonality)
        return data


def run_baselines(dataset, intervals_method, verbose=False, seed=0):
    with CodeTimer('Read and Parse data   ', verbose):
        data = HierarchicalDataset.load_process_data(dataset=dataset)
        Y_df = data['Y_df'][["unique_id", 'ds', 'y']]
        S_df, tags = data['S_df'], data['tags']
        horizon = data['horizon']
        seasonality = data['seasonality']
        freq = data['freq']

        # Train/Test Splits
        Y_test_df  = Y_df.groupby('unique_id').tail(horizon)
        Y_train_df = Y_df.drop(Y_test_df.index)
        Y_test_df  = Y_test_df.set_index('unique_id')
        Y_train_df = Y_train_df.set_index('unique_id')

        dataset_str = f'{dataset}, h={horizon} '
        dataset_str += f'n_series={len(S_df)}, n_bottom={len(S_df.columns)} \n'
        dataset_str += f'test ds=[{min(Y_test_df.ds), max(Y_test_df.ds)}] '
        print(dataset_str)

    with CodeTimer('Fit/Predict Model	  ', verbose):
        # Read to avoid unnecesary AutoARIMA computation
        yhat_file = f'./data/{dataset}/Y_hat.csv'
        yfitted_file = f'./data/{dataset}/Y_fitted.csv'
        yrec_file = f'./data/{dataset}/{intervals_method}_rec.csv'

        if os.path.exists(yhat_file):
            Y_hat_df = pd.read_csv(yhat_file)
            Y_fitted_df = pd.read_csv(yfitted_file)

        else:
            if not os.path.exists(f'./data/{dataset}'):
                os.makedirs(f'./data/{dataset}')			
            fcst = StatsForecast(
                df=Y_train_df, 
                models=[AutoARIMA(season_length=seasonality)],
                fallback_model=[Naive()],
                freq=freq, 
                n_jobs=-1
            )
            Y_hat_df = fcst.forecast(h=horizon, fitted=True, level=LEVEL)
            Y_fitted_df = fcst.forecast_fitted_values()

            Y_hat_df = Y_hat_df.reset_index()
            Y_fitted_df = Y_fitted_df.reset_index()
            Y_hat_df.to_csv(yhat_file, index=False)
            Y_fitted_df.to_csv(yfitted_file, index=False)

        Y_hat_df = Y_hat_df.set_index('unique_id')
        Y_fitted_df = Y_fitted_df.set_index('unique_id')

    with CodeTimer('Reconcile Predictions ', verbose):
        if is_strictly_hierarchical(S=S_df.values.astype(np.float32), 
            tags={key: S_df.index.get_indexer(val) for key, val in tags.items()}):
            reconcilers = [
                BottomUp(),
                TopDown(method='average_proportions'),
                TopDown(method='proportion_averages'),
                MinTrace(method='ols'),
                MinTrace(method='wls_var'),
                MinTrace(method='mint_shrink'),
                #ERM(method='reg_bu', lambda_reg=100) # Extremely inneficient
                ERM(method='closed')
            ]
        else:
            reconcilers = [
                BottomUp(),
                MinTrace(method='ols'),
                MinTrace(method='wls_var'),
                MinTrace(method='mint_shrink'),
                #ERM(method='reg_bu', lambda_reg=100) # Extremely inneficient
                ERM(method='closed')
            ]
        
        hrec = HierarchicalReconciliation(reconcilers=reconcilers)
        Y_rec_df = hrec.bootstrap_reconcile(Y_hat_df=Y_hat_df,
                                            Y_df=Y_fitted_df,
                                            S_df=S_df, tags=tags,
                                            level=LEVEL,
                                            intervals_method=intervals_method,
                                            num_samples=10, num_seeds=10)

        # Matching Y_test/Y_rec/S index ordering
        Y_test_df = Y_test_df.reset_index()
        Y_test_df.unique_id = Y_test_df.unique_id.astype('category')
        Y_test_df.unique_id = Y_test_df.unique_id.cat.set_categories(S_df.index)
        Y_test_df = Y_test_df.sort_values(by=['unique_id', 'ds'])

        Y_rec_df = Y_rec_df.reset_index()
        Y_rec_df.unique_id = Y_rec_df.unique_id.astype('category')
        Y_rec_df.unique_id = Y_rec_df.unique_id.cat.set_categories(S_df.index)
        Y_rec_df = Y_rec_df.sort_values(by=['seed', 'unique_id', 'ds'])

        Y_rec_df.to_csv(yrec_file, index=False)

        # Parsing model level columns
        flat_cols = list(hrec.level_names.keys())
        for model in hrec.level_names:
            flat_cols += hrec.level_names[model]
        for model in hrec.sample_names:
            flat_cols += hrec.sample_names[model]
        y_rec  = Y_rec_df[flat_cols]
        model_columns = y_rec.columns

        n_series = len(S_df)
        n_seeds = len(Y_rec_df.seed.unique())
        y_rec  = y_rec.values.reshape(n_seeds, n_series, horizon, len(model_columns))
        y_test = Y_test_df['y'].values.reshape(n_series, horizon)
        y_train = Y_train_df['y'].values.reshape(n_series, -1)

    with CodeTimer('Evaluate Base Forecasts  ', verbose):
        crps_results = {'Dataset': [dataset] * len(['Overall'] + list(tags.keys())),
                        'Level': ['Overall'] + list(tags.keys()),}
        msse_results = {'Dataset': [dataset] * len(['Overall'] + list(tags.keys())),
                        'Level': ['Overall'] + list(tags.keys()),}
        Y_hat_quantiles = Y_hat_df.drop(columns=['ds', 'AutoARIMA'])
        y_hat_quantiles_np = Y_hat_quantiles.values.reshape(n_series, horizon, len(QUANTILES))
        y_hat_np = Y_hat_df['AutoARIMA'].values.reshape(n_series, -1)

        crps_results['AutoARIMA'] = HierarchicalDataset._get_hierarchical_scrps(
                                                            Y=y_test,
                                                            Yq_hat=y_hat_quantiles_np,
                                                            q_to_pred=QUANTILES,
                                                            hier_idxs=data['hier_idxs'])
        msse_results['AutoARIMA'] = HierarchicalDataset._get_hierarchical_msse(
                                                            Y=y_test,
                                                            Y_hat=y_hat_np,
                                                            Y_train=y_train,
                                                            hier_idxs=data['hier_idxs'])

    with CodeTimer('Evaluate Models CRPS  ', verbose):
        for model in hrec.level_names.keys():
            crps_results[model] = []
            for level in crps_results['Level']:
                if level=='Overall':
                    row_idxs = np.arange(len(S_df))
                else:
                    row_idxs = S_df.index.get_indexer(tags[level])
                col_idxs = model_columns.get_indexer(hrec.level_names[model])
                _y = y_test[row_idxs,:]
                _y_rec_seeds = y_rec[:,row_idxs,:,:][:,:,:,col_idxs]

                level_model_crps =[]
                for seed_idx in range(y_rec.shape[0]):
                    _y_rec = _y_rec_seeds[seed_idx,:,:,:]
                    level_model_crps.append(scaled_crps(y=_y, y_hat=_y_rec,
                                                        quantiles=QUANTILES))
                level_model_crps = f'{np.mean(level_model_crps):.4f}±{(1.96 * np.std(level_model_crps)):.4f}'
                crps_results[model].append(level_model_crps)

        crps_results = pd.DataFrame(crps_results)

    with CodeTimer('Evaluate Models MSSE  ', verbose):
        for model in hrec.level_names.keys():
            msse_results[model] = []
            for level in msse_results['Level']:
                if level=='Overall':
                    row_idxs = np.arange(len(S_df))
                else:
                    row_idxs = S_df.index.get_indexer(tags[level])
                col_idx = model_columns.get_loc(model)
                _y = y_test[row_idxs,:]
                _y_train = y_train[row_idxs,:]
                _y_hat_seeds = y_rec[:,row_idxs,:,:][:,:,:,col_idx]

                level_model_msse = []
                for seed_idx in range(y_rec.shape[0]):
                    _y_hat = _y_hat_seeds[seed_idx,:,:]
                    level_model_msse.append(msse(y=_y, y_hat=_y_hat, y_train=_y_train))
                level_model_msse = f'{np.mean(level_model_msse):.4f}'
                msse_results[model].append(level_model_msse)

        msse_results = pd.DataFrame(msse_results)

        return crps_results, msse_results

if __name__ == '__main__':
    
    # Parse execution parameters
    verbose = True
    parser = argparse.ArgumentParser()
    parser.add_argument("-intervals_method", "--intervals_method", type=str)
    parser.add_argument("-dataset", "--dataset", type=str)

    args = parser.parse_args()
    intervals_method = args.intervals_method
    dataset = args.dataset

    assert intervals_method in ['bootstrap', 'normality', 'permbu'], \
        "Select `--intervals_method` from ['bootstrap', 'normality', 'permbu']"

    available_datasets = ['Labour', 'Traffic', 'OldTraffic',
                          'TourismSmall', 'TourismLarge', 'OldTourismLarge', 'Wikitwo']
    assert dataset in available_datasets, \
        "Select `--dataset` from ['Labour', 'Traffic', 'OldTraffic', \
            'TourismSmall', 'TourismLarge', 'OldTourismLarge', 'Wikitwo']"

    print(f'\n {intervals_method.upper()} {dataset} statistical baselines evaluation \n')

    LEVEL = np.arange(0, 100, 2)
    qs = [[50-lv/2, 50+lv/2] for lv in LEVEL]
    QUANTILES = np.sort(np.concatenate(qs)/100)

    # Run experiments
    crps_results_list = []
    msse_results_list = []
    try:
        crps_results, msse_results = run_baselines(dataset=dataset,
                    intervals_method=intervals_method, verbose=verbose)
        crps_results_list.append(crps_results)
        msse_results_list.append(msse_results)
    except Exception as e:
        print('failed ', dataset)
        print(str(e))
    print('\n\n')

    crps_results_df = pd.concat(crps_results_list)
    msse_results_df = pd.concat(msse_results_list)

    crps_results_df.to_csv(f'./data/{intervals_method}_crps.csv', index=False)
    msse_results_df.to_csv(f'./data/{intervals_method}_msse.csv', index=False)

    print('='*(200+24))
    print(f'{intervals_method} sCRPS:')
    print(crps_results_df)
    print('\n\n'+'='*(200+24))
    print(f'{intervals_method} relMSE:')
    print(msse_results_df)
