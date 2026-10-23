import os
import fire
import pandas as pd
import pickle

from hierarchicalforecast.utils import aggregate_temporal
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import (
    BottomUp, BottomUpSparse, TopDown, 
    TopDownSparse, 
    MiddleOut, 
    MiddleOutSparse, 
    MinTrace, 
    MinTraceSparse, 
    OptimalCombination,
    # ERM,
)

from statsforecast.models import AutoETS
from statsforecast.core import StatsForecast

def main():

    # Read data
    Y_df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/tourism.csv')
    Y_df = Y_df.rename({'Trips': 'y', 'Quarter': 'ds'}, axis=1)
    Y_df.insert(0, 'Country', 'Australia')
    Y_df = Y_df[['Country', 'Region', 'State', 'Purpose', 'ds', 'y']]
    Y_df['ds'] = Y_df['ds'].str.replace(r'(\d+) (Q\d)', r'\1-\2', regex=True)
    Y_df['ds'] = pd.PeriodIndex(Y_df["ds"], freq='Q').to_timestamp()
    Y_df["unique_id"] = Y_df["Country"] + "/" + Y_df["State"] + "/" + Y_df["Region"] + "/" + Y_df["Purpose"]

    # split data into train and test
    horizon = 8
    Y_test_df = Y_df.groupby("unique_id", as_index=False).tail(horizon)
    Y_train_df = Y_df.drop(Y_test_df.index)    

    # Temporal Hierarchical Aggregation
    spec_temporal = {"year": 4, "semiannual": 2, "quarter": 1}
    Y_train_df, S_train_df, tags_train = aggregate_temporal(df=Y_train_df, spec=spec_temporal)
    Y_test_df, S_test_df, tags_test = aggregate_temporal(df=Y_test_df,  spec=spec_temporal)

    # Create forecasts
    Y_hat_dfs = []
    id_cols = ["unique_id", "temporal_id", "ds"]
    # We will train a model for each temporal level
    for level, temporal_ids_train in tags_train.items():
        # Filter the data for the level
        Y_level_train = Y_train_df.query("temporal_id in @temporal_ids_train")
        temporal_ids_test = tags_test[level] # noqa F841
        Y_level_test = Y_test_df.query("temporal_id in @temporal_ids_test")
        # For each temporal level we have a different frequency and forecast horizon
        freq_level = pd.infer_freq(Y_level_train["ds"].unique())
        horizon_level = Y_level_test["ds"].nunique()
        # Train a model and create forecasts
        fcst = StatsForecast(models=[AutoETS(model='ZZZ')], freq=freq_level, n_jobs=-1)
        Y_hat_df_level = fcst.forecast(df=Y_level_train[["ds", "unique_id", "y"]], h=horizon_level)
        # Add the test set to the forecast
        Y_hat_df_level = Y_hat_df_level.merge(Y_level_test.drop(columns="y"), on=["ds", "unique_id"], how="left")
        # Put cols in the right order (for readability)
        Y_hat_cols = id_cols + [col for col in Y_hat_df_level.columns if col not in id_cols]
        Y_hat_df_level = Y_hat_df_level[Y_hat_cols]
        # Append the forecast to the list
        Y_hat_dfs.append(Y_hat_df_level)

    Y_hat_df = pd.concat(Y_hat_dfs, ignore_index=True)

    reconcilers = [
                   BottomUp(),
                   BottomUpSparse(),
                   TopDown(method="forecast_proportions"),
                   TopDownSparse(method="forecast_proportions"),
                #    TopDown(method="average_proportions"),
                #    TopDownSparse(method="average_proportions"),
                #    TopDown(method="proportion_averages"),
                #    TopDownSparse(method="proportion_averages"),
                   MiddleOut(middle_level="semiannual", top_down_method="forecast_proportions"),
                   MiddleOutSparse(middle_level="semiannual", top_down_method="forecast_proportions"),
                   MinTrace(method='ols'),
                   MinTrace(method='wls_struct'),
                #    MinTrace(method='wls_var'),
                #    MinTrace(method='mint_cov'),
                #    MinTrace(method='mint_shrink'),
                   MinTraceSparse(method='ols'),
                   MinTraceSparse(method='wls_struct'),
                #    MinTraceSparse(method='wls_var'),
                   OptimalCombination(method='ols'),
                   OptimalCombination(method='wls_struct'),
                #    ERM(method='closed'),
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)

    Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, 
                            S_df=S_test_df, 
                            tags=tags_test, 
                            temporal=True,
                            )

    execution_times = pd.Series(hrec.execution_times).reset_index()

    if not os.path.exists('./data'):
        os.makedirs('./data')
    Y_rec_df.to_csv('./data/Y_rec.csv', index=False)
    Y_test_df.to_csv('./data/Y_test.csv', index=False)
    Y_train_df.to_csv('./data/Y_train.csv', index=False)
    with open('./data/tags.pickle', 'wb') as handle:
        pickle.dump(tags_test, handle, protocol=pickle.HIGHEST_PROTOCOL)      

    execution_times.to_csv('./data/execution_times.csv', index=False)

if __name__ == '__main__':
    fire.Fire(main)
