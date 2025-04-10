import os
import fire
import pandas as pd

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import (
    BottomUp, BottomUpSparse, TopDown, TopDownSparse, MiddleOut, MiddleOutSparse, 
    MinTrace, 
    MinTraceSparse, 
    OptimalCombination,
    ERM,
)

from src.data import get_data


def main():
    Y_train_df, Y_test_df, Y_hat_df, Y_fitted_df, S_df, tags = get_data()

    reconcilers = [BottomUp(),
                   BottomUpSparse(),
                   TopDown(method="forecast_proportions"),
                   TopDownSparse(method="forecast_proportions"),
                   TopDown(method="average_proportions"),
                   TopDownSparse(method="average_proportions"),
                   TopDown(method="proportion_averages"),
                   TopDownSparse(method="proportion_averages"),
                   MiddleOut(middle_level="Country/State", top_down_method="average_proportions"),
                   MiddleOutSparse(middle_level="Country/State", top_down_method="average_proportions"),
                   MinTrace(method='ols'),
                   MinTrace(method='wls_struct'),
                   MinTrace(method='wls_var'),
                   MinTrace(method='mint_cov'),
                   MinTrace(method='mint_shrink'),
                   MinTraceSparse(method='ols'),
                   MinTraceSparse(method='wls_struct'),
                   MinTraceSparse(method='wls_var'),
                   OptimalCombination(method='ols'),
                   OptimalCombination(method='wls_struct'),
                   ERM(method='closed'),
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df,
                               Y_df=Y_fitted_df, S=S_df, tags=tags)

    execution_times = pd.Series(hrec.execution_times).reset_index()

    if not os.path.exists('./data'):
        os.makedirs('./data')
    Y_rec_df.to_csv('./data/Y_rec.csv', index=False)
    execution_times.to_csv('./data/execution_times.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)
