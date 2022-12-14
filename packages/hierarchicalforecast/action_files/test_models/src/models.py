import os
import time

import fire
import numpy as np
import pandas as pd

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import (
    BottomUp, MinTrace, 
)

from src.data import get_data


def main():
    Y_train_df, Y_test_df, Y_hat_df, Y_fitted_df, S_df, tags = get_data()

    reconcilers = [BottomUp(),
                   MinTrace(method='mint_shrink'),
                   MinTrace(method='ols')]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df,
                               Y_df=Y_fitted_df, S=S_df, tags=tags)

    Y_rec_df = Y_rec_df.reset_index()
    execution_times = pd.Series(hrec.execution_times).reset_index()

    if not os.path.exists('./data'):
        os.makedirs('./data')
    Y_rec_df.to_csv('./data/Y_rec.csv', index=False)
    execution_times.to_csv('./data/execution_times.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)
