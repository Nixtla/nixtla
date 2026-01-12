import os
import pickle

import fire
import pandas as pd
import polars as pl
from src.data import get_tourism
from statsforecast.core import StatsForecast
from statsforecast.models import AutoETS

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import (
    ERM,
    BottomUp,
    BottomUpSparse,
    MiddleOut,
    MiddleOutSparse,
    MinTrace,
    MinTraceSparse,
    OptimalCombination,
    TopDown,
    TopDownSparse,
)
from hierarchicalforecast.utils import aggregate

SPECS = {
        "strict": [
                    ['Country'],
                    ['Country', 'State'],
                    ['Country', 'State', 'Region'],
                    ['Country', 'State', 'Region', 'Purpose']
                    ],
        "non-strict": [
                    ['Country'],
                    ['Country', 'State'],
                    ['Country', 'Purpose'],
                    ['Country', 'State', 'Region'],
                    ['Country', 'State', 'Purpose'],
                    ['Country', 'State', 'Region', 'Purpose']
                    ],
                    }


def main(hierarchy: str = "non-strict", type: str = "point", engine: str = 'pandas') -> None:
    if type == "probabilistic":
        level = [80, 90]
    elif type == "point":
        level = None
    # Get data
    Y_df = get_tourism()
    freq = "QS"
    if engine == 'polars':
        Y_df = pl.from_pandas(Y_df)
        freq = "1q"

    # Hierarchical Aggregation
    spec = SPECS[hierarchy]
    Y_df, S_df, tags = aggregate(Y_df, spec)

    # Train/Test Splits
    if engine == 'pandas':
        Y_test_df = Y_df.groupby('unique_id').tail(8)
        Y_train_df = Y_df.drop(Y_test_df.index)
    elif engine == 'polars':
        Y_test_df = Y_df.group_by('unique_id').tail(8)
        Y_train_df = Y_df.filter(pl.col('ds') < Y_test_df['ds'].min())

    sf = StatsForecast(models=[AutoETS(season_length=4, model='ZZA')],
                       freq=freq, n_jobs=-1)
    Y_hat_df = sf.forecast(df=Y_train_df, h=8, fitted=True, level=level)
    Y_fitted_df = sf.forecast_fitted_values()

    # Base reconcilers
    reconcilers = [
                   BottomUp(),
                   MinTrace(method='ols'),
                   MinTrace(method='wls_struct'),
                   MinTrace(method='wls_var'),
                   MinTrace(method='mint_shrink'),
                   MinTrace(method='emint'),
                   OptimalCombination(method='ols'),
                   OptimalCombination(method='wls_struct'),
                   ERM(method='closed'),
    ]

    # Add reconcilers that handle strict hierarchies only
    if hierarchy == "strict":
        reconcilers += [
                TopDown(method="average_proportions"),
                TopDown(method="proportion_averages"),
                MinTrace(method='mint_cov'),
                MiddleOut(middle_level="Country/State", top_down_method="average_proportions"),
                MiddleOut(middle_level="Country/State", top_down_method="proportion_averages"),
                MiddleOutSparse(middle_level="Country/State", top_down_method="average_proportions"),
                MiddleOutSparse(middle_level="Country/State", top_down_method="proportion_averages"),
        ]
        if level is None:
            reconcilers += [
                    TopDown(method="forecast_proportions"),
            ]
            if engine == 'pandas':
                reconcilers += [
                    TopDownSparse(method="forecast_proportions"),
                    ]

    # Add sparse reconcilers only if using pandas engine
    if engine == 'pandas':
        if hierarchy == "strict":
            reconcilers += [
                    BottomUpSparse(),
                    TopDownSparse(method="average_proportions"),
                    TopDownSparse(method="proportion_averages"),
                    MinTraceSparse(method='ols'),
                    MinTraceSparse(method='wls_struct'),
                    MinTraceSparse(method='wls_var'),
            ]
        else:
            reconcilers += [
                    BottomUpSparse(),
                    MinTraceSparse(method='ols'),
                    MinTraceSparse(method='wls_struct'),
                    MinTraceSparse(method='wls_var'),
            ]

    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df,
                               Y_df=Y_fitted_df,
                               S_df=S_df,
                               tags=tags,
                               level=level)

    execution_times = pd.Series(hrec.execution_times).reset_index()

    if not os.path.exists('./data'):
        os.makedirs('./data')
    if engine == 'polars':
        Y_test_df = Y_test_df.to_pandas()
        Y_train_df = Y_train_df.to_pandas()
        Y_rec_df = Y_rec_df.to_pandas()
    Y_rec_df.to_csv('./data/Y_rec.csv', index=False)
    Y_test_df.to_csv('./data/Y_test.csv', index=False)
    Y_train_df.to_csv('./data/Y_train.csv', index=False)
    execution_times.to_csv('./data/execution_times.csv', index=False)
    with open('./data/tags.pickle', 'wb') as handle:
        pickle.dump(tags, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    fire.Fire(main)
