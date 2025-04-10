import pickle
import numpy as np
import pandas as pd

import hierarchicalforecast.evaluation as hfe
from utilsforecast.losses import mae, rmse

def evaluate():
    execution_times = pd.read_csv('data/execution_times.csv')
    models = [f"{x[0]} ({x[1]:.2f} secs)" for x in execution_times.values]

    Y_rec_df = pd.read_csv('data/Y_rec.csv')
    Y_test_df = pd.read_csv('data/Y_test.csv')

    with open('data/tags.pickle', 'rb') as handle:
        tags = pickle.load(handle)

    Y_hat_df=Y_rec_df.merge(Y_test_df, on=["ds", "unique_id", "temporal_id"], how="left")
    
    evaluation = hfe.evaluate(
            df=Y_hat_df.drop(columns="unique_id"),
            tags=tags, 
            metrics=[mae, rmse],
            id_col='temporal_id'
    )
    numeric_cols = evaluation.select_dtypes(include="number").columns
    evaluation[numeric_cols] = evaluation[numeric_cols].map('{:.3}'.format).astype(np.float64)
    evaluation.columns = ['level', 'metric', 'Base'] + models
    return evaluation

if __name__ == '__main__':
    evaluation = evaluate()
    evaluation.to_csv('./data/evaluation.csv')
    print(evaluation.T)
