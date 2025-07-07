import fire
import pickle
import numpy as np
import pandas as pd

import hierarchicalforecast.evaluation as hfe
from utilsforecast.losses import rmse, mase,scaled_crps
from functools import partial

def eval(type: str = "point") -> pd.DataFrame:
    mase_p = partial(mase, seasonality=4)
    if type == "probabilistic":
        level = [80, 90]
        metrics = [rmse, mase_p, scaled_crps]
    elif type == "point":
        level = None
        metrics = [rmse, mase_p]        
    else:
        raise ValueError("Type must be either 'point' or 'probabilistic'.")
    execution_times = pd.read_csv('data/execution_times.csv')
    models = [f"{x[0]} ({x[1]:.2f} secs)" for x in execution_times.values]

    Y_rec_df = pd.read_csv('data/Y_rec.csv')
    Y_test_df = pd.read_csv('data/Y_test.csv')
    Y_train_df = pd.read_csv('data/Y_train.csv')

    with open('data/tags.pickle', 'rb') as handle:
        tags = pickle.load(handle)

    evaluation = hfe.evaluate(
            df=Y_rec_df.merge(Y_test_df, on=['unique_id', 'ds'], how="left"),
            metrics = metrics,
            level=level,
            tags=tags, 
            train_df=Y_train_df
    )
    numeric_cols = evaluation.select_dtypes(include="number").columns
    evaluation[numeric_cols] = evaluation[numeric_cols].map('{:.3}'.format).astype(np.float64)
    evaluation.columns = ['level', 'metric', 'Base'] + models
    print(evaluation.T)
    evaluation.to_csv('./data/evaluation.csv')

if __name__ == '__main__':
    fire.Fire(eval)