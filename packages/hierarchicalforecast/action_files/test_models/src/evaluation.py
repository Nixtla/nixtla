import pickle
import numpy as np
import pandas as pd
from itertools import product

from hierarchicalforecast.evaluation import HierarchicalEvaluation


def rmse(y, y_hat):
    return np.mean(np.sqrt(np.mean((y-y_hat)**2, axis=1)))

def mase(y, y_hat, y_insample, seasonality=4):
    errors = np.mean(np.abs(y - y_hat), axis=1)
    scale = np.mean(np.abs(y_insample[:, seasonality:] - y_insample[:, :-seasonality]), axis=1)
    return np.mean(errors / scale)


def evaluate():
    execution_times = pd.read_csv(f'data/execution_times.csv')
    models = [f'{x[0]} ({x[1]:.2f} secs)' for x in execution_times.values]

    Y_rec_df = pd.read_csv('data/Y_rec.csv')
    Y_test_df = pd.read_csv('data/Y_test.csv')
    Y_train_df = pd.read_csv('data/Y_train.csv')

    Y_rec_df = Y_rec_df.set_index('unique_id')
    Y_test_df = Y_test_df.set_index('unique_id')
    Y_train_df = Y_train_df.set_index('unique_id')

    with open('data/tags.pickle', 'rb') as handle:
        tags = pickle.load(handle)

    eval_tags = {}
    eval_tags['Total'] = tags['Country']
    eval_tags['Purpose'] = tags['Country/Purpose']
    eval_tags['State'] = tags['Country/State']
    eval_tags['Regions'] = tags['Country/State/Region']
    eval_tags['Bottom'] = tags['Country/State/Region/Purpose']
    eval_tags['All'] = np.concatenate(list(tags.values()))

    evaluator = HierarchicalEvaluation(evaluators=[mase])
    evaluation = evaluator.evaluate(
            Y_hat_df=Y_rec_df, Y_test_df=Y_test_df,
            tags=eval_tags, Y_df=Y_train_df
    )
    evaluation = evaluation.drop('Overall')

    evaluation.columns = ['Base'] + models
    evaluation = evaluation.applymap('{:.2f}'.format)
    return evaluation


if __name__ == '__main__':
    evaluation = evaluate()
    evaluation.to_csv('./data/evaluation.csv')
    print(evaluation.T)
