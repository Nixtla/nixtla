import os
import fire
import pickle
import pandas as pd

from statsforecast.models import ETS
from statsforecast.core import StatsForecast

from hierarchicalforecast.utils import aggregate


def get_data():
    # If data exists read it
    if os.path.isfile('data/Y_test.csv'):
        Y_test_df = pd.read_csv('data/Y_test.csv')
        Y_train_df = pd.read_csv('data/Y_train.csv')
        Y_hat_df = pd.read_csv('data/Y_hat.csv')
        Y_fitted_df = pd.read_csv('data/Y_fitted.csv')
        S_df = pd.read_csv('data/S.csv')

        with open('data/tags.pickle', 'rb') as handle:
            tags = pickle.load(handle)
        
        return Y_train_df, Y_test_df, Y_hat_df, Y_fitted_df, S_df, tags

    # Read and Parse Data
    Y_df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/tourism.csv')
    Y_df = Y_df.rename({'Trips': 'y', 'Quarter': 'ds'}, axis=1)
    Y_df.insert(0, 'Country', 'Australia')
    Y_df = Y_df[['Country', 'Region', 'State', 'Purpose', 'ds', 'y']]
    Y_df['ds'] = Y_df['ds'].str.replace(r'(\d+) (Q\d)', r'\1-\2', regex=True)
    Y_df['ds'] = pd.to_datetime(Y_df['ds'])

    # Hierarchical Aggregation
    spec = [
        ['Country'],
        ['Country', 'State'], 
        ['Country', 'Purpose'], 
        ['Country', 'State', 'Region'], 
        ['Country', 'State', 'Purpose'], 
        ['Country', 'State', 'Region', 'Purpose']
    ]

    Y_df, S_df, tags = aggregate(Y_df, spec)
    Y_df = Y_df.reset_index()

    # Train/Test Splits
    Y_test_df = Y_df.groupby('unique_id').tail(8)
    Y_train_df = Y_df.drop(Y_test_df.index)

    Y_test_df = Y_test_df.set_index('unique_id')
    Y_train_df = Y_train_df.set_index('unique_id')

    sf = StatsForecast(df=Y_train_df,
                       models=[ETS(season_length=4, model='ZZA')],
                       freq='QS', n_jobs=-1)
    Y_hat_df = sf.forecast(h=8, fitted=True)
    Y_fitted_df = sf.forecast_fitted_values()
    
    Y_test_df = Y_test_df.reset_index()
    Y_train_df = Y_train_df.reset_index()

    # Save Data
    Y_test_df.to_csv('./data/Y_test.csv', index=False)
    Y_train_df.to_csv('./data/Y_train.csv', index=False)

    Y_hat_df.to_csv('./data/Y_hat.csv', index=False)
    Y_fitted_df.to_csv('./data/Y_fitted.csv', index=False)
    S_df.to_csv('./data/S.csv', index=False)

    with open('./data/tags.pickle', 'wb') as handle:
        pickle.dump(tags, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    return Y_train_df, Y_test_df, Y_hat_df, Y_fitted_df, S_df, tags

def save_data():
    Y_train_df, Y_test_df, Y_hat_df, Y_fitted_df, S_df, tags = get_data()

    Y_test_df.to_csv('./data/Y_test.csv', index=False)
    Y_train_df.to_csv('./data/Y_train.csv', index=False)

    Y_hat_df.to_csv('./data/Y_hat.csv', index=False)
    Y_fitted_df.to_csv('./data/Y_fitted.csv', index=False)
    S_df.to_csv('./data/S.csv', index=False)

    with open('./data/tags.pickle', 'wb') as handle:
        pickle.dump(tags, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    fire.Fire(save_data)
