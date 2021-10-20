import pandas as pd
from nixtlats.data.datasets.m5 import M5, M5Evaluation


if __name__=='__main__':
    forecast = pd.read_csv('prophet-forecasts-m5.csv')
    forecast = forecast.set_index(['unique_id', 'ds']).unstack()
    forecast = forecast.droplevel(0, 1).reset_index()
    
    directory = 'data/'
    *_, s_df = M5.load(directory)
    forecast = forecast.merge(s_df, how='left', on=['unique_id'])
    evaluation = M5Evaluation.evaluate(directory, forecast)
    
    print(evaluation)

