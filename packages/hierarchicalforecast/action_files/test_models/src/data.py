import pandas as pd

def get_tourism():
    # Read data
    Y_df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/tourism.csv')
    Y_df = Y_df.rename({'Trips': 'y', 'Quarter': 'ds'}, axis=1)
    Y_df.insert(0, 'Country', 'Australia')
    Y_df = Y_df[['Country', 'Region', 'State', 'Purpose', 'ds', 'y']]
    Y_df['ds'] = Y_df['ds'].str.replace(r'(\d+) (Q\d)', r'\1-\2', regex=True)
    Y_df['ds'] = pd.PeriodIndex(Y_df["ds"], freq='Q').to_timestamp()

    return Y_df
