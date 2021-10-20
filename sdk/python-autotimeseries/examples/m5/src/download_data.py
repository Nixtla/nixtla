import argparse
import logging

import pandas as pd
from pathlib import Path
from nixtlats.data.datasets.m5 import M5


def main(directory: str, output_format: str = 'parquet') -> None:
    """
    Downloads M5 data and splits it in train/test sets.

    Parameters
    ----------
    directory: str
        Directory where data will be stored.
    """
    horizon = 28
    logger.info('Loading data')
    Y, X, S = M5.load(directory)
    renamer = {'unique_id': 'item_id', 
               'ds': 'timestamp', 
               'y': 'demand',
               'item_id': 'sku_id'}
    Y.rename(renamer, axis=1, inplace=True)
    logger.info(Y.head())
    X.rename(renamer, axis=1, inplace=True)
    X = X.loc[:, ~X.columns.str.startswith('event')]
    logger.info(X.head())
    S.rename(renamer, axis=1, inplace=True)
    logger.info(S.head())

    def split_train_test(df, horizon):
        df_test = df.groupby('item_id').tail(horizon).copy()
        df_train = df.drop(df_test.index)
        
        return df_train, df_test
    
    logger.info('Splitting data')
    Y_train, Y_test = split_train_test(Y, horizon)

    dir_path = Path(directory)
    train_path = dir_path / 'm5' / output_format / 'train'
    test_path = dir_path / 'm5' / output_format / 'test'

    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
    # defining writer
    writer = getattr(pd.DataFrame, f'to_{output_format}')
    logger.info('Saving training data')
    writer(Y_train, train_path / f'target.{output_format}', index=False)
    writer(X, train_path / f'temporal.{output_format}', index=False)
    writer(S, train_path / f'static.{output_format}', index=False)
    
    logger.info('Saving testing data')
    writer(Y_test, test_path / f'target.{output_format}', index=False)
    writer(Y_test.drop('demand', axis=1), test_path / f'target_dates.{output_format}', index=False)

    logger.info('Processing calendar')
    calendar = pd.read_csv(f'{directory}/m5/datasets/calendar.csv')
    dfs = []
    for col in ['event_name_1', 'event_name_2']:
        df = calendar[['date', col]].dropna().rename(columns={col: 'event'}) 
        dfs.append(df)
    dfs = pd.concat(dfs).sort_values('event')
    events = dfs['event'].unique()

    events_str = []
    for event in events:
        dates = dfs.query(f'event==@event')['date'].unique()
        dates = ','.join(dates)
        str_event = event.replace(' ', '_')
        str_event = str_event.replace('-', '')
        str_event = str_event.replace('\'', '')
        str_event = f'{str_event}={dates}'
        events_str.append(str_event)
    
    events_str = '/'.join(events_str)

    logger.info('Saving calendar')
    with open(train_path / 'calendar-holidays.txt', 'w') as f:
        f.write(events_str)
        

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str)
    parser.add_argument('--output-format', type=str, default='parquet')

    args = parser.parse_args()
    
    main(args.directory, args.output_format)


