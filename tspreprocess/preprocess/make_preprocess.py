import argparse
import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def check_nans(df):
    """ Auxiliary function for data wrangling logs """
    n_rows = len(df)
    
    check_df = {'col': [], 'dtype': [], 'nan_prc': []}
    for col in df.columns:
        check_df['col'].append(col)
        check_df['dtype'].append(df[col].dtype)
        check_df['nan_prc'].append(df[col].isna().sum()/n_rows)
    
    check_df = pd.DataFrame(check_df)
    print("\n")
    print(f"dataframe n_rows {n_rows}")
    print(check_df)
    print("\n")

    
def one_hot_encoding(df, unique_id): 
    """
    Creates a new dataframe with one hot encoded variables
    Arguments
    ---------
    df: Pandas DataFrame
        Original data with categorical or object columns to encode
    unique_id: str
        String index that identifies each unique observation in df
    
    Returns
    -------
    one_hot_concat_df: Pandas DataFrame
        Processed data with one hot encoded variables.
        The column names identify each category and its levels.
         i.e.  category_[level1], category_[level2] ...
    """
    encoder = OneHotEncoder()
    columns = list(df.columns)
    columns.remove(unique_id)
    one_hot_concat_df = pd.DataFrame(df[unique_id].values, columns=[unique_id])
    for col in columns:
        dummy_columns = [f'{col}_[{x}]' for x in list(df[col].unique())]
        dummy_values  = encoder.fit_transform(df[col].values.reshape(-1,1)).toarray()
        one_hot_df    = pd.DataFrame(dummy_values, columns=dummy_columns)        
        one_hot_concat_df = pd.concat([one_hot_concat_df, one_hot_df], axis=1)
    return one_hot_concat_df

def numpy_balance(*arrs):
    N = len(arrs)
    out =  np.transpose(np.meshgrid(*arrs, indexing='ij'),
                        np.roll(np.arange(N + 1), -1)).reshape(-1, N)
    return out

def numpy_ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out

def numpy_bfill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), mask.shape[1] - 1)
    idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out

def temporal_preprocessing(temporal, unique_id, ds, 
                           zfill_cols=[], ffill_cols=[], 
                           original_date_range=True):
    """
    Creates a new panel dataframe with balanced temporal data.
    Arguments
    ---------
    temporal: Pandas DataFrame
        Original data with temporal observations to process
    unique_id: str
        String index that identifies each unique trajectory in df
    ds: str
        String index that identifies dates for each trajectory.
    zfill_cols: str list
        String list with columns to be filled with zeros after balance.
    ffill_cols: str list
        String list with columns to be filled with forward fill ('ffill')
        after balance.
    original_date_range: bool
        Boolean with option to filter the trajectories to original date range.
        If false completely balanced panel is returned.
    
    Returns
    -------
    balanced_df: Pandas DataFrame
        Preprocessed data balanced and filled nans.
    """
    df = temporal.copy()
    UIDS  = temporal[unique_id].unique()
    DATES = temporal[ds].unique()
    
    # Balance panel (fixed UIDS)
    balanced_prod = numpy_balance(UIDS, DATES)
    balanced_df   = pd.DataFrame(balanced_prod, columns=[unique_id, ds])
    balanced_df[ds] = balanced_df[ds].astype(DATES.dtype)

    df.set_index([unique_id, ds], inplace=True)
    balanced_df.set_index([unique_id, ds], inplace=True)
    balanced_df = balanced_df.merge(df, how='left', 
                                    left_on=[unique_id, ds],
                                    right_index=True).reset_index()
    
    # FFill and ZFill
    for col in zfill_cols:
        balanced_df[col] = balanced_df[col].fillna(0)

    for col in ffill_cols:
        col_values = balanced_df[col].astype('float32').values        
        col_values = col_values.reshape(len(UIDS), len(DATES))
        col_values = numpy_ffill(col_values)
        #col_values = numpy_bfill(col_values)
        balanced_df[col] = col_values.flatten()
        #balanced_df[col] = balanced_df[col].fillna(0)

    # Match original date interval through filter
    if original_date_range:
        date_range_df = temporal.groupby(unique_id).agg({ds: ['min', 'max']})
        date_range_df = date_range_df.droplevel(1, axis=1).reset_index()
        date_range_df.columns = [unique_id, 'min_ds', 'max_ds']

        date_range_df.set_index([unique_id], inplace=True)
        balanced_df.set_index([unique_id], inplace=True)

        balanced_df = balanced_df.merge(date_range_df, how='left', 
                                        left_on=[unique_id],
                                        right_index=True).reset_index()
        balanced_df['ds_in_range'] = (balanced_df[ds] >= balanced_df['min_ds']) & \
                                     (balanced_df[ds] <= balanced_df['max_ds'])
        balanced_df = balanced_df[balanced_df['ds_in_range']]
        balanced_df.drop(['ds_in_range', 'min_ds', 'max_ds'], axis=1, inplace=True) 

    return balanced_df


class TSPreprocess:
    """Preprocess time series."""

    def __init__(self, filename: str,
                 filename_output: str,
                 kind: str,
                 unique_id_column: str,
                 ds_column: str, y_column: str) -> 'TSPreprocess':
        self.filename = filename
        if filename_output is None:
            self.filename_output = f'{kind}-preprocessed.csv'
        else:
            self.filename_output = filename_output
        self.kind = kind
        self.unique_id_column = unique_id_column
        self.ds_column = ds_column
        self.y_column = y_column
        self.df: pd.DataFrame

        self.df = self._read_file()

    def _read_file(self) -> pd.DataFrame:
        logger.info('Reading file...')
        df = pd.read_csv(f'/opt/ml/processing/input/{self.filename}')
        logger.info('File read.')
        renamer = {self.unique_id_column: 'unique_id',
                   self.ds_column: 'ds',
                   self.y_column: 'y'}

        df.rename(columns=renamer, inplace=True)
        
        if self.kind == 'balance':
            df['ds'] = pd.to_datetime(df['ds'])

        return df

    def get_one_hot_encoded(self) -> None:
        """One-hot encodes."""
        logger.info('One hot encoding...')
        df_ohe = one_hot_encoding(self.df, 'unique_id')
        logger.info('OHE finished.')
        
        logger.info('Writing file...')
        df_ohe.to_csv(f'/opt/ml/processing/output/{self.filename_output}',
                        index=False)
        logger.info('File written...')
    
    def get_balanced_data(self) -> None:
        """Balance data."""
        logger.info('Balancing data...')
        balanced_data = temporal_preprocessing(self.df, 'unique_id', 'ds', 
                                               ['y'])
        logger.info('Finished.')
        
        logger.info('Writing file...')
        balanced_data.to_csv(f'/opt/ml/processing/output/{self.filename_output}',
                             index=False)
        logger.info('File written...')

    def preprocess(self) -> None:
        if self.kind == 'onehot':
            self.get_one_hot_encoded()
        elif self.kind == 'balance':
            self.get_balanced_data()
        else:
            raise ValueError(
                '`kind` only accepts "onehot" or "balance"'
                ' as input'
            )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--kind', type=str, required=True,
                        help='onehot or balance')
    parser.add_argument('--filename-output', type=str, default=None)
    parser.add_argument('--unique-id-column', type=str, default='unique_id')
    parser.add_argument('--ds-column', type=str, default='ds')
    parser.add_argument('--y-column', type=str, default='y')

    args = parser.parse_args()
    tspreprocess = TSPreprocess(**vars(args))

    tspreprocess.preprocess()
