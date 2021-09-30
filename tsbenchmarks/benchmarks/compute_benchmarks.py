import argparse
import logging

import pandas as pd
#from evaluate import evaluate_my_model
#from losses import mae

#################################################################
# Cell
from math import sqrt
from typing import Optional, Union

import numpy as np

# Cell
def divide_no_nan(a, b):
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    div[div != div] = 0.0
    div[div == float('inf')] = 0.0
    return div

# Cell
def metric_protections(y: np.ndarray, y_hat: np.ndarray, weights: np.ndarray):
    assert (weights is None) or (np.sum(weights) > 0), 'Sum of weights cannot be 0'
    assert (weights is None) or (weights.shape == y_hat.shape), 'Wrong weight dimension'

# Cell
def smape(y: np.ndarray, y_hat: np.ndarray,
          weights: Optional[np.ndarray] = None,
          axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculates Symmetric Mean Absolute Percentage Error.

    SMAPE measures the relative prediction accuracy of a
    forecasting method by calculating the relative deviation
    of the prediction and the true value scaled by the sum of the
    absolute values for the prediction and true value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 200% which is desireble compared to normal MAPE that
    may be undetermined.

    Parameters
    ----------
    y: numpy array
        Actual test values.
    y_hat: numpy array
        Predicted values.
    weights: numpy array
        Weights for weighted average.
    axis: None or int, optional
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the
        elements of the input array.
        If axis is negative it counts
        from the last to the first axis.

    Returns
    -------
    smape: numpy array or double
        Return the smape along the specified axis.
    """
    metric_protections(y, y_hat, weights)

    delta_y = np.abs(y - y_hat)
    scale = np.abs(y) + np.abs(y_hat)
    smape = divide_no_nan(delta_y, scale)
    smape = 200 * np.average(smape, weights=weights, axis=axis)

    if isinstance(smape, float):
        assert smape <= 200, 'SMAPE should be lower than 200'
    else:
        assert all(smape <= 200), 'SMAPE should be lower than 200'

    return smape

# Cell
def mase(y: np.ndarray, y_hat: np.ndarray,
         y_train: np.ndarray,
         seasonality: int,
         weights: Optional[np.ndarray] = None,
         axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculates the Mean Absolute Scaled Error.

    MASE measures the relative prediction accuracy of a
    forecasting method by comparinng the mean absolute errors
    of the prediction and the true value against the mean
    absolute errors of the seasonal naive model.

    Parameters
    ----------
    y: numpy array
        Actual test values.
    y_hat: numpy array
        Predicted values.
    y_train: numpy array
        Actual insample values for Seasonal Naive predictions.
    seasonality: int
        Main frequency of the time series
        Hourly 24,  Daily 7, Weekly 52,
        Monthly 12, Quarterly 4, Yearly 1.
    weights: numpy array
        Weights for weighted average.
    axis: None or int, optional
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the
        elements of the input array.
        If axis is negative it counts
        from the last to the first axis.

    Returns
    -------
    mase: numpy array or double
        Return the mase along the specified axis.

    References
    ----------
    [1] https://robjhyndman.com/papers/mase.pdf
    """
    delta_y = np.abs(y - y_hat)
    delta_y = np.average(delta_y, weights=weights, axis=axis)

    scale = np.abs(y_train[:-seasonality] - y_train[seasonality:])
    scale = np.average(scale, axis=axis)

    mase = delta_y / scale

    return mase

#################################################################
import logging
import requests
import subprocess
import zipfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cell
def download_file(directory: str, source_url: str, decompress: bool = False) -> None:
    """Download data from source_ulr inside directory.

    Parameters
    ----------
    directory: str, Path
        Custom directory where data will be downloaded.
    source_url: str
        URL where data is hosted.
    decompress: bool
        Wheter decompress downloaded file. Default False.
    """
    if isinstance(directory, str):
        directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    filename = source_url.split('/')[-1]
    filepath = Path(f'{directory}/{filename}')

    # Streaming, so we can iterate over the response.
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(source_url, stream=True, headers=headers)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte

    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filepath, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
            f.flush()
    t.close()

    if total_size != 0 and t.n != total_size:
        logger.error('ERROR, something went wrong downloading data')

    size = filepath.stat().st_size
    logger.info(f'Successfully downloaded {filename}, {size}, bytes.')

    if decompress:
        if '.zip' in filepath.suffix:
            logger.info('Decompressing zip file...')
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(directory)
        else:
            from patoolib import extract_archive
            extract_archive(filepath, outdir=directory)
        logger.info(f'Successfully decompressed {filepath}')

# Cell
@dataclass
class Info:
    """
    Info Dataclass of datasets.
    Args:
        groups (Tuple): Tuple of str groups
        class_groups (Tuple): Tuple of dataclasses.
    """
    groups: Tuple[str]
    class_groups: Tuple[dataclass]

    def get_group(self, group: str):
        """Gets dataclass of group."""
        if group not in self.groups:
            raise Exception(f'Unkown group {group}')

        return self.class_groups[self.groups.index(group)]

    def __getitem__(self, group: str):
        """Gets dataclass of group."""
        if group not in self.groups:
            raise Exception(f'Unkown group {group}')

        return self.class_groups[self.groups.index(group)]

    def __iter__(self):
        for group in self.groups:
            yield group, self.get_group(group)

#################################################################
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# from utils import download_file, Info
# from losses import smape, mase

# Cell
@dataclass
class Yearly:
    seasonality: int = 1
    horizon: int = 6
    freq: str = 'Y'
    name: str = 'Yearly'
    n_ts: int = 23_000

@dataclass
class Quarterly:
    seasonality: int = 4
    horizon: int = 8
    freq: str = 'Q'
    name: str = 'Quarterly'
    n_ts: int = 24_000

@dataclass
class Monthly:
    seasonality: int = 12
    horizon: int = 18
    freq: str = 'M'
    name: str = 'Monthly'
    n_ts: int = 48_000

@dataclass
class Weekly:
    seasonality: int = 1
    horizon: int = 13
    freq: str = 'W'
    name: str = 'Weekly'
    n_ts: int = 359

@dataclass
class Daily:
    seasonality: int = 1
    horizon: int = 14
    freq: str = 'D'
    name: str = 'Daily'
    n_ts: int = 4_227

@dataclass
class Hourly:
    seasonality: int = 24
    horizon: int = 48
    freq: str = 'H'
    name: str = 'Hourly'
    n_ts: int = 414


@dataclass
class Other:
    seasonality: int = 1
    horizon: int = 8
    freq: str = 'D'
    name: str = 'Other'
    n_ts: int = 5_000
    included_groups: Tuple = ('Weekly', 'Daily', 'Hourly')

# Cell
M4Info = Info(groups=('Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly', 'Other'),
              class_groups=(Yearly, Quarterly, Monthly, Weekly, Daily, Hourly, Other))

# Cell
@dataclass
class M4:

    source_url: str = 'https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/'
    naive2_forecast_url: str = 'https://github.com/Nixtla/m4-forecasts/raw/master/forecasts/submission-Naive2.zip'

    @staticmethod
    def load(directory: str,
             group: str,
             cache: bool = True) -> Tuple[pd.DataFrame,
                                          Optional[pd.DataFrame],
                                          Optional[pd.DataFrame]]:
        """Downloads and loads M4 data.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        group: str
            Group name.
            Allowed groups: 'Yearly', 'Quarterly', 'Monthly',
                            'Weekly', 'Daily', 'Hourly'.
        cache: bool
            If `True` saves and loads

        Notes
        -----
        [1] Returns train+test sets.
        """
        path = f'{directory}/m4/datasets'
        file_cache = f'{path}/{group}.p'

        if os.path.exists(file_cache) and cache:
            df_train, df_test, X_df, S_df = pd.read_pickle(file_cache)

            return df_train, df_test, X_df, S_df

        if group == 'Other':
            #Special case.
            included_dfs = [M4.load(directory, gr) \
                            for gr in M4Info['Other'].included_groups]
            df, *_ = zip(*included_dfs)
            df = pd.concat(df)
        else:
            M4.download(directory, group)
            path = f'{directory}/m4/datasets'
            class_group = M4Info[group]
            S_df = pd.read_csv(f'{directory}/m4/datasets/M4-info.csv',
                               usecols=['M4id','category'])
            S_df['category'] = S_df['category'].astype('category').cat.codes
            S_df.rename({'M4id': 'unique_id'}, axis=1, inplace=True)
            S_df = S_df[S_df['unique_id'].str.startswith(class_group.name[0])]

            def read_and_melt(file):
                df = pd.read_csv(file)
                df.columns = ['unique_id'] + list(range(1, df.shape[1]))
                df = pd.melt(df, id_vars=['unique_id'], var_name='ds', value_name='y')
                df = df.dropna()

                return df

            df_train = read_and_melt(file=f'{path}/{group}-train.csv')
            df_test = read_and_melt(file=f'{path}/{group}-test.csv')

            # len_train = df_train.groupby('unique_id').agg({'ds': 'max'}).reset_index()
            # len_train.columns = ['unique_id', 'len_serie']
            # df_test = df_test.merge(len_train, on=['unique_id'])
            # df_test['ds'] = df_test['ds'] + df_test['len_serie']
            # df_test.drop('len_serie', axis=1, inplace=True)

            # df = pd.concat([df_train, df_test])
            # df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

            S_df = S_df.sort_values('unique_id').reset_index(drop=True)

        X_df = None
        if cache:
            pd.to_pickle((df_train, df_test, X_df, S_df), file_cache)

        return df_train, df_test, None, S_df

    @staticmethod
    def download(directory: str, group: str) -> None:
        """Download M4 Dataset."""
        path = f'{directory}/m4/datasets/'
        if not os.path.exists(path):
            #for group in M4Info.groups:
            download_file(path, f'{M4.source_url}/Train/{group}-train.csv')
            download_file(path, f'{M4.source_url}/Test/{group}-test.csv')
            download_file(path, f'{M4.source_url}/M4-info.csv')
            download_file(path, M4.naive2_forecast_url, decompress=True)

# Cell
class M4Evaluation:

    @staticmethod
    def load_benchmark(directory: str, group: str,
                       source_url: Optional[str] = None) -> np.ndarray:
        """Downloads and loads a bechmark forecasts.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        group: str
            Group name.
            Allowed groups: 'Yearly', 'Quarterly', 'Monthly',
                            'Weekly', 'Daily', 'Hourly'.
        source_url: str, optional
            Optional benchmark url obtained from
            https://github.com/Nixtla/m4-forecasts/tree/master/forecasts.
            If `None` returns Naive2.

        Returns
        -------
        benchmark: numpy array
            Numpy array of shape (n_series, horizon).
        """
        path = f'{directory}/m4/datasets'
        initial = group[0]
        if source_url is not None:
            filename = source_url.split('/')[-1].replace('.rar', '.csv')
            filepath = f'{path}/{filename}'
            if not os.path.exists(filepath):
                download_file(path, source_url, decompress=True)

        else:
            filepath = f'{path}/submission-Naive2.csv'

        benchmark = pd.read_csv(filepath)
        benchmark = benchmark[benchmark['id'].str.startswith(initial)]
        benchmark = benchmark.set_index('id').dropna(1)
        benchmark = benchmark.sort_values('id').values

        return benchmark

    @staticmethod
    def evaluate(directory: str, group: str,
                 y_hat: Union[np.ndarray, str]) -> pd.DataFrame:
        """Evaluates y_hat according to M4 methodology.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        group: str
            Group name.
            Allowed groups: 'Yearly', 'Quarterly', 'Monthly',
                            'Weekly', 'Daily', 'Hourly'.
        y_hat: numpy array, str
            Group forecasts as numpy array or
            benchmark url from
            https://github.com/Nixtla/m4-forecasts/tree/master/forecasts.

        Returns
        -------
        evaluation: pandas dataframe
            DataFrame with columns OWA, SMAPE, MASE
            and group as index.
        """
        if isinstance(y_hat, str):
            y_hat = M4Evaluation.load_benchmark(directory, group, y_hat)

        initial = group[0]
        class_group = M4Info[group]
        horizon = class_group.horizon
        seasonality = class_group.seasonality
        path = f'{directory}/m4/datasets'

        y_hat = y_hat['y'].values.reshape(-1, horizon)

        df_train, df_test, _, S_df = M4.load(directory='data', group=group)

        y_train = df_train.groupby('unique_id')['y']
        y_train = y_train.apply(lambda x: x.values)
        y_train = y_train.values

        y_test = df_test['y'].values.reshape(-1, horizon)

        naive2 = M4Evaluation.load_benchmark(directory, group)
        smape_y_hat = smape(y_test, y_hat)
        smape_naive2 = smape(y_test, naive2)

        mases_y_hat = [mase(y_test[i], y_hat[i], y_train[i], seasonality)
                              for i in range(class_group.n_ts)]
       
        mase_y_hat = np.mean(mases_y_hat)
        mase_naive2 = np.mean([mase(y_test[i], naive2[i], y_train[i], seasonality)
                               for i in range(class_group.n_ts)])

        owa = .5 * (mase_y_hat / mase_naive2 + smape_y_hat / smape_naive2)

        evaluation = pd.DataFrame({'SMAPE': smape_y_hat,
                                   'MASE': mase_y_hat,
                                   'OWA': owa},
                                   index=[group])

        smape_ts = smape(y_test, y_hat, axis=0)

        return evaluation, mases_y_hat, smape_ts

#################################################################

# Cell
@dataclass
class M5:

    # original data available from Kaggle directly
    # pip install kaggle --upgrade
    # kaggle competitions download -c m5-forecasting-accuracy
    source_url: str = 'https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip'

    @staticmethod
    def download(directory: str) -> None:
        """Downloads M5 Competition Dataset."""
        path = f'{directory}/m5/datasets'
        if not os.path.exists(path):
            download_file(directory=path,
                          source_url=M5.source_url,
                          decompress=True)

    @staticmethod
    def load(directory: str, cache: bool = True) -> Tuple[pd.DataFrame,
                                                          pd.DataFrame,
                                                          pd.DataFrame]:
        """Downloads and loads M5 data.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        cache: bool
            If `True` saves and loads.

        Notes
        -----
        [1] Returns train+test sets.
        [2] Based on https://www.kaggle.com/lemuz90/m5-preprocess.
        """
        path = f'{directory}/m5/datasets'
        file_cache = f'{path}/m5.p'

        if os.path.exists(file_cache) and cache:
            Y_df, X_df, S_df = pd.read_pickle(file_cache)

            return Y_df, X_df, S_df

        M5.download(directory)
        # Calendar data
        cal_dtypes = {
            'wm_yr_wk': np.uint16,
            'event_name_1': 'category',
            'event_type_1': 'category',
            'event_name_2': 'category',
            'event_type_2': 'category',
            'snap_CA': np.uint8,
            'snap_TX': np.uint8,
            'snap_WI': np.uint8,
        }
        cal = pd.read_csv(f'{path}/calendar.csv',
                          dtype=cal_dtypes,
                          usecols=list(cal_dtypes.keys()) + ['date'],
                          parse_dates=['date'])
        cal['d'] = np.arange(cal.shape[0]) + 1
        cal['d'] = 'd_' + cal['d'].astype('str')
        cal['d'] = cal['d'].astype('category')

        event_cols = [k for k in cal_dtypes if k.startswith('event')]
        for col in event_cols:
            cal[col] = cal[col].cat.add_categories('nan').fillna('nan')

        # Prices
        prices_dtypes = {
            'store_id': 'category',
            'item_id': 'category',
            'wm_yr_wk': np.uint16,
            'sell_price': np.float32
        }

        prices = pd.read_csv(f'{path}/sell_prices.csv',
                             dtype=prices_dtypes)

        # Sales
        sales_dtypes = {
            'item_id': prices.item_id.dtype,
            'dept_id': 'category',
            'cat_id': 'category',
            'store_id': 'category',
            'state_id': 'category',
            **{f'd_{i+1}': np.float32 for i in range(1969)}
        }
        # Reading train and test sets
        sales_train = pd.read_csv(f'{path}/sales_train_evaluation.csv',
                                  dtype=sales_dtypes)
        sales_test = pd.read_csv(f'{path}/sales_test_evaluation.csv',
                                 dtype=sales_dtypes)
        sales = sales_train.merge(sales_test, how='left',
                                  on=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
        sales['id'] = sales[['item_id', 'store_id']].astype(str).agg('_'.join, axis=1).astype('category')
        # Long format
        long = sales.melt(id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                          var_name='d', value_name='y')
        long['d'] = long['d'].astype(cal.d.dtype)
        long = long.merge(cal, on=['d'])
        long = long.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'])
        long = long.drop(columns=['d', 'wm_yr_wk'])

        def first_nz_mask(values, index):
            """Return a boolean mask where the True starts at the first non-zero value."""
            mask = np.full(values.size, True)
            for idx, value in enumerate(values):
                if value == 0:
                    mask[idx] = False
                else:
                    break
            return mask

        long = long.sort_values(['id', 'date'], ignore_index=True)
        keep_mask = long.groupby('id')['y'].transform(first_nz_mask, engine='numba')
        long = long[keep_mask.astype(bool)]
        long.rename(columns={'id': 'unique_id', 'date': 'ds'}, inplace=True)
        Y_df = long.filter(items=['unique_id', 'ds', 'y'])
        cats = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        S_df = long.filter(items=['unique_id'] + cats)
        S_df = S_df.drop_duplicates(ignore_index=True)
        X_df = long.drop(columns=['y'] + cats)

        if cache:
            pd.to_pickle((Y_df, X_df, S_df), file_cache)

        return Y_df, X_df, S_df


# Cell
class M5Evaluation:

    levels: dict =  dict(
        Level1=['total'],
        Level2=['state_id'],
        Level3=['store_id'],
        Level4=['cat_id'],
        Level5=['dept_id'],
        Level6=['state_id', 'cat_id'],
        Level7=['state_id', 'dept_id'],
        Level8=['store_id', 'cat_id'],
        Level9=['store_id', 'dept_id'],
        Level10=['item_id'],
        Level11=['state_id', 'item_id'],
        Level12=['item_id', 'store_id']
    )

    @staticmethod
    def aggregate_levels(y_hat: pd.DataFrame,
                         categories: pd.DataFrame = None) -> pd.DataFrame:
        """Aggregates the 30_480 series to get 42_840."""
        y_hat_cat = y_hat.assign(total='Total')

        df_agg = []
        for level, agg in M5Evaluation.levels.items():
            df = y_hat_cat.groupby(agg).sum().reset_index()
            renamer = dict(zip(agg, ['Agg_Level_1', 'Agg_Level_2']))
            df.rename(columns=renamer, inplace=True)
            df.insert(0, 'Level_id', level)
            df_agg.append(df)
        df_agg = pd.concat(df_agg)
        df_agg = df_agg.fillna('X')
        df_agg = df_agg.set_index(['Level_id', 'Agg_Level_1', 'Agg_Level_2'])
        df_agg.columns = [f'd_{i+1}' for i in range(df_agg.shape[1])]

        return df_agg

    @staticmethod
    def evaluate(directory: str,
                 y_hat: pd.DataFrame) -> pd.DataFrame:
        """Evaluates y_hat according to M4 methodology.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        y_hat: pandas datafrae, str
            Forecasts as wide pandas dataframe with columns
            ['unique_id'] and forecasts or
            benchmark url from
            https://github.com/Nixtla/m5-forecasts/tree/main/forecasts.

        Returns
        -------
        evaluation: pandas dataframe
            DataFrame with columns OWA, SMAPE, MASE
            and group as index.
        """
        print('Downlading M5 data')
        M5.download(directory)

        # Merge with s_df
        y_hat = y_hat.set_index(['unique_id', 'ds']).unstack()
        y_hat = y_hat.droplevel(0, 1).reset_index()
        *_, s_df = M5.load(directory)
        y_hat = y_hat.merge(s_df, how='left', on=['unique_id'])

        path = f'{directory}/m5/datasets'
 
        weights = pd.read_csv(f'{path}/weights_evaluation.csv')
        sales = pd.read_csv(f'{path}/sales_train_evaluation.csv')
        y_test = pd.read_csv(f'{path}/sales_test_evaluation.csv')

        # sales
        sales = M5Evaluation.aggregate_levels(sales)
        def scale(x):
            x = x.values
            x = x[np.argmax(x!=0):]
            scale = ((x[1:] - x[:-1]) ** 2).mean()
            return scale
        scales = sales.agg(scale, 1).rename('scale').reset_index()

        # y_test
        y_test = M5Evaluation.aggregate_levels(y_test)

        #y_hat
        y_hat = M5Evaluation.aggregate_levels(y_hat)

        score = (y_test - y_hat) ** 2
        score = score.mean(1)
        score = score.rename('rmse').reset_index()
        score = score.merge(weights, how='left',
                            on=['Level_id', 'Agg_Level_1', 'Agg_Level_2'])
        score = score.merge(scales, how='left',
                            on=['Level_id', 'Agg_Level_1', 'Agg_Level_2'])
        score['wrmsse'] = (score['rmse'] / score['scale']).pow(1 / 2) * score['weight']
        score = score.groupby('Level_id')[['wrmsse']].sum()
        score = score.loc[M5Evaluation.levels.keys()]
        total = score.mean().rename('Total').to_frame().T
        score = pd.concat([total, score])

        return score

#################################################################
import matplotlib.pyplot as plt

#from data import M4Evaluation

def loss_benchmark(metrics, model_losses, losses_dict, root_dir):
    for metric in metrics:
        loss_dict = losses_dict[metric]
        loss_dict['YourModel'] = model_losses[metric]
        final_dict = dict(sorted(loss_dict.items(), key=lambda item: item[1]))

        colors = len(final_dict)*['#6e8e9e']
        colors[list(final_dict.keys()).index('YourModel')] = '#e1ad9b'

        plt.figure(figsize=(10,5))
        plt.bar(final_dict.keys(), final_dict.values(), color=colors)
        plt.xlabel('Model')
        plt.ylabel(metric)
        plt.grid()
        plt.savefig(root_dir+f'{metric}.pdf')

def loss_per_serie(losses, root_dir):
    plt.figure(figsize=(15,7))
    plt.hist(losses, bins=50, color='#6e8e9e')
    plt.grid()
    plt.ylabel('Number of series')
    plt.xlabel('LOSS')
    plt.savefig(root_dir+'loss_per_serie.pdf')

def loss_per_timestamp(losses, root_dir):
    plt.figure(figsize=(15,7))
    plt.plot(losses)
    plt.xlabel('H')
    plt.ylabel('SMAPE')
    plt.grid()
    plt.savefig(root_dir+'smape_per_timestamp.pdf')


def evaluate_M4(dataset, forecasts, root_dir):
    METRICS = {'sMAPE', 'MASE'}
    LOSSES_DICT = {
                    'M4-Daily':{'sMAPE':{'ESRNN':3.170,
                                            'FFORMA':3.097,
                                            'Theta':3.053,
                                            'ARIMA': 3.193,
                                            'ETS': 3.046,
                                            'Naive1': 3.045,
                                            'Naive2': 3.045,
                                            'RNN': 5.964,
                                            'MLP': 9.321},
                                'MASE':{'ESRNN': 3.446,
                                            'FFORMA': 3.344,
                                            'Theta': 3.262,
                                            'ARIMA': 3.410,
                                            'ETS': 3.253,
                                            'Naive1': 3.278,
                                            'Naive2': 3.278,
                                            'RNN': 6.232,
                                            'MLP': 12.973}},
                    'M4-Yearly':{'sMAPE':{'NBEATS': 13.114,
                                            'ESRNN': 13.176,
                                            'FFORMA': 13.528,
                                            'Theta': 14.593,
                                            'ARIMA': 15.168,
                                            'ETS': 15.356,
                                            'Naive1': 16.342,
                                            'Naive2': 16.342,
                                            'RNN': 22.398,
                                            'MLP': 21.764},
                                'MASE':{'ESRNN': 2.980,
                                            'FFORMA': 3.060,
                                            'Theta': 3.382,
                                            'ARIMA': 3.402,
                                            'ETS': 3.444,
                                            'Naive1': 3.974,
                                            'Naive2': 3.974,
                                            'RNN': 4.946,
                                            'MLP': 4.946}},
                    'M4-Quarterly':{'sMAPE':{'NBEATS': 9.154,
                                            'ESRNN': 9.679,
                                            'FFORMA': 9.733,
                                            'Theta': 10.311,
                                            'ARIMA': 10.431,
                                            'ETS': 10.291,
                                            'Naive1': 11.610,
                                            'Naive2': 11.012,
                                            'RNN': 17.027,
                                            'MLP': 18.500},
                                    'MASE':{'ESRNN': 1.118,
                                            'FFORMA': 1.111,
                                            'Theta': 1.232,
                                            'ARIMA': 1.165,
                                            'ETS': 1.161,
                                            'Naive1': 1.477,
                                            'Naive2': 1.371,
                                            'RNN': 2.016,
                                            'MLP': 2.314}},
                    'M4-Monthly':{'sMAPE':{'NBEATS': 12.041,
                                            'ESRNN': 12.126,
                                            'FFORMA': 12.639,
                                            'Theta': 13.002,
                                            'ARIMA': 13.443,
                                            'ETS': 13.525,
                                            'Naive1': 15.256,
                                            'Naive2': 14.427,
                                            'RNN': 24.056,
                                            'MLP': 24.333},
                                'MASE':{'ESRNN': 0.884,
                                            'FFORMA': 0.893,
                                            'Theta': 0.970,
                                            'ARIMA': 0.930,
                                            'ETS': 0.948,
                                            'Naive1': 1.205,
                                            'Naive2': 1.063,
                                            'RNN': 1.601,
                                            'MLP': 1.925}},
                    'M4-Weekly':{'sMAPE':{'ESRNN': 7.817,
                                            'FFORMA': 7.625,
                                            'Theta': 9.093,
                                            'ARIMA': 8.653,
                                            'ETS': 8.727,
                                            'Naive1': 9.161,
                                            'Naive2': 9.161,
                                            'RNN': 15.220,
                                            'MLP': 21.349},
                                'MASE':{'ESRNN': 2.356, 
                                            'FFORMA': 2.108,
                                            'Theta': 2.637,
                                            'ARIMA': 2.556,
                                            'ETS': 2.527,
                                            'Naive1': 2.777,
                                            'Naive2': 2.777,
                                            'RNN': 5.132,
                                            'MLP': 13.568}},
                    'M4-Hourly':{'sMAPE':{'ESRNN': 9.328,
                                            'FFORMA': 11.506,
                                            'Theta': 18.138,
                                            'ARIMA': 13.980,
                                            'ETS': 17.307,
                                            'Naive1': 43.003,
                                            'Naive2': 18.383,
                                            'RNN': 14.698,
                                            'MLP': 13.842},
                                'MASE':{'ESRNN': 0.893,
                                            'FFORMA': 0.819,
                                            'Theta': 2.455,
                                            'ARIMA': 0.943,
                                            'ETS': 1.824,
                                            'Naive1': 11.608,
                                            'Naive2': 2.395,
                                            'RNN': 3.048,
                                            'MLP': 2.607}}
                  }

    group = dataset.split('-')[1]
    losses_dict = LOSSES_DICT[dataset]
    print('DATASET: M4 dataset, GROUP: ', group)
    evaluation, mases_y_hat, smape_ts = M4Evaluation.evaluate(y_hat=forecasts, directory='./data', group=group)
    model_losses = {'sMAPE': evaluation['SMAPE'][0], 'MASE': evaluation['MASE'][0]}

    # Bar plot comparing models
    loss_benchmark(metrics= METRICS, model_losses=model_losses, losses_dict=losses_dict, root_dir=root_dir)

    # Histogram of MASE by time serie
    loss_per_serie(losses=mases_y_hat, root_dir=root_dir)

    # SMAPE by horizon
    loss_per_timestamp(losses=smape_ts, root_dir=root_dir)


def evaluate_M5(forecasts, root_dir):
    METRICS = {'WRMSSE'}
    LOSSES_DICT = {
                   'WRMSSE':{'Naive': 1.752,
                             'sNaive': 0.847,
                             'MLP': 0.977,
                             'RF': 1.010,
                             'YJ_STU_1st': 0.520,
                             'Matthias_2nd': 0.528,
                             'mf_3rd': 0.536,
                             'Random_prediction_50th': 0.576,}
                  }

    wrmsse = M5Evaluation.evaluate(y_hat=forecasts, directory='./data')

    model_losses = {'WRMSSE': wrmsse['wrmsse'].values[0]}

    # Bar plot comparing models
    loss_benchmark(metrics= METRICS, model_losses=model_losses, losses_dict=LOSSES_DICT, root_dir=root_dir)
    
    LOSSES_DICT['WRMSSE']['YourModel'] = model_losses['WRMSSE']

    losses_df = pd.DataFrame.from_dict(LOSSES_DICT).rename_axis('model').reset_index()
    losses_df.sort_values(by='WRMSSE', inplace=True)

    return losses_df

    
def evaluate_my_model(forecasts, dataset, root_dir, train_time=None):
    assert dataset in ['M4-Daily', 'M4-Yearly', 'M4-Quarterly', 'M4-Monthly', 'M4-Weekly', 'M4-Hourly', 'M5'], 'Dataset {} not supported!'.format(dataset)

    if 'M4' in dataset:
        evaluate_M4(dataset=dataset, forecasts=forecasts, root_dir=root_dir)

    if 'M5' in dataset:
        output = evaluate_M5(forecasts=forecasts, root_dir=root_dir)
        
        return output


###############################################################################

class TSBenchmarks:
    """Benchmarks"""

    def __init__(self, dataset: str,
                 filename: str,
                 unique_id_column: str,
                 ds_column: str, y_column: str) -> 'TSBenchmarks':
        self.dataset = dataset
        self.filename = filename
        self.unique_id_column = unique_id_column
        self.ds_column = ds_column
        self.y_column = y_column
        self.df: pd.DataFrame

        self.df = self._read_file()

    def _read_file(self) -> pd.DataFrame:
        logger.info('Reading file...')
        df = pd.read_csv(f'/opt/ml/processing/input/{self.filename}')
        logger.info('File readed.')
        renamer = {self.unique_id_column: 'unique_id',
                   self.ds_column: 'ds',
                   self.y_column: 'y'}

        df.rename(columns=renamer, inplace=True)

        return df

    def benchmark(self) -> None:
        """ Compute metrics """
        logger.info('Computing metrics...')
        root_dir = '/opt/ml/processing/output/'
        model_metrics = evaluate_my_model(forecasts=self.df, dataset=self.dataset, root_dir=root_dir,train_time=None)

        model_metrics.to_csv(root_dir+'benchmarks.csv', index=False)
        logger.info('File written...')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--unique-id-column', type=str, default='unique_id')
    parser.add_argument('--ds-column', type=str, default='ds')
    parser.add_argument('--y-column', type=str, default='y')

    args = parser.parse_args()

    benchmarking = TSBenchmarks(**vars(args))

    benchmarking.benchmark()
