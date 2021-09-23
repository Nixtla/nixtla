import argparse
import logging

import pandas as pd
from mlforecast.core import TimeSeries
from mlforecast.forecast import Forecast
from multiprocessing import cpu_count
from tsfresh.feature_extraction import extract_features, EfficientFCParameters
from tsfeatures import tsfeatures, acf_features
from tsfeatures.tsfeatures_r import tsfeatures_r
from window_ops.rolling import rolling_mean


class TSFeatures:
    """Computes static or temporal features."""

    def __init__(self, freq: int, filename: str,
                 unique_id_column: str,
                 ds_column: str, y_column: str,
                 kind: str) -> 'TSFeatures':
        self.freq = freq
        self.filename = filename
        self.unique_id_column = unique_id_column
        self.ds_column = ds_column
        self.y_column = y_column
        self.kind = kind
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

        if self.kind == 'temporal':
            df['ds'] = pd.to_datetime(df['ds'])

        return df

    def get_static_features(self) -> pd.DataFrame:
        """Computes static features."""
        logger.info('Computing features...')
        features = tsfeatures(self.df, freq=self.freq, features=[acf_features])
        features = features.set_index('unique_id')

        logger.info('Computing extra features...')
        features_fresh = extract_features(self.df, column_id='unique_id',
                                          column_sort='ds',
                                          column_value='y',
                                          n_jobs=cpu_count() - 1,
                                          default_fc_parameters=EfficientFCParameters())
        features_fresh = features_fresh.rename_axis('unique_id')
        features_fresh = features_fresh.fillna('')

        logger.info('Merging features...')
        features = features.join(features_fresh).reset_index()
        logger.info('Merging finished...')

        return features

    def get_temporal_features(self) -> pd.DataFrame:
        """Calculates temporal features."""
        ts = TimeSeries(
            freq='D',
            lags=list(range(self.freq, 2 * self.freq + 1)),
            lag_transforms = {
               self.freq:  [(rolling_mean, self.freq), (rolling_mean, 2 * self.freq)],
               2 * self.freq: [(rolling_mean, self.freq), (rolling_mean, 2 * self.freq)],
            },
            date_features=['year', 'month', 'day', 'dayofweek', 'quarter', 'week'],
            num_threads=None
        )

        fcst = Forecast(None, ts)
        features = fcst.preprocess(self.df.set_index('unique_id'))
        features = features.reset_index()

        return features

    def get_features(self) -> None:
        """Calculates features based on kind."""
        if self.kind == 'static':
            features = self.get_static_features()
        elif self.kind == 'temporal':
            features = self.get_temporal_features()
        else:
            raise ValueError('Kink must be either "static" or "temporal".')

        logger.info('Writing file...')
        features.to_csv(f'/opt/ml/processing/output/{self.kind}-features.csv',
                        index=False)
        logger.info('File written...')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument('--freq', type=int, required=True)
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--unique-id-column', type=str, default='unique_id')
    parser.add_argument('--ds-column', type=str, default='ds')
    parser.add_argument('--y-column', type=str, default='y')
    parser.add_argument('--kind', type=str, default='static')


    args = parser.parse_args()

    fasttsfeatures = TSFeatures(**vars(args))

    fasttsfeatures.get_features()
