# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import argparse
import json
import logging
from typing import Callable, Dict, List

#import hdays as hdays_part2
import holidays as hdays_part1
import numpy as np
import pandas as pd


def get_holiday_names(country):
    """Return all possible holiday names of given country.

    Parameters
    ----------
    country: country name

    Returns
    -------
    A set of all possible holiday names of given country
    """
    years = np.arange(1995, 2045)
    # try:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         holiday_names = getattr(hdays_part2, country)(years=years).values()
    # except AttributeError:
    try:
        holiday_names = getattr(hdays_part1, country)(years=years).values()
    except AttributeError as e:
        raise AttributeError(
            "Holidays in {} are not currently supported!".format(country)) from e

    return set(holiday_names)

def make_holidays_df(year_list, country, events, province=None, state=None):
    """Make dataframe of holidays for given years and countries

    Parameters
    ----------
    year_list: a list of years
    country: country name
    events: dict
        Dictionary, the key is the name event and the value is the date.

    Returns
    -------
    Dataframe with 'ds' and 'holiday', which can directly feed
    to 'holidays' params in Prophet
    """
    # try:
    #     holidays = getattr(hdays_part2, country)(years=year_list, expand=False)
    # except AttributeError:
    try:
        holidays = getattr(hdays_part1, country)(prov=province, state=state, years=year_list, expand=False)
    except AttributeError as e:
        raise AttributeError(
            "Holidays in {} are not currently supported!".format(country)) from e
    holidays_df = pd.DataFrame([(date, holidays.get_list(date)) for date in holidays], columns=['ds', 'holiday'])
    if events is not None:
        events_df = pd.DataFrame([(k, y) for k, v in events.items() for y in v], columns=['holiday', 'ds'])
        holidays_df = pd.concat([holidays_df, events_df])
    holidays_df = holidays_df.explode('holiday')
    holidays_df.reset_index(inplace=True, drop=True)
    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])

    return holidays_df

def distance_to_holiday(holiday_dates, dates):
    """Compute min distance between dates and holiday_dates

    Parameters
    ----------
    holiday_dates: a list of days where the holiday occurs
    dates: a range of days to compute the distance against

    Returns
    -------
    Numpy array with distance in days to the holiday_dates
    """

    # Get holidays around dates
    dates = pd.DatetimeIndex(dates)
    dates_np = np.array(dates)#.astype('datetime64[W]')
    #print(dates_np)

    #holiday_dates = get_holiday_dates(holiday, dates)
    holiday_dates_np = np.array(pd.DatetimeIndex(holiday_dates))#.astype('datetime64[W]')
    #print(holiday_dates_np)

    # Compute day distance to holiday
    distance = np.expand_dims(dates_np, axis=1) - np.expand_dims(holiday_dates_np, axis=0)
    distance = np.abs(distance)
    distance = np.min(distance, axis=1)
    distance[distance>183] = 365 - distance[distance>183]
    # Convert to minutes
    distance = distance.astype(float)
    distance /= 6e10
    distance = distance.astype(int)

    return distance

def make_holidays_distance_df(dates, year_list, country, events=None):
    """Make dataframe of distance in days to holidays
    for holiday dates and date range

    Parameters
    ----------
    holiday_dates: a list of days where the holiday occurs
    dates: a range of days to compute the distance against

    Returns
    -------
    Dataframe with 'ds' and 'holiday distance', referenced
    by the name of the holiday
    """
    holidays_df = make_holidays_df(year_list=year_list,
                                   country=country,
                                   events=events)

    distance_dict = {'ds': dates}
    for holiday in holidays_df.holiday.unique():
        holiday_dates = holidays_df[holidays_df.holiday==holiday]['ds']
        holiday_dates = holiday_dates.tolist()

        distance_dict[holiday] = distance_to_holiday(holiday_dates, dates)

    holidays_distance_df = pd.DataFrame(distance_dict)
    holidays_distance_df.set_index('ds', inplace=True)
    
    # Clean columns
    new_cols = holidays_distance_df.columns
    new_cols = new_cols.str.replace(' ', '_')
    new_cols = new_cols.str.replace("'|\.|\(|\)|-", '', regex=True)
    new_cols = new_cols.str.lower()
    holidays_distance_df.columns = new_cols

    return holidays_distance_df

class CalendarFeatures:
    """Computes calendar features.
    
    Notes
    -----
    [1] The argument events should be a str dict ('{'key':'value'}') 
    """

    def __init__(self, filename: str,
                 filename_output: str,
                 country: str,
                 events: Dict[str, List[str]],
                 scale: bool,
                 unique_id_column: str,
                 ds_column: str, y_column: str) -> 'CalendarFeatures':
        self.filename = filename
        self.filename_output = filename_output
        self.country = country
        self.events = events
        self.scale = scale
        self.unique_id_column = unique_id_column
        self.ds_column = ds_column
        self.y_column = y_column
        
        self.ext: str = self.filename.split('/')[-1].split('.')[-1]
        self.reader: Callable = getattr(pd, f'read_{self.ext}')
        self.writer: Callable = getattr(pd.DataFrame, f'to_{self.ext}')
        
        self.df: pd.DataFrame
        self.df = self._read_file()

    def _read_file(self) -> pd.DataFrame:
        logger.info('Reading file...')
        df = self.reader(f'/opt/ml/processing/input/{self.filename}')
        logger.info('File read.')
        renamer = {self.unique_id_column: 'unique_id',
                   self.ds_column: 'ds',
                   self.y_column: 'y'}

        df.rename(columns=renamer, inplace=True)

        df['ds'] = pd.to_datetime(df['ds'])

        return df

    def get_calendar_features(self) -> None:
        """Computes calendar features."""
        logger.info('Computing features...')
        dates = self.df['ds'].unique()
        holidays = make_holidays_distance_df(dates=dates,
                                             year_list=list(range(1990, 2025)),
                                             country=self.country,
                                             events=self.events)
        # hack, it should be an argument
        holidays = (holidays == 0).astype(int)
        holidays = holidays.dropna(axis=1, inplace=True)

        # remove duplicated columns
        holidays = holidays[holidays.index.isin(dates)]
        holidays = holidays.loc[:,~holidays.T.duplicated(keep='first')]

        # scale if requested
        if self.scale:
            holidays -= holidays.min(axis=0)
            holidays /= (holidays.max(axis=0) - holidays.min(axis=0)) 
            holidays = holidays.round(4)
        
        logger.info('Merging features...')
        features = self.df.set_index('ds').merge(holidays, 
                                                 how='left', 
                                                 left_on=['ds'],
                                                 right_index=True)
        features.reset_index(inplace=True)
        logger.info('Merging finished...')

        logger.info('Writing file...')
        self.writer(features, f'/opt/ml/processing/output/{self.filename_output}',
                    index=False)
        logger.info('File written...')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--country', type=str, required=True)
    parser.add_argument('--events', type=str,
                        metavar='KEY1=VALUE1/KEY2=VALUE2 or txt file', 
                        default=None)
    parser.add_argument('--scale', type=bool, default=False)
    parser.add_argument('--filename-output', type=str, default='calendar-features.parquet')
    parser.add_argument('--unique-id-column', type=str, default='unique_id')
    parser.add_argument('--ds-column', type=str, default='ds')
    parser.add_argument('--y-column', type=str, default='y')


    args = parser.parse_args()
    if args.events is not None and args.events.endswith('.txt'):
        events_file = f'/opt/ml/processing/input/{args.events}'
        with open(events_file) as f:
            args.events = f.readlines()[0]

    def parse_var(s):
        """
        Parse a key, value pair, separated by '='
        That's the reverse of ShellArgs.
        
        On the command line (argparse) a declaration will typically look like:
            foo=hello
        or
            foo="hello world"
        """
        items = s.split('=')
        key = items[0].strip() # we remove blanks around keys, as is logical
        if len(items) > 1:
            # rejoin the rest:
            value = items[1].split(',')
        return (key, value)

    def parse_vars(items):
        """
        Parse a series of key-value pairs and return a dictionary
        """
        d = {}

        if items:
            for item in items:
                key, value = parse_var(item)
                d[key] = value
        return d
      
    args.events = parse_vars(args.events.split('/')) if args.events not in [None, ''] else None

    calendarfeatures = CalendarFeatures(**vars(args))

    calendarfeatures.get_calendar_features()
