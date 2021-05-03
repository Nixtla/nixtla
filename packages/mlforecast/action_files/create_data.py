import argparse
from pathlib import Path

from mlforecast.utils import generate_daily_series


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()

    series = generate_daily_series(20)
    if args.distributed:
        import dask.dataframe as dd
        series = dd.from_pandas(series, npartitions=2)
    data_path = Path('data')
    data_path.mkdir(exist_ok=True)
    series.to_parquet('data/train')
