from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from xiuhmolpilli.arena import FoundationalTimeSeriesArena
from .utils import models
from .test_eval import generate_exp_dataset


def generate_data(freq: str, tmpdir: str) -> str:
    df = generate_exp_dataset(n_series=5, freq=freq, return_df=True)
    df_parquet_path = Path(tmpdir) / f"dataset_{freq}.parquet"
    df.to_parquet(df_parquet_path)
    return str(df_parquet_path)


def test_foundational_time_series_arena():
    cwd = Path.cwd()
    with TemporaryDirectory(dir=cwd) as tmpdir:
        parquet_data_paths = [generate_data(freq, tmpdir) for freq in ["H", "MS"]]
        arena = FoundationalTimeSeriesArena(
            models=models,
            parquet_data_paths=parquet_data_paths,
            results_dir=tmpdir,
        )
        arena.compete()
        eval_df = pd.read_csv(arena.evaluation_path)
        arena.compete()
        eval_df_2 = pd.read_csv(arena.evaluation_path)
        print(eval_df)
        print(eval_df_2)
        assert eval_df.equals(eval_df_2)
        print(eval_df)
