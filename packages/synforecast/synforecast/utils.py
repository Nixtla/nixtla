"""Utilities for generating synthetic time series datasets."""

from __future__ import annotations

import narwhals.stable.v2 as nw
from narwhals.stable.v2.typing import IntoDataFrameT

from synforecast.base import BaseGenerator, _categorize_ids
from synforecast.presets import balanced_pool

__all__ = ["generate_series"]


def generate_series(
    n_series: int,
    freq: str | int = "D",
    min_length: int = 50,
    max_length: int = 500,
    generators: list[BaseGenerator] | None = None,
    engine: str = "pandas",
    seed: int = 0,
) -> IntoDataFrameT:
    """Generate a synthetic panel of time series.

    Series are drawn from a balanced pool of generators covering diverse
    temporal behaviors (or from `generators` when provided) and returned in
    long format, mirroring `utilsforecast.data.generate_series`.

    Args:
        n_series (int): Number of series to generate.
        freq (str | int): Frequency of the data, as a pandas offset alias
            (e.g. 'D', 'h', '5min', 'MS') or an integer for an integer time
            index. Defaults to 'D'.
        min_length (int): Minimum length of each series. Defaults to 50.
        max_length (int): Maximum length of each series. Defaults to 500.
        generators (list[BaseGenerator], optional): Generators to draw from.
            Defaults to `synforecast.balanced_pool`. Ignores min_length /
            max_length / freq / engine / seed when provided.
        engine (str): Output dataframe library. Defaults to 'pandas'.
        seed (int): Random seed. Defaults to 0.

    Returns:
        DataFrame in long format with columns [`unique_id`, `ds`, `y`].
    """
    if generators is None:
        generators = balanced_pool(
            min_length=min_length,
            max_length=max_length,
            freq=freq,
            seed=seed,
            engine=engine,
        )

    # Spread the requested series across the pool
    n_gens = len(generators)
    counts = [n_series // n_gens] * n_gens
    for i in range(n_series % n_gens):
        counts[i] += 1

    dfs = []
    start_id = 0
    for gen, count in zip(generators, counts, strict=True):
        if count == 0:
            continue
        df = gen.generate(n_series=count, start_id=start_id)
        dfs.append(
            nw.from_native(df).with_columns(
                nw.col(gen.id_col).cast(nw.String()).cast(nw.Int64())
            )
        )
        start_id += count

    id_col = generators[0].id_col
    return _categorize_ids(nw.concat(dfs), id_col).to_native()
