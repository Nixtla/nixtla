"""Utilities for generating synthetic time series datasets."""

from __future__ import annotations

from typing import Any, cast

import narwhals.stable.v2 as nw
from narwhals.stable.v2.typing import IntoDataFrame

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
    with_generator_col: bool = False,
) -> IntoDataFrame:
    """Generate a synthetic panel of time series.

    Series are drawn from a balanced pool of generators covering diverse
    temporal behaviors (or from `generators` when provided) and returned in
    long format, mirroring `utilsforecast.data.generate_series`.

    Series are spread evenly across the generator list from the front, so
    when ``n_series`` is smaller than the pool only the first ``n_series``
    generators contribute. The default pool is ordered round-robin across
    its behavioral niches, so a small panel still spans distinct behaviors:
    the first 15 generators cover all 15 niches.

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
        with_generator_col (bool): When True, add a `generator` column with
            the alias of the generator that produced each series.
            Defaults to False.

    Returns:
        DataFrame in long format with columns [`unique_id`, `ds`, `y`],
        plus `generator` when `with_generator_col` is True.
    """
    if n_series < 1:
        raise ValueError(f"n_series must be a positive integer, got {n_series}.")

    if generators is None:
        generators = balanced_pool(
            min_length=min_length,
            max_length=max_length,
            freq=freq,
            seed=seed,
            engine=engine,
        )
    elif not generators:
        raise ValueError("generators must be a non-empty list of generators.")

    # Spread the requested series across the pool
    n_gens = len(generators)
    counts = [n_series // n_gens] * n_gens
    for i in range(n_series % n_gens):
        counts[i] += 1

    dfs: list[nw.DataFrame[Any]] = []
    start_id = 0
    for gen, count in zip(generators, counts, strict=True):
        if count == 0:
            continue
        df: IntoDataFrame = gen.generate(n_series=count, start_id=start_id)
        frame = nw.from_native(df).with_columns(
            nw.col(gen.id_col).cast(nw.String()).cast(nw.Int64())
        )
        if with_generator_col:
            if "generator" in frame.columns:
                raise ValueError(
                    "with_generator_col=True reserves the output column "
                    f"'generator', but {type(gen).__name__} already produces "
                    "that column. Rename the conflicting generator column or "
                    "set with_generator_col=False."
                )
            frame = frame.with_columns(
                nw.lit(gen.alias or type(gen).__name__).alias("generator")
            )
        dfs.append(frame)
        start_id += count

    id_col = generators[0].id_col
    # narwhals' concat stubs return the non-stable DataFrame class; the
    # runtime object is the stable one the inputs are.
    combined = cast("nw.DataFrame[Any]", nw.concat(dfs))
    return _categorize_ids(combined, id_col).to_native()
