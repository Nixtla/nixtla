"""Tests for the top-level ``generate_series`` convenience function."""

import pandas as pd
import polars as pl
import pytest

from synforecast import generate_series
from synforecast.generators import RandomWalkGenerator, SeasonalGenerator


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_returns_long_format(engine):
    """Output is a long-format frame of the requested backend with the
    expected columns."""
    df = generate_series(n_series=6, engine=engine, seed=0)

    if engine == "pandas":
        assert isinstance(df, pd.DataFrame)
        cols = list(df.columns)
    else:
        assert isinstance(df, pl.DataFrame)
        cols = df.columns

    assert cols == ["unique_id", "ds", "y"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_produces_requested_number_of_series(engine):
    """Exactly ``n_series`` distinct ids are produced, spread across the pool."""
    n_series = 10
    df = generate_series(n_series=n_series, engine=engine, seed=0)

    if engine == "pandas":
        n_unique = df["unique_id"].nunique()
    else:
        n_unique = df["unique_id"].n_unique()

    assert n_unique == n_series


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_ids_are_contiguous_integers(engine):
    """Ids are integer categories 0..n_series-1 (matching utilsforecast)."""
    n_series = 8
    df = generate_series(n_series=n_series, engine=engine, seed=1)

    if engine == "pandas":
        ids = sorted(int(x) for x in df["unique_id"].unique())
    else:
        ids = sorted(int(x) for x in df["unique_id"].unique().to_list())

    assert ids == list(range(n_series))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_deterministic_for_fixed_seed(engine):
    """Same seed reproduces identical output; a different seed does not."""
    a = generate_series(n_series=5, engine=engine, seed=42)
    b = generate_series(n_series=5, engine=engine, seed=42)
    c = generate_series(n_series=5, engine=engine, seed=43)

    if engine == "pandas":
        pd.testing.assert_frame_equal(a, b)
        assert not a["y"].equals(c["y"])
    else:
        assert a.equals(b)
        assert not a["y"].equals(c["y"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_series_lengths_within_bounds(engine):
    """Each series length falls inside [min_length, max_length]."""
    min_length, max_length = 30, 60
    df = generate_series(
        n_series=12,
        min_length=min_length,
        max_length=max_length,
        engine=engine,
        seed=7,
    )

    if engine == "pandas":
        lengths = df.groupby("unique_id", observed=True).size().tolist()
    else:
        lengths = df.group_by("unique_id").len()["len"].to_list()

    assert all(min_length <= n <= max_length for n in lengths)


def test_more_series_than_generators():
    """Requesting more series than generators in the pool still yields the
    exact requested count (leftovers spread across the first generators)."""
    df = generate_series(n_series=57, engine="polars", seed=3)
    assert df["unique_id"].n_unique() == 57


def test_custom_generators_are_used():
    """A user-supplied generator list overrides the default pool."""
    gens = [
        RandomWalkGenerator(
            min_length=40, max_length=40, freq="D", engine="polars", seed=11
        ),
        SeasonalGenerator(
            min_length=40, max_length=40, freq="D", engine="polars", seed=12
        ),
    ]
    df = generate_series(n_series=4, generators=gens)

    assert df["unique_id"].n_unique() == 4
    # All series honor the custom fixed length.
    lengths = df.group_by("unique_id").len()["len"].to_list()
    assert all(n == 40 for n in lengths)


def test_integer_freq_index():
    """An integer frequency produces an integer time index."""
    df = generate_series(n_series=3, freq=1, engine="polars", seed=0)
    assert df["ds"].dtype in (pl.Int64, pl.Int32)
