"""Tests for parallel generation and n_jobs invariance."""

import narwhals.stable.v2 as nw
import numpy as np
import pytest

import synforecast.base as _base_mod
import synforecast.dataset as _dataset_mod
from synforecast import SynSet, balanced_pool
from synforecast.generators import (
    GARCHGenerator,
    RandomWalkGenerator,
    SARIMAGenerator,
)

BASE = {"min_length": 50, "max_length": 50, "freq": "D", "seed": 42}

PARAMS = pytest.mark.parametrize(
    "cls,extra",
    [
        (RandomWalkGenerator, {}),
        (SARIMAGenerator, {"p": 1, "q": 1}),
        (GARCHGenerator, {}),
    ],
    ids=["RandomWalk", "SARIMA", "GARCH"],
)

ENGINES = pytest.mark.parametrize("engine", ["pandas", "polars"])


def _col(df, name: str) -> np.ndarray:
    return nw.from_native(df)[name].to_numpy()


def _sorted_values(df) -> np.ndarray:
    nw_df = nw.from_native(df)
    nw_df = nw_df.with_columns(nw.col("unique_id").cast(nw.String()).cast(nw.Int64()))
    return nw_df.sort(["unique_id", "ds"])["y"].to_numpy()


class TestParallelGeneration:
    """Parallel generation produces valid results."""

    @PARAMS
    @ENGINES
    def test_produces_valid_output(self, cls, extra, engine):
        gen = cls(**{**BASE, **extra}, engine=engine)
        df = gen.generate(n_series=8)
        ids = _col(df, "unique_id")
        assert len(np.unique(ids)) == 8
        values = _col(df, "y")
        assert len(values) == 8 * 50
        assert np.isfinite(values).all()


class TestReproducibility:
    """Same seed gives identical output, independent of n_jobs."""

    @PARAMS
    def test_same_seed_identical(self, cls, extra):
        df1 = cls(**{**BASE, **extra}).generate(n_series=8)
        df2 = cls(**{**BASE, **extra}).generate(n_series=8)
        np.testing.assert_array_equal(_col(df1, "y"), _col(df2, "y"))

    @PARAMS
    def test_n_jobs_invariance(self, cls, extra):
        df1 = cls(**{**BASE, **extra}).generate(n_series=8, n_jobs=1)
        df2 = cls(**{**BASE, **extra}).generate(n_series=8, n_jobs=4)
        df3 = cls(**{**BASE, **extra}).generate(n_series=8)
        np.testing.assert_array_equal(_col(df1, "y"), _col(df2, "y"))
        np.testing.assert_array_equal(_col(df1, "y"), _col(df3, "y"))

    def test_synset_same_seed_identical(self):
        pools = [
            balanced_pool(min_length=30, max_length=30, freq="D", seed=7)[:4]
            for _ in range(2)
        ]
        df1 = SynSet(pools[0]).generate(n_series_per_generator=4)
        df2 = SynSet(pools[1]).generate(n_series_per_generator=4)
        np.testing.assert_array_equal(_sorted_values(df1), _sorted_values(df2))

    def test_synset_n_jobs_invariance(self):
        pools = [
            balanced_pool(min_length=30, max_length=30, freq="D", seed=7)[:4]
            for _ in range(2)
        ]
        df1 = SynSet(pools[0]).generate(n_series_per_generator=4, n_jobs=1)
        df2 = SynSet(pools[1]).generate(n_series_per_generator=4, n_jobs=4)
        np.testing.assert_array_equal(_sorted_values(df1), _sorted_values(df2))


class TestBatchFallback:
    """When the Rust batch module is unavailable, fall back to the threaded path."""

    @ENGINES
    def test_generator_falls_back_without_rust(self, monkeypatch, engine):
        monkeypatch.setattr(_base_mod, "_rs_batch", None)
        gen = RandomWalkGenerator(**BASE, engine=engine)
        df = gen.generate(n_series=6)
        assert len(np.unique(_col(df, "unique_id"))) == 6

    def test_fallback_reproducible(self, monkeypatch):
        monkeypatch.setattr(_base_mod, "_rs_batch", None)
        df1 = RandomWalkGenerator(**BASE).generate(n_series=6)
        df2 = RandomWalkGenerator(**BASE).generate(n_series=6)
        np.testing.assert_array_equal(_col(df1, "y"), _col(df2, "y"))

    def test_fallback_n_jobs_invariance(self, monkeypatch):
        monkeypatch.setattr(_base_mod, "_rs_batch", None)
        df1 = RandomWalkGenerator(**BASE).generate(n_series=6, n_jobs=1)
        df2 = RandomWalkGenerator(**BASE).generate(n_series=6, n_jobs=3)
        np.testing.assert_array_equal(_col(df1, "y"), _col(df2, "y"))

    def test_synset_falls_back_without_rust(self, monkeypatch):
        monkeypatch.setattr(_base_mod, "_rs_batch", None)
        monkeypatch.setattr(_dataset_mod, "_rs_batch", None)
        generators = balanced_pool(min_length=30, max_length=30, freq="D", seed=42)
        df = SynSet(generators[:3]).generate(n_series_per_generator=3)
        assert len(np.unique(_col(df, "unique_id"))) == 9
