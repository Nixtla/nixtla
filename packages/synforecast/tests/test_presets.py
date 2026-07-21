"""Tests for balanced_pool preset."""

import polars as pl

from synforecast import SynSet, balanced_pool
from synforecast.base import BaseGenerator


class TestBalancedPool:
    """Tests for the balanced_pool function."""

    def test_returns_list_of_generators(self) -> None:
        """Test that balanced_pool returns a list of BaseGenerator instances."""
        generators = balanced_pool()
        assert isinstance(generators, list)
        assert len(generators) == 42
        for gen in generators:
            assert isinstance(gen, BaseGenerator)

    def test_custom_length_and_freq(self) -> None:
        """Test that custom base parameters are applied to all generators."""
        generators = balanced_pool(
            min_length=50, max_length=75, freq="h", engine="polars"
        )
        for gen in generators:
            assert gen.min_length == 50
            assert gen.max_length == 75
            assert gen.freq == "h"

    def test_custom_seed(self) -> None:
        """Test that seeds are offset per generator."""
        generators = balanced_pool(seed=100)
        seeds = [gen.seed for gen in generators]
        assert seeds[0] == 100
        assert seeds[1] == 101
        assert len(set(seeds)) == len(seeds)  # all unique

    def test_none_seed(self) -> None:
        """Test that seed=None produces None seeds for all generators."""
        generators = balanced_pool(seed=None)
        for gen in generators:
            assert gen.seed is None

    def test_base_kwargs_passed(self) -> None:
        """Test that extra kwargs are forwarded to all generators."""
        generators = balanced_pool(engine="pandas")
        for gen in generators:
            assert gen.engine == "pandas"

    def test_with_synset(self) -> None:
        """Test integration with SynSet."""
        generators = balanced_pool(
            min_length=30, max_length=30, freq="D", engine="polars"
        )
        dataset = SynSet(generators)
        df = dataset.generate(n_series_per_generator=1)
        assert isinstance(df, pl.DataFrame)
        n_series = df["unique_id"].n_unique()
        assert n_series == 42

    def test_all_generators_produce_output(self) -> None:
        """Test that every generator in the pool produces valid time series."""
        generators = balanced_pool(
            min_length=50, max_length=50, freq="D", engine="polars"
        )
        for i, gen in enumerate(generators):
            df = gen.generate(n_series=1)
            assert isinstance(df, pl.DataFrame), (
                f"Generator {i} ({gen.alias}) did not return a DataFrame"
            )
            assert len(df) > 0, f"Generator {i} ({gen.alias}) returned empty DataFrame"

    def test_reproducibility(self) -> None:
        """Test that the same seed produces identical results."""
        gen1 = balanced_pool(
            min_length=50, max_length=50, freq="D", engine="polars", seed=42
        )
        gen2 = balanced_pool(
            min_length=50, max_length=50, freq="D", engine="polars", seed=42
        )

        df1 = gen1[0].generate(n_series=1)
        df2 = gen2[0].generate(n_series=1)

        assert df1["y"].to_list() == df2["y"].to_list()

    def test_diverse_generator_types(self) -> None:
        """Test that the pool contains generators from multiple classes."""
        generators = balanced_pool()
        class_names = {type(gen).__name__ for gen in generators}
        assert len(class_names) == 15
        expected = {
            "SARIMAGenerator",
            "ETSGenerator",
            "FractionalBrownianMotionGenerator",
            "RegimeSwitchingGenerator",
            "GARCHGenerator",
            "CyclicGenerator",
            "IntermittentDemandGenerator",
            "EnergyLoadGenerator",
            "IoTSensorGenerator",
            "VitalSignsGenerator",
            "GaussianProcessGenerator",
            "ChaoticSystemGenerator",
            "INARGenerator",
            "BoundedProcessGenerator",
            "LevyProcessGenerator",
        }
        assert class_names == expected

    def test_balanced_pool_with_patterns(self) -> None:
        """Test balanced_pool with anomalies, changepoints, and missing data."""
        generators = balanced_pool(
            min_length=100,
            max_length=100,
            freq="D",
            engine="polars",
            seed=42,
            anomalies=True,
            changepoints=True,
            missing_data=True,
        )
        assert len(generators) > 0
        for gen in generators:
            df = gen.generate(n_series=1)
            assert isinstance(df, pl.DataFrame)
            assert len(df) > 0


class TestPretrainingPool:
    """Tests for the pretraining_pool function."""

    META = {"TSIGenerator", "TCMGenerator", "KernelSynthGenerator"}

    def test_includes_meta_and_balanced_by_default(self) -> None:
        from synforecast import pretraining_pool

        pool = pretraining_pool(min_length=64, max_length=64, engine="polars")
        classes = {type(g).__name__ for g in pool}
        # Meta-generators that balanced_pool excludes are present...
        assert classes >= self.META
        # ...alongside the full balanced pool.
        assert len(pool) == 42 + 3 * 3

    def test_meta_only(self) -> None:
        from synforecast import pretraining_pool

        pool = pretraining_pool(
            include_balanced=False, n_meta_variants=2, min_length=64,
            max_length=64, engine="polars",
        )
        assert len(pool) == 6
        assert {type(g).__name__ for g in pool} == self.META

    def test_all_seeds_unique(self) -> None:
        from synforecast import pretraining_pool

        seeds = [g.seed for g in pretraining_pool(seed=42)]
        assert len(set(seeds)) == len(seeds)

    def test_none_seed(self) -> None:
        from synforecast import pretraining_pool

        assert all(g.seed is None for g in pretraining_pool(seed=None))

    def test_rejects_zero_variants(self) -> None:
        import pytest

        from synforecast import pretraining_pool

        with pytest.raises(ValueError):
            pretraining_pool(n_meta_variants=0)

    def test_with_synset(self) -> None:
        from synforecast import SynSet, pretraining_pool

        pool = pretraining_pool(
            min_length=48, max_length=48, freq="D", engine="polars", seed=1
        )
        df = SynSet(pool).generate(n_series_per_generator=1)
        assert isinstance(df, pl.DataFrame)
        assert df["unique_id"].n_unique() == len(pool)
        assert len(df) > 0
