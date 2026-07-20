"""Tests for start_id functionality in generators."""

import polars as pl

from synforecast.generators import RandomWalkGenerator, SeasonalGenerator


class TestStartId:
    """Tests for start_id parameter in generate()."""

    def test_default_start_id(self) -> None:
        """Test that default start_id is 0."""
        params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "h",
            "engine": "polars",
            "seed": 42,
        }

        generator = RandomWalkGenerator(**params)
        df = generator.generate(n_series=3)
        # Should have IDs 0, 1, and 2.
        series_ids = sorted(df["unique_id"].unique().to_list())
        expected_ids = ["0", "1", "2"]
        assert series_ids == expected_ids

    def test_custom_start_id(self) -> None:
        """Test that custom start_id works correctly."""
        params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "h",
            "engine": "polars",
            "seed": 42,
        }

        generator = RandomWalkGenerator(**params)
        df = generator.generate(n_series=3, start_id=10)
        # Should have IDs 10, 11, and 12.
        series_ids = sorted(df["unique_id"].unique().to_list())
        expected_ids = ["10", "11", "12"]
        assert series_ids == expected_ids

    def test_start_id_zero(self) -> None:
        """Test that start_id=0 works explicitly."""
        params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "h",
            "engine": "polars",
            "seed": 42,
        }

        generator = SeasonalGenerator(**params)
        df = generator.generate(n_series=2, start_id=0)
        series_ids = sorted(df["unique_id"].unique().to_list())
        expected_ids = ["0", "1"]
        assert series_ids == expected_ids

    def test_start_id_with_single_series(self) -> None:
        """Test start_id with single series generation."""
        params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "h",
            "engine": "polars",
            "seed": 42,
        }

        generator = RandomWalkGenerator(**params)
        df = generator.generate(n_series=1, start_id=5)
        series_ids = df["unique_id"].unique().to_list()
        assert series_ids == ["5"]

    def test_start_id_large_value(self) -> None:
        """Test start_id with a large value."""
        params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "h",
            "engine": "polars",
            "seed": 42,
        }

        generator = SeasonalGenerator(**params)
        df = generator.generate(n_series=2, start_id=1000)
        series_ids = sorted(df["unique_id"].unique().to_list())
        expected_ids = ["1000", "1001"]
        assert series_ids == expected_ids

    def test_sequential_generations_with_start_id(self) -> None:
        """Test multiple sequential generations with different start_ids."""
        params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "h",
            "engine": "polars",
            "seed": 42,
        }

        generator = RandomWalkGenerator(**params)

        # Generate first batch
        df1 = generator.generate(n_series=3, start_id=0)

        # Generate second batch
        df2 = generator.generate(n_series=2, start_id=3)

        # Combine both
        combined = pl.concat([df1, df2])

        # Should have IDs 0 through 4.
        series_ids = sorted(combined["unique_id"].unique().to_list())
        expected_ids = ["0", "1", "2", "3", "4"]
        assert series_ids == expected_ids
