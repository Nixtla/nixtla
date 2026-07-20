"""Tests for SynSet dataset class."""

import logging

import pandas as pd
import polars as pl
import pytest

from synforecast.dataset import SynSet
from synforecast.generators import RandomWalkGenerator, SeasonalGenerator


class TestSynSet:
    """Tests for SynSet class."""

    def test_basic_generation(self) -> None:
        """Test basic dataset generation with multiple generators."""
        # Create two generators
        rw_params = {
            "min_length": 50,
            "max_length": 100,
            "freq": "h",
            "engine": "polars",
            "seed": 42,
        }
        seasonal_params = {
            "min_length": 50,
            "max_length": 100,
            "freq": "h",
            "engine": "polars",
            "seed": 43,
        }

        rw_gen = RandomWalkGenerator(**rw_params)
        seasonal_gen = SeasonalGenerator(**seasonal_params)

        # Create dataset
        dataset = SynSet([rw_gen, seasonal_gen])
        df = dataset.generate(n_series_per_generator=3)
        # Check DataFrame structure
        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"unique_id", "ds", "y"}

        # Check data types
        assert df["unique_id"].dtype == pl.Categorical
        assert df["ds"].dtype == pl.Datetime
        assert df["y"].dtype == pl.Float64

        # Check number of series (3 from each generator = 6 total)
        n_series = df["unique_id"].n_unique()
        assert n_series == 6

    def test_unique_series_ids(self) -> None:
        """Test that series IDs are unique across all generators."""
        rw_params = {
            "min_length": 100,
            "max_length": 100,
            "freq": "h",
            "engine": "polars",
            "seed": 42,
        }
        seasonal_params = {
            "min_length": 100,
            "max_length": 100,
            "freq": "h",
            "engine": "polars",
            "seed": 43,
        }

        rw_gen = RandomWalkGenerator(**rw_params)
        seasonal_gen = SeasonalGenerator(**seasonal_params)

        dataset = SynSet([rw_gen, seasonal_gen])
        df = dataset.generate(n_series_per_generator=2)
        # Get unique series IDs
        series_ids = sorted(df["unique_id"].unique().to_list())

        # Should have IDs 0, 1, 2, and 3.
        expected_ids = ["0", "1", "2", "3"]
        assert series_ids == expected_ids

    def test_single_generator(self) -> None:
        """Test dataset with a single generator."""
        rw_params = {
            "min_length": 50,
            "max_length": 100,
            "freq": "h",
            "engine": "polars",
            "seed": 42,
        }

        rw_gen = RandomWalkGenerator(**rw_params)
        dataset = SynSet([rw_gen])
        df = dataset.generate(n_series_per_generator=5)
        # Should have 5 series
        n_series = df["unique_id"].n_unique()
        assert n_series == 5

    def test_multiple_generators(self) -> None:
        """Test dataset with more than two generators."""
        params1 = {
            "min_length": 50,
            "max_length": 100,
            "freq": "h",
            "seed": 42,
            "engine": "polars",
        }
        params2 = {
            "min_length": 50,
            "max_length": 100,
            "freq": "h",
            "seed": 43,
            "engine": "polars",
        }
        params3 = {
            "min_length": 50,
            "max_length": 100,
            "freq": "h",
            "seed": 44,
            "engine": "polars",
        }

        gen1 = RandomWalkGenerator(**params1)
        gen2 = SeasonalGenerator(**params2)
        gen3 = RandomWalkGenerator(**params3)

        dataset = SynSet([gen1, gen2, gen3])
        df = dataset.generate(n_series_per_generator=2)
        # Should have 6 series (2 from each of 3 generators)
        n_series = df["unique_id"].n_unique()
        assert n_series == 6

    def test_custom_column_names(self) -> None:
        """Test that custom column names are preserved."""
        rw_params = {
            "min_length": 50,
            "max_length": 100,
            "freq": "h",
            "engine": "polars",
            "seed": 42,
            "id_col": "series_id",
            "time_col": "timestamp",
            "target_col": "value",
        }
        seasonal_params = {
            "min_length": 50,
            "max_length": 100,
            "freq": "h",
            "engine": "polars",
            "seed": 43,
            "id_col": "series_id",
            "time_col": "timestamp",
            "target_col": "value",
        }

        rw_gen = RandomWalkGenerator(**rw_params)
        seasonal_gen = SeasonalGenerator(**seasonal_params)

        dataset = SynSet([rw_gen, seasonal_gen])
        df = dataset.generate(n_series_per_generator=2)
        # Check that custom column names are used
        assert set(df.columns) == {"series_id", "timestamp", "value"}

    def test_empty_generators_list(self) -> None:
        """Test that empty generators list raises error."""
        with pytest.raises(ValueError, match="generators list cannot be empty"):
            SynSet([])

    def test_invalid_generator_type(self) -> None:
        """Test that non-BaseGenerator objects raise error."""
        rw_params = {
            "min_length": 50,
            "max_length": 100,
            "freq": "h",
            "seed": 42,
            "engine": "polars",
        }
        rw_gen = RandomWalkGenerator(**rw_params)

        with pytest.raises(
            ValueError, match="Generator at index 1 is not an instance of BaseGenerator"
        ):
            SynSet([rw_gen, "not a generator"])

    def test_different_frequencies(self) -> None:
        """Test generators with different frequencies."""
        rw_params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "h",
            "engine": "polars",
            "seed": 42,
        }
        seasonal_params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "D",
            "engine": "polars",
            "seed": 43,
        }

        rw_gen = RandomWalkGenerator(**rw_params)
        seasonal_gen = SeasonalGenerator(**seasonal_params)

        dataset = SynSet([rw_gen, seasonal_gen])
        df = dataset.generate(n_series_per_generator=2)
        # Should have 4 series total
        n_series = df["unique_id"].n_unique()
        assert n_series == 4

        # Each series should have the correct frequency
        # (This is implicitly tested by the timestamps being generated correctly)
        assert len(df) == 200  # 2 series * 50 length * 2 generators

    def test_inconsistent_id_col(self) -> None:
        """Test that inconsistent id_col raises error."""
        rw_params = {
            "min_length": 50,
            "max_length": 100,
            "freq": "h",
            "engine": "polars",
            "seed": 42,
            "id_col": "series_id",
        }
        seasonal_params = {
            "min_length": 50,
            "max_length": 100,
            "freq": "h",
            "engine": "polars",
            "seed": 43,
            "id_col": "unique_id",  # Different from first generator
        }

        rw_gen = RandomWalkGenerator(**rw_params)
        seasonal_gen = SeasonalGenerator(**seasonal_params)

        with pytest.raises(ValueError, match="id_col"):
            SynSet([rw_gen, seasonal_gen])

    def test_inconsistent_time_col(self) -> None:
        """Test that inconsistent time_col raises error."""
        rw_params = {
            "min_length": 50,
            "max_length": 100,
            "freq": "h",
            "engine": "polars",
            "seed": 42,
            "time_col": "timestamp",
        }
        seasonal_params = {
            "min_length": 50,
            "max_length": 100,
            "freq": "h",
            "engine": "polars",
            "seed": 43,
            "time_col": "ds",  # Different from first generator
        }

        rw_gen = RandomWalkGenerator(**rw_params)
        seasonal_gen = SeasonalGenerator(**seasonal_params)

        with pytest.raises(ValueError, match="time_col"):
            SynSet([rw_gen, seasonal_gen])

    def test_inconsistent_target_col(self) -> None:
        """Test that inconsistent target_col raises error."""
        rw_params = {
            "min_length": 50,
            "max_length": 100,
            "freq": "h",
            "engine": "polars",
            "seed": 42,
            "target_col": "value",
        }
        seasonal_params = {
            "min_length": 50,
            "max_length": 100,
            "freq": "h",
            "engine": "polars",
            "seed": 43,
            "target_col": "y",  # Different from first generator
        }

        rw_gen = RandomWalkGenerator(**rw_params)
        seasonal_gen = SeasonalGenerator(**seasonal_params)

        with pytest.raises(ValueError, match="target_col"):
            SynSet([rw_gen, seasonal_gen])

    def test_different_frequencies_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that different frequencies emit a warning."""
        rw_params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "h",
            "engine": "polars",
            "seed": 42,
        }
        seasonal_params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "D",  # Different frequency
            "engine": "polars",
            "seed": 43,
        }

        rw_gen = RandomWalkGenerator(**rw_params)
        seasonal_gen = SeasonalGenerator(**seasonal_params)

        # Capture log output
        with caplog.at_level(logging.WARNING):
            dataset = SynSet([rw_gen, seasonal_gen])

        # Check that warning was logged
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"
        assert "freq='D'" in caplog.records[0].message
        assert "freq='h'" in caplog.records[0].message
        assert "Mixing different frequencies" in caplog.records[0].message

        # Should still be able to generate data
        df = dataset.generate(n_series_per_generator=2)

        assert df["unique_id"].n_unique() == 4

    def test_backend_polars(self) -> None:
        """Test that engine='polars' returns a Polars DataFrame."""
        rw_params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "h",
            "seed": 42,
            "engine": "polars",
        }

        rw_gen = RandomWalkGenerator(**rw_params)
        df = rw_gen.generate(n_series=2)

        # Should return a Polars DataFrame
        assert isinstance(df, pl.DataFrame)

        # Check structure
        assert set(df.columns) == {"unique_id", "ds", "y"}
        assert df["unique_id"].n_unique() == 2
        assert len(df) == 100  # 2 series * 50 length

    def test_backend_pandas(self) -> None:
        """Test that engine='pandas' returns a Pandas DataFrame."""
        import pandas as pd

        rw_params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "h",
            "seed": 42,
            "engine": "pandas",
        }

        rw_gen = RandomWalkGenerator(**rw_params)
        df = rw_gen.generate(n_series=2)

        # Should return a Pandas DataFrame
        assert isinstance(df, pd.DataFrame)

        # Check structure
        assert set(df.columns) == {"unique_id", "ds", "y"}
        assert df["unique_id"].nunique() == 2
        assert len(df) == 100  # 2 series * 50 length

    def test_engine_default_is_pandas(self) -> None:
        """Test that the default engine is pandas."""
        rw_gen = RandomWalkGenerator(min_length=50, max_length=50, freq="h", seed=42)

        # Check that default engine is pandas
        assert rw_gen.engine == "pandas"

        df = rw_gen.generate(n_series=1)
        assert isinstance(df, pd.DataFrame)

    def test_synset_backend_polars(self) -> None:
        """Test SynSet with polars backend generators."""
        rw_params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "h",
            "seed": 42,
            "engine": "polars",
        }
        seasonal_params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "h",
            "seed": 43,
            "engine": "polars",
        }

        rw_gen = RandomWalkGenerator(**rw_params)
        seasonal_gen = SeasonalGenerator(**seasonal_params)

        dataset = SynSet([rw_gen, seasonal_gen])
        df = dataset.generate(n_series_per_generator=2)

        # Should return a Polars DataFrame
        assert isinstance(df, pl.DataFrame)
        assert df["unique_id"].n_unique() == 4

    def test_synset_backend_pandas(self) -> None:
        """Test SynSet with pandas backend generators."""
        import pandas as pd

        rw_params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "h",
            "seed": 42,
            "engine": "pandas",
        }
        seasonal_params = {
            "min_length": 50,
            "max_length": 50,
            "freq": "h",
            "seed": 43,
            "engine": "pandas",
        }

        rw_gen = RandomWalkGenerator(**rw_params)
        seasonal_gen = SeasonalGenerator(**seasonal_params)

        dataset = SynSet([rw_gen, seasonal_gen])
        df = dataset.generate(n_series_per_generator=2)

        # Should return a Pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        assert df["unique_id"].nunique() == 4
