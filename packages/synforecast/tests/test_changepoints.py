"""Tests for changepoint injection: types, locations, magnitudes, flags.

Direct `_core` tests cover both the Rust-delegating path (module default)
and the pure-Python fallback (via monkeypatched ``_HAS_RUST``).
"""

import numpy as np
import pytest

import synforecast._core as core_module
from synforecast._core import _add_changepoints
from synforecast.exogenous import ExogenousConfig
from synforecast.generators import RandomWalkGenerator, VARGenerator
from tests.helpers import to_pandas


@pytest.fixture(params=["rust", "python"])
def core_path(request, monkeypatch):
    """Run direct _core tests on both the Rust and pure-Python branches."""
    if request.param == "python":
        monkeypatch.setattr(core_module, "_HAS_RUST", False)
    elif not core_module._HAS_RUST:
        pytest.skip("Rust backend not available")
    return request.param


def _inject(values, seed, n_cp, locs, cp_type, level=(), trend=(), variance=()):
    return _add_changepoints(
        values,
        np.random.default_rng(seed),
        n_cp,
        np.asarray(locs, dtype=float),
        cp_type,
        np.asarray(level, dtype=float),
        np.asarray(trend, dtype=float),
        np.asarray(variance, dtype=float),
    )


class TestLevelChangepoints:
    @pytest.mark.usefixtures("core_path")
    def test_exact_segment_levels(self):
        """Level changes on a zero series produce exact step segments."""
        out, meta = _inject(np.zeros(100), 0, 2, [0.3, 0.6], "level", level=[5.0, -3.0])
        np.testing.assert_array_equal(meta["changepoint_indices"], [30, 60])
        assert np.all(out[:30] == 0.0)
        assert np.all(out[30:60] == 5.0)
        assert np.all(out[60:] == 2.0)  # cumulative: 5 - 3

    @pytest.mark.usefixtures("core_path")
    def test_cumulative_effects(self):
        out, _ = _inject(
            np.zeros(200), 1, 3, [0.2, 0.5, 0.8], "level", level=[10.0, 10.0, 10.0]
        )
        assert np.all(out[160:] == 30.0)

    @pytest.mark.usefixtures("core_path")
    def test_random_locations_within_bounds(self):
        out, meta = _inject(np.zeros(200), 2, 3, [], "level")
        idx = np.asarray(meta["changepoint_indices"])
        assert len(idx) == 3
        # locations drawn from uniform(0.1, 0.9)
        assert np.all(idx >= 20) and np.all(idx < 180)
        # each changepoint actually shifted the level
        for i in idx:
            assert out[i] != out[i - 1]


class TestTrendChangepoints:
    @pytest.mark.usefixtures("core_path")
    def test_exact_trend_slope(self):
        out, meta = _inject(np.zeros(100), 3, 1, [0.5], "trend", trend=[0.2])
        assert np.all(out[:50] == 0.0)
        np.testing.assert_allclose(out[50:], 0.2 * np.arange(50), atol=1e-12)

    @pytest.mark.usefixtures("core_path")
    def test_two_trends_accumulate(self):
        out, _ = _inject(np.zeros(100), 4, 2, [0.2, 0.6], "trend", trend=[0.1, 0.1])
        # after second changepoint slope is 0.2 per step
        diffs = np.diff(out[61:])
        np.testing.assert_allclose(diffs, 0.2, atol=1e-12)


class TestVarianceChangepoints:
    @pytest.mark.stats
    @pytest.mark.usefixtures("core_path")
    def test_local_variance_scales(self):
        """Local deviations roughly double after a variance change of 2."""
        rng_in = np.random.default_rng(123)
        base = rng_in.normal(0.0, 1.0, 2000)
        out, _ = _inject(base.copy(), 5, 1, [0.5], "variance", variance=[2.0])

        def local_dev_std(x):
            kernel = np.ones(11) / 11.0
            m = np.convolve(x, kernel, mode="same")
            return np.std(x[20:-20] - m[20:-20])

        ratio = local_dev_std(out[1000:]) / local_dev_std(out[:1000])
        assert 1.5 < ratio < 2.6, f"variance ratio {ratio}"

    @pytest.mark.usefixtures("core_path")
    def test_variance_change_is_bounded(self):
        """Regression: variance changepoints must not diverge (the Rust
        implementation used a running-window sum that blew up)."""
        rng_in = np.random.default_rng(7)
        base = rng_in.normal(0.0, 1.0, 2000)
        out, _ = _inject(base.copy(), 6, 1, [0.2], "variance", variance=[2.0])
        assert np.max(np.abs(out)) < 100.0

    @pytest.mark.usefixtures("core_path")
    def test_variance_below_one_dampens(self):
        rng_in = np.random.default_rng(8)
        base = rng_in.normal(0.0, 1.0, 1000)
        out, _ = _inject(base.copy(), 9, 1, [0.5], "variance", variance=[0.5])
        assert np.std(out[500:]) < np.std(out[:500])


class TestMixedChangepoints:
    @pytest.mark.usefixtures("core_path")
    def test_mixed_applies_level_and_trend(self):
        out, meta = _inject(
            np.zeros(100),
            10,
            1,
            [0.5],
            "mixed",
            level=[10.0],
            trend=[0.5],
            variance=[1.0],
        )
        assert np.all(out[:50] == 0.0)
        np.testing.assert_allclose(out[50:], 10.0 + 0.5 * np.arange(50), atol=1e-12)


class TestEdgeCases:
    @pytest.mark.usefixtures("core_path")
    def test_fewer_locations_than_num_changepoints(self):
        """Regression: partial location lists are padded, not IndexError."""
        out, meta = _inject(np.zeros(100), 11, 3, [0.5], "level", level=[5.0])
        idx = np.asarray(meta["changepoint_indices"])
        assert len(idx) == 3
        assert idx[0] == 50

    def test_location_one_maps_to_last_index_python(self, monkeypatch):
        """Python path: relative location 1.0 lands on the final point."""
        monkeypatch.setattr(core_module, "_HAS_RUST", False)
        out, meta = _inject(np.zeros(10), 12, 1, [1.0], "level", level=[7.0])
        assert meta["changepoint_indices"][0] == 9
        assert out[9] == 7.0
        assert np.all(out[:9] == 0.0)

    def test_extra_locations_truncated_python(self, monkeypatch):
        """Python path: metadata reports exactly num_changepoints indices."""
        monkeypatch.setattr(core_module, "_HAS_RUST", False)
        out, meta = _inject(np.zeros(100), 13, 1, [0.3, 0.6, 0.9], "level", level=[5.0])
        np.testing.assert_array_equal(meta["changepoint_indices"], [30])
        assert np.all(out[30:] == 5.0)

    def test_generator_validation(self):
        base = {"min_length": 100, "max_length": 100, "freq": "D"}
        with pytest.raises(ValueError, match="num_changepoints"):
            RandomWalkGenerator(**base, changepoints=True, num_changepoints=-1)
        with pytest.raises(ValueError, match="changepoint_type"):
            RandomWalkGenerator(
                **base, changepoints=True, changepoint_type="invalid_type"
            )


class TestGeneratorAPI:
    BASE = {"min_length": 200, "max_length": 200, "freq": "D", "seed": 42}

    def test_no_changepoints_by_default(self, engine):
        gen = RandomWalkGenerator(
            **self.BASE, engine=engine, drift=0.0, volatility=1e-9
        )
        pdf = to_pandas(gen.generate(n_series=1))
        assert np.all(np.abs(pdf["y"].to_numpy()) < 1.0)

    def test_zero_changepoints(self, engine):
        gen = RandomWalkGenerator(
            **self.BASE,
            engine=engine,
            drift=0.0,
            volatility=1e-9,
            changepoints=True,
            num_changepoints=0,
        )
        pdf = to_pandas(gen.generate(n_series=1))
        assert np.all(np.abs(pdf["y"].to_numpy()) < 1.0)

    def test_level_change_magnitude_and_flag(self, engine):
        """A level changepoint on a near-flat walk shifts the mean by the
        configured amount at exactly the flagged position."""
        gen = RandomWalkGenerator(
            **self.BASE,
            engine=engine,
            drift=0.0,
            volatility=1e-9,
            changepoints=True,
            num_changepoints=1,
            changepoint_type="level",
            changepoint_level_changes=[50.0],
            changepoint_locations=[0.5],
            exogenous=ExogenousConfig(changepoint_flags=True),
        )
        pdf = to_pandas(gen.generate(n_series=2))
        assert "changepoint_flag" in pdf.columns
        for _, group in pdf.groupby("unique_id", observed=True):
            y = group["y"].to_numpy()
            flags = np.flatnonzero(group["changepoint_flag"].to_numpy())
            np.testing.assert_array_equal(flags, [100])
            assert abs(np.mean(y[100:]) - np.mean(y[:100]) - 50.0) < 1e-6

    def test_trend_change_magnitude(self, engine):
        gen = RandomWalkGenerator(
            **self.BASE,
            engine=engine,
            drift=0.0,
            volatility=1e-9,
            changepoints=True,
            num_changepoints=1,
            changepoint_type="trend",
            changepoint_trend_changes=[0.5],
            changepoint_locations=[0.5],
        )
        pdf = to_pandas(gen.generate(n_series=1))
        y = pdf["y"].to_numpy()
        np.testing.assert_allclose(np.diff(y[101:]), 0.5, atol=1e-6)
        np.testing.assert_allclose(np.diff(y[:100]), 0.0, atol=1e-6)

    def test_variance_changepoint_bounded_threaded_path(self, engine):
        """Variance changepoints through generate() stay bounded on the
        threaded path. VAR has no Rust batch path, so it uses the threaded
        pipeline (per-series _core injection).
        """
        gen = VARGenerator(
            **self.BASE,
            engine=engine,
            changepoints=True,
            num_changepoints=1,
            changepoint_type="variance",
            changepoint_variance_changes=[2.0],
            changepoint_locations=[0.5],
        )
        pdf = to_pandas(gen.generate(n_series=1, n_jobs=1))
        assert np.all(np.isfinite(pdf["y"].to_numpy()))
        assert np.max(np.abs(pdf["y"].to_numpy())) < 1e4

    def test_variance_changepoint_bounded_rust_batch_path(self):
        """Variance changepoints through the Rust batch path stay bounded
        (regression for the fixed running-window divergence in
        rust/src/pattern_injection.rs add_changepoints)."""
        gen = RandomWalkGenerator(
            min_length=1000,
            max_length=1000,
            freq="D",
            seed=42,
            changepoints=True,
            num_changepoints=1,
            changepoint_type="variance",
            changepoint_variance_changes=[2.0],
            changepoint_locations=[0.2],
        )
        pdf = to_pandas(gen.generate(n_series=1))
        assert np.max(np.abs(pdf["y"].to_numpy())) < 1e4

    def test_changepoints_with_anomalies_and_missingness(self, engine):
        gen = RandomWalkGenerator(
            **self.BASE,
            engine=engine,
            changepoints=True,
            num_changepoints=2,
            changepoint_type="level",
            changepoint_level_changes=[30.0, -20.0],
            anomalies=True,
            anomaly_fraction=0.05,
            spike_magnitude=25.0,
            missing_data=True,
            missing_rate=0.1,
        )
        pdf = to_pandas(gen.generate(n_series=1))
        assert len(pdf) == 200
        assert pdf["y"].isna().sum() > 0
