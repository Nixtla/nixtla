"""Tests for anomaly injection: counts, magnitudes, flags, and edge cases.

Direct `_core` tests cover both the Rust-delegating path (module default)
and the pure-Python fallback (via monkeypatched ``_HAS_RUST``).
"""

import numpy as np
import pytest

import synforecast._core as core_module
from synforecast._core import _add_anomalies
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


def _inject(values, seed, types, fraction, spike=10.0, dip=-10.0, shift=5.0, dur=10):
    return _add_anomalies(
        values, np.random.default_rng(seed), types, fraction, spike, dip, shift, dur
    )


class TestSpikeAndDip:
    def test_spike_count_and_magnitude_python(self, monkeypatch):
        """Python path: exactly floor(n*fraction) points get +spike_magnitude."""
        monkeypatch.setattr(core_module, "_HAS_RUST", False)
        out, meta = _inject(np.zeros(1000), 0, ["spike"], 0.05, spike=7.5)
        expected = int(1000 * 0.05)
        spiked = np.flatnonzero(out != 0.0)
        assert len(spiked) == expected
        assert np.allclose(out[spiked], 7.5)
        assert set(meta["anomaly_indices"].tolist()) == set(spiked.tolist())

    def test_dip_count_and_magnitude_python(self, monkeypatch):
        monkeypatch.setattr(core_module, "_HAS_RUST", False)
        out, meta = _inject(np.zeros(1000), 1, ["dip"], 0.05, dip=-4.25)
        expected = int(1000 * 0.05)
        dipped = np.flatnonzero(out != 0.0)
        assert len(dipped) == expected
        assert np.allclose(out[dipped], -4.25)

    @pytest.mark.usefixtures("core_path")
    def test_spike_magnitude_both_paths(self):
        """Both paths: every modified point is an integer multiple of the
        magnitude (the Rust path may stack duplicates), and the count of
        injected events matches floor(n*fraction)."""
        out, meta = _inject(np.zeros(2000), 2, ["spike"], 0.05, spike=10.0)
        expected = int(2000 * 0.05)
        assert len(meta["anomaly_indices"]) == expected
        modified = out[out != 0.0]
        assert len(modified) > 0
        multiples = modified / 10.0
        assert np.allclose(multiples, np.round(multiples))
        assert np.all(multiples >= 1)
        # Rust samples with replacement: distinct points <= events
        assert len(modified) <= expected

    @pytest.mark.usefixtures("core_path")
    def test_metadata_covers_modified_points(self):
        out, meta = _inject(np.zeros(500), 3, ["spike", "dip"], 0.1)
        modified = set(np.flatnonzero(out != 0.0).tolist())
        reported = set(np.asarray(meta["anomaly_indices"]).tolist())
        assert modified <= reported

    def test_mixed_types_magnitudes(self, monkeypatch):
        """Python path: with spike+dip, each anomaly equals one magnitude."""
        monkeypatch.setattr(core_module, "_HAS_RUST", False)
        out, _ = _inject(np.zeros(1000), 4, ["spike", "dip"], 0.1, spike=8.0, dip=-3.0)
        modified = out[out != 0.0]
        assert len(modified) == 100
        assert np.all(np.isin(modified, [8.0, -3.0]))
        # Types are picked uniformly; with n=100 both should appear
        assert (modified == 8.0).any() and (modified == -3.0).any()


class TestLevelShift:
    def test_shift_duration_python(self, monkeypatch):
        """Python path: a level shift raises exactly `duration` points."""
        monkeypatch.setattr(core_module, "_HAS_RUST", False)
        length, dur = 400, 12
        # fraction chosen so exactly one anomaly is injected
        out, meta = _inject(
            np.zeros(length), 5, ["level_shift"], 1.5 / length, shift=6.0, dur=dur
        )
        loc = int(meta["anomaly_indices"][0])
        end = min(loc + dur, length)
        expected = np.zeros(length)
        expected[loc:end] = 6.0
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.usefixtures("core_path")
    def test_shift_truncated_at_series_end(self):
        """A level shift near the end must not run past the series."""
        # fraction 0.02 on length 50 -> exactly one anomaly per call;
        # duration 100 always runs past the end and must be truncated
        for seed in range(30):
            out, meta = _inject(
                np.zeros(50), seed, ["level_shift"], 0.02, shift=3.0, dur=100
            )
            assert len(meta["anomaly_indices"]) == 1
            loc = int(meta["anomaly_indices"][0])
            assert np.all(out[:loc] == 0.0)
            assert np.all(out[loc:] == 3.0)

    @pytest.mark.usefixtures("core_path")
    def test_shift_affects_more_points_than_events(self):
        out, meta = _inject(np.zeros(500), 6, ["level_shift"], 0.01, shift=2.0, dur=20)
        assert (out != 0.0).sum() > len(meta["anomaly_indices"])


class TestEdgeCases:
    @pytest.mark.usefixtures("core_path")
    def test_zero_fraction_no_anomalies(self):
        out, meta = _inject(np.ones(100), 7, ["spike"], 0.0)
        assert np.all(out == 1.0)
        assert len(meta["anomaly_indices"]) == 0

    def test_fraction_one_python(self, monkeypatch):
        """Python path: fraction 1.0 makes every point an anomaly."""
        monkeypatch.setattr(core_module, "_HAS_RUST", False)
        out, meta = _inject(np.zeros(50), 8, ["spike"], 1.0, spike=1.0)
        assert np.all(out == 1.0)
        assert len(set(meta["anomaly_indices"].tolist())) == 50

    def test_generator_validation(self):
        base = {"min_length": 100, "max_length": 100, "freq": "D"}
        with pytest.raises(ValueError, match="anomaly_fraction"):
            RandomWalkGenerator(**base, anomalies=True, anomaly_fraction=1.5)
        with pytest.raises(ValueError, match="anomaly_type"):
            RandomWalkGenerator(**base, anomalies=True, anomaly_types=["invalid"])


class TestGeneratorAPI:
    BASE = {"min_length": 400, "max_length": 400, "freq": "D", "seed": 42}

    def test_no_anomalies_by_default(self, engine):
        gen = RandomWalkGenerator(
            **self.BASE, engine=engine, drift=0.0, volatility=1e-9
        )
        pdf = to_pandas(gen.generate(n_series=1))
        assert np.all(np.abs(pdf["y"].to_numpy()) < 1.0)

    def test_anomaly_flag_aligns_with_spikes(self, engine):
        """With a near-flat walk, |y| > spike/2 iff the point was spiked;
        the flag column must mark exactly those points."""
        gen = RandomWalkGenerator(
            **self.BASE,
            engine=engine,
            drift=0.0,
            volatility=1e-9,
            anomalies=True,
            anomaly_fraction=0.05,
            anomaly_types=["spike"],
            spike_magnitude=100.0,
            exogenous=ExogenousConfig(anomaly_flags=True),
        )
        pdf = to_pandas(gen.generate(n_series=3))
        assert "anomaly_flag" in pdf.columns
        y = pdf["y"].to_numpy()
        flags = pdf["anomaly_flag"].to_numpy().astype(bool)
        np.testing.assert_array_equal(y > 50.0, flags)

    def test_anomaly_flag_count_matches_fraction(self, engine):
        """Flagged count per series is close to floor(n*fraction); the Rust
        path samples with replacement so a few collisions are allowed."""
        n, fraction = 400, 0.05
        gen = RandomWalkGenerator(
            **self.BASE,
            engine=engine,
            anomalies=True,
            anomaly_fraction=fraction,
            anomaly_types=["spike"],
            spike_magnitude=100.0,
            exogenous=ExogenousConfig(anomaly_flags=True),
        )
        pdf = to_pandas(gen.generate(n_series=5))
        expected = int(n * fraction)
        for _, group in pdf.groupby("unique_id", observed=True):
            count = int(group["anomaly_flag"].sum())
            assert 0.75 * expected <= count <= expected

    def test_dip_magnitude_through_api(self, engine):
        gen = RandomWalkGenerator(
            **self.BASE,
            engine=engine,
            drift=0.0,
            volatility=1e-9,
            anomalies=True,
            anomaly_fraction=0.05,
            anomaly_types=["dip"],
            dip_magnitude=-50.0,
        )
        pdf = to_pandas(gen.generate(n_series=1))
        y = pdf["y"].to_numpy()
        dipped = y[y < -25.0]
        assert len(dipped) > 0
        # Each dipped value is a multiple of the magnitude (stacking allowed)
        multiples = dipped / -50.0
        assert np.allclose(multiples, np.round(multiples))

    def test_threaded_fallback_generator_anomalies(self, engine):
        """VAR has no Rust batch path -> exercises the threaded pipeline."""
        gen = VARGenerator(
            min_length=200,
            max_length=200,
            freq="D",
            engine=engine,
            seed=42,
            anomalies=True,
            anomaly_fraction=0.1,
            anomaly_types=["spike"],
            spike_magnitude=1000.0,
        )
        pdf = to_pandas(gen.generate(n_series=3))
        for _, group in pdf.groupby("unique_id", observed=True):
            assert (group["y"].to_numpy() > 500.0).any()

    def test_anomalies_with_missingness(self, engine):
        gen = RandomWalkGenerator(
            **self.BASE,
            engine=engine,
            anomalies=True,
            anomaly_fraction=0.1,
            spike_magnitude=50.0,
            missing_data=True,
            missing_rate=0.2,
        )
        pdf = to_pandas(gen.generate(n_series=1))
        assert len(pdf) == 400
        assert pdf["y"].isna().sum() > 0
