"""Tests for series classification (SynAugment.analyze internals).

Regression guard for the analyzer that previously recommended
RegimeSwitchingGenerator for nearly every series: regime detection now scores a
piecewise-constant fit against a linear null, and seasonality requires a genuine
ACF peak rather than any high monotonic value.
"""

import numpy as np

from synforecast._analysis import (
    classify_series,
    detect_regime_changes,
    detect_seasonality,
)

N = 240


def _series() -> dict[str, np.ndarray]:
    """A spread of distinct, deterministic series types."""
    rng = np.random.default_rng(0)
    t = np.arange(N)
    x_ar1 = np.zeros(N)
    for i in range(1, N):
        x_ar1[i] = 0.7 * x_ar1[i - 1] + rng.normal()
    intermittent = np.zeros(N)
    intermittent[rng.choice(N, 30, replace=False)] = rng.integers(1, 20, 30)
    return {
        "random_walk": np.cumsum(rng.normal(0, 1, N)),
        "seasonal": 50 + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1, N),
        "linear_trend": 0.5 * t + rng.normal(0, 3, N),
        "two_regime": np.concatenate(
            [rng.normal(0, 0.3, 120), rng.normal(8, 0.3, 120)]
        ),
        "white_noise": rng.normal(0, 1, N),
        "gbm": 100 * np.exp(np.cumsum(rng.normal(0.003, 0.02, N))),
        "intermittent": intermittent,
        "ar1": x_ar1,
    }


class TestRegimeDetection:
    """detect_regime_changes must fire only on genuine level shifts."""

    def test_detects_clean_level_shift(self) -> None:
        rng = np.random.default_rng(1)
        series = np.concatenate([rng.normal(0, 0.3, 120), rng.normal(8, 0.3, 120)])
        result = detect_regime_changes(series)
        assert result["has_regimes"]
        assert result["n_regimes"] >= 2

    def test_rejects_trend(self) -> None:
        rng = np.random.default_rng(2)
        series = 0.5 * np.arange(N) + rng.normal(0, 3, N)
        assert not detect_regime_changes(series)["has_regimes"]

    def test_rejects_random_walk(self) -> None:
        rng = np.random.default_rng(3)
        series = np.cumsum(rng.normal(0, 1, N))
        assert not detect_regime_changes(series)["has_regimes"]

    def test_rejects_white_noise(self) -> None:
        rng = np.random.default_rng(4)
        assert not detect_regime_changes(rng.normal(0, 1, N))["has_regimes"]

    def test_short_series_has_no_regimes(self) -> None:
        assert not detect_regime_changes(np.arange(20.0))["has_regimes"]


class TestSeasonalityDetection:
    """detect_seasonality must key on an ACF peak, not monotonic decay."""

    def test_detects_seasonal_period(self) -> None:
        rng = np.random.default_rng(5)
        t = np.arange(N)
        series = 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1, N)
        result = detect_seasonality(series)
        assert result["has_seasonality"]
        assert abs(result["period"] - 24) <= 1

    def test_random_walk_is_not_seasonal(self) -> None:
        rng = np.random.default_rng(6)
        # A random walk has a high but monotonically decaying ACF; the old
        # argmax rule mislabeled it seasonal at a tiny lag.
        assert not detect_seasonality(np.cumsum(rng.normal(0, 1, N)))[
            "has_seasonality"
        ]

    def test_level_shift_is_not_seasonal(self) -> None:
        rng = np.random.default_rng(7)
        series = np.concatenate([rng.normal(0, 0.3, 120), rng.normal(8, 0.3, 120)])
        assert not detect_seasonality(series)["has_seasonality"]


class TestClassificationDiscrimination:
    """classify_series must not collapse everything onto one generator."""

    def test_recommendations_are_diverse(self) -> None:
        recs = [classify_series(s)["recommended_generator"] for s in _series().values()]
        # The original bug produced RegimeSwitchingGenerator for ~all series.
        assert len(set(recs)) >= 4
        assert recs.count("RegimeSwitchingGenerator") <= 1

    def test_seasonal_series_recommends_seasonal_family(self) -> None:
        rng = np.random.default_rng(8)
        t = np.arange(N)
        series = 50 + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1, N)
        assert classify_series(series)["recommended_generator"] in (
            "SeasonalGenerator",
            "SARIMAGenerator",
        )

    def test_level_shift_recommends_regime_switching(self) -> None:
        rng = np.random.default_rng(9)
        series = np.concatenate([rng.normal(0, 0.3, 120), rng.normal(8, 0.3, 120)])
        assert (
            classify_series(series)["recommended_generator"]
            == "RegimeSwitchingGenerator"
        )

    def test_intermittent_recommends_intermittent_demand(self) -> None:
        rng = np.random.default_rng(10)
        series = np.zeros(N)
        series[rng.choice(N, 30, replace=False)] = rng.integers(1, 20, 30)
        assert (
            classify_series(series)["recommended_generator"]
            == "IntermittentDemandGenerator"
        )
