"""Tests for diagnostics.scoring."""

from __future__ import annotations

import numpy as np
import pytest

from conformalforecast.diagnostics.scoring import (
    coverage_width_summary,
    mean_interval_width,
    pinball_loss,
    winkler_score,
)


class TestWinklerScore:
    def test_hand_checked_covered(self) -> None:
        # Interval (1, 3), truth 2, alpha 0.1 -> width only: 3 - 1 = 2.
        interval = np.array([[[[1.0, 3.0]]]], dtype=np.float64)
        truth = np.array([[[2.0]]], dtype=np.float64)
        score = winkler_score(interval, truth, alpha=0.1)
        assert score.shape == (1, 1, 1)
        assert score[0, 0, 0] == pytest.approx(2.0)

    def test_hand_checked_above_upper(self) -> None:
        # Interval (1, 3), truth 5, alpha 0.1.
        # width=2; penalty=(2/0.1) * (5 - 3) = 40. total = 42.
        interval = np.array([[[[1.0, 3.0]]]], dtype=np.float64)
        truth = np.array([[[5.0]]], dtype=np.float64)
        score = winkler_score(interval, truth, alpha=0.1)
        assert score[0, 0, 0] == pytest.approx(42.0)

    def test_hand_checked_below_lower(self) -> None:
        # Interval (1, 3), truth -1, alpha 0.2.
        # width=2; penalty=(2/0.2) * (1 - (-1)) = 20. total = 22.
        interval = np.array([[[[1.0, 3.0]]]], dtype=np.float64)
        truth = np.array([[[-1.0]]], dtype=np.float64)
        score = winkler_score(interval, truth, alpha=0.2)
        assert score[0, 0, 0] == pytest.approx(22.0)

    def test_non_negative_on_random_data(self) -> None:
        rng = np.random.default_rng(0)
        truth = rng.normal(size=(3, 50, 4))
        interval = np.stack([truth - 0.5, truth + 0.5], axis=-1)
        # Add some that don't cover truth.
        interval[..., 0] = interval[..., 0] + rng.normal(scale=0.3, size=interval[..., 0].shape)
        scores = winkler_score(interval, truth, alpha=0.1)
        assert (scores >= 0).all()

    def test_saturated_intervals_stay_finite(self) -> None:
        # Online methods (ACI, AgACI) can produce intervals saturated at the
        # per-cell quantile sentinel when alpha_t drifts. Width and Winkler
        # must remain finite under that regime.
        from conformalforecast._quantiles import _SATURATION_SENTINEL

        truth = np.zeros((2, 5, 3), dtype=np.float64)
        lower = np.full_like(truth, -_SATURATION_SENTINEL)
        upper = np.full_like(truth, _SATURATION_SENTINEL)
        interval = np.stack([lower, upper], axis=-1)
        widths = interval[..., 1] - interval[..., 0]
        assert np.isfinite(widths).all()
        scores = winkler_score(interval, truth, alpha=0.1)
        assert np.isfinite(scores).all()

    def test_invalid_alpha_raises(self) -> None:
        interval = np.zeros((1, 1, 1, 2))
        truth = np.zeros((1, 1, 1))
        for bad_alpha in [0.0, 1.0, -0.1, 1.5]:
            with pytest.raises(ValueError, match="alpha"):
                winkler_score(interval, truth, alpha=bad_alpha)

    def test_shape_mismatch_raises(self) -> None:
        interval = np.zeros((1, 2, 1, 2))
        truth = np.zeros((1, 3, 1))
        with pytest.raises(ValueError, match="truth"):
            winkler_score(interval, truth, alpha=0.1)


class TestWinklerNotEvaluable:
    """Winkler needs the truth: a cell without one cannot be scored.

    ``np.where(NaN < lower, ...)`` is ``0.0``, so a finite-bounds/NaN-truth cell
    used to come back as the bare width -- the *best* Winkler that width can
    score. That silently flattered ``mean_winkler_score`` in proportion to how
    gappy the panel was, which makes the metric incomparable across panels, the
    one job a proper scoring rule has.
    """

    ALPHA = 0.1

    def _gappy(self) -> tuple[np.ndarray, np.ndarray]:
        """20 cells of width 2: one genuine miss, six missing truths."""
        n = 20
        interval = np.zeros((1, n, 1, 2))
        interval[..., 0], interval[..., 1] = -1.0, 1.0
        truth = np.zeros((1, n, 1))
        truth[0, 0, 0] = 2.0  # 1.0 outside the upper bound -> penalty 2/alpha
        truth[0, 5:11, 0] = np.nan  # 30% of the holdout has no observed target
        return interval, truth

    def test_nan_truth_is_not_scored(self) -> None:
        interval, truth = self._gappy()
        score = winkler_score(interval, truth, alpha=self.ALPHA)
        assert np.isnan(score[0, 6, 0])

    def test_a_real_miss_still_scores(self) -> None:
        """Masking must not swallow the penalty it exists to apply."""
        interval, truth = self._gappy()
        score = winkler_score(interval, truth, alpha=self.ALPHA)
        # width 2 + (2 / 0.1) * 1.0 = 22.
        assert score[0, 0, 0] == pytest.approx(22.0)

    def test_mean_winkler_excludes_missing_truths(self) -> None:
        # 13 hits at 2.0 plus one miss at 22.0 over 14 evaluable cells = 48/14.
        # Counting the six missing truths as penalty-free 2.0s gives 60/20 = 3.0,
        # which is the bias this masking removes.
        interval, truth = self._gappy()
        summary = coverage_width_summary(interval, truth, alpha=self.ALPHA)
        assert summary["mean_winkler"] == pytest.approx(48.0 / 14.0)

    def test_nan_bound_is_not_scored(self) -> None:
        """Regression guard -- already held, since the width carries the NaN."""
        interval, truth = self._gappy()
        interval[0, 1] = np.nan
        score = winkler_score(interval, truth, alpha=self.ALPHA)
        assert np.isnan(score[0, 1, 0])

    def test_dense_data_is_unaffected(self) -> None:
        """The masking must be invisible whenever every truth is observed."""
        interval, truth = self._gappy()
        truth[0, 5:11, 0] = 0.0  # fill the gaps in: all covered
        score = winkler_score(interval, truth, alpha=self.ALPHA)
        assert not np.isnan(score).any()
        assert float(score.mean()) == pytest.approx(60.0 / 20.0)


class TestMeanIntervalWidthNotEvaluable:
    """Width needs no target, so a NaN truth must not shrink its sample."""

    def test_nan_bound_is_skipped(self) -> None:
        interval = np.zeros((1, 4, 1, 2))
        interval[..., 0], interval[..., 1] = -1.0, 1.0
        interval[0, 2] = np.nan
        # Three surviving samples, all of width 2.
        np.testing.assert_allclose(mean_interval_width(interval), [[2.0]])

    def test_all_bounds_nan_is_nan(self) -> None:
        interval = np.full((1, 3, 1, 2), np.nan)
        assert np.isnan(mean_interval_width(interval)).all()


class TestMeanIntervalWidth:
    def test_shape_and_value(self) -> None:
        # All widths are 2.
        interval = np.zeros((2, 5, 3, 2))
        interval[..., 0] = -1.0
        interval[..., 1] = 1.0
        out = mean_interval_width(interval)
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out, 2.0)

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="interval"):
            mean_interval_width(np.zeros((2, 5, 3)))


class TestPinballLoss:
    def test_q_half_equals_abs_diff_over_two(self) -> None:
        # rho_{0.5}(y, y_hat) = max(0.5*(y-y_hat), -0.5*(y-y_hat)) = 0.5*|y-y_hat|
        prediction = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64)
        truth = np.array([[[2.0, 0.0, 3.0]]], dtype=np.float64)
        out = pinball_loss(prediction, truth, quantile=0.5)
        assert out.shape == (1, 1, 3)
        np.testing.assert_allclose(out, np.array([[[0.5, 1.0, 0.0]]]))

    def test_known_pinball_values(self) -> None:
        # q=0.9; truth=10, prediction=8: diff=2, rho=max(0.9*2, -0.1*2)=1.8
        # q=0.9; truth=8, prediction=10: diff=-2, rho=max(-1.8, 0.2)=0.2
        prediction = np.array([[[8.0, 10.0]]], dtype=np.float64)
        truth = np.array([[[10.0, 8.0]]], dtype=np.float64)
        out = pinball_loss(prediction, truth, quantile=0.9)
        np.testing.assert_allclose(out, np.array([[[1.8, 0.2]]]))

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(2)
        prediction = rng.normal(size=(2, 30, 3))
        truth = rng.normal(size=(2, 30, 3))
        for q in [0.1, 0.5, 0.9]:
            out = pinball_loss(prediction, truth, quantile=q)
            assert (out >= 0).all()

    def test_invalid_quantile_raises(self) -> None:
        prediction = np.zeros((1, 1, 1))
        truth = np.zeros((1, 1, 1))
        for bad_q in [0.0, 1.0, -0.1, 1.1]:
            with pytest.raises(ValueError, match="quantile"):
                pinball_loss(prediction, truth, quantile=bad_q)

    def test_shape_mismatch_raises(self) -> None:
        prediction = np.zeros((1, 2, 1))
        truth = np.zeros((1, 3, 1))
        with pytest.raises(ValueError, match="prediction"):
            pinball_loss(prediction, truth, quantile=0.5)


class TestCoverageWidthSummary:
    def test_returns_expected_keys(self) -> None:
        rng = np.random.default_rng(0)
        truth = rng.normal(size=(1, 20, 1))
        interval = np.stack([truth - 1.0, truth + 1.0], axis=-1)
        summary = coverage_width_summary(interval, truth, alpha=0.1)
        assert set(summary.keys()) == {
            "marginal_coverage",
            "target_coverage",
            "mean_width",
            "mean_winkler",
        }
        for v in summary.values():
            assert isinstance(v, float)
        assert summary["target_coverage"] == pytest.approx(0.9)
        assert summary["marginal_coverage"] == pytest.approx(1.0)
        assert summary["mean_width"] == pytest.approx(2.0)

    def test_invalid_alpha_raises(self) -> None:
        interval = np.zeros((1, 1, 1, 2))
        truth = np.zeros((1, 1, 1))
        with pytest.raises(ValueError, match="alpha"):
            coverage_width_summary(interval, truth, alpha=0.0)
