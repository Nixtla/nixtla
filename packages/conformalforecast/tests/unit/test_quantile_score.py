"""Tests for QuantileScore (CQR nonconformity score)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import BaseModel

from conformalforecast.nonconformity.quantile import QuantileScore


def test_is_model_and_still_constructs_bare() -> None:
    instance = QuantileScore()
    assert isinstance(instance, BaseModel)
    assert QuantileScore.model_json_schema()["properties"].keys() == {"kind"}
    assert QuantileScore.model_validate(instance.model_dump()) == instance


class TestScore:
    """QuantileScore.score returns correct values and shapes."""

    def test_hand_checked_values(self) -> None:
        score_fn = QuantileScore()
        # (n_series=1, n_samples=2, horizon=1, 2): two points
        # First: q_lo=1, q_hi=3, y=2 → max(1-2, 2-3) = max(-1, -1) = -1
        # Second: q_lo=2, q_hi=4, y=5 → max(2-5, 5-4) = max(-3, 1) = 1
        prediction = np.array([[[[1.0, 3.0]], [[2.0, 4.0]]]])
        truth = np.array([[[2.0], [5.0]]])
        scores = score_fn.score(prediction, truth)
        np.testing.assert_array_almost_equal(scores, np.array([[[-1.0], [1.0]]]))

    def test_sign_convention_inside_interval(self) -> None:
        """Truth inside the predicted interval → non-positive score."""
        score_fn = QuantileScore()
        prediction = np.array([[[[0.0, 10.0]]]])  # q_lo=0, q_hi=10
        truth = np.array([[[5.0]]])  # safely inside
        scores = score_fn.score(prediction, truth)
        assert scores.item() <= 0

    def test_sign_convention_outside_interval(self) -> None:
        """Truth outside the predicted interval → positive score."""
        score_fn = QuantileScore()
        prediction = np.array([[[[0.0, 10.0]]]])
        truth_above = np.array([[[15.0]]])
        truth_below = np.array([[[-3.0]]])
        assert score_fn.score(prediction, truth_above).item() > 0
        assert score_fn.score(prediction, truth_below).item() > 0

    @pytest.mark.parametrize(
        "n_series,n_samples,horizon",
        [(1, 1, 1), (1, 50, 1), (5, 1, 12), (5, 50, 12)],
    )
    def test_shape_preserved(self, n_series: int, n_samples: int, horizon: int) -> None:
        score_fn = QuantileScore()
        rng = np.random.default_rng(0)
        q_lo = rng.standard_normal((n_series, n_samples, horizon)) - 1.0
        q_hi = rng.standard_normal((n_series, n_samples, horizon)) + 1.0
        prediction = np.stack([q_lo, q_hi], axis=-1)
        truth = rng.standard_normal((n_series, n_samples, horizon))

        scores = score_fn.score(prediction, truth)
        assert scores.shape == (n_series, n_samples, horizon)

    def test_invalid_prediction_shape_raises(self) -> None:
        score_fn = QuantileScore()
        # 3-D prediction (missing trailing axis of size 2) is invalid.
        prediction = np.zeros((1, 1, 1))
        truth = np.zeros((1, 1, 1))
        with pytest.raises(ValueError, match="n_samples, horizon, 2"):
            score_fn.score(prediction, truth)

    def test_invalid_trailing_axis_raises(self) -> None:
        score_fn = QuantileScore()
        # Last axis size 3 instead of 2.
        prediction = np.zeros((1, 1, 1, 3))
        truth = np.zeros((1, 1, 1))
        with pytest.raises(ValueError, match="n_samples, horizon, 2"):
            score_fn.score(prediction, truth)

    def test_truth_shape_mismatch_raises(self) -> None:
        score_fn = QuantileScore()
        prediction = np.zeros((1, 1, 1, 2))
        truth = np.zeros((1, 1, 2))  # wrong horizon
        with pytest.raises(ValueError, match="truth.shape"):
            score_fn.score(prediction, truth)


class TestInvert:
    """QuantileScore.invert widens the predicted interval."""

    def test_invert_widens_interval(self) -> None:
        """Positive threshold strictly widens the interval componentwise."""
        score_fn = QuantileScore()
        prediction = np.array([[[[1.0, 3.0], [2.0, 5.0]]]])  # (1, 1, 2, 2)
        threshold = np.array([[0.5, 1.0]])  # (n_series=1, horizon=2)

        interval = score_fn.invert(prediction, threshold)

        assert interval.shape == (1, 1, 2, 2)
        # lower = q_lo - threshold; upper = q_hi + threshold
        np.testing.assert_array_almost_equal(interval[0, 0, :, 0], [0.5, 1.0])
        np.testing.assert_array_almost_equal(interval[0, 0, :, 1], [3.5, 6.0])
        # widening: lower < q_lo and upper > q_hi
        assert np.all(interval[..., 0] < prediction[..., 0])
        assert np.all(interval[..., 1] > prediction[..., 1])

    def test_invert_scalar_threshold(self) -> None:
        score_fn = QuantileScore()
        prediction = np.array([[[[1.0, 3.0]]]])
        interval = score_fn.invert(prediction, np.float64(2.0))
        np.testing.assert_array_almost_equal(interval[0, 0, 0, :], [-1.0, 5.0])

    def test_invert_1d_threshold(self) -> None:
        score_fn = QuantileScore()
        # (n_series=2, n_samples=1, horizon=2, 2)
        prediction = np.array([[[[1.0, 3.0], [2.0, 4.0]]], [[[10.0, 20.0], [30.0, 40.0]]]])
        threshold = np.array([0.5, 2.0])  # (n_series=2,)
        interval = score_fn.invert(prediction, threshold)
        assert interval.shape == (2, 1, 2, 2)
        np.testing.assert_array_almost_equal(interval[0, 0, :, 0], [0.5, 1.5])
        np.testing.assert_array_almost_equal(interval[1, 0, :, 0], [8.0, 28.0])

    @pytest.mark.parametrize(
        "n_series,n_samples,horizon",
        [(1, 1, 1), (1, 50, 1), (5, 1, 12), (5, 50, 12)],
    )
    def test_invert_output_shape(self, n_series: int, n_samples: int, horizon: int) -> None:
        score_fn = QuantileScore()
        rng = np.random.default_rng(0)
        q_lo = rng.standard_normal((n_series, n_samples, horizon)) - 1.0
        q_hi = q_lo + 1.0 + rng.random((n_series, n_samples, horizon))
        prediction = np.stack([q_lo, q_hi], axis=-1)
        threshold = rng.random((n_series, horizon))

        interval = score_fn.invert(prediction, threshold)
        assert interval.shape == (n_series, n_samples, horizon, 2)

    def test_invert_invalid_prediction_shape_raises(self) -> None:
        score_fn = QuantileScore()
        prediction = np.zeros((1, 1, 1))  # 3-D, missing trailing axis
        with pytest.raises(ValueError, match="n_samples, horizon, 2"):
            score_fn.invert(prediction, np.array([[0.0]]))
