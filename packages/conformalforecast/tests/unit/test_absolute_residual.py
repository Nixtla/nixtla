"""Tests for AbsoluteResidual score function."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from conformalforecast.nonconformity.absolute import AbsoluteResidual


def test_is_model_and_still_constructs_bare() -> None:
    instance = AbsoluteResidual()
    assert isinstance(instance, BaseModel)
    assert AbsoluteResidual.model_json_schema()["properties"].keys() == {"kind"}
    assert AbsoluteResidual.model_validate(instance.model_dump()) == instance


class TestScore:
    """AbsoluteResidual.score returns correct values and shapes."""

    def test_hand_checked_values(self) -> None:
        score_fn = AbsoluteResidual()
        # (n_series=2, n_samples=3, horizon=2)
        prediction = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            ]
        )
        truth = np.array(
            [
                [[1.5, 1.0], [3.0, 5.0], [4.0, 6.0]],
                [[12.0, 18.0], [30.0, 45.0], [49.0, 60.0]],
            ]
        )
        scores = score_fn.score(prediction, truth)
        expected = np.array(
            [
                [[0.5, 1.0], [0.0, 1.0], [1.0, 0.0]],
                [[2.0, 2.0], [0.0, 5.0], [1.0, 0.0]],
            ]
        )
        np.testing.assert_array_almost_equal(scores, expected)

    def test_shape_preserved(self) -> None:
        score_fn = AbsoluteResidual()
        rng = np.random.default_rng(0)
        for n_series, n_samples, horizon in [(1, 10, 1), (5, 20, 12)]:
            pred = rng.standard_normal((n_series, n_samples, horizon))
            truth = rng.standard_normal((n_series, n_samples, horizon))
            scores = score_fn.score(pred, truth)
            assert scores.shape == (n_series, n_samples, horizon)


class TestInvert:
    """AbsoluteResidual.invert produces symmetric intervals."""

    def test_symmetric_interval_2d_threshold(self) -> None:
        score_fn = AbsoluteResidual()
        # prediction: (n_series=1, n_samples=2, horizon=3)
        prediction = np.array([[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]])
        # threshold: (n_series=1, horizon=3)
        threshold = np.array([[1.0, 2.0, 3.0]])

        interval = score_fn.invert(prediction, threshold)

        assert interval.shape == (1, 2, 3, 2)
        # lower = prediction - threshold
        np.testing.assert_array_almost_equal(
            interval[..., 0],
            np.array([[[9.0, 18.0, 27.0], [39.0, 48.0, 57.0]]]),
        )
        # upper = prediction + threshold
        np.testing.assert_array_almost_equal(
            interval[..., 1],
            np.array([[[11.0, 22.0, 33.0], [41.0, 52.0, 63.0]]]),
        )

    def test_scalar_threshold(self) -> None:
        score_fn = AbsoluteResidual()
        prediction = np.array([[[5.0, 10.0]]])  # (1, 1, 2)
        threshold = np.float64(2.0)

        interval = score_fn.invert(prediction, threshold)

        assert interval.shape == (1, 1, 2, 2)
        np.testing.assert_array_almost_equal(interval[0, 0, :, 0], [3.0, 8.0])
        np.testing.assert_array_almost_equal(interval[0, 0, :, 1], [7.0, 12.0])

    def test_1d_threshold(self) -> None:
        score_fn = AbsoluteResidual()
        # (n_series=2, n_samples=1, horizon=3)
        prediction = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
        # (n_series=2,) — same threshold across all horizons per series
        threshold = np.array([0.5, 1.0])

        interval = score_fn.invert(prediction, threshold)

        assert interval.shape == (2, 1, 3, 2)
        np.testing.assert_array_almost_equal(interval[0, 0, :, 0], [0.5, 1.5, 2.5])
        np.testing.assert_array_almost_equal(interval[1, 0, :, 0], [3.0, 4.0, 5.0])

    def test_output_shape(self) -> None:
        score_fn = AbsoluteResidual()
        rng = np.random.default_rng(42)
        for n_series, n_samples, horizon in [(1, 1, 1), (3, 10, 6)]:
            pred = rng.standard_normal((n_series, n_samples, horizon))
            threshold = rng.random((n_series, horizon))
            interval = score_fn.invert(pred, threshold)
            assert interval.shape == (n_series, n_samples, horizon, 2)
