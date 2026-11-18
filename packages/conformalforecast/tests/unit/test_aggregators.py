"""Tests for online expert aggregators."""

from __future__ import annotations

import numpy as np
import pytest

from conformalforecast.aggregators import EWA, OnlineAggregator
from conformalforecast.base import CalibrationError


def _initialized(n_experts=4, n_series=2, horizon=3, eta=1.0) -> EWA:
    agg = EWA(eta=eta)
    agg.initialize(n_experts=n_experts, n_series=n_series, horizon=horizon)
    return agg


class TestEWAConstruction:
    """EWA no longer needs a panel shape to construct."""

    def test_ewa_constructs_without_a_panel_shape(self) -> None:
        agg = EWA(eta=2.0)
        assert agg.eta == 2.0
        assert agg.cumulative_losses_ is None

    def test_weights_before_initialize_raises(self) -> None:
        with pytest.raises(CalibrationError, match="initialize"):
            EWA().weights()

    def test_update_before_initialize_raises(self) -> None:
        with pytest.raises(CalibrationError, match="initialize"):
            EWA().update(np.zeros((3, 2, 4)))

    def test_initialize_allocates_and_validates(self) -> None:
        agg = EWA()
        agg.initialize(n_experts=3, n_series=2, horizon=4)
        assert agg.cumulative_losses_.shape == (3, 2, 4)
        assert np.allclose(agg.cumulative_losses_, 0.0)
        assert np.allclose(agg.weights().sum(axis=0), 1.0)

        with pytest.raises(ValueError, match="n_experts must be >= 1"):
            EWA().initialize(n_experts=0, n_series=2, horizon=4)


class TestEWAInitialization:
    """Initial state of an EWA aggregator."""

    def test_initial_weights_are_uniform(self) -> None:
        agg = _initialized(n_experts=4, n_series=2, horizon=3, eta=1.0)
        w = agg.weights()
        assert w.shape == (4, 2, 3)
        np.testing.assert_allclose(w, np.full_like(w, 1 / 4), rtol=1e-12)

    def test_initial_cumulative_losses_zero(self) -> None:
        agg = _initialized(n_experts=3, n_series=1, horizon=2, eta=0.5)
        np.testing.assert_array_equal(agg.cumulative_losses_, np.zeros((3, 1, 2)))


class TestEWAUpdates:
    """Behaviour of weights as cumulative losses accumulate."""

    def test_lower_loss_expert_dominates(self) -> None:
        """The expert with consistently lower loss should monotonically gain weight."""
        agg = _initialized(n_experts=3, n_series=1, horizon=1, eta=1.0)

        prev_winner_weight = float(agg.weights()[0, 0, 0])
        for _ in range(10):
            losses = np.array([[[0.1]], [[0.5]], [[0.9]]], dtype=np.float64)
            agg.update(losses)
            current_winner_weight = float(agg.weights()[0, 0, 0])
            assert current_winner_weight > prev_winner_weight - 1e-12
            prev_winner_weight = current_winner_weight

        final_weights = agg.weights()
        assert final_weights[0, 0, 0] > final_weights[1, 0, 0]
        assert final_weights[1, 0, 0] > final_weights[2, 0, 0]

    def test_weights_sum_to_one_along_expert_axis(self) -> None:
        rng = np.random.default_rng(0)
        agg = _initialized(n_experts=5, n_series=2, horizon=4, eta=0.7)
        for _ in range(20):
            losses = rng.uniform(0.0, 1.0, size=(5, 2, 4))
            agg.update(losses)
            sums = agg.weights().sum(axis=0)
            np.testing.assert_allclose(sums, np.ones((2, 4)), rtol=1e-12)

    def test_numerical_stability_large_losses(self) -> None:
        """eta * cumulative_loss may have magnitude 1e6; weights must remain finite."""
        agg = _initialized(n_experts=4, n_series=2, horizon=3, eta=1.0)
        losses = np.array(
            [
                np.full((2, 3), 1e6),
                np.full((2, 3), 5e5),
                np.full((2, 3), 1e5),
                np.full((2, 3), 0.0),
            ],
            dtype=np.float64,
        )
        agg.update(losses)
        w = agg.weights()
        assert np.all(np.isfinite(w))
        assert not np.any(np.isnan(w))
        np.testing.assert_allclose(w.sum(axis=0), np.ones((2, 3)), rtol=1e-12)
        # Best expert (lowest loss) gets ~all the weight.
        assert w[3, 0, 0] > 0.999


class TestEWAValidation:
    """Input-validation paths for EWA."""

    def test_wrong_shape_losses_raises(self) -> None:
        agg = _initialized(n_experts=2, n_series=1, horizon=3, eta=1.0)
        with pytest.raises(ValueError, match="losses must have shape"):
            agg.update(np.zeros((2, 1, 4)))
        with pytest.raises(ValueError, match="losses must have shape"):
            agg.update(np.zeros((3, 1, 3)))

    @pytest.mark.parametrize("eta", [0.0, -0.5])
    def test_invalid_eta_raises(self, eta: float) -> None:
        with pytest.raises(ValueError, match="eta"):
            EWA(eta=eta)

    @pytest.mark.parametrize(
        "n_experts,n_series,horizon",
        [(0, 1, 1), (1, 0, 1), (1, 1, 0)],
    )
    def test_invalid_dimensions_raises(self, n_experts: int, n_series: int, horizon: int) -> None:
        with pytest.raises(ValueError):
            EWA().initialize(n_experts=n_experts, n_series=n_series, horizon=horizon)


class TestSubclassability:
    """OnlineAggregator can be subclassed by user code."""

    def test_custom_subclass_works(self) -> None:
        class UniformAggregator(OnlineAggregator):
            def weights(self) -> np.ndarray:
                cumulative = self._require_initialized()
                del cumulative
                w = np.full(
                    (self.n_experts, self.n_series, self.horizon),
                    1.0 / self.n_experts,
                    dtype=np.float64,
                )
                return w

        agg = UniformAggregator()
        agg.initialize(n_experts=3, n_series=2, horizon=4)
        w = agg.weights()
        np.testing.assert_allclose(w, np.full_like(w, 1 / 3), rtol=1e-12)
        # update() still validates shape.
        with pytest.raises(ValueError, match="losses must have shape"):
            agg.update(np.zeros((2, 2, 4)))
