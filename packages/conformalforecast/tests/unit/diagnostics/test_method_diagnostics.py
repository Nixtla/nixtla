"""Tests for diagnostics.method_diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from conformalforecast.adapters.callable import CallableAdapter
from conformalforecast.diagnostics.method_diagnostics import (
    aci_state,
    agaci_state,
    method_state,
    nexcp_state,
    spci_state,
)
from conformalforecast.methods.aci import ACIForecaster, AdaptiveConformalInference
from conformalforecast.methods.agaci import AgACIForecaster, AggregatedAdaptiveConformalInference
from conformalforecast.methods.nexcp import NexCPForecaster, NonexchangeableConformalPrediction
from conformalforecast.methods.spci import SequentialPredictiveConformalInference, SPCIForecaster
from conformalforecast.methods.split import SplitConformal
from tests.helpers import MockQuantileRegressor


def _zero_predict_fn(n_series: int, horizon: int):
    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.zeros((n_series, horizon))

    return predict_fn


def _make_iid(rng: np.random.Generator, n_series: int, horizon: int, n: int):
    histories = [rng.normal(size=(n_series, 20)) for _ in range(n)]
    truths = rng.normal(size=(n_series, n, horizon))
    return histories, truths


# ===========================================================================
# aci_state
# ===========================================================================


class TestAciState:
    def _calibrated(self, gamma: float = 0.05) -> ACIForecaster:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 3
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AdaptiveConformalInference(alpha=0.1, gamma=gamma)
        histories, truths = _make_iid(rng, n_series, horizon, 60)
        fitted = method.calibrate(adapter, histories, truths)
        return fitted

    def test_keys_and_shapes(self) -> None:
        fitted = self._calibrated()
        state = aci_state(fitted)
        assert set(state.keys()) == {
            "alpha_t",
            "alpha_t_minus_alpha",
            "n_observations",
            "gamma",
        }
        assert state["alpha_t"].shape == (2, 3)
        assert state["alpha_t_minus_alpha"].shape == (2, 3)
        assert isinstance(state["n_observations"], int)
        assert state["gamma"] == pytest.approx(0.05)

    def test_drift_changes_after_updates(self) -> None:
        fitted = self._calibrated()
        before = aci_state(fitted)["alpha_t_minus_alpha"].copy()
        # Force misses by passing huge truths.
        for _ in range(20):
            result = fitted.predict(np.zeros((2, 20)))
            fitted.update(result.point, np.full((2, 1, 3), 1e6, dtype=np.float64))
        after = aci_state(fitted)["alpha_t_minus_alpha"]
        assert not np.allclose(before, after)

    def test_wrong_type_raises(self) -> None:
        split = SplitConformal(alpha=0.1)
        with pytest.raises(TypeError, match="ACIForecaster"):
            aci_state(split)  # type: ignore[arg-type]

    def test_method_rather_than_fitted_raises(self) -> None:
        """The unfitted method has no state; the fitted object is what to pass."""
        method = AdaptiveConformalInference(alpha=0.1, gamma=0.05)
        with pytest.raises(TypeError, match="ACIForecaster"):
            aci_state(method)  # type: ignore[arg-type]

    def test_split_fitted_is_the_wrong_type(self) -> None:
        rng = np.random.default_rng(0)
        preds = rng.normal(size=(1, 50, 1))
        fitted = SplitConformal(alpha=0.1).fit(preds, preds + rng.normal(size=(1, 50, 1)))
        with pytest.raises(TypeError, match="ACIForecaster"):
            aci_state(fitted)  # type: ignore[arg-type]


# ===========================================================================
# agaci_state
# ===========================================================================


class TestAgaciState:
    def _calibrated(self) -> AgACIForecaster:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 3
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AggregatedAdaptiveConformalInference(alpha=0.1, gammas=(0.01, 0.05, 0.1))
        histories, truths = _make_iid(rng, n_series, horizon, 60)
        fitted = method.calibrate(adapter, histories, truths)
        return fitted

    def test_keys_and_shapes(self) -> None:
        fitted = self._calibrated()
        state = agaci_state(fitted)
        expected_keys = {
            "alpha_t_per_expert",
            "gammas",
            "weights_lower",
            "weights_upper",
            "best_expert_per_cell_lower",
            "best_expert_per_cell_upper",
            "n_observations",
        }
        assert set(state.keys()) == expected_keys
        n_experts, n_series, horizon = 3, 2, 3
        assert state["alpha_t_per_expert"].shape == (n_experts, n_series, horizon)
        assert state["weights_lower"].shape == (n_experts, n_series, horizon)
        assert state["weights_upper"].shape == (n_experts, n_series, horizon)
        assert state["best_expert_per_cell_lower"].shape == (n_series, horizon)
        assert state["best_expert_per_cell_upper"].shape == (n_series, horizon)
        assert state["gammas"] == (0.01, 0.05, 0.1)

    def test_weights_sum_to_one_per_cell(self) -> None:
        fitted = self._calibrated()
        state = agaci_state(fitted)
        np.testing.assert_allclose(state["weights_lower"].sum(axis=0), 1.0, atol=1e-12)
        np.testing.assert_allclose(state["weights_upper"].sum(axis=0), 1.0, atol=1e-12)

    def test_wrong_type_raises(self) -> None:
        split = SplitConformal(alpha=0.1)
        with pytest.raises(TypeError, match="AgACIForecaster"):
            agaci_state(split)  # type: ignore[arg-type]


# ===========================================================================
# nexcp_state
# ===========================================================================


class TestNexcpState:
    def _calibrated(self, rho: float = 0.99) -> NexCPForecaster:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 2
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = NonexchangeableConformalPrediction(alpha=0.1, rho=rho)
        histories, truths = _make_iid(rng, n_series, horizon, 100)
        fitted = method.calibrate(adapter, histories, truths)
        return fitted

    def test_ess_equals_n_when_rho_one(self) -> None:
        fitted = self._calibrated(rho=1.0)
        state = nexcp_state(fitted)
        assert state["effective_sample_size"] == pytest.approx(state["n_observations"])

    def test_ess_lt_n_when_rho_lt_one(self) -> None:
        fitted = self._calibrated(rho=0.99)
        state = nexcp_state(fitted)
        n = state["n_observations"]
        assert 50 < state["effective_sample_size"] < n

    def test_keys_and_weight_bounds(self) -> None:
        fitted = self._calibrated(rho=0.99)
        state = nexcp_state(fitted)
        expected_keys = {
            "rho",
            "n_observations",
            "effective_sample_size",
            "weights_minimum",
            "weights_maximum",
            "score_quantile",
        }
        assert set(state.keys()) == expected_keys
        assert state["weights_maximum"] == pytest.approx(1.0)
        assert state["weights_minimum"] < state["weights_maximum"]
        assert state["score_quantile"].shape == (1, 2)

    def test_wrong_type_raises(self) -> None:
        split = SplitConformal(alpha=0.1)
        with pytest.raises(TypeError, match="NexCPForecaster"):
            nexcp_state(split)  # type: ignore[arg-type]


# ===========================================================================
# spci_state
# ===========================================================================


class TestSpciState:
    def _calibrated(self) -> SPCIForecaster:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 3
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = SequentialPredictiveConformalInference(
            alpha=0.1,
            window_size=10,
            regressor=MockQuantileRegressor(),
            beta_grid_size=5,
            refit_every=2,
        )
        histories, truths = _make_iid(rng, n_series, horizon, 80)
        fitted = method.calibrate(adapter, histories, truths)
        return fitted

    def test_keys_and_values(self) -> None:
        fitted = self._calibrated()
        state = spci_state(fitted)
        expected_keys = {
            "window_size",
            "n_observations",
            "n_regressors",
            "regressor_class",
            "refit_every",
            "steps_since_refit",
            "residuals_shape",
        }
        assert set(state.keys()) == expected_keys
        assert state["n_regressors"] == 2 * 3
        assert state["regressor_class"] == "MockQuantileRegressor"
        assert state["window_size"] == 10
        assert state["refit_every"] == 2
        assert state["residuals_shape"] == (2, 80, 3)

    def test_wrong_type_raises(self) -> None:
        split = SplitConformal(alpha=0.1)
        with pytest.raises(TypeError, match="SPCIForecaster"):
            spci_state(split)  # type: ignore[arg-type]


# ===========================================================================
# method_state dispatcher
# ===========================================================================


class TestMethodStateDispatch:
    def test_aci(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AdaptiveConformalInference(alpha=0.1, gamma=0.05)
        histories, truths = _make_iid(rng, 1, 1, 50)
        fitted = method.calibrate(adapter, histories, truths)
        state = method_state(fitted)
        assert "alpha_t" in state

    def test_split_returns_empty_dict(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SplitConformal(alpha=0.1)
        histories, truths = _make_iid(rng, 1, 1, 50)
        fitted = method.calibrate(adapter, histories, truths)
        state = method_state(fitted)
        assert state == {}
