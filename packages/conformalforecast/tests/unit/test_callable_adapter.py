"""Tests for CallableAdapter."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from conformalforecast.adapters.callable import CallableAdapter
from conformalforecast.capabilities import (
    SupportsBootstrap,
    SupportsCrossValidation,
    SupportsCrossValidationQuantiles,
    SupportsQuantiles,
    SupportsRefit,
)


def _naive_forecast(history: np.ndarray) -> np.ndarray:
    """Repeat the last value for each series across the horizon."""
    # history: (n_series, T) -> (n_series, horizon)
    # We'll use a closure to pass horizon; tests create lambdas instead.
    return np.repeat(history[:, -1:], history.shape[0], axis=1)


class TestConstruction:
    """CallableAdapter construction validation."""

    def test_valid_construction(self) -> None:
        adapter = CallableAdapter(
            predict_fn=lambda h: h[:, -1:],
            horizon=1,
            n_series=1,
        )
        assert adapter.horizon == 1
        assert adapter.n_series == 1

    def test_non_callable_raises(self) -> None:
        with pytest.raises(ValueError, match="callable"):
            CallableAdapter(predict_fn="not a function", horizon=1, n_series=1)  # type: ignore[arg-type]

    def test_invalid_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            CallableAdapter(predict_fn=lambda h: h, horizon=0, n_series=1)

    def test_invalid_n_series_raises(self) -> None:
        with pytest.raises(ValueError, match="n_series"):
            CallableAdapter(predict_fn=lambda h: h, horizon=1, n_series=0)

    def test_adapter_construction_is_keyword_only_and_validated(self) -> None:
        with pytest.raises(TypeError):
            CallableAdapter(_naive_forecast, 3, 2)  # type: ignore[misc]

        adapter = CallableAdapter(predict_fn=_naive_forecast, horizon=3, n_series=2)
        assert adapter.horizon == 3
        assert adapter.n_series == 2

        for bad in ({"horizon": 0}, {"n_series": 0}):
            with pytest.raises(ValidationError):
                CallableAdapter(predict_fn=_naive_forecast, **{"horizon": 3, "n_series": 2, **bad})

    def test_extra_kwargs_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            CallableAdapter(
                predict_fn=_naive_forecast,
                horizon=3,
                n_series=2,
                horzon=3,  # type: ignore[call-arg]
            )

    def test_adapter_schema_describes_horizon_and_n_series(self) -> None:
        # predict_fn is SkipJsonSchema-hidden: pydantic cannot render a
        # JsonSchema for a callable, and no schema-filling caller could supply
        # one anyway.
        props = CallableAdapter.model_json_schema()["properties"]
        assert props["horizon"]["minimum"] == 1
        assert props["n_series"]["minimum"] == 1
        assert props["horizon"]["description"]
        assert "predict_fn" not in props


class TestCapabilityMixins:
    """The mixins are models, and ``supports()`` still gates on ``isinstance``."""

    def test_mixins_are_models_and_isinstance_still_gates_capabilities(self) -> None:
        adapter = CallableAdapter(predict_fn=_naive_forecast, horizon=3, n_series=2)
        assert adapter.supports(SupportsCrossValidation) is False
        assert issubclass(SupportsQuantiles, BaseModel)

    def test_mixins_cannot_be_instantiated(self) -> None:
        for cap in (
            SupportsRefit,
            SupportsQuantiles,
            SupportsBootstrap,
            SupportsCrossValidation,
            SupportsCrossValidationQuantiles,
        ):
            with pytest.raises(TypeError):
                cap()  # type: ignore[abstract]


class TestPredict:
    """CallableAdapter.predict shape and validation."""

    @pytest.mark.parametrize(
        "n_series,horizon",
        [(1, 1), (1, 12), (5, 1), (5, 12)],
    )
    def test_output_shape(self, n_series: int, horizon: int) -> None:
        adapter = CallableAdapter(
            predict_fn=lambda h, hz=horizon: np.ones((h.shape[0], hz)),
            horizon=horizon,
            n_series=n_series,
        )
        history = np.random.default_rng(0).standard_normal((n_series, 50))
        result = adapter.predict(history)
        assert result.shape == (n_series, 1, horizon)

    def test_wrong_history_ndim_raises(self) -> None:
        adapter = CallableAdapter(
            predict_fn=lambda h: h[:, -1:],
            horizon=1,
            n_series=1,
        )
        with pytest.raises(ValueError, match="2-D"):
            adapter.predict(np.array([1.0, 2.0, 3.0]))

    def test_wrong_history_n_series_raises(self) -> None:
        adapter = CallableAdapter(
            predict_fn=lambda h: h[:, -1:],
            horizon=1,
            n_series=2,
        )
        with pytest.raises(ValueError, match="leading axis"):
            adapter.predict(np.ones((3, 10)))

    def test_bad_predict_fn_output_shape_raises(self) -> None:
        adapter = CallableAdapter(
            predict_fn=lambda h: h,  # returns (n_series, T), wrong horizon
            horizon=5,
            n_series=1,
        )
        with pytest.raises(ValueError, match="predict_fn must return shape"):
            adapter.predict(np.ones((1, 10)))


class TestPredictBatch:
    """CallableAdapter.predict_batch stacks correctly."""

    @pytest.mark.parametrize("n_series", [1, 5])
    def test_output_shape(self, n_series: int) -> None:
        horizon = 6
        adapter = CallableAdapter(
            predict_fn=lambda h, hz=horizon: np.ones((h.shape[0], hz)),
            horizon=horizon,
            n_series=n_series,
        )
        rng = np.random.default_rng(0)
        histories = [rng.standard_normal((n_series, 50)) for _ in range(10)]
        result = adapter.predict_batch(histories)
        assert result.shape == (n_series, 10, horizon)


class TestExogForwarding:
    """CallableAdapter forwards exogenous X iff the wrapped callable accepts it."""

    def test_callable_without_x_ignores_exog(self) -> None:
        adapter = CallableAdapter(
            predict_fn=lambda h: np.zeros((h.shape[0], 2)), horizon=2, n_series=1
        )
        X = np.ones((1, 1, 2, 3))
        out = adapter.predict(np.array([[1.0, 2.0, 3.0]]), X=X)
        assert out.shape == (1, 1, 2)

    def test_callable_with_x_receives_exog(self) -> None:
        seen = {}

        def fn(h, X=None):
            seen["X"] = X
            return np.zeros((h.shape[0], 2))

        adapter = CallableAdapter(predict_fn=fn, horizon=2, n_series=1)
        X = np.ones((1, 1, 2, 3))
        adapter.predict(np.array([[1.0, 2.0, 3.0]]), X=X)
        assert seen["X"] is not None
        np.testing.assert_array_equal(seen["X"], X)

    def test_predict_batch_slices_x_per_window(self) -> None:
        received: list[np.ndarray] = []

        def fn(h, X=None):
            received.append(X)
            return np.zeros((h.shape[0], 2))

        adapter = CallableAdapter(predict_fn=fn, horizon=2, n_series=1)
        hists = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])]
        X = np.arange(1 * 2 * 2 * 1, dtype=np.float64).reshape(1, 2, 2, 1)
        adapter.predict_batch(hists, X=X)
        assert len(received) == 2
        np.testing.assert_array_equal(received[0], X[:, 0:1, :, :])
        np.testing.assert_array_equal(received[1], X[:, 1:2, :, :])
