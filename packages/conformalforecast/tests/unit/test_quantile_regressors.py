"""Tests for the QuantileRegressor ABC and QRFQuantileRegressor."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import PrivateAttr

from conformalforecast.quantile_regressors.base import QuantileRegressor
from conformalforecast.quantile_regressors.qrf import QRFQuantileRegressor, _load_backend

try:
    importlib.import_module("quantile_forest")

    _HAS_QF = True
except ImportError:
    _HAS_QF = False

qf_required = pytest.mark.skipif(not _HAS_QF, reason="quantile-forest is not installed")


class MockQuantileRegressor(QuantileRegressor):
    """Returns ``np.quantile(y, q)`` unconditionally on the fitted target.

    Useful as the test workhorse for the SPCI pipeline: it exercises the
    full :class:`QuantileRegressor` ABC without pulling in the
    quantile-forest dependency, and behaves like an empirical quantile of
    calibration residuals (so SPCI's coverage degrades to split-conformal
    semantics on iid data).
    """

    _y: NDArray[np.floating] | None = PrivateAttr(default=None)

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> None:
        self._y = np.asarray(y, dtype=np.float64)

    def predict_quantile(self, X: NDArray[np.floating], q: float) -> NDArray[np.floating]:
        if self._y is None:
            raise RuntimeError("MockQuantileRegressor must be fit first.")
        return np.full(X.shape[0], float(np.quantile(self._y, q)), dtype=np.float64)


# ===========================================================================
# Mock regressor (ABC sanity)
# ===========================================================================


class TestMockRegressor:
    """The mock exercises the ABC contract without external dependencies."""

    def test_fit_and_predict(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        y = rng.normal(size=50)

        reg = MockQuantileRegressor()
        reg.fit(X, y)
        out_lo = reg.predict_quantile(X[:5], 0.1)
        out_hi = reg.predict_quantile(X[:5], 0.9)

        assert out_lo.shape == (5,)
        assert out_hi.shape == (5,)
        # Mock returns the empirical quantile of y broadcast across rows.
        np.testing.assert_allclose(out_lo, np.full(5, float(np.quantile(y, 0.1))))
        np.testing.assert_allclose(out_hi, np.full(5, float(np.quantile(y, 0.9))))

    def test_predict_before_fit_raises(self) -> None:
        reg = MockQuantileRegressor()
        with pytest.raises(RuntimeError, match="fit"):
            reg.predict_quantile(np.zeros((1, 3)), 0.5)


# ===========================================================================
# Backend import is deferred to fit(), not construction
# ===========================================================================


class TestBackendLoadDeferredToFit:
    """The [spci] ImportError must surface at fit(), not construction.

    Construction has to stay side-effect-free because
    SequentialPredictiveConformalInference builds a QRFQuantileRegressor as
    its *default* ``regressor``, and a pydantic default that imports an
    optional dependency to be validated would make
    ``SequentialPredictiveConformalInference.model_json_schema()`` raise on a
    machine without ``quantile-forest`` installed.
    """

    def test_construction_does_not_import_backend(self) -> None:
        reg = QRFQuantileRegressor(n_estimators=5, random_state=0)
        assert reg._forest is None

    def test_fit_surfaces_missing_backend_as_import_error(self, monkeypatch) -> None:
        from conformalforecast.quantile_regressors import qrf as qrf_module

        def _raise() -> None:
            raise ImportError(
                "QRFQuantileRegressor requires the 'quantile-forest' package. "
                "Install it with: pip install conformalforecast[spci]"
            )

        monkeypatch.setattr(qrf_module, "_load_backend", _raise)
        reg = QRFQuantileRegressor(n_estimators=5, random_state=0)
        with pytest.raises(ImportError, match=r"conformalforecast\[spci\]"):
            reg.fit(np.zeros((3, 2)), np.zeros(3))

    @qf_required
    def test_load_backend_resolves_the_real_estimator(self) -> None:
        assert _load_backend().__name__ == "RandomForestQuantileRegressor"


# ===========================================================================
# Equality (extra="allow" passthrough)
# ===========================================================================


class TestQRFEquality:
    """``QRFQuantileRegressor`` is the one class with ``extra="allow"``."""

    def test_passthrough_kwargs_count_towards_equality(self) -> None:
        """Kwargs land in ``__pydantic_extra__``, not ``__dict__``.

        Without ``ArrayModel._comparable_fields`` folding the extras in,
        two regressors that forward different arguments to the backend
        would compare equal — construction needs no backend, so this runs
        without ``quantile-forest`` installed.
        """
        a = QRFQuantileRegressor(max_depth=3)
        b = QRFQuantileRegressor(max_depth=7)
        assert a == QRFQuantileRegressor(max_depth=3)
        assert a != b


# ===========================================================================
# QRFQuantileRegressor (gated on quantile-forest install)
# ===========================================================================


@qf_required
class TestQRFRegressor:
    """QRF fits and predicts on synthetic data, respects quantile ordering."""

    def test_fit_and_predict_shape(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        y = rng.normal(size=200)

        reg = QRFQuantileRegressor(n_estimators=20, random_state=0)
        reg.fit(X, y)

        out = reg.predict_quantile(X[:7], 0.5)
        assert out.shape == (7,)
        assert out.dtype == np.float64

    def test_quantile_ordering_on_monotone_data(self) -> None:
        """Lower-q predictions should not exceed higher-q predictions."""
        rng = np.random.default_rng(0)
        n = 300
        X = rng.normal(size=(n, 2))
        # Target depends on X[:, 0] plus heteroskedastic noise so quantiles
        # vary across rows.
        y = X[:, 0] + rng.normal(size=n) * (1.0 + np.abs(X[:, 1]))

        reg = QRFQuantileRegressor(n_estimators=50, random_state=0)
        reg.fit(X, y)

        # Predict on a fresh sample of rows so the test isn't trivially
        # satisfied by training-set memorization.
        X_test = rng.normal(size=(30, 2))
        q_lo = reg.predict_quantile(X_test, 0.1)
        q_hi = reg.predict_quantile(X_test, 0.9)
        assert (q_lo <= q_hi + 1e-9).all()

    def test_predict_before_fit_raises(self) -> None:
        reg = QRFQuantileRegressor(n_estimators=10, random_state=0)
        with pytest.raises(RuntimeError, match="fit"):
            reg.predict_quantile(np.zeros((2, 3)), 0.5)


# ===========================================================================
# Batched predict_quantiles
# ===========================================================================


class TestBatchedQuantilesFallback:
    """The ABC's default loops over the scalar method, so custom regressors work."""

    def test_default_matches_scalar_calls(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 3))
        y = rng.normal(size=40)
        reg = MockQuantileRegressor()
        reg.fit(X, y)

        qs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        batched = reg.predict_quantiles(X[:6], qs)
        assert batched.shape == (6, qs.shape[0])
        for j, q in enumerate(qs):
            np.testing.assert_array_equal(batched[:, j], reg.predict_quantile(X[:6], float(q)))

    def test_single_quantile_keeps_two_dimensions(self) -> None:
        rng = np.random.default_rng(1)
        reg = MockQuantileRegressor()
        reg.fit(rng.normal(size=(20, 2)), rng.normal(size=20))
        assert reg.predict_quantiles(np.zeros((4, 2)), np.array([0.5])).shape == (4, 1)


@qf_required
class TestQRFBatchedQuantiles:
    """QRF overrides the default with one forest traversal; results must agree."""

    def test_matches_per_level_calls(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        y = X[:, 0] + rng.normal(size=200)
        reg = QRFQuantileRegressor(n_estimators=25, random_state=0)
        reg.fit(X, y)

        X_test = rng.normal(size=(15, 3))
        qs = np.array([0.02, 0.1, 0.5, 0.9, 0.98])
        batched = reg.predict_quantiles(X_test, qs)
        assert batched.shape == (15, qs.shape[0])
        for j, q in enumerate(qs):
            np.testing.assert_allclose(batched[:, j], reg.predict_quantile(X_test, float(q)))

    def test_predict_before_fit_raises(self) -> None:
        reg = QRFQuantileRegressor(n_estimators=10, random_state=0)
        with pytest.raises(RuntimeError, match="fit"):
            reg.predict_quantiles(np.zeros((2, 3)), np.array([0.5]))


# ===========================================================================
# pydantic round-trip
# ===========================================================================


def test_config_round_trips_through_model_dump() -> None:
    """``model_validate(model_dump())`` reproduces an unfitted instance.

    ``_forest`` is a ``PrivateAttr``, so it never enters ``model_dump()`` --
    the round trip only has to reproduce the three declared config fields,
    same as the six ``ConformalMethod`` table in
    ``tests/unit/methods/test_pydantic_conversion.py``.
    """
    reg = QRFQuantileRegressor(n_estimators=17, min_samples_leaf=3, random_state=2)
    assert QRFQuantileRegressor.model_validate(reg.model_dump()) == reg


def test_extra_kwargs_survive_the_round_trip() -> None:
    """``model_extra`` (the RandomForestQuantileRegressor passthrough) round-trips too.

    Both sides of the final ``==`` carry the *same* extra (``max_depth=4``),
    so this holds whether or not ``__pydantic_extra__`` counts towards
    equality -- it does not exercise that merge. See
    ``test_extra_kwargs_count_toward_equality`` for a test that does.
    """
    reg = QRFQuantileRegressor(n_estimators=10, max_depth=4)
    dumped = reg.model_dump()
    assert dumped["max_depth"] == 4
    assert QRFQuantileRegressor.model_validate(dumped) == reg


def test_extra_kwargs_count_toward_equality() -> None:
    """``__pydantic_extra__`` values must count towards ``ArrayModel.__eq__``.

    ``max_depth`` is not a declared field -- it lives only in
    ``__pydantic_extra__`` (``model_config = ConfigDict(extra="allow")``).
    Two instances differing only there are the only diverging state, so if
    ``ArrayModel._comparable_fields`` dropped extras, this would wrongly
    return equal.
    """
    assert QRFQuantileRegressor(max_depth=3) != QRFQuantileRegressor(max_depth=9)
