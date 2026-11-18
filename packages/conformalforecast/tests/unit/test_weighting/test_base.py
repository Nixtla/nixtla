import numpy as np
import pytest
from pydantic import ValidationError

from conformalforecast.weighting import (
    DensityRatio,
    DensityRatioEstimator,
    PrecomputedWeights,
    UniformWeights,
)


class _Stub:
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.tile(np.array([0.5, 0.5]), (len(X), 1))


def test_stub_satisfies_protocol_without_subclassing():
    assert isinstance(_Stub(), DensityRatioEstimator)


def test_uniform_weights_are_ones_and_never_adapted():
    w = UniformWeights()
    w.fit_source(None, (2, 5, 3))
    assert w.is_adapted_ is False
    np.testing.assert_array_equal(w.calibration_weights(), np.ones((2, 5, 3)))
    np.testing.assert_array_equal(w.test_weights(None), np.ones((2, 1, 3)))


def test_uniform_test_weights_follow_supplied_exog_shape():
    w = UniformWeights()
    w.fit_source(None, (2, 5, 3))
    X = np.zeros((2, 7, 3, 4))
    np.testing.assert_array_equal(w.test_weights(X), np.ones((2, 7, 3)))


def test_precomputed_weights_returned_verbatim():
    given = np.arange(30, dtype=np.float64).reshape(2, 5, 3)
    w = PrecomputedWeights(weights=given)
    w.fit_source(None, (2, 5, 3))
    np.testing.assert_array_equal(w.calibration_weights(), given)
    assert w.is_adapted_ is True


def test_precomputed_weights_reject_shape_mismatch():
    w = PrecomputedWeights(weights=np.ones((2, 5, 3)))
    with pytest.raises(ValueError, match="shape"):
        w.fit_source(None, (2, 6, 3))


def test_precomputed_weights_reject_negative():
    with pytest.raises(ValueError, match="non-negative"):
        PrecomputedWeights(weights=-np.ones((2, 5, 3)))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_precomputed_weights_reject_non_finite(bad_value):
    w = np.ones((2, 5, 3))
    w[0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        PrecomputedWeights(weights=w)


def test_calibration_weights_before_fit_source_raises():
    with pytest.raises(RuntimeError, match="fit_source"):
        UniformWeights().calibration_weights()


def test_weight_function_forbids_extra_keywords():
    """A typo'd knob must fail at construction, not be silently ignored."""
    with pytest.raises(ValidationError):
        UniformWeights(nonexistent_knob=1)


def test_density_ratio_rejects_a_non_estimator():
    """The Protocol is runtime-checkable, so pydantic gates on it."""
    with pytest.raises(ValidationError):
        DensityRatio(estimator="not an estimator")


@pytest.mark.parametrize("kwargs", [{"cv": -1}, {"clip_quantile": 0.0}, {"clip_quantile": 1.5}])
def test_density_ratio_field_bounds(kwargs):
    """Field constraints replace the hand-written raise-checks."""
    with pytest.raises(ValidationError):
        DensityRatio(estimator=_Stub(), **kwargs)


def test_weight_functions_compare_elementwise_safely():
    """ArrayModel.__eq__ must not raise on the array-valued field."""
    a = PrecomputedWeights(weights=np.ones((1, 2, 3)))
    b = PrecomputedWeights(weights=np.ones((1, 2, 3)))
    assert a == b
    assert a != PrecomputedWeights(weights=np.zeros((1, 2, 3)))


def test_fit_source_rejects_a_mismatched_valid_mask():
    w = UniformWeights()
    with pytest.raises(ValueError, match="valid mask shape"):
        w.fit_source(None, (2, 5, 3), valid=np.ones((2, 4, 3), dtype=bool))


def test_precomputed_weights_forwards_the_valid_mask():
    """PrecomputedWeights overrides fit_source for its own shape check; the
    mask must still reach the base so a subclass can read valid_."""
    valid = np.ones((2, 5, 3), dtype=bool)
    valid[0, :2] = False
    w = PrecomputedWeights(weights=np.ones((2, 5, 3)))
    w.fit_source(None, (2, 5, 3), valid=valid)
    np.testing.assert_array_equal(w.valid_, valid)
