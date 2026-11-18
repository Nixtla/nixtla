"""Tests for SplitConformal's array surface: shapes, determinism, lifecycle."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from conformalforecast.base import CalibrationResult
from conformalforecast.methods.fixed_offset import FixedOffsetForecaster
from conformalforecast.methods.split import SplitConformal
from conformalforecast.nonconformity.absolute import AbsoluteResidual
from conformalforecast.nonconformity.signed import SignedResidual


def _make_data(
    n_series: int, n_windows: int, horizon: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    preds = rng.normal(size=(n_series, n_windows, horizon))
    truths = preds + rng.normal(size=(n_series, n_windows, horizon))
    return preds, truths


class TestShapes:
    @pytest.mark.parametrize("n_series", [1, 5])
    @pytest.mark.parametrize("horizon", [1, 12])
    def test_fit_and_interval_shapes(self, n_series: int, horizon: int) -> None:
        preds, truths = _make_data(n_series, 50, horizon)
        fitted = SplitConformal(alpha=0.1).fit(preds, truths)

        assert isinstance(fitted, FixedOffsetForecaster)
        result = fitted.calibration_
        assert isinstance(result, CalibrationResult)
        assert result.n_calibration_samples.shape == (n_series,)
        assert (result.n_calibration_samples == 50).all()
        assert fitted.score_quantile_.shape == (n_series, horizon)

        new_preds = _make_data(n_series, 7, horizon, seed=1)[0]
        iv = fitted.interval(new_preds)
        assert iv.shape == (n_series, 7, horizon, 2)
        assert (iv[..., 0] <= iv[..., 1]).all()

    def test_interval_with_x_ignored(self) -> None:
        preds, truths = _make_data(2, 50, 3)
        X = np.zeros((2, 50, 3, 4))
        fitted = SplitConformal(alpha=0.1).fit(preds, truths, X=X)
        iv_with_x = fitted.interval(preds, X=np.zeros((2, 50, 3, 4)))
        iv_without = fitted.interval(preds)
        np.testing.assert_array_equal(iv_with_x, iv_without)

    def test_float32_preserved(self) -> None:
        preds, truths = _make_data(2, 50, 3)
        fitted = SplitConformal(alpha=0.1).fit(preds.astype(np.float32), truths.astype(np.float32))
        iv = fitted.interval(preds.astype(np.float32))
        assert iv.dtype == np.float32


class TestLifecycle:
    def test_fit_returns_new_object_and_leaves_method_untouched(self) -> None:
        preds, truths = _make_data(2, 50, 3)
        method = SplitConformal(alpha=0.1)
        fitted = method.fit(preds, truths)
        assert fitted is not method
        assert fitted.method_ is not method
        assert method == SplitConformal(**method.model_dump())

    def test_two_fits_are_independent(self) -> None:
        method = SplitConformal(alpha=0.1)
        preds_a, truths_a = _make_data(1, 50, 3, seed=0)
        preds_b, truths_b = _make_data(1, 50, 3, seed=1)
        a = method.fit(preds_a, truths_a)
        snapshot = a.score_quantile_.copy()
        b = method.fit(preds_b * 10.0, truths_b * 10.0)
        np.testing.assert_array_equal(a.score_quantile_, snapshot)
        assert not np.array_equal(a.score_quantile_, b.score_quantile_)

    def test_result_arrays_defensively_copied(self) -> None:
        preds, truths = _make_data(1, 50, 3)
        fitted = SplitConformal(alpha=0.1).fit(preds, truths)
        quantile_snapshot = fitted.calibration_.diagnostics["score_quantile"]
        assert isinstance(quantile_snapshot, np.ndarray)
        quantile_snapshot[:] = -999.0
        assert not np.any(fitted.score_quantile_ == -999.0)

    def test_interval_shape_mismatch_raises(self) -> None:
        # Test n_series mismatch: fit with 1 series, predict with 5
        preds, truths = _make_data(1, 30, 3)
        fitted = SplitConformal(alpha=0.1).fit(preds, truths)
        mismatched_preds = _make_data(5, 4, 3, seed=1)[0]
        with pytest.raises(ValueError, match="score_quantile"):
            fitted.interval(mismatched_preds)

        # Test horizon mismatch: fit with horizon 3, predict with horizon 12
        preds, truths = _make_data(1, 30, 3)
        fitted = SplitConformal(alpha=0.1).fit(preds, truths)
        mismatched_horizon = _make_data(1, 4, 12, seed=1)[0]
        with pytest.raises(ValueError, match="score_quantile"):
            fitted.interval(mismatched_horizon)


class TestDeterminism:
    def test_same_config_same_arrays_same_output(self) -> None:
        preds, truths = _make_data(3, 60, 4)
        k1 = SplitConformal(alpha=0.1).fit(preds, truths)
        k2 = SplitConformal(alpha=0.1).fit(preds, truths)
        np.testing.assert_array_equal(k1.score_quantile_, k2.score_quantile_)
        new_preds = _make_data(3, 5, 4, seed=9)[0]
        np.testing.assert_array_equal(k1.interval(new_preds), k2.interval(new_preds))


class TestCoverage:
    def test_empirical_coverage_matches_alpha(self) -> None:
        # Gaussian noise, alpha=0.1 -> target coverage 0.9.
        # Tolerance 0.03 with 3*500*4 = 6000 test cells (documented; matches
        # the tolerance used by tests/unit/test_split.py for the method layer).
        rng = np.random.default_rng(42)
        n_series, n_cal, n_test, horizon = 3, 500, 500, 4
        preds_cal = rng.normal(size=(n_series, n_cal, horizon))
        truths_cal = preds_cal + rng.normal(size=(n_series, n_cal, horizon))

        fitted = SplitConformal(alpha=0.1).fit(preds_cal, truths_cal)

        preds_test = rng.normal(size=(n_series, n_test, horizon))
        truths_test = preds_test + rng.normal(size=(n_series, n_test, horizon))
        iv = fitted.interval(preds_test)

        covered = (truths_test >= iv[..., 0]) & (truths_test <= iv[..., 1])
        assert abs(covered.mean() - 0.9) < 0.03


class TestNaNMasking:
    def test_ragged_equals_per_series_dense_fit(self) -> None:
        # Fitting a ragged panel must equal fitting each series separately
        # on its dense sub-array (spec: NaN-masking test requirement).
        rng = np.random.default_rng(7)
        horizon = 3
        lengths = [40, 25]  # series 1 is shorter
        n_windows = max(lengths)

        preds = np.full((2, n_windows, horizon), np.nan)
        truths = np.full((2, n_windows, horizon), np.nan)
        for i, n in enumerate(lengths):
            preds[i, -n:] = rng.normal(size=(n, horizon))
            truths[i, -n:] = preds[i, -n:] + rng.normal(size=(n, horizon))

        ragged = SplitConformal(alpha=0.1).fit(preds, truths)
        np.testing.assert_array_equal(ragged.calibration_.n_calibration_samples, lengths)

        for i, n in enumerate(lengths):
            dense = SplitConformal(alpha=0.1).fit(preds[i : i + 1, -n:], truths[i : i + 1, -n:])
            np.testing.assert_array_equal(ragged.score_quantile_[i : i + 1], dense.score_quantile_)

    def test_insufficient_series_named(self) -> None:
        from conformalforecast.base import InsufficientCalibrationSamples

        rng = np.random.default_rng(3)
        preds = rng.normal(size=(2, 20, 2))
        truths = preds + rng.normal(size=(2, 20, 2))
        # Series 1 keeps only 5 effective windows < ceil(1/0.1) = 10.
        preds[1, :15] = np.nan
        truths[1, :15] = np.nan

        method = SplitConformal(alpha=0.1)
        with pytest.raises(InsufficientCalibrationSamples) as excinfo:
            method.fit(preds, truths)
        assert excinfo.value.series_indices == [1]


class TestAsymmetric:
    def test_requires_both_levels(self) -> None:
        with pytest.raises(ValidationError, match="both be provided"):
            SplitConformal(alpha=0.1, symmetric=False, alpha_lo=0.05)

    def test_levels_must_bracket_the_median(self) -> None:
        with pytest.raises(ValidationError, match="0 < alpha_lo < 0.5 < alpha_hi < 1"):
            SplitConformal(alpha=0.1, symmetric=False, alpha_lo=0.6, alpha_hi=0.7)

    def test_levels_must_sum_to_alpha(self) -> None:
        with pytest.raises(ValidationError, match="must equal alpha"):
            SplitConformal(alpha=0.1, symmetric=False, alpha_lo=0.2, alpha_hi=0.95)

    def test_symmetric_true_rejects_explicit_levels(self) -> None:
        with pytest.raises(ValidationError, match="cannot be specified when symmetric=True"):
            SplitConformal(alpha=0.1, alpha_lo=0.02, alpha_hi=0.92)

    def test_symmetric_true_rejects_matching_equal_tailed_levels(self) -> None:
        """Deliberately differs from CQR, which accepts a matching pair: here
        the pair asks for per-tail coverage (the equal-tailed interval), which
        the equal-width symmetric interval does not provide."""
        with pytest.raises(ValidationError, match="symmetric=False, alpha_lo=0.05, alpha_hi=0.95"):
            SplitConformal(alpha=0.1, alpha_lo=0.05, alpha_hi=0.95)

    def test_default_score_is_signed(self) -> None:
        """A signed residual is what carries the asymmetry."""
        rng = np.random.default_rng(0)
        m = SplitConformal(alpha=0.1, symmetric=False, alpha_lo=0.05, alpha_hi=0.95)
        fitted = m.fit(rng.normal(size=(1, 200, 1)), rng.normal(size=(1, 200, 1)))
        assert isinstance(fitted.score_fn_, SignedResidual)
        assert fitted.score_quantile_.shape == (1, 1, 2)

    def test_rejects_a_symmetric_score(self) -> None:
        """An unsigned score makes both one-sided quantiles non-negative, and
        AbsoluteResidual.invert then broadcasts the (n_series, horizon, 2)
        quantile into a wrong-shaped interval instead of rejecting it."""
        with pytest.raises(ValidationError, match="symmetric=False requires a signed score"):
            SplitConformal(
                alpha=0.1,
                symmetric=False,
                alpha_lo=0.05,
                alpha_hi=0.95,
                score=AbsoluteResidual(),
            )

    def test_symmetric_still_accepts_any_score(self) -> None:
        """The gate narrows the asymmetric branch only."""
        rng = np.random.default_rng(0)
        m = SplitConformal(alpha=0.1, score=SignedResidual())
        fitted = m.fit(rng.normal(size=(1, 200, 1)), rng.normal(size=(1, 200, 1)))
        assert isinstance(fitted.score_fn_, SignedResidual)

    def test_shifts_the_interval_for_a_biased_forecaster(self) -> None:
        rng = np.random.default_rng(1)
        preds = np.zeros((1, 400, 1))
        truths = rng.normal(loc=3.0, scale=1.0, size=(1, 400, 1))
        m = SplitConformal(alpha=0.1, symmetric=False, alpha_lo=0.05, alpha_hi=0.95)
        with pytest.warns(UserWarning, match="entirely"):
            fitted = m.fit(preds, truths)
        iv = fitted.interval(np.zeros((1, 1, 1)))
        assert iv[0, 0, 0, 0] > 0.0  # the whole interval sits above the forecast

    def test_no_warning_for_an_unbiased_forecaster(self) -> None:
        import warnings

        rng = np.random.default_rng(2)
        preds = np.zeros((1, 400, 1))
        truths = rng.normal(size=(1, 400, 1))
        m = SplitConformal(alpha=0.1, symmetric=False, alpha_lo=0.05, alpha_hi=0.95)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m.fit(preds, truths)

    def test_coverage_is_nominal(self) -> None:
        rng = np.random.default_rng(3)
        preds = np.zeros((1, 2000, 1))
        truths = rng.normal(size=(1, 2000, 1))
        m = SplitConformal(alpha=0.1, symmetric=False, alpha_lo=0.05, alpha_hi=0.95)
        fitted = m.fit(preds, truths)
        test_truths = rng.normal(size=(1, 4000, 1))
        iv = fitted.interval(np.zeros((1, 4000, 1)))
        covered = ((test_truths >= iv[..., 0]) & (test_truths <= iv[..., 1])).mean()
        assert abs(covered - 0.9) < 0.02  # 4000 draws: +/- ~0.01 at 2 sigma

    def test_symmetric_path_unchanged(self) -> None:
        """The default must produce the exact interval it did before."""
        rng = np.random.default_rng(4)
        preds = rng.normal(size=(2, 100, 3))
        truths = rng.normal(size=(2, 100, 3))
        m = SplitConformal(alpha=0.1)
        fitted = m.fit(preds, truths)
        assert m.symmetric is True
        assert isinstance(fitted.score_fn_, AbsoluteResidual)
        assert fitted.score_quantile_.shape == (2, 3)
        iv = fitted.interval(preds[:, :5])
        centre = preds[:, :5]
        np.testing.assert_allclose(iv[..., 1] - centre, centre - iv[..., 0])

    def test_per_tail_floor_is_stricter_than_ceil_1_over_alpha(self) -> None:
        """Each tail's one-sided quantile enforces its own ceil(1/level) floor.

        15 windows clears ceil(1/alpha) = ceil(1/0.1) = 10 but not the
        per-tail floor ceil(1/alpha_lo) = ceil(1/0.05) = 20, so asymmetric
        mode must raise where symmetric mode, given the same data, fits fine.
        """
        from conformalforecast.base import InsufficientCalibrationSamples

        rng = np.random.default_rng(5)
        preds = rng.normal(size=(1, 15, 1))
        truths = preds + rng.normal(size=(1, 15, 1))

        asymmetric = SplitConformal(alpha=0.1, symmetric=False, alpha_lo=0.05, alpha_hi=0.95)
        with pytest.raises(InsufficientCalibrationSamples, match="alpha_lo") as excinfo:
            asymmetric.fit(preds, truths)
        assert excinfo.value.series_indices == [0]

        # Same data, coarser floor: fits fine.
        fitted = SplitConformal(alpha=0.1).fit(preds, truths)
        assert isinstance(fitted, FixedOffsetForecaster)

    def test_upper_tail_error_names_the_value_the_user_set(self) -> None:
        """The upper tail's quantile is taken at 1 - alpha_hi, so the message
        must quote alpha_hi itself rather than the derived level."""
        from conformalforecast.base import InsufficientCalibrationSamples

        rng = np.random.default_rng(15)
        preds = rng.normal(size=(1, 15, 1))
        truths = preds + rng.normal(size=(1, 15, 1))

        # alpha_lo=0.09 clears its floor of ceil(1/0.09) = 12; the upper tail's
        # 1 - alpha_hi = 0.01 does not (floor 100), so it is the tail that raises.
        m = SplitConformal(alpha=0.1, symmetric=False, alpha_lo=0.09, alpha_hi=0.99)
        with pytest.raises(InsufficientCalibrationSamples, match=r"from alpha_hi=0\.99"):
            m.fit(preds, truths)

    def test_float32_preserved(self) -> None:
        rng = np.random.default_rng(6)
        preds = rng.normal(size=(2, 60, 3)).astype(np.float32)
        truths = (preds + rng.normal(size=(2, 60, 3))).astype(np.float32)
        m = SplitConformal(alpha=0.1, symmetric=False, alpha_lo=0.05, alpha_hi=0.95)
        fitted = m.fit(preds, truths)
        assert fitted.score_quantile_.dtype == np.float32
        iv = fitted.interval(preds[:, :5])
        assert iv.dtype == np.float32

    def test_ragged_equals_per_series_dense_fit(self) -> None:
        rng = np.random.default_rng(7)
        horizon = 2
        lengths = [40, 25]  # series 1 is shorter
        n_windows = max(lengths)

        preds = np.full((2, n_windows, horizon), np.nan)
        truths = np.full((2, n_windows, horizon), np.nan)
        for i, n in enumerate(lengths):
            preds[i, -n:] = rng.normal(size=(n, horizon))
            truths[i, -n:] = preds[i, -n:] + rng.normal(size=(n, horizon))

        m = SplitConformal(alpha=0.1, symmetric=False, alpha_lo=0.05, alpha_hi=0.95)
        ragged = m.fit(preds, truths)
        np.testing.assert_array_equal(ragged.calibration_.n_calibration_samples, lengths)

        for i, n in enumerate(lengths):
            dense = m.fit(preds[i : i + 1, -n:], truths[i : i + 1, -n:])
            np.testing.assert_array_equal(ragged.score_quantile_[i : i + 1], dense.score_quantile_)


def test_asymmetric_round_trips_through_model_dump() -> None:
    """The asymmetric path, where the levels are genuine user config."""
    m = SplitConformal(alpha=0.1, symmetric=False, alpha_lo=0.02, alpha_hi=0.92)
    assert SplitConformal.model_validate(m.model_dump()) == m


def test_asymmetric_without_levels_raises() -> None:
    """``model_copy`` skips validation, so the missing levels surface in ``fit``."""
    preds, truths = _make_data(2, 50, 3)
    m = SplitConformal(alpha=0.1).model_copy(update={"symmetric": False})
    with pytest.raises(ValueError, match="skipped validation"):
        m.fit(preds, truths)


class TestCalibrationIntervalsDefault:
    """The base-class default: broadcast the fitted state over every window.

    That is exact for any method whose offsets are a constant of the whole
    calibration set, which is every shipped method except SPCI.
    """

    def test_identical_to_interval(self) -> None:
        preds, truths = _make_data(3, 50, 4)
        fitted = SplitConformal(alpha=0.1).fit(preds, truths)
        np.testing.assert_array_equal(fitted.calibration_intervals(preds), fitted.interval(preds))

    def test_no_cell_is_unevaluable(self) -> None:
        preds, truths = _make_data(2, 40, 3)
        fitted = SplitConformal(alpha=0.1).fit(preds, truths)
        assert not np.isnan(fitted.calibration_intervals(preds)).any()
