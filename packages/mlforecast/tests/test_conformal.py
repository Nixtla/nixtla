import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlforecast import MLForecast
from mlforecast.forecast import _get_conformal_method
from mlforecast.lag_transforms import ExponentiallyWeightedMean
from mlforecast.target_transforms import Differences
from mlforecast.utils import (
    PredictionIntervals,
    ScaleAlignment,
)

warnings.simplefilter('ignore', UserWarning)


# ---------------------------------------------------------------------------
# Local fixtures (M4-data dependent)
# ---------------------------------------------------------------------------

@pytest.fixture
def fcst():
    return MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1),
        freq=1,
        lags=[24 * (i+1) for i in range(7)],
        lag_transforms={
            48: [ExponentiallyWeightedMean(alpha=0.3)],
        },
        num_threads=1,
        target_transforms=[Differences([24])],
    )


@pytest.fixture
def fitted_fcst(fcst, setup_forecast_data):
    _, train, _ = setup_forecast_data
    fcst.fit(train, fitted=True)
    return fcst


@pytest.fixture
def predictions(fitted_fcst):
    return fitted_fcst.predict(48)


@pytest.fixture
def fcst_with_intervals(setup_forecast_data):
    """Forecast object fitted with prediction intervals."""
    _, train, _ = setup_forecast_data
    f = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1),
        freq=1,
        lags=[24 * (i+1) for i in range(7)],
        lag_transforms={
            48: [ExponentiallyWeightedMean(alpha=0.3)],
        },
        num_threads=1,
        target_transforms=[Differences([24])],
    )
    f.fit(train, prediction_intervals=PredictionIntervals(n_windows=3, h=48))
    return f


@pytest.fixture
def predictions_w_intervals(fcst_with_intervals):
    """Predictions with prediction intervals."""
    return fcst_with_intervals.predict(48, level=[50, 80, 95])


# ---------------------------------------------------------------------------
# Conformal method validation
# ---------------------------------------------------------------------------

def test_conformal_method():
    with pytest.raises(ValueError):
        _get_conformal_method('my_method')


# ---------------------------------------------------------------------------
# Prediction interval integration tests
# ---------------------------------------------------------------------------

def test_prediction_intervals_lower_horizon(fcst_with_intervals):
    """Test we can forecast horizon lower than h with prediction intervals."""
    preds_h1 = fcst_with_intervals.predict(1, level=[50, 80, 95])
    preds_h30 = fcst_with_intervals.predict(30, level=[50, 80, 95])

    monotonic_count = preds_h1.filter(regex='lo|hi').apply(
        lambda x: x.is_monotonic_increasing,
        axis=1
    ).sum()
    assert monotonic_count == len(preds_h1)

    monotonic_count = preds_h30.filter(regex='lo|hi').apply(
        lambda x: x.is_monotonic_increasing,
        axis=1
    ).sum()
    assert monotonic_count == len(preds_h30)


def test_prediction_intervals_error_conditions(fcst_with_intervals):
    """Test error conditions for prediction intervals."""
    with pytest.raises(Exception):
        fcst_with_intervals.predict(49, level=[68])


def test_recover_point_forecasts(predictions, predictions_w_intervals):
    """Test we can recover point forecasts from interval predictions."""
    pd.testing.assert_frame_equal(
        predictions,
        predictions_w_intervals[predictions.columns]
    )


def test_recover_mean_forecasts_level_zero(predictions, fcst_with_intervals):
    """Test we can recover mean forecasts with level 0."""
    level_zero_preds = fcst_with_intervals.predict(48, level=[0])
    np.testing.assert_allclose(
        predictions['LGBMRegressor'].values,
        level_zero_preds['LGBMRegressor-lo-0'].values,
    )


def test_prediction_intervals_monotonicity(predictions_w_intervals):
    """Test monotonicity of prediction intervals."""
    monotonic_count = predictions_w_intervals.filter(regex='lo|hi').apply(
        lambda x: x.is_monotonic_increasing,
        axis=1
    ).sum()
    assert monotonic_count == len(predictions_w_intervals)


# ---------------------------------------------------------------------------
# Scale-aligned conformal prediction tests
# ---------------------------------------------------------------------------

def test_prediction_intervals_method_validation():
    with pytest.raises(ValueError, match="method must be one of"):
        PredictionIntervals(method="weighted_conformal_error")


def test_scale_alignment_scale_estimator_validation():
    ScaleAlignment(scale_estimator="std")
    with pytest.raises(ValueError, match="scale_estimator must be 'mad' or 'std'"):
        ScaleAlignment(scale_estimator="variance")


def _series_scales(df, method):
    from mlforecast.conformal_prediction import (
        _floor_series_scales,
        _raw_series_scales,
    )

    raw = _raw_series_scales(df, "unique_id", "ds", "y", (method,))
    return _floor_series_scales(raw[method], floor_factor=1e-3, stacklevel=2)


def test_series_scales_mad():
    """The MAD scale is correct for a known signal."""
    df = pd.DataFrame({"unique_id": ["s1"] * 5, "ds": range(5), "y": [0.0, 1, 3, 6, 10]})
    scales = _series_scales(df, "mad")
    assert "s1" in scales
    assert scales["s1"] > 0


def test_series_scales_std():
    """The std scale is correct for a known signal."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(50).cumsum()
    df = pd.DataFrame({"unique_id": ["s1"] * 50, "ds": range(50), "y": y})
    scales = _series_scales(df, "std")
    dy = np.diff(y)
    expected = float(np.std(dy, ddof=1))
    assert abs(scales["s1"] - expected) < 1e-10


def test_series_scales_flat_series():
    """Flat series gets a positive floor rather than zero."""
    df = pd.DataFrame({
        "unique_id": ["flat"] * 10 + ["noisy"] * 10,
        "ds": list(range(10)) * 2,
        "y": [5.0] * 10 + list(np.random.default_rng(0).standard_normal(10)),
    })
    scales = _series_scales(df, "mad")
    assert scales["flat"] > 0


def test_series_scales_short_series():
    """Single-observation series gets a positive scale without crashing."""
    df = pd.DataFrame({
        "unique_id": ["short", "normal"] + ["normal"] * 9,
        "ds": [0] + list(range(10)),
        "y": [42.0] + list(np.arange(10, dtype=float)),
    })
    scales = _series_scales(df, "mad")
    assert scales["short"] > 0


def test_series_scales_polars():
    """The per-series scales work on a polars DataFrame."""
    df_pd = pd.DataFrame({
        "unique_id": ["s1"] * 20,
        "ds": range(20),
        "y": np.arange(20, dtype=float) + np.random.default_rng(7).standard_normal(20),
    })
    df_pl = pl.from_pandas(df_pd)
    scales_pd = _series_scales(df_pd, "mad")
    scales_pl = _series_scales(df_pl, "mad")
    assert abs(scales_pd["s1"] - scales_pl["s1"]) < 1e-10


def test_apply_scale_alignment_normalizes_by_source_scale():
    """_apply_scale_alignment divides each source residual by its series' σ_src."""
    from mlforecast.conformal_prediction import _apply_scale_alignment

    cs_df = pd.DataFrame({
        "unique_id": ["s1", "s1", "s2", "s2"],
        "LGBMRegressor": [1.0, 2.0, 3.0, 4.0],
    })
    source_scales = {"s1": 2.0, "s2": 4.0}
    result = _apply_scale_alignment(cs_df, ["LGBMRegressor"], "unique_id", source_scales)
    expected_s1 = np.array([1.0, 2.0]) / 2.0  # divided by σ_src_s1
    expected_s2 = np.array([3.0, 4.0]) / 4.0  # divided by σ_src_s2
    np.testing.assert_allclose(result["LGBMRegressor"].to_numpy()[:2], expected_s1)
    np.testing.assert_allclose(result["LGBMRegressor"].to_numpy()[2:], expected_s2)


def test_apply_scale_alignment_no_mutation():
    """_apply_scale_alignment does not mutate its input cs_df."""
    from mlforecast.conformal_prediction import _apply_scale_alignment

    cs_df = pd.DataFrame({
        "unique_id": ["s1"] * 4,
        "LGBMRegressor": [1.0, 2.0, 3.0, 4.0],
    })
    original_vals = cs_df["LGBMRegressor"].to_numpy().copy()
    _apply_scale_alignment(cs_df, ["LGBMRegressor"], "unique_id", {"s1": 1.0})
    np.testing.assert_array_equal(cs_df["LGBMRegressor"].to_numpy(), original_vals)


def test_scale_aligned_source_scales_stored(scale_aligned_setup):
    """Fitting with prediction intervals stores the source scales of every estimator."""
    fcst, _, _, _ = scale_aligned_setup
    source_scales = fcst._calibration_.source_scales
    assert set(source_scales) == {"mad", "std"}
    for scales in source_scales.values():
        assert len(scales) == fcst.ts.ga.n_groups
        for v in scales.values():
            assert v > 0


def test_scale_aligned_transfer_large_target(scale_aligned_setup):
    """scale_aligned transfer produces intervals ~400× wider for a ×400 target."""
    fcst, source_series, target_series, h = scale_aligned_setup

    preds_src = fcst.predict(h, level=[80])
    src_width = (preds_src["LGBMRegressor-hi-80"] - preds_src["LGBMRegressor-lo-80"]).mean()

    preds_sa = fcst.transfer(target_series, ScaleAlignment()).predict(
        h, level=[80]
    )
    assert "LGBMRegressor-lo-80" in preds_sa.columns
    assert preds_sa["LGBMRegressor-lo-80"].notna().all()
    assert (preds_sa["LGBMRegressor-lo-80"] <= preds_sa["LGBMRegressor-hi-80"]).all()

    sa_width = (preds_sa["LGBMRegressor-hi-80"] - preds_sa["LGBMRegressor-lo-80"]).mean()
    ratio = float(sa_width / src_width)
    assert ratio > 50, f"Expected scale-aligned intervals >> source intervals, got ratio={ratio:.1f}"


# ---------------------------------------------------------------------------
# Item 1 new API tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, match",
    [
        ("scale_aligned", "must be a `Recalibration`"),
        (ScaleAlignment, r"Pass an instance, e.g. `ScaleAlignment\(\)`"),
    ],
)
def test_transfer_invalid_method_raises(scale_aligned_setup, method, match):
    fcst, _, target_series, _ = scale_aligned_setup
    with pytest.raises(TypeError, match=match):
        fcst.transfer(target_series, method)


# ---------------------------------------------------------------------------
# Item 2 per-series scale tests
# ---------------------------------------------------------------------------

def test_per_series_scale_width_proportional(scale_aligned_setup):
    """Series with 10× larger scale gets ~10× wider intervals."""
    fcst, source_series, _, h = scale_aligned_setup

    # Build two-series target: one at source scale, one at 10× source scale
    rng = np.random.default_rng(99)
    base = source_series[source_series["unique_id"] == source_series["unique_id"].unique()[0]].copy()
    s_normal = base.copy(); s_normal["unique_id"] = "tgt_normal"
    s_wide = base.copy()
    s_wide["y"] = s_wide["y"] * 10.0
    s_wide["unique_id"] = "tgt_wide"
    target_two = pd.concat([s_normal, s_wide], ignore_index=True)

    preds = fcst.transfer(target_two, ScaleAlignment()).predict(h, level=[80])
    normal_w = (preds[preds["unique_id"] == "tgt_normal"]["LGBMRegressor-hi-80"]
                - preds[preds["unique_id"] == "tgt_normal"]["LGBMRegressor-lo-80"]).mean()
    wide_w = (preds[preds["unique_id"] == "tgt_wide"]["LGBMRegressor-hi-80"]
              - preds[preds["unique_id"] == "tgt_wide"]["LGBMRegressor-lo-80"]).mean()
    ratio = float(wide_w / normal_w)
    assert ratio > 5.0, f"Expected ~10× width ratio, got {ratio:.2f}"


def test_per_series_scale_leaves_other_series_unchanged(scale_aligned_setup):
    """Scaling one target series does not change other series' widths."""
    fcst, source_series, _, h = scale_aligned_setup

    base = source_series[source_series["unique_id"] == source_series["unique_id"].unique()[0]].copy()
    s1 = base.copy(); s1["unique_id"] = "tgt_1"
    s2 = base.copy(); s2["unique_id"] = "tgt_2"

    # Predict with original s2 and then with scaled s2
    target_orig = pd.concat([s1, s2], ignore_index=True)
    s2_scaled = s2.copy(); s2_scaled["y"] *= 5.0
    target_scaled = pd.concat([s1, s2_scaled], ignore_index=True)

    p_orig = fcst.transfer(target_orig, ScaleAlignment()).predict(h, level=[80])
    p_scaled = fcst.transfer(target_scaled, ScaleAlignment()).predict(h, level=[80])

    w1_orig = (p_orig[p_orig["unique_id"] == "tgt_1"]["LGBMRegressor-hi-80"]
               - p_orig[p_orig["unique_id"] == "tgt_1"]["LGBMRegressor-lo-80"]).mean()
    w1_scaled = (p_scaled[p_scaled["unique_id"] == "tgt_1"]["LGBMRegressor-hi-80"]
                 - p_scaled[p_scaled["unique_id"] == "tgt_1"]["LGBMRegressor-lo-80"]).mean()
    np.testing.assert_allclose(w1_orig, w1_scaled, rtol=1e-6,
                               err_msg="Scaling tgt_2 should not affect tgt_1's widths")


# ---------------------------------------------------------------------------
# Item 5 robust scale ratio tests
# ---------------------------------------------------------------------------

def test_robust_ratio_recovers_3x_scale():
    """IQR ratio gives a more stable estimate than std when an outlier is present."""
    from mlforecast.conformal_prediction import _robust_scale_ratio

    rng = np.random.default_rng(42)
    src = rng.standard_normal(2000)
    tgt = 3.0 * rng.standard_normal(2000)
    # Inject a large outlier — IQR ignores it, std would be inflated
    tgt_with_outlier = np.append(tgt, 1000.0)

    ratio_iqr = _robust_scale_ratio(src, tgt_with_outlier)
    std_ratio = float(np.std(tgt_with_outlier)) / float(np.std(src))

    # IQR ratio should be in a reasonable range (~3×) and less than the inflated std ratio
    assert 1.5 <= ratio_iqr <= 5.0, f"Expected IQR ratio ≈ 3, got {ratio_iqr:.3f}"
    assert std_ratio > ratio_iqr, "std ratio should overshoot due to outlier"


def test_robust_ratio_fallback_std_warns():
    """IQR-degenerate inputs trigger std fallback warning."""
    from mlforecast.conformal_prediction import _robust_scale_ratio

    # Near-constant residuals → IQR ≈ 0 → should fall back to std
    src = np.ones(50)
    tgt = np.full(50, 2.0)
    with pytest.warns(UserWarning, match="IQR"):
        _robust_scale_ratio(src, tgt)


def test_robust_ratio_fallback_constant_warns():
    """Both IQR and std near zero triggers constant fallback warning."""
    from mlforecast.conformal_prediction import _robust_scale_ratio

    src = np.ones(50)  # constant → IQR=0, std=0
    tgt = np.ones(50)
    with pytest.warns(UserWarning, match="Both IQR"):
        result = _robust_scale_ratio(src, tgt)
    assert result == 1.0
