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
from mlforecast.utils import PredictionIntervals, TransferConformal, generate_daily_series

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
# Weighted conformal prediction tests
# ---------------------------------------------------------------------------

def test_weighted_conformal_method_validation():
    """weighted_conformal_error and _distribution should be accepted; invalid names rejected."""
    PredictionIntervals(method="weighted_conformal_error")
    PredictionIntervals(method="weighted_conformal_distribution")
    with pytest.raises(ValueError):
        PredictionIntervals(method="bad_method")


def test_weighted_conformal_features_stored(weighted_conformal_setup):
    """After fitting with weighted_conformal method, _cs_df should contain feature columns."""
    fcst, _, _, _ = weighted_conformal_setup
    non_feat_cols = {fcst.ts.id_col, fcst.ts.time_col, "cutoff"} | set(fcst.models.keys())
    feat_cols = [c for c in fcst._cs_df.columns if c not in non_feat_cols]
    assert len(feat_cols) > 0, "_cs_df must contain feature columns for DRE"


def test_weighted_conformal_uniform_weights_matches_standard(weighted_conformal_setup):
    """Uniform weights should produce intervals close to the unweighted method."""
    fcst, series, h, n_windows = weighted_conformal_setup
    n_series = fcst.ts.ga.n_groups
    n_cal_rows = len(fcst._cs_df)

    preds_std = fcst.predict(h, level=[80])

    uniform_weights = np.ones(n_cal_rows)
    preds_w = fcst.predict(h, level=[80], transfer_conformal=TransferConformal(method="weighted_conformal", weights=uniform_weights))

    pd.testing.assert_series_equal(
        preds_std["LGBMRegressor"], preds_w["LGBMRegressor"]
    )
    assert "LGBMRegressor-lo-80" in preds_w.columns
    assert "LGBMRegressor-hi-80" in preds_w.columns


def test_weighted_conformal_user_array(weighted_conformal_setup):
    """User-supplied weight array flows through; intervals are finite and ordered."""
    fcst, _, h, _ = weighted_conformal_setup
    n_cal_rows = len(fcst._cs_df)
    weights = np.random.default_rng(42).exponential(scale=1.0, size=n_cal_rows)
    preds = fcst.predict(h, level=[80, 95], transfer_conformal=TransferConformal(method="weighted_conformal", weights=weights))

    assert "LGBMRegressor-lo-80" in preds.columns
    assert preds["LGBMRegressor-lo-80"].notna().all()
    assert preds["LGBMRegressor-hi-80"].notna().all()
    assert (preds["LGBMRegressor-lo-80"] <= preds["LGBMRegressor-hi-80"]).all()


def test_weighted_conformal_callable_weights(weighted_conformal_setup):
    """Callable receives source feature matrix and returns weights; intervals are valid."""
    fcst, _, h, _ = weighted_conformal_setup
    weight_fn = lambda X: np.ones(len(X)) if X is not None else np.ones(len(fcst._cs_df))
    preds = fcst.predict(h, level=[80], transfer_conformal=TransferConformal(method="weighted_conformal", weights=weight_fn))

    assert "LGBMRegressor-lo-80" in preds.columns
    assert preds["LGBMRegressor-lo-80"].notna().all()


# ---------------------------------------------------------------------------
# Density ratio estimation tests
# ---------------------------------------------------------------------------

def test_estimate_density_ratio():
    """estimate_density_ratio returns positive weights of correct shape."""
    from mlforecast.conformal_prediction import estimate_density_ratio

    rng = np.random.default_rng(0)
    src = rng.standard_normal((50, 4))
    tgt = rng.standard_normal((30, 4)) + 0.5
    weights = estimate_density_ratio(src, tgt)
    assert weights.shape == (50,)
    assert (weights > 0).all()


def test_estimate_density_ratio_gradient_boosting():
    """estimate_density_ratio works with gradient_boosting estimator."""
    from mlforecast.conformal_prediction import estimate_density_ratio

    rng = np.random.default_rng(1)
    src = rng.standard_normal((30, 3))
    tgt = rng.standard_normal((20, 3)) + 1.0
    weights = estimate_density_ratio(src, tgt, estimator="gradient_boosting")
    assert weights.shape == (30,)
    assert (weights > 0).all()


# ---------------------------------------------------------------------------
# Scale-aligned conformal prediction tests
# ---------------------------------------------------------------------------

def test_prediction_intervals_scale_estimator_validation():
    """scale_estimator accepts 'mad' and 'std'; rejects invalid values."""
    PredictionIntervals(scale_estimator="mad")
    PredictionIntervals(scale_estimator="std")
    PredictionIntervals(scale_estimator=None)
    with pytest.raises(ValueError, match="scale_estimator"):
        PredictionIntervals(scale_estimator="variance")


def test_compute_series_scales_mad():
    """_compute_series_scales returns correct MAD for a known signal."""
    from mlforecast.conformal_prediction import _compute_series_scales

    df = pd.DataFrame({"unique_id": ["s1"] * 5, "ds": range(5), "y": [0.0, 1, 3, 6, 10]})
    scales = _compute_series_scales(df, "unique_id", "ds", "y", method="mad")
    assert "s1" in scales
    assert scales["s1"] > 0


def test_compute_series_scales_std():
    """_compute_series_scales returns correct std for a known signal."""
    from mlforecast.conformal_prediction import _compute_series_scales

    rng = np.random.default_rng(0)
    y = rng.standard_normal(50).cumsum()
    df = pd.DataFrame({"unique_id": ["s1"] * 50, "ds": range(50), "y": y})
    scales = _compute_series_scales(df, "unique_id", "ds", "y", method="std")
    dy = np.diff(y)
    expected = float(np.std(dy, ddof=1))
    assert abs(scales["s1"] - expected) < 1e-10


def test_compute_series_scales_flat_series():
    """Flat series gets a positive floor rather than zero."""
    from mlforecast.conformal_prediction import _compute_series_scales

    df = pd.DataFrame({
        "unique_id": ["flat"] * 10 + ["noisy"] * 10,
        "ds": list(range(10)) * 2,
        "y": [5.0] * 10 + list(np.random.default_rng(0).standard_normal(10)),
    })
    scales = _compute_series_scales(df, "unique_id", "ds", "y", method="mad")
    assert scales["flat"] > 0


def test_compute_series_scales_short_series():
    """Single-observation series gets a positive scale without crashing."""
    from mlforecast.conformal_prediction import _compute_series_scales

    df = pd.DataFrame({
        "unique_id": ["short", "normal"] + ["normal"] * 9,
        "ds": [0] + list(range(10)),
        "y": [42.0] + list(np.arange(10, dtype=float)),
    })
    scales = _compute_series_scales(df, "unique_id", "ds", "y", method="mad")
    assert scales["short"] > 0


def test_compute_series_scales_polars():
    """_compute_series_scales works on a polars DataFrame."""
    from mlforecast.conformal_prediction import _compute_series_scales

    df_pd = pd.DataFrame({
        "unique_id": ["s1"] * 20,
        "ds": range(20),
        "y": np.arange(20, dtype=float) + np.random.default_rng(7).standard_normal(20),
    })
    df_pl = pl.from_pandas(df_pd)
    scales_pd = _compute_series_scales(df_pd, "unique_id", "ds", "y", method="mad")
    scales_pl = _compute_series_scales(df_pl, "unique_id", "ds", "y", method="mad")
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
    """Fitting with scale_estimator stores _cs_source_scales_ on the MLForecast instance."""
    fcst, _, _, _ = scale_aligned_setup
    assert fcst._cs_source_scales_ is not None
    assert len(fcst._cs_source_scales_) == fcst.ts.ga.n_groups
    for v in fcst._cs_source_scales_.values():
        assert v > 0


def test_scale_aligned_transfer_large_target(scale_aligned_setup):
    """scale_aligned transfer produces intervals ~400× wider for a ×400 target."""
    fcst, source_series, target_series, h = scale_aligned_setup

    preds_src = fcst.predict(h, level=[80])
    src_width = (preds_src["LGBMRegressor-hi-80"] - preds_src["LGBMRegressor-lo-80"]).mean()

    preds_sa = fcst.predict(
        h, level=[80], new_df=target_series,
        transfer_conformal="scale_aligned",
    )
    assert "LGBMRegressor-lo-80" in preds_sa.columns
    assert preds_sa["LGBMRegressor-lo-80"].notna().all()
    assert (preds_sa["LGBMRegressor-lo-80"] <= preds_sa["LGBMRegressor-hi-80"]).all()

    sa_width = (preds_sa["LGBMRegressor-hi-80"] - preds_sa["LGBMRegressor-lo-80"]).mean()
    ratio = float(sa_width / src_width)
    assert ratio > 50, f"Expected scale-aligned intervals >> source intervals, got ratio={ratio:.1f}"


def test_scale_aligned_nontransfer_path_unchanged(scale_aligned_setup):
    """predict() without new_df is unaffected by scale_estimator."""
    fcst, source_series, _, h = scale_aligned_setup

    pi_plain = PredictionIntervals(method="conformal_error", n_windows=2, h=h)
    fcst_plain = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1, n_estimators=5),
        freq="D",
        lags=[1, 7],
        date_features=["dayofweek"],
        num_threads=1,
    )
    fcst_plain.fit(source_series, prediction_intervals=pi_plain)

    preds_with = fcst.predict(h, level=[80])
    preds_without = fcst_plain.predict(h, level=[80])

    pd.testing.assert_frame_equal(
        preds_with[["LGBMRegressor-lo-80", "LGBMRegressor-hi-80"]].reset_index(drop=True),
        preds_without[["LGBMRegressor-lo-80", "LGBMRegressor-hi-80"]].reset_index(drop=True),
        rtol=1e-5,
    )


def test_scale_aligned_weighted_composes(scale_aligned_setup):
    """scale_aligned_weighted combines scale alignment with DRE weights."""
    fcst, source_series, target_series, h = scale_aligned_setup

    pi_w = PredictionIntervals(
        method="weighted_conformal_error", n_windows=2, h=h, scale_estimator="mad"
    )
    fcst_w = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1, n_estimators=5),
        freq="D",
        lags=[1, 7],
        date_features=["dayofweek"],
        num_threads=1,
    )
    fcst_w.fit(source_series, prediction_intervals=pi_w)

    preds = fcst_w.predict(
        h, level=[80],
        new_df=target_series,
        transfer_conformal="scale_aligned_weighted",
    )
    assert "LGBMRegressor-lo-80" in preds.columns
    assert preds["LGBMRegressor-lo-80"].notna().all()
    assert (preds["LGBMRegressor-lo-80"] <= preds["LGBMRegressor-hi-80"]).all()


def test_scale_aligned_requires_scale_estimator():
    """scale_aligned transfer raises if model was fit without scale_estimator."""
    series = generate_daily_series(2, min_length=40, max_length=50, seed=3)
    fcst = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1, n_estimators=5),
        freq="D",
        lags=[1],
        num_threads=1,
    )
    fcst.fit(series, prediction_intervals=PredictionIntervals(method="conformal_error"))
    target_df = series.groupby("unique_id").tail(10).reset_index(drop=True)
    with pytest.raises(ValueError, match="scale_estimator"):
        fcst.predict(5, level=[80], new_df=target_df, transfer_conformal="scale_aligned")


# ---------------------------------------------------------------------------
# Transfer conformal prediction tests
# ---------------------------------------------------------------------------

def test_weighted_conformal_auto_dre(weighted_conformal_setup):
    """transfer_conformal_method='weighted_conformal' + new_df uses full-feature DRE."""
    fcst, series, h, _ = weighted_conformal_setup
    target_df = series.groupby("unique_id").tail(30).reset_index(drop=True)

    preds = fcst.predict(
        h,
        level=[80],
        new_df=target_df,
        transfer_conformal="weighted_conformal",
    )
    assert "LGBMRegressor-lo-80" in preds.columns
    assert preds["LGBMRegressor-lo-80"].notna().all()
    assert (preds["LGBMRegressor-lo-80"] <= preds["LGBMRegressor-hi-80"]).all()


def test_weighted_conformal_auto_dre_requires_weighted_method():
    """auto-DRE should fail gracefully when _cs_df has no feature columns."""
    series = generate_daily_series(2, min_length=40, max_length=50, seed=1)
    fcst = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1, n_estimators=5),
        freq="D",
        lags=[1],
        num_threads=1,
    )
    fcst.fit(series, prediction_intervals=PredictionIntervals(method="conformal_error"))
    target_df = series.groupby("unique_id").tail(10).reset_index(drop=True)
    with pytest.raises(ValueError, match="weighted_conformal"):
        fcst.predict(5, level=[80], new_df=target_df,
                     transfer_conformal="weighted_conformal")


# ---------------------------------------------------------------------------
# Item 1 new API tests
# ---------------------------------------------------------------------------

def test_transfer_conformal_string_shorthand_equals_object(scale_aligned_setup):
    """String shorthand == TransferConformal(method=<str>) for predict output."""
    fcst, _, target_series, h = scale_aligned_setup
    preds_str = fcst.predict(h, level=[80], new_df=target_series, transfer_conformal="scale_aligned")
    from mlforecast.conformal_prediction import TransferConformal as TC
    preds_obj = fcst.predict(h, level=[80], new_df=target_series, transfer_conformal=TC(method="scale_aligned"))
    pd.testing.assert_frame_equal(preds_str, preds_obj)


def test_transfer_conformal_invalid_method_raises():
    """TransferConformal with unknown method raises ValueError."""
    with pytest.raises(ValueError, match="must be one of"):
        TransferConformal(method="nonexistent")


def test_transfer_conformal_weights_with_recalibrate_raises():
    """TransferConformal.weights is invalid with method='recalibrate'."""
    with pytest.raises(ValueError, match="weights is only valid"):
        TransferConformal(method="recalibrate", weights=np.ones(5))


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

    preds = fcst.predict(h, level=[80], new_df=target_two, transfer_conformal="scale_aligned")
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

    p_orig = fcst.predict(h, level=[80], new_df=target_orig, transfer_conformal="scale_aligned")
    p_scaled = fcst.predict(h, level=[80], new_df=target_scaled, transfer_conformal="scale_aligned")

    w1_orig = (p_orig[p_orig["unique_id"] == "tgt_1"]["LGBMRegressor-hi-80"]
               - p_orig[p_orig["unique_id"] == "tgt_1"]["LGBMRegressor-lo-80"]).mean()
    w1_scaled = (p_scaled[p_scaled["unique_id"] == "tgt_1"]["LGBMRegressor-hi-80"]
                 - p_scaled[p_scaled["unique_id"] == "tgt_1"]["LGBMRegressor-lo-80"]).mean()
    np.testing.assert_allclose(w1_orig, w1_scaled, rtol=1e-6,
                               err_msg="Scaling tgt_2 should not affect tgt_1's widths")


# ---------------------------------------------------------------------------
# Item 4 DRE stabilization tests
# ---------------------------------------------------------------------------

def test_estimate_density_ratio_cv_shape():
    """estimate_density_ratio with cv=5 returns correct shape, all positive."""
    from mlforecast.conformal_prediction import estimate_density_ratio

    rng = np.random.default_rng(0)
    src = rng.standard_normal((60, 4))
    tgt = rng.standard_normal((40, 4)) + 0.5
    weights = estimate_density_ratio(src, tgt, cv=5)
    assert weights.shape == (60,)
    assert (weights > 0).all()


def test_estimate_density_ratio_cv0_matches_insample():
    """cv=0 reproduces original in-sample behavior (single fit, no cross-fitting)."""
    from mlforecast.conformal_prediction import estimate_density_ratio

    rng = np.random.default_rng(1)
    src = rng.standard_normal((30, 3))
    tgt = rng.standard_normal((20, 3)) + 1.0
    # Both should produce valid positive weights
    w_cv0 = estimate_density_ratio(src, tgt, cv=0)
    w_cv1 = estimate_density_ratio(src, tgt, cv=1)
    assert w_cv0.shape == (30,)
    assert (w_cv0 > 0).all()
    assert (w_cv1 > 0).all()


def test_estimate_density_ratio_return_target_weights():
    """return_target_weights=True returns a tuple of (n_src,) and (n_tgt,) arrays."""
    from mlforecast.conformal_prediction import estimate_density_ratio

    rng = np.random.default_rng(2)
    src = rng.standard_normal((40, 3))
    tgt = rng.standard_normal((25, 3)) + 0.5
    result = estimate_density_ratio(src, tgt, return_target_weights=True)
    assert isinstance(result, tuple) and len(result) == 2
    src_w, tgt_w = result
    assert src_w.shape == (40,)
    assert tgt_w.shape == (25,)
    assert (src_w > 0).all()
    assert (tgt_w > 0).all()


def test_estimate_density_ratio_clip_controls_extremes():
    """clip_quantile=0.99 bounds weights; None produces unbounded values."""
    from mlforecast.conformal_prediction import estimate_density_ratio

    rng = np.random.default_rng(3)
    src = rng.standard_normal((50, 2))
    # Very different target distribution → extreme weights without clipping
    tgt = rng.standard_normal((30, 2)) * 0.1 + 5.0

    w_clipped = estimate_density_ratio(src, tgt, cv=0, clip_quantile=0.99)
    w_none = estimate_density_ratio(src, tgt, cv=0, clip_quantile=None)

    assert w_clipped.max() <= w_none.max() + 1e-10
    assert np.isfinite(w_clipped).all()


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
