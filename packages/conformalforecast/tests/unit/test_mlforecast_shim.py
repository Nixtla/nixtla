"""Shim unit tests: mlforecast's hook signature, driven with hand-built frames.

These do not need mlforecast — only ``utilsforecast`` (its DataFrame-agnostic
helper layer), which the shim soft-imports. The registration tests at the bottom
do need mlforecast and skip separately.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("utilsforecast")
pd = pytest.importorskip("pandas")

from conformalforecast.base import InsufficientCalibrationSamples  # noqa: E402
from conformalforecast.compat.mlforecast_shim import make_interval_method  # noqa: E402
from conformalforecast.methods import (  # noqa: E402
    AdaptiveConformalInference,
    ConformalizedQuantileRegression,
    SequentialPredictiveConformalInference,
    SplitConformal,
)
from conformalforecast.nonconformity import AbsoluteResidual, SignedResidual  # noqa: E402

# Methods require ceil(1/alpha) windows per series: 11 for level=90.
N_WINDOWS = 12

HOOK_KWARGS = {
    "model_names": ["model"],
    "level": [90],
    "cs_n_windows": N_WINDOWS,
    "cs_h": 2,
    "n_series": 2,
    "horizon": 2,
}


def _frames(
    n_series: int = 2, horizon: int = 2, n_windows: int = N_WINDOWS
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build an mlforecast-shaped (fcst_df, cs_df) pair with known scores."""
    rng = np.random.default_rng(0)
    ids = np.repeat([f"id{i}" for i in range(n_series)], horizon)
    fcst_df = pd.DataFrame({"unique_id": ids, "ds": np.tile(np.arange(horizon), n_series)})
    fcst_df["model"] = 100.0
    # cs_df is window-major: (n_windows, n_series, cs_h) flattened.
    scores = np.abs(rng.normal(size=(n_windows, n_series, horizon))) + 1.0
    cs_df = pd.DataFrame(
        {
            "unique_id": np.tile(
                np.repeat([f"id{i}" for i in range(n_series)], horizon), n_windows
            ),
            "cutoff": np.repeat(np.arange(n_windows), n_series * horizon),
            "ds": np.tile(np.arange(horizon), n_windows * n_series),
            "model": scores.ravel(),
        }
    )
    return fcst_df, cs_df, scores


class TestFactoryValidation:
    def test_rejects_method_whose_default_score_is_not_absolute(self) -> None:
        with pytest.raises(ValueError, match="AbsoluteResidual"):
            make_interval_method(SequentialPredictiveConformalInference)

    def test_rejects_method_consuming_quantiles(self) -> None:
        with pytest.raises(ValueError, match="consumes quantile forecasts"):
            make_interval_method(ConformalizedQuantileRegression)

    def test_rejects_a_configured_score_that_is_not_absolute(self) -> None:
        with pytest.raises(ValueError, match="AbsoluteResidual"):
            make_interval_method(AdaptiveConformalInference, score=SignedResidual())

    def test_accepts_a_configured_absolute_score(self) -> None:
        fcst_df, cs_df, _ = _frames()
        default = make_interval_method(SplitConformal)(fcst_df, cs_df, **HOOK_KWARGS)
        configured = make_interval_method(SplitConformal, score=AbsoluteResidual())(
            fcst_df, cs_df, **HOOK_KWARGS
        )
        pd.testing.assert_frame_equal(configured, default)


class TestSplitConformalHook:
    def test_columns_and_shape(self) -> None:
        fcst_df, cs_df, _ = _frames()
        fn = make_interval_method(SplitConformal)
        out = fn(
            fcst_df,
            cs_df,
            model_names=["model"],
            level=[80, 90],
            cs_n_windows=N_WINDOWS,
            cs_h=2,
            n_series=2,
            horizon=2,
        )
        assert list(out.columns)[-4:] == [
            "model-lo-90",
            "model-lo-80",
            "model-hi-80",
            "model-hi-90",
        ]
        assert len(out) == len(fcst_df)

    def test_intervals_are_ordered_and_nested(self) -> None:
        fcst_df, cs_df, _ = _frames()
        fn = make_interval_method(SplitConformal)
        out = fn(
            fcst_df,
            cs_df,
            model_names=["model"],
            level=[80, 90],
            cs_n_windows=N_WINDOWS,
            cs_h=2,
            n_series=2,
            horizon=2,
        )
        assert (out["model-lo-90"] <= out["model-lo-80"]).all()
        assert (out["model-lo-80"] <= out["model-hi-80"]).all()
        assert (out["model-hi-80"] <= out["model-hi-90"]).all()

    def test_matches_hand_computed_order_statistic(self) -> None:
        """SplitConformal width == the exact ceil((1-a)(n+1)) score, per (series, step)."""
        fcst_df, cs_df, scores = _frames(n_series=2, horizon=2, n_windows=N_WINDOWS)
        fn = make_interval_method(SplitConformal)
        out = fn(
            fcst_df,
            cs_df,
            model_names=["model"],
            level=[80],
            cs_n_windows=N_WINDOWS,
            cs_h=2,
            n_series=2,
            horizon=2,
        )
        # alpha = 0.2, n = 12 -> k = ceil(0.8 * 13) = 11 -> the 11th smallest score.
        k = 11
        ordered = np.sort(scores, axis=0)  # (n_windows, n_series, horizon)
        expected = ordered[k - 1].ravel()  # (n_series, horizon) -> series-major flat
        np.testing.assert_allclose(out["model-hi-80"].to_numpy() - 100.0, expected, rtol=1e-12)

    def test_constant_width_when_cs_h_is_one(self) -> None:
        rng = np.random.default_rng(1)
        n_series, horizon, n_windows = 2, 3, N_WINDOWS
        ids = np.repeat([f"id{i}" for i in range(n_series)], horizon)
        fcst_df = pd.DataFrame({"unique_id": ids, "ds": np.tile(np.arange(horizon), n_series)})
        fcst_df["model"] = 10.0
        scores = np.abs(rng.normal(size=(n_windows, n_series, 1))) + 1.0
        cs_df = pd.DataFrame({"model": scores.ravel()})
        fn = make_interval_method(SplitConformal)
        out = fn(
            fcst_df,
            cs_df,
            model_names=["model"],
            level=[90],
            cs_n_windows=n_windows,
            cs_h=1,
            n_series=n_series,
            horizon=horizon,
        )
        widths = (out["model-hi-90"] - out["model-lo-90"]).to_numpy().reshape(n_series, horizon)
        # Width is constant across the horizon for each series (mlforecast semantics).
        np.testing.assert_allclose(widths, np.tile(widths[:, :1], (1, horizon)), rtol=1e-12)


class TestCalibrationWindowRequirement:
    """Methods need ceil(1/alpha) windows per series; mlforecast's defaults do not."""

    def test_too_few_windows_raises_naming_the_series(self) -> None:
        # mlforecast's PredictionIntervals default (and minimum) is n_windows=2,
        # which cannot serve level=90 (alpha=0.1 needs 11 windows).
        fcst_df, cs_df, _ = _frames(n_series=2, horizon=2, n_windows=2)
        fn = make_interval_method(SplitConformal)
        with pytest.raises(InsufficientCalibrationSamples) as excinfo:
            fn(
                fcst_df,
                cs_df,
                model_names=["model"],
                level=[90],
                cs_n_windows=2,
                cs_h=2,
                n_series=2,
                horizon=2,
            )
        assert excinfo.value.series_indices == [0, 1]

    def test_alpha_is_exact_for_integer_levels(self) -> None:
        """level=90 must give alpha=0.1 exactly, so ceil(1/alpha) is 10, not 11.

        ``1 - 90/100`` is 0.09999999999999998 in binary floating point, which
        rounds ``ceil(1/alpha)`` up to 11 and rejects a perfectly adequate
        10-window calibration set. ``(100 - 90)/100`` is exact.
        """
        fcst_df, cs_df, _ = _frames(n_series=1, horizon=1, n_windows=10)
        fn = make_interval_method(SplitConformal)
        out = fn(
            fcst_df,
            cs_df,
            model_names=["model"],
            level=[90],
            cs_n_windows=10,
            cs_h=1,
            n_series=1,
            horizon=1,
        )
        assert np.isfinite(out["model-hi-90"].to_numpy()).all()


class TestOnlineMethodHook:
    def test_aci_method_runs_and_produces_finite_intervals(self) -> None:
        fcst_df, cs_df, _ = _frames(n_series=2, horizon=2, n_windows=N_WINDOWS)
        fn = make_interval_method(AdaptiveConformalInference, gamma=0.01)
        out = fn(
            fcst_df,
            cs_df,
            model_names=["model"],
            level=[90],
            cs_n_windows=N_WINDOWS,
            cs_h=2,
            n_series=2,
            horizon=2,
        )
        assert np.isfinite(out["model-lo-90"].to_numpy()).all()
        assert (out["model-lo-90"] <= out["model-hi-90"]).all()

    def test_deterministic(self) -> None:
        fcst_df, cs_df, _ = _frames()
        a = make_interval_method(SplitConformal)(fcst_df, cs_df, **HOOK_KWARGS)
        b = make_interval_method(SplitConformal)(fcst_df, cs_df, **HOOK_KWARGS)
        pd.testing.assert_frame_equal(a, b)


class TestUnsupportedMlforecast11Arguments:
    """mlforecast 1.1 added weighted and transfer conformal; the shim rejects both."""

    def test_weights_rejected(self) -> None:
        fcst_df, cs_df, _ = _frames()
        fn = make_interval_method(SplitConformal)
        with pytest.raises(NotImplementedError, match="weights"):
            fn(fcst_df, cs_df, weights=np.ones(10), **HOOK_KWARGS)

    def test_target_weights_rejected(self) -> None:
        fcst_df, cs_df, _ = _frames()
        fn = make_interval_method(SplitConformal)
        with pytest.raises(NotImplementedError, match="target_weights"):
            fn(fcst_df, cs_df, target_weights=np.ones(4), **HOOK_KWARGS)

    def test_is_transfer_rejected(self) -> None:
        fcst_df, cs_df, _ = _frames()
        fn = make_interval_method(SplitConformal)
        with pytest.raises(NotImplementedError, match="is_transfer"):
            fn(fcst_df, cs_df, is_transfer=True, **HOOK_KWARGS)

    def test_explicit_none_and_false_are_fine(self) -> None:
        """The values mlforecast passes on the ordinary path must not trip the guards."""
        fcst_df, cs_df, _ = _frames()
        fn = make_interval_method(SplitConformal)
        out = fn(fcst_df, cs_df, weights=None, is_transfer=False, **HOOK_KWARGS)
        assert "model-hi-90" in out.columns


class TestRegistration:
    def test_register_inserts_into_mlforecast_registry(self) -> None:
        pytest.importorskip("mlforecast")
        from mlforecast.conformal_prediction import _INTERVAL_METHODS, get_conformal_method

        from conformalforecast.compat.mlforecast_shim import register_interval_method

        name = "conformal_method_split_test"
        try:
            fn = register_interval_method(name, SplitConformal)
            assert _INTERVAL_METHODS[name] is fn
            assert get_conformal_method(name) is fn
        finally:
            _INTERVAL_METHODS.pop(name, None)

    def test_register_refuses_to_clobber_without_overwrite(self) -> None:
        pytest.importorskip("mlforecast")
        from conformalforecast.compat.mlforecast_shim import register_interval_method

        with pytest.raises(ValueError, match="already registered"):
            register_interval_method("conformal_error", SplitConformal)
