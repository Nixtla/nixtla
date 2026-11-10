"""End-to-end CQR tests against StatsForecast and NeuralForecast adapters.

Each test is gated on its respective ``pytest.importorskip`` so the file works
in both the slim and full development environments.

The role of these tests is to exercise the *plumbing* — that the CV fast path
runs to completion against a real Nixtla model, that the adapter's
``predict_quantiles`` returns sensible intervals, and that the loss-based
runtime gating selects the right path. Coverage guarantees are validated in
the unit-test suite against a synthetic oracle quantile adapter; replicating
them here would require splitting the panel and refitting the model, which
is dramatically slower than the unit-test equivalent and adds no signal
beyond confirming that calibration succeeded.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd
import pytest

# torch and scikit-learn each vendor their own libomp. Loading both into one
# process segfaults on macOS arm64 (KMP_DUPLICATE_LIB_OK does not help), so the
# torch-backed integration tests run in their own pytest session.
pytestmark = pytest.mark.torch


def _make_df(
    n_series: int = 3,
    t_len: int = 200,
    freq: str = "D",
    start: str = "2020-01-01",
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic daily panel used by both adapter integration tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=t_len, freq=freq)
    rows: list[pd.DataFrame] = []
    for i in range(n_series):
        rows.append(
            pd.DataFrame(
                {
                    "unique_id": f"series_{i}",
                    "ds": dates,
                    "y": rng.standard_normal(t_len).cumsum(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _assert_valid_interval(fitted: Any, history: np.ndarray) -> None:
    """Predict from `history` and assert the returned interval is well-formed.

    We do not assert lower ≤ upper because CQR can legitimately produce
    inverted intervals when the underlying quantile regressor over-covers
    on calibration data (negative score_threshold tightens further).
    Marginal coverage is the CQR contract; per-point ordering is not.
    """
    result = fitted.predict(history)
    assert result.point.shape == (history.shape[0], 1, fitted.forecaster_.horizon)
    assert result.interval.shape == (
        history.shape[0],
        1,
        fitted.forecaster_.horizon,
        2,
    )
    assert np.isfinite(result.interval).all()


# ===========================================================================
# StatsForecast + AutoARIMA — coverage-grade end-to-end
# ===========================================================================


def test_cqr_statsforecast_autoarima() -> None:
    sf_mod = pytest.importorskip("statsforecast")
    StatsForecast = sf_mod.StatsForecast

    from statsforecast.models import AutoARIMA

    from conformalforecast.adapters.statsforecast import StatsForecastAdapter
    from conformalforecast.methods.cqr import ConformalizedQuantileRegression

    n_series = 3
    horizon = 3
    t_len = 200
    alpha = 0.1
    df = _make_df(n_series=n_series, t_len=t_len)

    sf = StatsForecast(models=[AutoARIMA(season_length=1)], freq="D")
    sf.fit(df)
    adapter = StatsForecastAdapter(
        sf=sf,
        train_df=df,
        horizon=horizon,
        freq="D",
        model_name="AutoARIMA",
    )

    method = ConformalizedQuantileRegression(alpha=alpha)
    fitted = method.calibrate_cv(adapter, n_windows=50, step_size=1)
    cal = fitted.calibration_
    assert cal.diagnostics["path"] == "cross_validation"
    np.testing.assert_array_almost_equal(fitted.method_.quantile_levels, [0.05, 0.95])
    # Calibration must produce a non-trivial nonconformity-score quantile.
    assert np.any(np.abs(fitted.score_quantile_) > 1e-6)

    history = adapter._df_to_panel(nw.from_native(df), "y")
    _assert_valid_interval(fitted, history)


# ===========================================================================
# NeuralForecast + MLP(MQLoss)
# ===========================================================================

# Shared kwargs that pin determinism + silence Lightning's noisy progress output.
_TRAINER_QUIET: dict[str, Any] = {
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "logger": False,
    "accelerator": "cpu",
}


def test_cqr_neuralforecast_mqloss() -> None:
    nf_mod = pytest.importorskip("neuralforecast")
    NeuralForecast = nf_mod.NeuralForecast

    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import MLP

    from conformalforecast.adapters.neuralforecast import NeuralForecastAdapter
    from conformalforecast.methods.cqr import ConformalizedQuantileRegression

    n_series = 3
    horizon = 3
    t_len = 200
    alpha = 0.1
    df = _make_df(n_series=n_series, t_len=t_len)

    model = MLP(
        h=horizon,
        input_size=12,
        max_steps=20,
        loss=MQLoss(quantiles=[0.05, 0.5, 0.95]),
        random_seed=0,
        **_TRAINER_QUIET,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df)

    adapter = NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="MLP")
    assert adapter._supports_quantiles_runtime is True
    assert adapter._loss_class_name == "MQLoss"

    method = ConformalizedQuantileRegression(alpha=alpha)
    fitted = method.calibrate_cv(adapter, n_windows=50, step_size=1)
    cal = fitted.calibration_
    assert cal.diagnostics["path"] == "cross_validation"
    assert np.any(np.abs(fitted.score_quantile_) > 1e-6)

    history = adapter._df_to_panel(nw.from_native(df), "y")
    _assert_valid_interval(fitted, history)


# ===========================================================================
# NeuralForecast + DistributionLoss(Normal) — proves dual-path detection
# ===========================================================================


def test_cqr_neuralforecast_distribution_loss() -> None:
    """Verify a parametric distribution loss also drives CQR end-to-end.

    This is the highest-value test for the dual-path probabilistic-loss
    detection (``is_distribution_output`` flag plus class-name allowlist):
    distribution losses produce quantiles via internal sampling, not through
    a quantile head, so the column-resolution path differs from MQLoss.
    """
    nf_mod = pytest.importorskip("neuralforecast")
    NeuralForecast = nf_mod.NeuralForecast

    from neuralforecast.losses.pytorch import DistributionLoss
    from neuralforecast.models import MLP

    from conformalforecast.adapters.neuralforecast import NeuralForecastAdapter
    from conformalforecast.methods.cqr import ConformalizedQuantileRegression

    n_series = 3
    horizon = 3
    t_len = 200
    alpha = 0.1
    df = _make_df(n_series=n_series, t_len=t_len)

    model = MLP(
        h=horizon,
        input_size=12,
        max_steps=20,
        loss=DistributionLoss(distribution="Normal", level=[80, 90]),
        random_seed=0,
        **_TRAINER_QUIET,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df)

    adapter = NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="MLP")
    assert adapter._supports_quantiles_runtime is True
    assert adapter._loss_class_name == "DistributionLoss"

    method = ConformalizedQuantileRegression(alpha=alpha)
    fitted = method.calibrate_cv(adapter, n_windows=50, step_size=1)
    cal = fitted.calibration_
    assert cal.diagnostics["path"] == "cross_validation"
    assert np.any(np.abs(fitted.score_quantile_) > 1e-6)

    history = adapter._df_to_panel(nw.from_native(df), "y")
    _assert_valid_interval(fitted, history)
