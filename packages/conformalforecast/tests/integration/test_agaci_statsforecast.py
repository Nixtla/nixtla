"""Integration test: AgACI over StatsForecastAdapter via the CV calibration path.

Uses AutoARIMA on three short synthetic series. AgACI runs ``K`` parallel ACI
experts; the number of refits in the holdout cycle equals ``n_holdout``, so
keeping ``n_holdout`` modest matters more than for the ACI integration test.
"""

from __future__ import annotations

import narwhals as nw
import numpy as np
import pandas as pd
import pytest

sf_mod = pytest.importorskip("statsforecast")
StatsForecast = sf_mod.StatsForecast

from statsforecast.models import AutoARIMA  # noqa: E402

from conformalforecast.adapters.statsforecast import StatsForecastAdapter  # noqa: E402
from conformalforecast.methods.agaci import AggregatedAdaptiveConformalInference  # noqa: E402


def _make_data(n_series: int, t_len: int, freq: str = "D", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=t_len, freq=freq)
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


class TestAgACIStatsForecast:
    """End-to-end AgACI on a real AutoARIMA model via the CV calibration path."""

    def test_cv_calibration_then_online_cycle(self) -> None:
        n_series = 3
        train_len = 200
        n_cal = 100
        n_holdout = 100
        horizon = 1
        alpha = 0.1
        freq = "D"
        t_len = train_len + n_holdout + horizon

        df = _make_data(n_series=n_series, t_len=t_len, freq=freq, seed=0)
        df_train = df.groupby("unique_id").head(train_len).reset_index(drop=True)
        sf = StatsForecast(models=[AutoARIMA(season_length=1)], freq=freq)
        sf.fit(df_train)

        adapter = StatsForecastAdapter(
            sf=sf,
            train_df=df_train,
            horizon=horizon,
            freq=freq,
            model_name="AutoARIMA",
        )

        method = AggregatedAdaptiveConformalInference(alpha=alpha)
        fitted = method.calibrate_cv(adapter, n_windows=n_cal, step_size=1, refit=False)
        cal = fitted.calibration_

        assert cal.diagnostics["path"] == "cross_validation"
        assert (cal.n_calibration_samples == n_cal).all()

        full_panel = adapter._df_to_panel(nw.from_native(df), "y")
        covered = 0
        total = 0
        for w in range(n_holdout):
            new_history = full_panel[:, : train_len + w]
            adapter.refit(new_history)

            truth = full_panel[:, train_len + w : train_len + w + horizon]

            result = fitted.predict(new_history)
            assert result.point.shape == (n_series, 1, horizon)
            assert result.interval.shape == (n_series, 1, horizon, 2)

            truth_3d = truth[:, np.newaxis, :]
            in_interval = (truth_3d >= result.interval[..., 0]) & (
                truth_3d <= result.interval[..., 1]
            )
            covered += int(in_interval.sum())
            total += int(in_interval.size)

            fitted.update(result.point, truth_3d)

        coverage = covered / total
        assert abs(coverage - (1 - alpha)) < 0.06, (
            f"Empirical coverage {coverage:.4f} deviates from target "
            f"{1 - alpha:.2f} by more than 0.06."
        )

        # Aggregator weights should not remain uniform after a full holdout cycle.
        K = method.n_experts
        n_series_, horizon_ = adapter.n_series, adapter.horizon
        uniform = np.full((K, n_series_, horizon_), 1.0 / K)
        w_lower = fitted.aggregator_lower_.weights()
        w_upper = fitted.aggregator_upper_.weights()
        assert not np.allclose(w_lower, uniform, atol=1e-3), (
            "aggregator_lower_ weights should drift from uniform after the holdout cycle."
        )
        assert not np.allclose(w_upper, uniform, atol=1e-3), (
            "aggregator_upper_ weights should drift from uniform after the holdout cycle."
        )
