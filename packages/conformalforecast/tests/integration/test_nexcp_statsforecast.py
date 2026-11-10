"""Integration test: NexCP over StatsForecastAdapter via the CV calibration path.

Uses a Naive model for the same reason as :mod:`test_aci_statsforecast`: the
synthetic series are random walks for which Naive is the optimal point
forecaster. The property under test is NexCP's coverage tracking, not the
forecaster's accuracy.
"""

from __future__ import annotations

import narwhals as nw
import numpy as np
import pandas as pd
import pytest

sf_mod = pytest.importorskip("statsforecast")
StatsForecast = sf_mod.StatsForecast

from statsforecast.models import Naive  # noqa: E402

from conformalforecast.adapters.statsforecast import StatsForecastAdapter  # noqa: E402
from conformalforecast.methods.nexcp import NonexchangeableConformalPrediction  # noqa: E402


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


class TestNexCPStatsForecast:
    """End-to-end NexCP on a StatsForecast Naive model via the CV path."""

    def test_cv_calibration_then_online_cycle(self) -> None:
        n_series = 3
        train_len = 200
        n_cal = 100
        n_holdout = 100
        horizon = 1
        alpha = 0.1
        rho = 0.99
        freq = "D"
        t_len = train_len + n_holdout + horizon

        df = _make_data(n_series=n_series, t_len=t_len, freq=freq, seed=0)
        df_train = df.groupby("unique_id").head(train_len).reset_index(drop=True)

        sf = StatsForecast(models=[Naive()], freq=freq)
        sf.fit(df_train)

        adapter = StatsForecastAdapter(
            sf=sf,
            train_df=df_train,
            horizon=horizon,
            freq=freq,
            model_name="Naive",
        )

        method = NonexchangeableConformalPrediction(alpha=alpha, rho=rho)
        fitted = method.calibrate_cv(adapter, n_windows=n_cal, step_size=1, refit=False)
        cal = fitted.calibration_

        assert cal.diagnostics["path"] == "cross_validation"
        assert cal.diagnostics["rho"] == rho
        assert (cal.n_calibration_samples == n_cal).all()

        # ESS for rho=0.99, n=100: closed-form gives ~63.4 — close to but
        # below n_windows, as expected.
        from conformalforecast.diagnostics.method_diagnostics import nexcp_state

        ess = nexcp_state(fitted)["effective_sample_size"]
        assert 0 < ess < n_cal, f"ESS {ess:.2f} should be in (0, {n_cal})."
        assert ess > n_cal / 2, f"ESS {ess:.2f} should be > n_cal / 2 for rho=0.99."

        # Holdout walk: refit the adapter so the prediction anchor advances.
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
