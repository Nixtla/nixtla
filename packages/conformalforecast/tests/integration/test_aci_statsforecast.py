"""Integration test: ACI over StatsForecastAdapter via the CV calibration path.

Uses a Naive model rather than AutoARIMA to keep the test runtime modest:
ACI's online holdout loop refits the adapter at each step (so the prediction
anchor advances with new observations), and 100 AutoARIMA refits on a 250-step
panel would dominate the suite. Naive is functionally appropriate here because
the synthetic series are random walks, for which Naive is the optimal point
forecaster — residuals are pure noise increments and ACI's coverage tracking
is the property under test, not the forecaster's accuracy.
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
from conformalforecast.methods.aci import AdaptiveConformalInference  # noqa: E402


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


class TestACIStatsForecast:
    """End-to-end ACI on a real Naive model via the CV calibration path."""

    def test_cv_calibration_then_online_cycle(self) -> None:
        n_series = 3
        train_len = 200
        n_cal = 100
        n_holdout = 100
        horizon = 1
        alpha = 0.1
        gamma = 0.05
        freq = "D"
        # Total panel must cover both calibration backtest and holdout walk.
        t_len = train_len + n_holdout + horizon

        df = _make_data(n_series=n_series, t_len=t_len, freq=freq, seed=0)

        # Fit the underlying model on the first ``train_len`` points per series.
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

        method = AdaptiveConformalInference(alpha=alpha, gamma=gamma)
        fitted = method.calibrate_cv(adapter, n_windows=n_cal, step_size=1, refit=False)
        cal = fitted.calibration_

        assert cal.diagnostics["path"] == "cross_validation"
        assert (cal.n_calibration_samples == n_cal).all()
        # Calibration loop should drift alpha_t away from the initial value.
        initial_alpha = np.full_like(fitted.alpha_t_, alpha)
        assert not np.allclose(fitted.alpha_t_, initial_alpha), (
            "alpha_t_ should drift from initial alpha after the calibration loop."
        )

        # Online holdout cycle. Refit at each step so the adapter's anchor
        # advances with new observations.
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
