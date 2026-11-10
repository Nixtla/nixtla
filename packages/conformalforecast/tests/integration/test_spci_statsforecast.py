"""Integration test: SPCI over StatsForecastAdapter via the CV calibration path.

Uses Naive (rather than AutoARIMA) because the synthetic series are random
walks, for which Naive is the optimal point forecaster. The property under
test is SPCI's interval calibration and asymmetric-offset behaviour, not the
forecaster's accuracy. Naive also keeps the test runtime modest: the online
holdout loop refits the adapter at each step, and AutoARIMA refits dominate
the suite.
"""

from __future__ import annotations

import narwhals as nw
import numpy as np
import pandas as pd
import pytest

sf_mod = pytest.importorskip("statsforecast")
pytest.importorskip("quantile_forest")
StatsForecast = sf_mod.StatsForecast

from statsforecast.models import Naive  # noqa: E402

from conformalforecast.adapters.statsforecast import StatsForecastAdapter  # noqa: E402
from conformalforecast.methods.spci import (  # noqa: E402
    SequentialPredictiveConformalInference,
)


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


class TestSPCIStatsForecast:
    """End-to-end SPCI on a real Naive model via the CV calibration path."""

    def test_cv_calibration_then_online_cycle(self) -> None:
        n_series = 3
        train_len = 400
        n_cal = 300
        n_holdout = 30
        horizon = 1
        alpha = 0.1
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

        method = SequentialPredictiveConformalInference(
            alpha=alpha, window_size=50, beta_grid_size=5, refit_every=5
        )
        fitted = method.calibrate_cv(adapter, n_windows=n_cal, step_size=1, refit=False)
        cal = fitted.calibration_

        assert cal.diagnostics["path"] == "cross_validation"
        from conformalforecast.diagnostics.method_diagnostics import spci_state

        assert spci_state(fitted)["regressor_class"] == "QRFQuantileRegressor"
        assert (cal.n_calibration_samples == n_cal).all()

        full_panel = adapter._df_to_panel(nw.from_native(df), "y")
        covered = 0
        total = 0
        intervals: list[np.ndarray] = []
        points: list[np.ndarray] = []
        for w in range(n_holdout):
            new_history = full_panel[:, : train_len + w]
            adapter.refit(new_history)

            truth = full_panel[:, train_len + w : train_len + w + horizon]
            result = fitted.predict(new_history)
            assert result.point.shape == (n_series, 1, horizon)
            assert result.interval.shape == (n_series, 1, horizon, 2)
            intervals.append(result.interval.copy())
            points.append(result.point.copy())

            truth_3d = truth[:, np.newaxis, :]
            in_interval = (truth_3d >= result.interval[..., 0]) & (
                truth_3d <= result.interval[..., 1]
            )
            covered += int(in_interval.sum())
            total += int(in_interval.size)

            fitted.update(result.point, truth_3d)

        coverage = covered / total
        # Looser than the synthetic tolerance: real-ish data + QRF variance
        # + finite calibration.
        assert abs(coverage - (1 - alpha)) < 0.08, (
            f"Empirical coverage {coverage:.4f} deviates from target "
            f"{1 - alpha:.2f} by more than 0.08."
        )

        # At least one (series, horizon, sample) should exhibit a visibly
        # asymmetric offset around the point forecast — confirming the
        # interval shape isn't forced symmetric.
        asym_found = False
        for interval, point in zip(intervals, points, strict=True):
            lower_offset = interval[..., 0] - point  # (n_series, 1, horizon)
            upper_offset = interval[..., 1] - point
            if not np.allclose(-lower_offset, upper_offset, atol=1e-3):
                asym_found = True
                break
        assert asym_found, "Expected at least one cell to show asymmetric offsets."
