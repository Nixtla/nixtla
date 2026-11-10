"""Integration test: SplitConformal over StatsForecastAdapter via the CV path.

Split CP is the package's reference method; this is its end-to-end check
against a real StatsForecast model. ``Naive`` on random walks keeps the runtime
modest and is the optimal point forecaster for the synthetic data, so residuals
are pure noise increments and coverage is the property under test.
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
from conformalforecast.methods.split import SplitConformal  # noqa: E402


def _make_data(n_series: int, t_len: int, freq: str = "D", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=t_len, freq=freq)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": f"series_{i}",
                    "ds": dates,
                    "y": rng.standard_normal(t_len).cumsum(),
                }
            )
            for i in range(n_series)
        ],
        ignore_index=True,
    )


class TestSplitConformalStatsForecast:
    def _fit(self, df: pd.DataFrame, horizon: int, freq: str = "D") -> StatsForecastAdapter:
        sf = StatsForecast(models=[Naive()], freq=freq)
        sf.fit(df)
        return StatsForecastAdapter(
            sf=sf, train_df=df, horizon=horizon, freq=freq, model_name="Naive"
        )

    def test_cv_calibration_shapes_and_diagnostics(self) -> None:
        n_series, horizon, n_windows = 3, 4, 40
        df = _make_data(n_series=n_series, t_len=160)
        adapter = self._fit(df, horizon=horizon)

        method = SplitConformal(alpha=0.1)
        fitted = method.calibrate_cv(adapter, n_windows=n_windows, step_size=1)
        cal = fitted.calibration_

        assert cal.diagnostics["path"] == "cross_validation"
        assert cal.n_calibration_samples.shape == (n_series,)
        assert (cal.n_calibration_samples == n_windows).all()
        q = np.asarray(cal.diagnostics["score_quantile"])
        assert q.shape == (n_series, horizon)
        assert (q >= 0).all()

        result = fitted.predict(adapter._df_to_panel(nw.from_native(df), "y"))
        assert result.point.shape == (n_series, 1, horizon)
        assert result.interval.shape == (n_series, 1, horizon, 2)
        assert (result.interval[..., 0] <= result.point).all()
        assert (result.point <= result.interval[..., 1]).all()

    def test_holdout_coverage_near_nominal(self) -> None:
        """Calibrate on the train segment, then check coverage on unseen steps.

        Tolerance: empirical coverage in [0.75, 1.0] for a 0.9 target over
        3 series x 30 one-step holdout windows (90 checks). The band is wide
        because 90 Bernoulli(0.9) draws have a ~0.03 standard error and split CP
        offers only marginal, not conditional, coverage on a random walk.
        """
        n_series, horizon, n_windows, n_holdout = 3, 1, 60, 30
        train_len = 200
        df = _make_data(n_series=n_series, t_len=train_len + n_holdout + horizon, seed=1)
        df_train = df.groupby("unique_id").head(train_len).reset_index(drop=True)

        adapter = self._fit(df_train, horizon=horizon)
        method = SplitConformal(alpha=0.1)
        fitted = method.calibrate_cv(adapter, n_windows=n_windows, step_size=1)

        full_panel = adapter._df_to_panel(nw.from_native(df), "y")
        covered = 0
        total = 0
        for w in range(n_holdout):
            new_history = full_panel[:, : train_len + w]
            # Advance the adapter's forecast anchor with the new observations.
            adapter.refit(new_history)
            result = fitted.predict(new_history)
            truth = full_panel[:, train_len + w : train_len + w + horizon][:, np.newaxis, :]
            in_interval = (truth >= result.interval[..., 0]) & (truth <= result.interval[..., 1])
            covered += int(in_interval.sum())
            total += int(in_interval.size)

        coverage = covered / total
        assert 0.75 <= coverage <= 1.0, f"coverage {coverage:.3f} outside [0.75, 1.0]"

    def test_determinism(self) -> None:
        df = _make_data(n_series=2, t_len=120, seed=2)
        results = []
        for _ in range(2):
            adapter = self._fit(df, horizon=3)
            method = SplitConformal(alpha=0.1)
            fitted = method.calibrate_cv(adapter, n_windows=20, step_size=1)
            results.append(fitted.predict(adapter._df_to_panel(nw.from_native(df), "y")).interval)
        np.testing.assert_array_equal(results[0], results[1])
