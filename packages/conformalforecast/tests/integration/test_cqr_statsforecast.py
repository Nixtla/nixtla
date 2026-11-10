"""Integration test: CQR over a real StatsForecast quantile model.

Notes
-----
``StatsForecastAdapter`` does not implement :class:`SupportsQuantiles` in v0.1
(its docstring scopes quantile output as out-of-scope). To exercise CQR
end-to-end against a real forecasting library, this test defines a small
test-local adapter that wraps a fitted ``StatsForecast`` and exposes
``predict_quantiles`` via the ``level=`` parameter on ``forecast()``. The
production adapter is intentionally untouched.

Only symmetric quantile pairs ``(alpha/2, 1 - alpha/2)`` are supported by
this test adapter, which is all CQR's default ``symmetric=True`` requires.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from pydantic import PrivateAttr, model_validator

sf_mod = pytest.importorskip("statsforecast")
StatsForecast = sf_mod.StatsForecast

from statsforecast.models import AutoARIMA  # noqa: E402

from conformalforecast.base import Forecast, ForecasterAdapter, Series  # noqa: E402
from conformalforecast.capabilities import SupportsQuantiles  # noqa: E402
from conformalforecast.methods.cqr import ConformalizedQuantileRegression  # noqa: E402

# ---------------------------------------------------------------------------
# Test-local quantile adapter
# ---------------------------------------------------------------------------


class _StatsForecastQuantileAdapter(ForecasterAdapter, SupportsQuantiles):
    """Wrap a fitted StatsForecast model to produce symmetric quantile pairs.

    Only supports calls with ``quantiles == (q, 1 - q)`` for some ``q < 0.5``.
    """

    sf: Any
    train_df: pd.DataFrame
    freq: str
    model_name: str

    _series_ids: tuple[str, ...] = PrivateAttr(default=())

    @model_validator(mode="before")
    @classmethod
    def _derive_n_series(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["n_series"] = data["train_df"]["unique_id"].nunique()
        return data

    @model_validator(mode="after")
    def _build(self) -> _StatsForecastQuantileAdapter:
        self._series_ids = tuple(sorted(self.train_df["unique_id"].unique()))
        return self

    def predict(self, history: Series) -> Forecast:
        # Not exercised by CQR; CQR only calls predict_quantiles. We still
        # implement it to satisfy the abstract contract.
        df = self._panel_to_df(history)
        out = self.sf.forecast(h=self.horizon, df=df)
        panel = self._df_to_panel(out, self.model_name)
        return panel[:, np.newaxis, :]

    def predict_quantiles(
        self,
        history: Series,
        quantiles: NDArray[np.floating],
    ) -> Forecast:
        history = self._validate_history(history)
        q_arr = np.asarray(quantiles, dtype=np.float64)
        if q_arr.shape != (2,):
            raise ValueError(f"This adapter only supports a quantile pair; got {q_arr.shape}")
        q_lo, q_hi = float(q_arr[0]), float(q_arr[1])
        if not np.isclose(q_lo + q_hi, 1.0):
            raise ValueError(
                "This adapter only supports symmetric quantile pairs "
                f"(q_lo + q_hi == 1); got ({q_lo}, {q_hi})."
            )
        # Translate (q_lo, q_hi) -> confidence level percentage in (0, 100).
        level_pct = (q_hi - q_lo) * 100.0

        df = self._panel_to_df(history)
        out = self.sf.forecast(h=self.horizon, df=df, level=[level_pct])

        # statsforecast formats the column suffix using the float repr of the
        # level it received, so look the columns up by prefix instead.
        lo_prefix = f"{self.model_name}-lo-"
        hi_prefix = f"{self.model_name}-hi-"
        lo_cols = [c for c in out.columns if c.startswith(lo_prefix)]
        hi_cols = [c for c in out.columns if c.startswith(hi_prefix)]
        if len(lo_cols) != 1 or len(hi_cols) != 1:
            raise RuntimeError(
                f"Expected exactly one lo/hi column pair in StatsForecast "
                f"output; got lo={lo_cols}, hi={hi_cols}"
            )
        lo_col, hi_col = lo_cols[0], hi_cols[0]

        lo_panel = self._df_to_panel(out, lo_col)  # (n_series, horizon)
        hi_panel = self._df_to_panel(out, hi_col)
        # Stack on a new "quantile" axis -> (n_series, 2, horizon)
        return np.stack([lo_panel, hi_panel], axis=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _panel_to_df(self, panel: NDArray[np.floating]) -> pd.DataFrame:
        # Reconstruct timestamps anchored at the train df's per-series end.
        rows: list[pd.DataFrame] = []
        for i, sid in enumerate(self._series_ids):
            end = self.train_df.loc[self.train_df["unique_id"] == sid, "ds"].max()
            T = panel.shape[1]
            timestamps = pd.date_range(end=end, periods=T, freq=self.freq)
            rows.append(pd.DataFrame({"unique_id": sid, "ds": timestamps, "y": panel[i]}))
        return pd.concat(rows, ignore_index=True)

    def _df_to_panel(self, df: pd.DataFrame, value_col: str) -> NDArray[np.floating]:
        pivot = df.pivot(index="unique_id", columns="ds", values=value_col)
        pivot = pivot.loc[list(self._series_ids)].sort_index(axis=1)
        return pivot.to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCQRStatsForecast:
    """End-to-end CQR on a real ARIMA model produces calibrated intervals."""

    def test_calibrate_and_predict(self) -> None:
        n_series = 3
        t_len = 200
        horizon = 3
        alpha = 0.1
        freq = "D"

        df = _make_data(n_series=n_series, t_len=t_len, freq=freq, seed=0)
        sf = StatsForecast(models=[AutoARIMA(season_length=1)], freq=freq)
        sf.fit(df)

        adapter = _StatsForecastQuantileAdapter(
            sf=sf,
            train_df=df,
            horizon=horizon,
            freq=freq,
            model_name="AutoARIMA",
        )

        method = ConformalizedQuantileRegression(alpha=alpha)

        # Build a calibration set with rolling backtest windows. We split the
        # train_df into a "calibration prefix" + "evaluation truth" pair per
        # window (mirroring how a user would do this without CV).
        n_cal = 20
        eval_split = t_len - horizon - n_cal  # earliest cutoff
        cal_histories: list[NDArray[np.floating]] = []
        truths_list: list[NDArray[np.floating]] = []
        full_panel = adapter._df_to_panel(df, "y")
        for w in range(n_cal):
            cutoff = eval_split + w
            cal_histories.append(full_panel[:, :cutoff])
            truths_list.append(full_panel[:, cutoff : cutoff + horizon])
        cal_truths = np.stack(truths_list, axis=1)  # (n_series, n_cal, horizon)

        fitted = method.calibrate(adapter, cal_histories, cal_truths)
        cal_result = fitted.calibration_

        assert cal_result.diagnostics["path"] == "loop"
        assert (cal_result.n_calibration_samples == n_cal).all()
        # Calibration is doing real (non-trivial) work.
        assert np.any(np.abs(fitted.score_quantile_) > 1e-6)

        # Holdout evaluation.
        holdout_n = 30
        holdout_split = t_len - horizon - holdout_n
        covered = 0
        total = 0
        for w in range(holdout_n):
            cutoff = holdout_split + w
            history = full_panel[:, :cutoff]
            truth = full_panel[:, cutoff : cutoff + horizon]

            result = fitted.predict(history)
            assert result.point.shape == (n_series, 1, horizon)
            assert result.interval.shape == (n_series, 1, horizon, 2)

            truth_3d = truth[:, np.newaxis, :]
            covered += (
                (truth_3d >= result.interval[..., 0]) & (truth_3d <= result.interval[..., 1])
            ).sum()
            total += truth_3d.size

        empirical_coverage = covered / total
        # Looser tolerance than unit tests: real ARIMA on a small synthetic
        # series, only 30 holdout windows.
        assert abs(empirical_coverage - (1 - alpha)) < 0.1, (
            f"Empirical coverage {empirical_coverage:.4f} deviates from "
            f"target {1 - alpha:.2f} by more than 0.1"
        )
