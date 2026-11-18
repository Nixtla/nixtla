"""Integration tests for MLForecastAdapter."""

from __future__ import annotations

import narwhals as nw
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

mlf_mod = pytest.importorskip("mlforecast")
MLForecast = mlf_mod.MLForecast

from mlforecast.lag_transforms import ExpandingMean  # noqa: E402
from mlforecast.target_transforms import Differences  # noqa: E402
from mlforecast.utils import PredictionIntervals  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from conformalforecast.adapters.mlforecast import MLForecastAdapter  # noqa: E402
from conformalforecast.capabilities import (  # noqa: E402
    SupportsBootstrap,
    SupportsCrossValidation,
    SupportsQuantiles,
    SupportsRefit,
)
from conformalforecast.methods.split import SplitConformal  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(
    n_series: int = 1,
    t_len: int = 100,
    freq: str = "D",
    start: str = "2020-01-01",
    seed: int = 0,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.DataFrame:
    """Create a synthetic long-format DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=t_len, freq=freq)
    rows: list[pd.DataFrame] = []
    for i in range(n_series):
        sid = f"series_{i}"
        rows.append(
            pd.DataFrame(
                {
                    id_col: sid,
                    time_col: dates,
                    target_col: rng.standard_normal(t_len).cumsum(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _make_adapter(
    n_series: int = 1,
    t_len: int = 100,
    horizon: int = 3,
    freq: str = "D",
    model_name: str = "Ridge",
    **mlf_kwargs: object,
) -> tuple[MLForecastAdapter, pd.DataFrame]:
    """Fit and return an adapter plus its training DataFrame."""
    df = _make_df(n_series=n_series, t_len=t_len, freq=freq)
    base_kwargs: dict[str, object] = {"lags": [1, 2, 3]}
    base_kwargs.update(mlf_kwargs)
    mlf = MLForecast(models={model_name: Ridge()}, freq=freq, **base_kwargs)
    mlf.fit(df)
    adapter = MLForecastAdapter(
        mlf=mlf, train_df=df, horizon=horizon, freq=freq, model_name=model_name
    )
    return adapter, df


# ===========================================================================
# Construction tests
# ===========================================================================


class TestConstruction:
    """Successful construction and validation failures."""

    def test_single_model(self) -> None:
        adapter, _ = _make_adapter(n_series=1, horizon=3)
        assert adapter.n_series == 1
        assert adapter.horizon == 3

    def test_multi_model_mlf(self) -> None:
        """Construct from a multi-model MLForecast, selecting one model."""
        df = _make_df(n_series=2, t_len=100)
        mlf = MLForecast(
            models={"Ridge": Ridge(), "Ridge2": Ridge(alpha=2.0)},
            freq="D",
            lags=[1, 2, 3],
        )
        mlf.fit(df)
        adapter = MLForecastAdapter(mlf=mlf, train_df=df, horizon=5, freq="D", model_name="Ridge")
        assert adapter.n_series == 2
        assert adapter.model_name == "Ridge"

    # --- validation failure tests ---

    def test_not_fitted_raises(self) -> None:
        df = _make_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        with pytest.raises(ValueError, match="not fitted"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="D", model_name="Ridge")

    def test_bad_model_name_raises(self) -> None:
        df = _make_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df)
        with pytest.raises(ValueError, match="model_name 'NoSuchModel'"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="D", model_name="NoSuchModel")

    def test_missing_columns_raises(self) -> None:
        df = _make_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df)
        bad_df = df.rename(columns={"y": "target"})
        with pytest.raises(ValueError, match="missing required columns"):
            MLForecastAdapter(mlf=mlf, train_df=bad_df, horizon=3, freq="D", model_name="Ridge")

    def test_invalid_freq_raises(self) -> None:
        df = _make_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df)
        with pytest.raises(ValueError, match="Invalid frequency"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="BOGUS", model_name="Ridge")

    def test_non_contiguous_raises(self) -> None:
        df = _make_df(t_len=100)
        df = df.drop(index=50).reset_index(drop=True)
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        # MLForecast itself complains about missing dates; silence by using validate_data=False
        mlf.fit(df, validate_data=False)
        with pytest.raises(ValueError, match="Non-contiguous"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="D", model_name="Ridge")

    def test_nan_target_raises(self) -> None:
        clean_df = _make_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(clean_df)
        bad_df = clean_df.copy()
        bad_df.loc[10, "y"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            MLForecastAdapter(mlf=mlf, train_df=bad_df, horizon=3, freq="D", model_name="Ridge")

    def test_non_overlapping_series_both_survive(self) -> None:
        """A3/B1: series with non-overlapping ranges no longer get intersected away.

        Each series individually has plenty of span for horizon=3, so both
        survive construction — unlike the removed common-range intersection
        that would have discarded data (or raised) here.
        """
        dates1 = pd.date_range("2020-01-01", periods=50, freq="D")
        dates2 = pd.date_range("2020-02-15", periods=50, freq="D")
        rng = np.random.default_rng(0)
        df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "unique_id": "a",
                        "ds": dates1,
                        "y": rng.standard_normal(50).cumsum(),
                    }
                ),
                pd.DataFrame(
                    {
                        "unique_id": "b",
                        "ds": dates2,
                        "y": rng.standard_normal(50).cumsum(),
                    }
                ),
            ],
            ignore_index=True,
        )
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df)
        adapter = MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="D", model_name="Ridge")
        assert set(adapter._series_ids) == {"a", "b"}

    def test_short_series_raises(self) -> None:
        """A series that individually spans fewer than 2*horizon steps raises,
        naming the offending series, even though the other series has plenty
        of data (no common-range truncation masks this)."""
        dates_a = pd.date_range("2020-01-01", periods=50, freq="D")
        dates_b = pd.date_range("2020-01-01", periods=10, freq="D")
        rng = np.random.default_rng(0)
        df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "unique_id": "a",
                        "ds": dates_a,
                        "y": rng.standard_normal(50).cumsum(),
                    }
                ),
                pd.DataFrame(
                    {
                        "unique_id": "b",
                        "ds": dates_b,
                        "y": rng.standard_normal(10).cumsum(),
                    }
                ),
            ],
            ignore_index=True,
        )
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df)
        with pytest.raises(ValueError, match=r"\['b'\] span fewer than"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=20, freq="D", model_name="Ridge")

    def test_passing_a_derived_field_raises(self) -> None:
        df = _make_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df)
        with pytest.raises(ValidationError, match="derived from train_df"):
            MLForecastAdapter(
                mlf=mlf,
                train_df=df,
                horizon=3,
                freq="D",
                model_name="Ridge",
                n_series=99,  # type: ignore[call-arg]
            )

    def test_not_mlforecast_instance_raises(self) -> None:
        df = _make_df()
        with pytest.raises(ValueError, match="MLForecast instance"):
            MLForecastAdapter(
                mlf="not_mlf",  # type: ignore[arg-type]
                train_df=df,
                horizon=3,
                freq="D",
                model_name="Ridge",
            )

    def test_prediction_intervals_rejected(self) -> None:
        """A fit with prediction_intervals must be rejected at construction."""
        df = _make_df(n_series=1, t_len=100)
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(
            df,
            prediction_intervals=PredictionIntervals(
                n_windows=2, h=3, method="conformal_distribution"
            ),
        )
        # Confirm the test setup exercises the rejection path.
        assert mlf._cs_df is not None
        with pytest.raises(ValueError, match="produces its own intervals"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="D", model_name="Ridge")

    def test_max_horizon_too_small_rejected(self) -> None:
        df = _make_df(n_series=1, t_len=100)
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df, max_horizon=5)
        with pytest.raises(ValueError, match="max_horizon=5"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=10, freq="D", model_name="Ridge")

    def test_horizons_missing_steps_rejected(self) -> None:
        df = _make_df(n_series=1, t_len=100)
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df, horizons=[1, 3, 5])
        with pytest.raises(ValueError, match="missing required steps"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=5, freq="D", model_name="Ridge")

    def test_static_features_rejected(self) -> None:
        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        rows = []
        for sid in ["a", "b"]:
            rows.append(
                pd.DataFrame(
                    {
                        "unique_id": sid,
                        "ds": dates,
                        "y": rng.standard_normal(100).cumsum(),
                        # constant feature so MLForecast accepts it as static
                        "feat_x": float(rng.standard_normal()),
                    }
                )
            )
        df = pd.concat(rows, ignore_index=True)
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df, static_features=["feat_x"])
        with pytest.raises(ValueError, match="Static features are not supported"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="D", model_name="Ridge")

    def test_lag_and_target_transforms_pass_through(self) -> None:
        """Differences + ExpandingMean should construct cleanly and predict."""
        df = _make_df(n_series=2, t_len=100)
        mlf = MLForecast(
            models={"Ridge": Ridge()},
            freq="D",
            lags=[1, 2, 3],
            target_transforms=[Differences([1])],
            lag_transforms={1: [ExpandingMean()]},
        )
        mlf.fit(df)
        adapter = MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="D", model_name="Ridge")
        history = adapter._df_to_panel(nw.from_native(df), "y")
        out = adapter.predict(history)
        assert out.shape == (2, 1, 3)
        assert not np.isnan(out).any()


# ===========================================================================
# predict tests
# ===========================================================================


class TestPredict:
    """predict() shape, determinism, and anchor semantics."""

    @pytest.mark.parametrize(
        "n_series,horizon",
        [(1, 1), (1, 12), (5, 1), (5, 12)],
    )
    def test_output_shape(self, n_series: int, horizon: int) -> None:
        adapter, df = _make_adapter(n_series=n_series, horizon=horizon, t_len=100)
        history = adapter._df_to_panel(nw.from_native(df), "y")
        result = adapter.predict(history)
        assert result.shape == (n_series, 1, horizon)

    def test_determinism(self) -> None:
        adapter, df = _make_adapter(n_series=2, horizon=5)
        history = adapter._df_to_panel(nw.from_native(df), "y")
        r1 = adapter.predict(history)
        r2 = adapter.predict(history)
        np.testing.assert_array_equal(r1, r2)

    def test_different_histories_produce_different_forecasts(self) -> None:
        adapter, df = _make_adapter(n_series=1, horizon=3, t_len=100)
        history_full = adapter._df_to_panel(nw.from_native(df), "y")
        history_short = history_full[:, :50]
        r1 = adapter.predict(history_full)
        r2 = adapter.predict(history_short)
        assert not np.array_equal(r1, r2)


# ===========================================================================
# refit tests
# ===========================================================================


class TestRefit:
    """refit() updates the model and anchoring."""

    def test_refit_with_longer_panel_shifts_end(self) -> None:
        adapter, df = _make_adapter(n_series=1, horizon=3, t_len=80)
        old_ends = adapter._series_ends.to_list()

        new_df = _make_df(n_series=1, t_len=100, seed=1)
        new_panel = adapter._df_to_panel(nw.from_native(new_df), "y")

        adapter.refit(new_panel)

        assert adapter._series_ends.to_list() == [e + pd.Timedelta(days=20) for e in old_ends]
        result = adapter.predict(new_panel)
        assert result.shape == (1, 1, 3)

    def test_refit_with_different_series_raises(self) -> None:
        adapter, _ = _make_adapter(n_series=2, horizon=3)
        bad_panel = np.random.default_rng(0).standard_normal((3, 100))
        with pytest.raises(ValueError, match="leading axis"):
            adapter.refit(bad_panel)


# ===========================================================================
# cross_validate tests
# ===========================================================================


class TestCrossValidate:
    """cross_validate() shapes and validation."""

    @pytest.mark.parametrize("n_series", [1, 3])
    def test_output_shape(self, n_series: int) -> None:
        horizon = 3
        n_windows = 4
        step_size = 2
        adapter, _ = _make_adapter(n_series=n_series, horizon=horizon, t_len=100)
        preds, truths = adapter.cross_validate(n_windows=n_windows, step_size=step_size)
        assert preds.shape == (n_series, n_windows, horizon)
        assert truths.shape == (n_series, n_windows, horizon)

    def test_refit_true_vs_false_same_shape(self) -> None:
        adapter, _ = _make_adapter(n_series=1, horizon=3, t_len=100)
        p1, t1 = adapter.cross_validate(n_windows=3, step_size=1, refit=False)
        p2, t2 = adapter.cross_validate(n_windows=3, step_size=1, refit=True)
        assert p1.shape == p2.shape
        assert t1.shape == t2.shape

    def test_invalid_n_windows_raises(self) -> None:
        adapter, _ = _make_adapter()
        with pytest.raises(ValueError, match="n_windows"):
            adapter.cross_validate(n_windows=0, step_size=1)

    def test_invalid_step_size_raises(self) -> None:
        adapter, _ = _make_adapter()
        with pytest.raises(ValueError, match="step_size"):
            adapter.cross_validate(n_windows=2, step_size=0)


# ===========================================================================
# Capability declarations
# ===========================================================================


class TestCapabilities:
    """isinstance checks for capability mixins."""

    def test_supports_refit(self) -> None:
        adapter, _ = _make_adapter()
        assert isinstance(adapter, SupportsRefit)

    def test_supports_cross_validation(self) -> None:
        adapter, _ = _make_adapter()
        assert isinstance(adapter, SupportsCrossValidation)

    def test_not_supports_quantiles(self) -> None:
        adapter, _ = _make_adapter()
        assert not isinstance(adapter, SupportsQuantiles)

    def test_not_supports_bootstrap(self) -> None:
        adapter, _ = _make_adapter()
        assert not isinstance(adapter, SupportsBootstrap)


# ===========================================================================
# SplitConformal calibration via cross_validate
# ===========================================================================


class TestSplitConformalCV:
    """SplitConformal.calibrate_cv(n_windows=...) dispatches to forecaster.cross_validate."""

    def test_cv_path_matches_explicit_path(self) -> None:
        """CV-based and predict_batch-based calibration produce the same quantile.

        MLForecast's ``cross_validation(refit=False)`` trains the model exactly
        once on the data up to the first cutoff (``T - h - (n_windows-1)*step_size``
        steps). The explicit path therefore needs to refit on the same prefix
        before producing batch predictions for the equivalence to hold.
        """
        adapter, df = _make_adapter(n_series=1, horizon=3, t_len=100)

        method_cv = SplitConformal(alpha=0.1)
        fitted_cv = method_cv.calibrate_cv(adapter, n_windows=10, step_size=1, refit=False)
        cal_cv = fitted_cv.calibration_

        cv_preds, cv_truths = adapter.cross_validate(n_windows=10, step_size=1, refit=False)
        full = adapter._df_to_panel(nw.from_native(df), "y")
        T = full.shape[1]
        cal_histories = [full[:, : T - 3 - (10 - 1 - w)] for w in range(10)]

        # Match the training data MLForecast's CV used internally.
        adapter.refit(full[:, : T - 3 - (10 - 1)])

        method_explicit = SplitConformal(alpha=0.1)
        fitted_explicit = method_explicit.calibrate(adapter, cal_histories, cv_truths)
        cal_explicit = fitted_explicit.calibration_

        np.testing.assert_array_almost_equal(
            cal_cv.diagnostics["score_quantile"], cal_explicit.diagnostics["score_quantile"]
        )
        assert np.array_equal(cal_cv.n_calibration_samples, cal_explicit.n_calibration_samples)

    def test_cv_calibration_then_predict(self) -> None:
        """End-to-end: calibrate via CV, then predict produces a valid interval."""
        adapter, df = _make_adapter(n_series=1, horizon=3, t_len=100)
        method = SplitConformal(alpha=0.1)
        fitted = method.calibrate_cv(adapter, n_windows=10, step_size=1)

        history = adapter._df_to_panel(nw.from_native(df), "y")
        result = fitted.predict(history)

        assert result.point.shape == (1, 1, 3)
        assert result.interval.shape == (1, 1, 3, 2)
        assert np.all(result.interval[..., 0] <= result.point)
        assert np.all(result.point <= result.interval[..., 1])


# ===========================================================================
# Anchoring: calendar features must be derived from the history's own dates
# ===========================================================================


class TestPredictAnchor:
    """The value-level consequence of anchoring ``predict`` at a fixed end.

    MLForecast derives ``date_features`` from the timestamps of the frame it is
    handed. When a shorter history was stamped as if it ended at the *training*
    end, every observation carried the wrong day-of-week, so the features -- and
    the forecast -- were wrong. Only the last window of a rolling
    ``predict_batch`` loop was correct.
    """

    def _weekly_df(self, t_len: int = 120) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-01-01", periods=t_len, freq="D")
        y = (
            10
            + 3 * np.sin(2 * np.pi * dates.dayofweek / 7)
            + 0.05 * np.arange(t_len)
            + rng.normal(scale=0.1, size=t_len)
        )
        return pd.DataFrame({"unique_id": "series_0", "ds": dates, "y": y})

    def test_prefix_history_matches_a_direct_mlforecast_call(self) -> None:
        df = self._weekly_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1], date_features=["dayofweek"])
        mlf.fit(df)
        adapter = MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="D", model_name="Ridge")
        panel = df.pivot(index="unique_id", columns="ds", values="y").to_numpy()

        for cut in (100, 105, 111):
            got = adapter.predict(panel[:, :cut])[0, 0, :]
            expected = mlf.predict(h=3, new_df=df.iloc[:cut])["Ridge"].to_numpy()
            np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)

    def test_anchor_shifts_with_panel_width(self) -> None:
        adapter, df = _make_adapter(n_series=2, horizon=3, t_len=100)
        panel = df.pivot(index="unique_id", columns="ds", values="y").to_numpy()

        assert adapter._anchor_shift(panel) == 0
        assert adapter._anchor_shift(panel[:, :-25]) == -25

    def test_recent_window_history_matches_a_direct_mlforecast_call(self) -> None:
        """A fixed-width sliding window ends where training ends, not in the past.

        Deriving the anchor from the panel width alone dated these rows from the
        *start* of training, so `date_features` were computed off by the width
        difference: right values, wrong calendar, and nothing raised.
        """
        df = self._weekly_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1], date_features=["dayofweek"])
        mlf.fit(df)
        adapter = MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="D", model_name="Ridge")
        panel = df.pivot(index="unique_id", columns="ds", values="y").to_numpy()

        for window in (30, 60, 90):
            got = adapter.predict(panel[:, -window:])[0, 0, :]
            expected = mlf.predict(h=3, new_df=df.iloc[-window:])["Ridge"].to_numpy()
            np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)

    def test_history_that_is_not_training_data_raises(self) -> None:
        adapter, df = _make_adapter(n_series=1, horizon=3, t_len=100)
        panel = df.pivot(index="unique_id", columns="ds", values="y").to_numpy()

        # An interior window is neither a prefix nor a recent window, so its
        # dates are unknowable -- say so rather than guess.
        with pytest.raises(ValueError, match="at either edge"):
            adapter.predict(panel[:, 20:60])


# ===========================================================================
# Exogenous features
# ===========================================================================


def _exog_df(t_len: int = 100, freq: str = "D", seed: int = 0) -> pd.DataFrame:
    """Long frame whose target is a strong positive function of ``price``."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=t_len, freq=freq)
    price = rng.standard_normal(t_len).cumsum()
    return pd.DataFrame(
        {
            "unique_id": "series_0",
            "ds": dates,
            "price": price,
            "y": 10 + 3 * price + rng.standard_normal(t_len) * 0.5,
        }
    )


def _exog_adapter(horizon: int = 3) -> tuple[MLForecastAdapter, pd.DataFrame]:
    df = _exog_df()
    mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
    mlf.fit(df, static_features=[])
    adapter = MLForecastAdapter(
        mlf=mlf,
        train_df=df,
        horizon=horizon,
        freq="D",
        model_name="Ridge",
        exog_features=["price"],
    )
    return adapter, df


class TestExogenousFeatures:
    """`MLForecast.predict` rebuilds its features from `new_df`.

    A target-only history is rejected with "The feature names should match
    those that were passed during fit", so `_panel_to_df` joins the regressors'
    history back on -- the same fix the StatsForecast adapter carries.
    """

    def test_predict_with_exogenous_features(self) -> None:
        adapter, df = _exog_adapter()
        panel = adapter._df_to_panel(nw.from_native(df), "y")
        out = adapter.predict(panel, X=np.ones((adapter.n_series, 1, 3, 1)))
        assert out.shape == (adapter.n_series, 1, 3)
        assert np.isfinite(out).all()

    def test_predict_actually_uses_the_future_regressors(self) -> None:
        """Shape alone would pass if the columns were silently ignored."""
        adapter, df = _exog_adapter()
        panel = adapter._df_to_panel(nw.from_native(df), "y")
        low = adapter.predict(panel, X=np.full((adapter.n_series, 1, 3, 1), -5.0))
        high = adapter.predict(panel, X=np.full((adapter.n_series, 1, 3, 1), 5.0))
        assert not np.allclose(low, high)
        # y is built as a positive function of 'price'.
        assert np.all(low < high)

    def test_predict_on_a_truncated_history(self) -> None:
        """The calibrate loop's path: each window is dated earlier than the last."""
        adapter, df = _exog_adapter()
        panel = adapter._df_to_panel(nw.from_native(df), "y")
        X = np.ones((adapter.n_series, 1, 3, 1))
        full = adapter.predict(panel, X=X)
        short = adapter.predict(panel[:, :-10], X=X)
        assert short.shape == full.shape
        assert np.isfinite(short).all()
        assert not np.allclose(full, short)

    def test_calibrate_and_predict_round_trip(self) -> None:
        """The workflow the missing support blocked, end to end through a method."""
        adapter, df = _exog_adapter()
        panel = adapter._df_to_panel(nw.from_native(df), "y")
        n_windows, horizon = 8, 3
        histories = [panel[:, : panel.shape[1] - horizon - k] for k in range(n_windows)][::-1]
        truths = np.stack([panel[:, h.shape[1] : h.shape[1] + horizon] for h in histories], axis=1)

        method = SplitConformal(alpha=0.2)
        fitted = method.calibrate(
            adapter, histories, truths, X=np.ones((adapter.n_series, n_windows, horizon, 1))
        )
        result = fitted.calibration_
        assert result.n_calibration_samples.shape == (adapter.n_series,)

        out = fitted.predict(panel, X=np.ones((adapter.n_series, 1, horizon, 1)))
        assert out.point.shape == (adapter.n_series, 1, horizon)
        assert np.all(out.interval[..., 0] <= out.point)
        assert np.all(out.point <= out.interval[..., 1])

    def test_cross_validate_uses_the_regressors(self) -> None:
        """CV takes no X and needs none: MLForecast splits the frame's exog itself.

        It only works if the adapter declares `static_features=[]` on its
        internal fit and cross_validation calls -- otherwise mlforecast infers
        `price` to be static and raises when its values change over time.
        """
        adapter, _ = _exog_adapter()
        preds, truths = adapter.cross_validate(n_windows=6, step_size=1)
        assert preds.shape == (adapter.n_series, 6, 3)
        assert not np.isnan(preds).any()
        assert not np.isnan(truths).any()

    def test_regressors_sharpen_the_interval(self) -> None:
        """The reason to want this: a better point forecast is a tighter interval."""
        adapter, df = _exog_adapter()
        with_exog = SplitConformal(alpha=0.1)
        fitted_with = with_exog.calibrate_cv(adapter, n_windows=20, step_size=1)

        bare_df = df[["unique_id", "ds", "y"]]
        bare_mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        bare_mlf.fit(bare_df, static_features=[])
        without = SplitConformal(alpha=0.1)
        fitted_without = without.calibrate_cv(
            MLForecastAdapter(
                mlf=bare_mlf, train_df=bare_df, horizon=3, freq="D", model_name="Ridge"
            ),
            n_windows=20,
            step_size=1,
        )

        # Same series, same model, same windows -- the only difference is
        # whether the forecaster could read 'price', which the target is built
        # from. Smaller residuals give a smaller conformal quantile.
        assert float(np.mean(fitted_with.score_quantile_)) < float(
            np.mean(fitted_without.score_quantile_)
        )

    def test_history_beyond_the_training_frame_raises(self) -> None:
        adapter, df = _exog_adapter()
        panel = adapter._df_to_panel(nw.from_native(df), "y")
        extended = np.concatenate([panel, np.zeros((adapter.n_series, 5))], axis=1)
        with pytest.raises(ValueError, match="No values for exogenous features"):
            adapter.predict(extended, X=np.ones((adapter.n_series, 1, 3, 1)))

    def test_exog_features_must_name_real_columns(self) -> None:
        df = _exog_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df, static_features=[])
        with pytest.raises(ValueError, match=r"exog_features \['nope'\] are not columns"):
            MLForecastAdapter(
                mlf=mlf,
                train_df=df,
                horizon=3,
                freq="D",
                model_name="Ridge",
                exog_features=["nope"],
            )

    def test_x_without_exog_features_raises(self) -> None:
        adapter, df = _make_adapter(t_len=100, horizon=3)
        panel = adapter._df_to_panel(nw.from_native(df), "y")
        with pytest.raises(ValueError, match="constructed without"):
            adapter.predict(panel, X=np.ones((adapter.n_series, 1, 3, 1)))

    def test_refit_is_refused_and_gated_in_supports(self) -> None:
        adapter, df = _exog_adapter()
        assert adapter.supports(SupportsRefit) is False
        # Capabilities that do not depend on exog are untouched.
        assert adapter.supports(SupportsCrossValidation) is True
        with pytest.raises(ValueError, match="refit is not available"):
            adapter.refit(adapter._df_to_panel(nw.from_native(df), "y"))

    def test_non_exog_adapter_still_supports_refit(self) -> None:
        adapter, _ = _make_adapter(t_len=100, horizon=3)
        assert adapter.supports(SupportsRefit) is True


# ===========================================================================
# Ragged end dates
# ===========================================================================


def _ragged_end_df(freq: str = "D") -> pd.DataFrame:
    """Both series start together, but 'b' stopped reporting ten steps early."""
    rng = np.random.default_rng(0)
    frames = [
        pd.DataFrame(
            {
                "unique_id": sid,
                "ds": pd.date_range("2020-01-01", periods=n, freq=freq),
                "y": rng.standard_normal(n).cumsum(),
            }
        )
        for sid, n in (("a", 60), ("b", 50))
    ]
    return pd.concat(frames, ignore_index=True)


class TestRaggedEndDates:
    """A series that stopped reporting early is an ordinary panel, not an error.

    Regression coverage for the reported ``KeyError: "['a'] not in index"``,
    which MLForecast reached exactly as StatsForecast did.
    """

    def _adapter(self, horizon: int = 3) -> tuple[MLForecastAdapter, pd.DataFrame]:
        df = _ragged_end_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df)
        adapter = MLForecastAdapter(
            mlf=mlf, train_df=df, horizon=horizon, freq="D", model_name="Ridge"
        )
        return adapter, df

    @pytest.mark.parametrize("aligned", [False, True])
    def test_calibrate_cv_succeeds(self, aligned: bool) -> None:
        adapter, _ = self._adapter()
        method = SplitConformal(alpha=0.2)
        fitted = method.calibrate_cv(adapter, n_windows=6, step_size=1, aligned=aligned)
        result = fitted.calibration_

        assert result.n_calibration_samples.shape == (2,)
        np.testing.assert_array_equal(result.n_calibration_samples, [6, 6])

    def test_predict_keeps_the_horizon_axis_and_the_right_values(self) -> None:
        """Shape alone is not enough: the old pivot also mis-slotted the values."""
        adapter, _ = self._adapter()
        method = SplitConformal(alpha=0.2)
        fitted = method.calibrate_cv(adapter, n_windows=6, step_size=1)

        result = fitted.predict(adapter._train_panel)
        assert result.point.shape == (2, 1, 3)
        assert result.interval.shape == (2, 1, 3, 2)
        assert not np.isnan(result.point).any()

        # Independent path: MLForecast forecasting from its own stored history.
        # Agreement proves the panel -> DataFrame -> predict round trip keeps
        # each series on its own step axis.
        expected = adapter.mlf.predict(h=3)
        for row, sid in enumerate(adapter._series_ids):
            want = expected.loc[expected["unique_id"] == sid, "Ridge"].to_numpy()
            np.testing.assert_allclose(result.point[row, 0], want)


# ===========================================================================
# Custom column names
# ===========================================================================

_CUSTOM = {"id_col": "item", "time_col": "time", "target_col": "value"}


def _custom_name_adapter(
    n_series: int = 2,
    t_len: int = 100,
    horizon: int = 3,
    freq: str = "D",
) -> tuple[MLForecastAdapter, pd.DataFrame]:
    """Adapter over an ``(item, time, value)`` frame, fit with those names."""
    df = _make_df(n_series=n_series, t_len=t_len, freq=freq, **_CUSTOM)
    mlf = MLForecast(models={"Ridge": Ridge()}, freq=freq, lags=[1, 2, 3])
    mlf.fit(df, static_features=[], **_CUSTOM)
    adapter = MLForecastAdapter(
        mlf=mlf, train_df=df, horizon=horizon, freq=freq, model_name="Ridge", **_CUSTOM
    )
    return adapter, df


class TestCustomColumnNames:
    """``id_col``/``time_col``/``target_col`` must reach the MLForecast calls.

    ``MLForecast.predict`` reuses the names ``fit`` recorded, so it was already
    fine; ``cross_validation`` and the ``fit`` calls behind ``refit`` and the
    post-CV repair take them as arguments and defaulted to ``unique_id``/``ds``/
    ``y``, which raised ``KeyError: 'unique_id'``.
    """

    def test_predict(self) -> None:
        adapter, df = _custom_name_adapter()
        forecast = adapter.predict(adapter._train_panel)
        assert forecast.shape == (2, 1, 3)
        assert np.isfinite(forecast).all()

    def test_cross_validate(self) -> None:
        adapter, _ = _custom_name_adapter()
        predictions, truths = adapter.cross_validate(n_windows=5, step_size=1)
        assert predictions.shape == truths.shape == (2, 5, 3)
        assert np.isfinite(predictions).all()
        assert np.isfinite(truths).all()

    def test_predict_still_works_after_cross_validate(self) -> None:
        """The post-CV repair refit must also carry the names."""
        adapter, _ = _custom_name_adapter()
        adapter.cross_validate(n_windows=5, step_size=1)
        assert np.isfinite(adapter.predict(adapter._train_panel)).all()

    def test_refit(self) -> None:
        adapter, _ = _custom_name_adapter()
        adapter.refit(adapter._train_panel)
        assert np.isfinite(adapter.predict(adapter._train_panel)).all()

    def test_end_to_end_split_conformal(self) -> None:
        adapter, _ = _custom_name_adapter(t_len=120)
        method = SplitConformal(alpha=0.1)
        fitted = method.calibrate_cv(adapter, n_windows=20, step_size=1)
        result = fitted.predict(adapter._train_panel)
        assert result.interval.shape == (2, 1, 3, 2)
        assert np.isfinite(result.interval).all()

    def test_output_matches_the_default_named_frame(self) -> None:
        """Renaming the columns must not change a single number."""
        custom_adapter, _ = _custom_name_adapter(t_len=120)
        default_adapter, _ = _make_adapter(n_series=2, t_len=120, horizon=3)

        np.testing.assert_array_equal(custom_adapter._train_panel, default_adapter._train_panel)
        np.testing.assert_array_equal(
            custom_adapter.predict(custom_adapter._train_panel),
            default_adapter.predict(default_adapter._train_panel),
        )
        custom_cv = custom_adapter.cross_validate(n_windows=5, step_size=1)
        default_cv = default_adapter.cross_validate(n_windows=5, step_size=1)
        np.testing.assert_array_equal(custom_cv[0], default_cv[0])
        np.testing.assert_array_equal(custom_cv[1], default_cv[1])

    def test_names_disagreeing_with_the_fitted_model_are_refused(self) -> None:
        """``MLForecast.predict`` takes no names, so a mismatch cannot be fixed later.

        Reported at construction, where the user can act on it, rather than as a
        ``KeyError`` from the reshaping code.
        """
        df = _make_df(n_series=2, t_len=100, **_CUSTOM)
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        default_df = df.rename(columns={"item": "unique_id", "time": "ds", "value": "y"})
        mlf.fit(default_df, static_features=[])
        with pytest.raises(ValueError, match="Column names disagree"):
            MLForecastAdapter(
                mlf=mlf, train_df=df, horizon=3, freq="D", model_name="Ridge", **_CUSTOM
            )
