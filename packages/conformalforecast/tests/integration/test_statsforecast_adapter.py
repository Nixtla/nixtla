"""Integration tests for StatsForecastAdapter."""

from __future__ import annotations

import narwhals as nw
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

sf_mod = pytest.importorskip("statsforecast")
StatsForecast = sf_mod.StatsForecast

from statsforecast.models import AutoARIMA, AutoETS, Naive, SeasonalNaive  # noqa: E402

from conformalforecast.adapters.statsforecast import StatsForecastAdapter  # noqa: E402
from conformalforecast.base import UnsupportedCapability  # noqa: E402
from conformalforecast.capabilities import (  # noqa: E402
    SupportsBootstrap,
    SupportsCrossValidation,
    SupportsCrossValidationQuantiles,
    SupportsQuantiles,
    SupportsRefit,
)
from conformalforecast.methods.cqr import (  # noqa: E402
    ConformalizedQuantileRegression,
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


def _fitted_sf(
    n_series: int = 1,
    t_len: int = 100,
    freq: str = "D",
    model_cls: type = Naive,
    **model_kwargs: object,
) -> tuple[StatsForecast, pd.DataFrame, str]:
    """Build and fit a StatsForecast instance, returning ``(sf, train_df, model_name)``."""
    df = _make_df(n_series=n_series, t_len=t_len, freq=freq)
    model = model_cls(**model_kwargs) if model_kwargs else model_cls()
    sf = StatsForecast(models=[model], freq=freq)
    sf.fit(df)
    model_name = model.alias if hasattr(model, "alias") else type(model).__name__
    return sf, df, model_name


def _ragged_fitted_sf(freq: str = "D") -> tuple[StatsForecast, pd.DataFrame, str]:
    """Build and fit a StatsForecast instance on a ragged panel.

    Series 'a' spans 40 days starting 2020-01-01. Series 'b' starts 10 steps
    later (2020-01-11) and spans only 30 days — both end on the same day, so
    the CV region (near the shared end) is fully overlapping while the early
    history is ragged (no common-range truncation drops 'b').
    """
    rng = np.random.default_rng(0)
    dates_a = pd.date_range("2020-01-01", periods=40, freq=freq)
    dates_b = pd.date_range("2020-01-11", periods=30, freq=freq)
    df = pd.concat(
        [
            pd.DataFrame({"unique_id": "a", "ds": dates_a, "y": rng.standard_normal(40).cumsum()}),
            pd.DataFrame({"unique_id": "b", "ds": dates_b, "y": rng.standard_normal(30).cumsum()}),
        ],
        ignore_index=True,
    )
    model = Naive()
    sf = StatsForecast(models=[model], freq=freq)
    sf.fit(df)
    model_name = model.alias if hasattr(model, "alias") else type(model).__name__
    return sf, df, model_name


def _ragged_end_fitted_sf(freq: str = "D") -> tuple[StatsForecast, pd.DataFrame, str]:
    """Build and fit a StatsForecast instance on a panel with ragged *end* dates.

    Both series start on 2020-01-01, but 'b' stopped reporting 10 steps before
    'a' — the ordinary "this series was retired" shape. StatsForecast derives CV
    cutoffs relative to each series' own end, so the two series land on disjoint
    cutoff sets and the reshaped cube is genuinely NaN-holed. This is the panel
    that used to raise ``KeyError: "['a'] not in index"``.
    """
    rng = np.random.default_rng(0)
    dates_a = pd.date_range("2020-01-01", periods=40, freq=freq)
    dates_b = pd.date_range("2020-01-01", periods=30, freq=freq)
    df = pd.concat(
        [
            pd.DataFrame({"unique_id": "a", "ds": dates_a, "y": rng.standard_normal(40).cumsum()}),
            pd.DataFrame({"unique_id": "b", "ds": dates_b, "y": rng.standard_normal(30).cumsum()}),
        ],
        ignore_index=True,
    )
    model = Naive()
    sf = StatsForecast(models=[model], freq=freq)
    sf.fit(df)
    model_name = model.alias if hasattr(model, "alias") else type(model).__name__
    return sf, df, model_name


def _fitted_sf_with_exog(
    cols: list[str],
    t_len: int = 60,
    freq: str = "D",
) -> tuple[StatsForecast, pd.DataFrame, str]:
    """Fit an AutoARIMA model with future-exogenous regressor columns.

    Returns ``(sf, train_df, model_name)``; ``train_df`` carries ``cols`` as
    historical exogenous values alongside ``unique_id``/``ds``/``y``.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=t_len, freq=freq)
    data: dict[str, object] = {
        "unique_id": "series_0",
        "ds": dates,
    }
    y = rng.standard_normal(t_len).cumsum()
    for col in cols:
        exog_vals = rng.standard_normal(t_len).cumsum()
        data[col] = exog_vals
        y = y + exog_vals
    data["y"] = y
    df = pd.DataFrame(data)
    model = AutoARIMA()
    sf = StatsForecast(models=[model], freq=freq)
    sf.fit(df[["unique_id", "ds", "y", *cols]])
    model_name = model.alias if hasattr(model, "alias") else type(model).__name__
    # The frame keeps `cols`, as the docstring says: the adapter needs the
    # regressors' history to hand back to sf.forecast. Returning a target-only
    # frame here is what made the exogenous predict test unpassable.
    return sf, df[["unique_id", "ds", "y", *cols]], model_name


def train_df_to_panel(
    train_df: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    freq: str = "D",
) -> np.ndarray:
    """Build a ragged ``(n_series, T)`` panel from a long-format train_df.

    ``T`` spans the earliest start to the latest end across all series (at
    ``freq``); series that start later or end earlier than the panel-wide
    range are left/right NaN-padded to align on the shared calendar grid.
    """
    series_ids = sorted(train_df[id_col].unique())
    full_range = pd.date_range(train_df[time_col].min(), train_df[time_col].max(), freq=freq)
    panel = np.full((len(series_ids), len(full_range)), np.nan, dtype=np.float64)
    for i, sid in enumerate(series_ids):
        sub = train_df.loc[train_df[id_col] == sid].set_index(time_col)[target_col]
        panel[i, :] = sub.reindex(full_range).to_numpy(dtype=np.float64)
    return panel


def _make_adapter(
    n_series: int = 1,
    t_len: int = 100,
    horizon: int = 3,
    freq: str = "D",
    model_cls: type = Naive,
    **model_kwargs: object,
) -> tuple[StatsForecastAdapter, pd.DataFrame]:
    """Fit and return an adapter plus its training DataFrame."""
    sf, df, model_name = _fitted_sf(
        n_series=n_series, t_len=t_len, freq=freq, model_cls=model_cls, **model_kwargs
    )
    adapter = StatsForecastAdapter(
        sf=sf, train_df=df, horizon=horizon, freq=freq, model_name=model_name
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

    def test_multi_model_sf(self) -> None:
        """Construct from a multi-model SF, selecting one model."""
        df = _make_df(n_series=2, t_len=100)
        sf = StatsForecast(models=[Naive(), AutoETS(season_length=1)], freq="D")
        sf.fit(df)
        adapter = StatsForecastAdapter(sf=sf, train_df=df, horizon=5, freq="D", model_name="Naive")
        assert adapter.n_series == 2
        assert adapter.model_name == "Naive"

    # --- validation failure tests ---

    def test_not_fitted_raises(self) -> None:
        df = _make_df()
        sf = StatsForecast(models=[Naive()], freq="D")
        with pytest.raises(ValueError, match="not fitted"):
            StatsForecastAdapter(sf=sf, train_df=df, horizon=3, freq="D", model_name="Naive")

    def test_bad_model_name_raises(self) -> None:
        df = _make_df()
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.fit(df)
        with pytest.raises(ValueError, match="model_name 'NoSuchModel'"):
            StatsForecastAdapter(sf=sf, train_df=df, horizon=3, freq="D", model_name="NoSuchModel")

    def test_missing_columns_raises(self) -> None:
        df = _make_df()
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.fit(df)
        bad_df = df.rename(columns={"y": "target"})
        with pytest.raises(ValueError, match="missing required columns"):
            StatsForecastAdapter(sf=sf, train_df=bad_df, horizon=3, freq="D", model_name="Naive")

    def test_invalid_freq_raises(self) -> None:
        df = _make_df()
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.fit(df)
        with pytest.raises(ValueError, match="Invalid frequency"):
            StatsForecastAdapter(sf=sf, train_df=df, horizon=3, freq="BOGUS", model_name="Naive")

    def test_non_contiguous_raises(self) -> None:
        df = _make_df(t_len=100)
        # Drop a row to create a gap
        df = df.drop(index=50).reset_index(drop=True)
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.fit(df)
        with pytest.raises(ValueError, match="Non-contiguous"):
            StatsForecastAdapter(sf=sf, train_df=df, horizon=3, freq="D", model_name="Naive")

    def test_nan_target_raises(self) -> None:
        df = _make_df()
        df.loc[10, "y"] = np.nan
        sf = StatsForecast(models=[Naive()], freq="D")
        # Fit on clean data
        clean_df = _make_df()
        sf.fit(clean_df)
        with pytest.raises(ValueError, match="NaN"):
            StatsForecastAdapter(sf=sf, train_df=df, horizon=3, freq="D", model_name="Naive")

    def test_non_overlapping_series_both_survive(self) -> None:
        """Series with non-overlapping ranges no longer get intersected away.

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
                        "y": rng.standard_normal(50),
                    }
                ),
                pd.DataFrame(
                    {
                        "unique_id": "b",
                        "ds": dates2,
                        "y": rng.standard_normal(50),
                    }
                ),
            ],
            ignore_index=True,
        )
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.fit(df)
        adapter = StatsForecastAdapter(sf=sf, train_df=df, horizon=3, freq="D", model_name="Naive")
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
                        "y": rng.standard_normal(50),
                    }
                ),
                pd.DataFrame(
                    {
                        "unique_id": "b",
                        "ds": dates_b,
                        "y": rng.standard_normal(10),
                    }
                ),
            ],
            ignore_index=True,
        )
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.fit(df)
        with pytest.raises(ValueError, match=r"\['b'\] span fewer than"):
            StatsForecastAdapter(sf=sf, train_df=df, horizon=20, freq="D", model_name="Naive")

    def test_construction_is_keyword_only_and_n_series_is_derived(self) -> None:
        sf, train_df, model_name = _fitted_sf()
        with pytest.raises(TypeError):
            StatsForecastAdapter(sf, train_df, 3, "D", model_name)  # type: ignore[misc]

        adapter = StatsForecastAdapter(
            sf=sf, train_df=train_df, horizon=3, freq="D", model_name=model_name
        )
        assert adapter.n_series == train_df["unique_id"].nunique()
        assert adapter.horizon == 3

    def test_passing_a_derived_field_raises(self) -> None:
        sf, train_df, model_name = _fitted_sf()
        with pytest.raises(ValidationError, match="derived from train_df"):
            StatsForecastAdapter(
                sf=sf,
                train_df=train_df,
                horizon=3,
                freq="D",
                model_name=model_name,
                n_series=99,  # type: ignore[call-arg]
            )

    def test_not_statsforecast_instance_raises(self) -> None:
        df = _make_df()
        with pytest.raises(ValueError, match="StatsForecast instance"):
            StatsForecastAdapter(
                sf="not_sf",  # type: ignore[arg-type]
                train_df=df,
                horizon=3,
                freq="D",
                model_name="Naive",
            )


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
        # Use different subset of history (different last values)
        history_short = history_full[:, :50]
        r1 = adapter.predict(history_full)
        r2 = adapter.predict(history_short)
        # Naive model repeats last value, so different histories → different forecasts
        assert not np.array_equal(r1, r2)


# ===========================================================================
# refit tests
# ===========================================================================


class TestRefit:
    """refit() updates the model and anchoring."""

    def test_refit_with_longer_panel_shifts_end(self) -> None:
        adapter, df = _make_adapter(n_series=1, horizon=3, t_len=80)
        old_ends = adapter._series_ends.to_list()

        # Create a longer panel (20 extra steps = 20 new days)
        new_df = _make_df(n_series=1, t_len=100, seed=1)
        new_panel = adapter._df_to_panel(nw.from_native(new_df), "y")

        adapter.refit(new_panel)

        # Every series' end timestamp should have shifted forward by 20 days
        assert adapter._series_ends.to_list() == [e + pd.Timedelta(days=20) for e in old_ends]
        # Predict still works
        result = adapter.predict(new_panel)
        assert result.shape == (1, 1, 3)

    def test_refit_updates_train_df_so_divergent_adapters_differ(self) -> None:
        """train_df is the training frame, not the construction argument.

        ArrayModel.__eq__ compares declared fields elementwise, so a stale
        ``train_df`` would make two adapters refit to *different* data compare
        equal. There is one training frame and ``refit`` replaces it.
        """
        sf, df, model_name = _fitted_sf(n_series=1, t_len=80)
        # Same wrapped model and frame, so the two start out equal; only the
        # training frame diverges below.
        a, b = (
            StatsForecastAdapter(sf=sf, train_df=df, horizon=3, freq="D", model_name=model_name)
            for _ in range(2)
        )
        assert a == b

        panel_a = a._df_to_panel(nw.from_native(_make_df(n_series=1, t_len=100, seed=1)), "y")
        panel_b = b._df_to_panel(nw.from_native(_make_df(n_series=1, t_len=100, seed=2)), "y")
        a.refit(panel_a)
        b.refit(panel_b)

        assert not a.train_df.equals(b.train_df)
        assert a != b

    def test_refit_with_different_series_raises(self) -> None:
        adapter, _ = _make_adapter(n_series=2, horizon=3)
        # Create panel with 3 series instead of 2
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

    def test_supports_quantiles(self) -> None:
        adapter, _ = _make_adapter()
        assert isinstance(adapter, SupportsQuantiles)

    def test_supports_cross_validation_quantiles(self) -> None:
        adapter, _ = _make_adapter()
        assert isinstance(adapter, SupportsCrossValidationQuantiles)

    def test_not_supports_bootstrap(self) -> None:
        adapter, _ = _make_adapter()
        assert not isinstance(adapter, SupportsBootstrap)


# ===========================================================================
# SplitConformal calibration via cross_validate
# ===========================================================================


# ===========================================================================
# Quantile prediction
# ===========================================================================


def _make_quantile_adapter(
    n_series: int = 1,
    t_len: int = 100,
    horizon: int = 3,
    freq: str = "D",
) -> tuple[StatsForecastAdapter, pd.DataFrame]:
    """Adapter backed by AutoETS — a model that supports level-based intervals."""
    return _make_adapter(
        n_series=n_series,
        t_len=t_len,
        horizon=horizon,
        freq=freq,
        model_cls=AutoETS,
        season_length=1,
    )


class TestPredictQuantiles:
    """predict_quantiles shape, validation, and column resolution."""

    @pytest.mark.parametrize("quantiles", [[0.1, 0.9], [0.05, 0.95]])
    def test_output_shape(self, quantiles: list[float]) -> None:
        adapter, df = _make_quantile_adapter(n_series=2, horizon=4, t_len=100)
        history = adapter._df_to_panel(nw.from_native(df), "y")
        result = adapter.predict_quantiles(history, np.array(quantiles))
        assert result.shape == (2, len(quantiles), 4)
        # Lower < Upper for symmetric pairs.
        np.testing.assert_array_less(result[:, 0, :], result[:, 1, :])

    def test_asymmetric_quantiles_rejected(self) -> None:
        adapter, df = _make_quantile_adapter()
        history = adapter._df_to_panel(nw.from_native(df), "y")
        with pytest.raises(ValueError, match="symmetric pair"):
            adapter.predict_quantiles(history, np.array([0.1, 0.8]))

    def test_median_rejected(self) -> None:
        adapter, df = _make_quantile_adapter()
        history = adapter._df_to_panel(nw.from_native(df), "y")
        with pytest.raises(ValueError, match="0.5 .median."):
            adapter.predict_quantiles(history, np.array([0.1, 0.5, 0.9]))

    def test_out_of_range_rejected(self) -> None:
        adapter, df = _make_quantile_adapter()
        history = adapter._df_to_panel(nw.from_native(df), "y")
        with pytest.raises(ValueError, match="0 < q < 1"):
            adapter.predict_quantiles(history, np.array([0.0, 1.0]))

    def test_missing_quantile_column_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the underlying forecast output is missing the expected column, raise.

        Most modern StatsForecast models (Naive, AutoETS, AutoARIMA, …) compute
        analytical level-based intervals, so it is hard to elicit a missing
        ``-lo-/-hi-`` column from a real fitted model. We monkey-patch the
        StatsForecast ``forecast`` to drop interval columns, simulating the
        case where the user picked a model that does not support level=.
        """
        adapter, df = _make_quantile_adapter(n_series=1, t_len=100, horizon=3)
        history = adapter._df_to_panel(nw.from_native(df), "y")

        real_forecast = adapter.sf.forecast

        def fake_forecast(*args: object, **kwargs: object) -> pd.DataFrame:
            result = real_forecast(*args, **kwargs)
            interval_cols = [c for c in result.columns if "-lo-" in c or "-hi-" in c]
            return result.drop(columns=interval_cols)

        monkeypatch.setattr(adapter.sf, "forecast", fake_forecast)

        with pytest.raises(ValueError, match="did not return quantile column"):
            adapter.predict_quantiles(history, np.array([0.05, 0.95]))


class TestCrossValidateQuantiles:
    """cross_validate_quantiles shapes, ordering, and refit behavior."""

    def test_output_shapes(self) -> None:
        adapter, _ = _make_quantile_adapter(n_series=2, horizon=3, t_len=100)
        preds, truths = adapter.cross_validate_quantiles(
            n_windows=4, step_size=1, quantiles=np.array([0.1, 0.9])
        )
        assert preds.shape == (2, 4, 3, 2)
        assert truths.shape == (2, 4, 3)

    def test_quantile_order_preserved(self) -> None:
        """Output quantile axis follows the input order, not sorted."""
        adapter, _ = _make_quantile_adapter(n_series=1, horizon=3, t_len=100)
        preds_lo_hi, _ = adapter.cross_validate_quantiles(
            n_windows=3, step_size=1, quantiles=np.array([0.1, 0.9])
        )
        preds_hi_lo, _ = adapter.cross_validate_quantiles(
            n_windows=3, step_size=1, quantiles=np.array([0.9, 0.1])
        )
        # Reverse order of the last axis should yield the swapped result.
        np.testing.assert_array_almost_equal(preds_lo_hi[..., ::-1], preds_hi_lo)

    def test_refit_true_and_false_produce_shapes(self) -> None:
        adapter, _ = _make_quantile_adapter(n_series=1, horizon=3, t_len=100)
        p_no, _ = adapter.cross_validate_quantiles(
            n_windows=3, step_size=1, quantiles=np.array([0.1, 0.9]), refit=False
        )
        p_yes, _ = adapter.cross_validate_quantiles(
            n_windows=3, step_size=1, quantiles=np.array([0.1, 0.9]), refit=True
        )
        assert p_no.shape == p_yes.shape == (1, 3, 3, 2)

    def test_invalid_n_windows_raises(self) -> None:
        adapter, _ = _make_quantile_adapter()
        with pytest.raises(ValueError, match="n_windows"):
            adapter.cross_validate_quantiles(
                n_windows=0, step_size=1, quantiles=np.array([0.1, 0.9])
            )

    def test_invalid_step_size_raises(self) -> None:
        adapter, _ = _make_quantile_adapter()
        with pytest.raises(ValueError, match="step_size"):
            adapter.cross_validate_quantiles(
                n_windows=2, step_size=0, quantiles=np.array([0.1, 0.9])
            )


# ===========================================================================
# CQR end-to-end via the CV path is exercised in test_cqr_nixtla.py.
# ===========================================================================


class TestSplitConformalCV:
    """SplitConformal.calibrate_cv(n_windows=...) dispatches to forecaster.cross_validate."""

    def test_cv_path_matches_explicit_path(self) -> None:
        """CV-based and predict_batch-based calibration produce the same quantile."""
        adapter, df = _make_adapter(n_series=1, horizon=3, t_len=100)

        # Fast path: delegate to cross_validate
        method_cv = SplitConformal(alpha=0.1)
        fitted_cv = method_cv.calibrate_cv(adapter, n_windows=10, step_size=1, refit=False)
        cal_cv = fitted_cv.calibration_

        # Slow path: replicate cross_validate manually via predict_batch
        cv_preds, cv_truths = adapter.cross_validate(n_windows=10, step_size=1, refit=False)
        # Build histories matching the CV cutoffs.
        full = adapter._df_to_panel(nw.from_native(df), "y")
        T = full.shape[1]
        cal_histories = [full[:, : T - 3 - (10 - 1 - w)] for w in range(10)]

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
        # Interval should bracket the point forecast.
        assert np.all(result.interval[..., 0] <= result.point)
        assert np.all(result.point <= result.interval[..., 1])


# ===========================================================================
# Ragged panel end-to-end
# ===========================================================================


def test_ragged_panel_end_to_end() -> None:
    """A panel where series have different lengths keeps all data
    (no silent common-range truncation) and forecasts end-to-end."""
    sf, train_df, model_name = _ragged_fitted_sf()  # series 'b' starts 10 steps late
    adapter = StatsForecastAdapter(
        sf=sf, train_df=train_df, horizon=3, freq="D", model_name=model_name
    )
    # Both series survive construction (no intersection dropped 'b').
    assert set(adapter._series_ids) == {"a", "b"}

    method = SplitConformal(alpha=0.2)
    fitted = method.calibrate_cv(adapter, n_windows=8, step_size=1)
    result = fitted.calibration_
    assert result.n_calibration_samples.shape == (2,)
    pred = fitted.predict(train_df_to_panel(train_df))  # helper: latest panel
    assert pred.interval.shape[0] == 2


def test_cv_modes_smoke_statsforecast() -> None:
    """Smoke test: ``aligned`` has no observable effect on a *same-end* panel.

    ``StatsForecast.cross_validation`` derives cutoffs per series relative to
    that series' OWN end date, not a shared calendar. For a same-end panel
    (as built by ``_ragged_fitted_sf``) this means every series' cutoffs
    coincide, so the resulting CV cube is already per-series-dense (no NaN)
    regardless of ``aligned``, and the two modes are provably identical.

    The divergence the flag exists for — NaN preservation (aligned=True) vs.
    densification (aligned=False) — needs *ragged end dates* to appear, and is
    covered by ``TestRaggedEndDates`` below.
    """
    sf, train_df, model_name = _ragged_fitted_sf()  # from Task 7
    adapter = StatsForecastAdapter(
        sf=sf, train_df=train_df, horizon=2, freq="D", model_name=model_name
    )

    preds_a, truths_a = adapter.cross_validate(n_windows=6, step_size=1, aligned=True)
    preds_d, truths_d = adapter.cross_validate(n_windows=6, step_size=1, aligned=False)

    # Both series present in both modes.
    assert preds_a.shape[0] == 2
    assert preds_d.shape[0] == 2

    # StatsForecast's per-series-relative cutoffs make the output dense.
    assert not np.isnan(preds_a).any()
    assert not np.isnan(preds_d).any()
    assert not np.isnan(truths_a).any()
    assert not np.isnan(truths_d).any()

    # aligned has no observable effect: identical output either way.
    np.testing.assert_array_equal(preds_a, preds_d)
    np.testing.assert_array_equal(truths_a, truths_d)


class TestRaggedEndDates:
    """A series that stopped reporting early is an ordinary panel, not an error.

    Regression coverage for the reported ``KeyError: "['a'] not in index"`` out
    of the old calendar-union pivot: with different end dates each series gets
    its own cutoff
    set, so a cutoff group is missing whichever series does not reach it.
    """

    def _adapter(self, horizon: int = 3) -> tuple[StatsForecastAdapter, pd.DataFrame]:
        sf, train_df, model_name = _ragged_end_fitted_sf()
        adapter = StatsForecastAdapter(
            sf=sf, train_df=train_df, horizon=horizon, freq="D", model_name=model_name
        )
        return adapter, train_df

    @pytest.mark.parametrize("aligned", [False, True])
    def test_calibrate_cv_succeeds(self, aligned: bool) -> None:
        adapter, _ = self._adapter()
        method = SplitConformal(alpha=0.2)
        fitted = method.calibrate_cv(adapter, n_windows=8, step_size=1, aligned=aligned)
        result = fitted.calibration_

        assert result.n_calibration_samples.shape == (2,)
        # Every series contributes all 8 of its own windows either way: dense
        # mode densifies them, aligned mode leaves NaN the method then masks.
        np.testing.assert_array_equal(result.n_calibration_samples, [8, 8])

    def test_cutoff_axis_is_the_union_and_aligned_modes_diverge(self) -> None:
        """The two modes are no longer interchangeable once the ends differ."""
        adapter, _ = self._adapter(horizon=2)
        preds_a, truths_a = adapter.cross_validate(n_windows=6, step_size=1, aligned=True)
        preds_d, truths_d = adapter.cross_validate(n_windows=6, step_size=1, aligned=False)

        # aligned: the union of both series' disjoint cutoff sets, NaN-holed.
        assert preds_a.shape == (2, 12, 2)
        assert np.isnan(preds_a).any() and np.isnan(truths_a).any()
        # Each series is valid in exactly its own 6 windows.
        valid = ~np.isnan(preds_a).any(axis=2)
        np.testing.assert_array_equal(valid.sum(axis=1), [6, 6])

        # dense: each series' own 6 windows, no NaN left for the method to mask.
        assert preds_d.shape == (2, 6, 2)
        assert not np.isnan(preds_d).any() and not np.isnan(truths_d).any()
        # Same numbers, just compacted.
        np.testing.assert_array_equal(preds_d[0], preds_a[0][valid[0]])
        np.testing.assert_array_equal(truths_d[1], truths_a[1][valid[1]])

    def test_quantile_cv_densifies_instead_of_raising(self) -> None:
        """The CQR path used to call any NaN an internal library error."""
        sf, train_df, model_name = _ragged_end_fitted_sf()
        # Naive has no interval support; refit with a model that does.
        sf = StatsForecast(models=[AutoETS(season_length=1)], freq="D")
        sf.fit(train_df)
        adapter = StatsForecastAdapter(
            sf=sf, train_df=train_df, horizon=2, freq="D", model_name="AutoETS"
        )

        qp, truths = adapter.cross_validate_quantiles(
            n_windows=6, step_size=1, quantiles=np.array([0.1, 0.9])
        )
        # Densified to each series' own 6 windows, quantiles on the last axis.
        assert qp.shape == (2, 6, 2, 2)
        assert truths.shape == (2, 6, 2)
        assert not np.isnan(qp).any()
        assert not np.isnan(truths).any()
        assert np.all(qp[..., 0] <= qp[..., 1])

    def test_predict_keeps_the_horizon_axis_and_the_right_values(self) -> None:
        """The union pivot used to widen this past ``horizon``, silently.

        Shape alone is not enough: on the old pivot each series also landed on
        its own diagonal, so half the cells were NaN and the rest were shifted.
        The model is ``Naive``, so the whole forecast is each series' own last
        observation and the values can be asserted exactly.
        """
        adapter, train_df = self._adapter()
        method = SplitConformal(alpha=0.2)
        fitted = method.calibrate_cv(adapter, n_windows=8, step_size=1)

        result = fitted.predict(adapter._train_panel)
        assert result.point.shape == (2, 1, 3)
        assert result.interval.shape == (2, 1, 3, 2)
        assert not np.isnan(result.point).any()
        assert not np.isnan(result.interval).any()

        for row, sid in enumerate(adapter._series_ids):
            last = train_df.loc[train_df["unique_id"] == sid, "y"].to_numpy()[-1]
            np.testing.assert_allclose(result.point[row, 0], np.full(3, last))


# ===========================================================================
# Exogenous X on predict (Task 9)
# ===========================================================================


def _exog_setup(horizon: int = 3) -> tuple[StatsForecastAdapter, pd.DataFrame]:
    sf, train_df, model_name = _fitted_sf_with_exog(["price"])
    adapter = StatsForecastAdapter(
        sf=sf,
        train_df=train_df,
        horizon=horizon,
        freq="D",
        model_name=model_name,
        exog_features=["price"],
    )
    return adapter, train_df


def test_predict_with_exogenous_features() -> None:
    """Reference Nixtla exog path: forecast through the adapter with a 4-D X.

    ``sf.forecast(df=..., X_df=...)`` is a stateless fit-and-predict that
    re-derives its regressor count from ``df``. This used to be ``xfail``
    because ``_panel_to_df`` built a target-only frame, so ``df`` implied zero
    regressors while ``X_df`` carried one and StatsForecast raised
    ``Expected X to have shape (3, 2), but got (3, 3)``. ``_panel_to_df`` now
    joins the regressors' history back on from the training frame.
    """
    adapter, train_df = _exog_setup()
    panel = train_df_to_panel(train_df)
    X = np.ones((adapter.n_series, 1, 3, 1))  # future 'price' = 1.0
    out = adapter.predict(panel, X=X)
    assert out.shape == (adapter.n_series, 1, 3)
    assert np.isfinite(out).all()


def test_predict_actually_uses_the_future_regressors() -> None:
    """Shape alone would pass if the columns were silently ignored again."""
    adapter, train_df = _exog_setup()
    panel = train_df_to_panel(train_df)
    low = adapter.predict(panel, X=np.full((adapter.n_series, 1, 3, 1), -5.0))
    high = adapter.predict(panel, X=np.full((adapter.n_series, 1, 3, 1), 5.0))
    assert not np.allclose(low, high)
    # The helper builds y as a positive function of 'price'.
    assert np.all(low < high)


def test_predict_on_a_truncated_history_picks_up_its_own_regressors() -> None:
    """The calibrate loop's path: each window is dated earlier than the last.

    The exog join keys on (id, ds), and `panel_to_df_ragged` stamps those dates
    from the same anchor `exog_to_futr_df` continues from — so a prefix panel
    must pick up the regressor values for *its* dates, not the training end's.
    """
    adapter, train_df = _exog_setup()
    panel = train_df_to_panel(train_df)
    X = np.ones((adapter.n_series, 1, 3, 1))

    full = adapter.predict(panel, X=X)
    short = adapter.predict(panel[:, :-10], X=X)
    assert short.shape == full.shape
    assert np.isfinite(short).all()
    # Different history, different forecast origin, so different numbers.
    assert not np.allclose(full, short)


def test_calibrate_and_predict_round_trip_with_exog() -> None:
    """The workflow the defect actually blocked, end to end through a method."""
    adapter, train_df = _exog_setup()
    panel = train_df_to_panel(train_df)

    n_windows, horizon = 8, 3
    histories = [panel[:, : panel.shape[1] - horizon - k] for k in range(n_windows)][::-1]
    truths = np.stack([panel[:, h.shape[1] : h.shape[1] + horizon] for h in histories], axis=1)
    X_cal = np.ones((adapter.n_series, n_windows, horizon, 1))

    method = SplitConformal(alpha=0.2)
    fitted = method.calibrate(adapter, histories, truths, X=X_cal)
    result = fitted.calibration_
    assert result.n_calibration_samples.shape == (adapter.n_series,)

    out = fitted.predict(panel, X=np.ones((adapter.n_series, 1, horizon, 1)))
    assert out.point.shape == (adapter.n_series, 1, horizon)
    assert out.interval.shape == (adapter.n_series, 1, horizon, 2)
    assert np.all(out.interval[..., 0] <= out.point)
    assert np.all(out.point <= out.interval[..., 1])


def test_history_beyond_the_training_frame_raises_naming_the_dates() -> None:
    """Past the training end the regressors are unknown -- say so, don't send NaN."""
    adapter, train_df = _exog_setup()
    panel = train_df_to_panel(train_df)
    extended = np.concatenate([panel, np.zeros((adapter.n_series, 5))], axis=1)
    with pytest.raises(ValueError, match=r"No values for exogenous features.*2020-03"):
        adapter.predict(extended, X=np.ones((adapter.n_series, 1, 3, 1)))


def test_exog_features_must_name_real_columns() -> None:
    """Caught at construction, not as a shape error from inside statsforecast."""
    sf, train_df, model_name = _fitted_sf_with_exog(["price"])
    with pytest.raises(ValueError, match=r"exog_features \['nope'\] are not columns"):
        StatsForecastAdapter(
            sf=sf,
            train_df=train_df,
            horizon=3,
            freq="D",
            model_name=model_name,
            exog_features=["nope"],
        )


def test_undeclared_regressor_columns_are_rejected() -> None:
    """`exog_features` must account for every regressor column in `train_df`.

    `sf.forecast` re-derives its regressor set from the frame `predict` hands
    it, which carries only the declared features -- but `cross_validate` passes
    `train_df` whole, so StatsForecast reads every non-key column there. Leave a
    column out of `exog_features` and calibration measures a model with more
    regressors than prediction uses, with nothing raised.
    """
    sf, train_df, model_name = _fitted_sf_with_exog(["price", "promo"])
    with pytest.raises(ValueError, match=r"promo"):
        StatsForecastAdapter(
            sf=sf,
            train_df=train_df,
            horizon=3,
            freq="D",
            model_name=model_name,
            exog_features=["price"],
        )


def test_regressor_columns_without_exog_features_are_rejected() -> None:
    """The likelier version of the same split: pass the fitted frame, forget the kwarg.

    `calibrate_cv` then runs an exogenous model while `predict` runs a bare one,
    so the conformal quantile is measured against forecasts the method will
    never produce.
    """
    sf, train_df, model_name = _fitted_sf_with_exog(["price"])
    with pytest.raises(ValueError, match=r"price"):
        StatsForecastAdapter(sf=sf, train_df=train_df, horizon=3, freq="D", model_name=model_name)


def test_refit_is_refused_for_an_exog_adapter() -> None:
    """A history panel is target-only, so a refit would drop the regressors."""
    adapter, train_df = _exog_setup()
    with pytest.raises(ValueError, match="refit is not available"):
        adapter.refit(train_df_to_panel(train_df))


# ===========================================================================
# Anchoring: predict dates follow the history panel's own width
# ===========================================================================


class TestPredictAnchor:
    """A shorter history must be dated earlier, not stamped with the training end."""

    def test_shorter_history_is_dated_earlier(self) -> None:
        adapter, df = _make_adapter(n_series=2, horizon=3, t_len=100)
        full = adapter._df_to_panel(nw.from_native(df), "y")

        assert adapter._anchor_shift(full) == 0

        # 40 columns short -> every series ends 40 steps earlier.
        assert adapter._anchor_shift(full[:, :-40]) == -40

    def test_recent_window_is_dated_at_the_training_end(self) -> None:
        """A sliding window keeps its end; only a prefix moves it back.

        Both are 60 columns wide, so width alone cannot tell them apart -- which
        is why the anchor is located by value instead.
        """
        adapter, df = _make_adapter(n_series=2, horizon=3, t_len=100)
        full = adapter._df_to_panel(nw.from_native(df), "y")

        assert adapter._anchor_shift(full[:, -60:]) == 0
        assert adapter._anchor_shift(full[:, :60]) == -40

    def test_unlocatable_history_raises_instead_of_guessing(self) -> None:
        adapter, df = _make_adapter(n_series=2, horizon=3, t_len=100)
        full = adapter._df_to_panel(nw.from_native(df), "y")
        with pytest.raises(ValueError, match="at either edge"):
            adapter.predict(full[:, 20:60])

    def test_rolling_windows_get_distinct_forecast_dates(self) -> None:
        """The defect this guards: every window used to be stamped with the same dates."""
        adapter, df = _make_adapter(n_series=1, horizon=3, t_len=100)
        full = adapter._df_to_panel(nw.from_native(df), "y")

        last_dates = []
        for cut in (60, 70, 80):
            frame = adapter._panel_to_df(full[:, :cut])
            last_dates.append(frame["ds"].max())

        assert len(set(last_dates)) == 3, last_dates
        assert last_dates[1] - last_dates[0] == pd.Timedelta(days=10)
        assert last_dates[2] - last_dates[1] == pd.Timedelta(days=10)

    def test_forecast_timestamps_follow_the_history(self) -> None:
        """The forecast frame's dates continue the history, not the training end."""
        adapter, df = _make_adapter(n_series=1, horizon=3, t_len=100)
        full = adapter._df_to_panel(nw.from_native(df), "y")

        cut = 70
        history_df = adapter._panel_to_df(full[:, :cut])
        result_df = adapter.sf.forecast(h=3, df=history_df.to_native())
        expected_first = history_df["ds"].max() + pd.Timedelta(days=1)
        assert result_df["ds"].min() == expected_first
        assert expected_first < adapter._series_ends[adapter._series_ids.index("series_0")]

    def test_positional_models_are_value_identical_either_way(self) -> None:
        """Why the statsforecast coverage above is date-level, not value-level.

        SeasonalNaive and friends index backward from the end of the array, so
        their *values* never depended on the anchor -- only their timestamps did.
        The value-level regression this fix prevents is in the MLForecast
        adapter, where calendar features are derived from the dates; see
        `test_mlforecast_adapter.py::TestPredictAnchor`.
        """
        adapter, df = _make_adapter(
            n_series=1, horizon=1, t_len=100, model_cls=SeasonalNaive, season_length=7
        )
        full = adapter._df_to_panel(nw.from_native(df), "y")
        for cut in (70, 75, 93):
            got = adapter.predict(full[:, :cut])[0, 0, 0]
            assert got == pytest.approx(full[0, cut - 7]), cut


# ===========================================================================
# predict_quantiles refuses to silently drop exogenous regressors
# ===========================================================================


def test_predict_quantiles_with_exog_raises() -> None:
    """SupportsQuantiles has no X, so an exog adapter must refuse, not guess."""
    sf, train_df, model_name = _fitted_sf_with_exog(["price"])
    adapter = StatsForecastAdapter(
        sf=sf,
        train_df=train_df,
        horizon=3,
        freq="D",
        model_name=model_name,
        exog_features=["price"],
    )
    panel = train_df_to_panel(train_df)
    with pytest.raises(UnsupportedCapability, match="predict_quantiles"):
        adapter.predict_quantiles(panel, np.array([0.1, 0.9]))


def _exog_adapter() -> StatsForecastAdapter:
    sf, train_df, model_name = _fitted_sf_with_exog(["price"])
    return StatsForecastAdapter(
        sf=sf,
        train_df=train_df,
        horizon=3,
        freq="D",
        model_name=model_name,
        exog_features=["price"],
    )


def test_supports_reports_no_quantiles_for_an_exog_adapter() -> None:
    """The gate belongs in `supports`, so it fires at construction."""
    adapter = _exog_adapter()
    assert adapter.supports(SupportsQuantiles) is False
    assert adapter.supports(SupportsCrossValidationQuantiles) is False
    # Refit is gated too: a history panel carries target values only, so
    # refitting an exog model would silently retrain it without its regressors.
    assert adapter.supports(SupportsRefit) is False
    # Capabilities that do not depend on exog are untouched. CV needs no X --
    # StatsForecast splits the training frame's regressor columns per window.
    assert adapter.supports(SupportsCrossValidation) is True


def test_cqr_on_an_exog_adapter_fails_at_calibrate_not_predict() -> None:
    """Without this, the user pays for a full CV calibration before finding out."""
    adapter = _exog_adapter()
    with pytest.raises(UnsupportedCapability):
        ConformalizedQuantileRegression(alpha=0.1).calibrate_cv(adapter, n_windows=5)


def test_non_exog_adapter_still_supports_quantiles() -> None:
    adapter, _ = _make_adapter(n_series=2, horizon=3, t_len=100)
    assert adapter.supports(SupportsQuantiles) is True
    ConformalizedQuantileRegression(alpha=0.1).calibrate_cv(adapter, n_windows=15)


def test_cross_validate_with_exog_uses_the_regressors() -> None:
    """CV takes no X and needs none: StatsForecast splits the training frame's exog.

    Pinned because the adapter docstring claimed the opposite. The comparison is
    against the same panel fit without the regressor column at all -- if exog
    were being dropped, the two would agree.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    x = rng.standard_normal(120)
    df = pd.DataFrame(
        {
            "unique_id": "series_0",
            "ds": dates,
            "y": 10 + np.arange(120) * 0.1 + 5 * x,
            "x1": x,
        }
    )

    def cv(frame: pd.DataFrame, exog: list[str] | None) -> np.ndarray:
        sf = StatsForecast(models=[AutoARIMA()], freq="D")
        sf.fit(frame)
        adapter = StatsForecastAdapter(
            sf=sf, train_df=frame, horizon=4, freq="D", model_name="AutoARIMA", exog_features=exog
        )
        return adapter.cross_validate(n_windows=5, step_size=1)[0]

    with_exog = cv(df, ["x1"])
    without_exog = cv(df.drop(columns=["x1"]), None)
    assert np.isfinite(with_exog).all()
    assert not np.allclose(with_exog, without_exog)


# ===========================================================================
# Custom column names
# ===========================================================================

_CUSTOM = {"id_col": "item", "time_col": "time", "target_col": "value"}


def _custom_name_adapter(
    n_series: int = 2,
    t_len: int = 100,
    horizon: int = 3,
    freq: str = "D",
) -> tuple[StatsForecastAdapter, pd.DataFrame]:
    """Adapter over an ``(item, time, value)`` frame, fit with those names."""
    df = _make_df(n_series=n_series, t_len=t_len, freq=freq, **_CUSTOM)
    sf = StatsForecast(models=[Naive()], freq=freq)
    sf.fit(df, **_CUSTOM)
    adapter = StatsForecastAdapter(
        sf=sf, train_df=df, horizon=horizon, freq=freq, model_name="Naive", **_CUSTOM
    )
    return adapter, df


class TestCustomColumnNames:
    """``id_col``/``time_col``/``target_col`` must reach the StatsForecast calls.

    ``sf.forecast`` and ``sf.cross_validation`` are stateless fit-and-predict
    calls that re-read the column names from their own kwargs every time, so
    without forwarding them the adapter built a frame with custom names and then
    handed it to a call that looked for ``unique_id``/``ds``/``y``.
    """

    def test_predict(self) -> None:
        adapter, df = _custom_name_adapter()
        panel = train_df_to_panel(df, **_CUSTOM)
        forecast = adapter.predict(panel)
        assert forecast.shape == (2, 1, 3)
        assert np.isfinite(forecast).all()

    def test_predict_batch(self) -> None:
        adapter, df = _custom_name_adapter()
        panel = train_df_to_panel(df, **_CUSTOM)
        histories = [panel[:, : panel.shape[1] - k] for k in (2, 1, 0)]
        forecast = adapter.predict_batch(histories)
        assert forecast.shape == (2, 3, 3)
        assert np.isfinite(forecast).all()

    def test_cross_validate(self) -> None:
        adapter, _ = _custom_name_adapter()
        predictions, truths = adapter.cross_validate(n_windows=5, step_size=1)
        assert predictions.shape == truths.shape == (2, 5, 3)
        assert np.isfinite(predictions).all()
        assert np.isfinite(truths).all()

    def test_predict_quantiles(self) -> None:
        df = _make_df(n_series=2, t_len=100, **_CUSTOM)
        sf = StatsForecast(models=[AutoETS(season_length=1)], freq="D")
        sf.fit(df, **_CUSTOM)
        adapter = StatsForecastAdapter(
            sf=sf, train_df=df, horizon=3, freq="D", model_name="AutoETS", **_CUSTOM
        )
        quantiles = np.array([0.05, 0.95])
        panel = train_df_to_panel(df, **_CUSTOM)
        out = adapter.predict_quantiles(panel, quantiles)
        assert out.shape == (2, 2, 3)
        assert np.isfinite(out).all()

    def test_cross_validate_quantiles(self) -> None:
        df = _make_df(n_series=2, t_len=100, **_CUSTOM)
        sf = StatsForecast(models=[AutoETS(season_length=1)], freq="D")
        sf.fit(df, **_CUSTOM)
        adapter = StatsForecastAdapter(
            sf=sf, train_df=df, horizon=3, freq="D", model_name="AutoETS", **_CUSTOM
        )
        preds, truths = adapter.cross_validate_quantiles(
            n_windows=5, step_size=1, quantiles=np.array([0.05, 0.95])
        )
        assert preds.shape == (2, 5, 3, 2)
        assert truths.shape == (2, 5, 3)
        assert np.isfinite(preds).all()

    def test_refit(self) -> None:
        adapter, df = _custom_name_adapter()
        panel = train_df_to_panel(df, **_CUSTOM)
        adapter.refit(panel)
        assert np.isfinite(adapter.predict(panel)).all()

    def test_end_to_end_split_conformal(self) -> None:
        adapter, df = _custom_name_adapter(t_len=120)
        method = SplitConformal(alpha=0.1)
        fitted = method.calibrate_cv(adapter, n_windows=20, step_size=1)
        result = fitted.predict(train_df_to_panel(df, **_CUSTOM))
        assert result.interval.shape == (2, 1, 3, 2)
        assert np.isfinite(result.interval).all()

    def test_output_matches_the_default_named_frame(self) -> None:
        """Renaming the columns must not change a single number.

        The strongest statement of the contract: the names are labels on the
        frame the adapter builds internally, nothing more.
        """
        custom_adapter, custom_df = _custom_name_adapter(t_len=120)
        default_adapter, default_df = _make_adapter(n_series=2, t_len=120, horizon=3)

        custom_panel = train_df_to_panel(custom_df, **_CUSTOM)
        default_panel = train_df_to_panel(default_df)
        np.testing.assert_array_equal(custom_panel, default_panel)

        np.testing.assert_array_equal(
            custom_adapter.predict(custom_panel), default_adapter.predict(default_panel)
        )
        custom_cv = custom_adapter.cross_validate(n_windows=5, step_size=1)
        default_cv = default_adapter.cross_validate(n_windows=5, step_size=1)
        np.testing.assert_array_equal(custom_cv[0], default_cv[0])
        np.testing.assert_array_equal(custom_cv[1], default_cv[1])
