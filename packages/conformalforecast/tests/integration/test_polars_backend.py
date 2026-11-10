"""pandas and polars frames give the same numbers, and polars never meets pandas.

The pandas paths are pinned in depth by the per-adapter suites; this file proves
the polars path matches them, and pins the frame-type boundary.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")
pytest.importorskip("statsforecast")

from statsforecast import StatsForecast  # noqa: E402
from statsforecast.models import AutoARIMA, Naive  # noqa: E402

from conformalforecast.adapters.statsforecast import StatsForecastAdapter  # noqa: E402
from conformalforecast.methods import SplitConformal  # noqa: E402

FREQ = {"pandas": "D", "polars": "1d"}
Q = np.array([0.1, 0.9])


def _data(*, exog: bool = False, int_ds: bool = False, seed: int = 0) -> dict[str, list[Any]]:
    """Ragged panel: 'b' starts ten steps after 'a' and ends five steps before it.

    ``int_ds`` makes ``ds`` an integer step instead of a daily timestamp.
    """
    rng = np.random.default_rng(seed)
    if int_ds:
        a: list[Any] = list(range(60))
        b: list[Any] = list(range(10, 55))
    else:
        a = [dt.datetime(2020, 1, 1) + dt.timedelta(days=i) for i in range(60)]
        b = [dt.datetime(2020, 1, 11) + dt.timedelta(days=i) for i in range(45)]
    data: dict[str, list[Any]] = {
        "unique_id": ["a"] * len(a) + ["b"] * len(b),
        "ds": a + b,
        "y": rng.standard_normal(len(a) + len(b)).cumsum().tolist(),
    }
    if exog:
        data["x"] = rng.standard_normal(len(a) + len(b)).tolist()
    return data


def _native(backend: str, data: dict[str, list[Any]]) -> Any:
    return pd.DataFrame(data) if backend == "pandas" else pl.DataFrame(data)


def _spy(obj: Any, name: str, seen: list[str]) -> None:
    """Record the top-level module of every frame passed to ``obj.name``."""
    original = getattr(obj, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        seen.extend(
            type(v).__module__.split(".")[0]
            for v in (*args, *kwargs.values())
            if hasattr(v, "columns")
        )
        return original(*args, **kwargs)

    setattr(obj, name, wrapper)


def _sf_adapter(backend: str, *, exog: bool = False, int_ds: bool = False) -> StatsForecastAdapter:
    df = _native(backend, _data(exog=exog, int_ds=int_ds))
    model = AutoARIMA(season_length=1) if exog else Naive()
    freq = 1 if int_ds else FREQ[backend]
    sf = StatsForecast(models=[model], freq=freq)
    sf.fit(df)
    return StatsForecastAdapter(
        sf=sf,
        train_df=df,
        horizon=3,
        freq=freq,
        model_name=repr(model),
        exog_features=["x"] if exog else None,
    )


@pytest.fixture(scope="module")
def sf_pair() -> dict[str, StatsForecastAdapter]:
    return {b: _sf_adapter(b) for b in ("pandas", "polars")}


class TestStatsForecastBackends:
    def test_train_df_is_kept_in_the_users_backend(self, sf_pair) -> None:
        for backend, adapter in sf_pair.items():
            assert type(adapter.train_df).__module__.split(".")[0] == backend

    @pytest.mark.parametrize("cut", ["full", "prefix", "recent"])
    def test_predict_matches_pandas(self, sf_pair, cut: str) -> None:
        full = sf_pair["pandas"]._train_panel  # the right-aligned training panel
        history = {"full": full, "prefix": full[:, :-5], "recent": full[:, -20:]}[cut]
        got = {b: a.predict(history) for b, a in sf_pair.items()}
        np.testing.assert_allclose(got["polars"], got["pandas"], rtol=1e-10)

    @pytest.mark.parametrize("aligned", [False, True])
    def test_cross_validate_matches_pandas(self, sf_pair, aligned: bool) -> None:
        got = {b: a.cross_validate(4, 1, aligned=aligned) for b, a in sf_pair.items()}
        for p, q in zip(got["polars"], got["pandas"], strict=True):
            np.testing.assert_allclose(p, q, rtol=1e-10, equal_nan=True)

    def test_quantiles_match_pandas(self, sf_pair) -> None:
        history = sf_pair["pandas"]._train_panel
        np.testing.assert_allclose(
            sf_pair["polars"].predict_quantiles(history, Q),
            sf_pair["pandas"].predict_quantiles(history, Q),
            rtol=1e-10,
        )
        for p, q in zip(
            sf_pair["polars"].cross_validate_quantiles(3, 1, Q),
            sf_pair["pandas"].cross_validate_quantiles(3, 1, Q),
            strict=True,
        ):
            np.testing.assert_allclose(p, q, rtol=1e-10)

    def test_split_conformal_end_to_end_matches_pandas(self, sf_pair) -> None:
        history = sf_pair["pandas"]._train_panel
        got = {
            b: SplitConformal(alpha=0.2).calibrate_cv(a, n_windows=5).predict(history)
            for b, a in sf_pair.items()
        }
        np.testing.assert_allclose(got["polars"].interval, got["pandas"].interval, rtol=1e-10)

    def test_pyarrow_backed_pandas_matches_numpy_backed(self, sf_pair) -> None:
        pytest.importorskip("pyarrow")
        df = _native("pandas", _data()).convert_dtypes(dtype_backend="pyarrow")
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.fit(df)
        adapter = StatsForecastAdapter(sf=sf, train_df=df, horizon=3, freq="D", model_name="Naive")
        history = sf_pair["pandas"]._train_panel[:, :-5]
        np.testing.assert_allclose(
            adapter.predict(history), sf_pair["pandas"].predict(history), rtol=1e-10
        )

    def test_refit_matches_pandas(self) -> None:
        pair = {b: _sf_adapter(b) for b in ("pandas", "polars")}
        shorter = pair["pandas"]._train_panel[:, :-4]
        for adapter in pair.values():
            adapter.refit(shorter)
        assert type(pair["polars"].train_df).__module__.split(".")[0] == "polars"
        np.testing.assert_allclose(
            pair["polars"].predict(shorter), pair["pandas"].predict(shorter), rtol=1e-10
        )

    def test_integer_ds_matches_pandas(self) -> None:
        pair = {b: _sf_adapter(b, int_ds=True) for b in ("pandas", "polars")}
        history = pair["pandas"]._train_panel[:, :-5]
        np.testing.assert_allclose(
            pair["polars"].predict(history), pair["pandas"].predict(history), rtol=1e-10
        )
        fitted = {
            b: SplitConformal(alpha=0.2).calibrate_cv(a, n_windows=5) for b, a in pair.items()
        }
        np.testing.assert_allclose(
            fitted["polars"].predict(history).interval,
            fitted["pandas"].predict(history).interval,
            rtol=1e-10,
        )

    @pytest.mark.parametrize("refit", [True, 2])
    def test_calibrate_cv_refit_keeps_polars_and_matches_pandas(self, refit: bool | int) -> None:
        pair = {b: _sf_adapter(b) for b in ("pandas", "polars")}
        fitted = {
            b: SplitConformal(alpha=0.2).calibrate_cv(a, n_windows=5, refit=refit)
            for b, a in pair.items()
        }
        assert type(fitted["polars"].forecaster_.train_df).__module__.split(".")[0] == "polars"
        history = pair["pandas"]._train_panel
        np.testing.assert_allclose(
            fitted["polars"].predict(history).interval,
            fitted["pandas"].predict(history).interval,
            rtol=1e-10,
        )

    def test_exog_predict_matches_pandas(self) -> None:
        pair = {b: _sf_adapter(b, exog=True) for b in ("pandas", "polars")}
        history = pair["pandas"]._train_panel[:, :-5]
        X = np.random.default_rng(1).standard_normal((2, 1, 3, 1))
        np.testing.assert_allclose(
            pair["polars"].predict(history, X=X),
            pair["pandas"].predict(history, X=X),
            rtol=1e-8,
        )

    def test_polars_frames_reach_the_library(self) -> None:
        adapter = _sf_adapter("polars")
        seen: list[str] = []
        for name in ("forecast", "cross_validation", "fit"):
            _spy(adapter.sf, name, seen)
        adapter.predict(adapter._train_panel)
        adapter.cross_validate(2, 1)
        adapter.refit(adapter._train_panel[:, :-2])
        assert seen and set(seen) == {"polars"}

    def test_wrong_dialect_freq_is_rejected_at_construction(self) -> None:
        df = _native("polars", _data())
        sf = StatsForecast(models=[Naive()], freq="1d")
        sf.fit(df)
        with pytest.raises(ValueError, match=r"Invalid frequency 'D'.*polars interval"):
            StatsForecastAdapter(sf=sf, train_df=df, horizon=3, freq="D", model_name="Naive")

    def test_day_30_monthly_series_crossing_feb_29(self) -> None:
        # polars anchors '1mo' at the start (Jan 30, Feb 29, Mar 30, ...), and
        # Apr 30 is a month end, which utilsforecast would snap when stepping.
        ds = pl.datetime_range(
            pl.datetime(2023, 9, 30), pl.datetime(2024, 8, 30), "1mo", eager=True
        )
        df = pl.DataFrame({"unique_id": "a", "ds": ds, "y": np.arange(len(ds), dtype=float)})
        sf = StatsForecast(models=[Naive()], freq="1mo")
        sf.fit(df)
        adapter = StatsForecastAdapter(
            sf=sf, train_df=df, horizon=2, freq="1mo", model_name="Naive"
        )
        full = adapter._train_panel
        np.testing.assert_array_equal(adapter.predict(full)[0, 0], [11.0, 11.0])
        np.testing.assert_array_equal(adapter.predict(full[:, :-4])[0, 0], [7.0, 7.0])
        history = adapter._panel_to_df(full[:, :-4])
        assert history["ds"].to_list() == ds.to_list()[:-4]

    @pytest.mark.parametrize("bad", ["not a frame", 3])
    def test_non_frames_are_rejected(self, bad: object) -> None:
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.fit(_native("pandas", _data()))
        with pytest.raises(ValueError, match="train_df must be a pandas or polars DataFrame"):
            StatsForecastAdapter(sf=sf, train_df=bad, horizon=3, freq="D", model_name="Naive")


mlforecast = pytest.importorskip("mlforecast")
from mlforecast import MLForecast  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from conformalforecast.adapters.mlforecast import MLForecastAdapter  # noqa: E402


def _mlf_adapter(backend: str, *, exog: bool = False, int_ds: bool = False) -> MLForecastAdapter:
    df = _native(backend, _data(exog=exog, int_ds=int_ds))
    freq = 1 if int_ds else FREQ[backend]
    mlf = MLForecast(models={"Ridge": Ridge()}, freq=freq, lags=[1, 2, 3])
    mlf.fit(df, static_features=[])
    return MLForecastAdapter(
        mlf=mlf,
        train_df=df,
        horizon=3,
        freq=freq,
        model_name="Ridge",
        exog_features=["x"] if exog else None,
    )


@pytest.fixture(scope="module")
def mlf_pair() -> dict[str, MLForecastAdapter]:
    return {b: _mlf_adapter(b) for b in ("pandas", "polars")}


class TestMLForecastBackends:
    def test_train_df_is_kept_in_the_users_backend(
        self, mlf_pair: dict[str, MLForecastAdapter]
    ) -> None:
        for backend, adapter in mlf_pair.items():
            assert type(adapter.train_df).__module__.split(".")[0] == backend

    @pytest.mark.parametrize("cut", ["full", "prefix", "recent"])
    def test_predict_matches_pandas(self, mlf_pair: dict[str, MLForecastAdapter], cut: str) -> None:
        full = mlf_pair["pandas"]._train_panel
        history = {"full": full, "prefix": full[:, :-5], "recent": full[:, -20:]}[cut]
        np.testing.assert_allclose(
            mlf_pair["polars"].predict(history), mlf_pair["pandas"].predict(history), rtol=1e-10
        )

    @pytest.mark.parametrize("aligned", [False, True])
    def test_cross_validate_matches_pandas(
        self, mlf_pair: dict[str, MLForecastAdapter], aligned: bool
    ) -> None:
        for p, q in zip(
            mlf_pair["polars"].cross_validate(3, 1, aligned=aligned),
            mlf_pair["pandas"].cross_validate(3, 1, aligned=aligned),
            strict=True,
        ):
            np.testing.assert_allclose(p, q, rtol=1e-10, equal_nan=True)

    def test_refit_matches_pandas(self) -> None:
        pair = {b: _mlf_adapter(b) for b in ("pandas", "polars")}
        shorter = pair["pandas"]._train_panel[:, :-4]
        for adapter in pair.values():
            adapter.refit(shorter)
        np.testing.assert_allclose(
            pair["polars"].predict(shorter), pair["pandas"].predict(shorter), rtol=1e-10
        )

    def test_exog_predict_matches_pandas(self) -> None:
        pair = {b: _mlf_adapter(b, exog=True) for b in ("pandas", "polars")}
        history = pair["pandas"]._train_panel[:, :-5]
        X = np.random.default_rng(1).standard_normal((2, 1, 3, 1))
        np.testing.assert_allclose(
            pair["polars"].predict(history, X=X),
            pair["pandas"].predict(history, X=X),
            rtol=1e-10,
        )

    def test_polars_frames_reach_the_library(self) -> None:
        adapter = _mlf_adapter("polars")
        seen: list[str] = []
        for name in ("predict", "cross_validation", "fit"):
            _spy(adapter.mlf, name, seen)
        adapter.predict(adapter._train_panel)
        adapter.cross_validate(2, 1)
        adapter.refit(adapter._train_panel[:, :-2])
        assert seen and set(seen) == {"polars"}

    def test_integer_ds_matches_pandas(self) -> None:
        pair = {b: _mlf_adapter(b, int_ds=True) for b in ("pandas", "polars")}
        history = pair["pandas"]._train_panel[:, :-5]
        np.testing.assert_allclose(
            pair["polars"].predict(history), pair["pandas"].predict(history), rtol=1e-10
        )
        fitted = {
            b: SplitConformal(alpha=0.2).calibrate_cv(a, n_windows=5) for b, a in pair.items()
        }
        np.testing.assert_allclose(
            fitted["polars"].predict(history).interval,
            fitted["pandas"].predict(history).interval,
            rtol=1e-10,
        )

    @pytest.mark.parametrize("refit", [True, 2])
    def test_calibrate_cv_refit_keeps_polars_and_matches_pandas(self, refit: bool | int) -> None:
        pair = {b: _mlf_adapter(b) for b in ("pandas", "polars")}
        fitted = {
            b: SplitConformal(alpha=0.2).calibrate_cv(a, n_windows=5, refit=refit)
            for b, a in pair.items()
        }
        assert type(fitted["polars"].forecaster_.train_df).__module__.split(".")[0] == "polars"
        history = pair["pandas"]._train_panel
        np.testing.assert_allclose(
            fitted["polars"].predict(history).interval,
            fitted["pandas"].predict(history).interval,
            rtol=1e-10,
        )
