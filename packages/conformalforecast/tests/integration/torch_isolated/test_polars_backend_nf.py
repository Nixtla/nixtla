"""NeuralForecast: pandas and polars frames give the same numbers."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.torch

pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")
nf_mod = pytest.importorskip("neuralforecast")

from neuralforecast import NeuralForecast  # noqa: E402
from neuralforecast.losses.pytorch import MAE, MQLoss  # noqa: E402
from neuralforecast.models import MLP  # noqa: E402

from conformalforecast.adapters.neuralforecast import NeuralForecastAdapter  # noqa: E402

FREQ = {"pandas": "D", "polars": "1d"}
QUIET: dict[str, Any] = {
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "logger": False,
    "accelerator": "cpu",
}
# Two separate trainings with one seed: identical on CPU in practice; the
# tolerance only absorbs last-bit differences in float32 accumulation.
RTOL = 1e-5


def _data() -> dict[str, list[Any]]:
    rng = np.random.default_rng(0)
    a = [dt.datetime(2020, 1, 1) + dt.timedelta(days=i) for i in range(60)]
    b = [dt.datetime(2020, 1, 11) + dt.timedelta(days=i) for i in range(45)]
    return {
        "unique_id": ["a"] * len(a) + ["b"] * len(b),
        "ds": a + b,
        "y": rng.standard_normal(len(a) + len(b)).cumsum().tolist(),
    }


def _adapter(backend: str, loss: Any = None) -> NeuralForecastAdapter:
    df = pd.DataFrame(_data()) if backend == "pandas" else pl.DataFrame(_data())
    model = MLP(h=3, input_size=12, max_steps=10, loss=loss or MAE(), random_seed=0, **QUIET)
    nf = NeuralForecast(models=[model], freq=FREQ[backend])
    nf.fit(df)
    return NeuralForecastAdapter(nf=nf, train_df=df, freq=FREQ[backend], model_name="MLP")


@pytest.fixture(scope="module")
def pair() -> dict[str, NeuralForecastAdapter]:
    return {b: _adapter(b) for b in ("pandas", "polars")}


@pytest.mark.parametrize("cut", ["full", "recent"])
def test_predict_matches_pandas(pair: dict[str, NeuralForecastAdapter], cut: str) -> None:
    full = pair["pandas"]._train_panel
    history = full if cut == "full" else full[:, -20:]
    np.testing.assert_allclose(
        pair["polars"].predict(history), pair["pandas"].predict(history), rtol=RTOL
    )


def test_cross_validate_matches_pandas(pair: dict[str, NeuralForecastAdapter]) -> None:
    polars_out = pair["polars"].cross_validate(2, 1)
    pandas_out = pair["pandas"].cross_validate(2, 1)
    for p, q in zip(polars_out, pandas_out, strict=True):
        np.testing.assert_allclose(p, q, rtol=RTOL, equal_nan=True)


def test_quantiles_match_pandas() -> None:
    q = np.array([0.1, 0.9])
    quantile_pair = {b: _adapter(b, loss=MQLoss(quantiles=[0.1, 0.9])) for b in FREQ}
    history = quantile_pair["pandas"]._train_panel
    np.testing.assert_allclose(
        quantile_pair["polars"].predict_quantiles(history, q),
        quantile_pair["pandas"].predict_quantiles(history, q),
        rtol=RTOL,
    )


def test_train_df_stays_polars_after_refit() -> None:
    adapter = _adapter("polars")
    adapter.refit(adapter._train_panel[:, :-2])
    assert type(adapter.train_df).__module__.split(".")[0] == "polars"
