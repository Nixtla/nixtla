"""Lazy adapter re-exports: names resolve, but heavy libraries stay unimported."""

from __future__ import annotations

import subprocess
import sys

import pytest

from conformalforecast import adapters


def test_dir_advertises_all_adapters() -> None:
    assert set(adapters.__all__) == {
        "CallableAdapter",
        "MLForecastAdapter",
        "NeuralForecastAdapter",
        "StatsForecastAdapter",
    }
    assert set(adapters.__all__) <= set(dir(adapters))


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="DartsAdapter"):
        adapters.DartsAdapter  # noqa: B018


@pytest.mark.parametrize(
    "name", ["StatsForecastAdapter", "MLForecastAdapter", "NeuralForecastAdapter"]
)
def test_lazy_names_resolve_or_raise_install_hint(name: str) -> None:
    """With the extra installed the class resolves; without it, the hint is shown."""
    try:
        obj = getattr(adapters, name)
    except ImportError as exc:  # extra not installed
        assert "conformalforecast[nixtla]" in str(exc)
    else:
        assert obj.__name__ == name


def test_importing_package_does_not_import_optional_libraries() -> None:
    """Run in a subprocess: this session may already have the libraries loaded.

    Covers every optional extra, not just ``[nixtla]``: ``quantile-forest`` and
    its scikit-learn dependency belong to ``[spci]`` and are reached only
    through :class:`~conformalforecast.QRFQuantileRegressor`'s constructor.
    """
    code = (
        "import sys; import conformalforecast; "
        "heavy = [m for m in ('statsforecast', 'mlforecast', 'neuralforecast', 'torch', "
        "'sklearn', 'quantile_forest', 'matplotlib', 'pandas', 'polars') "
        "if m in sys.modules]; print(heavy)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"eagerly imported: {out.stdout.strip()}"


def test_qrf_regressor_name_resolves_without_importing_backend() -> None:
    """The class is eagerly importable and constructible; the backend loads at fit()."""
    code = (
        "import sys; from conformalforecast import QRFQuantileRegressor; "
        "QRFQuantileRegressor(); "
        "assert QRFQuantileRegressor.__name__ == 'QRFQuantileRegressor'; "
        "print('quantile_forest' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
