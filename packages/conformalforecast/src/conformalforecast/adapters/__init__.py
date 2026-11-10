"""Forecaster adapters.

:class:`~conformalforecast.adapters.callable.CallableAdapter` is imported
eagerly — it is pure numpy. The library-backed adapters are re-exported
**lazily** via PEP 562 ``__getattr__`` so that ``import conformalforecast``
never pulls in statsforecast, mlforecast, neuralforecast (or torch). A missing
extra surfaces the adapter module's own ``ImportError``, which already carries
the exact ``pip install conformalforecast[nixtla]`` command.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .callable import CallableAdapter

if TYPE_CHECKING:  # static analysers see the real classes
    from .mlforecast import MLForecastAdapter
    from .neuralforecast import NeuralForecastAdapter
    from .statsforecast import StatsForecastAdapter

# Public name -> defining submodule (relative).
_LAZY_ADAPTERS: dict[str, str] = {
    "StatsForecastAdapter": ".statsforecast",
    "MLForecastAdapter": ".mlforecast",
    "NeuralForecastAdapter": ".neuralforecast",
}

__all__ = [
    "CallableAdapter",
    "MLForecastAdapter",
    "NeuralForecastAdapter",
    "StatsForecastAdapter",
]


def __getattr__(name: str) -> Any:
    """
    Resolve a lazily re-exported adapter class.

    Parameters
    ----------
    name : str
        Attribute name being looked up on this module.

    Returns
    -------
    Any
        The adapter class.

    Raises
    ------
    AttributeError
        If ``name`` is not a known adapter.
    ImportError
        Propagated from the adapter module when its extra is not installed.
    """
    module_name = _LAZY_ADAPTERS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
