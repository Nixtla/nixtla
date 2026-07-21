"""SynForecast: Synthetic Time Series Generation."""

from importlib.metadata import version

__version__: str = version(distribution_name="synforecast")

from synforecast.base import BaseGenerator
from synforecast.dataset import SynAugment, SynSet
from synforecast.multivariatize import Multivariatizer
from synforecast.presets import balanced_pool
from synforecast.utils import generate_series

__all__ = [
    "BaseGenerator",
    "Multivariatizer",
    "SynAugment",
    "SynSet",
    "balanced_pool",
    "generate_series",
]
