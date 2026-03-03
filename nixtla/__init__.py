from importlib.metadata import version
from .nixtla_client import NixtlaClient

__version__ = version("nixtla")
__all__ = ["NixtlaClient"]