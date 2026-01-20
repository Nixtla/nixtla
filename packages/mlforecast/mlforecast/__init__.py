from importlib.metadata import version

__version__ = version("mlforecast")
__all__ = ['MLForecast']
from mlforecast.forecast import MLForecast
