__all__ = []


try:
    from catboost import CatBoostRegressor
except ImportError:

    class CatBoostRegressor:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            raise ImportError("Please install catboost to use this model.")


try:
    from lightgbm import LGBMRegressor
except ImportError:

    class LGBMRegressor:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            raise ImportError("Please install lightgbm to use this model.")


try:
    from xgboost import XGBRegressor
except ImportError:

    class XGBRegressor:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            raise ImportError("Please install xgboost to use this model.")


try:
    from window_ops.shift import shift_array
except ImportError:
    import numpy as np
    from utilsforecast.compat import njit

    @njit
    def shift_array(x, offset):
        if offset >= x.size or offset < 0:
            return np.full_like(x, np.nan)
        if offset == 0:
            return x.copy()
        out = np.empty_like(x)
        out[:offset] = np.nan
        out[offset:] = x[:-offset]
        return out
