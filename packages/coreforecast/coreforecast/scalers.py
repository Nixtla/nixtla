import numpy as np

from .grouped_array import GroupedArray


class BaseLocalScaler:
    scaler_type: str

    def fit(self, ga: GroupedArray) -> "BaseLocalScaler":
        self.stats_ = ga.scaler_fit(self.scaler_type)
        return self

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga.scaler_transform(self.stats_)

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        return ga.scaler_inverse_transform(self.stats_)


class LocalMinMaxScaler(BaseLocalScaler):
    scaler_type = "MinMax"


class LocalStandardScaler(BaseLocalScaler):
    scaler_type = "Standard"


class LocalRobustScaler(BaseLocalScaler):
    def __init__(self, scale: str):
        if scale == "iqr":
            self.scaler_type = "RobustIqr"
        else:
            self.scaler_type = "RobustMad"
