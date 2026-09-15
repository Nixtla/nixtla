"""Type aliases, validators and endpoint constants shared across the package.

This module sits below every other one and imports nothing from the package, so
`nixtla_client`, `jobs` and `_preprocessing` can all take their vocabulary from
here rather than from each other.
"""

import datetime
from typing import TYPE_CHECKING, Annotated, Any, Dict, Literal, Optional, TypeVar, Union

import annotated_types
import pandas as pd
from pydantic import AfterValidator, BaseModel, TypeAdapter

if TYPE_CHECKING:
    try:
        from polars import DataFrame as PolarsDataFrame
    except ModuleNotFoundError:
        pass
    try:
        from dask.dataframe import DataFrame as DaskDataFrame
    except ModuleNotFoundError:
        pass
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
    except ModuleNotFoundError:
        pass
    try:
        from ray.data import Dataset as RayDataset
    except ModuleNotFoundError:
        pass

AnyDFType = TypeVar(
    "AnyDFType",
    "DaskDataFrame",
    pd.DataFrame,
    "PolarsDataFrame",
    "RayDataset",
    "SparkDataFrame",
)
DistributedDFType = TypeVar(
    "DistributedDFType",
    "DaskDataFrame",
    "RayDataset",
    "SparkDataFrame",
)


def validate_extra_params(value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate that the dictionary doesn't contain complex structures."""
    primitives = (str, int, float, bool, type(None))
    if value is None:
        return value

    for _, v in value.items():
        if isinstance(v, dict):
            for _, nv in v.items():
                # nested structure allowed but they can support primitive values only
                if not isinstance(nv, primitives):
                    raise TypeError(f"Invalid value type: {type(nv).__name__}")
        elif isinstance(v, (dict, list, tuple, set)):
            for nv in v:
                if not isinstance(nv, primitives):
                    raise TypeError(f"Invalid value type: {type(nv).__name__}")
        elif not isinstance(v, primitives):
            raise TypeError(f"Invalid value type: {type(v).__name__}")
    return value


_PositiveInt = Annotated[int, annotated_types.Gt(0)]
_NonNegativeInt = Annotated[int, annotated_types.Ge(0)]
_ExtraParamDataType = Annotated[
    Optional[Dict[str, Any]], AfterValidator(validate_extra_params)
]
extra_param_checker = TypeAdapter(_ExtraParamDataType)
_Loss = Literal["default", "mae", "mse", "rmse", "mape", "smape", "poisson"]
_Model = str
_FinetuneDepth = Literal[1, 2, 3, 4, 5]
_Freq = Union[str, int, pd.offsets.BaseOffset]
_FreqType = TypeVar("_FreqType", str, int, pd.offsets.BaseOffset)
_ThresholdMethod = Literal["univariate", "multivariate"]
_ExplainMethod = Literal["granger", "transfer_entropy"]
_FeatureContributionsType = Literal[
    "shapley", "intervention", "granger", "transfer_entropy"
]
# Only used to derive distinct in-range per-partition seeds; the seed's range
# itself is validated server-side.
_MIN_SEED = -(2**63)
_MAX_SEED = 2**64 - 1


class FinetunedModel(BaseModel, extra="allow"):  # type: ignore
    id: str
    created_at: datetime.datetime
    created_by: str
    base_model_id: str
    steps: int
    depth: int
    loss: _Loss
    model: _Model
    freq: str


# Client concurrency limit for all partitioned async requests. This bounds
# submission pressure; it is not a guarantee of the deployment's team job cap.
_MAX_CONCURRENT_ASYNC_JOBS = 5

# The server is mid-rename: the async route already uses the post-rename name
_ANOMALY_DETECTION_ENDPOINT = "v2/anomaly_detection"
_ONLINE_ANOMALY_DETECTION_ENDPOINT = "v2/online_anomaly_detection"
