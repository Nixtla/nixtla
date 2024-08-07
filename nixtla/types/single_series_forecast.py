# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime
from ..core.pydantic_utilities import pydantic_v1
from .model import Model
from .single_series_forecast_fewshot_loss import SingleSeriesForecastFewshotLoss
from .single_series_forecast_finetune_loss import SingleSeriesForecastFinetuneLoss
from .single_series_forecast_level_item import SingleSeriesForecastLevelItem


class SingleSeriesForecast(pydantic_v1.BaseModel):
    fewshot_steps: typing.Optional[int] = None
    fewshot_loss: typing.Optional[SingleSeriesForecastFewshotLoss] = None
    model: typing.Optional[Model] = pydantic_v1.Field(default=None)
    """
    Model to use as a string. Options are: `timegpt-1`, and `timegpt-1-long-horizon.` We recommend using `timegpt-1-long-horizon` for forecasting if you want to predict more than one seasonal period given the frequency of your data.
    """

    freq: typing.Optional[str] = pydantic_v1.Field(default=None)
    """
    The frequency of the data represented as a string. 'D' for daily, 'M' for monthly, 'H' for hourly, and 'W' for weekly frequencies are available.
    """

    level: typing.Optional[typing.List[SingleSeriesForecastLevelItem]] = None
    fh: typing.Optional[int] = pydantic_v1.Field(default=None)
    """
    The forecasting horizon. This represents the number of time steps into the future that the forecast should predict.
    """

    y: typing.Optional[typing.Any] = None
    x: typing.Optional[typing.Dict[str, typing.Optional[typing.List[float]]]] = None
    clean_ex_first: typing.Optional[bool] = pydantic_v1.Field(default=None)
    """
    A boolean flag that indicates whether the API should preprocess (clean) the exogenous signal before applying the large time model. If True, the exogenous signal is cleaned; if False, the exogenous variables are applied after the large time model.
    """

    finetune_steps: typing.Optional[int] = pydantic_v1.Field(default=None)
    """
    The number of tuning steps used to train the large time model on the data. Set this value to 0 for zero-shot inference, i.e., to make predictions without any further model tuning.
    """

    finetune_loss: typing.Optional[SingleSeriesForecastFinetuneLoss] = pydantic_v1.Field(default=None)
    """
    The loss used to train the large time model on the data. Select from ['default', 'mae', 'mse', 'rmse', 'mape', 'smape']. It will only be used if finetune_steps larger than 0. Default is a robust loss function that is less sensitive to outliers.
    """

    def json(self, **kwargs: typing.Any) -> str:
        kwargs_with_defaults: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        return super().json(**kwargs_with_defaults)

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        kwargs_with_defaults: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        return super().dict(**kwargs_with_defaults)

    class Config:
        frozen = True
        smart_union = True
        extra = pydantic_v1.Extra.allow
        json_encoders = {dt.datetime: serialize_datetime}
