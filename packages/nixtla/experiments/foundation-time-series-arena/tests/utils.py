from xiuhmolpilli.models.benchmarks import (
    AutoARIMA,
    NixtlaProphet,
    SeasonalNaive,
    AutoNHITS,
    AutoTFT,
    AutoLGBM,
)
from xiuhmolpilli.models.foundational import Chronos, LagLlama, Moirai, TimeGPT, TimesFM

models = [
    # benchmarks
    AutoARIMA(),
    NixtlaProphet(),
    SeasonalNaive(),
    # neural benchmarks
    AutoNHITS(),
    AutoTFT(),
    # ml
    AutoLGBM(),
    # foundational models
    Chronos("amazon/chronos-t5-tiny"),
    LagLlama(),
    Moirai("Salesforce/moirai-1.0-R-small"),
    TimeGPT(),
    TimesFM(),
]
