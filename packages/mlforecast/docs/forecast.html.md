---
description: Full pipeline encapsulation
output-file: forecast.html
title: MLForecast
---

##

::: mlforecast.forecast.MLForecast
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - save
        - load
        - update
        - make_future_dataframe
        - get_missing_future
        - predict
        - preprocess
        - fit_models
        - cross_validation
        - cv
        - from_cv
      heading_level: 3
      show_root_heading: true
      show_source: true
