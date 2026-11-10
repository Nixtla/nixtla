---
title: Capabilities
description: Optional adapter features, declared as mixins and checked when a forecaster is bound
---

::: conformalforecast.capabilities.SupportsRefit
    options:
      filters:
        - "!^_"
        - "!^model_config$"

::: conformalforecast.capabilities.SupportsQuantiles
    options:
      filters:
        - "!^_"
        - "!^model_config$"

::: conformalforecast.capabilities.SupportsBootstrap
    options:
      filters:
        - "!^_"
        - "!^model_config$"

::: conformalforecast.capabilities.SupportsCrossValidation
    options:
      filters:
        - "!^_"
        - "!^model_config$"

::: conformalforecast.capabilities.SupportsCrossValidationQuantiles
    options:
      filters:
        - "!^_"
        - "!^model_config$"
