---
title: Nonconformity scores
description: Score functions used to turn predictions and truths into calibration scores
---

A score is method config: pass it as `score=` when constructing a method, or
leave it out to take the method's default. Each concrete score carries a `kind`
tag (`"absolute"`, `"signed"`, `"quantile"`), which is what lets a configured
method survive `model_validate(model_dump())`.

A custom score subclasses `ScoreFunction` (or one of the built-ins) and declares
its own tag, e.g. `kind: Literal["scaled"] = "scaled"`. Subclassing a built-in
without declaring a new tag raises `TypeError` when the class is defined,
because the score would otherwise reload as its parent. To reload a dumped
method holding a custom score, import the module that defines the score first.

::: conformalforecast.nonconformity.AbsoluteResidual
    options:
      filters:
        - "!^_"
        - "!^model_config$"

::: conformalforecast.nonconformity.SignedResidual
    options:
      filters:
        - "!^_"
        - "!^model_config$"

::: conformalforecast.nonconformity.QuantileScore
    options:
      filters:
        - "!^_"
        - "!^model_config$"

::: conformalforecast.nonconformity.ScoreField
