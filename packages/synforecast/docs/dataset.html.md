---
title: Dataset
description: Dataset composition and augmentation
---

::: synforecast.dataset.SynSet
    options:
      members:
        - generate

::: synforecast.augment.BaseAugmenter
    options:
      members:
        - apply
        - model_copy

::: synforecast.augment.FitAndSimulate
    options:
      members:
        - apply
        - apply_series

::: synforecast.augment.Mixup
    options:
      members:
        - apply

::: synforecast.augment.MovingBlockBootstrap
    options:
      members:
        - apply

::: synforecast.augment.DBA
    options:
      members:
        - apply

::: synforecast.augment.analyze

::: synforecast.dataset.SynAugment
    options:
      members:
        - analyze
        - augment
        - augment_single_series
        - mixup
        - mbb
        - dba
