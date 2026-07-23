---
title: Pretraining Generators
description: TSI, TCM, and KernelSynth generators for foundation-model pretraining
---

::: synforecast.generators.tsi.TSIGenerator
    options:
      members:
        - generate_single_series

::: synforecast.generators.tcm.TCMGenerator
    options:
      members:
        - generate_single_series
        - generate

::: synforecast.generators.kernel_synth.KernelSynthGenerator
    options:
      members:
        - generate_single_series
