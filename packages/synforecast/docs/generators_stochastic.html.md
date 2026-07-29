---
title: Stochastic Generators
description: GARCH, Ornstein-Uhlenbeck, GBM, Jump Diffusion, Poisson, Cyclic, fBm, Hawkes, Stochastic Volatility, Regime Switching, Chaotic System, Bounded Process, and Lévy Process generators
---

::: synforecast.generators.garch.GARCHGenerator
    options:
      members:
        - generate_single_series

::: synforecast.generators.ornstein_uhlenbeck.OrnsteinUhlenbeckGenerator
    options:
      members:
        - generate_single_series

::: synforecast.generators.geometric_brownian_motion.GeometricBrownianMotionGenerator
    options:
      members:
        - generate_single_series

::: synforecast.generators.jump_diffusion.JumpDiffusionGenerator
    options:
      members:
        - generate_single_series

::: synforecast.generators.poisson_process.PoissonProcessGenerator
    options:
      members:
        - generate_single_series

::: synforecast.generators.cyclic.CyclicGenerator
    options:
      members:
        - generate_single_series

::: synforecast.generators.fractional_brownian_motion.FractionalBrownianMotionGenerator
    options:
      members:
        - generate_single_series
        - get_model_info
        - estimate_hurst

::: synforecast.generators.hawkes_process.HawkesProcessGenerator
    options:
      members:
        - generate_single_series
        - simulate_with_events
        - get_model_info
        - estimate_parameters

::: synforecast.generators.stochastic_volatility.StochasticVolatilityGenerator
    options:
      members:
        - generate_single_series
        - generate_with_volatility
        - get_model_info
        - implied_volatility_smile

::: synforecast.generators.regime_switching.RegimeSwitchingGenerator
    options:
      members:
        - generate_single_series
        - generate_with_regimes
        - get_model_info

::: synforecast.generators.chaotic_system.ChaoticSystemGenerator
    options:
      members:
        - generate_single_series
        - get_model_info

::: synforecast.generators.bounded_process.BoundedProcessGenerator
    options:
      members:
        - generate_single_series
        - get_model_info

::: synforecast.generators.levy_process.LevyProcessGenerator
    options:
      members:
        - generate_single_series
        - get_model_info
