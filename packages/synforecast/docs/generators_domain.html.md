---
title: Domain-Specific Generators
description: IntermittentDemand, IoTSensor, EnergyLoad, StateSpace, DailyActiveUsers, VitalSigns, and Clickstream generators
---

::: synforecast.generators.intermittent_demand.IntermittentDemandGenerator
    options:
      members:
        - generate_single_series

::: synforecast.generators.iot_sensor.IoTSensorGenerator
    options:
      members:
        - generate_single_series
        - generate

::: synforecast.generators.energy_load.EnergyLoadGenerator
    options:
      members:
        - generate_single_series

::: synforecast.generators.state_space.StateSpaceGenerator
    options:
      members:
        - generate_single_series
        - generate_with_states

::: synforecast.generators.daily_active_users.DailyActiveUsersGenerator
    options:
      members:
        - generate_single_series
        - generate

::: synforecast.generators.vital_signs.VitalSignsGenerator
    options:
      members:
        - generate_single_series
        - generate_all_vitals
        - get_model_info

::: synforecast.generators.clickstream.ClickstreamGenerator
    options:
      members:
        - generate_single_series
        - generate_full_metrics
        - generate_funnel
        - get_model_info
