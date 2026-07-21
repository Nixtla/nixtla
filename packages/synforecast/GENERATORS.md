# SynForecast Generators

Reference for all synthetic time series generators in SynForecast, organized by category.

## Overview

SynForecast provides **31 generators** organized into five categories:
- **Statistical** (5): Classical time series models
- **Stochastic** (13): Stochastic process-based generators
- **Multivariate** (3): Multi-dimensional time series
- **Domain-Specific** (7): Industry/application-focused generators
- **Pretraining** (3): Diversity-targeted generators for foundation-model corpora

All generators share the same constructor and `generate()` interface. Constructors are keyword-only:

```python
gen = RandomWalkGenerator(min_length=50, max_length=100, freq="D", seed=42)
df = gen.generate(n_series=10)
```

`generate(n_series, start_id=0, n_jobs=-1)` returns a long-format DataFrame with columns `[unique_id, ds, y]`:
- `unique_id`: integer categorical series identifier
- `ds`: `datetime64[ns]` timestamp (or `int64` when `freq` is an integer)
- `y`: `float64` value

The output is a pandas DataFrame by default; set `engine` to change the output library.

---

## Common Parameters

Defined on `BaseGenerator` (`synforecast/base.py`) and accepted by every generator.

### Core

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_length` | required | Minimum length of each series (> 0) |
| `max_length` | required | Maximum length of each series (>= `min_length`) |
| `freq` | required | Pandas offset alias (`'D'`, `'h'`, `'5min'`, `'MS'`, `'W-MON'`, ...) or an integer for an integer time index |
| `engine` | `"pandas"` | Output dataframe library: `pandas`, `polars`, `cudf`, `modin`, `pyarrow` |
| `alias` | `None` | Name of the generator (default: class name) |
| `id_col` | `"unique_id"` | Name of the ID column |
| `time_col` | `"ds"` | Name of the timestamp column |
| `target_col` | `"y"` | Name of the value column |
| `start_datetime` | `"2000-01-01"` | First timestamp of every series (any format accepted by `pandas.Timestamp`); ignored when `freq` is an integer |
| `seed` | `None` | Random seed for reproducibility |
| `innovation_distribution` | `"normal"` | Innovation/noise distribution: `normal`, `t`, `laplace`, `uniform`, `skew_normal` |
| `innovation_params` | `None` | Distribution parameters, e.g. `{"df": 5}` for `t` (df > 2), `{"alpha": 5}` for `skew_normal` |
| `exogenous` | `None` | `ExogenousConfig` for exogenous variable generation (pattern flags, datetime features, correlated columns) |

### Missing data injection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `missing_data` | `False` | Enable missing data patterns |
| `missing_pattern` | `"random"` | Pattern: `random`, `block`, `seasonal` |
| `missing_rate` | `0.1` | Proportion of missing values (0-1) |
| `missing_block_size` | `3` | Size of missing blocks for `block` pattern |
| `missing_seasonal_period` | `7` | Period for `seasonal` pattern |

### Anomaly injection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `anomalies` | `False` | Enable anomaly injection |
| `anomaly_fraction` | `0.05` | Fraction of points that are anomalies |
| `anomaly_types` | `["spike", "dip"]` | Types: `spike`, `dip`, `level_shift` |
| `spike_magnitude` | `10.0` | Magnitude of spikes |
| `dip_magnitude` | `-10.0` | Magnitude of dips |
| `level_shift_magnitude` | `20.0` | Magnitude of level shifts |
| `level_shift_duration` | `10` | Duration of level shifts in time steps |

### Changepoint injection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `changepoints` | `False` | Enable changepoint injection |
| `num_changepoints` | `2` | Number of changepoints |
| `changepoint_type` | `"level"` | Type: `level`, `trend`, `variance`, `mixed` |
| `changepoint_level_changes` | `None` | Size of level changes (random when `None`) |
| `changepoint_trend_changes` | `None` | Size of trend changes (random when `None`) |
| `changepoint_variance_changes` | `None` | Size of variance changes (random when `None`) |
| `changepoint_locations` | `None` | Relative positions 0-1 (random when `None`) |

Per-generator sections below list only generator-specific parameters.

---

## Statistical Generators

### RandomWalkGenerator

Simple random walk with configurable drift and volatility. **Applications**: finance, econometrics, baseline testing.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `drift` | `0.0` | Mean of the random steps |
| `volatility` | `1.0` | Standard deviation of random steps (>= 0) |
| `start_value` | `0.0` | Initial value for all series |

### SeasonalGenerator

Deterministic seasonal pattern combined with trend and noise. **Applications**: retail, energy, transportation, tourism.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seasonality_period` | `24` | Period of seasonality (>= 1) |
| `seasonality_amplitude` | `10.0` | Amplitude of seasonal component |
| `trend` | `0.0` | Linear trend coefficient |
| `noise_level` | `1.0` | Standard deviation of noise (>= 0) |
| `base_level` | `50.0` | Base level of the series |

### SARIMAGenerator

Seasonal ARIMA with seasonal multiplicative polynomials and exogenous regressors. **Applications**: econometrics, demand forecasting, macroeconomics.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `p` | `1` | AR order (>= 0) |
| `d` | `0` | Differencing order (0-2) |
| `q` | `1` | MA order (>= 0) |
| `P` | `1` | Seasonal AR order (>= 0) |
| `D` | `0` | Seasonal differencing order (0-2) |
| `Q` | `1` | Seasonal MA order (>= 0) |
| `seasonal_period` | `12` | Seasonal period (>= 1) |
| `mean` | `0.0` | Process mean (for stationary models with d=0, D=0) |
| `drift` | `0.0` | Drift term (for integrated models with d>0 or D>0) |
| `noise_std` | `1.0` | Standard deviation of innovation noise (> 0) |
| `burn_in` | `None` | Burn-in period; `None` for automatic |
| `validate_stationarity` | `True` | Validate AR parameters for stationarity |
| `ar_params` | `None` | AR coefficients phi_1..phi_p |
| `ma_params` | `None` | MA coefficients theta_1..theta_q |
| `seasonal_ar_params` | `None` | Seasonal AR coefficients |
| `seasonal_ma_params` | `None` | Seasonal MA coefficients |
| `exog_coefficients` | `None` | Coefficients for exogenous regressors |

### ETSGenerator

Exponential smoothing state space (ETS) model supporting all 30 variants. **Applications**: forecasting, retail, supply chain.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `error_type` | `"add"` | Error type: `add` (A) or `mul` (M) |
| `trend_type` | `"add"` | Trend type: `add` (A), `mul` (M), or `None` (N) |
| `seasonal_type` | `"add"` | Seasonal type: `add` (A), `mul` (M), or `None` (N) |
| `seasonal_period` | `12` | Seasonal period m (>= 1) |
| `level` | `100.0` | Initial level l_0 |
| `trend` | `0.0` | Initial trend b_0 |
| `seasonal` | `None` | Initial seasonal factors |
| `alpha` | `0.3` | Level smoothing parameter (0-1) |
| `beta` | `0.1` | Trend smoothing parameter (0-1) |
| `gamma` | `0.1` | Seasonal smoothing parameter (0-1) |
| `phi` | `0.98` | Damping parameter (0-1) |
| `damped` | `False` | Use damped trend |
| `noise_std` | `1.0` | Standard deviation of noise (> 0) |
| `box_cox_lambda` | `None` | Box-Cox lambda (`None` = no transformation) |

### INARGenerator

Integer-valued autoregressive (INAR) counts via binomial thinning. **Applications**: epidemiology, insurance, operations research, ecology.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `p` | `1` | INAR order (>= 1) |
| `alpha` | `None` | Thinning probabilities (each in [0,1]) |
| `innovation_type` | `"poisson"` | Innovation distribution: `poisson`, `negative_binomial` |
| `innovation_mean` | `5.0` | Mean of innovation distribution (> 0) |
| `innovation_dispersion` | `2.0` | Dispersion for negative binomial (> 0) |

---

## Stochastic Generators

### GARCHGenerator

GARCH returns with time-varying volatility clustering. **Applications**: quantitative finance, risk management, volatility modeling.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `p` | `1` | GARCH order (>= 1) |
| `q` | `1` | ARCH order (>= 1) |
| `omega` | `0.1` | Constant term in variance equation |
| `alpha` | `None` | ARCH parameters |
| `beta` | `None` | GARCH parameters |
| `mu` | `0.0` | Mean of returns |
| `initial_variance` | `1.0` | Initial variance |

### OrnsteinUhlenbeckGenerator

Mean-reverting Ornstein-Uhlenbeck process, the continuous-time analog of AR(1). **Applications**: interest rates, pairs trading, physics, biology.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theta` | `0.5` | Speed of mean reversion (> 0) |
| `mu` | `0.0` | Long-term mean |
| `sigma` | `1.0` | Volatility (> 0) |
| `initial_value` | `0.0` | Initial value |
| `dt` | `1.0` | Time step (> 0) |

### GeometricBrownianMotionGenerator

Geometric Brownian Motion paths, the foundation of the Black-Scholes model. **Applications**: stock prices, options pricing, portfolio management.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mu` | `0.05` | Drift coefficient |
| `sigma` | `0.2` | Volatility (> 0) |
| `initial_value` | `100.0` | Initial value (> 0) |
| `dt` | `1.0` | Time step (> 0) |

### JumpDiffusionGenerator

Diffusion with random jumps (Merton model) capturing sudden market movements. **Applications**: finance, insurance, rare event modeling.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mu` | `0.05` | Drift coefficient |
| `sigma` | `0.2` | Diffusion volatility (> 0) |
| `lambda_jump` | `0.1` | Jump intensity, jumps per unit time (>= 0) |
| `jump_mean` | `0.0` | Mean of jump size (log scale) |
| `jump_std` | `0.1` | Std of jump size (log scale, >= 0) |
| `initial_value` | `100.0` | Initial value (> 0) |
| `dt` | `1.0` | Time step (> 0) |

### PoissonProcessGenerator

Poisson process arrivals. **Applications**: queuing theory, reliability engineering, telecommunications.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_rate` | `5.0` | Rate parameter (> 0) |
| `cumulative` | `False` | Return cumulative counts |

### CyclicGenerator

Multiple cyclic components with varying periods and amplitudes. **Applications**: environmental science, astronomy, biology.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_level` | `100.0` | Base level of series |
| `trend` | `0.0` | Linear trend coefficient |
| `cycle_period_mean` | `50.0` | Mean cycle period (> 0) |
| `cycle_period_std` | `10.0` | Std of cycle period variation (>= 0) |
| `cycle_amplitude_mean` | `20.0` | Mean cycle amplitude |
| `cycle_amplitude_std` | `5.0` | Std of cycle amplitude variation (>= 0) |
| `num_cycles` | `3` | Number of cycles to generate (> 0) |
| `noise_std` | `1.0` | Standard deviation of noise (>= 0) |

### FractionalBrownianMotionGenerator

Fractional Brownian motion with long-range dependence controlled by the Hurst exponent. **Applications**: network traffic, hydrology, finance, turbulence.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hurst` | `0.5` | Hurst exponent H in (0, 1) |
| `sigma` | `1.0` | Volatility/scale parameter (> 0) |
| `method` | `"fft"` | Generation method: `fft` (O(n log n)), `cholesky`, `hosking` |
| `return_increments` | `False` | Return fGn increments instead of cumulative fBm |
| `initial_value` | `0.0` | Starting value for the process |

### HawkesProcessGenerator

Self-exciting point process where past events increase future event probability. **Applications**: order flow, seismology, social media, epidemiology.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `baseline_intensity` | `1.0` | Background event rate mu (> 0) |
| `excitation_amplitude` | `0.5` | Jump in intensity per event alpha (>= 0) |
| `decay_rate` | `1.0` | Rate of intensity decay beta (> 0) |
| `kernel` | `"exponential"` | Excitation kernel: `exponential`, `power_law` |
| `power_law_exponent` | `1.5` | Exponent for power-law kernel (> 1) |
| `output_type` | `"counts"` | Output: `counts`, `intensity`, `events` |
| `max_events` | `10000` | Maximum events to simulate (> 0) |

### StochasticVolatilityGenerator

Heston or SABR model where volatility follows its own stochastic process. **Applications**: derivatives pricing, risk management, quantitative finance.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"heston"` | Model: `heston`, `sabr` |
| `initial_price` | `100.0` | Starting price S0 (> 0) |
| `initial_vol` | `0.04` | Starting variance V0 (SABR starts at sqrt(initial_vol)) |
| `drift` | `0.05` | Price drift mu |
| `mean_vol` | `0.04` | Long-run variance theta (Heston, > 0) |
| `vol_mean_reversion` | `2.0` | Variance mean-reversion speed kappa (> 0) |
| `vol_of_vol` | `0.3` | Volatility of volatility (Heston sigma_v, SABR alpha) |
| `correlation` | `-0.7` | Price-volatility correlation rho in [-1, 1] |
| `beta` | `0.5` | CEV exponent (SABR only; 0 = normal, 1 = lognormal) |
| `dt` | `1/252` | Time step (1/252 = daily with 252 trading days) |
| `output_type` | `"price"` | Output: `price`, `returns`, `volatility` |

### RegimeSwitchingGenerator

Markov-switching series transitioning between regimes with distinct dynamics. **Applications**: macroeconomics, business cycles, bull/bear markets.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_regimes` | `2` | Number of regimes/states (>= 2) |
| `regime_means` | `None` | Mean for each regime |
| `regime_variances` | `None` | Variance for each regime |
| `regime_ar_coeffs` | `None` | AR(1) coefficient for each regime |
| `transition_matrix` | `None` | Regime transition probability matrix |
| `initial_regime` | `None` | Starting regime (0-indexed) |

### ChaoticSystemGenerator

Deterministic chaotic dynamics: Lorenz attractor, logistic map, Mackey-Glass. **Applications**: physics, climate science, neuroscience, nonlinear forecasting benchmarks.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `system` | `"lorenz"` | System: `lorenz`, `logistic`, `mackey_glass` |
| `sigma` | `10.0` | Lorenz sigma |
| `rho` | `28.0` | Lorenz rho |
| `beta_param` | `2.6667` | Lorenz beta |
| `dt` | `0.01` | Integration step size (> 0) |
| `logistic_r` | `3.9` | Logistic map parameter r |
| `mg_beta` | `0.2` | Mackey-Glass beta |
| `mg_gamma` | `0.1` | Mackey-Glass gamma |
| `mg_n` | `10.0` | Mackey-Glass exponent n |
| `mg_tau` | `17` | Mackey-Glass delay tau (>= 1) |
| `observation_noise` | `0.0` | Observation noise std (>= 0) |
| `initial_perturbation` | `0.01` | Initial condition perturbation scale (>= 0) |

### BoundedProcessGenerator

Series constrained to a finite interval via Beta-AR or logit-normal models. **Applications**: market share, utilization rates, bounded environmental and healthcare metrics.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"beta_ar"` | Model: `beta_ar`, `logit_normal` |
| `phi` | `0.8` | AR coefficient in [-1, 1] |
| `omega` | `0.1` | Intercept for beta_ar conditional mean |
| `kappa` | `20.0` | Beta precision parameter (> 0) |
| `sigma` | `0.3` | Logit-normal innovation std (> 0) |
| `initial_value` | `0.5` | Starting value in (0, 1) |
| `lower` | `0.0` | Lower bound |
| `upper` | `1.0` | Upper bound |

### LevyProcessGenerator

Heavy-tailed alpha-stable process via the Chambers-Mallows-Stuck algorithm. **Applications**: tail risk, insurance catastrophes, heavy-tailed network traffic, anomalous diffusion.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `1.5` | Stability parameter in (0, 2] |
| `beta_skew` | `0.0` | Skewness parameter in [-1, 1] |
| `scale` | `1.0` | Scale parameter (> 0) |
| `location` | `0.0` | Location parameter |
| `cumulative` | `True` | Return cumulative sum (Levy flight) |
| `initial_value` | `0.0` | Starting value for cumulative mode |

---

## Multivariate Generators

### CopulaGenerator

Multivariate series with specified marginal distributions and dependency structure. **Applications**: portfolio risk, correlated assets, insurance aggregate claims.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `copula_type` | `"gaussian"` | Copula type: `gaussian`, `t` |
| `correlation_matrix` | `None` | Correlation matrix for n_series |
| `df` | `5.0` | Degrees of freedom for t-copula (> 0) |
| `marginal_distributions` | `[{"type": "normal", "loc": 0.0, "scale": 1.0}]` | Marginal distributions |

### VARGenerator

Vector autoregressive (VAR) series with inter-series dependencies. **Applications**: macroeconomics, multi-asset systems, Granger causality studies.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lag_order` | `1` | VAR lag order (>= 1) |
| `coef_matrices` | `None` | Coefficient matrices for each lag |
| `intercept` | `None` | Intercept vector |
| `innovation_covariance` | `None` | Covariance matrix of innovations |

### GaussianProcessGenerator

Smooth, correlated series from Gaussian Process priors with configurable kernels. **Applications**: geostatistics, Bayesian optimization, robotics, environmental science.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel` | `"rbf"` | Kernel: `rbf`, `matern_0.5`, `matern_1.5`, `matern_2.5`, `periodic` |
| `length_scale` | `20.0` | Kernel length scale (> 0) |
| `amplitude` | `1.0` | Signal amplitude (> 0) |
| `period` | `50.0` | Period for periodic kernel (> 0) |
| `mean` | `0.0` | Mean function value |
| `noise_variance` | `1e-06` | Observation noise variance (>= 0) |

---

## Domain-Specific Generators

### IntermittentDemandGenerator

Sporadic demand with many zero values, typical of spare parts and slow movers. **Applications**: supply chain, inventory management, spare parts forecasting.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `demand_probability` | `0.2` | Probability of non-zero demand (0-1) |
| `demand_distribution` | `"poisson"` | Distribution for non-zero demand: `poisson`, `negative_binomial`, `lognormal`, `gamma` |
| `demand_mean` | `5.0` | Mean of demand when non-zero (> 0) |
| `demand_std` | `2.0` | Std of demand (>= 0) |
| `intermittent_pattern` | `"random"` | Pattern: `random`, `clustered`, `seasonal` |
| `cluster_size` | `3` | Size of demand clusters (>= 1) |
| `seasonal_period` | `12` | Period for seasonal intermittency (>= 1) |
| `seasonal_peak_prob` | `0.4` | Peak probability in seasonal pattern (0-1) |
| `min_demand` | `1` | Minimum non-zero demand value (>= 1) |

### IoTSensorGenerator

IoT sensor readings with drift, noise, failures, and battery effects. **Applications**: IoT, manufacturing, predictive maintenance.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_sensors` | `1` | Number of sensors to generate (>= 1) |
| `sensor_type` | `"temperature"` | Sensor: `temperature`, `humidity`, `pressure`, `light`, `motion`, `generic` |
| `base_value` | `None` | Base sensor reading |
| `trend` | `0.0` | Linear drift rate per time step |
| `seasonal_period` | `0` | Seasonal cycle length (>= 0) |
| `seasonal_amplitude` | `0.0` | Amplitude of seasonal variation |
| `measurement_noise` | `0.1` | Std of measurement noise |
| `drift_rate` | `0.0` | Rate of sensor drift |
| `drift_noise` | `0.01` | Random variation in drift |
| `calibration_error` | `0.0` | Initial calibration offset |
| `battery_life` | `None` | Steps until battery degradation starts |
| `battery_degradation_rate` | `0.001` | Rate of quality loss |
| `failure_probability` | `0.0` | Probability of sensor failure (0-1) |
| `failure_type` | `"intermittent"` | Failure type: `intermittent`, `complete`, `stuck` |
| `failure_duration` | `10` | Duration of intermittent failures (>= 1) |
| `stuck_value` | `None` | Value when stuck |
| `spatial_correlation` | `0.5` | Correlation between nearby sensors (0-1) |

### EnergyLoadGenerator

Electricity demand with daily/weekly/yearly seasonality and weather effects. **Applications**: energy, utilities, smart grid.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_load` | `100.0` | Base load in kW or MW (> 0) |
| `load_type` | `"residential"` | Load type: `residential`, `commercial`, `industrial` |
| `daily_pattern` | `True` | Enable daily cycle |
| `daily_amplitude` | `30.0` | Amplitude of daily variation |
| `weekly_pattern` | `True` | Enable weekly cycle |
| `weekly_amplitude` | `15.0` | Amplitude of weekly variation |
| `yearly_pattern` | `True` | Enable yearly cycle |
| `yearly_amplitude` | `20.0` | Amplitude of yearly variation |
| `temperature_sensitive` | `True` | Enable temperature effects |
| `temperature_sensitivity` | `2.0` | Load change per degree |
| `base_temperature` | `20.0` | Base temperature in Celsius |
| `morning_peak_hour` | `8` | Hour of morning peak (0-23) |
| `evening_peak_hour` | `19` | Hour of evening peak (0-23) |
| `peak_amplitude` | `40.0` | Additional load during peaks |
| `holiday_effect` | `0.3` | Load reduction on holidays (0-1) |
| `holiday_days` | `[]` | Day indices for holidays |
| `extreme_weather_prob` | `0.0` | Probability of extreme weather (0-1) |
| `extreme_weather_impact` | `1.5` | Load multiplier during extreme weather |
| `noise_std` | `5.0` | Standard deviation of random noise |

### StateSpaceGenerator

General linear or custom state space models with transition and observation equations. **Applications**: control systems, econometrics, signal processing, Kalman filter testing.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `state_dim` | `1` | Dimension of hidden state vector (>= 1) |
| `obs_dim` | `1` | Dimension of observation vector (>= 1) |
| `transition_matrix` | `None` | State transition matrix F |
| `observation_matrix` | `None` | Observation matrix H |
| `state_covariance` | `None` | State noise covariance Q |
| `obs_covariance` | `None` | Observation noise covariance R |
| `transition_fn` | `None` | Custom state transition function |
| `observation_fn` | `None` | Custom observation function |
| `initial_state` | `None` | Initial state x[0] |
| `initial_state_covariance` | `None` | Initial state uncertainty |

### DailyActiveUsersGenerator

DAU patterns for digital products with growth, weekly seasonality, and event spikes. **Applications**: product analytics, mobile apps, SaaS.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_users` | `10000.0` | Base number of daily active users (> 0) |
| `growth_rate` | `0.0005` | Daily organic growth rate |
| `growth_rate_std` | `0.0` | Std dev for per-series growth rate perturbation (>= 0) |
| `app_type` | `"consumer"` | App type: `consumer`, `business`, `gaming` |
| `weekly_pattern` | `True` | Enable weekly seasonality |
| `weekend_factor` | `None` | Multiplier for weekend activity (> 0) |
| `event_probability` | `0.02` | Daily probability of an event (0-1) |
| `event_impact_min` | `1.2` | Minimum event impact multiplier (>= 1) |
| `event_impact_max` | `2.0` | Maximum event impact multiplier |
| `event_decay_rate` | `0.1` | Rate at which event impact decays (0-1) |
| `noise_std` | `0.05` | Standard deviation of noise |
| `event_col` | `"event"` | Name of the event indicator column |

### VitalSignsGenerator

Physiologically realistic vital signs with patient archetypes. **Applications**: healthcare, medical devices, clinical research.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patient_type` | `"healthy"` | Archetype: `healthy`, `cardiac`, `sepsis`, `respiratory`, `hypertensive` |
| `vital_sign` | `"heart_rate"` | Signal: `heart_rate`, `systolic_bp`, `diastolic_bp`, `respiratory_rate`, `spo2`, `temperature` |
| `include_circadian` | `True` | Include circadian rhythm effects |
| `include_hrv` | `True` | Include heart rate variability |
| `include_events` | `True` | Include random physiological events |
| `event_probability` | `0.01` | Probability of event per timestep (0-1) |

### ClickstreamGenerator

Web traffic with sessions, pageviews, conversions, bounce rates, and bot traffic. **Applications**: web analytics, digital marketing, e-commerce.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_sessions` | `100.0` | Baseline sessions per time unit (> 0) |
| `traffic_source` | `"mixed"` | Source: `organic`, `paid`, `direct`, `referral`, `mixed` |
| `conversion_rate` | `0.03` | Base conversion rate (0-1) |
| `bounce_rate` | `0.4` | Base bounce rate, single-page sessions (0-1) |
| `avg_session_depth` | `3.5` | Average pages per non-bounced session (> 0) |
| `include_seasonality` | `True` | Include time-of-day and day-of-week patterns |
| `include_bots` | `True` | Include bot traffic |
| `bot_fraction` | `0.15` | Fraction of traffic from bots (0-1) |
| `output_type` | `"sessions"` | Metric: `sessions`, `pageviews`, `conversions`, `bounce_rate` |

---

## Pretraining Generators

Designed for diverse pretraining corpora: each series samples a fresh random
configuration, so a pool spans
trend-only, pure-seasonal, causally-structured, and noise-dominated regimes.

The `pretraining_pool()` preset collects these three meta-generators (plus the
interpretable `balanced_pool` by default) into a breadth-maximizing corpus:

```python
from synforecast import SynSet, pretraining_pool

df = SynSet(pretraining_pool(min_length=512, max_length=512, freq="h")).generate(
    n_series_per_generator=1
)
```

KernelSynth and Gaussian-process sampling require covariance factorizations,
whose cost grows cubically with series length. Scale the length and number of
series gradually when building a large corpus.

### TSIGenerator

Trend + seasonality + irregularity composition with randomized presence,
types, and weights per series. **Applications**: pretraining data, augmentation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trend_types` | all 6 | Trend shapes sampled per series: `none`, `linear`, `exponential`, `logistic`, `piecewise_linear`, `damped` |
| `n_seasonal_range` | `(0, 3)` | Number of seasonal harmonics per series |
| `seasonal_periods` | mixed pool | Periods in steps; integer (7, 12, 24, ..., 365.25) and non-integer/co-prime values |
| `seasonal_amplitude_range` | log-uniform | Amplitude draw range per harmonic |
| `amplitude_modulation_prob` | prob | Slowly varying amplitude envelope |
| `harmonics_prob` | prob | Add decaying 2f/3f overtones (non-sinusoidal shapes) |
| `irregular_types` | all 5 | Noise process: `gaussian`, `ar1`, `garch_like`, `student_t`, `laplace` |
| `noise_scale_range` | `(0.5, 12.0)` | Noise std as a fraction of the structural signal std |
| `multiplicative_prob` | `0.3` | Probability of multiplicative (vs. additive) composition |

### TCMGenerator

Random temporal structural causal model: a sparse dependency graph over
latent variables and lags is sampled per series, rolled out with linear or
mildly nonlinear edge functions, and one node is observed. **Applications**:
pretraining data with genuine causal/lead-lag structure.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_vars_range` | `(1, 5)` | Number of latent variables |
| `max_lag_range` | `(1, 24)` | Maximum dependency lag |
| `edge_probability_range` | `(0.05, 0.3)` | Per-(var, lag) edge probability |
| `edge_kinds` | all 5 | Edge functions: `linear`, `tanh`, `relu`, `product`, `threshold` |
| `stability_margin` | `0.95` | Spectral-radius bound for the linear part |
| `noise_types` | all 3 | Node innovations: `gaussian`, `student_t`, `laplace` |
| `heteroscedastic_prob` | prob | Slow random noise-scale envelope per node |
| `multivariate` | `False` | When True, `generate(n_series)` observes n_series nodes of one shared causal system as correlated series |

### KernelSynthGenerator

Samples each series from a Gaussian-process prior whose kernel is a random
composition of `1..max_kernels` base kernels combined with `+`/`*` operators.
This adapts the KernelSynth recipe used to pretrain the Chronos models (Ansari
et al. 2024) and the Apache-2.0-licensed
[reference implementation](https://github.com/amazon-science/chronos-forecasting/blob/main/scripts/kernel-synth.py).
SynForecast adds a configurable kernel bank, time-step seasonal periods,
bounded retries, divergence guards, and optional standardization.
**Applications**: pretraining data with controllable temporal structure.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_kernels` | `5` | Base kernels composed per series (drawn from `1..max_kernels`) |
| `seasonal_periods` | broad pool | Periodic-kernel periods in time steps (4–730) |
| `rbf_length_scales` | `[0.1, 1.0, 10.0]` | RBF length scales on the normalized grid |
| `rational_quadratic_alphas` | `[0.1, 1.0, 10.0]` | Rational-quadratic shape parameters |
| `linear_sigmas` | `[0.0, 1.0, 10.0]` | Linear (DotProduct) `sigma_0` offsets |
| `white_noise_levels` | `[0.1, 1.0]` | White-kernel diagonal noise levels |
| `include_constant` | `True` | Include a constant kernel in the bank |
| `standardize` | `True` | Standardize each series to zero mean, unit variance |

### Multivariatizer

Not a generator: `synforecast.Multivariatizer` wraps any univariate
generator and produces cross-dependent channels sharing one length.
Couplings (sampled per call): `mixing` — instantaneous random correlation
via a well-conditioned mixing matrix; `leadlag` — channels as lagged,
scaled, noise-perturbed transforms of others. Channel diversity statistics
of the wrapped generator are preserved.

```python
from synforecast import Multivariatizer
from synforecast.generators import TSIGenerator

mv = Multivariatizer(
    base=TSIGenerator(min_length=512, max_length=512, freq="D"),
    couplings=["mixing", "leadlag"],
    seed=42,
)
df = mv.generate(n_series=4)
```

TSI and TCM use the Rust batch path when the extension is available;
KernelSynth is pure NumPy (it runs on the threaded fallback path). Reproduce
performance measurements on your hardware with the scripts in `benchmarks/`;
benchmark results are not API guarantees.

---

## Quick Reference Table

| Generator | Category | Primary Domain | Key Features |
|-----------|----------|----------------|--------------|
| RandomWalk | Statistical | Finance | Drift, volatility |
| Seasonal | Statistical | Retail/Energy | Periodic patterns |
| SARIMA | Statistical | Forecasting | AR, MA, differencing, seasonal |
| ETS | Statistical | Forecasting | 30 model variants |
| INAR | Statistical | Epidemiology/Insurance | Integer counts, binomial thinning |
| GARCH | Stochastic | Finance | Volatility clustering |
| OrnsteinUhlenbeck | Stochastic | Finance/Physics | Mean reversion |
| GeometricBrownianMotion | Stochastic | Finance | Log-normal prices |
| JumpDiffusion | Stochastic | Finance | Sudden jumps |
| PoissonProcess | Stochastic | Operations | Event arrivals |
| Cyclic | Stochastic | Science | Multiple frequencies |
| FractionalBrownianMotion | Stochastic | Networks/Finance | Long-range dependence, Hurst exponent |
| HawkesProcess | Stochastic | Finance/Social | Self-excitation |
| StochasticVolatility | Stochastic | Derivatives | Heston/SABR models |
| RegimeSwitching | Stochastic | Macro/Finance | Markov switching |
| ChaoticSystem | Stochastic | Physics/Climate | Lorenz, logistic, Mackey-Glass |
| BoundedProcess | Stochastic | Marketing/Economics | Beta-AR, logit-normal, bounded interval |
| LevyProcess | Stochastic | Finance/Insurance | Alpha-stable, heavy tails |
| Copula | Multivariate | Risk | Dependency structure |
| VAR | Multivariate | Macro | Vector autoregression |
| GaussianProcess | Multivariate | Geostatistics | RBF, Matern, periodic kernels |
| IntermittentDemand | Domain | Supply Chain | Sporadic demand |
| IoTSensor | Domain | Manufacturing | Sensor data, failures |
| EnergyLoad | Domain | Utilities | Grid demand |
| StateSpace | Domain | Control | Custom state equations |
| DailyActiveUsers | Domain | Product | User engagement |
| VitalSigns | Domain | Healthcare | Patient monitoring |
| Clickstream | Domain | Web Analytics | Sessions, conversions |
| TSI | Pretraining | Foundation models | Randomized trend/seasonal/irregular composition |
| TCM | Pretraining | Foundation models | Random causal graphs, nonlinear lead-lag structure |
| KernelSynth | Pretraining | Foundation models | GP samples from randomly composed kernels (Chronos recipe) |

---

## Usage Example

All generators follow the same keyword-only constructor and `generate()` interface:

```python
from synforecast.generators import RegimeSwitchingGenerator

generator = RegimeSwitchingGenerator(
    min_length=100,
    max_length=200,
    freq="D",
    n_regimes=2,
    regime_means=[0.0, 5.0],
    seed=42,
)
df = generator.generate(n_series=10)

# df is a pandas DataFrame (default engine) in long format with columns:
#   unique_id (integer categorical), ds (datetime64[ns]), y (float64)
```

See the [`nbs/docs/generators`](https://github.com/Nixtla/synforecast/tree/main/nbs/docs/generators)
directory for executable guides to each generator.

---

## References and attribution

Unless noted otherwise, SynForecast implements the models and numerical methods
independently. These are the primary sources for named models or algorithms;
the KernelSynth entry explicitly identifies its reference implementation:

- Hyndman, Koehler, Ord, and Snyder (2008), *Forecasting with Exponential
  Smoothing: The State Space Approach*,
  [doi:10.1007/978-3-540-71918-2](https://doi.org/10.1007/978-3-540-71918-2)
  (ETS state equations).
- Bollerslev (1986), “Generalized Autoregressive Conditional
  Heteroskedasticity,”
  [doi:10.1016/0304-4076(86)90063-1](https://doi.org/10.1016/0304-4076%2886%2990063-1)
  (GARCH).
- Merton (1976), “Option Pricing When Underlying Stock Returns Are
  Discontinuous,”
  [doi:10.1016/0304-405X(76)90022-2](https://doi.org/10.1016/0304-405X%2876%2990022-2)
  (jump diffusion).
- Mandelbrot and Van Ness (1968), “Fractional Brownian Motions, Fractional
  Noises and Applications,”
  [doi:10.1137/1010093](https://doi.org/10.1137/1010093) (fractional Brownian
  motion).
- Hawkes (1971), “Spectra of Some Self-Exciting and Mutually Exciting Point
  Processes,” [doi:10.1093/biomet/58.1.83](https://doi.org/10.1093/biomet/58.1.83),
  and Ogata (1981), “On Lewis' Simulation Method for Point Processes,”
  [doi:10.1109/TIT.1981.1056305](https://doi.org/10.1109/TIT.1981.1056305)
  (Hawkes process and thinning simulation).
- Chambers, Mallows, and Stuck (1976), “A Method for Simulating Stable Random
  Variables,”
  [doi:10.1080/01621459.1976.10480344](https://doi.org/10.1080/01621459.1976.10480344)
  (alpha-stable draws used by `LevyProcessGenerator`).
- Bahrpeyma et al. (2021), “A Methodology for Validating Diversity in
  Synthetic Time Series Generation,”
  [doi:10.1016/j.mex.2021.101459](https://doi.org/10.1016/j.mex.2021.101459)
  (trend/seasonality/irregularity composition). `TSIGenerator` extends this
  construction with additional component families and safeguards.
- Runge et al. (2023), “Causal Inference for Time Series,”
  [doi:10.1038/s43017-023-00431-y](https://doi.org/10.1038/s43017-023-00431-y)
  (temporal structural-causal framing). `TCMGenerator`'s graph sampler and
  rollout are original SynForecast design choices, not an implementation
  published in that review.
- Ansari et al. (2024), “Chronos: Learning the Language of Time Series,”
  [arXiv:2403.07815](https://arxiv.org/abs/2403.07815), and the official
  Apache-2.0-licensed
  [KernelSynth implementation](https://github.com/amazon-science/chronos-forecasting/blob/main/scripts/kernel-synth.py)
  (KernelSynth and TSMixup recipes; the linked script is the KernelSynth
  reference implementation). SynForecast's deviations are described above
  and in the corresponding API documentation.
- Ansari et al. (2025), “Chronos-2: From Univariate to Universal Forecasting,”
  [arXiv:2510.15821](https://arxiv.org/abs/2510.15821) (motivation for the
  cotemporaneous and sequential couplings in `Multivariatizer`).
- Wichura (1988), “Algorithm AS 241: The Percentage Points of the Normal
  Distribution,” [doi:10.2307/2347330](https://doi.org/10.2307/2347330)
  (inverse-normal approximation in `_distributions.py`).
