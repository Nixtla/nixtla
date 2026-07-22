"""Type stubs for the Rust _lib extension module (PyO3/maturin)."""

import numpy as np
import numpy.typing as npt

class pattern_injection:
    @staticmethod
    def add_changepoints(
        values: npt.NDArray[np.float64],
        seed: int,
        num_changepoints: int,
        locations: npt.NDArray[np.float64],
        changepoint_type: str,
        level_changes: npt.NDArray[np.float64],
        trend_changes: npt.NDArray[np.float64],
        variance_changes: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.int64]]]: ...
    @staticmethod
    def add_missingness(
        values: npt.NDArray[np.float64],
        seed: int,
        pattern: str,
        missing_rate: float,
        missing_block_size: int,
        missing_seasonal_period: int,
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.int64]]]: ...
    @staticmethod
    def add_anomalies(
        values: npt.NDArray[np.float64],
        seed: int,
        anomaly_types: list[str],
        anomaly_fraction: float,
        spike_magnitude: float,
        dip_magnitude: float,
        level_shift_magnitude: float,
        level_shift_duration: int,
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.int64]]]: ...

class statistical:
    @staticmethod
    def random_walk(
        length: int,
        drift: float,
        volatility: float,
        start_value: float,
        seed: int,
        innov_dist: int,
        innov_param: float,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def seasonal(
        length: int,
        seasonality_period: int,
        seasonality_amplitude: float,
        trend: float,
        noise_level: float,
        base_level: float,
        seed: int,
        innov_dist: int = 0,
        innov_param: float = 0.0,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def sarima(
        length: int,
        full_ar_poly: npt.NDArray[np.float64],
        full_ma_poly: npt.NDArray[np.float64],
        d: int,
        D: int,
        seasonal_period: int,
        mean: float,
        drift_val: float,
        noise_std: float,
        burn_in: int,
        seed: int,
        innov_dist: int,
        innov_param: float,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def ets(
        length: int,
        error_type: int,
        trend_type: int,
        seasonal_type: int,
        seasonal_period: int,
        level: float,
        trend_init: float,
        seasonal_init: npt.NDArray[np.float64],
        alpha: float,
        beta_param: float,
        gamma: float,
        phi: float,
        damped: bool,
        noise_std: float,
        seed: int,
        innov_dist: int,
        innov_param: float,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def inar(
        length: int,
        p: int,
        alpha_arr: npt.NDArray[np.float64],
        innov_type: int,
        innov_mean: float,
        innov_dispersion: float,
        seed: int,
    ) -> npt.NDArray[np.float64]: ...

class stochastic:
    @staticmethod
    def ornstein_uhlenbeck(
        length: int,
        theta: float,
        mu: float,
        sigma: float,
        initial_value: float,
        dt: float,
        seed: int,
        innov_dist: int,
        innov_param: float,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def geometric_brownian_motion(
        length: int,
        mu: float,
        sigma: float,
        initial_value: float,
        dt: float,
        seed: int,
        innov_dist: int,
        innov_param: float,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def jump_diffusion(
        length: int,
        mu: float,
        sigma: float,
        lambda_jump: float,
        jump_mean: float,
        jump_std: float,
        initial_value: float,
        dt: float,
        seed: int,
        innov_dist: int,
        innov_param: float,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def poisson_process(
        length: int,
        lambda_rate: float,
        cumulative: bool,
        seed: int,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def cyclic(
        length: int,
        base_level: float,
        trend: float,
        cycle_period_mean: float,
        cycle_period_std: float,
        cycle_amplitude_mean: float,
        cycle_amplitude_std: float,
        num_cycles: int,
        noise_std: float,
        seed: int,
        innov_dist: int,
        innov_param: float,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def garch(
        length: int,
        p: int,
        q: int,
        omega: float,
        alpha_arr: npt.NDArray[np.float64],
        beta_arr: npt.NDArray[np.float64],
        mu: float,
        initial_variance: float,
        seed: int,
        innov_dist: int,
        innov_param: float,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def hawkes_process(
        length: int,
        baseline_intensity: float,
        excitation_amplitude: float,
        decay_rate: float,
        kernel_type: int,
        power_law_exponent: float,
        output_type: int,
        max_events: int,
        seed: int,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def chaotic_system(
        length: int,
        system_id: int,
        sigma: float,
        rho: float,
        beta: float,
        dt: float,
        logistic_r: float,
        mg_beta: float,
        mg_gamma: float,
        mg_n: float,
        mg_tau: int,
        observation_noise: float,
        initial_perturbation: float,
        seed: int,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def bounded_process(
        length: int,
        model_id: int,
        phi: float,
        omega: float,
        kappa: float,
        sigma_param: float,
        initial_value: float,
        seed: int,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def levy_process(
        length: int,
        alpha: float,
        beta_skew: float,
        scale: float,
        location: float,
        cumulative: bool,
        initial_value: float,
        seed: int,
    ) -> npt.NDArray[np.float64]: ...

class volatility:
    @staticmethod
    def stochastic_volatility(
        length: int,
        model_type: int,
        initial_price: float,
        initial_vol: float,
        drift: float,
        mean_vol: float,
        vol_mean_reversion: float,
        vol_of_vol: float,
        correlation: float,
        beta_param: float,
        dt: float,
        output_type: int,
        seed: int,
        innov_dist: int,
        innov_param: float,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def regime_switching(
        length: int,
        n_regimes: int,
        regime_means: npt.NDArray[np.float64],
        regime_variances: npt.NDArray[np.float64],
        regime_ar_coeffs: npt.NDArray[np.float64],
        transition_matrix: npt.NDArray[np.float64],
        stationary_probs: npt.NDArray[np.float64],
        initial_regime: int,
        seed: int,
        innov_dist: int,
        innov_param: float,
    ) -> npt.NDArray[np.float64]: ...

class multivariate:
    @staticmethod
    def copula(
        length: int,
        n_variables: int,
        copula_type: int,
        df: float,
        correlation_matrix: npt.NDArray[np.float64],
        marginal_distribution: int,
        marginal_param1: float,
        marginal_param2: float,
        seed: int,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def var_process(
        length: int,
        n_variables: int,
        order: int,
        coef_matrices: npt.NDArray[np.float64],
        intercept: npt.NDArray[np.float64],
        innovation_cov: npt.NDArray[np.float64],
        seed: int,
        innov_dist: int,
        innov_param: float,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def state_space(
        length: int,
        state_dim: int,
        obs_dim: int,
        F_mat: npt.NDArray[np.float64],
        H_mat: npt.NDArray[np.float64],
        Q_mat: npt.NDArray[np.float64],
        R_mat: npt.NDArray[np.float64],
        initial_state: npt.NDArray[np.float64],
        seed: int,
        innov_dist: int,
        innov_param: float,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def fbm(
        length: int,
        hurst: float,
        sigma: float,
        initial_value: float,
        cumulative: bool,
        method: int,
        seed: int,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def gaussian_process(
        length: int,
        kernel_id: int,
        length_scale: float,
        amplitude: float,
        period: float,
        mean: float,
        noise_variance: float,
        seed: int,
    ) -> npt.NDArray[np.float64]: ...

class domain:
    @staticmethod
    def intermittent_demand(
        length: int,
        demand_probability: float,
        demand_distribution: int,
        demand_mean: float,
        demand_std: float,
        intermittent_pattern: int,
        cluster_size: int,
        seasonal_period: int,
        seasonal_peak_prob: float,
        min_demand: float,
        seed: int,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def daily_active_users(
        length: int,
        base_users: float,
        growth_rate: float,
        weekend_factor: float,
        weekly_pattern: bool,
        event_probability: float,
        event_impact_min: float,
        event_impact_max: float,
        event_decay_rate: float,
        noise_std: float,
        step_hours: float,
        seed: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]: ...
    @staticmethod
    def energy_load(
        length: int,
        base_load: float,
        load_type: int,
        daily_pattern: bool,
        daily_amplitude: float,
        weekly_pattern: bool,
        weekly_amplitude: float,
        yearly_pattern: bool,
        yearly_amplitude: float,
        temperature_sensitive: bool,
        temperature_sensitivity: float,
        base_temperature: float,
        morning_peak_hour: int,
        evening_peak_hour: int,
        peak_amplitude: float,
        holiday_effect: float,
        holiday_days: npt.NDArray[np.int32],
        extreme_weather_prob: float,
        extreme_weather_impact: float,
        noise_std: float,
        step_hours: float,
        seed: int,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def vital_signs(
        length: int,
        baseline_mean: float,
        baseline_std: float,
        min_val: float,
        max_val: float,
        include_circadian: bool,
        circadian_magnitude: float,
        include_hrv: bool,
        include_events: bool,
        event_probability: float,
        vital_sign_type: int,
        seed: int,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def iot_sensor(
        length: int,
        base_value: float,
        trend: float,
        amplitude: float,
        period: float,
        measurement_noise: float,
        drift_rate: float,
        drift_noise: float,
        battery_degradation: bool,
        battery_life: float,
        battery_degradation_rate: float,
        calibration_offset: float,
        failure_mode: int,
        failure_probability: float,
        failure_duration: int,
        stuck_value: float | None,
        spatial_correlation: float,
        seed: int,
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def clickstream(
        length: int,
        base_sessions: float,
        conversion_mult: float,
        bounce_mult: float,
        depth_mult: float,
        seasonality_amp: float,
        conversion_rate: float,
        bounce_rate: float,
        avg_session_depth: float,
        include_seasonality: bool,
        include_bots: bool,
        bot_fraction: float,
        output_type: int,
        seed: int,
    ) -> npt.NDArray[np.float64]: ...

class batch:
    @staticmethod
    def generate_batch(
        gen_type: int,
        scalar_params: npt.NDArray[np.float64],
        array_params: list[npt.NDArray[np.float64]],
        lengths: npt.NDArray[np.int32],
        gen_seeds: npt.NDArray[np.uint64],
        pi_seeds: npt.NDArray[np.uint64],
        pi_config_dict: dict,
        n_workers: int = 0,
    ) -> list[dict[str, npt.NDArray]]: ...
    @staticmethod
    def generate_multi_batch(
        specs: list[dict],
        n_workers: int = 0,
    ) -> list[list[dict[str, npt.NDArray]]]: ...

class distributions:
    @staticmethod
    def norm_cdf(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def norm_ppf(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def t_cdf(x: npt.NDArray[np.float64], df: float) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def gamma_ppf(
        u: npt.NDArray[np.float64], a: float, scale: float
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def expon_ppf(
        u: npt.NDArray[np.float64], scale: float
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def lognorm_ppf(
        u: npt.NDArray[np.float64], s: float, scale: float
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def uniform_ppf(
        u: npt.NDArray[np.float64], loc: float, scale: float
    ) -> npt.NDArray[np.float64]: ...
