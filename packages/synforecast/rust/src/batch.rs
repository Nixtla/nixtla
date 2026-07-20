use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::generators::{domain, multivariate, statistical, stochastic, tcm, tsi, volatility};
use crate::pattern_injection as pi;

/// Return a cached rayon `ThreadPool` for the given worker count.
///
/// The pool is created once on first use and reused on subsequent calls with
/// the same `n_workers` value.  If `n_workers` changes, the old pool is
/// dropped and a new one is built.
fn get_or_create_pool(n_workers: usize) -> Result<Arc<rayon::ThreadPool>, String> {
    static CACHED: Mutex<Option<(usize, Arc<rayon::ThreadPool>)>> = Mutex::new(None);
    let mut guard = CACHED.lock().unwrap();
    if let Some((w, pool)) = guard.as_ref() {
        if *w == n_workers {
            return Ok(Arc::clone(pool));
        }
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_workers)
            .build()
            .map_err(|e| format!("failed to build rayon thread pool: {e}"))?,
    );
    *guard = Some((n_workers, Arc::clone(&pool)));
    Ok(pool)
}

// Generator type IDs — must match Python-side _GEN_TYPE_MAP
pub const GEN_RANDOM_WALK: i32 = 0;
pub const GEN_SEASONAL: i32 = 1;
pub const GEN_SARIMA: i32 = 2;
pub const GEN_ETS: i32 = 3;
pub const GEN_INAR: i32 = 4;
pub const GEN_ORNSTEIN_UHLENBECK: i32 = 5;
pub const GEN_GEOMETRIC_BROWNIAN_MOTION: i32 = 6;
pub const GEN_JUMP_DIFFUSION: i32 = 7;
pub const GEN_POISSON_PROCESS: i32 = 8;
pub const GEN_CYCLIC: i32 = 9;
pub const GEN_GARCH: i32 = 10;
pub const GEN_HAWKES_PROCESS: i32 = 11;
pub const GEN_CHAOTIC_SYSTEM: i32 = 12;
pub const GEN_BOUNDED_PROCESS: i32 = 13;
pub const GEN_LEVY_PROCESS: i32 = 14;
pub const GEN_STOCHASTIC_VOLATILITY: i32 = 15;
pub const GEN_REGIME_SWITCHING: i32 = 16;
pub const GEN_STATE_SPACE: i32 = 19;
pub const GEN_FBM: i32 = 20;
pub const GEN_GAUSSIAN_PROCESS: i32 = 21;
pub const GEN_INTERMITTENT_DEMAND: i32 = 22;
pub const GEN_ENERGY_LOAD: i32 = 24;
pub const GEN_VITAL_SIGNS: i32 = 25;
pub const GEN_CLICKSTREAM: i32 = 27;
pub const GEN_TSI: i32 = 28;
pub const GEN_TCM: i32 = 29;

/// Per-series result returned from batch generation.
#[derive(Debug)]
pub struct SeriesResult {
    pub values: Vec<f64>,
    pub cp_indices: Vec<i64>,
    pub anom_indices: Vec<i64>,
    pub miss_indices: Vec<i64>,
}

/// Configuration for pattern injection, shared across all series in a batch.
///
/// String fields accept: `changepoint_type` ("level" | "trend" | "variance"),
/// `missing_pattern` ("random" | "block" | "seasonal"),
/// `anomaly_types` (subset of "spike", "dip", "level_shift").
#[derive(Debug, Clone)]
pub struct PatternInjectionConfig {
    pub changepoints: bool,
    pub num_changepoints: i32,
    pub changepoint_type: String,
    pub changepoint_locations: Vec<f64>,
    pub changepoint_level_changes: Vec<f64>,
    pub changepoint_trend_changes: Vec<f64>,
    pub changepoint_variance_changes: Vec<f64>,
    pub anomalies: bool,
    pub anomaly_types: Vec<String>,
    pub anomaly_fraction: f64,
    pub spike_magnitude: f64,
    pub dip_magnitude: f64,
    pub level_shift_magnitude: f64,
    pub level_shift_duration: i32,
    pub missing_data: bool,
    pub missing_pattern: String,
    pub missing_rate: f64,
    pub missing_block_size: i32,
    pub missing_seasonal_period: i32,
}

/// Specification for one generator in a multi-batch call.
#[derive(Debug)]
pub struct GeneratorSpec {
    pub gen_type: i32,
    pub scalar_params: Vec<f64>,
    pub array_params: Vec<Vec<f64>>,
    pub lengths: Vec<i32>,
    pub gen_seeds: Vec<u64>,
    pub pi_seeds: Vec<u64>,
    pub pi_config: PatternInjectionConfig,
}

/// Validate that `sp` and `ap` have the required minimum lengths for a generator.
macro_rules! check_params {
    ($sp:expr, $n_sp:expr, $ap:expr, 0, $name:expr) => {
        if $sp.len() < $n_sp {
            return Err(format!(
                "{}: expected >= {} scalar params, got {}",
                $name,
                $n_sp,
                $sp.len()
            ));
        }
    };
    ($sp:expr, $n_sp:expr, $ap:expr, $n_ap:expr, $name:expr) => {
        if $sp.len() < $n_sp {
            return Err(format!(
                "{}: expected >= {} scalar params, got {}",
                $name,
                $n_sp,
                $sp.len()
            ));
        }
        if $ap.len() < $n_ap {
            return Err(format!(
                "{}: expected >= {} array params, got {}",
                $name,
                $n_ap,
                $ap.len()
            ));
        }
    };
}

/// Dispatch to the correct generator based on type ID.
///
/// Returns `Err` if `gen_type` is unknown or if `sp`/`ap` have insufficient length.
fn dispatch_generator(
    gen_type: i32,
    out: &mut [f64],
    sp: &[f64],
    ap: &[Vec<f64>],
    seed: u64,
) -> Result<(), String> {
    match gen_type {
        GEN_RANDOM_WALK => {
            // sp: [drift, volatility, start_value, innov_dist, innov_param]
            check_params!(sp, 5, ap, 0, "random_walk");
            statistical::random_walk(out, sp[0], sp[1], sp[2], seed, sp[3] as i32, sp[4]);
        }
        GEN_SEASONAL => {
            // sp: [period, amplitude, trend, noise_level, base_level,
            //      innov_dist, innov_param]
            check_params!(sp, 7, ap, 0, "seasonal");
            statistical::seasonal(
                out,
                sp[0] as i32,
                sp[1],
                sp[2],
                sp[3],
                sp[4],
                seed,
                sp[5] as i32,
                sp[6],
            );
        }
        GEN_SARIMA => {
            // sp: [d, d_seasonal, seasonal_period, mean, drift_val, noise_std,
            //      burn_in, innov_dist, innov_param]
            // ap: [full_ar_poly, full_ma_poly]
            check_params!(sp, 9, ap, 2, "sarima");
            statistical::sarima(
                out,
                &ap[0],
                &ap[1],
                sp[0] as i32,
                sp[1] as i32,
                sp[2] as i32,
                sp[3],
                sp[4],
                sp[5],
                sp[6] as i32,
                seed,
                sp[7] as i32,
                sp[8],
            );
        }
        GEN_ETS => {
            // sp: [error_type, trend_type, seasonal_type, seasonal_period,
            //      level, trend_init, alpha, beta_param, gamma, phi,
            //      damped, noise_std, innov_dist, innov_param]
            // ap: [seasonal_init]
            check_params!(sp, 14, ap, 1, "ets");
            statistical::ets(
                out,
                sp[0] as i32,
                sp[1] as i32,
                sp[2] as i32,
                sp[3] as i32,
                sp[4],
                sp[5],
                &ap[0],
                sp[6],
                sp[7],
                sp[8],
                sp[9],
                sp[10] != 0.0,
                sp[11],
                seed,
                sp[12] as i32,
                sp[13],
            );
        }
        GEN_INAR => {
            // sp: [p, innov_type, innov_mean, innov_dispersion]
            // ap: [alpha_arr]
            check_params!(sp, 4, ap, 1, "inar");
            statistical::inar(out, sp[0] as i32, &ap[0], sp[1] as i32, sp[2], sp[3], seed);
        }
        GEN_ORNSTEIN_UHLENBECK => {
            // sp: [theta, mu, sigma, initial_value, dt, innov_dist, innov_param]
            check_params!(sp, 7, ap, 0, "ornstein_uhlenbeck");
            stochastic::ornstein_uhlenbeck(
                out,
                sp[0],
                sp[1],
                sp[2],
                sp[3],
                sp[4],
                seed,
                sp[5] as i32,
                sp[6],
            );
        }
        GEN_GEOMETRIC_BROWNIAN_MOTION => {
            // sp: [mu, sigma, initial_value, dt, innov_dist, innov_param]
            check_params!(sp, 6, ap, 0, "geometric_brownian_motion");
            stochastic::geometric_brownian_motion(
                out,
                sp[0],
                sp[1],
                sp[2],
                sp[3],
                seed,
                sp[4] as i32,
                sp[5],
            );
        }
        GEN_JUMP_DIFFUSION => {
            // sp: [mu, sigma, lambda_jump, jump_mean, jump_std,
            //      initial_value, dt, innov_dist, innov_param]
            check_params!(sp, 9, ap, 0, "jump_diffusion");
            stochastic::jump_diffusion(
                out,
                sp[0],
                sp[1],
                sp[2],
                sp[3],
                sp[4],
                sp[5],
                sp[6],
                seed,
                sp[7] as i32,
                sp[8],
            );
        }
        GEN_POISSON_PROCESS => {
            // sp: [lambda_rate, cumulative]
            check_params!(sp, 2, ap, 0, "poisson_process");
            stochastic::poisson_process(out, sp[0], sp[1] != 0.0, seed);
        }
        GEN_CYCLIC => {
            // sp: [base_level, trend, period_mean, period_std,
            //      amp_mean, amp_std, num_cycles, noise_std]
            check_params!(sp, 8, ap, 0, "cyclic");
            stochastic::cyclic(
                out,
                sp[0],
                sp[1],
                sp[2],
                sp[3],
                sp[4],
                sp[5],
                sp[6] as i32,
                sp[7],
                seed,
            );
        }
        GEN_GARCH => {
            // sp: [p, q, omega, mu, initial_variance, innov_dist, innov_param]
            // ap: [alpha_arr, beta_arr]
            check_params!(sp, 7, ap, 2, "garch");
            stochastic::garch(
                out,
                sp[0] as i32,
                sp[1] as i32,
                sp[2],
                &ap[0],
                &ap[1],
                sp[3],
                sp[4],
                seed,
                sp[5] as i32,
                sp[6],
            );
        }
        GEN_HAWKES_PROCESS => {
            // sp: [baseline, excitation, decay, kernel_type,
            //      power_law_exp, output_type, max_events]
            check_params!(sp, 7, ap, 0, "hawkes_process");
            stochastic::hawkes_process(
                out,
                sp[0],
                sp[1],
                sp[2],
                sp[3] as i32,
                sp[4],
                sp[5] as i32,
                sp[6] as i32,
                seed,
            );
        }
        GEN_CHAOTIC_SYSTEM => {
            // sp: [system_id, sigma, rho, beta, dt, logistic_r,
            //      mg_beta, mg_gamma, mg_n, mg_tau, obs_noise, init_perturb]
            check_params!(sp, 12, ap, 0, "chaotic_system");
            stochastic::chaotic_system(
                out,
                sp[0] as i32,
                sp[1],
                sp[2],
                sp[3],
                sp[4],
                sp[5],
                sp[6],
                sp[7],
                sp[8],
                sp[9] as i32,
                sp[10],
                sp[11],
                seed,
            );
        }
        GEN_BOUNDED_PROCESS => {
            // sp: [model_id, phi, omega, kappa, sigma_param, initial_value,
            //      lower, upper]
            check_params!(sp, 8, ap, 0, "bounded_process");
            stochastic::bounded_process(
                out,
                sp[0] as i32,
                sp[1],
                sp[2],
                sp[3],
                sp[4],
                sp[5],
                sp[6],
                sp[7],
                seed,
            );
        }
        GEN_LEVY_PROCESS => {
            // sp: [alpha, beta_skew, scale, location, cumulative, initial_value]
            check_params!(sp, 6, ap, 0, "levy_process");
            stochastic::levy_process(out, sp[0], sp[1], sp[2], sp[3], sp[4] != 0.0, sp[5], seed);
        }
        GEN_STOCHASTIC_VOLATILITY => {
            // sp: [model_type, initial_price, initial_vol, drift, mean_vol,
            //      vol_mr, vol_of_vol, correlation, beta_param, dt,
            //      output_type, innov_dist, innov_param]
            check_params!(sp, 13, ap, 0, "stochastic_volatility");
            volatility::stochastic_volatility(
                out,
                sp[0] as i32,
                sp[1],
                sp[2],
                sp[3],
                sp[4],
                sp[5],
                sp[6],
                sp[7],
                sp[8],
                sp[9],
                sp[10] as i32,
                seed,
                sp[11] as i32,
                sp[12],
            );
        }
        GEN_REGIME_SWITCHING => {
            // sp: [n_regimes, initial_regime (< 0 => draw from stationary),
            //      innov_dist, innov_param]
            // ap: [regime_means, regime_variances, regime_ar_coeffs,
            //      transition_matrix, stationary_probs]
            check_params!(sp, 4, ap, 5, "regime_switching");
            volatility::regime_switching(
                out,
                sp[0] as i32,
                &ap[0],
                &ap[1],
                &ap[2],
                &ap[3],
                &ap[4],
                sp[1] as i32,
                seed,
                sp[2] as i32,
                sp[3],
            );
        }
        GEN_STATE_SPACE => {
            // sp: [state_dim, obs_dim, innov_dist, innov_param]
            // ap: [f_mat, h_mat, q_mat, r_mat, initial_state, initial_state_cov]
            check_params!(sp, 4, ap, 6, "state_space");
            multivariate::state_space(
                out,
                sp[0] as usize,
                sp[1] as usize,
                &ap[0],
                &ap[1],
                &ap[2],
                &ap[3],
                &ap[4],
                &ap[5],
                seed,
                sp[2] as i32,
                sp[3],
            );
        }
        GEN_FBM => {
            // sp: [hurst, sigma, initial_value, cumulative, method]
            check_params!(sp, 5, ap, 0, "fbm");
            multivariate::fbm(out, sp[0], sp[1], sp[2], sp[3] != 0.0, sp[4] as i32, seed);
        }
        GEN_GAUSSIAN_PROCESS => {
            // sp: [kernel_id, length_scale, amplitude, period, mean, noise_variance]
            check_params!(sp, 6, ap, 0, "gaussian_process");
            multivariate::gaussian_process(
                out,
                sp[0] as i32,
                sp[1],
                sp[2],
                sp[3],
                sp[4],
                sp[5],
                seed,
            );
        }
        GEN_INTERMITTENT_DEMAND => {
            // sp: [demand_prob, demand_dist, demand_mean, demand_std,
            //      pattern, cluster_size, seasonal_period, seasonal_peak_prob,
            //      min_demand]
            check_params!(sp, 9, ap, 0, "intermittent_demand");
            domain::intermittent_demand(
                out,
                sp[0],
                sp[1] as i32,
                sp[2],
                sp[3],
                sp[4] as i32,
                sp[5] as i32,
                sp[6] as i32,
                sp[7],
                sp[8],
                seed,
            );
        }
        GEN_ENERGY_LOAD => {
            // sp: [base_load, load_type, daily_pattern, daily_amplitude,
            //      weekly_pattern, weekly_amplitude, yearly_pattern,
            //      yearly_amplitude, temperature_sensitive,
            //      temperature_sensitivity, base_temperature,
            //      morning_peak_hour, evening_peak_hour, peak_amplitude,
            //      holiday_effect, extreme_weather_prob, extreme_weather_impact,
            //      noise_std, step_hours]
            // ap: [holiday_days_f64] (optional, may be absent)
            check_params!(sp, 19, ap, 0, "energy_load");
            let holiday_days_i32: Vec<i32> = if !ap.is_empty() {
                ap[0].iter().map(|&x| x as i32).collect()
            } else {
                vec![]
            };
            domain::energy_load(
                out,
                sp[0],
                sp[1] as i32,
                sp[2] != 0.0,
                sp[3],
                sp[4] != 0.0,
                sp[5],
                sp[6] != 0.0,
                sp[7],
                sp[8] != 0.0,
                sp[9],
                sp[10],
                sp[11] as i32,
                sp[12] as i32,
                sp[13],
                sp[14],
                &holiday_days_i32,
                sp[15],
                sp[16],
                sp[17],
                sp[18],
                seed,
            );
        }
        GEN_VITAL_SIGNS => {
            // sp: [baseline_mean, baseline_std, min_val, max_val,
            //      include_circadian, circadian_magnitude, include_hrv,
            //      include_events, event_probability, vital_sign_type]
            check_params!(sp, 10, ap, 0, "vital_signs");
            domain::vital_signs(
                out,
                sp[0],
                sp[1],
                sp[2],
                sp[3],
                sp[4] != 0.0,
                sp[5],
                sp[6] != 0.0,
                sp[7] != 0.0,
                sp[8],
                sp[9] as i32,
                seed,
            );
        }
        GEN_CLICKSTREAM => {
            // sp: [base_sessions, conversion_mult, bounce_mult, depth_mult,
            //      seasonality_amp, conversion_rate, bounce_rate,
            //      avg_session_depth, include_seasonality, include_bots,
            //      bot_fraction, output_type]
            check_params!(sp, 12, ap, 0, "clickstream");
            domain::clickstream(
                out,
                sp[0],
                sp[1],
                sp[2],
                sp[3],
                sp[4],
                sp[5],
                sp[6],
                sp[7],
                sp[8] != 0.0,
                sp[9] != 0.0,
                sp[10],
                sp[11] as i32,
                seed,
            );
        }
        GEN_TSI => {
            // sp: [trend_slope_lo, trend_slope_hi, trend_growth_lo,
            //      trend_growth_hi, n_breakpoints_lo, n_breakpoints_hi,
            //      level_lo, level_hi, n_seasonal_lo, n_seasonal_hi,
            //      seasonal_amp_lo, seasonal_amp_hi,
            //      amplitude_modulation_prob, harmonics_prob,
            //      noise_scale_lo, noise_scale_hi, ar1_phi_lo, ar1_phi_hi,
            //      tail_df_lo, tail_df_hi, multiplicative_prob,
            //      scale_lo, scale_hi]
            // ap: [trend_type_ids, seasonal_periods, irregular_type_ids]
            check_params!(sp, 23, ap, 3, "tsi");
            tsi::tsi(out, sp, &ap[0], &ap[1], &ap[2], seed);
        }
        GEN_TCM => {
            // sp: [n_vars_lo, n_vars_hi, max_lag_lo, max_lag_hi,
            //      edge_prob_lo, edge_prob_hi, coef_lo, coef_hi,
            //      stability_margin, clamp_threshold,
            //      noise_scale_lo, noise_scale_hi, heteroscedastic_prob]
            // ap: [edge_kind_ids, noise_type_ids]
            check_params!(sp, 13, ap, 2, "tcm");
            tcm::tcm(out, sp, &ap[0], &ap[1], seed);
        }
        _ => return Err(format!("unknown generator type: {gen_type}")),
    }
    Ok(())
}

/// Apply pattern injection (changepoints, anomalies, missingness) to a single series.
#[inline]
fn apply_pattern_injection(
    values: &mut [f64],
    pi_config: &PatternInjectionConfig,
    cp_seed: u64,
    anom_seed: u64,
    miss_seed: u64,
) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let cp_indices = if pi_config.changepoints {
        let r = pi::add_changepoints(
            values,
            cp_seed,
            pi_config.num_changepoints,
            &pi_config.changepoint_locations,
            &pi_config.changepoint_type,
            &pi_config.changepoint_level_changes,
            &pi_config.changepoint_trend_changes,
            &pi_config.changepoint_variance_changes,
        );
        r.changepoint_indices
    } else {
        vec![]
    };

    let anom_indices = if pi_config.anomalies {
        let r = pi::add_anomalies(
            values,
            anom_seed,
            &pi_config.anomaly_types,
            pi_config.anomaly_fraction,
            pi_config.spike_magnitude,
            pi_config.dip_magnitude,
            pi_config.level_shift_magnitude,
            pi_config.level_shift_duration,
        );
        r.anomaly_indices
    } else {
        vec![]
    };

    let miss_indices = if pi_config.missing_data {
        let r = pi::add_missingness(
            values,
            miss_seed,
            &pi_config.missing_pattern,
            pi_config.missing_rate,
            pi_config.missing_block_size,
            pi_config.missing_seasonal_period,
        );
        r.missing_indices
    } else {
        vec![]
    };

    (cp_indices, anom_indices, miss_indices)
}

/// Generate multiple series of a single generator type in parallel via rayon.
///
/// Each series gets its own pre-sampled seed, making them fully independent and
/// safe to generate in parallel. Pattern injection is applied per-series inside
/// the rayon scope.
///
/// When `n_workers > 0`, a cached rayon thread pool with that many threads is
/// used (created once, reused on subsequent calls with the same count).
/// When `n_workers == 0`, the global rayon pool is used (defaults to all
/// available cores).
///
/// # Errors
///
/// Returns `Err` if `gen_type` is unknown, `scalar_params`/`array_params` have
/// insufficient length, or `pi_seeds.len() < lengths.len() * 3`.
pub fn generate_batch(
    gen_type: i32,
    scalar_params: &[f64],
    array_params: &[Vec<f64>],
    lengths: &[i32],
    gen_seeds: &[u64],
    pi_seeds: &[u64],
    pi_config: &PatternInjectionConfig,
    n_workers: usize,
) -> Result<Vec<SeriesResult>, String> {
    let n_series = lengths.len();
    if pi_seeds.len() < n_series * 3 {
        return Err(format!(
            "pi_seeds length {} < {} (n_series × 3)",
            pi_seeds.len(),
            n_series * 3,
        ));
    }

    let work = || {
        lengths
            .par_iter()
            .zip(gen_seeds.par_iter())
            .zip(pi_seeds.par_chunks(3))
            .map(|((&len, &gen_seed), pi_chunk)| {
                let l = len as usize;
                let mut values = vec![0.0f64; l];

                dispatch_generator(gen_type, &mut values, scalar_params, array_params, gen_seed)?;

                let (cp_indices, anom_indices, miss_indices) = apply_pattern_injection(
                    &mut values,
                    pi_config,
                    pi_chunk[0],
                    pi_chunk[1],
                    pi_chunk[2],
                );

                Ok(SeriesResult {
                    values,
                    cp_indices,
                    anom_indices,
                    miss_indices,
                })
            })
            .collect()
    };

    if n_workers > 0 {
        get_or_create_pool(n_workers)?.install(work)
    } else {
        work()
    }
}

/// Generate series from multiple generators in parallel via rayon.
///
/// Flattens all series from all generators into one rayon work pool so that
/// work-stealing balances heterogeneous generator workloads naturally.
///
/// When `n_workers > 0`, a cached rayon thread pool with that many threads is
/// used (see `get_or_create_pool`). When `n_workers == 0`, the global rayon
/// pool is used (defaults to all available cores).
///
/// # Errors
///
/// Returns `Err` if any generator spec has an unknown `gen_type` or insufficient
/// `scalar_params`/`array_params`.
pub fn generate_multi_batch(
    specs: &[GeneratorSpec],
    n_workers: usize,
) -> Result<Vec<Vec<SeriesResult>>, String> {
    // Build flat task list: (spec_index, series_index)
    let total_tasks: usize = specs.iter().map(|s| s.lengths.len()).sum();
    let mut tasks: Vec<(usize, usize)> = Vec::with_capacity(total_tasks);
    for (gi, spec) in specs.iter().enumerate() {
        for si in 0..spec.lengths.len() {
            tasks.push((gi, si));
        }
    }

    let work = || {
        tasks
            .into_par_iter()
            .map(|(gi, si)| {
                let spec = &specs[gi];
                let len = spec.lengths[si] as usize;
                let mut values = vec![0.0f64; len];

                dispatch_generator(
                    spec.gen_type,
                    &mut values,
                    &spec.scalar_params,
                    &spec.array_params,
                    spec.gen_seeds[si],
                )?;

                let (cp_indices, anom_indices, miss_indices) = apply_pattern_injection(
                    &mut values,
                    &spec.pi_config,
                    spec.pi_seeds[si * 3],
                    spec.pi_seeds[si * 3 + 1],
                    spec.pi_seeds[si * 3 + 2],
                );

                Ok((
                    gi,
                    si,
                    SeriesResult {
                        values,
                        cp_indices,
                        anom_indices,
                        miss_indices,
                    },
                ))
            })
            .collect::<Result<Vec<_>, String>>()
    };

    let flat_results = if n_workers > 0 {
        get_or_create_pool(n_workers)?.install(work)?
    } else {
        work()?
    };

    // Regroup by generator index, pre-allocating each inner Vec
    let mut grouped: Vec<Vec<(usize, SeriesResult)>> = specs
        .iter()
        .map(|s| Vec::with_capacity(s.lengths.len()))
        .collect();
    for (gi, si, result) in flat_results {
        grouped[gi].push((si, result));
    }
    // Sort each group by series index and extract results
    Ok(grouped
        .into_iter()
        .map(|mut group| {
            group.sort_by_key(|(si, _)| *si);
            group.into_iter().map(|(_, r)| r).collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pi_config() -> PatternInjectionConfig {
        PatternInjectionConfig {
            changepoints: false,
            num_changepoints: 0,
            changepoint_type: "level".to_string(),
            changepoint_locations: vec![],
            changepoint_level_changes: vec![],
            changepoint_trend_changes: vec![],
            changepoint_variance_changes: vec![],
            anomalies: false,
            anomaly_types: vec![],
            anomaly_fraction: 0.0,
            spike_magnitude: 0.0,
            dip_magnitude: 0.0,
            level_shift_magnitude: 0.0,
            level_shift_duration: 0,
            missing_data: false,
            missing_pattern: "random".to_string(),
            missing_rate: 0.0,
            missing_block_size: 1,
            missing_seasonal_period: 1,
        }
    }

    #[test]
    fn test_generate_batch_random_walk() {
        let sp = vec![0.0, 1.0, 0.0, 0.0, 0.0];
        let lengths = vec![100, 100, 100];
        let gen_seeds = vec![42, 43, 44];
        let pi_seeds = vec![0; 9];
        let pi = default_pi_config();

        let results = generate_batch(
            GEN_RANDOM_WALK,
            &sp,
            &[],
            &lengths,
            &gen_seeds,
            &pi_seeds,
            &pi,
            0,
        )
        .unwrap();
        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.values.len(), 100);
            assert!(r.values.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_generate_batch_deterministic() {
        let sp = vec![0.0, 1.0, 0.0, 0.0, 0.0];
        let lengths = vec![50, 50];
        let gen_seeds = vec![42, 43];
        let pi_seeds = vec![0; 6];
        let pi = default_pi_config();

        let r1 = generate_batch(
            GEN_RANDOM_WALK,
            &sp,
            &[],
            &lengths,
            &gen_seeds,
            &pi_seeds,
            &pi,
            0,
        )
        .unwrap();
        let r2 = generate_batch(
            GEN_RANDOM_WALK,
            &sp,
            &[],
            &lengths,
            &gen_seeds,
            &pi_seeds,
            &pi,
            0,
        )
        .unwrap();
        assert_eq!(r1[0].values, r2[0].values);
        assert_eq!(r1[1].values, r2[1].values);
    }

    #[test]
    fn test_generate_batch_with_pattern_injection() {
        let sp = vec![0.0, 1.0, 0.0, 0.0, 0.0];
        let lengths = vec![100];
        let gen_seeds = vec![42];
        let pi_seeds = vec![100, 200, 300];
        let pi = PatternInjectionConfig {
            changepoints: true,
            num_changepoints: 2,
            changepoint_type: "level".to_string(),
            anomalies: true,
            anomaly_types: vec!["spike".to_string()],
            anomaly_fraction: 0.05,
            spike_magnitude: 10.0,
            missing_data: true,
            missing_pattern: "random".to_string(),
            missing_rate: 0.1,
            ..default_pi_config()
        };

        let results = generate_batch(
            GEN_RANDOM_WALK,
            &sp,
            &[],
            &lengths,
            &gen_seeds,
            &pi_seeds,
            &pi,
            0,
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert!(!results[0].cp_indices.is_empty());
        assert!(!results[0].anom_indices.is_empty());
        assert!(!results[0].miss_indices.is_empty());
    }

    #[test]
    fn test_generate_batch_variable_lengths() {
        let sp = vec![0.0, 1.0, 0.0, 0.0, 0.0];
        let lengths = vec![50, 100, 200];
        let gen_seeds = vec![42, 43, 44];
        let pi_seeds = vec![0; 9];
        let pi = default_pi_config();

        let results = generate_batch(
            GEN_RANDOM_WALK,
            &sp,
            &[],
            &lengths,
            &gen_seeds,
            &pi_seeds,
            &pi,
            0,
        )
        .unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].values.len(), 50);
        assert_eq!(results[1].values.len(), 100);
        assert_eq!(results[2].values.len(), 200);
    }

    #[test]
    fn test_generate_multi_batch() {
        let spec1 = GeneratorSpec {
            gen_type: GEN_RANDOM_WALK,
            scalar_params: vec![0.0, 1.0, 0.0, 0.0, 0.0],
            array_params: vec![],
            lengths: vec![50, 50],
            gen_seeds: vec![42, 43],
            pi_seeds: vec![0; 6],
            pi_config: default_pi_config(),
        };
        let spec2 = GeneratorSpec {
            gen_type: GEN_SEASONAL,
            scalar_params: vec![24.0, 10.0, 0.0, 1.0, 50.0, 0.0, 0.0],
            array_params: vec![],
            lengths: vec![100, 100, 100],
            gen_seeds: vec![44, 45, 46],
            pi_seeds: vec![0; 9],
            pi_config: default_pi_config(),
        };

        let results = generate_multi_batch(&[spec1, spec2], 0).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 3);
        assert_eq!(results[0][0].values.len(), 50);
        assert_eq!(results[1][0].values.len(), 100);
    }

    #[test]
    fn test_dispatch_invalid_gen_type() {
        let mut out = vec![0.0f64; 64];
        let result = dispatch_generator(999, &mut out, &[], &[], 42);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("unknown generator type"),
            "error should mention unknown generator type"
        );
    }

    #[test]
    fn test_dispatch_insufficient_scalar_params() {
        let mut out = vec![0.0f64; 64];
        // random_walk requires 5 scalar params, pass only 2
        let result = dispatch_generator(GEN_RANDOM_WALK, &mut out, &[0.0, 1.0], &[], 42);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("scalar params"),
            "error should mention scalar params"
        );
    }

    #[test]
    fn test_dispatch_insufficient_array_params() {
        let mut out = vec![0.0f64; 64];
        // sarima requires 2 array params, pass 0
        let sp = vec![0.0; 9];
        let result = dispatch_generator(GEN_SARIMA, &mut out, &sp, &[], 42);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("array params"),
            "error should mention array params"
        );
    }

    #[test]
    fn test_dispatch_all_generators_without_array_params() {
        let test_cases: Vec<(i32, Vec<f64>, &str)> = vec![
            (
                GEN_RANDOM_WALK,
                vec![0.0, 1.0, 0.0, 0.0, 0.0],
                "random_walk",
            ),
            (
                GEN_SEASONAL,
                vec![24.0, 10.0, 0.0, 1.0, 50.0, 0.0, 0.0],
                "seasonal",
            ),
            (
                GEN_ORNSTEIN_UHLENBECK,
                vec![1.0, 0.0, 0.5, 0.0, 0.01, 0.0, 0.0],
                "ornstein_uhlenbeck",
            ),
            (
                GEN_GEOMETRIC_BROWNIAN_MOTION,
                vec![0.05, 0.2, 100.0, 0.01, 0.0, 0.0],
                "geometric_brownian_motion",
            ),
            (
                GEN_JUMP_DIFFUSION,
                vec![0.05, 0.2, 0.1, 0.0, 0.1, 100.0, 0.01, 0.0, 0.0],
                "jump_diffusion",
            ),
            (GEN_POISSON_PROCESS, vec![2.0, 0.0], "poisson_process"),
            (
                GEN_CYCLIC,
                vec![50.0, 0.0, 20.0, 2.0, 10.0, 1.0, 3.0, 1.0],
                "cyclic",
            ),
            (
                GEN_HAWKES_PROCESS,
                vec![1.0, 0.5, 1.0, 0.0, 2.0, 0.0, 1000.0],
                "hawkes_process",
            ),
            (
                GEN_CHAOTIC_SYSTEM,
                vec![
                    0.0,
                    10.0,
                    28.0,
                    8.0 / 3.0,
                    0.01,
                    3.9,
                    0.2,
                    0.1,
                    10.0,
                    17.0,
                    0.0,
                    0.0,
                ],
                "chaotic_system",
            ),
            (
                GEN_BOUNDED_PROCESS,
                vec![0.0, 0.5, 0.2, 10.0, 0.1, 0.5, 0.0, 1.0],
                "bounded_process",
            ),
            (
                GEN_LEVY_PROCESS,
                vec![1.5, 0.0, 1.0, 0.0, 0.0, 0.0],
                "levy_process",
            ),
            (
                GEN_STOCHASTIC_VOLATILITY,
                vec![
                    0.0, 100.0, 0.2, 0.05, 0.2, 2.0, 0.3, -0.7, 0.5, 0.01, 0.0, 0.0, 0.0,
                ],
                "stochastic_volatility",
            ),
            (GEN_FBM, vec![0.7, 1.0, 0.0, 1.0, 2.0], "fbm"),
            (
                GEN_GAUSSIAN_PROCESS,
                vec![0.0, 10.0, 1.0, 1.0, 0.0, 0.01],
                "gaussian_process",
            ),
            (
                GEN_INTERMITTENT_DEMAND,
                vec![0.3, 0.0, 10.0, 2.0, 0.0, 3.0, 7.0, 0.8, 1.0],
                "intermittent_demand",
            ),
            (
                GEN_ENERGY_LOAD,
                vec![
                    1000.0, 0.0, 1.0, 200.0, 1.0, 100.0, 0.0, 0.0, 0.0, 0.0, 20.0, 8.0, 18.0,
                    150.0, 0.9, 0.01, 300.0, 50.0, 1.0,
                ],
                "energy_load",
            ),
            (
                GEN_VITAL_SIGNS,
                vec![75.0, 5.0, 40.0, 200.0, 1.0, 3.0, 1.0, 0.0, 0.01, 0.0],
                "vital_signs",
            ),
            (
                GEN_CLICKSTREAM,
                vec![
                    1000.0, 0.03, 0.4, 3.5, 0.3, 0.03, 0.4, 3.5, 1.0, 0.0, 0.05, 0.0,
                ],
                "clickstream",
            ),
        ];

        for (gen_type, sp, name) in &test_cases {
            let mut out = vec![0.0f64; 64];
            let result = dispatch_generator(*gen_type, &mut out, sp, &[], 42);
            assert!(result.is_ok(), "{name} dispatch failed: {:?}", result.err());
            assert!(
                out.iter().all(|v| v.is_finite()),
                "{name} produced non-finite values"
            );
        }
    }

    #[test]
    fn test_dispatch_generators_with_array_params() {
        type DispatchCase<'a> = (i32, Vec<f64>, Vec<Vec<f64>>, &'a str);
        let test_cases: Vec<DispatchCase<'_>> = vec![
            (
                GEN_SARIMA,
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 50.0, 0.0, 0.0],
                vec![vec![1.0, -0.5], vec![1.0]],
                "sarima",
            ),
            (
                GEN_ETS,
                vec![
                    0.0, 0.0, 0.0, 1.0, 100.0, 0.0, 0.3, 0.1, 0.1, 0.98, 0.0, 1.0, 0.0, 0.0,
                ],
                vec![vec![0.0]],
                "ets",
            ),
            (GEN_INAR, vec![1.0, 0.0, 5.0, 1.0], vec![vec![0.5]], "inar"),
            (
                GEN_GARCH,
                vec![1.0, 1.0, 0.01, 0.0, 0.01, 0.0, 0.0],
                vec![vec![0.1], vec![0.8]],
                "garch",
            ),
            (
                GEN_REGIME_SWITCHING,
                vec![2.0, -1.0, 0.0, 0.0],
                vec![
                    vec![0.0, 1.0],
                    vec![1.0, 2.0],
                    vec![0.5, 0.5],
                    vec![0.95, 0.05, 0.1, 0.9],
                    vec![2.0 / 3.0, 1.0 / 3.0],
                ],
                "regime_switching",
            ),
            (
                GEN_TSI,
                vec![
                    -8.0, 8.0, 1.0, 4.0, 1.0, 3.0, -10.0, 10.0, 0.0, 3.0, 0.2, 3.0, 0.4, 0.4, 0.5,
                    12.0, 0.3, 0.95, 2.5, 12.0, 0.3, 0.1, 100.0,
                ],
                vec![
                    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    vec![7.0, 12.0, 24.0, 365.25],
                    vec![0.0, 1.0, 2.0, 3.0, 4.0],
                ],
                "tsi",
            ),
            (
                GEN_TCM,
                vec![
                    1.0, 5.0, 1.0, 24.0, 0.05, 0.3, 0.1, 0.8, 0.95, 1e6, 0.5, 2.0, 0.2,
                ],
                vec![vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0, 2.0]],
                "tcm",
            ),
            (
                GEN_STATE_SPACE,
                vec![2.0, 1.0, 0.0, 0.0],
                vec![
                    vec![1.0, 1.0, 0.0, 1.0],
                    vec![1.0, 0.0],
                    vec![0.1, 0.0, 0.0, 0.1],
                    vec![1.0],
                    vec![0.0, 0.0],
                    vec![1.0, 0.0, 0.0, 1.0],
                ],
                "state_space",
            ),
        ];

        for (gen_type, sp, ap, name) in &test_cases {
            let mut out = vec![0.0f64; 100];
            let result = dispatch_generator(*gen_type, &mut out, sp, ap, 42);
            assert!(result.is_ok(), "{name} dispatch failed: {:?}", result.err());
            assert!(
                out.iter().all(|v| v.is_finite()),
                "{name} produced non-finite values"
            );
        }
    }

    #[test]
    fn test_generate_batch_returns_error_on_bad_type() {
        let pi = default_pi_config();
        let result = generate_batch(999, &[], &[], &[100], &[42], &[0, 0, 0], &pi, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_multi_batch_returns_error_on_bad_type() {
        let spec = GeneratorSpec {
            gen_type: 999,
            scalar_params: vec![],
            array_params: vec![],
            lengths: vec![100],
            gen_seeds: vec![42],
            pi_seeds: vec![0; 3],
            pi_config: default_pi_config(),
        };
        let result = generate_multi_batch(&[spec], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_batch_custom_thread_pool() {
        let sp = vec![0.0, 1.0, 0.0, 0.0, 0.0];
        let lengths = vec![100, 100, 100, 100];
        let gen_seeds = vec![42, 43, 44, 45];
        let pi_seeds = vec![0; 12];
        let pi = default_pi_config();

        // n_workers=2: use a custom 2-thread pool
        let results = generate_batch(
            GEN_RANDOM_WALK,
            &sp,
            &[],
            &lengths,
            &gen_seeds,
            &pi_seeds,
            &pi,
            2,
        )
        .unwrap();
        assert_eq!(results.len(), 4);

        // Results should be identical to the global pool (deterministic per seed)
        let results_global = generate_batch(
            GEN_RANDOM_WALK,
            &sp,
            &[],
            &lengths,
            &gen_seeds,
            &pi_seeds,
            &pi,
            0,
        )
        .unwrap();
        for (a, b) in results.iter().zip(results_global.iter()) {
            assert_eq!(a.values, b.values);
        }
    }

    #[test]
    fn test_generate_multi_batch_custom_thread_pool() {
        let spec = GeneratorSpec {
            gen_type: GEN_RANDOM_WALK,
            scalar_params: vec![0.0, 1.0, 0.0, 0.0, 0.0],
            array_params: vec![],
            lengths: vec![50, 50],
            gen_seeds: vec![42, 43],
            pi_seeds: vec![0; 6],
            pi_config: default_pi_config(),
        };

        let results = generate_multi_batch(&[spec], 2).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 2);
    }
}
