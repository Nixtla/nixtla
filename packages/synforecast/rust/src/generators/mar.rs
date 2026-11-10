//! Native mixture autoregressive (MAR) generator.
//!
//! Model: Wong and Li (2000), doi:10.1111/1467-9868.00222. Sampling is inspired
//! by GRATIS (Kang et al., 2020), https://arxiv.org/abs/1903.02787; the sampler
//! is not a reproduction of that package. PACF conversion uses the inverse
//! Levinson-Durbin recursion. Full references and deviations: GENERATORS.md.
//!
//! Scalar parameters: max components, max AR order, seasonal period (zero for
//! none), Dirichlet concentration, intercept scale, innovation-scale lower
//! and upper bounds, burn-in, standardize flag, fixed-mode flag, innovation
//! distribution id, and innovation parameter. Fixed mode additionally uses
//! arrays for weights, orders, flattened coefficients, intercepts, and scales.

use crate::rng::SfRng;

const MAX_ABS: f64 = 1e8;
const MIN_STD: f64 = 1e-8;
const MAX_RETRIES: usize = 8;

#[derive(Clone)]
struct Params {
    weights: Vec<f64>,
    coefficients: Vec<Vec<f64>>,
    intercepts: Vec<f64>,
    scales: Vec<f64>,
}

fn pacf_to_ar(pacf: &[f64]) -> Vec<f64> {
    let mut coefficients = Vec::new();
    for &reflection in pacf {
        let previous = coefficients.clone();
        coefficients.resize(previous.len() + 1, 0.0);
        for i in 0..previous.len() {
            coefficients[i] = previous[i] - reflection * previous[previous.len() - 1 - i];
        }
        *coefficients.last_mut().unwrap() = reflection;
    }
    coefficients
}

fn apply_seasonal_factor(
    coefficients: &[f64],
    seasonal_period: usize,
    seasonal_phi: f64,
) -> Vec<f64> {
    let mut base = Vec::with_capacity(coefficients.len() + 1);
    base.push(1.0);
    base.extend(coefficients.iter().map(|value| -*value));
    let mut product = vec![0.0; base.len() + seasonal_period];
    for (index, &value) in base.iter().enumerate() {
        product[index] += value;
        product[index + seasonal_period] -= seasonal_phi * value;
    }
    product.into_iter().skip(1).map(|value| -value).collect()
}

fn sample_params(sp: &[f64], rng: &mut SfRng) -> Params {
    let n_components = rng.integers(1, sp[0] as i32 + 1) as usize;
    let mut weights: Vec<f64> = (0..n_components).map(|_| rng.gamma(sp[3], 1.0)).collect();
    let weight_sum = weights.iter().sum::<f64>();
    for weight in &mut weights {
        *weight /= weight_sum;
    }

    let seasonal_period = sp[2] as usize;
    let mut coefficients = Vec::with_capacity(n_components);
    for _ in 0..n_components {
        let order = rng.integers(1, sp[1] as i32 + 1) as usize;
        let pacf: Vec<f64> = (0..order).map(|_| rng.uniform(-0.9, 0.9)).collect();
        let mut component = pacf_to_ar(&pacf);
        if seasonal_period > 0 && rng.bernoulli(0.5) {
            component = apply_seasonal_factor(&component, seasonal_period, rng.uniform(-0.9, 0.9));
        }
        coefficients.push(component);
    }

    let intercepts = (0..n_components).map(|_| rng.normal(0.0, sp[4])).collect();
    let (log_lo, log_hi) = (sp[5].ln(), sp[6].ln());
    let scales = (0..n_components)
        .map(|_| {
            if log_lo == log_hi {
                sp[5]
            } else {
                rng.uniform(log_lo, log_hi).exp()
            }
        })
        .collect();
    Params {
        weights,
        coefficients,
        intercepts,
        scales,
    }
}

fn fixed_params(ap: &[Vec<f64>]) -> Result<Params, String> {
    let n_components = ap[0].len();
    if n_components == 0
        || ap[1].len() != n_components
        || ap[3].len() != n_components
        || ap[4].len() != n_components
    {
        return Err("mar: inconsistent fixed component counts".to_string());
    }
    let mut offset = 0usize;
    let mut coefficients = Vec::with_capacity(n_components);
    for &encoded_order in &ap[1] {
        let order = encoded_order as usize;
        let end = offset
            .checked_add(order)
            .ok_or_else(|| "mar: fixed AR coefficient offset overflow".to_string())?;
        if !encoded_order.is_finite()
            || encoded_order.fract() != 0.0
            || order == 0
            || end > ap[2].len()
        {
            return Err("mar: invalid fixed AR coefficient layout".to_string());
        }
        coefficients.push(ap[2][offset..end].to_vec());
        offset = end;
    }
    if offset != ap[2].len() {
        return Err("mar: unused fixed AR coefficients".to_string());
    }
    Ok(Params {
        weights: ap[0].clone(),
        coefficients,
        intercepts: ap[3].clone(),
        scales: ap[4].clone(),
    })
}

fn choose_component(weights: &[f64], rng: &mut SfRng) -> usize {
    let draw = rng.uniform01();
    let mut cumulative = 0.0;
    for (index, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if draw < cumulative {
            return index;
        }
    }
    weights.len() - 1
}

fn simulate(
    out: &mut [f64],
    params: &Params,
    burn_in: usize,
    innov_dist: i32,
    innov_param: f64,
    rng: &mut SfRng,
) {
    let max_order = params.coefficients.iter().map(Vec::len).max().unwrap_or(1);
    let mut values = vec![0.0; out.len() + burn_in + max_order];
    rng.normal_array(&mut values[..max_order], 0.0, 1.0);
    for step in max_order..values.len() {
        let component_index = choose_component(&params.weights, rng);
        let component = &params.coefficients[component_index];
        let autoregression = component
            .iter()
            .enumerate()
            .map(|(lag, coefficient)| coefficient * values[step - lag - 1])
            .sum::<f64>();
        values[step] = params.intercepts[component_index]
            + autoregression
            + rng.sample_innovation(params.scales[component_index], innov_dist, innov_param);
    }
    out.copy_from_slice(&values[max_order + burn_in..]);
}

fn population_std(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

/// Generate one finite MAR series into `out`.
pub fn mar(out: &mut [f64], sp: &[f64], ap: &[Vec<f64>], seed: u64) -> Result<(), String> {
    if sp.len() < 12 {
        return Err(format!(
            "mar: expected >= 12 scalar params, got {}",
            sp.len()
        ));
    }
    if !sp.iter().all(|value| value.is_finite()) {
        return Err("mar: scalar params must be finite".to_string());
    }
    if !(1.0..=1e6).contains(&sp[0])
        || !(1.0..=1e6).contains(&sp[1])
        || !(0.0..=1e9).contains(&sp[2])
        || sp[3] <= 0.0
        || sp[4] < 0.0
        || sp[5] <= 0.0
        || sp[6] < sp[5]
        || !(0.0..=1e9).contains(&sp[7])
    {
        return Err("mar: invalid configuration".to_string());
    }
    let fixed = if sp[9] != 0.0 {
        if ap.len() < 5 {
            return Err(format!(
                "mar: expected >= 5 array params in fixed mode, got {}",
                ap.len()
            ));
        }
        Some(fixed_params(ap)?)
    } else {
        None
    };

    let mut rng = SfRng::new(seed);
    for _ in 0..MAX_RETRIES {
        let sampled;
        let params = if let Some(params) = fixed.as_ref() {
            params
        } else {
            sampled = sample_params(sp, &mut rng);
            &sampled
        };
        simulate(out, params, sp[7] as usize, sp[10] as i32, sp[11], &mut rng);
        let std = if out.len() > 1 {
            population_std(out)
        } else {
            1.0
        };
        if out
            .iter()
            .all(|value| value.is_finite() && value.abs() < MAX_ABS)
            && (out.len() <= 1 || std > MIN_STD)
        {
            if sp[8] != 0.0 && out.len() > 1 {
                let mean = out.iter().sum::<f64>() / out.len() as f64;
                for value in out {
                    *value = (*value - mean) / std;
                }
            }
            return Ok(());
        }
    }
    if fixed.is_some() {
        return Err(format!(
            "mar: fixed configuration produced no finite, bounded, non-constant series in {MAX_RETRIES} attempts"
        ));
    }
    rng.normal_array(out, 0.0, 1.0);
    if sp[8] != 0.0 && out.len() > 1 {
        let mean = out.iter().sum::<f64>() / out.len() as f64;
        let std = population_std(out);
        for value in out {
            *value = (*value - mean) / std;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_orders_reject_overflow_and_fractional_values() {
        let mut arrays = vec![
            vec![0.5, 0.5],
            vec![1.0, f64::MAX],
            vec![0.0],
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        ];
        assert!(fixed_params(&arrays).err().unwrap().contains("overflow"));
        arrays[1] = vec![1.0, 1.5];
        assert!(fixed_params(&arrays).is_err());
    }

    #[test]
    fn pacf_and_seasonal_polynomials_match_hand_calculation() {
        let coefficients = pacf_to_ar(&[0.5, -0.25, 0.2]);
        assert_eq!(coefficients.len(), 3);
        for (got, expected) in coefficients.iter().zip([0.675, -0.375, 0.2]) {
            assert!((got - expected).abs() < 1e-12);
        }
        for (period, expected) in [
            (2, vec![0.3, 0.5, -0.21, 0.14]),
            (4, vec![0.3, -0.2, 0.0, 0.7, -0.21, 0.14]),
        ] {
            let got = apply_seasonal_factor(&[0.3, -0.2], period, 0.7);
            assert_eq!(got.len(), expected.len());
            for (got, expected) in got.iter().zip(expected) {
                assert!((got - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn recurrence_preserves_lag_order_and_discards_burn_in() {
        let mut initial = [0.0; 2];
        SfRng::new(42).normal_array(&mut initial, 0.0, 1.0);
        let a = 2.0 + 0.3 * initial[1] - 0.2 * initial[0];
        let b = 2.0 + 0.3 * a - 0.2 * initial[1];
        let c = 2.0 + 0.3 * b - 0.2 * a;
        let d = 2.0 + 0.3 * c - 0.2 * b;
        let params = Params {
            weights: vec![1.0],
            coefficients: vec![vec![0.3, -0.2]],
            intercepts: vec![2.0],
            scales: vec![0.0],
        };
        let mut out = [0.0; 2];
        simulate(&mut out, &params, 2, 0, 0.0, &mut SfRng::new(42));
        assert!((out[0] - c).abs() < 1e-12);
        assert!((out[1] - d).abs() < 1e-12);
    }

    fn random_scalars() -> Vec<f64> {
        vec![
            3.0, 5.0, 12.0, 1.0, 1.0, 0.1, 2.0, 100.0, 1.0, 0.0, 0.0, 0.0,
        ]
    }

    #[test]
    fn random_mar_is_finite_standardized_and_deterministic() {
        let mut a = vec![0.0; 256];
        let mut b = vec![0.0; 256];
        mar(&mut a, &random_scalars(), &[], 42).unwrap();
        mar(&mut b, &random_scalars(), &[], 42).unwrap();
        assert_eq!(a, b);
        assert!(a.iter().all(|value| value.is_finite()));
        assert!(a.iter().sum::<f64>().abs() < 1e-10);
        assert!((population_std(&a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn non_finite_scalars_are_rejected() {
        let mut sp = random_scalars();
        sp[4] = f64::INFINITY;
        assert!(mar(&mut [0.0; 8], &sp, &[], 1).is_err());
        sp = random_scalars();
        sp[0] = 1e12;
        assert!(mar(&mut [0.0; 8], &sp, &[], 1).is_err());
    }

    #[test]
    fn fixed_mar_errors_instead_of_falling_back() {
        let mut sp = random_scalars();
        sp[8] = 0.0;
        sp[9] = 1.0;
        let ap = vec![vec![1.0], vec![1.0], vec![0.5], vec![1e9], vec![1.0]];
        let mut out = vec![0.0; 64];
        assert!(mar(&mut out, &sp, &ap, 7).is_err());
    }

    #[test]
    fn fixed_mar_preserves_raw_moments() {
        let mut sp = random_scalars();
        sp[8] = 0.0;
        sp[9] = 1.0;
        let ap = vec![vec![1.0], vec![1.0], vec![0.5], vec![4.0], vec![0.1]];
        let mut out = vec![0.0; 1000];
        mar(&mut out, &sp, &ap, 7).unwrap();
        let mean = out.iter().sum::<f64>() / out.len() as f64;
        assert!((mean - 8.0).abs() < 0.25);
    }
}
