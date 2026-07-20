use crate::rng::SfRng;
use std::f64::consts::PI;

pub struct ChangePointResult {
    pub changepoint_indices: Vec<i64>,
}

pub fn add_changepoints(
    values: &mut [f64],
    seed: u64,
    num_changepoints: i32,
    locations: &[f64],
    changepoint_type: &str,
    level_changes_in: &[f64],
    trend_changes_in: &[f64],
    variance_changes_in: &[f64],
) -> ChangePointResult {
    let length = values.len() as i32;
    let mut rng = SfRng::new(seed);
    let n_cp = num_changepoints as usize;

    // Generate changepoint locations
    let mut locs = vec![0.0_f64; n_cp];
    if locations.is_empty() {
        for loc in locs.iter_mut() {
            *loc = rng.uniform(0.1, 0.9);
        }
        locs.sort_by(|a, b| a.total_cmp(b));
    } else {
        for (i, loc) in locs.iter_mut().enumerate() {
            *loc = if i < locations.len() {
                locations[i]
            } else {
                rng.uniform(0.1, 0.9)
            };
        }
    }

    // Convert to indices; clip so a relative location of 1.0 maps to the
    // last valid index instead of a silent out-of-bounds no-op.
    let max_idx = (length as i64 - 1).max(0);
    let cp_indices: Vec<i64> = locs
        .iter()
        .map(|&l| ((l * length as f64) as i64).clamp(0, max_idx))
        .collect();

    // Generate change sizes
    let has_level = changepoint_type == "level" || changepoint_type == "mixed";
    let has_trend = changepoint_type == "trend" || changepoint_type == "mixed";
    let has_variance = changepoint_type == "variance" || changepoint_type == "mixed";

    let mut level_changes = vec![0.0_f64; n_cp];
    let mut trend_changes = vec![0.0_f64; n_cp];
    let mut variance_changes = vec![1.0_f64; n_cp];

    if has_level {
        for (i, lc) in level_changes.iter_mut().enumerate() {
            *lc = if i < level_changes_in.len() {
                level_changes_in[i]
            } else {
                rng.uniform(-20.0, 20.0)
            };
        }
    }
    if has_trend {
        for (i, tc) in trend_changes.iter_mut().enumerate() {
            *tc = if i < trend_changes_in.len() {
                trend_changes_in[i]
            } else {
                rng.uniform(-0.5, 0.5)
            };
        }
    }
    if has_variance {
        for (i, vc) in variance_changes.iter_mut().enumerate() {
            *vc = if i < variance_changes_in.len() {
                variance_changes_in[i]
            } else {
                rng.uniform(0.5, 2.0)
            };
        }
    }

    // Apply changepoints
    for cp in 0..n_cp {
        let pos = cp_indices[cp] as usize;

        // Level change
        for v in &mut values[pos..] {
            *v += level_changes[cp];
        }

        // Trend change
        for (i, v) in values[pos..].iter_mut().enumerate() {
            *v += trend_changes[cp] * i as f64;
        }

        // Variance change: scale each point's deviation from the rolling
        // mean of the last 11 (already-modified) values, matching the Python
        // implementation in synforecast/_core.py exactly. The window mean
        // is recomputed from current values each step; an incremental sum of
        // pre-modification values diverges for factors > 1.
        if variance_changes[cp] != 1.0 {
            const WINDOW: usize = 10;
            for t in pos..values.len() {
                let win_start = t.saturating_sub(WINDOW);
                let window = &values[win_start..=t];
                let mean_val = window.iter().sum::<f64>() / window.len() as f64;
                let deviation = values[t] - mean_val;
                values[t] = mean_val + deviation * variance_changes[cp];
            }
        }
    }

    ChangePointResult {
        changepoint_indices: cp_indices,
    }
}

pub struct MissingnessResult {
    pub missing_indices: Vec<i64>,
}

pub fn add_missingness(
    values: &mut [f64],
    seed: u64,
    pattern: &str,
    missing_rate: f64,
    missing_block_size: i32,
    missing_seasonal_period: i32,
) -> MissingnessResult {
    let length = values.len() as i32;

    // Endpoint semantics (all patterns): rate 0 is a no-op, rate 1 marks
    // every point missing with exact metadata. Mirrors _core.py.
    if missing_rate <= 0.0 {
        return MissingnessResult {
            missing_indices: vec![],
        };
    }
    if missing_rate >= 1.0 {
        for v in values.iter_mut() {
            *v = f64::NAN;
        }
        return MissingnessResult {
            missing_indices: (0..length as i64).collect(),
        };
    }

    let mut rng = SfRng::new(seed);
    let estimated_missing = (length as f64 * missing_rate) as usize;
    let mut all_missing = Vec::with_capacity(estimated_missing);

    match pattern {
        "random" => {
            // Exactly floor(length * missing_rate) distinct points, sampled
            // without replacement so the NaN fraction matches the rate.
            let n_missing = (length as f64 * missing_rate) as i32;
            if n_missing > 0 {
                for idx in rng.choice(length, n_missing) {
                    values[idx as usize] = f64::NAN;
                    all_missing.push(idx as i64);
                }
            }
        }
        "block" => {
            assert!(
                missing_block_size > 0,
                "missing_block_size must be > 0 for block pattern"
            );
            let n_missing = (length as f64 * missing_rate) as i32;
            let n_blocks = (n_missing / missing_block_size).max(1);
            // Start drawn from [0, max_start] inclusive so blocks can reach
            // the end of the series; when the series is shorter than the
            // block size, a truncated block starts at 0.
            let max_start = (length - missing_block_size).max(0);
            for _ in 0..n_blocks {
                let start = rng.integers(0, max_start + 1);
                let end = (start + missing_block_size).min(length);
                for t in start..end {
                    values[t as usize] = f64::NAN;
                    all_missing.push(t as i64);
                }
            }
        }
        "seasonal" => {
            assert!(
                missing_seasonal_period > 0,
                "missing_seasonal_period must be > 0 for seasonal pattern"
            );
            for i in 0..length {
                let phase = i % missing_seasonal_period;
                let phase_prob = missing_rate
                    * (1.0 + (2.0 * PI * phase as f64 / missing_seasonal_period as f64).sin());
                if rng.uniform(0.0, 1.0) < phase_prob {
                    values[i as usize] = f64::NAN;
                    all_missing.push(i as i64);
                }
            }
        }
        _ => {}
    }

    // Sorted unique indices so the metadata matches the injected positions
    // exactly even when blocks overlap (mirrors the Python implementation).
    all_missing.sort_unstable();
    all_missing.dedup();

    MissingnessResult {
        missing_indices: all_missing,
    }
}

pub struct AnomalyResult {
    pub anomaly_indices: Vec<i64>,
}

pub fn add_anomalies(
    values: &mut [f64],
    seed: u64,
    anomaly_types: &[String],
    anomaly_fraction: f64,
    spike_magnitude: f64,
    dip_magnitude: f64,
    level_shift_magnitude: f64,
    level_shift_duration: i32,
) -> AnomalyResult {
    let length = values.len() as i32;
    let mut rng = SfRng::new(seed);
    let num_anomalies = (length as f64 * anomaly_fraction) as i32;

    let mut anomaly_locs: Vec<i64> = Vec::with_capacity(num_anomalies as usize);
    if num_anomalies > 0 {
        let n_types = anomaly_types.len() as i32;
        // Sample locations without replacement so exactly num_anomalies
        // distinct points are affected and magnitudes never stack at a
        // duplicated location (matches the Python implementation).
        anomaly_locs.extend(
            rng.choice(length, num_anomalies)
                .into_iter()
                .map(|x| x as i64),
        );

        for &loc_i64 in anomaly_locs.iter().take(num_anomalies as usize) {
            let loc = loc_i64 as usize;
            let type_idx = rng.integers(0, n_types) as usize;
            let atype = &anomaly_types[type_idx];

            match atype.as_str() {
                "spike" => {
                    values[loc] += spike_magnitude;
                }
                "dip" => {
                    values[loc] += dip_magnitude;
                }
                "level_shift" => {
                    let end = ((loc as i32 + level_shift_duration) as usize).min(values.len());
                    for v in &mut values[loc..end] {
                        *v += level_shift_magnitude;
                    }
                }
                _ => {}
            }
        }
    }

    AnomalyResult {
        anomaly_indices: anomaly_locs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_changepoints_level() {
        let mut values = vec![0.0; 100];
        let locs = [0.5]; // changepoint at index 50
        let level = [10.0];
        let trend = [];
        let variance = [];
        let result = add_changepoints(
            &mut values,
            42,
            1,
            &locs,
            "level",
            &level,
            &trend,
            &variance,
        );
        assert_eq!(result.changepoint_indices.len(), 1);
        // Before changepoint: values should be 0
        assert!((values[0]).abs() < 1e-10);
        assert!((values[49]).abs() < 1e-10);
        // After changepoint: values should be shifted by 10
        assert!((values[50] - 10.0).abs() < 1e-10);
        assert!((values[99] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_changepoints_trend() {
        let mut values = vec![0.0; 100];
        let locs = [0.5]; // changepoint at index 50
        let level = [];
        let trend = [1.0]; // slope of 1 per step
        let variance = [];
        add_changepoints(
            &mut values,
            42,
            1,
            &locs,
            "trend",
            &level,
            &trend,
            &variance,
        );
        // Before changepoint: all zeros
        assert!((values[0]).abs() < 1e-10);
        // After changepoint: linear increase
        assert!((values[51] - 1.0).abs() < 1e-10); // 1 step after cp
        assert!((values[60] - 10.0).abs() < 1e-10); // 10 steps after cp
    }

    #[test]
    fn test_add_changepoints_random_locations() {
        let mut values = vec![0.0; 200];
        let result = add_changepoints(&mut values, 42, 3, &[], "level", &[], &[], &[]);
        assert_eq!(result.changepoint_indices.len(), 3);
        // All indices should be within bounds
        for &idx in &result.changepoint_indices {
            assert!((0..200).contains(&idx));
        }
    }

    #[test]
    fn test_add_changepoints_variance_bounded() {
        // Regression: the old incremental window sum mixed pre- and
        // post-modification values and diverged exponentially for factors > 1.
        let mut values: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
        let locs = [0.2];
        let variance = [2.0];
        add_changepoints(&mut values, 42, 1, &locs, "variance", &[], &[], &variance);
        for (i, v) in values.iter().enumerate() {
            assert!(v.abs() < 100.0, "variance changepoint diverged at {i}: {v}");
        }
    }

    #[test]
    fn test_add_changepoints_location_one_maps_to_last_index() {
        let mut values = vec![0.0; 100];
        let locs = [1.0];
        let level = [10.0];
        let result = add_changepoints(&mut values, 42, 1, &locs, "level", &level, &[], &[]);
        assert_eq!(result.changepoint_indices[0], 99);
        assert!((values[99] - 10.0).abs() < 1e-10);
        assert!(values[98].abs() < 1e-10);
    }

    #[test]
    fn test_add_missingness_random() {
        let mut values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = add_missingness(&mut values, 42, "random", 0.2, 1, 1);
        // Exactly 20 distinct missing values (sampled without replacement)
        let nan_count = values.iter().filter(|v| v.is_nan()).count();
        assert_eq!(nan_count, 20);
        assert_eq!(result.missing_indices.len(), nan_count);
    }

    #[test]
    fn test_add_missingness_block_short_series() {
        // length <= block_size must still inject a (truncated) block
        let mut values = vec![1.0; 5];
        add_missingness(&mut values, 42, "block", 0.5, 10, 1);
        assert!(
            values.iter().all(|v| v.is_nan()),
            "short series should be fully covered by a truncated block"
        );
    }

    #[test]
    fn test_add_missingness_block() {
        let mut values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = add_missingness(&mut values, 42, "block", 0.2, 10, 1);
        let nan_count = values.iter().filter(|v| v.is_nan()).count();
        assert!(nan_count > 0, "block missingness should produce NaN");
        assert!(!result.missing_indices.is_empty());
    }

    #[test]
    fn test_add_missingness_seasonal() {
        let mut values: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let result = add_missingness(&mut values, 42, "seasonal", 0.2, 1, 10);
        let nan_count = values.iter().filter(|v| v.is_nan()).count();
        assert!(nan_count > 0, "seasonal missingness should produce NaN");
        assert!(!result.missing_indices.is_empty());
    }

    #[test]
    fn test_add_missingness_rate_zero_is_noop() {
        for pattern in ["random", "block", "seasonal"] {
            let mut values: Vec<f64> = (0..100).map(|i| i as f64).collect();
            let result = add_missingness(&mut values, 42, pattern, 0.0, 3, 7);
            assert!(
                values.iter().all(|v| !v.is_nan()),
                "{pattern}: rate 0 must not inject NaNs"
            );
            assert!(result.missing_indices.is_empty());
        }
    }

    #[test]
    fn test_add_missingness_rate_one_all_missing() {
        for pattern in ["random", "block", "seasonal"] {
            let mut values: Vec<f64> = (0..100).map(|i| i as f64).collect();
            let result = add_missingness(&mut values, 42, pattern, 1.0, 3, 7);
            assert!(
                values.iter().all(|v| v.is_nan()),
                "{pattern}: rate 1 must mark every point missing"
            );
            let expected: Vec<i64> = (0..100).collect();
            assert_eq!(result.missing_indices, expected);
        }
    }

    #[test]
    fn test_add_anomalies_spike() {
        let mut values = vec![0.0; 100];
        let types = vec!["spike".to_string()];
        let result = add_anomalies(&mut values, 42, &types, 0.1, 100.0, -100.0, 50.0, 10);
        // Should have ~10 anomalies
        assert!(!result.anomaly_indices.is_empty());
        // At least one value should be non-zero (spiked)
        assert!(values.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_add_anomalies_level_shift() {
        let mut values = vec![0.0; 100];
        let types = vec!["level_shift".to_string()];
        let result = add_anomalies(&mut values, 42, &types, 0.05, 100.0, -100.0, 50.0, 10);
        assert!(!result.anomaly_indices.is_empty());
        // Level shifts affect consecutive values
        let nonzero_count = values.iter().filter(|&&v| v != 0.0).count();
        assert!(
            nonzero_count > result.anomaly_indices.len(),
            "level shifts should affect more positions than anomaly count"
        );
    }

    #[test]
    fn test_add_anomalies_zero_fraction() {
        let mut values = vec![1.0; 50];
        let types = vec!["spike".to_string()];
        let result = add_anomalies(&mut values, 42, &types, 0.0, 100.0, -100.0, 50.0, 10);
        assert!(result.anomaly_indices.is_empty());
        // All values should remain 1.0
        for v in &values {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }
}
