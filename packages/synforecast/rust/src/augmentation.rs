//! Native algorithms shared by augmentation and feature-targeted generation.
//!
//! Sources: Petitjean et al. (2011), doi:10.1016/j.patcog.2010.09.013 (DBA);
//! Forestier et al. (2017), doi:10.1109/ICDM.2017.106 (weighted augmentation);
//! Bergmeir et al. (2016), doi:10.1016/j.ijforecast.2015.07.002 (MBB recipe).
//! Decomposition and strength formulas: https://otexts.com/fpp3/stlfeatures.html
//! and https://otexts.com/fpp3/classical-decomposition.html. Here decomposition
//! is classical, with extended endpoints; it is not STL/Box-Cox bagging.
//! See GENERATORS.md and tests/README.md for conventions and independent checks.

use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::{fft, rng::SfRng};

pub type Decomposition = (Vec<f64>, Vec<f64>, Vec<f64>);

/// Number of native coverage features; matches Python FEATURE_NAMES.
pub const FEATURE_SET_SIZE: usize = 12;

#[derive(PartialEq)]
struct Neighbor {
    distance: f64,
    index: usize,
}

impl Eq for Neighbor {}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn validate_values(values: &[f64]) -> Result<(), String> {
    if values.len() < 3 {
        return Err("values must contain at least 3 observations".to_string());
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err("values must contain only finite observations".to_string());
    }
    Ok(())
}

fn validate_dtw_inputs(a: &[f64], b: &[f64]) -> Result<(), String> {
    if a.is_empty() || b.is_empty() {
        return Err("DTW inputs must be non-empty one-dimensional arrays".to_string());
    }
    if !a.iter().chain(b).all(|value| value.is_finite()) {
        return Err("DTW inputs must be finite".to_string());
    }
    Ok(())
}

fn band_width(n: usize, m: usize, band: Option<usize>) -> usize {
    band.unwrap_or(n.max(m)).max(n.abs_diff(m)).min(n.max(m))
}

const MAX_ALIGNMENT_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_DBA_ITERATIONS: usize = 1000;

fn parent_cells(n: usize, m: usize, width: usize) -> Result<usize, String> {
    let row_width = width.saturating_mul(2).saturating_add(1).min(m);
    n.checked_mul(row_width).filter(|&cells| cells <= MAX_ALIGNMENT_BYTES)
        .ok_or_else(|| "DTW alignment exceeds the 64 MiB parent-storage limit; shorten series or reduce the band".to_string())
}

/// Banded DTW dynamic program over two rolling rows.
///
/// Returns the accumulated squared cost. When `parents` is provided it must
/// hold `n * min(2 * width + 1, m)` cells; row `i` stores the columns
/// `lower..=upper` of its band contiguously, starting at column `lower`.
fn dtw_core(a: &[f64], b: &[f64], width: usize, mut parents: Option<&mut [u8]>) -> f64 {
    let (n, m) = (a.len(), b.len());
    let row_width = width.saturating_mul(2).saturating_add(1).min(m);
    let mut previous = vec![f64::INFINITY; m + 1];
    let mut current = vec![f64::INFINITY; m + 1];
    previous[0] = 0.0;
    for i in 1..=n {
        let lower = 1.max(i.saturating_sub(width));
        let upper = m.min(i.saturating_add(width));
        current[lower - 1] = f64::INFINITY;
        if upper < m {
            current[upper + 1] = f64::INFINITY;
        }
        for j in lower..=upper {
            let mut best = previous[j - 1];
            let mut direction = 0u8;
            if previous[j] < best {
                best = previous[j];
                direction = 1;
            }
            if current[j - 1] < best {
                best = current[j - 1];
                direction = 2;
            }
            current[j] = (a[i - 1] - b[j - 1]).powi(2) + best;
            if let Some(parents) = parents.as_deref_mut() {
                parents[(i - 1) * row_width + (j - lower)] = direction;
            }
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[m]
}

/// Return square-root DTW distance without materializing an alignment path.
pub fn dtw_distance(a: &[f64], b: &[f64], band: Option<usize>) -> Result<f64, String> {
    validate_dtw_inputs(a, b)?;
    let cost = dtw_core(a, b, band_width(a.len(), b.len(), band), None);
    if !cost.is_finite() {
        return Err("DTW alignment is infeasible for the requested band".to_string());
    }
    Ok(cost.sqrt())
}

/// Return square-root DTW distance and an optimal alignment path.
pub fn dtw_alignment(
    a: &[f64],
    b: &[f64],
    band: Option<usize>,
) -> Result<(f64, Vec<(usize, usize)>), String> {
    validate_dtw_inputs(a, b)?;
    let (n, m) = (a.len(), b.len());
    let width = band_width(n, m, band);
    let row_width = width.saturating_mul(2).saturating_add(1).min(m);
    let cells = parent_cells(n, m, width)?;
    let mut parents = Vec::new();
    parents
        .try_reserve_exact(cells)
        .map_err(|_| "unable to allocate DTW parent storage")?;
    parents.resize(cells, u8::MAX);
    let cost = dtw_core(a, b, width, Some(&mut parents));
    if !cost.is_finite() {
        return Err("DTW alignment is infeasible for the requested band".to_string());
    }

    let infeasible = || "DTW alignment is infeasible for the requested band".to_string();
    let (mut i, mut j) = (n, m);
    let capacity = n
        .checked_add(m)
        .filter(|n| *n <= MAX_ALIGNMENT_BYTES / std::mem::size_of::<(usize, usize)>())
        .ok_or("DTW path exceeds 64 MiB")?;
    let mut path = Vec::new();
    path.try_reserve_exact(capacity)
        .map_err(|_| "unable to allocate DTW path")?;
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        let lower = 1.max(i.saturating_sub(width));
        if j < lower || j > m.min(i.saturating_add(width)) {
            return Err(infeasible());
        }
        match parents[(i - 1) * row_width + (j - lower)] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            2 => j -= 1,
            _ => return Err(infeasible()),
        }
    }
    path.reverse();
    Ok((cost.sqrt(), path))
}

/// Retain only the k nearest neighbors per series, breaking ties by input index.
/// Each unordered pair is evaluated once, in bounded parallel chunks. Storage
/// is O(n * k) plus one chunk, rather than a full pair list and distance matrix.
pub fn nearest_dtw_neighbors(
    series: &[Vec<f64>],
    window_fraction: f64,
    n_neighbors: usize,
) -> Result<Vec<Vec<(usize, f64)>>, String> {
    if !(window_fraction > 0.0 && window_fraction <= 1.0) {
        return Err("window_fraction must satisfy 0 < value <= 1".to_string());
    }
    if n_neighbors == 0 {
        return Err("n_neighbors must be >= 1".to_string());
    }
    for values in series {
        validate_dtw_inputs(values, values)?;
    }
    let k = n_neighbors.min(series.len().saturating_sub(1));
    let mut nearest: Vec<BinaryHeap<Neighbor>> = (0..series.len())
        .map(|_| BinaryHeap::with_capacity(k))
        .collect();
    const CHUNK_SIZE: usize = 4096;
    let mut pairs = Vec::with_capacity(CHUNK_SIZE);
    let mut evaluate = |pairs: &[(usize, usize)]| -> Result<(), String> {
        let distances = pairs
            .par_iter()
            .map(|&(i, j)| {
                let (a, b) = (&series[i], &series[j]);
                let band = ((window_fraction * a.len().max(b.len()) as f64).ceil() as usize)
                    .max(a.len().abs_diff(b.len()) + 1);
                dtw_distance(a, b, Some(band))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (&(i, j), distance) in pairs.iter().zip(distances) {
            for (source, index) in [(i, j), (j, i)] {
                let heap = &mut nearest[source];
                let neighbor = Neighbor { distance, index };
                if heap.len() < k {
                    heap.push(neighbor);
                } else if let Some(mut worst) = heap.peek_mut() {
                    if neighbor < *worst {
                        *worst = neighbor;
                    }
                }
            }
        }
        Ok(())
    };
    for i in 0..series.len() {
        for j in i + 1..series.len() {
            pairs.push((i, j));
            if pairs.len() == CHUNK_SIZE {
                evaluate(&pairs)?;
                pairs.clear();
            }
        }
    }
    evaluate(&pairs)?;
    Ok(nearest
        .into_iter()
        .map(|heap| {
            heap.into_sorted_vec()
                .into_iter()
                .map(|neighbor| (neighbor.index, neighbor.distance))
                .collect()
        })
        .collect())
}

/// Compute one weighted DBA barycenter restricted to reference length.
pub fn dba_barycenter(
    reference: &[f64],
    neighbors: &[Vec<f64>],
    weights: &[f64],
    n_iterations: usize,
    band: Option<usize>,
) -> Result<Vec<f64>, String> {
    Ok(dba_barycenters(
        reference,
        neighbors,
        &[weights.to_vec()],
        n_iterations,
        band,
        || Ok(()),
    )?
    .remove(0))
}

type Alignment = Vec<(usize, usize)>;

fn alignment_paths(
    center: &[f64],
    series: &[&[f64]],
    band: Option<usize>,
    parallel: bool,
) -> Result<Vec<Alignment>, String> {
    let mut paths = Vec::with_capacity(series.len());
    // Bound simultaneous parent allocations even on machines with many cores.
    for chunk in series.chunks(8) {
        let align = |values: &&[f64]| dtw_alignment(center, values, band).map(|(_, path)| path);
        let results: Result<Vec<_>, _> = if parallel {
            chunk.par_iter().map(align).collect()
        } else {
            chunk.iter().map(align).collect()
        };
        paths.extend(results?);
    }
    Ok(paths)
}

/// Refine all copies together, sharing iteration-one paths. `check` runs on
/// the calling thread between bounded work chunks (Python signal handling).
pub fn dba_barycenters(
    reference: &[f64],
    neighbors: &[Vec<f64>],
    weights: &[Vec<f64>],
    n_iterations: usize,
    band: Option<usize>,
    check: impl Fn() -> Result<(), String>,
) -> Result<Vec<Vec<f64>>, String> {
    if !(1..=MAX_DBA_ITERATIONS).contains(&n_iterations) {
        return Err("n_iterations must be in [1, 1000]".to_string());
    }
    if weights.is_empty() || weights.len() > 1000 {
        return Err("DBA requires between 1 and 1000 copies".to_string());
    }
    if reference
        .len()
        .checked_mul(weights.len())
        .is_none_or(|n| n > MAX_ALIGNMENT_BYTES / 8)
    {
        return Err("DBA output exceeds 64 MiB; reduce copies or series length".to_string());
    }
    let series: Vec<&[f64]> = std::iter::once(reference)
        .chain(neighbors.iter().map(Vec::as_slice))
        .collect();
    for values in &series {
        validate_dtw_inputs(values, values)?;
    }
    for row in weights {
        let total: f64 = row.iter().sum();
        if row.len() != series.len()
            || row.iter().any(|w| !w.is_finite() || *w < 0.0)
            || !total.is_finite()
            || total <= 0.0
        {
            return Err("weights must be non-negative and match all input series".to_string());
        }
    }
    // Every path has at most n+m-1 cells. Bound retained paths per copy;
    // initial paths are shared, later copies execute in chunks of at most 8.
    let path_cells = series
        .iter()
        .try_fold(0usize, |total, values| {
            total
                .checked_add(reference.len())?
                .checked_add(values.len())
        })
        .ok_or("DBA alignment cache size overflow")?;
    if path_cells > MAX_ALIGNMENT_BYTES / std::mem::size_of::<(usize, usize)>() {
        return Err(
            "DBA alignment cache exceeds 64 MiB; shorten series or reduce neighbors".to_string(),
        );
    }
    check()?;
    let initial = alignment_paths(reference, &series, band, true)?;
    let mut centers = vec![reference.to_vec(); weights.len()];
    for iteration in 0..n_iterations {
        for (group, weight_group) in centers.chunks_mut(8).zip(weights.chunks(8)) {
            check()?;
            group
                .par_iter_mut()
                .zip(weight_group.par_iter())
                .try_for_each(|(center, row)| -> Result<(), String> {
                    let next;
                    let paths = if iteration == 0 {
                        &initial
                    } else {
                        next = alignment_paths(center, &series, band, weights.len() == 1)?;
                        &next
                    };
                    let mut sums = vec![0.0; center.len()];
                    let mut totals = vec![0.0; center.len()];
                    // Keep accumulation order independent of worker scheduling.
                    for ((values, path), weight) in series.iter().zip(paths).zip(row) {
                        for &(i, j) in path {
                            sums[i] += weight * values[j];
                            totals[i] += weight;
                        }
                    }
                    for i in 0..center.len() {
                        if totals[i] > 0.0 {
                            center[i] = sums[i] / totals[i];
                        }
                    }
                    Ok(())
                })?;
        }
    }
    check()?;
    Ok(centers)
}

fn moving_average(values: &[f64], period: Option<usize>) -> Vec<f64> {
    let n = values.len();
    let (mut width, mut half_endpoints) = match period {
        None => {
            let mut window = (n / 10) | 1;
            window = window.max(3);
            if window > n {
                window = if n % 2 == 1 { n } else { n - 1 };
            }
            (window, false)
        }
        Some(period) if period % 2 == 1 => (period, false),
        Some(period) => (period + 1, true),
    };
    if width > n {
        width = if n % 2 == 1 { n } else { n - 1 };
        half_endpoints = false;
    }
    // Center before accumulating and compensate rounding error as the window
    // slides. Even periods retain the classical half-weighted endpoints.
    let baseline = values[0];
    let mut sum = 0.0;
    let mut correction = 0.0;
    let add = |sum: &mut f64, correction: &mut f64, value: f64| {
        let adjusted = value - *correction;
        let next = *sum + adjusted;
        *correction = (next - *sum) - adjusted;
        *sum = next;
    };
    for value in &values[..width] {
        add(&mut sum, &mut correction, value - baseline);
    }
    let denominator = if half_endpoints { width - 1 } else { width } as f64;
    let mut computed = Vec::with_capacity(n - width + 1);
    for start in 0..=n - width {
        if start > 0 {
            add(&mut sum, &mut correction, -(values[start - 1] - baseline));
            add(
                &mut sum,
                &mut correction,
                values[start + width - 1] - baseline,
            );
        }
        let endpoints = if half_endpoints {
            0.5 * (values[start] - baseline) + 0.5 * (values[start + width - 1] - baseline)
        } else {
            0.0
        };
        computed.push(baseline + (sum - endpoints) / denominator);
    }
    let left = (width - 1) / 2;
    let right = n - computed.len() - left;
    let mut trend = Vec::with_capacity(n);
    trend.extend(std::iter::repeat_n(computed[0], left));
    trend.extend_from_slice(&computed);
    trend.extend(std::iter::repeat_n(*computed.last().unwrap(), right));
    trend
}

/// Classical moving-average decomposition used by the Python feature helpers.
pub fn classical_decompose(values: &[f64], period: Option<usize>) -> Result<Decomposition, String> {
    validate_values(values)?;
    if period == Some(0) || period == Some(1) {
        return Err("period must be >= 2 when provided".to_string());
    }
    let usable_period = period.filter(|period| *period <= values.len() / 2);
    let trend = moving_average(values, usable_period);
    let mut seasonal = vec![0.0; values.len()];
    if let Some(period) = usable_period {
        let mut phase_means = vec![0.0; period];
        for (phase, phase_mean) in phase_means.iter_mut().enumerate() {
            let mut total = 0.0;
            let mut count = 0;
            for index in (phase..values.len()).step_by(period) {
                total += values[index] - trend[index];
                count += 1;
            }
            *phase_mean = total / count as f64;
        }
        let phase_mean = phase_means.iter().sum::<f64>() / period as f64;
        for value in &mut phase_means {
            *value -= phase_mean;
        }
        for (index, value) in seasonal.iter_mut().enumerate() {
            *value = phase_means[index % period];
        }
    }
    let remainder = values
        .iter()
        .zip(&trend)
        .zip(&seasonal)
        .map(|((&value, &trend), &seasonal)| value - trend - seasonal)
        .collect();
    Ok((trend, seasonal, remainder))
}

fn variance(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64
}

fn spectral_entropy(values: &[f64]) -> Result<f64, String> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut centered: Vec<f64> = values.iter().map(|value| value - mean).collect();
    if variance(&centered) <= f64::EPSILON {
        return Ok(0.0);
    }
    let n_bins = values.len() / 2;
    if n_bins <= 1 {
        return Ok(0.0);
    }
    let spectrum = fft::rfft_half(&mut centered)?;
    let bins = &spectrum[1..=n_bins];
    let total = bins.iter().map(|value| value.norm_sqr()).sum::<f64>();
    if total <= f64::EPSILON {
        return Ok(0.0);
    }
    let entropy = bins
        .iter()
        .filter_map(|value| {
            let power = value.norm_sqr();
            if power > 0.0 {
                let probability = power / total;
                Some(-probability * probability.ln())
            } else {
                None
            }
        })
        .sum::<f64>();
    Ok((entropy / (n_bins as f64).ln()).clamp(0.0, 1.0))
}

/// Return spectral entropy, trend strength, seasonal strength, and ACF(1).
pub fn compute_features(
    values: &[f64],
    period: Option<usize>,
) -> Result<(f64, f64, f64, f64), String> {
    validate_values(values)?;
    if period == Some(0) || period == Some(1) {
        return Err("period must be >= 2 when provided".to_string());
    }
    let decomposition = classical_decompose(values, period)?;
    features_from_decomposition(values, period, &decomposition)
}

fn features_from_decomposition(
    values: &[f64],
    period: Option<usize>,
    decomposition: &Decomposition,
) -> Result<(f64, f64, f64, f64), String> {
    let (trend, seasonal, remainder) = decomposition;
    let trend_plus_remainder: Vec<f64> = trend
        .iter()
        .zip(remainder)
        .map(|(trend, remainder)| trend + remainder)
        .collect();
    let trend_denominator = variance(&trend_plus_remainder);
    let trend_strength = if trend_denominator <= f64::EPSILON {
        0.0
    } else {
        (1.0 - variance(remainder) / trend_denominator).clamp(0.0, 1.0)
    };
    let seasonal_strength = if period.is_none()
        || period.is_some_and(|period| period > values.len() / 2)
        || seasonal.iter().all(|value| *value == 0.0)
    {
        0.0
    } else {
        let seasonal_plus_remainder: Vec<f64> = seasonal
            .iter()
            .zip(remainder)
            .map(|(seasonal, remainder)| seasonal + remainder)
            .collect();
        let denominator = variance(&seasonal_plus_remainder);
        if denominator <= f64::EPSILON {
            0.0
        } else {
            (1.0 - variance(remainder) / denominator).clamp(0.0, 1.0)
        }
    };
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = variance(values);
    let acf1 = if var == 0.0 {
        0.0
    } else {
        values[..values.len() - 1]
            .iter()
            .zip(&values[1..])
            .map(|(left, right)| (left - mean) * (right - mean))
            .sum::<f64>()
            / values.len() as f64
            / var
    };
    Ok((
        spectral_entropy(values)?,
        trend_strength,
        seasonal_strength,
        acf1,
    ))
}

/// Biased, mean-centered sample ACF, with the same denominator at every lag.
fn feature_acf(values: &[f64], lag: usize) -> f64 {
    if values.len() <= lag {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let denominator = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
    if denominator <= f64::EPSILON {
        return 0.0;
    }
    values[..values.len() - lag]
        .iter()
        .zip(&values[lag..])
        .map(|(a, b)| (a - mean) * (b - mean))
        .sum::<f64>()
        / denominator
}

/// Unit population variance; scale first to avoid overflow on finite inputs.
fn normalize_features(values: &[f64]) -> Vec<f64> {
    let magnitude = values.iter().map(|x| x.abs()).fold(0.0, f64::max);
    if magnitude == 0.0 {
        return vec![0.0; values.len()];
    }
    let scaled: Vec<f64> = values.iter().map(|x| x / magnitude).collect();
    let mean = scaled.iter().sum::<f64>() / scaled.len() as f64;
    let std = variance(&scaled).sqrt();
    if std == 0.0 {
        return vec![0.0; values.len()];
    }
    scaled.iter().map(|x| (x - mean) / std).collect()
}

/// Native coverage candidate schema; order matches Python FEATURE_NAMES.
/// Short but otherwise valid input returns NaNs for unavailable features.
pub fn compute_feature_set(
    values: &[f64],
    period: Option<usize>,
    window_size: Option<usize>,
) -> Result<Vec<f64>, String> {
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return Err("values must be non-empty and finite".to_string());
    }
    if period == Some(0) || period == Some(1) {
        return Err("period must be >= 2 when provided".to_string());
    }
    if window_size.is_some_and(|width| width < 2) {
        return Err("window_size must be >= 2 when provided".to_string());
    }
    let n = values.len();
    if n < 3 {
        return Ok(vec![f64::NAN; FEATURE_SET_SIZE]);
    }
    let normalized = normalize_features(values);
    let decomposition = classical_decompose(&normalized, period)?;
    let (entropy, trend, seasonal, acf1) =
        features_from_decomposition(&normalized, period, &decomposition)?;
    let unusable_period = period.is_some_and(|p| p > n / 2);
    let (trend, seasonal) = if unusable_period {
        (f64::NAN, f64::NAN)
    } else {
        (trend, seasonal)
    };
    let x_acf10 = if n > 10 {
        (1..=10)
            .map(|lag| feature_acf(&normalized, lag).powi(2))
            .sum()
    } else {
        f64::NAN
    };
    let differences: Vec<f64> = normalized
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();
    let diff1_acf1 = feature_acf(&differences, 1);
    let seas_acf1 = period.map_or(0.0, |p| feature_acf(&normalized, p));
    let spike = if unusable_period {
        f64::NAN
    } else {
        let remainder = &decomposition.2;
        let mean = remainder.iter().sum::<f64>() / n as f64;
        let deviations: Vec<f64> = remainder.iter().map(|x| x - mean).collect();
        let total = deviations.iter().map(|x| x * x).sum::<f64>();
        let loo_variances: Vec<f64> = deviations
            .iter()
            .map(|x| ((total - n as f64 / (n - 1) as f64 * x * x) / (n - 1) as f64).max(0.0))
            .collect();
        variance(&loo_variances)
    };
    let width = window_size.or(period).unwrap_or(10);
    let (lumpiness, max_level_shift, max_var_shift) = if width <= n / 2 {
        let tile_variances: Vec<f64> = normalized.chunks_exact(width).map(variance).collect();
        let lumpiness = variance(&tile_variances);
        let mut sum = normalized[..width].iter().sum::<f64>();
        let mut squares = normalized[..width].iter().map(|x| x * x).sum::<f64>();
        let mut means = Vec::with_capacity(n - width + 1);
        let mut variances = Vec::with_capacity(n - width + 1);
        for start in 0..=n - width {
            if start > 0 {
                let old = normalized[start - 1];
                let new = normalized[start + width - 1];
                sum += new - old;
                squares += new * new - old * old;
            }
            let mean = sum / width as f64;
            means.push(mean);
            variances.push((squares / width as f64 - mean * mean).max(0.0));
        }
        let level = (0..=n - 2 * width)
            .map(|i| (means[i] - means[i + width]).abs())
            .fold(0.0, f64::max);
        let var = (0..=n - 2 * width)
            .map(|i| (variances[i] - variances[i + width]).abs())
            .fold(0.0, f64::max);
        (lumpiness, level, var)
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    };
    let mut scratch = normalized.clone();
    let (left, upper, _) = scratch.select_nth_unstable_by(n / 2, f64::total_cmp);
    let upper = *upper;
    let median = if n.is_multiple_of(2) {
        let lower = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lower + upper)
    } else {
        upper
    };
    let crossing_points = normalized
        .windows(2)
        .filter(|pair| (pair[0] <= median) != (pair[1] <= median))
        .count() as f64;
    Ok(vec![
        entropy,
        trend,
        seasonal,
        acf1,
        x_acf10,
        diff1_acf1,
        seas_acf1,
        spike,
        lumpiness,
        max_level_shift,
        max_var_shift,
        crossing_points,
    ])
}

pub fn compute_feature_set_batch(
    values: &[f64],
    offsets: &[usize],
    period: Option<usize>,
    window_size: Option<usize>,
) -> Result<Vec<f64>, String> {
    if offsets.len() < 2
        || offsets[0] != 0
        || *offsets.last().unwrap() != values.len()
        || offsets.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err("offsets must increase strictly from 0 to len(values)".to_string());
    }
    let n_series = offsets.len() - 1;
    let mut order: Vec<usize> = (0..n_series).collect();
    order.sort_by_key(|&i| offsets[i + 1] - offsets[i]);
    let rows = order
        .par_iter()
        .map(|&i| {
            compute_feature_set(&values[offsets[i]..offsets[i + 1]], period, window_size)
                .map(|row| (i, row))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let width = FEATURE_SET_SIZE;
    let mut output = vec![0.0; n_series * width];
    for (i, row) in rows {
        output[i * width..(i + 1) * width].copy_from_slice(&row);
    }
    Ok(output)
}

/// Bootstrap decomposition remainders using randomly selected moving blocks.
pub fn moving_block_bootstrap(
    values: &[f64],
    period: Option<usize>,
    block_size: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    let decomposition = classical_decompose(values, period)?;
    if block_size < 2 || block_size > values.len() {
        return Err("block_size must be in [2, len(values)]".to_string());
    }
    Ok(bootstrap_remainder(&decomposition, block_size, seed))
}

/// Decompose once and independently resample the remainder for each seed.
pub fn moving_block_bootstrap_many(
    values: &[f64],
    period: Option<usize>,
    block_size: usize,
    seeds: &[u64],
) -> Result<Vec<Vec<f64>>, String> {
    let decomposition = classical_decompose(values, period)?;
    if block_size < 2 || block_size > values.len() {
        return Err("block_size must be in [2, len(values)]".to_string());
    }
    Ok(seeds
        .iter()
        .map(|&seed| bootstrap_remainder(&decomposition, block_size, seed))
        .collect())
}

fn bootstrap_remainder(decomposition: &Decomposition, block_size: usize, seed: u64) -> Vec<f64> {
    let (trend, seasonal, remainder) = decomposition;
    let length = remainder.len();
    let mut rng = SfRng::new(seed);
    let n_blocks = length / block_size + 2;
    let mut sampled = Vec::with_capacity(n_blocks * block_size);
    for _ in 0..n_blocks {
        let start = rng.integers(0, (length - block_size + 1) as i32) as usize;
        sampled.extend_from_slice(&remainder[start..start + block_size]);
    }
    let offset = rng.integers(0, block_size as i32) as usize;
    trend
        .iter()
        .zip(seasonal)
        .zip(&sampled[offset..offset + length])
        .map(|((&trend, &seasonal), &remainder)| trend + seasonal + remainder)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_budget_and_overflow_are_rejected() {
        assert!(parent_cells(10_000, 10_000, 10_000).is_err());
        assert!(parent_cells(usize::MAX, usize::MAX, usize::MAX).is_err());
        assert_eq!(parent_cells(100, 100, 2).unwrap(), 500);
    }

    #[test]
    fn dba_checks_cancellation_between_work_chunks() {
        use std::cell::Cell;
        let calls = Cell::new(0);
        let result = dba_barycenters(
            &[0., 1., 2.],
            &[vec![1., 2., 3.]],
            &[vec![0.5, 0.5]],
            5,
            Some(1),
            || {
                calls.set(calls.get() + 1);
                if calls.get() == 3 {
                    Err("cancelled".to_string())
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(result.unwrap_err(), "cancelled");
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn dtw_identity_and_path() {
        let values = [1.0, 2.0, 3.0];
        let (distance, path) = dtw_alignment(&values, &values, Some(0)).unwrap();
        assert_eq!(distance, 0.0);
        assert_eq!(path, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn banded_dtw_matches_unbanded_when_band_covers_everything() {
        let a: Vec<f64> = (0..40).map(|i| (i as f64 * 0.3).sin()).collect();
        let b: Vec<f64> = (0..37).map(|i| (i as f64 * 0.31 + 0.2).sin()).collect();
        let (full, path_full) = dtw_alignment(&a, &b, None).unwrap();
        let (wide, path_wide) = dtw_alignment(&a, &b, Some(100)).unwrap();
        assert_eq!(full, wide);
        assert_eq!(path_full, path_wide);
        assert_eq!(dtw_distance(&a, &b, None).unwrap(), full);
        let (narrow, _) = dtw_alignment(&a, &b, Some(4)).unwrap();
        assert!(narrow >= full);
        assert_eq!(dtw_distance(&a, &b, Some(4)).unwrap(), narrow);
    }

    #[test]
    fn decomposition_reconstructs_input() {
        use std::f64::consts::PI;
        let values: Vec<f64> = (0..48)
            .map(|index| index as f64 / 10.0 + (2.0 * PI * index as f64 / 12.0).sin())
            .collect();
        let (trend, seasonal, remainder) = classical_decompose(&values, Some(12)).unwrap();
        for index in 0..values.len() {
            assert!(
                (values[index] - trend[index] - seasonal[index] - remainder[index]).abs() < 1e-12
            );
        }
    }

    #[test]
    fn dba_and_mbb_are_deterministic() {
        let reference = vec![0.0, 1.0, 2.0, 1.0];
        let neighbors = vec![vec![0.0, 1.2, 1.8, 1.0]];
        let a = dba_barycenter(&reference, &neighbors, &[1.0, 1.0], 2, Some(1)).unwrap();
        let b = dba_barycenter(&reference, &neighbors, &[1.0, 1.0], 2, Some(1)).unwrap();
        assert_eq!(a, b);

        let values: Vec<f64> = (0..24).map(|index| index as f64).collect();
        assert_eq!(
            moving_block_bootstrap(&values, Some(4), 4, 9).unwrap(),
            moving_block_bootstrap(&values, Some(4), 4, 9).unwrap()
        );
    }

    #[test]
    fn coverage_acf_and_window_oracles() {
        assert!((feature_acf(&[-1.0, 1.0, -1.0, 1.0], 1) + 0.75).abs() < 1e-12);
        assert!(feature_acf(&[1.0, 2.0], 2).is_nan());
        let values: Vec<f64> = (0..40).map(|i| if i < 20 { 0.0 } else { 1.0 }).collect();
        let features = compute_feature_set(&values, None, None).unwrap();
        assert!((features[9] - 2.0).abs() < 1e-12);
        assert_eq!(features[11], 1.0);
        let variance_change: Vec<f64> = (0..40)
            .map(|i| {
                let magnitude = if i < 20 { 1.0 } else { 3.0 };
                if i % 2 == 0 {
                    -magnitude
                } else {
                    magnitude
                }
            })
            .collect();
        let features = compute_feature_set(&variance_change, None, None).unwrap();
        assert!((features[8] - 0.64).abs() < 1e-12);
        assert!((features[10] - 1.6).abs() < 1e-12);
    }

    #[test]
    fn coverage_features_are_scale_invariant() {
        let values: Vec<f64> = (0..120)
            .map(|i| (i as f64 * 0.7).sin() + 0.3 * (i as f64 * 0.11).cos() + i as f64 * 0.01)
            .collect();
        let reference = compute_feature_set(&values, Some(12), None).unwrap();
        for scale in [1e-12, 1e-9, 1e160] {
            let scaled: Vec<f64> = values.iter().map(|x| x * scale).collect();
            let features = compute_feature_set(&scaled, Some(12), None).unwrap();
            for (actual, expected) in features.iter().zip(&reference) {
                assert!((actual - expected).abs() <= 1e-8 * expected.abs().max(1e-2));
            }
        }
    }

    #[test]
    fn coverage_unusable_period_leaves_decomposition_features_undefined() {
        let values: Vec<f64> = (0..20).map(|i| (i as f64).sin()).collect();
        let features = compute_feature_set(&values, Some(12), Some(5)).unwrap();
        assert!(features[1].is_nan() && features[2].is_nan() && features[7].is_nan());
        assert!(features[0].is_finite() && features[3].is_finite());
        assert!(features[8].is_finite());
        let usable = compute_feature_set(&values, Some(10), None).unwrap();
        assert!(usable[1].is_finite() && usable[2].is_finite() && usable[7].is_finite());
    }

    #[test]
    fn coverage_batch_matches_per_series_in_input_order() {
        let series: Vec<Vec<f64>> = [31usize, 2, 64, 17, 64]
            .iter()
            .map(|&n| (0..n).map(|i| ((i * 7 + n) % 11) as f64).collect())
            .collect();
        let mut offsets = vec![0];
        for values in &series {
            offsets.push(offsets.last().unwrap() + values.len());
        }
        let flat: Vec<f64> = series.concat();
        let rows = compute_feature_set_batch(&flat, &offsets, Some(4), None).unwrap();
        for (i, values) in series.iter().enumerate() {
            let expected = compute_feature_set(values, Some(4), None).unwrap();
            let actual = &rows[i * FEATURE_SET_SIZE..(i + 1) * FEATURE_SET_SIZE];
            for (a, e) in actual.iter().zip(&expected) {
                assert!(a == e || (a.is_nan() && e.is_nan()));
            }
        }
        assert!(compute_feature_set_batch(&flat, &[0, 3], None, None).is_err());
        assert!(compute_feature_set_batch(&flat, &[0, 0, flat.len()], None, None).is_err());
    }

    #[test]
    fn coverage_short_and_constant_inputs() {
        assert!(compute_feature_set(&[], None, None).is_err());
        assert!(compute_feature_set(&[1.0, f64::INFINITY, 2.0], None, None).is_err());
        assert!(compute_feature_set(&[1.0; 3], Some(1), None).is_err());
        assert!(compute_feature_set(&[1.0; 2], None, None)
            .unwrap()
            .iter()
            .all(|x| x.is_nan()));
        let features = compute_feature_set(&[1.0; 100], Some(12), None).unwrap();
        assert!(features.iter().all(|x| *x == 0.0));
        let features = compute_feature_set(&[1.0; 10], Some(12), None).unwrap();
        assert!(features[2].is_nan());
        assert!(features[4].is_nan());
        assert!(features[8].is_nan());
        assert_eq!(compute_features(&[1.0; 10], Some(12)).unwrap().2, 0.0);
        assert!(compute_feature_set(&[1.0; 13], None, Some(1)).is_err());
        let short: Vec<f64> = (0..13).map(|x| x as f64).collect();
        assert!(compute_feature_set(&short, None, None).unwrap()[8].is_nan());
        let explicit = compute_feature_set(&short, None, Some(5)).unwrap();
        assert!(explicit.iter().all(|x| x.is_finite()));
    }
}
