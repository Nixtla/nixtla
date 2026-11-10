//! Overlay pipeline shared by the native batch path and the per-series Python path.
//!
//! Stages run on each series in a fixed order: base shrink, calendar cycle,
//! waveform, changepoints, anomalies, observation, missingness. Relative
//! magnitudes are measured once, on the untouched base series. The released
//! stages (changepoints, anomalies, missingness) draw from the legacy seeds
//! exactly as before; the newer stages draw from streams derived from the raw
//! per-series seeds, so switching one stage on never moves another.

use std::f64::consts::PI;

use crate::pattern_injection as pi;
use crate::rng::SfRng;

const STAGE_CALENDAR: u64 = 1;
const STAGE_WAVEFORM: u64 = 2;
const STAGE_OBSERVATION: u64 = 3;

/// Points on which a Fourier profile is normalized to peak 1.
const PROFILE_GRID: usize = 1440;

fn splitmix64(z: u64) -> u64 {
    let mut z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed of one stream of one stage, mixed from the three raw per-series seeds.
pub fn derive_seed(raw: [u64; 3], stage: u64, sub: u64) -> u64 {
    raw.iter()
        .fold(splitmix64((stage << 32) | sub), |h, &s| splitmix64(h ^ s))
}

pub struct ChangepointPlan {
    pub count: i32,
    pub kind: String,
    pub locations: Vec<f64>,
    pub level_changes: Vec<f64>,
    pub trend_changes: Vec<f64>,
    pub variance_changes: Vec<f64>,
    pub relative: bool,
}

pub struct AnomalyPlan {
    pub types: Vec<String>,
    pub fraction: f64,
    pub spike_magnitude: f64,
    pub dip_magnitude: f64,
    pub level_shift_magnitude: f64,
    pub level_shift_duration: i32,
    pub relative: bool,
}

pub struct MissingPlan {
    pub pattern: String,
    pub rate: f64,
    pub block_size: i32,
    pub seasonal_period: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Fourier { harmonics: usize },
    Levels { bins: usize },
}

pub struct CyclePlan {
    pub profile: Profile,
    /// Fixed profile: Fourier coefficients (cosines then sines) or one value per bin.
    pub values: Option<Vec<f64>>,
    /// Per-series normal noise added to fixed values.
    pub jitter: f64,
    pub weight: f64,
    /// Phase in [0, 1) of every step, for the longest series.
    pub phase: Vec<f64>,
}

pub struct CalendarPlan {
    pub cycles: Vec<CyclePlan>,
    pub multiplicative: bool,
    pub amplitude: f64,
    pub swing: f64,
    pub relative: bool,
    pub fallback_additive: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveShape {
    Pulse,
    Spike,
    Sawtooth,
    Trapezoid,
    Levels,
    Staircase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveTiming {
    Periodic,
    Jitter,
    Renewal,
}

pub struct WaveformPlan {
    pub shape: WaveShape,
    pub period: usize,
    pub duty: f64,
    pub width: usize,
    pub ramp: f64,
    pub levels: usize,
    pub timing: WaveTiming,
    pub jitter: f64,
    pub lognormal: bool,
    pub cv: f64,
    pub dip: bool,
    pub multiplicative: bool,
    pub amplitude: f64,
    pub swing: f64,
    pub relative: bool,
    pub fallback_additive: bool,
}

pub struct ObservationPlan {
    pub floor: Option<f64>,
    pub cap: Option<f64>,
    pub quantum: f64,
    pub hold: usize,
    pub stuck_fraction: f64,
    pub stuck_length: usize,
}

pub struct OverlayPlan {
    pub base_weight: f64,
    pub calendar: Option<CalendarPlan>,
    pub waveform: Option<WaveformPlan>,
    pub changepoints: Option<ChangepointPlan>,
    pub anomalies: Option<AnomalyPlan>,
    pub observation: Option<ObservationPlan>,
    pub missing: Option<MissingPlan>,
}

impl Default for OverlayPlan {
    fn default() -> Self {
        Self {
            base_weight: 1.0,
            calendar: None,
            waveform: None,
            changepoints: None,
            anomalies: None,
            observation: None,
            missing: None,
        }
    }
}

impl OverlayPlan {
    /// Longest series the calendar phases cover, or None without a calendar.
    pub fn phase_length(&self) -> Option<usize> {
        self.calendar
            .as_ref()
            .and_then(|c| c.cycles.iter().map(|cycle| cycle.phase.len()).min())
    }

    /// Fail when a series is longer than the calendar phases cover.
    pub fn check_length(&self, len: usize) -> Result<(), String> {
        match self.phase_length() {
            Some(covered) if len > covered => Err(format!(
                "series of length {len} exceeds the {covered} calendar phases computed"
            )),
            _ => Ok(()),
        }
    }

    fn needs_scale(&self) -> bool {
        self.changepoints.as_ref().is_some_and(|c| c.relative)
            || self.anomalies.as_ref().is_some_and(|a| a.relative)
            || self.calendar.as_ref().is_some_and(|c| c.relative)
            || self.waveform.as_ref().is_some_and(|w| w.relative)
    }
}

#[derive(Default)]
pub struct OverlayIndices {
    pub changepoints: Vec<i64>,
    pub anomalies: Vec<i64>,
    pub missing: Vec<i64>,
}

/// Run every enabled stage on one series in place.
///
/// `raw` are the series' three seeds as drawn; `legacy` are the seeds the
/// released stages consume (the raw seeds on the batch path, their NumPy
/// redraws on the Python path, so both paths keep their historical draws).
pub fn apply_overlays(
    values: &mut [f64],
    plan: &OverlayPlan,
    raw: [u64; 3],
    legacy: [u64; 3],
) -> OverlayIndices {
    // Relative magnitudes are sized on the untouched base series.
    let base_scale = if plan.needs_scale() {
        pi::overlay_scale(values)
    } else {
        1.0
    };
    if plan.base_weight != 1.0 {
        shrink_base(values, plan.base_weight);
    }
    if let Some(calendar) = &plan.calendar {
        add_calendar(values, calendar, raw, base_scale);
    }
    if let Some(waveform) = &plan.waveform {
        add_waveform(values, waveform, raw, base_scale);
    }
    let mut indices = OverlayIndices::default();
    if let Some(cp) = &plan.changepoints {
        indices.changepoints = pi::add_changepoints(
            values,
            legacy[0],
            cp.count,
            &cp.locations,
            &cp.kind,
            &cp.level_changes,
            &cp.trend_changes,
            &cp.variance_changes,
            cp.relative,
            if cp.relative { base_scale } else { 1.0 },
        )
        .changepoint_indices;
    }
    if let Some(an) = &plan.anomalies {
        indices.anomalies = pi::add_anomalies(
            values,
            legacy[1],
            &an.types,
            an.fraction,
            an.spike_magnitude,
            an.dip_magnitude,
            an.level_shift_magnitude,
            an.level_shift_duration,
            if an.relative { base_scale } else { 1.0 },
        )
        .anomaly_indices;
    }
    if let Some(obs) = &plan.observation {
        add_observation(values, obs, raw);
    }
    if let Some(miss) = &plan.missing {
        indices.missing = pi::add_missingness(
            values,
            legacy[2],
            &miss.pattern,
            miss.rate,
            miss.block_size,
            miss.seasonal_period,
        )
        .missing_indices;
    }
    indices
}

/// Whether every finite value is non-negative and at least one is finite.
fn non_negative(values: &[f64]) -> bool {
    let mut any = false;
    for &v in values.iter().filter(|v| v.is_finite()) {
        if v < 0.0 {
            return false;
        }
        any = true;
    }
    any
}

/// Shrink the series around its finite mean by `weight`; missing values stay missing.
fn shrink_base(values: &mut [f64], weight: f64) {
    let (sum, count) = values
        .iter()
        .filter(|v| v.is_finite())
        .fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    if count == 0 {
        return;
    }
    let mean = sum / count as f64;
    for v in values.iter_mut() {
        *v = mean + weight * (*v - mean);
    }
}

/// Apply a periodic profile in [-1, 1] additively or multiplicatively.
#[allow(clippy::too_many_arguments)]
fn apply_profile(
    values: &mut [f64],
    profile: &[f64],
    multiplicative: bool,
    fallback_additive: bool,
    swing: f64,
    amplitude: f64,
    relative: bool,
    base_scale: f64,
) {
    if multiplicative && non_negative(values) {
        for (v, p) in values.iter_mut().zip(profile) {
            *v *= 1.0 + swing * p;
        }
    } else if !multiplicative || fallback_additive {
        let amp = amplitude * if relative { base_scale } else { 1.0 };
        for (v, p) in values.iter_mut().zip(profile) {
            *v += amp * p;
        }
    }
}

// ---------------------------------------------------------------------------
// Calendar cycles
// ---------------------------------------------------------------------------

/// Fourier series at phase `theta` with harmonics computed by angle addition.
fn fourier(theta: f64, cos_coef: &[f64], sin_coef: &[f64]) -> f64 {
    let (s1, c1) = (2.0 * PI * theta).sin_cos();
    let (mut ck, mut sk) = (c1, s1);
    let mut total = 0.0;
    for (a, b) in cos_coef.iter().zip(sin_coef) {
        total += a * ck + b * sk;
        (ck, sk) = (ck * c1 - sk * s1, sk * c1 + ck * s1);
    }
    total
}

/// One cycle's profile with peak magnitude 1 (all zeros if it is flat).
fn cycle_profile(cycle: &CyclePlan, len: usize, rng: &mut SfRng) -> Vec<f64> {
    let phase = &cycle.phase[..len];
    let mut draw = |fixed: Option<f64>, scale: f64| match fixed {
        Some(v) if cycle.jitter > 0.0 => v + rng.normal(0.0, cycle.jitter),
        Some(v) => v,
        None => rng.normal(0.0, scale),
    };
    match cycle.profile {
        Profile::Fourier { harmonics } => {
            let fixed = cycle.values.as_deref();
            let mut cos_coef = vec![0.0; harmonics];
            let mut sin_coef = vec![0.0; harmonics];
            for k in 0..harmonics {
                let scale = 1.0 / (k + 1) as f64;
                cos_coef[k] = draw(fixed.map(|v| v[k]), scale);
                sin_coef[k] = draw(fixed.map(|v| v[harmonics + k]), scale);
            }
            let peak = (0..PROFILE_GRID)
                .map(|g| fourier(g as f64 / PROFILE_GRID as f64, &cos_coef, &sin_coef).abs())
                .fold(0.0, f64::max);
            if peak > 0.0 {
                phase
                    .iter()
                    .map(|&theta| fourier(theta, &cos_coef, &sin_coef) / peak)
                    .collect()
            } else {
                vec![0.0; len]
            }
        }
        Profile::Levels { bins } => {
            let fixed = cycle.values.as_deref();
            let mut levels: Vec<f64> = (0..bins).map(|b| draw(fixed.map(|v| v[b]), 1.0)).collect();
            let mean = levels.iter().sum::<f64>() / bins as f64;
            levels.iter_mut().for_each(|l| *l -= mean);
            let peak = levels.iter().fold(0.0_f64, |m, l| m.max(l.abs()));
            if peak > 0.0 {
                levels.iter_mut().for_each(|l| *l /= peak);
                phase
                    .iter()
                    .map(|&theta| levels[((theta * bins as f64) as usize).min(bins - 1)])
                    .collect()
            } else {
                vec![0.0; len]
            }
        }
    }
}

/// Weighted mean of every cycle's profile, so the combined peak is at most 1.
fn calendar_profile(len: usize, plan: &CalendarPlan, raw: [u64; 3]) -> Vec<f64> {
    let mut total = vec![0.0; len];
    let mut weight_sum = 0.0;
    for (i, cycle) in plan.cycles.iter().enumerate() {
        // Each cycle has its own stream, so adding a cycle never moves another.
        let mut rng = SfRng::new(derive_seed(raw, STAGE_CALENDAR, i as u64));
        let profile = cycle_profile(cycle, len, &mut rng);
        for (t, p) in total.iter_mut().zip(&profile) {
            *t += cycle.weight * p;
        }
        weight_sum += cycle.weight;
    }
    if weight_sum > 0.0 {
        total.iter_mut().for_each(|t| *t /= weight_sum);
    }
    total
}

fn add_calendar(values: &mut [f64], plan: &CalendarPlan, raw: [u64; 3], base_scale: f64) {
    let profile = calendar_profile(values.len(), plan, raw);
    apply_profile(
        values,
        &profile,
        plan.multiplicative,
        plan.fallback_additive,
        plan.swing,
        plan.amplitude,
        plan.relative,
        base_scale,
    );
}

// ---------------------------------------------------------------------------
// Waveforms
// ---------------------------------------------------------------------------

/// Nominal on-time of one cycle in steps; the whole cycle for sawtooth and level shapes.
pub fn waveform_on_steps(shape: WaveShape, period: usize, duty: f64, width: usize) -> usize {
    match shape {
        WaveShape::Spike => width.min(period - 1),
        WaveShape::Pulse | WaveShape::Trapezoid => {
            ((duty * period as f64).round() as usize).clamp(1, period - 1)
        }
        _ => period,
    }
}

fn whole_cycle(shape: WaveShape) -> bool {
    matches!(
        shape,
        WaveShape::Sawtooth | WaveShape::Levels | WaveShape::Staircase
    )
}

/// Cycle starts (the first at or before step 0, sorted) and on-durations covering `len` steps.
fn waveform_cycles(len: usize, plan: &WaveformPlan, rng: &mut SfRng) -> (Vec<i64>, Vec<i64>) {
    let period = plan.period as i64;
    let on = waveform_on_steps(plan.shape, plan.period, plan.duty, plan.width) as i64;
    let whole = whole_cycle(plan.shape);
    let len = len as i64;
    if plan.timing == WaveTiming::Renewal {
        let duration = |mean: f64, rng: &mut SfRng| -> i64 {
            let cv = plan.cv;
            let draw = if plan.lognormal {
                let sigma2 = cv.mul_add(cv, 1.0).ln();
                rng.lognormal(mean.ln() - sigma2 / 2.0, sigma2.sqrt())
            } else {
                rng.gamma(1.0 / (cv * cv), mean * cv * cv)
            };
            draw.min(1e9).round().max(1.0) as i64
        };
        let mut starts = Vec::new();
        let mut ons = Vec::new();
        let mut cursor = 0i64;
        loop {
            let (on_k, cycle) = if whole {
                let c = duration(period as f64, rng);
                (c, c)
            } else {
                let o = if plan.shape == WaveShape::Spike {
                    on
                } else {
                    duration(on as f64, rng)
                };
                (o, o + duration((period - on) as f64, rng))
            };
            if starts.is_empty() {
                // Start inside the first cycle so step 0 falls at a random point of the process.
                cursor = -(rng.below(cycle as u64) as i64);
            }
            starts.push(cursor);
            ons.push(on_k);
            cursor += cycle;
            // The offset is already subtracted, so this guarantees every step is covered.
            if cursor >= len {
                break;
            }
        }
        return (starts, ons);
    }
    // Cycles -1 .. len / period + 3 cover every step even after a half-period jitter.
    let n = (len / period + 5) as usize;
    let phase = rng.below(plan.period as u64) as i64;
    let mut starts: Vec<i64> = (0..n as i64).map(|k| (k - 1) * period - phase).collect();
    if plan.timing == WaveTiming::Jitter && plan.jitter > 0.0 {
        let half = period / 2;
        for s in starts.iter_mut() {
            let shift = rng.normal(0.0, plan.jitter * period as f64).round() as i64;
            *s += shift.clamp(-half, half);
        }
        starts.sort_unstable();
    }
    let ons = (0..n)
        .map(|k| {
            let gap = if k + 1 < n {
                starts[k + 1] - starts[k]
            } else {
                period
            };
            if whole {
                gap
            } else {
                on.min(gap)
            }
        })
        .collect();
    (starts, ons)
}

/// Level index (0..=top) of every cycle for the 'levels' and 'staircase' shapes.
fn waveform_levels(shape: WaveShape, top: i64, k: usize, rng: &mut SfRng) -> Vec<i64> {
    if shape == WaveShape::Levels {
        // Each cycle jumps to a uniformly drawn other level.
        let mut level = rng.integers(0, top as i32 + 1) as i64;
        (0..k)
            .map(|_| {
                level = (level + rng.integers(1, top as i32 + 1) as i64) % (top + 1);
                level
            })
            .collect()
    } else {
        // A one-level step per cycle in a direction that persists with probability 0.75, folded at both ends.
        let mut direction = if rng.uniform01() < 0.5 { -1i64 } else { 1 };
        let mut walk = rng.integers(0, top as i32 + 1) as i64;
        (0..k)
            .map(|_| {
                if rng.uniform01() < 0.25 {
                    direction = -direction;
                }
                walk += direction;
                let folded = walk.rem_euclid(2 * top);
                if folded <= top {
                    folded
                } else {
                    2 * top - folded
                }
            })
            .collect()
    }
}

/// Template in [0, 1] for one series: one sharp-edged shape per cycle.
fn waveform_template(len: usize, plan: &WaveformPlan, rng: &mut SfRng) -> Vec<f64> {
    let (starts, ons) = waveform_cycles(len, plan, rng);
    let k = starts.len();
    let top = plan.levels as i64 - 1;
    let levels = matches!(plan.shape, WaveShape::Levels | WaveShape::Staircase)
        .then(|| waveform_levels(plan.shape, top, k, rng));
    let mut out = vec![0.0; len];
    let mut c = 0usize;
    for (t, o) in out.iter_mut().enumerate() {
        let t = t as i64;
        // Equal starts (collided jitter) resolve to the later cycle, so empty cycles never show.
        while c + 1 < k && starts[c + 1] <= t {
            c += 1;
        }
        let pos = t - starts[c];
        let on = ons[c];
        *o = match plan.shape {
            WaveShape::Pulse | WaveShape::Spike => f64::from(u8::from(pos < on)),
            WaveShape::Trapezoid => {
                if pos < on {
                    let ramp = ((plan.ramp * on as f64).round() as i64).clamp(1, (on / 2).max(1));
                    let edge = (ramp + 1) as f64;
                    ((pos + 1) as f64 / edge)
                        .min((on - pos) as f64 / edge)
                        .min(1.0)
                } else {
                    0.0
                }
            }
            WaveShape::Sawtooth => ((pos + 1) as f64 / on as f64).min(1.0),
            WaveShape::Levels | WaveShape::Staircase => {
                levels.as_ref().map_or(0.0, |l| l[c] as f64 / top as f64)
            }
        };
    }
    out
}

fn add_waveform(values: &mut [f64], plan: &WaveformPlan, raw: [u64; 3], base_scale: f64) {
    if !values.iter().any(|v| v.is_finite()) {
        return;
    }
    let mut rng = SfRng::new(derive_seed(raw, STAGE_WAVEFORM, 0));
    let sign = if plan.dip { -1.0 } else { 1.0 };
    let template: Vec<f64> = waveform_template(values.len(), plan, &mut rng)
        .into_iter()
        .map(|t| sign * t)
        .collect();
    apply_profile(
        values,
        &template,
        plan.multiplicative,
        plan.fallback_additive,
        plan.swing,
        plan.amplitude,
        plan.relative,
        base_scale,
    );
}

// ---------------------------------------------------------------------------
// Observation process
// ---------------------------------------------------------------------------

/// Saturate, quantize, sample-and-hold and freeze readings; missing values stay missing and never spread.
fn add_observation(values: &mut [f64], plan: &ObservationPlan, raw: [u64; 3]) {
    let n = values.len();
    let finite: Vec<bool> = values.iter().map(|v| v.is_finite()).collect();
    if finite.iter().filter(|&&f| f).count() < 2 {
        return;
    }
    // Grid steps are in standard deviations of the series entering this stage.
    let unit = pi::overlay_scale(values);
    let (low, high) = values
        .iter()
        .filter(|v| v.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let span = high - low;
    // Bounds sit at fractions of the range, so a rare pulse is truncated rather than removed.
    let lower = plan.floor.map_or(f64::NEG_INFINITY, |f| low + f * span);
    let upper = plan.cap.map_or(f64::INFINITY, |c| low + c * span);
    for v in values.iter_mut().filter(|v| v.is_finite()) {
        *v = v.clamp(lower, upper);
    }
    if plan.quantum > 0.0 {
        let q = plan.quantum * unit;
        for v in values.iter_mut().filter(|v| v.is_finite()) {
            *v = (*v / q).round() * q;
        }
    }
    if plan.hold > 1 {
        let mut rng = SfRng::new(derive_seed(raw, STAGE_OBSERVATION, 0));
        let hold = plan.hold;
        let offset = rng.below(hold as u64) as usize;
        // Blocks of `hold` steps, the first one shortened by the random offset; each repeats its first finite reading.
        let mut start = 0usize;
        let mut end = hold - offset;
        while start < n {
            let end_c = end.min(n);
            if let Some(first) = (start..end_c).find(|&t| finite[t]) {
                let held = values[first];
                for t in start..end_c {
                    if finite[t] {
                        values[t] = held;
                    }
                }
            }
            start = end_c;
            end = start + hold;
        }
    }
    if plan.stuck_fraction > 0.0 {
        let mut rng = SfRng::new(derive_seed(raw, STAGE_OBSERVATION, 1));
        let mean_length = plan.stuck_length;
        let count = ((plan.stuck_fraction * n as f64 / mean_length as f64).round() as usize).max(1);
        let low_len = (mean_length / 2).max(2) as i32;
        let high_len = (mean_length + mean_length / 2 + 1) as i32;
        let lengths: Vec<usize> = (0..count)
            .map(|_| rng.integers(low_len, high_len) as usize)
            .collect();
        let starts: Vec<usize> = (0..count).map(|_| rng.below(n as u64) as usize).collect();
        // A frozen window repeats the last finite reading at or before its start.
        let mut last = vec![usize::MAX; n];
        let mut seen = usize::MAX;
        for t in 0..n {
            if finite[t] {
                seen = t;
            }
            last[t] = seen;
        }
        for (start, length) in starts.into_iter().zip(lengths) {
            if last[start] == usize::MAX {
                continue;
            }
            let frozen = values[last[start]];
            for t in start..(start + length).min(n) {
                if finite[t] {
                    values[t] = frozen;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn waveform(shape: WaveShape, timing: WaveTiming, cv: f64) -> WaveformPlan {
        WaveformPlan {
            shape,
            period: 96,
            duty: 0.1,
            width: 1,
            ramp: 0.25,
            levels: 4,
            timing,
            jitter: 0.05,
            lognormal: false,
            cv,
            dip: false,
            multiplicative: false,
            amplitude: 3.0,
            swing: 0.5,
            relative: true,
            fallback_additive: true,
        }
    }

    #[test]
    fn test_derived_seeds_differ_by_stage_and_stream() {
        let raw = [1, 2, 3];
        let a = derive_seed(raw, STAGE_CALENDAR, 0);
        assert_ne!(a, derive_seed(raw, STAGE_CALENDAR, 1));
        assert_ne!(a, derive_seed(raw, STAGE_WAVEFORM, 0));
        assert_ne!(a, derive_seed([1, 2, 4], STAGE_CALENDAR, 0));
        assert_eq!(a, derive_seed(raw, STAGE_CALENDAR, 0));
    }

    #[test]
    fn test_renewal_cycles_cover_the_series() {
        // Heavy-tailed durations used to leave the tail uncovered, pushing sawtooth past 1.
        for shape in [WaveShape::Sawtooth, WaveShape::Pulse, WaveShape::Levels] {
            for seed in 0..500 {
                let plan = waveform(shape, WaveTiming::Renewal, 1.0);
                let mut rng = SfRng::new(seed);
                let (starts, ons) = waveform_cycles(960, &plan, &mut rng);
                assert!(starts[0] <= 0);
                let last = starts.len() - 1;
                let end = starts[last]
                    + if whole_cycle(shape) {
                        ons[last]
                    } else {
                        i64::MAX / 2
                    };
                assert!(end >= 960, "seed {seed}: cycles end at {end}");
                let mut rng = SfRng::new(seed);
                let template = waveform_template(960, &plan, &mut rng);
                assert!(template.iter().all(|&t| (0.0..=1.0).contains(&t)));
            }
        }
    }

    #[test]
    fn test_periodic_pulse_repeats_exactly() {
        let plan = waveform(WaveShape::Pulse, WaveTiming::Periodic, 0.5);
        let template = waveform_template(960, &plan, &mut SfRng::new(3));
        assert_eq!(template[..864], template[96..]);
        assert_eq!(template[..96].iter().sum::<f64>(), 10.0);
    }

    #[test]
    fn test_calendar_profile_is_periodic_with_peak_one() {
        let phase: Vec<f64> = (0..480).map(|t| (t % 24) as f64 / 24.0).collect();
        let plan = CalendarPlan {
            cycles: vec![CyclePlan {
                profile: Profile::Fourier { harmonics: 3 },
                values: None,
                jitter: 0.0,
                weight: 1.0,
                phase,
            }],
            multiplicative: false,
            amplitude: 1.0,
            swing: 0.3,
            relative: false,
            fallback_additive: true,
        };
        let profile = calendar_profile(480, &plan, [5, 6, 7]);
        for t in 0..456 {
            assert!((profile[t] - profile[t + 24]).abs() < 1e-12);
        }
        let peak = profile.iter().fold(0.0_f64, |m, p| m.max(p.abs()));
        assert!(peak <= 1.0 + 1e-12 && peak > 0.9);
    }

    #[test]
    fn test_multiplicative_keeps_zeros_and_falls_back_on_negatives() {
        let profile = vec![0.5; 4];
        let mut counts = vec![0.0, 2.0, 0.0, 4.0];
        apply_profile(&mut counts, &profile, true, true, 0.4, 1.0, false, 1.0);
        assert_eq!(counts, vec![0.0, 2.4, 0.0, 4.8]);
        let mut signed = vec![-1.0, 1.0, f64::NAN, 0.0];
        apply_profile(&mut signed, &profile, true, true, 0.4, 2.0, false, 1.0);
        assert_eq!(signed[..2], [0.0, 2.0]);
        assert!(signed[2].is_nan());
        let mut skipped = vec![-1.0, 1.0];
        apply_profile(&mut skipped, &profile, true, false, 0.4, 2.0, false, 1.0);
        assert_eq!(skipped, vec![-1.0, 1.0]);
    }

    #[test]
    fn test_observation_keeps_missing_values_local() {
        let mut values: Vec<f64> = (0..200).map(|t| (t as f64 * 0.1).sin()).collect();
        values[50] = f64::NAN;
        values[0] = f64::NAN;
        let plan = ObservationPlan {
            floor: Some(0.1),
            cap: Some(0.9),
            quantum: 0.2,
            hold: 4,
            stuck_fraction: 0.2,
            stuck_length: 10,
        };
        add_observation(&mut values, &plan, [1, 2, 3]);
        let missing: Vec<usize> = (0..200).filter(|&t| values[t].is_nan()).collect();
        assert_eq!(missing, vec![0, 50]);
    }

    #[test]
    fn test_released_stages_alone_match_pattern_injection() {
        let base: Vec<f64> = (0..300).map(|t| (t as f64 * 0.05).sin() * 3.0).collect();
        let plan = OverlayPlan {
            changepoints: Some(ChangepointPlan {
                count: 2,
                kind: "mixed".into(),
                locations: vec![],
                level_changes: vec![],
                trend_changes: vec![],
                variance_changes: vec![],
                relative: true,
            }),
            anomalies: Some(AnomalyPlan {
                types: vec!["spike".into(), "level_shift".into()],
                fraction: 0.03,
                spike_magnitude: 5.0,
                dip_magnitude: -5.0,
                level_shift_magnitude: 2.0,
                level_shift_duration: 7,
                relative: true,
            }),
            missing: Some(MissingPlan {
                pattern: "block".into(),
                rate: 0.1,
                block_size: 4,
                seasonal_period: 7,
            }),
            ..OverlayPlan::default()
        };
        let mut fused = base.clone();
        let idx = apply_overlays(&mut fused, &plan, [9, 9, 9], [11, 12, 13]);

        let mut manual = base.clone();
        let scale = pi::overlay_scale(&manual);
        let cp = pi::add_changepoints(&mut manual, 11, 2, &[], "mixed", &[], &[], &[], true, scale);
        let an = pi::add_anomalies(
            &mut manual,
            12,
            &["spike".into(), "level_shift".into()],
            0.03,
            5.0,
            -5.0,
            2.0,
            7,
            scale,
        );
        let miss = pi::add_missingness(&mut manual, 13, "block", 0.1, 4, 7);
        assert_eq!(idx.changepoints, cp.changepoint_indices);
        assert_eq!(idx.anomalies, an.anomaly_indices);
        assert_eq!(idx.missing, miss.missing_indices);
        assert!(fused
            .iter()
            .zip(&manual)
            .all(|(a, b)| a == b || (a.is_nan() && b.is_nan())));
    }
}
