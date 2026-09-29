# Changelog

## Unreleased

### Breaking changes

- The free functions in `coreforecast.rolling`, `coreforecast.expanding` and
  `coreforecast.exponentially_weighted` now preserve a leading run of NaNs and
  compute the statistic over what follows it. They used to return all NaN as
  soon as the series started with one. With `skipna=True`, `min_samples` counts
  positions after that run, which is what pandas does. The seasonal variants
  split what follows the run into seasons, so a leading run whose length isn't a
  multiple of `season_length` now groups different positions together than it
  used to.
- Input without a float dtype (integer arrays, lists, Series) is now widened to
  float64 instead of float32. Values from `2**24` on were silently rounded
  before; the result now takes twice the memory. Pass a float32 array to get
  float32 back, which was already the only way to be sure of it.
- Arguments that used to read or write out of bounds are rejected with a
  `ValueError` or an `IndexError` instead: non-positive `window_size`,
  `min_samples` and `season_length`, negative `lag`, `d` and `k`, quantile
  levels outside `[0, 1]`, `indptr` entries that are negative or decreasing,
  out-of-range group indices, and `ds`, `stats`,
  `periods` and `tails` arrays whose size doesn't match the group count.
  Calling `update` on a lag transform built with `lag=0` also raises now: an
  update reads the value that isn't in the array yet, so there is nothing for
  it to consume. A negative `max_season_length` is a `ValueError` like the
  other counts rather than a `TypeError` from the binding.
- `GroupedArray.indptr` is stored and returned as `int64` instead of `int32`,
  so arrays are no longer limited to `2**31` elements. Any integer dtype is
  still accepted as input; the values that used to be rejected as "not
  representable with 32-bit integers" are now checked like every other offset.
  It is signed rather than unsigned so that arithmetic mixing it with numpy's
  default integer stays integral instead of promoting to float64.

### Bug fixes

- `ExpandingMean`, `ExpandingStd`, `ExpandingMin`, `ExpandingMax` and
  `ExponentiallyWeightedMean` ignored `skipna` in `update()`. Every other lag
  transform forwards it to a kernel, but these five carry their accumulator in
  Python and incorporated a NaN unconditionally, so one arriving through the
  incremental path made the statistic NaN for that group permanently even with
  `skipna=True`. `transform()` was always correct, so training features and
  the features produced while predicting recursively disagreed with nothing
  raised. With `skipna=True` the mean and std now leave their state untouched
  by a NaN, the min and max ignore it and the EWM forward-fills its mean, and
  a group whose whole lagged history was NaN, which the driver skips and whose
  stats row it fills with NaN, seeds from the first value it sees, as the
  transform does after a leading run of NaNs. Without `skipna` the updates are
  unchanged.
- `ExpandingMin`, `ExpandingMax` and `ExponentiallyWeightedMean` seeded the
  state of an empty group from the output position before its end, which
  belongs to another group, so a value the group got later through `update()`
  was folded into that group's statistic. `ExpandingMean` read the same
  position, and on an array with no elements the read raised an `IndexError`
  from all four. An empty group now starts from a NaN state, as with the
  other accumulators.
- `LocalBoxCoxScaler.stats_` had an uninitialised second column: the lambda
  kernels write one value per group and the array holding them was never
  cleared. Transforms were unaffected since that column is not read, but the
  attribute differed between identical fits. It is now deterministic padding:
  zero for a group that produced a lambda, NaN for an empty or all-NaN group,
  whose whole row the driver fills.

### Performance

- Rolling min and max, with their seasonal, update and expanding variants, use
  a block scan (van Herk / Gil-Werman) instead of a monotonic deque. The deque
  popped entries in a loop whose exit depended on the data, which mispredicted
  about once per element; the scan does three compare-selects per element with
  no such branch. About 3x faster on random data, 2.6 to 3.9 ns per element
  instead of 9.5 on a Neoverse N1, and the same on monotonic data. Results are
  unchanged.

### Build

- The Eigen submodule is gone; the handful of reductions it backed are plain
  loops now. Source builds no longer need to fetch it. The box-cox lambda
  estimators run about twice as fast because their transform is evaluated once
  instead of once per statistic; results may differ from previous versions in
  the last bits because the summation order changed.
- The kernels have their own tests under `tests/cpp`, built with
  `-DCOREFORECAST_BUILD_TESTS=ON` against the `external_libs/doctest`
  submodule; existing checkouts need `git submodule update --init` for it.
  They are not part of the sdist or the wheels.
- The private `_lib.rolling.rolling_quantile` and `seasonal_rolling_quantile`
  take `p` right after `data`, the order the public functions already used.

### Documentation

- The local scalers' `skipna` documentation said an interior NaN "may result in
  NaN statistics". It doesn't for min-max, where the result is whichever value
  wins an unordered comparison. With `skipna=False` only a leading run of NaNs
  is supported; pass `skipna=True` for anything else.
