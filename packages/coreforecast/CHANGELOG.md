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
  levels outside `[0, 1]`, `indptr` entries that are negative, decreasing or
  too large for an int32, out-of-range group indices, and `ds`, `stats`,
  `periods` and `tails` arrays whose size doesn't match the group count.
  Calling `update` on a lag transform built with `lag=0` also raises now: an
  update reads the value that isn't in the array yet, so there is nothing for
  it to consume.

### Documentation

- The local scalers' `skipna` documentation said an interior NaN "may result in
  NaN statistics". It doesn't for min-max, where the result is whichever value
  wins an unordered comparison. With `skipna=False` only a leading run of NaNs
  is supported; pass `skipna=True` for anything else.
