"""Shared helpers for Nixtla-family adapters.

The ``StatsForecastAdapter``, ``MLForecastAdapter``, and
``NeuralForecastAdapter`` all consume the same long-format Nixtla DataFrame
(``unique_id`` / ``ds`` / ``y``) and produce the same panel-shaped numpy
output. This module centralises the validation, panel↔DataFrame conversion,
and cross-validation reshaping logic so each adapter only contains the bits
that are specific to its underlying library.

The helpers take and return narwhals frames and numpy. Native frames exist only
in the adapter modules and in the boundary functions below. ``freq`` arithmetic
goes through utilsforecast, in the frame's own dialect. The functions are pure
and parameterised on column names; adapters keep thin instance-method shims that
forward to them, which preserves the ``adapter._df_to_panel(...)`` API the
integration tests use.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Annotated, Any

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame
from numpy.typing import NDArray
from pydantic import PlainValidator
from utilsforecast import processing as ufp

from ..base import InsufficientCalibrationSamples

# A library-specific resolver that maps (model_name, available_columns, q) to
# the column name that holds the quantile-q forecast. Each Nixtla library
# names columns differently (StatsForecast vs NeuralForecast vs MLForecast);
# this signature lets ``reshape_cv_quantiles`` and ``stack_quantile_panels``
# stay library-agnostic.
ColumnResolver = Callable[[str, list[str], float], str]


def quantiles_to_levels(quantiles: NDArray[np.floating]) -> list[int]:
    """Convert quantiles in (0, 1) to a sorted list of symmetric Nixtla level integers.

    The Nixtla family (StatsForecast, NeuralForecast, MLForecast) all expose
    intervals via integer ``level`` values: ``level=80`` →
    ``<model>-lo-80`` / ``<model>-hi-80`` for the 10th and 90th percentiles.
    The conformalforecast public API speaks in raw quantiles; this helper converts
    a user-supplied quantile array to the level representation Nixtla
    libraries actually consume.

    Parameters
    ----------
    quantiles : NDArray, shape (n_quantiles,)
        Values in ``(0, 1)``. Must come in symmetric pairs around 0.5
        (e.g. ``[0.05, 0.95]``); ``q == 0.5`` is rejected because the
        level API has no median column.

    Returns
    -------
    list of int
        Sorted unique levels.

    Raises
    ------
    ValueError
        If any quantile lies outside ``(0, 1)``, equals 0.5, or lacks its
        symmetric counterpart.
    """
    arr = np.asarray(quantiles, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"quantiles must be 1-D, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("quantiles must contain at least one value.")
    if not np.all((arr > 0.0) & (arr < 1.0)):
        raise ValueError(f"All quantiles must satisfy 0 < q < 1, got {arr.tolist()}.")
    if np.any(np.isclose(arr, 0.5)):
        raise ValueError(
            "Quantile 0.5 (median) is not supported by the level-based API. "
            "Pass strict (0, 0.5) ∪ (0.5, 1) values."
        )
    for q in arr:
        mirror = 1.0 - float(q)
        if not bool(np.any(np.isclose(arr, mirror))):
            raise ValueError(
                f"Quantile {float(q)} requires its symmetric pair {mirror} to "
                f"also be present in quantiles. Got {arr.tolist()}."
            )

    levels: set[int] = set()
    for q in arr:
        q_lo = float(min(float(q), 1.0 - float(q)))
        levels.add(int(round((1.0 - 2.0 * q_lo) * 100.0)))
    return sorted(levels)


def quantile_level(q: float) -> int:
    """Integer level for one quantile (e.g. 0.05 → 90)."""
    return int(round((1.0 - 2.0 * min(q, 1.0 - q)) * 100.0))


def sf_resolve_quantile_column(model_name: str, columns: list[str], q: float) -> str:
    """StatsForecast quantile column resolver.

    StatsForecast uses a deterministic naming convention:
    ``<model>-lo-<level>`` for ``q < 0.5`` and ``<model>-hi-<level>`` for
    ``q > 0.5``. If the expected column is missing, raise with a
    StatsForecast-specific message recommending interval-supporting models.
    """
    side = "lo" if q < 0.5 else "hi"
    col = f"{model_name}-{side}-{quantile_level(q)}"
    if col not in columns:
        raise ValueError(
            f"StatsForecast did not return quantile column '{col}'. "
            f"Model '{model_name}' may not support quantile prediction. "
            "Use a model with native interval support (AutoARIMA, AutoETS, etc.)."
        )
    return col


def nf_resolve_quantile_column(model_name: str, columns: list[str], q: float) -> str:
    """NeuralForecast quantile column resolver.

    NeuralForecast's column naming differs across loss types and versions.
    As of ``neuralforecast==3.1.x``:

    * ``DistributionLoss`` produces ``<model>-lo-<level>`` / ``<model>-hi-<level>``
      (no decimal suffix on the level).
    * ``MQLoss`` and friends produce ``<model>-lo-<level>.0`` /
      ``<model>-hi-<level>.0`` (with a ``.0`` decimal suffix).

    We try both conventions and return the first match. If neither column
    exists, raise ``ValueError`` listing the candidates considered.
    """
    side = "lo" if q < 0.5 else "hi"
    level = quantile_level(q)
    candidates = [
        f"{model_name}-{side}-{level}",
        f"{model_name}-{side}-{level}.0",
    ]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"NeuralForecast did not return quantile column for q={q}. "
        f"Tried {candidates}. Available columns: {columns}. For MQLoss, "
        "the requested quantile must match a symmetric pair the loss was "
        "trained with."
    )


# ----------------------------------------------------------------------
# Boundary: the only functions here that touch native frames or series
# ----------------------------------------------------------------------


def _pandas_or_polars(value: object) -> object:
    """Accept an eager pandas or polars DataFrame unchanged; reject anything else.

    ``pass_through=True`` keeps ``nw.from_native`` from raising ``TypeError`` on a
    non-frame -- pydantic turns only ``ValueError`` into a ``ValidationError``.
    """
    frame = nw.from_native(value, eager_only=True, pass_through=True)
    if not isinstance(frame, nw.DataFrame) or not (
        frame.implementation.is_pandas() or frame.implementation.is_polars()
    ):
        raise ValueError(
            f"train_df must be a pandas or polars DataFrame, got {type(value).__name__}."
        )
    return value


def count_series(df: object, id_col: str) -> int:
    """Distinct ids in ``df``; 1 when it is not a frame or lacks ``id_col``.

    The fallback lets a bad ``train_df`` reach ``TrainFrame``'s validator, which
    names the problem.
    """
    frame = nw.from_native(df, eager_only=True, pass_through=True)
    if isinstance(frame, nw.DataFrame) and id_col in frame.columns:
        return int(frame[id_col].n_unique())
    return 1


TrainFrame = Annotated[IntoDataFrame, PlainValidator(_pandas_or_polars)]
"""``train_df``'s type: a native pandas or polars frame, stored as given.

``PlainValidator`` replaces pydantic's built-in check, which cannot be built for
the ``IntoDataFrame`` Protocol. Neither library is imported to define it.
"""


def offset_times(times: nw.Series[Any], freq: str | int, n: int) -> nw.Series[Any]:
    """Every timestamp moved ``n`` steps of ``freq``, in the frame's own dialect.

    Parameters
    ----------
    times : nw.Series, shape (n,)
    freq : str or int
        A pandas alias for pandas, a polars interval for polars, an int for
        integer time columns.
    n : int
        Steps to move; negative moves backward.

    Returns
    -------
    nw.Series, shape (n,)
        Same length and backend as ``times``.
    """
    return _on_native(ufp.offset_times, times, freq, n)


def time_ranges(starts: nw.Series[Any], freq: str | int, periods: int) -> nw.Series[Any]:
    """``periods`` consecutive times from each start, series-major.

    Parameters
    ----------
    starts : nw.Series, shape (n_series,)
    freq : str or int
        As in :func:`offset_times`.
    periods : int
        Times per start.

    Returns
    -------
    nw.Series, shape (n_series * periods,)
        Block ``i`` (rows ``i * periods`` to ``(i + 1) * periods - 1``) starts at
        ``starts[i]``.
    """
    return _on_native(ufp.time_ranges, starts, freq, periods)


def _on_native(fn: Callable[..., Any], times: nw.Series[Any], *args: Any) -> nw.Series[Any]:
    """Apply a utilsforecast time function to ``times``' native series.

    utilsforecast cannot step a pandas ``ArrowDtype`` timestamp series, so one is
    stepped as numpy-backed datetimes and cast back, keeping frames built from the
    result joinable with ``train_df``.
    """
    native = times.to_native()
    arrow = getattr(native.dtype, "pyarrow_dtype", None) if times.dtype.is_temporal() else None
    if arrow is None:
        return nw.from_native(fn(native, *args), series_only=True)
    out = fn(native.astype(arrow.to_pandas_dtype()), *args).astype(native.dtype)
    return nw.from_native(out, series_only=True)


# ----------------------------------------------------------------------
# Index bookkeeping, in numpy
# ----------------------------------------------------------------------


def _group_layout(
    *keys: NDArray[Any],
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Group structure of rows sorted so that equal key tuples are contiguous.

    Parameters
    ----------
    *keys : NDArray, shape (n_rows,)
        Key columns; a group is a run of rows equal on every key.

    Returns
    -------
    group : NDArray, shape (n_rows,)
        Each row's 0-based group number, in row order.
    position : NDArray, shape (n_rows,)
        Each row's 0-based offset within its group.
    first_row : NDArray, shape (n_groups,)
        The row each group starts at.
    """
    n_rows = int(keys[0].shape[0])
    new = np.zeros(n_rows, dtype=bool)
    if n_rows:
        new[0] = True
        for key in keys:
            new[1:] |= key[1:] != key[:-1]
    first_row = np.flatnonzero(new)
    group = np.cumsum(new) - 1
    position = np.arange(n_rows) - first_row[group] if n_rows else np.zeros(0, dtype=np.intp)
    return group.astype(np.intp), position.astype(np.intp), first_row.astype(np.intp)


def _rows_of(ids: NDArray[Any], series_ids: tuple[Any, ...]) -> NDArray[np.intp]:
    """Position of each id in ``series_ids``. Call it on group-level ids, not rows."""
    where = {sid: i for i, sid in enumerate(series_ids)}
    return np.fromiter((where[sid] for sid in ids), dtype=np.intp, count=len(ids))


def validate_columns(df: nw.DataFrame[Any], *, id_col: str, time_col: str, target_col: str) -> None:
    required = {id_col, time_col, target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train_df is missing required columns: {sorted(missing)}")


def validate_fitted_column_names(
    fitted_names: dict[str, str | None],
    *,
    id_col: str,
    time_col: str,
    target_col: str,
    library_name: str,
    fit_call: str,
) -> None:
    """Check that the library recorded the same column names the adapter declares.

    MLForecast and NeuralForecast both take ``id_col``/``time_col``/``target_col``
    on ``fit`` and remember them: their ``predict`` has no such parameters and
    reuses the recorded names. So the adapter cannot pass custom names into a
    prediction call -- it can only read the output frame with the names it was
    given, and that only matches when the user fit the model with the same ones.
    A mismatch is a user error at construction, not a ``KeyError`` from inside
    the reshaping code several calls later.

    Parameters
    ----------
    fitted_names : dict of str to (str or None)
        The names the library recorded, keyed by ``"id_col"`` / ``"time_col"`` /
        ``"target_col"``. A ``None`` value means the attribute was not found on
        the instance -- an older library version -- and is skipped rather than
        reported as a mismatch.
    id_col, time_col, target_col : str
        The names the adapter was constructed with.
    library_name : str
        Library name for the error message (e.g. ``"MLForecast"``).
    fit_call : str
        The call to name in the remedy (e.g. ``"mlf.fit"``).

    Raises
    ------
    ValueError
        If any recorded name differs from the declared one.
    """
    declared = {"id_col": id_col, "time_col": time_col, "target_col": target_col}
    mismatched = {
        param: (recorded, declared[param])
        for param, recorded in fitted_names.items()
        if recorded is not None and recorded != declared[param]
    }
    if mismatched:
        detail = ", ".join(
            f"{param}: model has {recorded!r}, adapter got {want!r}"
            for param, (recorded, want) in sorted(mismatched.items())
        )
        raise ValueError(
            f"Column names disagree with the fitted {library_name} instance ({detail}). "
            f"{library_name}.predict reuses the names recorded at fit time, so the two "
            f"must match. Either construct the adapter with the fitted names, or refit "
            f"with {fit_call}(df, "
            f"id_col={id_col!r}, time_col={time_col!r}, target_col={target_col!r})."
        )


def validate_freq(df: nw.DataFrame[Any], *, time_col: str, freq: str | int) -> None:
    """Check that ``freq`` can step ``df``'s time column, in the frame's own dialect.

    utilsforecast raises ``ValueError``, ``TypeError`` or ``IndexError`` depending
    on the backend and the mismatch, so all of them become one message here.
    """
    try:
        offset_times(df[time_col].head(1), freq, 1)
    except Exception as exc:
        raise ValueError(
            f"Invalid frequency {freq!r} for a {df.implementation} frame whose "
            f"'{time_col}' column is {df.schema[time_col]}: {exc}. Pass the same freq "
            "you gave the forecasting library: a pandas alias such as 'D', 'h' or "
            "'MS' for pandas, a polars interval such as '1d', '1h' or '1mo' for "
            "polars, or an int for an integer time column."
        ) from exc


def validate_contiguity(
    df: nw.DataFrame[Any], *, id_col: str, time_col: str, freq: str | int
) -> None:
    """Verify each series has timestamps on a regular ``freq`` grid with no gaps.

    A gap, a duplicate or an off-grid stamp is reported at the first position
    where a series leaves its grid. On pandas each timestamp is compared with its
    predecessor stepped one ``freq``, and a first stamp, having no predecessor,
    must survive a step forward and back: an anchored ``freq`` such as ``'MS'``
    snaps an off-anchor stamp onto its grid. polars intervals are not anchored, so
    any first stamp starts a valid grid, and each series is compared with the
    range polars builds from that start -- stepping row by row would reject the
    clipped months of a '1mo' range anchored on day 29-31.
    """
    df = df.sort([id_col, time_col])
    ids = df[id_col].to_numpy()
    group, position, first_row = _group_layout(ids)
    times = df[time_col]
    firsts = times[first_row]
    if df.implementation.is_polars():
        # polars dates a range from its start (Jan 30, Feb 29, Mar 30 at '1mo'),
        # which stepping Feb 29 alone cannot reproduce, so each series is checked
        # against its own start-anchored grid. One call builds every grid.
        width = int(position.max(initial=0)) + 1
        rows = group * width + position
        grids = [time_ranges(firsts, freq, width)[rows]]
        if _all_month_ends(firsts, freq):
            # utilsforecast snaps every range to month ends when every start is
            # one, but a day-30 series starting Sep 30 is on polars' own grid
            # (Oct 30, ...). One start that is not a month end turns the snap off.
            sentinel = firsts.head(1).dt.offset_by("-1d")
            starts = nw.concat([firsts.to_frame(), sentinel.to_frame()])[firsts.name]
            grids.append(time_ranges(starts, freq, width)[rows])
        off = np.stack([times.to_numpy() != g.to_numpy() for g in grids])
        # A series is valid when it follows any one grid from start to end.
        failed = np.stack([np.bincount(group, o, minlength=len(first_row)) > 0 for o in off])
        series = np.flatnonzero(failed.all(axis=0))
        if series.size:
            in_series = group == series[0]
            first_off = [int(np.argmax(o & in_series)) for o in off]
            best = int(np.argmax(first_off))
            i = first_off[best]
            raise ValueError(
                f"Non-contiguous timestamps in series '{ids[i]}': unexpected {times[i]} "
                f"at position {int(position[i])} (previous: {times[i - 1]}, "
                f"expected: {grids[best][i]})."
            )
        return
    round_trip = offset_times(offset_times(firsts, freq, 1), freq, -1)
    off_anchor = np.flatnonzero(firsts.to_numpy() != round_trip.to_numpy())
    if off_anchor.size:
        g = int(off_anchor[0])
        raise ValueError(
            f"Non-contiguous timestamps in series '{ids[first_row[g]]}': first stamp "
            f"{firsts[g]} is not on the {freq!r} grid (nearest grid stamp: {round_trip[g]})."
        )
    expected = offset_times(times, freq, 1).to_numpy()[:-1]
    mismatch = np.flatnonzero((group[1:] == group[:-1]) & (times.to_numpy()[1:] != expected))
    if mismatch.size:
        i = int(mismatch[0]) + 1
        raise ValueError(
            f"Non-contiguous timestamps in series '{ids[i]}': unexpected {times[i]} "
            f"at position {int(position[i])} (previous: {times[i - 1]}, "
            f"expected: {offset_times(times[i - 1 : i], freq, 1)[0]})."
        )


def _all_month_ends(times: nw.Series[Any], freq: str | int) -> bool:
    """Whether utilsforecast would snap a polars ``freq`` step from ``times`` to month ends."""
    if not isinstance(freq, str) or "mo" not in freq or not len(times):
        return False
    return bool((times.dt.offset_by("1d").dt.month() != times.dt.month()).all())


def validate_no_nan(df: nw.DataFrame[Any], *, id_col: str, target_col: str) -> None:
    """Reject a missing target, whether stored as NaN (pandas) or null (polars)."""
    y = nw.col(target_col).cast(nw.Float64)
    offending = df.filter(y.is_null() | y.is_nan())[id_col].unique(maintain_order=True)
    if len(offending):
        raise ValueError(f"NaN values found in '{target_col}' for series: {offending.to_list()}")


def build_series_bounds(
    df: nw.DataFrame[Any], horizon: int, *, id_col: str, time_col: str
) -> tuple[tuple[Any, ...], nw.Series[Any]]:
    """Return ``(series_ids, series_ends)``, keeping every series at its own end.

    Must run after :func:`validate_contiguity`: on a gap-free grid a series spans
    ``rows - 1`` steps, so no calendar arithmetic is needed. Each series must
    span at least ``2 * horizon`` steps.

    Returns
    -------
    series_ids : tuple
        Sorted ids, the panel's row order.
    series_ends : nw.Series, shape (n_series,)
        Each series' last timestamp, aligned with ``series_ids``.

    Raises
    ------
    ValueError
        Naming any series whose span is shorter than ``2 * horizon``.
    """
    bounds = df.group_by(id_col).agg(nw.col(time_col).max(), nw.len().alias("__rows"))
    ids = bounds[id_col].to_list()
    # Python order, as panel.long_to_panel sorts; a frame sort follows category order.
    bounds = bounds[sorted(range(len(ids)), key=ids.__getitem__)]
    series_ids = tuple(bounds[id_col].to_list())
    too_short = [
        str(sid)
        for sid, rows in zip(series_ids, bounds["__rows"].to_list(), strict=True)
        if rows - 1 < 2 * horizon
    ]
    if too_short:
        raise ValueError(
            f"Series {too_short} span fewer than 2 * horizon = {2 * horizon} steps "
            f"and cannot support calibration."
        )
    return series_ids, bounds[time_col]


def df_to_ragged_panel(
    train_df: nw.DataFrame[Any],
    *,
    series_ids: tuple[Any, ...],
    id_col: str,
    time_col: str,
    target_col: str,
) -> NDArray[np.floating]:
    """Training values as a ``(n_series, T)`` panel, in the package's convention.

    ``T`` is the longest series' observation count and each series is *right*
    aligned within it, so column ``-1`` is every series' own last observation and
    a shorter series carries a contiguous leading run of NaN. This matches
    :func:`panel_to_df_ragged`, which dates a panel backwards from each series'
    own end.

    This is the inverse of :func:`panel_to_df_ragged`. Contrast
    :func:`df_to_step_panel`, whose column axis is a step offset from one
    forecast origin rather than each series' own recency; and note that neither
    pivots onto the union of timestamps, which would right-align only when every
    series ends on the same date.
    """
    df = train_df.filter(nw.col(id_col).is_in(list(series_ids))).sort([id_col, time_col])
    ids = df[id_col].to_numpy()
    group, position, first_row = _group_layout(ids)
    row = _rows_of(ids[first_row], series_ids)[group]
    counts = np.bincount(row, minlength=len(series_ids))
    panel = np.full((len(series_ids), int(counts.max())), np.nan, dtype=np.float64)
    panel[row, panel.shape[1] - counts[row] + position] = np.asarray(
        df[target_col].to_numpy(), dtype=np.float64
    )
    return panel


def _panels_match(left: NDArray[np.floating], right: NDArray[np.floating]) -> bool:
    """Value-equality for two equally-shaped panels, treating NaN as equal.

    Tolerant of last-bit float noise (a history may have round-tripped through a
    DataFrame) but not of anything larger -- the point of the comparison is to
    tell two *different* stretches of a series apart.
    """
    return bool(
        left.shape == right.shape
        and np.allclose(left, right, rtol=1e-9, atol=1e-12, equal_nan=True)
    )


def anchor_shift_for_panel(
    panel: NDArray[np.floating],
    *,
    train_panel: NDArray[np.floating],
) -> int:
    """Steps from the training end to ``panel``'s end, found by matching values.

    A panel carries no dates, so ``predict`` has to work out which stretch of
    time its columns represent. Deriving that from the width alone is wrong: a
    caller who passes the most recent ``T`` observations gets the *first* ``T``
    observations' dates, and mlforecast then builds ``date_features`` from them.
    The values are right, every calendar feature is wrong, and nothing raises.

    So the panel is located by value instead. Both panels are right-aligned per
    series (see :func:`df_to_ragged_panel`), which makes the three cases:

    - ``panel`` is the training panel's **prefix** -- the expanding-window
      ``calibrate`` loop, ``full[:, :cut]``. Its end is ``T - train_T`` steps
      before the training end.
    - ``panel`` is the training panel's **suffix** -- a fixed-width sliding
      window, ``full[:, -T:]``. It ends where training ends.
    - ``panel`` **extends** the training panel -- newer observations appended.
      Its end is ``T - train_T`` steps after the training end.

    Prefix is tested before suffix, so degenerate data that matches both (a
    constant series, say) keeps the historically documented reading.

    Parameters
    ----------
    panel : Series, shape (n_series, T)
    train_panel : Series, shape (n_series, train_T)
        The training values, from :func:`df_to_ragged_panel`.

    Returns
    -------
    int
        The shift :func:`panel_to_df_ragged` and :func:`exog_to_futr_df` take:
        0 for the training end, negative for a prefix, positive for an extension.

    Raises
    ------
    ValueError
        If ``panel`` is none of the three, so its dates cannot be inferred.
    """
    width, train_width = int(panel.shape[1]), int(train_panel.shape[1])
    overlap = min(width, train_width)

    # Compare on the overlap, once anchored at each edge. Left-aligned means the
    # two share a start, so the panel's end moved by the width difference;
    # right-aligned means they share an end, whatever the widths.
    if _panels_match(panel[:, :overlap], train_panel[:, :overlap]):
        delta = width - train_width
    elif _panels_match(panel[:, width - overlap :], train_panel[:, train_width - overlap :]):
        delta = 0
    else:
        raise ValueError(
            f"Cannot date a history of width {width}: aligned against the "
            f"{train_width}-column training panel at either edge, its values are not the "
            "training data, so the timestamps its columns stand for are unknown. Pass a "
            "prefix (`panel[:, :cut]`), a recent window (`panel[:, -w:]`), or an extension "
            "of the training data -- or calibrate with `calibrate_cv`, which never needs a "
            "history panel dated by hand. To forecast from data the adapter has not seen, "
            "call `refit` first."
        )

    return delta


def grid_stamps(
    like: nw.DataFrame[Any],
    offsets: NDArray[np.intp],
    *,
    series_ids: tuple[Any, ...],
    id_col: str,
    time_col: str,
    freq: str | int,
) -> nw.DataFrame[Any]:
    """``(id, time)`` rows at step ``offsets`` from each series' training end.

    A step at or before the end is read from ``like``'s own stamps -- validated
    contiguous -- so a history is dated exactly as the training frame is, with no
    calendar arithmetic. That matters for polars '1mo': utilsforecast snaps a step
    to month ends whenever every stamp it moves is one, so a series anchored on
    day 30 that ends on Apr 30 would step back to Mar 31. Only steps past the end
    use :func:`offset_times`, which is how the forecasting library dates them too.
    The id column is read from ``like`` as well, so it keeps its exact dtype.

    Parameters
    ----------
    like : nw.DataFrame
        The training frame.
    offsets : NDArray, shape (n_series, width)
        Steps from each series' last training stamp; 0 is that stamp. A step
        before the series' first stamp yields the first stamp: the caller drops
        those cells, which stand for the panel's left-NaN padding.

    Returns
    -------
    nw.DataFrame, n_series * width rows
        Columns ``[id_col, time_col]``, series-major in ``series_ids`` order.
    """
    df = like.select(id_col, time_col).sort([id_col, time_col])
    ids = df[id_col].to_numpy()
    _, _, first_row = _group_layout(ids)
    row = _rows_of(ids[first_row], series_ids)
    start = np.empty(len(series_ids), dtype=np.intp)
    last = np.empty(len(series_ids), dtype=np.intp)
    start[row] = first_row
    last[row] = np.append(first_row[1:], len(df)) - 1
    cell = np.maximum(last[:, None] + offsets, start[:, None])
    ahead = int(offsets.max(initial=0))
    if ahead > 0:
        tail = df[last]
        future = nw.maybe_reset_index(tail[np.repeat(np.arange(len(series_ids)), ahead)])
        future = future.with_columns(
            time_ranges(offset_times(tail[time_col], freq, 1), freq, ahead).alias(time_col)
        )
        beyond = len(df) + np.arange(len(series_ids))[:, None] * ahead + offsets - 1
        cell = np.where(offsets > 0, beyond, np.minimum(cell, last[:, None]))
        df = nw.concat([df, future])
    return nw.maybe_reset_index(df[cell.ravel()])


def panel_to_df_ragged(
    panel: NDArray[np.floating],
    *,
    series_ids: tuple[Any, ...],
    shift: int,
    id_col: str,
    time_col: str,
    target_col: str,
    freq: str | int,
    like: nw.DataFrame[Any],
) -> nw.DataFrame[Any]:
    """Convert a ``(n_series, T)`` panel to a long frame, per-series ragged.

    Each series' non-NaN values are timestamped backward from its own end,
    ``shift`` steps from its training end (see :func:`grid_stamps`). Left-NaN
    padding is dropped (not emitted). Because the panel is right-aligned, a full
    ``T``-step grid ending at each series' end covers every value, and the NaN
    mask selects the real ones.

    Parameters
    ----------
    panel : NDArray, shape (n_series, T)
    shift : int
        Steps from the training end to the panel's end, from
        :func:`anchor_shift_for_panel`.
    like : nw.DataFrame
        The training frame. The result has its backend and its id dtype.
    """
    n_series, width = panel.shape
    offsets = np.broadcast_to(shift - width + 1 + np.arange(width), (n_series, width))
    frame = grid_stamps(
        like, offsets, series_ids=series_ids, id_col=id_col, time_col=time_col, freq=freq
    ).with_columns(
        nw.new_series(target_col, panel.ravel(), backend=like.implementation),
        nw.new_series("__keep", ~np.isnan(panel).ravel(), backend=like.implementation),
    )
    return frame.filter(nw.col("__keep")).select(id_col, time_col, target_col)


def df_to_step_panel(
    df: nw.DataFrame[Any],
    value_col: str,
    *,
    series_ids: tuple[Any, ...],
    id_col: str,
    time_col: str,
    n_steps: int,
) -> NDArray[np.floating]:
    """Pivot one forecast origin to ``(n_series, n_steps)``, indexed by step.

    The horizon axis is each series' *own* step offset -- its rows sorted by
    ``time_col`` are steps ``0 .. n_steps - 1`` -- not a shared calendar.
    Forecasts for series that end on different dates carry different timestamps,
    so pivoting onto the union of those timestamps would widen the column axis
    past ``n_steps`` and scatter each series across its own diagonal. When every
    series does share an end date the two agree exactly.

    Every series in ``series_ids`` must be present with the full ``n_steps``.
    A forecast origin is not a cross-validation cutoff: "this window does not
    exist for this series" has no meaning here, and NaN-padding a missing series
    would put NaN into the returned intervals with nothing raised. The
    cross-validation path builds its own axes in :func:`reshape_cv_many`.

    Parameters
    ----------
    df : nw.DataFrame
        Long frame for a single forecast origin.
    value_col : str
        Column holding the values to extract.
    series_ids : tuple
        Row order of the result.
    id_col, time_col : str
    n_steps : int
        Width of the step axis, i.e. the horizon.

    Returns
    -------
    NDArray, shape (n_series, n_steps)

    Raises
    ------
    ValueError
        If any series is missing from ``df`` or does not contribute exactly
        ``n_steps`` rows, naming the series and what it contributed.
    """
    df = df.filter(nw.col(id_col).is_in(list(series_ids))).sort([id_col, time_col])
    ids = df[id_col].to_numpy()
    group, position, first_row = _group_layout(ids)
    sizes = np.diff(np.append(first_row, len(df)))
    present = {sid: int(n) for sid, n in zip(ids[first_row], sizes, strict=True)}
    wrong = [
        f"'{sid}' is missing" if sid not in present else f"'{sid}' has {present[sid]}"
        for sid in series_ids
        if present.get(sid) != n_steps
    ]
    if wrong:
        raise ValueError(
            f"Malformed forecast frame: expected {n_steps} rows for every series, "
            f"but {', '.join(wrong)}. Every series the adapter was fitted on must "
            "appear at a forecast origin, so a missing or partial one means the "
            "upstream output was truncated."
        )
    panel = np.full((len(series_ids), n_steps), np.nan, dtype=np.float64)
    row = _rows_of(ids[first_row], series_ids)[group]
    panel[row, position] = np.asarray(df[value_col].to_numpy(), dtype=np.float64)
    return panel


def reshape_cv_many(
    cv_df: nw.DataFrame[Any],
    value_cols: Sequence[str],
    *,
    series_ids: tuple[Any, ...],
    id_col: str,
    time_col: str,
    horizon: int,
    aligned: bool,
    strict_nan: bool = False,
    library_name: str = "The forecasting library",
) -> list[NDArray[np.floating]]:
    """Reshape Nixtla-format CV output to the arrays the methods consume.

    One pass over ``cv_df`` produces every requested column at once. Each row's
    destination is computed up front -- series on axis 0, window on axis 1, step
    within the cutoff group on axis 2 -- and written with a single scatter, so
    the allocation is the size of the *result* rather than of the calendar.

    That distinction matters for ragged panels. Libraries that derive cutoffs
    from each series' own end date (StatsForecast does) hand back disjoint cutoff
    sets when series end on different dates, and the union of those sets grows as
    ``n_series * n_windows``. ``aligned=True`` asks for that union and pays for
    it; ``aligned=False`` does not, and never builds it.

    Whether a ``(series, cutoff)`` cell exists is read from the frame's own
    grouping, so "absent" and "present but NaN" stay distinguishable without a
    second pass.

    Parameters
    ----------
    cv_df : nw.DataFrame
        Long-format cross-validation output, with a ``cutoff`` column. Rows for
        series outside ``series_ids`` are ignored.
    value_cols : sequence of str
        Columns to extract. All of them share one window selection, so the
        returned cubes line up cell for cell.
    series_ids : tuple
        Row order of the results.
    id_col, time_col : str
    horizon : int
        Width of the step axis.
    aligned : bool
        ``True`` puts the union of cutoffs on axis 1, NaN where a series has no
        cell. ``False`` gives each series its own most-recent windows, truncated
        to the smallest per-series count so the axis stays rectangular.
    strict_nan : bool
        Raise if a cell the frame *does* contain holds NaN. Used by the quantile
        path, where NaN can only mean an upstream defect.
    library_name : str
        Named in the ``strict_nan`` error message.

    Returns
    -------
    list of NDArray, each shape (n_series, n_windows, horizon)
        One array per entry in ``value_cols``, in the same order.

    Raises
    ------
    ValueError
        If a ``(series, cutoff)`` group does not hold exactly ``horizon`` rows,
        naming the offenders.
    InsufficientCalibrationSamples
        If ``aligned`` is False and a series has no usable window, naming those
        series and carrying their indices. This is a structural limit -- there
        is no rectangular array to return -- not a second copy of the method's
        ``ceil(1 / alpha)`` check, which stays in the method's ``fit`` because
        it is the only validation the embedded array surface and ``calibrate()``
        paths get.
    RuntimeError
        If ``strict_nan`` and a returned cell holds NaN.
    """
    cols = list(value_cols)
    df = cv_df.filter(nw.col(id_col).is_in(list(series_ids))).sort([id_col, "cutoff", time_col])
    ids = df[id_col].to_numpy()
    cell, step, first_row = _group_layout(ids, df["cutoff"].to_numpy())
    sizes = np.diff(np.append(first_row, len(df)))
    malformed = np.flatnonzero(sizes != horizon)
    if malformed.size:
        detail = ", ".join(
            f"'{ids[first_row[c]]}' at cutoff {df.item(int(first_row[c]), 'cutoff')} has {sizes[c]}"
            for c in malformed
        )
        raise ValueError(
            f"Malformed cross-validation frame: expected {horizon} rows per "
            f"(series, cutoff), but {detail}. A series may be absent from a "
            "cutoff entirely, but a partial horizon means the upstream output "
            "was truncated."
        )
    # Null (polars) and NaN (pandas) both reach numpy as NaN in a float array.
    values = np.column_stack([np.asarray(df[c].to_numpy(), dtype=np.float64) for c in cols])
    # A cell is usable iff every requested column is non-NaN across the horizon.
    holed = (
        np.logical_or.reduceat(np.isnan(values).any(axis=1), first_row)
        if len(df)
        else np.zeros(0, dtype=bool)
    )
    if strict_nan and bool(holed.any()):
        raise RuntimeError(
            f"{library_name} cross_validation produced NaN values for "
            "(series, cutoff) pairs it did return rows for. This indicates an "
            f"internal error in the {library_name} cross-validation output."
        )
    row_of_cell = _rows_of(ids[first_row], series_ids)
    if aligned:
        # The calendar axis is the deliverable here, so every present cell keeps
        # its column -- including a holed one, whose NaN is the answer. A series
        # with no usable cell at all survives as an all-NaN row, and the method
        # decides whether what is left suffices -- it is what knows ``alpha``.
        cutoffs, window_of_cell = np.unique(df["cutoff"].to_numpy()[first_row], return_inverse=True)
        n_windows = len(cutoffs)
        window_of_cell = window_of_cell.reshape(-1)
    else:
        usable = np.flatnonzero(~holed)
        counts = np.bincount(row_of_cell[usable], minlength=len(series_ids))
        # Structural, not a second copy of the method's ``ceil(1/alpha)`` check:
        # a series with no usable window makes ``n_windows`` zero, and there is
        # no rectangular array left to return. Typed so a caller catches the same
        # exception wherever it was noticed, with ids the method cannot see.
        starved = np.flatnonzero(counts == 0).tolist()
        if starved:
            raise InsufficientCalibrationSamples(
                f"Series {[str(series_ids[i]) for i in starved]} have no complete "
                "cross-validation window. Their histories end too early to reach "
                "any cutoff the other series define, so there is nothing to "
                "calibrate on. Drop them, extend them, or lower n_windows.",
                starved,
            )
        n_windows = int(counts.min())
        # Usable cells are in (series, cutoff) order, so counting back from each
        # series' end gives its age: 0 is the newest usable cutoff.
        _, rank, _ = _group_layout(row_of_cell[usable])
        age = counts[row_of_cell[usable]] - 1 - rank
        recent = age < n_windows
        window_of_cell = np.full(len(first_row), -1, dtype=np.intp)
        window_of_cell[usable[recent]] = n_windows - 1 - age[recent]
    keep = window_of_cell[cell] >= 0
    row = row_of_cell[cell][keep]
    window = window_of_cell[cell][keep]
    step = step[keep]
    out: list[NDArray[np.floating]] = []
    for j in range(len(cols)):
        cube = np.full((len(series_ids), n_windows, horizon), np.nan, dtype=np.float64)
        cube[row, window, step] = values[keep, j]
        out.append(cube)
    return out


def reshape_cv_quantiles(
    cv_df: nw.DataFrame[Any],
    *,
    model_name: str,
    target_col: str,
    q_arr: NDArray[np.floating],
    resolver: ColumnResolver,
    series_ids: tuple[Any, ...],
    id_col: str,
    time_col: str,
    horizon: int,
    library_name: str,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Reshape a Nixtla-format quantile CV DataFrame.

    Resolves the quantile column for each entry in ``q_arr`` via ``resolver``
    (which encapsulates each library's column-name conventions and "missing
    column" error message), then reshapes truths and every quantile column in a
    single :func:`reshape_cv_many` pass -- so they share one window selection --
    and stacks the panels with quantiles on the **last axis**.

    A series that does not reach a cutoff simply contributes no cell there, and
    the dense pass keeps each series' own windows, so the returned arrays carry
    no NaN. NaN in a cell the frame *does* contain is an upstream defect and
    raises ``RuntimeError``.

    Returns
    -------
    quantile_predictions : NDArray, shape ``(n_series, n_windows, horizon, n_quantiles)``
        Quantiles on the last axis, in the order of ``q_arr``.
    truths : NDArray, shape ``(n_series, n_windows, horizon)``
    """
    columns = list(cv_df.columns)
    dense = reshape_cv_many(
        cv_df,
        [target_col, *(resolver(model_name, columns, float(q)) for q in q_arr)],
        series_ids=series_ids,
        id_col=id_col,
        time_col=time_col,
        horizon=horizon,
        aligned=False,
        strict_nan=True,
        library_name=library_name,
    )
    truths = dense[0]
    quantile_predictions = np.stack(dense[1:], axis=-1)
    return quantile_predictions, truths


def validate_exog_columns(
    train_df: nw.DataFrame[Any],
    *,
    exog_features: list[str] | None,
) -> None:
    """Every declared regressor must carry its history in ``train_df``.

    ``predict`` re-supplies the historical values to the underlying library,
    which re-derives its regressor count from the frame it is handed. Without
    them the shapes disagree and the library raises from deep inside its own
    validation, naming neither the column nor the cause.

    Parameters
    ----------
    train_df : nw.DataFrame
    exog_features : list of str or None
        Nothing is checked when ``None``.

    Raises
    ------
    ValueError
        Naming the declared features that ``train_df`` does not carry.
    """
    if exog_features is None:
        return
    missing = [c for c in exog_features if c not in train_df.columns]
    if missing:
        raise ValueError(
            f"exog_features {missing} are not columns of train_df "
            f"(got {sorted(train_df.columns)}). Pass the frame the model was "
            "fitted on, including the regressors' historical values -- predict "
            "hands them back to the forecasting library alongside the future "
            "values in X."
        )


def attach_exog_history(
    history_df: nw.DataFrame[Any],
    *,
    train_df: nw.DataFrame[Any],
    exog_features: list[str],
    id_col: str,
    time_col: str,
) -> nw.DataFrame[Any]:
    """Join the regressors' historical values onto a dated history frame.

    The join key is ``(id, time)``, which is sound because
    :func:`panel_to_df_ragged` stamps its timestamps from the adapter's anchor --
    the same anchor :func:`exog_to_futr_df` continues from. A prefix panel is
    therefore dated earlier and picks up the matching historical rows.

    Shared by every Nixtla adapter that carries exogenous features: all three
    libraries reject a target-only history frame, each with its own opaque
    message (StatsForecast reports a shape mismatch, MLForecast a missing
    feature name, NeuralForecast a missing exogenous variable).

    Parameters
    ----------
    history_df : nw.DataFrame
        Long frame from :func:`panel_to_df_ragged`.
    train_df : nw.DataFrame
        The stored training frame, carrying ``exog_features``.
    exog_features : list of str
    id_col, time_col : str

    Returns
    -------
    nw.DataFrame
        ``history_df`` with the exogenous columns joined on.

    Raises
    ------
    ValueError
        If any history row has no regressor values. That means the panel extends
        past the training data, where the regressors are unknown by definition.
    """
    # The indicator, not the regressor values, marks an unmatched row: a NaN
    # regressor reads as null on pandas but not on polars.
    right = train_df.select(id_col, time_col, *exog_features).with_columns(__matched=nw.lit(True))
    merged = history_df.join(right, on=[id_col, time_col], how="left").sort([id_col, time_col])
    unknown = merged.filter(nw.col("__matched").is_null())
    merged = merged.drop("__matched")
    if len(unknown):
        dates = sorted(set(unknown[time_col].to_list()))
        shown = ", ".join(str(d) for d in dates[:5])
        more = f" (and {len(dates) - 5} more)" if len(dates) > 5 else ""
        raise ValueError(
            f"No values for exogenous features {exog_features} at "
            f"{len(unknown)} of {len(merged)} history steps, dated "
            f"{shown}{more}. The history extends past the training frame, so "
            "the regressors there are unknown. Call refit with a frame that "
            "carries them, or forecast from a history the adapter has seen."
        )
    return merged


def exog_to_futr_df(
    exog: NDArray[np.floating],
    *,
    series_ids: tuple[Any, ...],
    shift: int,
    feature_names: list[str],
    id_col: str,
    time_col: str,
    freq: str | int,
    like: nw.DataFrame[Any],
) -> nw.DataFrame[Any]:
    """Build a future-regressor frame (``X_df``) from a 4-D exogenous array.

    Parameters
    ----------
    exog : NDArray, shape (n_series, 1, horizon, n_features)
        Future exogenous features (single forecast origin).
    series_ids : tuple
    shift : int
        Steps from the training end to the history's end, from
        :func:`anchor_shift_for_panel`; future steps follow that end.
    feature_names : list of str
        Column names for the ``n_features`` axis, in order.
    id_col, time_col : str
    freq : str or int
        Step size in the frame's own dialect.
    like : nw.DataFrame
        The training frame. The result has its backend and its id dtype.

    Returns
    -------
    nw.DataFrame
        Columns ``[id_col, time_col, *feature_names]``, one row per
        ``(series, future step)``.

    Raises
    ------
    ValueError
        If ``exog`` is not 4-D, its leading axis is not ``len(series_ids)``,
        ``len(feature_names)`` != the feature axis size, or ``exog`` carries
        more than one forecast origin (``n_samples != 1``).
    """
    if exog.ndim != 4:
        raise ValueError(
            f"exog must be 4-D (n_series, n_samples, horizon, n_features), got shape {exog.shape}"
        )
    if exog.shape[0] != len(series_ids):
        raise ValueError(
            f"series_ids length {len(series_ids)} != exog leading axis {exog.shape[0]}."
        )
    if exog.shape[-1] != len(feature_names):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but exog has "
            f"{exog.shape[-1]} features."
        )
    if exog.shape[1] != 1:
        raise ValueError(
            f"exog must describe one forecast origin, shape (n_series, 1, horizon, "
            f"n_features); got {exog.shape}."
        )
    n_series, _, horizon, _ = exog.shape
    flat = exog[:, 0].reshape(n_series * horizon, -1)
    ends = grid_stamps(
        like,
        np.full((n_series, 1), shift),
        series_ids=series_ids,
        id_col=id_col,
        time_col=time_col,
        freq=freq,
    )
    times = time_ranges(offset_times(ends[time_col], freq, 1), freq, horizon)
    frame = nw.maybe_reset_index(ends[np.repeat(np.arange(n_series), horizon)])
    return frame.with_columns(
        times.alias(time_col),
        *(
            nw.new_series(name, flat[:, j], backend=like.implementation)
            for j, name in enumerate(feature_names)
        ),
    )


def stack_quantile_panels(
    result_df: nw.DataFrame[Any],
    *,
    model_name: str,
    q_arr: NDArray[np.floating],
    resolver: ColumnResolver,
    series_ids: tuple[Any, ...],
    id_col: str,
    time_col: str,
    horizon: int,
) -> NDArray[np.floating]:
    """Read each requested quantile column from ``result_df`` and stack.

    The forecast-side counterpart to :func:`reshape_cv_quantiles`. Used by
    ``predict_quantiles`` on each Nixtla adapter: the upstream library has
    just produced a wide DataFrame indexed by (id, ds) with one column per
    quantile, and this helper pivots each requested quantile to a panel via
    :func:`df_to_step_panel` -- so series forecast from different end dates
    still line up on the step axis -- and stacks them with quantiles on the
    **middle axis**.

    Returns
    -------
    NDArray, shape ``(n_series, n_quantiles, horizon)``
    """
    columns = list(result_df.columns)
    panels: list[NDArray[np.floating]] = []
    for q in q_arr:
        col = resolver(model_name, columns, float(q))
        panels.append(
            df_to_step_panel(
                result_df,
                col,
                series_ids=series_ids,
                id_col=id_col,
                time_col=time_col,
                n_steps=horizon,
            )
        )
    return np.stack(panels, axis=1)
