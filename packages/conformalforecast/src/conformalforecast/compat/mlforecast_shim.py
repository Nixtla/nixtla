"""Reference migration shim: a ``ConformalMethod`` behind mlforecast's interval hook.

Status
------
This is a **reference implementation for the upstream mlforecast migration PR**,
not a supported public API. The eventual mlforecast PR is expected to copy this
pattern in-repo, after which this module can be deleted.

What mlforecast's hook looks like (>= 1.1)
------------------------------------------
``mlforecast.conformal_prediction`` holds a module-level registry and a getter::

    _INTERVAL_METHODS = {"conformal_distribution": ..., "conformal_error": ...,
                         "weighted_conformal_distribution": ...,
                         "weighted_conformal_error": ...}

    def get_conformal_method(method: str) -> Callable: ...

Every registered function is called with ``fcst_df`` and ``cs_df`` positionally
and the rest by keyword::

    fn(fcst_df, cs_df, model_names=..., level=..., cs_h=..., cs_n_windows=...,
       n_series=..., horizon=..., weights=..., is_transfer=...)

plus ``target_weights=`` when the registered name does not start with
``"conformal_"``. :func:`register_interval_method` inserts a method-backed hook
into that registry; the lookup happens per call, so registering after import
works.

**Known upstream gap.** ``PredictionIntervals.__init__`` validates ``method``
against a hard-coded list rather than ``_INTERVAL_METHODS.keys()``, so a newly
registered name is reachable through :func:`get_conformal_method` but not
through ``PredictionIntervals(method=...)``. The upstream fix is one line
(``allowed_methods = list(_INTERVAL_METHODS)``). Until then, callers wanting an
end-to-end run override an existing key instead of adding a new one.

**Unsupported mlforecast 1.1 features.** ``weights`` / ``target_weights``
(density-ratio reweighting) and ``is_transfer=True`` (pooled cross-domain
calibration) raise :class:`NotImplementedError`. NexCP's ``rho`` decay is a
different mechanism and must not be silently substituted for DRE weights.

Layout mirrored from mlforecast
-------------------------------
- ``cs_df[model]`` is window-major: ``reshape(cs_n_windows, n_series, cs_h)``.
  The methods want ``(n_series, n_windows, horizon)``, hence one transpose.
  (``panel.long_to_panel`` is deliberately *not* used: ``cs_df`` is already a
  contiguous cube, not a long panel needing a pivot, and pivoting would re-sort
  the series axis away from mlforecast's own order.)
- ``fcst_df[model]`` is series-major of length ``n_series * horizon``.
- ``cs_h == 1`` with ``horizon > 1`` means one constant width for every step;
  the score column is tiled across the horizon, matching mlforecast.

Score-function limitation (upstream requirement)
------------------------------------------------
``cs_df`` holds ``abs(prediction - truth)``; mlforecast drops the target column
before the hook is called, so the *signed* residual is unrecoverable. The shim
reconstructs the method's input as ``predictions = 0``, ``truths = scores``,
which is exact under :class:`AbsoluteResidual` and wrong under any signed or
quantile score. Methods whose default score is anything else — and
``ConformalizedQuantileRegression``, which needs ``quantile_predictions``
mlforecast never computes — are rejected by :func:`make_interval_method`.

The upstream ask is small: ``mlforecast.conformal_prediction`` already exposes
``compute_conformity_scores(..., signed=False)``, but ``MLForecast.fit`` never
passes ``signed=True`` on the ordinary path (1.1 reserves signed scores for
transfer conformal, which bypasses this registry). Threading ``signed=`` from
``PredictionIntervals`` would unlock the signed-score methods here.

Calibration-window requirement
------------------------------
Methods validate that **every** series has at least ``ceil(1/alpha)`` effective
calibration windows and raise
:class:`~conformalforecast.base.InsufficientCalibrationSamples` otherwise. This
is stricter than mlforecast's native methods, which interpolate a quantile from
however few windows they were given (``PredictionIntervals(n_windows=2)`` is the
default, and its minimum). A method therefore needs
``PredictionIntervals(n_windows>=ceil(1/alpha))`` — 11 windows for ``level=90``,
21 for ``level=95``. The error names the offending series; it is a genuine
statistical constraint, not a shim limitation.

Also transparent to this shim: with ``PredictionIntervals(scale_estimator=...)``
mlforecast normalizes ``cs_df`` before the hook and rescales the emitted
interval columns afterwards, and ``_cs_df`` may carry extra feature columns for
the DRE path. Only ``cs_df[model]`` is read, so neither affects correctness.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..methods.base import ConformalMethod
from ..nonconformity.absolute import AbsoluteResidual

__all__ = ["make_interval_method", "register_interval_method"]


def _load_ufp() -> Any:
    """Soft-import ``utilsforecast.processing`` (an mlforecast dependency)."""
    try:
        import utilsforecast.processing as ufp
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The mlforecast shim needs utilsforecast (installed with mlforecast). "
            "Install it with: pip install conformalforecast[nixtla]"
        ) from exc
    return ufp


def _scores_cube(
    cs_values: NDArray[np.floating],
    *,
    cs_n_windows: int,
    n_series: int,
    cs_h: int,
    horizon: int,
) -> NDArray[np.floating]:
    """
    Reshape mlforecast's flat conformity scores to the method layout.

    Parameters
    ----------
    cs_values : NDArray, shape (cs_n_windows * n_series * cs_h,)
        Window-major flat scores from ``cs_df[model]``.
    cs_n_windows, n_series, cs_h : int
        The layout mlforecast recorded the scores with.
    horizon : int
        Requested forecast horizon.

    Returns
    -------
    NDArray, shape (n_series, cs_n_windows, horizon)
        Scores for the method. When ``cs_h == 1 < horizon`` the single column is
        tiled across the horizon (constant-width intervals).

    Raises
    ------
    ValueError
        If ``cs_values`` does not hold ``cs_n_windows * n_series * cs_h`` values,
        or ``cs_h`` is neither 1 nor >= ``horizon``.
    """
    expected = cs_n_windows * n_series * cs_h
    if cs_values.size != expected:
        raise ValueError(
            f"cs_df holds {cs_values.size} scores but the declared layout "
            f"({cs_n_windows}, {n_series}, {cs_h}) needs {expected}."
        )
    cube = cs_values.reshape(cs_n_windows, n_series, cs_h).transpose(1, 0, 2)
    if cs_h == 1 and horizon > 1:
        return np.repeat(cube, horizon, axis=2)
    if cs_h < horizon:
        raise ValueError(
            f"cs_h={cs_h} cannot serve horizon={horizon}; refit the intervals with "
            "PredictionIntervals(h=horizon) or h=1."
        )
    return cube[:, :, :horizon]


def make_interval_method(
    method_cls: type[ConformalMethod[Any]],
    **method_kwargs: Any,
) -> Callable[..., Any]:
    """
    Adapt a ``ConformalMethod`` to mlforecast's interval-hook signature.

    Parameters
    ----------
    method_cls : type[ConformalMethod]
        A method class, e.g. :class:`~conformalforecast.methods.split.SplitConformal`.
        ``alpha`` is supplied per requested level by the returned function, so
        do not pass it here.
    **method_kwargs
        Extra keyword arguments forwarded to ``method_cls`` (``gamma``, ``rho``,
        ``seed``, ``score``, ...). A ``score`` must be an
        :class:`AbsoluteResidual`, like the class default it replaces.

    Returns
    -------
    callable
        A function matching mlforecast 1.1's interval hook — ``fcst_df`` and
        ``cs_df`` positional, then keyword ``model_names``, ``level``,
        ``cs_n_windows``, ``cs_h``, ``n_series``, ``horizon``, ``weights``,
        ``is_transfer``, ``target_weights`` — that appends
        ``{model}-lo-{level}`` / ``{model}-hi-{level}`` columns. One method is
        constructed and fitted per ``(model, level)`` pair. Weighted and
        transfer conformal arguments raise ``NotImplementedError``.

    Raises
    ------
    ValueError
        If ``method_cls`` ``CONSUMES_QUANTILES`` (it needs quantile forecasts
        mlforecast does not record), or the score it would fit with is not an
        :class:`AbsoluteResidual` (see the module docstring).

    Examples
    --------
    >>> from conformalforecast.compat.mlforecast_shim import make_interval_method
    >>> from conformalforecast.methods import AdaptiveConformalInference
    >>> hook = make_interval_method(AdaptiveConformalInference, gamma=0.01)   # doctest: +SKIP
    """
    if method_cls.CONSUMES_QUANTILES:
        raise ValueError(
            f"{method_cls.__name__} consumes quantile forecasts, which mlforecast's conformity "
            "scores do not provide. Use a point-forecast method."
        )
    probe = method_cls(alpha=0.1, **method_kwargs)
    # The score the shim will fit with: the configured one, else the default.
    default_score = probe.score if probe.score is not None else probe._default_score()
    if not isinstance(default_score, AbsoluteResidual):
        raise ValueError(
            f"{method_cls.__name__}'s default score is "
            f"{type(default_score).__name__}, but mlforecast stores only absolute "
            "errors, so signed and quantile scores cannot be reconstructed. Pass a "
            "method whose default score is AbsoluteResidual."
        )

    def interval_method(
        fcst_df: Any,
        cs_df: Any,
        model_names: Sequence[str],
        level: Sequence[float],
        cs_n_windows: int,
        cs_h: int,
        n_series: int,
        horizon: int,
        weights: NDArray[np.floating] | None = None,
        is_transfer: bool = False,
        target_weights: NDArray[np.floating] | None = None,
    ) -> Any:
        """Append conformal interval columns to ``fcst_df``. See the factory docstring."""
        if weights is not None:
            raise NotImplementedError(
                "The method shim does not implement weighted conformal prediction: "
                "`weights` (density-ratio reweighting) is unsupported. NexCP's rho "
                "decay is a different mechanism and is not a substitute."
            )
        if target_weights is not None:
            raise NotImplementedError(
                "The method shim does not implement weighted conformal prediction: "
                "`target_weights` is unsupported."
            )
        if is_transfer:
            raise NotImplementedError(
                "The method shim does not implement transfer conformal prediction "
                "(`is_transfer=True` pools calibration across domains)."
            )
        ufp = _load_ufp()
        out = ufp.copy_if_pandas(fcst_df, deep=False)
        levels = sorted(level)
        for model in model_names:
            scores = _scores_cube(
                np.asarray(cs_df[model].to_numpy(), dtype=np.float64),
                cs_n_windows=cs_n_windows,
                n_series=n_series,
                cs_h=cs_h,
                horizon=horizon,
            )
            point = np.asarray(out[model].to_numpy(), dtype=np.float64).reshape(
                n_series, 1, horizon
            )
            # AbsoluteResidual(0, scores) == scores: the only faithful way to hand
            # mlforecast's precomputed |error| to a method that scores internally.
            pseudo_predictions = np.zeros_like(scores)

            los: list[NDArray[np.floating]] = []
            his: list[NDArray[np.floating]] = []
            for lv in levels:
                # (100 - lv) / 100 rather than 1 - lv/100: exact for integer levels.
                method = method_cls(alpha=(100.0 - lv) / 100.0, **method_kwargs)
                fitted = method.fit(pseudo_predictions, scores)
                iv = fitted.interval(point)
                los.append(iv[..., 0].reshape(-1))
                his.append(iv[..., 1].reshape(-1))

            lo_cols = [f"{model}-lo-{lv}" for lv in reversed(levels)]
            hi_cols = [f"{model}-hi-{lv}" for lv in levels]
            values = np.vstack(los[::-1] + his).T
            out = ufp.assign_columns(out, lo_cols + hi_cols, values)
        return out

    return interval_method


def register_interval_method(
    name: str,
    method_cls: type[ConformalMethod[Any]],
    *,
    overwrite: bool = False,
    **method_kwargs: Any,
) -> Callable[..., Any]:
    """
    Register a method-backed interval method in mlforecast's registry.

    Parameters
    ----------
    name : str
        Registry key, e.g. ``"conformal_method_split"``. Note the upstream gap
        described in the module docstring: ``PredictionIntervals`` validates
        ``method`` against a hard-coded list, so a brand-new name is reachable
        via ``get_conformal_method(name)`` but not yet via
        ``PredictionIntervals(method=name)``.
    method_cls : type[ConformalMethod]
        Method class; see :func:`make_interval_method`.
    overwrite : bool, default False
        Allow replacing an existing registry entry (e.g. shadowing
        ``"conformal_error"`` to compare against mlforecast's native method).
    **method_kwargs
        Forwarded to ``method_cls``.

    Returns
    -------
    callable
        The registered hook.

    Raises
    ------
    ImportError
        If mlforecast is not installed.
    ValueError
        If ``name`` is already registered and ``overwrite`` is False, or if
        ``method_cls`` is unsupported (see :func:`make_interval_method`).
    """
    try:
        from mlforecast.conformal_prediction import _INTERVAL_METHODS
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Registering an interval method needs mlforecast >= 1.1. "
            "Install it with: pip install conformalforecast[nixtla]"
        ) from exc
    if name in _INTERVAL_METHODS and not overwrite:
        raise ValueError(
            f"{name!r} is already registered in mlforecast's _INTERVAL_METHODS; "
            "pass overwrite=True to replace it."
        )
    hook = make_interval_method(method_cls, **method_kwargs)
    _INTERVAL_METHODS[name] = hook
    return hook
