"""Method-specific introspection: the internal state of a fitted method.

Every function here takes the fitted object a method's ``fit`` /
``calibrate`` / ``calibrate_cv`` returns, not the method itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..methods.aci import ACIForecaster
from ..methods.agaci import AgACIForecaster
from ..methods.base import ConformalForecaster
from ..methods.nexcp import NexCPForecaster
from ..methods.spci import SPCIForecaster


def _require(fitted: object, cls: type[ConformalForecaster], fn: str) -> None:
    if not isinstance(fitted, cls):
        raise TypeError(
            f"{fn} expects an {cls.__name__} -- the object the method's calibrate() "
            f"returns -- got {type(fitted).__name__}."
        )


def aci_state(fitted: ACIForecaster) -> dict[str, Any]:
    """
    Snapshot of ACI's adaptive state.

    Parameters
    ----------
    fitted : ACIForecaster

    Returns
    -------
    dict with keys:
        ``"alpha_t"`` : NDArray, shape (n_series, horizon)
            Current adaptive miscoverage rate per cell.
        ``"alpha_t_minus_alpha"`` : NDArray, shape (n_series, horizon)
            How far the adaptive rate has drifted from the target.
            Magnitude indicates how much adaptation occurred.
        ``"n_observations"`` : int
            Total observations (calibration + updates).
        ``"gamma"`` : float
            Learning rate.

    Raises
    ------
    TypeError
        If ``fitted`` is not an :class:`ACIForecaster`.
    """
    _require(fitted, ACIForecaster, "aci_state")
    alpha_t = fitted.alpha_t_.copy()
    return {
        "alpha_t": alpha_t,
        "alpha_t_minus_alpha": alpha_t - fitted.method_.alpha,
        "n_observations": int(fitted.n_observations_),
        "gamma": float(fitted.method_.gamma),
    }


def agaci_state(fitted: AgACIForecaster) -> dict[str, Any]:
    """
    Snapshot of AgACI's per-expert state and aggregator weights.

    Parameters
    ----------
    fitted : AgACIForecaster

    Returns
    -------
    dict with keys:
        ``"alpha_t_per_expert"`` : NDArray, shape (n_experts, n_series, horizon)
        ``"gammas"`` : tuple of float
        ``"weights_lower"`` : NDArray, shape (n_experts, n_series, horizon)
            Current normalized weights of the lower-bound aggregator.
        ``"weights_upper"`` : NDArray, shape (n_experts, n_series, horizon)
        ``"best_expert_per_cell_lower"`` : NDArray of int, shape (n_series, horizon)
            Index of the expert with highest weight per cell, for the
            lower-bound aggregator.
        ``"best_expert_per_cell_upper"`` : NDArray of int, shape (n_series, horizon)
        ``"n_observations"`` : int

    Raises
    ------
    TypeError
        If ``fitted`` is not an :class:`AgACIForecaster`.
    """
    _require(fitted, AgACIForecaster, "agaci_state")
    weights_lower = fitted.aggregator_lower_.weights()
    weights_upper = fitted.aggregator_upper_.weights()
    best_lower = np.argmax(weights_lower, axis=0).astype(np.int64)
    best_upper = np.argmax(weights_upper, axis=0).astype(np.int64)

    return {
        "alpha_t_per_expert": fitted.alpha_t_per_expert_.copy(),
        "gammas": fitted.method_.gammas,
        "weights_lower": weights_lower,
        "weights_upper": weights_upper,
        "best_expert_per_cell_lower": best_lower,
        "best_expert_per_cell_upper": best_upper,
        "n_observations": int(fitted.n_observations_),
    }


def nexcp_state(fitted: NexCPForecaster) -> dict[str, Any]:
    """
    Snapshot of NexCP's weighting state.

    Parameters
    ----------
    fitted : NexCPForecaster

    Returns
    -------
    dict with keys:
        ``"rho"`` : float
        ``"n_observations"`` : int
        ``"effective_sample_size"`` : float
            ``ESS = (sum w)^2 / sum(w^2)``. With ``rho = 1`` this equals
            ``n_observations``; with smaller ``rho`` it's less.
        ``"weights_minimum"`` : float
            Smallest weight in the buffer (the oldest sample's weight).
        ``"weights_maximum"`` : float
            Largest weight (the newest sample's weight = 1).
        ``"score_quantile"`` : NDArray, shape (n_series, horizon)
            Current weighted quantile threshold.

    Raises
    ------
    TypeError
        If ``fitted`` is not a :class:`NexCPForecaster`.
    """
    _require(fitted, NexCPForecaster, "nexcp_state")
    n_obs = int(fitted.n_observations_)
    rho = float(fitted.method_.rho)
    # Weights for the current buffer: position i (0-indexed, oldest first)
    # gets rho^(n_obs - 1 - i). Newest sample is 1, oldest is rho^(n_obs - 1).
    i = np.arange(n_obs)
    weights = rho ** (n_obs - 1 - i)
    sum_w = float(weights.sum())
    sum_w_sq = float((weights * weights).sum())
    ess = (sum_w * sum_w / sum_w_sq) if sum_w_sq > 0 else 0.0

    return {
        "rho": rho,
        "n_observations": n_obs,
        "effective_sample_size": ess,
        "weights_minimum": float(weights.min()),
        "weights_maximum": float(weights.max()),
        "score_quantile": fitted.score_quantile_.copy(),
    }


def spci_state(fitted: SPCIForecaster) -> dict[str, Any]:
    """
    Snapshot of SPCI's regressor inventory.

    Parameters
    ----------
    fitted : SPCIForecaster

    Returns
    -------
    dict with keys:
        ``"window_size"`` : int
        ``"n_observations"`` : int
        ``"n_regressors"`` : int
            Should equal ``n_series * horizon``.
        ``"regressor_class"`` : str
        ``"refit_every"`` : int
        ``"steps_since_refit"`` : int
            Where in the refit cycle we are.
        ``"residuals_shape"`` : tuple
            Shape of the residual buffer
            ``(n_series, n_observations, horizon)``.

    Raises
    ------
    TypeError
        If ``fitted`` is not an :class:`SPCIForecaster`.
    """
    _require(fitted, SPCIForecaster, "spci_state")
    regressors = fitted.quantile_regressors_
    # All regressor instances are clones of the same prototype; sample one
    # for its class name.
    sample = next(iter(regressors.values()))
    return {
        "window_size": int(fitted.method_.window_size),
        "n_observations": int(fitted.n_observations_),
        "n_regressors": len(regressors),
        "regressor_class": type(sample).__name__,
        "refit_every": int(fitted.method_.refit_every),
        "steps_since_refit": int(fitted.steps_since_refit_),
        "residuals_shape": tuple(fitted.residuals_.shape),
    }


def method_state(fitted: ConformalForecaster) -> dict[str, Any]:
    """
    Dispatch to the appropriate method-specific state function.

    Parameters
    ----------
    fitted : ConformalForecaster

    Returns
    -------
    dict
        Result of :func:`aci_state` / :func:`agaci_state` / :func:`nexcp_state`
        / :func:`spci_state`, or an empty dict for methods without
        method-specific diagnostics (e.g. the
        :class:`~conformalforecast.methods.fixed_offset.FixedOffsetForecaster`
        split conformal and CQR return).
    """
    if isinstance(fitted, ACIForecaster):
        return aci_state(fitted)
    if isinstance(fitted, AgACIForecaster):
        return agaci_state(fitted)
    if isinstance(fitted, NexCPForecaster):
        return nexcp_state(fitted)
    if isinstance(fitted, SPCIForecaster):
        return spci_state(fitted)
    return {}
