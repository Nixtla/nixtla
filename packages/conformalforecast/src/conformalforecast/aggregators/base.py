"""Base abstraction for online expert aggregators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict
from pydantic.json_schema import SkipJsonSchema

from .._model import ArrayModel
from ..base import CalibrationError


class OnlineAggregator(ArrayModel, ABC):
    """
    Online expert aggregator for AgACI and similar methods.

    Maintains per-expert cumulative losses and exposes normalized weights.
    Subclasses differ in how they convert losses to weights.

    The panel shape (`n_experts`, `n_series`, `horizon`) is unknown at
    construction time -- it is only known once the caller sees the
    calibration arrays -- so it is bound via :meth:`initialize`, not
    ``__init__``. Call :meth:`initialize` once before :meth:`weights` or
    :meth:`update`.
    """

    # extra="forbid" turns a typo'd or invented keyword into a ValidationError
    # instead of a silently ignored argument, for the same reason
    # ConformalMethod sets it: it matters most for a caller filling in
    # model_json_schema() rather than reading a signature. Set once here so
    # EWA and any third-party OnlineAggregator subclass inherit it rather
    # than redeclaring it.
    model_config = ConfigDict(extra="forbid")

    # Configuration-shaped, but set by initialize() rather than __init__
    # because the panel shape isn't known at construction time. No
    # trailing underscore is kept deliberately (see initialize()) so
    # update()'s shape check can keep reading them unchanged.
    n_experts: SkipJsonSchema[int | None] = None
    n_series: SkipJsonSchema[int | None] = None
    horizon: SkipJsonSchema[int | None] = None
    cumulative_losses_: SkipJsonSchema[NDArray[np.floating] | None] = None

    def initialize(self, n_experts: int, n_series: int, horizon: int) -> None:
        """
        Bind the panel shape and allocate ``cumulative_losses_``.

        Parameters
        ----------
        n_experts : int
            Number of experts being aggregated.
        n_series : int
            Number of series in the panel.
        horizon : int
            Forecast horizon length.

        Raises
        ------
        ValueError
            If any argument is < 1.
        """
        if n_experts < 1:
            raise ValueError(f"n_experts must be >= 1, got {n_experts}")
        if n_series < 1:
            raise ValueError(f"n_series must be >= 1, got {n_series}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.n_experts = n_experts
        self.n_series = n_series
        self.horizon = horizon
        self.cumulative_losses_ = np.zeros((n_experts, n_series, horizon), dtype=np.float64)

    def _require_initialized(self) -> NDArray[np.floating]:
        """Return ``cumulative_losses_``, raising if :meth:`initialize` was never called."""
        if self.cumulative_losses_ is None:
            raise CalibrationError(
                f"{type(self).__name__}.initialize(n_experts, n_series, horizon) "
                "must be called before weights() or update()."
            )
        return self.cumulative_losses_

    @abstractmethod
    def weights(self) -> NDArray[np.floating]:
        """
        Return current normalized weights.

        Returns
        -------
        NDArray, shape (n_experts, n_series, horizon)
            Sum along axis 0 equals 1 for every (series, horizon) cell.

        Raises
        ------
        CalibrationError
            If called before :meth:`initialize`.
        """

    def update(self, losses: NDArray[np.floating]) -> None:
        """
        Add per-expert losses to cumulative losses.

        Parameters
        ----------
        losses : NDArray, shape (n_experts, n_series, horizon)

        Raises
        ------
        CalibrationError
            If called before :meth:`initialize`.
        ValueError
            If ``losses`` does not have the expected shape.
        """
        cumulative = self._require_initialized()
        expected = (self.n_experts, self.n_series, self.horizon)
        if losses.shape != expected:
            raise ValueError(f"losses must have shape {expected}, got {losses.shape}")
        self.cumulative_losses_ = cumulative + losses
