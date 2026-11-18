__all__ = ["Hook", "Profiler"]


import functools
import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
)

import pandas as pd

if TYPE_CHECKING:
    from mlforecast.forecast import MLForecast

P = ParamSpec("P")
R = TypeVar("R")


class Hook:
    """Observe the stages of an :class:`mlforecast.MLForecast` object.

    Subclass this, override the hooks you need and pass instances through
    ``MLForecast(hooks=[...])``. Stages nest: a ``fit`` with prediction
    intervals runs a ``cross_validation``, which runs a ``fit`` and a ``predict``
    per window. The hooks are called for every level, so a hook can build a
    tree of what happened, react only to top-level calls or filter by stage.
    Hook arguments are live objects; mutating them is unsupported. Internal
    calibration uses a scratch forecaster, distinguishable by identity from
    the original ``MLForecast`` instance.

    Stages and the keyword arguments they carry:

    - ``preprocess``, ``fit``, ``fit_models``, ``predict``, ``cross_validation``: the bound arguments of the method call.
    - ``fit_model``: ``name``, ``model``, ``X``, ``y``, ``fit_kwargs`` and ``h`` (0-indexed horizon for direct models, None otherwise). Emitted once per estimator ``fit`` call.
    - ``predict_model``: ``name`` and ``model`` (a dict keyed by horizon index for direct models). Emitted once per model and covers its feature updates too. Its result is the model's predictions as a flat array ordered by series and then step, before target transforms are inverted.
    - ``cv_window``: ``i_window`` and ``cutoffs``.

    Hooks that started a stage get its ``on_end`` or ``on_error``, in reverse order, even if another one raises, and the first
    hook error then propagates in place of the stage's own: a stage whose ``on_end`` raises still ended, and the
    error reaches the enclosing stages as their ``on_error``.
    """

    def on_start(self, stage: str, fcst: "MLForecast", **kwargs: Any) -> None:
        """Called right before ``stage`` runs."""

    def on_end(
        self, stage: str, fcst: "MLForecast", result: Any, **kwargs: Any
    ) -> None:
        """Called right after ``stage`` returns ``result``."""

    def on_error(
        self, stage: str, fcst: "MLForecast", error: BaseException, **kwargs: Any
    ) -> None:
        """Called when ``stage`` raises ``error``, right before it propagates."""


def _notify(
    hooks: Iterable[Hook], hook: str, stage: str, obj: Any, **kwargs: Any
) -> None:
    first: Optional[BaseException] = None
    for hook_obj in hooks:
        try:
            getattr(hook_obj, hook)(stage, obj, **kwargs)
        except BaseException as error:
            if first is None:
                first = error
    if first is not None:
        raise first


class _SpanResult:
    __slots__ = ("result",)

    def __init__(self) -> None:
        self.result: Any = None


@contextmanager
def _span(obj: Any, stage: str, **kwargs: Any) -> Iterator[_SpanResult]:
    span = _SpanResult()
    hooks = obj.hooks
    if not hooks:
        yield span
        return
    started = []
    try:
        for hook_obj in hooks:
            hook_obj.on_start(stage, obj, **kwargs)
            started.append(hook_obj)
    except BaseException as error:
        _notify(reversed(started), "on_error", stage, obj, error=error, **kwargs)
        raise
    try:
        yield span
    except BaseException as error:
        _notify(reversed(started), "on_error", stage, obj, error=error, **kwargs)
        raise
    _notify(reversed(started), "on_end", stage, obj, result=span.result, **kwargs)


def _stage(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Notify ``self.hooks`` when the decorated method starts, ends or fails."""

    def decorator(method: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(method)

        @functools.wraps(method)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            self: Any = args[0]
            if not self.hooks:
                return method(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            info = dict(bound.arguments)
            del info["self"]
            with _span(self, name, **info) as span:
                span.result = method(*args, **kwargs)
            return span.result

        return wrapper

    return decorator


@dataclass
class Span:
    """Cost of one stage. ``children`` holds the stages that ran inside it."""

    stage: str
    seconds: float
    details: Dict[str, Any]
    rss_delta_bytes: Optional[int] = None
    children: List["Span"] = field(default_factory=list)
    error: Optional[str] = None


def _span_details(stage: str, kwargs: Dict[str, Any], result: Any) -> Dict[str, Any]:
    if stage == "preprocess":
        details = {"input_rows": kwargs["df"].shape[0]}
        if result is not None:
            features = result[0] if isinstance(result, tuple) else result
            details["output_rows"] = features.shape[0]
        return details
    if stage == "fit":
        return {"input_rows": kwargs["df"].shape[0]}
    if stage == "fit_model":
        X = kwargs["X"]
        return {
            "model": kwargs["name"],
            "model_horizon_index": kwargs["h"],
            "n_rows": X.shape[0],
            "n_features": X.shape[1],
        }
    if stage == "predict":
        return {"h": kwargs["h"]}
    if stage == "predict_model":
        return {"model": kwargs["name"]}
    if stage == "cross_validation":
        return {"n_windows": kwargs["n_windows"], "h": kwargs["h"]}
    if stage == "cv_window":
        return {"i_window": kwargs["i_window"]}
    return {}


class Profiler(Hook):
    """Record how long each stage takes and, optionally, its memory delta.

    Args:
        memory (bool): Record the change in the process' resident memory across each stage. Requires psutil. Defaults to False.

    Top-level calls end up in ``spans`` with everything that ran inside them as
    ``children``; ``to_df`` flattens them. A parent's seconds include its
    children, so summing all stages double counts. For example, the time spent fitting
    each model across a whole cross validation is
    ``profiler.to_df().query("stage == 'fit_model'").groupby("model")["seconds"].sum()``.
    Call ``reset()`` only between runs; it clears active spans too.
    """

    def __init__(self, memory: bool = False):
        self.memory = memory
        if memory:
            try:
                import psutil  # noqa: F401
            except ImportError as error:
                raise ImportError("Profiler(memory=True) requires psutil") from error
        self.spans: List[Span] = []
        self._stack: List[Tuple[str, float, Optional[int], List[Span]]] = []

    def _rss(self) -> Optional[int]:
        if not self.memory:
            return None
        import psutil

        return psutil.Process().memory_info().rss

    def on_start(self, stage: str, fcst: "MLForecast", **kwargs: Any) -> None:  # noqa: ARG002
        self._stack.append((stage, perf_counter(), self._rss(), []))

    def _close(
        self,
        stage: str,
        kwargs: Dict[str, Any],
        result: Any,
        error: Optional[BaseException],
    ) -> None:
        _, start, rss_start, children = self._stack.pop()
        rss_delta = None if rss_start is None else self._rss() - rss_start  # type: ignore[operator]
        span = Span(
            stage=stage,
            seconds=perf_counter() - start,
            details=_span_details(stage, kwargs, result),
            rss_delta_bytes=rss_delta,
            children=children,
            error=None if error is None else f"{type(error).__name__}: {error}",
        )
        parent = self._stack[-1][3] if self._stack else self.spans
        parent.append(span)

    def on_end(
        self,
        stage: str,
        fcst: "MLForecast",  # noqa: ARG002
        result: Any,
        **kwargs: Any,
    ) -> None:
        if not self._stack or self._stack[-1][0] != stage:
            raise RuntimeError(f"Profiler stage mismatch at {stage!r}")
        self._close(stage, kwargs, result, None)

    def on_error(
        self,
        stage: str,
        fcst: "MLForecast",  # noqa: ARG002
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        if not self._stack or self._stack[-1][0] != stage:
            return  # don't replace the stage's error with a mismatch
        self._close(stage, kwargs, None, error)

    def reset(self) -> None:
        self.spans = []
        self._stack = []

    def to_df(self) -> pd.DataFrame:
        """Flatten the recorded spans into one row per stage, in execution order."""
        rows: List[Dict[str, Any]] = []

        def visit(span: Span, depth: int) -> None:
            rows.append(
                {
                    "stage": span.stage,
                    "depth": depth,
                    "seconds": span.seconds,
                    "rss_delta_bytes": span.rss_delta_bytes,
                    "error": span.error,
                    **span.details,
                }
            )
            for child in span.children:
                visit(child, depth + 1)

        for span in self.spans:
            visit(span, 0)
        if not rows:
            return pd.DataFrame(
                columns=["stage", "depth", "seconds", "rss_delta_bytes", "error"]
            )
        return pd.DataFrame(rows)
