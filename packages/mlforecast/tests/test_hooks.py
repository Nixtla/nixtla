import copy
import os
import pickle
import sys

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from utilsforecast.losses import smape

import mlforecast
from mlforecast import MLForecast
from mlforecast.hooks import Hook, Profiler
from mlforecast.lag_transforms import RollingMean
from mlforecast.utils import PredictionIntervals, generate_daily_series


class Recorder(Hook):
    def __init__(self):
        self.events = []

    def on_start(self, stage, fcst, **kwargs):  # noqa: ARG002
        self.events.append(("start", stage, kwargs))

    def on_end(self, stage, fcst, result, **kwargs):  # noqa: ARG002
        self.events.append(("end", stage, kwargs))

    def on_error(self, stage, fcst, error, **kwargs):  # noqa: ARG002
        self.events.append(("error", stage, kwargs))

    def stages(self, hook):
        return [stage for h, stage, _ in self.events if h == hook]


class Raiser(Hook):
    """Raise once in ``on_end`` of ``stage``."""

    def __init__(self, stage):
        self.stage = stage
        self.raised = False

    def on_end(self, stage, fcst, result, **kwargs):  # noqa: ARG002
        if stage == self.stage and not self.raised:
            self.raised = True
            raise RuntimeError("end failed")


class Boom(BaseEstimator):
    def fit(self, X, y):  # noqa: ARG002
        raise RuntimeError("boom")


@pytest.fixture
def series():
    return generate_daily_series(4, min_length=20, max_length=30, equal_ends=True)


def _stages(span):
    return [child.stage for child in span.children]


def _count(span, stage):
    return (span.stage == stage) + sum(_count(child, stage) for child in span.children)


def test_hooks_are_off_by_default_and_methods_keep_their_identity():
    fcst = MLForecast(models=LinearRegression(), freq="D", lags=[1])
    assert fcst.hooks == []
    assert MLForecast.fit.__name__ == "fit"
    assert MLForecast.predict.__doc__.startswith("Compute the predictions")


@pytest.mark.parametrize("hooks", [[object()], Profiler()])
def test_hooks_validate_type(hooks):
    with pytest.raises(TypeError, match="sequence of Hook"):
        MLForecast(models=LinearRegression(), freq="D", hooks=hooks)


def test_hooks_accept_any_sequence():
    profiler = Profiler()
    fcst = MLForecast(models=LinearRegression(), freq="D", hooks=(profiler,))
    assert fcst.hooks == [profiler]


def test_empty_profiler_has_queryable_columns():
    assert Profiler().to_df().query("stage == 'fit_model'").empty


def test_direct_conformal_warning_points_to_caller(series):
    from mlforecast.conformal_prediction import (
        _floor_series_scales,
        _raw_series_scales,
    )

    frame = series.copy()
    first_id = frame["unique_id"].iloc[0]
    frame.loc[frame["unique_id"] == first_id, "y"] = 5.0
    raw = _raw_series_scales(frame, "unique_id", "ds", "y", ("mad",))
    with pytest.warns(UserWarning, match="near-zero scale") as caught:
        _floor_series_scales(raw["mad"], floor_factor=1e-3)
    assert caught[0].filename == __file__


def test_warning_stacklevel_stops_at_dirs_sharing_the_package_prefix():
    package_dir = os.path.dirname(mlforecast.__file__)
    caller = os.path.join(package_dir + "_ext", "run.py")
    code = "from mlforecast._warnings import _user_warning_stacklevel\nlevel = _user_warning_stacklevel()"
    namespace = {}
    exec(compile(code, caller, "exec"), namespace)
    assert namespace["level"] == 1


@pytest.mark.parametrize("with_hooks", [False, True])
@pytest.mark.parametrize("stage", ["preprocess", "fit", "cross_validation"])
def test_warnings_raised_inside_stages_point_to_the_caller(series, stage, with_hooks):
    """The stage wrapper adds a frame on every path, with or without hooks."""
    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1],
        hooks=[Profiler()] if with_hooks else None,
    )
    kwargs = {"max_horizon": 2, "horizon_features": {1: []}}
    calls = {
        "preprocess": lambda: fcst.preprocess(series, **kwargs),
        "fit": lambda: fcst.fit(series, **kwargs),
        "cross_validation": lambda: fcst.cross_validation(
            series, n_windows=1, h=2, **kwargs
        ),
    }
    with pytest.warns(UserWarning, match="has no effect") as caught:
        calls[stage]()
    (warning,) = [w for w in caught if "has no effect" in str(w.message)]
    assert warning.filename == __file__


@pytest.mark.parametrize("with_hooks", [False, True])
def test_pooled_validation_warning_points_to_the_caller(series, with_hooks):
    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1],
        lag_transforms={1: [RollingMean(3, global_=True)]},
        hooks=[Profiler()] if with_hooks else None,
    )
    with pytest.warns(UserWarning, match="gap-free") as caught:
        fcst.fit(series, validate_data=False)
    (warning,) = [w for w in caught if "gap-free" in str(w.message)]
    assert warning.filename == __file__


def test_stages_receive_bound_arguments(series):
    recorder = Recorder()
    fcst = MLForecast(models=LinearRegression(), freq="D", lags=[1], hooks=[recorder])
    fcst.fit(series).predict(3)

    assert recorder.stages("start") == [
        "fit",
        "preprocess",
        "fit_models",
        "fit_model",
        "predict",
        "predict_model",
    ]
    assert recorder.stages("end") == [
        "preprocess",
        "fit_model",
        "fit_models",
        "fit",
        "predict_model",
        "predict",
    ]
    (fit_model,) = [kw for h, s, kw in recorder.events if s == "fit_model"][:1]
    assert fit_model["name"] == "LinearRegression"
    assert fit_model["h"] is None
    assert fit_model["X"].shape[0] == len(fit_model["y"])
    (predict_model,) = [kw for h, s, kw in recorder.events if s == "predict_model"][:1]
    assert predict_model["name"] == "LinearRegression"
    assert predict_model["model"] is fcst.models_["LinearRegression"]
    (predict,) = [kw for h, s, kw in recorder.events if s == "predict"][:1]
    assert predict["h"] == 3
    # defaults are bound too
    assert predict["level"] is None
    (fit,) = [kw for h, s, kw in recorder.events if s == "fit"][:1]
    assert fit["df"] is series
    assert fit["id_col"] == "unique_id"
    (preprocess,) = [
        kw for h, s, kw in recorder.events if h == "start" and s == "preprocess"
    ]
    assert preprocess["df"] is series


def test_direct_models_emit_one_fit_model_per_horizon(series):
    recorder = Recorder()
    fcst = MLForecast(
        models=[LinearRegression(), LinearRegression()],
        freq="D",
        lags=[1],
        hooks=[recorder],
    )
    fcst.fit(series, max_horizon=2)

    fit_models = [kw for h, s, kw in recorder.events if h == "end" and s == "fit_model"]
    assert [(kw["name"], kw["h"]) for kw in fit_models] == [
        ("LinearRegression", 0),
        ("LinearRegression", 1),
        ("LinearRegression2", 0),
        ("LinearRegression2", 1),
    ]


@pytest.mark.parametrize("max_horizon", [None, 2])
def test_profiler_times_each_model_in_predict(series, max_horizon):
    profiler = Profiler()
    fcst = MLForecast(
        models=[LinearRegression(), LinearRegression()],
        freq="D",
        lags=[1],
        hooks=[profiler],
    )
    fcst.fit(series, max_horizon=max_horizon)
    profiler.reset()
    fcst.predict(2)

    (predict,) = profiler.spans
    assert _stages(predict) == ["predict_model", "predict_model"]
    assert [child.details for child in predict.children] == [
        {"model": "LinearRegression"},
        {"model": "LinearRegression2"},
    ]
    assert predict.seconds >= sum(child.seconds for child in predict.children)


@pytest.mark.parametrize("max_horizon", [None, 2])
def test_predict_model_result_is_the_models_predictions(series, max_horizon):
    class ResultRecorder(Hook):
        def __init__(self):
            self.results = {}

        def on_end(self, stage, fcst, result, **kwargs):  # noqa: ARG002
            if stage == "predict_model":
                self.results[kwargs["name"]] = result

    recorder = ResultRecorder()
    fcst = MLForecast(
        models=[LinearRegression(), LinearRegression(fit_intercept=False)],
        freq="D",
        lags=[1],
        hooks=[recorder],
    )
    fcst.fit(series, max_horizon=max_horizon)
    preds = fcst.predict(2)

    assert list(recorder.results) == ["LinearRegression", "LinearRegression2"]
    for name, result in recorder.results.items():
        np.testing.assert_allclose(result, preds[name].to_numpy())


def test_each_horizon_fit_model_gets_its_own_weights(series):
    series = series.assign(w=1.0)
    models_fit_kwargs = {"LinearRegression": {"sample_weight": None}}
    recorder = Recorder()
    fcst = MLForecast(models=LinearRegression(), freq="D", lags=[1], hooks=[recorder])
    fcst.fit(series, max_horizon=3, weight_col="w", models_fit_kwargs=models_fit_kwargs)

    n_rows = [
        (len(kwargs["X"]), len(kwargs["fit_kwargs"]["sample_weight"]))
        for hook, stage, kwargs in recorder.events
        if hook == "start" and stage == "fit_model"
    ]
    assert len(n_rows) == 3
    assert all(n_x == n_w for n_x, n_w in n_rows)
    assert models_fit_kwargs == {"LinearRegression": {"sample_weight": None}}


def test_profiler_nests_calibration_inside_fit(series):
    profiler = Profiler()
    fcst = MLForecast(models=LinearRegression(), freq="D", lags=[1], hooks=[profiler])
    fcst.fit(series, prediction_intervals=PredictionIntervals(n_windows=2, h=1))

    assert [span.stage for span in profiler.spans] == ["fit"]
    fit = profiler.spans[0]
    assert _stages(fit) == ["cross_validation", "preprocess", "fit_models"]
    cv = fit.children[0]
    assert cv.details == {"n_windows": 2, "h": 1}
    assert _stages(cv) == ["cv_window", "cv_window"]
    # refit=False during calibration: only the first window fits
    assert _stages(cv.children[0]) == ["fit", "predict"]
    assert _stages(cv.children[1]) == ["predict"]
    assert _count(fit, "fit_model") == 2
    assert fit.seconds >= sum(child.seconds for child in fit.children)


def test_profiler_keeps_every_cross_validation_window(series):
    profiler = Profiler()
    fcst = MLForecast(
        models=[LinearRegression(), LinearRegression()],
        freq="D",
        lags=[1],
        hooks=[profiler],
    )
    fcst.cross_validation(series, n_windows=3, h=2, refit=True)
    df = profiler.to_df()

    assert df["stage"].tolist()[:2] == ["cross_validation", "cv_window"]
    assert (df["stage"] == "cv_window").sum() == 3
    assert (df["stage"] == "fit").sum() == 3
    assert df.loc[df["stage"] == "cv_window", "i_window"].tolist() == [0, 1, 2]
    assert df.loc[df["stage"] == "cross_validation", "depth"].tolist() == [0]
    assert df.loc[df["stage"] == "fit_model", "depth"].tolist() == [4] * 6
    per_model = df.query("stage == 'fit_model'").groupby("model")["seconds"].sum()
    assert per_model.index.tolist() == ["LinearRegression", "LinearRegression2"]
    assert (per_model > 0).all()


def test_cv_window_receives_bound_arguments(series):
    recorder = Recorder()
    fcst = MLForecast(models=LinearRegression(), freq="D", lags=[1], hooks=[recorder])
    fcst.cross_validation(series, n_windows=2, h=2)
    (cv_start,) = [
        kw
        for hook, stage, kw in recorder.events
        if hook == "start" and stage == "cross_validation"
    ]
    assert cv_start["n_windows"] == 2
    assert cv_start["h"] == 2
    windows = [
        kw
        for hook, stage, kw in recorder.events
        if hook == "start" and stage == "cv_window"
    ]
    assert [window["i_window"] for window in windows] == [0, 1]
    assert all("cutoffs" in window for window in windows)


def test_memory_profiling_without_psutil_names_the_argument(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)  # makes `import psutil` fail
    with pytest.raises(ImportError, match=r"Profiler\(memory=True\) requires psutil"):
        Profiler(memory=True)


def test_profiler_can_record_memory(series):
    pytest.importorskip("psutil")
    profiler = Profiler(memory=True)
    fcst = MLForecast(models=LinearRegression(), freq="D", lags=[1], hooks=[profiler])
    fcst.fit(series)

    assert isinstance(profiler.spans[0].rss_delta_bytes, int)
    assert Profiler().on_start("fit", fcst) is None


def test_hooks_are_dropped_on_serialization(series):
    profiler = Profiler()
    fcst = MLForecast(models=LinearRegression(), freq="D", lags=[1], hooks=[profiler])
    fcst.fit(series)
    for restored in (copy.deepcopy(fcst), pickle.loads(pickle.dumps(fcst))):
        assert restored.hooks == []
        restored.predict(2)
    assert fcst.hooks == [profiler]
    assert [span.stage for span in profiler.spans] == ["fit"]


def test_hooks_are_kept_on_shallow_copy(series):
    profiler = Profiler()
    fcst = MLForecast(models=LinearRegression(), freq="D", lags=[1], hooks=[profiler])
    copy.copy(fcst).fit(series)
    assert fcst.hooks == [profiler]
    assert [span.stage for span in profiler.spans] == ["fit"]


def test_errors_are_reported_and_profiler_recovers(series):
    recorder = Recorder()
    profiler = Profiler()
    fcst = MLForecast(models=Boom(), freq="D", lags=[1], hooks=[recorder, profiler])
    with pytest.raises(RuntimeError, match="boom"):
        fcst.cross_validation(series, n_windows=2, h=1)

    assert recorder.stages("error") == [
        "fit_model",
        "fit_models",
        "fit",
        "cv_window",
        "cross_validation",
    ]
    assert profiler._stack == []
    assert profiler.spans[0].error.startswith("RuntimeError: boom")
    assert profiler.spans[0].children[0].error.startswith("RuntimeError: boom")

    fcst.models = {"LinearRegression": LinearRegression()}
    fcst.fit(series)
    assert [span.stage for span in profiler.spans] == ["cross_validation", "fit"]


def test_start_hook_failure_unwinds_profiler(series):
    class FailOnFit(Hook):
        def on_start(self, stage, fcst, **kwargs):  # noqa: ARG002
            if stage == "fit":
                raise RuntimeError("start failed")

    profiler = Profiler()
    fcst = MLForecast(
        models=LinearRegression(), freq="D", lags=[1], hooks=[profiler, FailOnFit()]
    )
    with pytest.raises(RuntimeError, match="start failed"):
        fcst.fit(series)
    assert profiler._stack == []
    assert profiler.spans[0].error == "RuntimeError: start failed"


def test_cv_window_start_failure_unwinds_profiler(series):
    class FailWindow(Hook):
        def on_start(self, stage, fcst, **kwargs):  # noqa: ARG002
            if stage == "cv_window":
                raise RuntimeError("window start failed")

    profiler = Profiler()
    fcst = MLForecast(
        models=LinearRegression(), freq="D", lags=[1], hooks=[profiler, FailWindow()]
    )
    with pytest.raises(RuntimeError, match="window start failed"):
        fcst.cross_validation(series, n_windows=2, h=1)
    assert profiler._stack == []
    assert profiler.spans[0].children[0].error == "RuntimeError: window start failed"


@pytest.mark.parametrize("model", [LinearRegression(), Boom()])
def test_hooks_finish_in_reverse_order(series, model):
    events = []

    class Logger(Hook):
        def __init__(self, label):
            self.label = label

        def on_start(self, stage, fcst, **kwargs):  # noqa: ARG002
            if stage == "fit_model":
                events.append(("start", self.label))

        def on_end(self, stage, fcst, result, **kwargs):  # noqa: ARG002
            if stage == "fit_model":
                events.append(("end", self.label))

        def on_error(self, stage, fcst, error, **kwargs):  # noqa: ARG002
            if stage == "fit_model":
                events.append(("error", self.label))

    fcst = MLForecast(
        models=model, freq="D", lags=[1], hooks=[Logger("a"), Logger("b")]
    )
    try:
        fcst.fit(series)
    except RuntimeError:
        pass
    finish = "end" if isinstance(model, LinearRegression) else "error"
    assert events == [("start", "a"), ("start", "b"), (finish, "b"), (finish, "a")]


@pytest.mark.parametrize("raiser_first", [True, False])
def test_raising_on_end_does_not_break_later_hooks(series, raiser_first):
    raiser = Raiser(stage="fit_model")
    profiler = Profiler()
    hooks = [raiser, profiler] if raiser_first else [profiler, raiser]
    fcst = MLForecast(models=LinearRegression(), freq="D", lags=[1], hooks=hooks)
    with pytest.raises(RuntimeError, match="end failed"):
        fcst.fit(series)
    fcst.fit(series)
    assert profiler._stack == []
    assert [span.stage for span in profiler.spans] == ["fit", "fit"]
    assert profiler.spans[0].error == "RuntimeError: end failed"
    assert profiler.spans[1].error is None


def test_cv_window_on_end_error_not_reported_as_error(series):
    recorder = Recorder()
    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1],
        hooks=[recorder, Raiser(stage="cv_window")],
    )
    with pytest.raises(RuntimeError, match="end failed"):
        fcst.cross_validation(series, n_windows=2, h=1)
    windows = [h for h, stage, _ in recorder.events if stage == "cv_window"]
    assert windows == ["start", "end"]
    assert recorder.stages("error") == ["cross_validation"]


def test_custom_callback_scores_every_window(series):
    class WindowScores(Hook):
        def __init__(self):
            self.scores = []

        def on_end(self, stage, fcst, result, **kwargs):
            if stage == "cv_window":
                score = smape(result, models=list(fcst.models))
                self.scores.append((kwargs["i_window"], score))

    scores = WindowScores()
    fcst = MLForecast(models=LinearRegression(), freq="D", lags=[1], hooks=[scores])
    fcst.cross_validation(series, n_windows=3, h=2)

    assert [i for i, _ in scores.scores] == [0, 1, 2]
    assert all(score.shape[0] == 4 for _, score in scores.scores)
