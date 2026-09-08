__all__ = ["RayLGBMForecast"]


from typing import Any, Dict

import lightgbm as lgb

from ._base import (
    _RAY_PARAMS,
    KeepLastMetrics,
    RayForecastBase,
    report_fitted_model,
    worker_n_jobs,
)


def _lgb_train_loop(config: Dict[str, Any]) -> None:
    import ray.train
    from ray.train.lightgbm import (
        RayTrainReportCallback,
        get_network_params,
        normalize_pandas_for_lightgbm,
    )

    class _ReportCallback(KeepLastMetrics, RayTrainReportCallback):
        pass

    shard = ray.train.get_dataset_shard("train")
    # since ray 2.56 to_pandas yields pd.ArrowDtype columns, which lightgbm's
    # input validation rejects, so they're mapped back to numpy dtypes here.
    df = normalize_pandas_for_lightgbm(shard.materialize().to_pandas())
    label = df.pop(config["target_col"])
    params = {
        **config["params"],
        "n_jobs": worker_n_jobs(config["params"].get("n_jobs")),
    }
    callback = _ReportCallback(checkpoint_at_end=False)
    # each worker only sees its own shard. ray's LightGBMConfig stashes the
    # network params in a per worker global rather than injecting them, so
    # without these every worker trains an independent model on 1/N of the data
    # and rank 0's is the one that gets checkpointed. Plain kwargs, as in
    # lightgbm.dask's _train_part.
    model = lgb.LGBMRegressor(
        **params, tree_learner="data_parallel", **get_network_params()
    )
    model.fit(
        df, label, eval_set=[(df, label)], eval_names=["train"], callbacks=[callback]
    )
    # the clamp is for this worker's thread pool; model_ is shipped to the
    # forecasting workers and returned by to_local, so it keeps what was asked for
    model.set_params(n_jobs=config["params"].get("n_jobs"))
    report_fitted_model(
        model, model.booster_, _ReportCallback.CHECKPOINT_NAME, callback.last_metrics
    )


class RayLGBMForecast(RayForecastBase, lgb.LGBMRegressor):
    """LightGBM forecaster trained with `ray.train.lightgbm.LightGBMTrainer`.

    The booster's parameters are taken as ``**kwargs`` and handled by
    ``LGBMRegressor`` itself; the ray arguments are keyword only
    so that they can't collide with them.

    ``num_workers`` sets the number of ray train workers. The previous
    ``lightgbm_ray`` based implementation derived that from ``n_jobs``
    (``RayParams(num_actors=n_jobs)``); ``n_jobs`` is now the per worker thread
    count, as it is for the local estimator.

    ``resources_per_worker`` is the CPU knob: it decides how many CPUs each
    worker is given and therefore how many threads the booster can use. It
    defaults to the cluster's CPUs split evenly across the workers and bounded by
    the smallest node, as ``xgboost_ray._autodetect_resources`` did, and ``n_jobs``
    can only lower it below that share. ``model_`` keeps the requested ``n_jobs``
    rather than the clamp.

    ``storage_path`` is where ray train writes the run. It defaults to a
    temporary directory that is discarded once the fitted model has been read
    back, so that ``~/ray_results`` doesn't grow by a run per model per fit; as
    with ray's own default, a local path only works on a single node, so point
    it at shared storage for a multi node cluster.

    ``fit`` takes a ray ``Dataset`` and a target column rather than the sklearn
    ``(X, y)`` pair, as the previous implementation did, so the inherited
    ``predict``/``score`` can't be used before fitting. The fitted estimator is
    exposed as ``model_``, a local ``lightgbm.LGBMRegressor`` that is sent to the
    workers in the forecasting step.
    """

    @classmethod
    def _get_param_names(cls):
        # sklearn reads the subclass' signature, which doesn't name the booster params
        return sorted([*lgb.LGBMRegressor._get_param_names(), *_RAY_PARAMS])

    def fit(self, dataset: Any, target_col: str) -> "RayLGBMForecast":  # type: ignore[override]
        from ray.train.lightgbm import LightGBMTrainer

        return self._train(LightGBMTrainer, _lgb_train_loop, dataset, target_col)  # type: ignore[return-value]
