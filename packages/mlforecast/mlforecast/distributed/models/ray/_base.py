__all__ = ["RayForecastBase"]


import contextlib
import pickle
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_RAY_PARAMS = ("num_workers", "resources_per_worker", "storage_path")
_MODEL_FILE = "model.pkl"


def worker_n_jobs(requested: Any) -> int:
    """Threads for the booster, from the CPUs this train worker was actually given.

    Both ``lightgbm_ray`` and ``xgboost_ray`` sized the thread pool from the
    actor's CPU share; without it N workers landing on one node each spawn
    threads for every core on the box. An explicit, smaller value is honoured.
    """
    import ray

    assigned = ray.get_runtime_context().get_assigned_resources().get("CPU", 1)
    assigned = max(1, int(assigned))
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        return assigned
    return assigned if requested <= 0 else min(requested, assigned)


class KeepLastMetrics:
    """Remember the last iteration's metrics so the final report can carry them."""

    last_metrics: Dict[str, Any] = {}

    def _report_metrics(self, report_dict: Dict[str, Any]) -> None:
        self.last_metrics = report_dict
        super()._report_metrics(report_dict)  # type: ignore[misc]


def report_fitted_model(
    model: Any, booster: Any, booster_file: str, metrics: Dict[str, Any]
) -> None:
    """Report ray's standard booster artifact along with the fitted estimator.

    The estimator is what becomes ``model_``, which is why it's checkpointed;
    the booster is kept next to it so that ``RayTrainReportCallback.get_model``
    still works on the result.

    ``ray.train.report`` is collective, so every worker has to call it;
    reporting from rank 0 only deadlocks.
    """
    import ray.train
    from ray.train import Checkpoint

    if ray.train.get_context().get_world_rank() != 0:
        ray.train.report(metrics)
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        booster.save_model(Path(tmp_dir, booster_file).as_posix())
        with open(Path(tmp_dir, _MODEL_FILE), "wb") as f:
            pickle.dump(model, f)
        ray.train.report(metrics, checkpoint=Checkpoint.from_directory(tmp_dir))


class RayForecastBase:
    """Mixin holding the ray.train plumbing; subclasses are real sklearn estimators.

    The training loop builds and fits the library's own estimator in the worker,
    as ``lightgbm.dask._train_part`` does, and checkpoints it. That keeps all of
    the parameter handling in the library instead of here.

    The ray specific arguments are keyword only so that they can't collide with
    the booster's parameters, which are taken as ``**kwargs``.
    """

    num_workers: int
    resources_per_worker: Optional[Dict[str, float]]
    storage_path: Optional[str]

    def __init__(
        self,
        *,
        num_workers: int = 1,
        resources_per_worker: Optional[Dict[str, float]] = None,
        storage_path: Optional[str] = None,
        **kwargs: Any,
    ):
        # cooperative: goes on to LGBMRegressor / XGBRegressor
        super().__init__(**kwargs)
        self.num_workers = num_workers
        self.resources_per_worker = resources_per_worker
        self.storage_path = storage_path

    def _resources_per_worker(self) -> Dict[str, float]:
        """The CPUs each worker gets, and therefore the booster's thread count.

        ``ScalingConfig`` assigns a single CPU per worker when this isn't set,
        which would make a default fit single threaded. ``xgboost_ray`` split the
        cluster's CPUs across its actors instead (``_autodetect_resources``);
        that's kept here so that the default isn't a slowdown, bound by the
        smallest node included.

        The share also becomes ray data's ``exclude_resources``
        (``DataParallelTrainer`` hands it ``scaling_config.total_resources``), so
        a worker that takes every CPU on its node leaves data none to execute
        with. ``_train`` materializes before building the trainer so that there
        is nothing left for data to do by then.
        """
        import ray

        if self.resources_per_worker is not None:
            return self.resources_per_worker
        cpus = int(ray.cluster_resources().get("CPU", 1))
        # a placement group bundle has to fit on a single node, so the cluster
        # wide share is bounded by the smallest one as well
        min_node_cpus = min(
            (
                node.get("Resources", {}).get("CPU", 0.0)
                for node in ray.nodes()
                if node.get("Alive", False)
            ),
            default=0.0,
        )
        share = min(int(min_node_cpus or 1), cpus // self.num_workers)
        return {"CPU": max(1, share)}

    def _train(
        self,
        trainer_cls: Any,
        train_loop: Callable[[Dict[str, Any]], None],
        dataset: Any,
        target_col: str,
    ) -> "RayForecastBase":
        from ray.train import RunConfig, ScalingConfig

        params = self.get_params()  # type: ignore[attr-defined]
        for name in _RAY_PARAMS:
            params.pop(name, None)
        # execute the dataset before the trainer exists. ray train reserves the
        # workers' CPUs away from ray data (`ScalingConfig.total_resources`
        # becomes data's `exclude_resources`), so a dataset with work still
        # pending once the placement group holds them has nothing left to run
        # with and blocks forever. Nothing is given up by doing it here: the
        # train loops build a `Dataset`/`DMatrix` from the whole shard anyway.
        dataset = dataset.materialize()
        with contextlib.ExitStack() as stack:
            storage_path = self.storage_path
            if storage_path is None:
                # the default (~/ray_results) would grow by one run per model per
                # fit, which neither of the previous wrappers did.
                storage_path = stack.enter_context(tempfile.TemporaryDirectory())
            trainer = trainer_cls(
                train_loop,
                train_loop_config={"params": params, "target_col": target_col},
                scaling_config=ScalingConfig(
                    num_workers=self.num_workers,
                    resources_per_worker=self._resources_per_worker(),
                ),
                run_config=RunConfig(storage_path=storage_path),
                datasets={"train": dataset},
            )
            with trainer.fit().checkpoint.as_directory() as ckpt_dir:
                with open(Path(ckpt_dir, _MODEL_FILE), "rb") as f:
                    self.model_ = pickle.load(f)
        return self
