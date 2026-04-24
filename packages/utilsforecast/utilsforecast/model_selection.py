"""Multi-objective model selection utilities."""

__all__ = ["ParetoFrontier"]

from typing import List, Optional

import numpy as np

from .compat import AnyDFType


class ParetoFrontier:
    """Utilities for Pareto frontier analysis."""

    _NON_MODEL_COLS = frozenset({"unique_id", "metric", "cutoff", "ds", "y"})
    _NON_METRIC_COLS = frozenset({"unique_id", "model", "cutoff", "ds", "y"})

    @staticmethod
    def is_dominated(data: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """Returns a boolean mask of which models are dominated.

        Args:
            data (np.ndarray): Shape (n, m) — metric values for n models, m metrics.
            directions (np.ndarray): Shape (m,) — 1 for minimization, -1 for maximization.

        Returns:
            np.ndarray: Shape (n,) boolean array — True where a model is dominated.
        """
        d = data * directions                                        # (n, m)
        better_or_equal = d[None, :, :] <= d[:, None, :]            # (n, n, m)
        strictly_better = d[None, :, :] <  d[:, None, :]            # (n, n, m)
        dominates = better_or_equal.all(axis=2) & strictly_better.any(axis=2)  # (n, n)
        np.fill_diagonal(dominates, False)
        return dominates.any(axis=1)

    @classmethod
    def find_non_dominated(
        cls,
        performance_df: AnyDFType,
        metrics: Optional[List[str]] = None,
        model_subset: Optional[List[str]] = None,
        maximization: Optional[List[str]] = None,
        id_col: str = "unique_id",
        cutoff_col: str = "cutoff",
    ) -> AnyDFType:
        """Returns the non-dominated models (Pareto frontier).

        Args:
            performance_df (AnyDFType): Output from evaluate, aggregated to one
                row per metric (e.g. via evaluate(..., agg_fn='mean')).
            metrics (List[str], optional): Metric names to consider for dominance
                comparison. Only used in the evaluate() output format (where a
                "metric" column exists). Defaults to all metrics.
            model_subset (List[str], optional): Subset of model columns to
                consider. Only used in the evaluate() output format.
                Defaults to all model columns.
            maximization (List[str], optional): Metrics where 'more is better'.
            id_col (str, optional): Column that identifies each series.
                Must match the id_col used in evaluate(). Defaults to 'unique_id'.
            cutoff_col (str, optional): Column that identifies the cutoff point.
                Must match the cutoff_col used in evaluate(). Defaults to 'cutoff'.
        """
        import narwhals.stable.v2 as nw
        df = nw.from_native(performance_df)
        columns = df.columns

        non_model_cols = cls._NON_MODEL_COLS | {id_col, cutoff_col}
        non_metric_cols = cls._NON_METRIC_COLS | {id_col, cutoff_col}

        if "metric" in columns:
            # evaluate() output format: models are columns, metrics are rows
            if model_subset is None:
                models = [c for c in columns if c not in non_model_cols]
            else:
                models = model_subset

            if not models:
                return performance_df

            actual_metrics = df["metric"].to_numpy()
            if metrics is not None:
                mask = [m in metrics for m in actual_metrics]
                df = df.filter(mask)
                actual_metrics = df["metric"].to_numpy()

            data = []
            for m in models:
                data.append(df[m].to_numpy())
            data = np.array(data)

            directions = np.ones(len(actual_metrics))
            if maximization:
                for i, m in enumerate(actual_metrics):
                    if m in maximization:
                        directions[i] = -1

            is_pareto = ~cls.is_dominated(data, directions)
            pareto_models = [m for m, p in zip(models, is_pareto) if p]
            keep_cols = [c for c in columns if c not in models or c in pareto_models]
            return df.select(*keep_cols).to_native()
        else:
            # Fallback: rows are candidates, columns are metrics
            if metrics is None:
                metrics = [c for c in columns if c not in non_metric_cols]

            if not metrics:
                return performance_df

            data = []
            for m in metrics:
                data.append(df[m].to_numpy())
            data = np.column_stack(data)

            directions = np.ones(len(metrics))
            if maximization:
                for i, m in enumerate(metrics):
                    if m in maximization:
                        directions[i] = -1

            is_pareto = (~cls.is_dominated(data, directions)).tolist()
            return df.filter(is_pareto).to_native()

    @staticmethod
    def plot_pareto_2d(
        performance_df: AnyDFType,
        metric_x: str,
        metric_y: str,
        maximize_x: bool = False,
        maximize_y: bool = False,
        show_dominated: bool = True,
        title: str = "Pareto Frontier",
    ):
        """Plots the 2D Pareto frontier."""
        import warnings
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib is required for plotting.")
            return None

        import narwhals.stable.v2 as nw

        nw_df = nw.from_native(performance_df)

        # Build a standard (rows=models, cols=metrics) narwhals frame for plotting
        if "metric" in nw_df.columns:
            row_x = nw_df.filter(nw.col("metric") == metric_x)
            row_y = nw_df.filter(nw.col("metric") == metric_y)

            if len(row_x) == 0 or len(row_y) == 0:
                raise ValueError(f"Metrics {metric_x} or {metric_y} not found in the 'metric' column.")

            models = [c for c in nw_df.columns if c not in ParetoFrontier._NON_MODEL_COLS]

            x_vals = [row_x[m].to_numpy()[0] for m in models]
            y_vals = [row_y[m].to_numpy()[0] for m in models]

            plot_df = nw.from_dict(
                {
                    "model": models,
                    metric_x: x_vals,
                    metric_y: y_vals,
                },
                backend=nw.get_native_namespace(nw_df.to_native()),
            )
        else:
            if "model" in nw_df.columns:
                plot_df = nw_df.select("model", metric_x, metric_y)
            else:
                plot_df = nw.from_dict(
                    {
                        "model": [str(i) for i in range(len(nw_df))],
                        metric_x: nw_df[metric_x].to_list(),
                        metric_y: nw_df[metric_y].to_list(),
                    },
                    backend=nw.get_native_namespace(nw_df.to_native()),
                )

        maximization = ([metric_x] if maximize_x else []) + ([metric_y] if maximize_y else [])
        pareto_df = nw.from_native(
            ParetoFrontier.find_non_dominated(
                plot_df.to_native(),
                metrics=[metric_x, metric_y],
                maximization=maximization,
            )
        )

        all_x = plot_df[metric_x].to_numpy()
        all_y = plot_df[metric_y].to_numpy()
        all_models = plot_df["model"].to_numpy()

        pareto_x = pareto_df[metric_x].to_numpy()
        pareto_y = pareto_df[metric_y].to_numpy()

        fig, ax = plt.subplots(figsize=(10, 6))

        if show_dominated:
            ax.scatter(all_x, all_y, color="grey", alpha=0.5, label="Dominated")
            for m, vx, vy in zip(all_models, all_x, all_y):
                ax.annotate(str(m), (vx, vy), alpha=0.7)

        ax.scatter(pareto_x, pareto_y, color="red", s=100, label="Pareto Optimal")

        sort_idx = np.argsort(pareto_x)
        ax.plot(pareto_x[sort_idx], pareto_y[sort_idx], "r--", alpha=0.5)

        ax.set_xlabel(metric_x)
        ax.set_ylabel(metric_y)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax
