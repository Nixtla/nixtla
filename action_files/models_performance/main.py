import logging
import os
from time import time
from typing import List, Optional, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mape, mse

from nixtla import NixtlaClient


logger = logging.getLogger(__name__)
load_dotenv()


class Experiment:
    """
    This class represents an experiment for evaluating the performance of different models.
    The main method, evaluate_performance, is intended to be called for different models.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        experiment_name: str,
        id_col: str,
        time_col: str,
        target_col: str,
        h: int,
        season_length: int,
        # Freq cannot be infered
        # because of StatsForecast
        freq: str,
        level: Optional[List[int]] = None,
        n_windows: int = 1,  # @A: this should be replaced with cross validation
    ):
        self.df = df
        self.experiment_name = experiment_name
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.h = h
        self.season_length = season_length
        self.freq = freq
        self.level = level
        self.n_windows = n_windows
        self.eval_index = [
            "experiment_name",
            "h",
            "season_length",
            "freq",
            "level",
            "n_windows",
            "metric",
        ]
        (
            self.df_train,
            self.df_test,
            self.df_cutoffs,
            self.has_id_col,
            self.comb_cv,
        ) = self._split_df(df)
        self.benchmark_models = ["SeasonalNaive", "Naive"]

    def _split_df(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, List]:
        has_id_col = self.id_col in df
        if has_id_col:
            df_test = df.groupby(self.id_col).tail(self.h)
            comb_cv = [self.id_col, self.time_col]
        else:
            df_test = df.tail(self.h)
            comb_cv = [self.time_col]
        df_train = df.drop(df_test.index)
        if has_id_col:
            df_cutoffs = (
                df_train.groupby(self.id_col)[[self.time_col]].max().reset_index()
            )
        else:
            df_cutoffs = df_train[[self.time_col]].max().to_frame().T
        df_cutoffs = df_cutoffs.rename(
            columns={
                self.time_col: "cutoff",
            }
        )
        return df_train, df_test, df_cutoffs, has_id_col, comb_cv

    def _evaluate_cv(
        self, cv_df: pd.DataFrame, total_time: float, model: str
    ) -> pd.DataFrame:
        metrics = [mae, mse, mape]
        if not self.has_id_col:
            cv_df = cv_df.assign(unique_id="ts_0")
        eval_df = cv_df.groupby("cutoff").apply(
            lambda df_cutoff: evaluate(
                df_cutoff,
                metrics=metrics,
                models=[model],
                id_col=self.id_col,
                time_col=self.time_col,
                target_col=self.target_col,
            )
        )
        eval_df = eval_df.reset_index().drop(columns="level_1")
        eval_df = eval_df.groupby(["metric"]).mean(numeric_only=True)
        eval_df = eval_df.reset_index()
        if len(eval_df) != len(metrics):
            raise ValueError(f"Expected only {len(metrics)} metrics")
        eval_df = pd.concat(
            [eval_df, pd.DataFrame({"metric": ["total_time"], model: [total_time]})]
        )
        for attr in reversed(self.eval_index):
            if attr not in eval_df.columns:
                eval_df.insert(0, attr, getattr(self, attr))
        return eval_df

    def _convert_fcst_df_to_cv_df(self, fcst_df: pd.DataFrame) -> pd.DataFrame:
        if self.has_id_col:
            # add cutoff column
            cv_df = fcst_df.merge(self.df_cutoffs, on=[self.id_col])
            # add y column
            merge_cols = [self.id_col, self.time_col]
        else:
            # add cutoff column
            cv_df = fcst_df.assign(cutoff=self.df_cutoffs["cutoff"].iloc[0])
            # add y column
            merge_cols = [self.time_col]
        cv_df = cv_df.merge(
            self.df_test[merge_cols + [self.target_col]],
            on=merge_cols,
        )
        return cv_df

    def evaluate_timegpt(self, model: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        init_time = time()
        # A: this sould be replaced with
        # cross validation
        timegpt = NixtlaClient()
        fcst_df = timegpt.forecast(
            df=self.df_train,
            X_df=(
                self.df_test.drop(columns=self.target_col)
                if self.df.shape[1] > 3
                else None
            ),
            h=self.h,
            freq=self.freq,
            level=self.level,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
            model=model,
        )
        cv_df = self._convert_fcst_df_to_cv_df(fcst_df)
        total_time = time() - init_time
        cv_df = cv_df.rename({"TimeGPT": model}, axis=1)
        eval_df = self._evaluate_cv(cv_df, total_time, model)
        return eval_df, cv_df.drop(columns=[self.target_col, "cutoff"])

    def evaluate_benchmark_performace(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        eval_df = []
        cv_df = []
        # wee need to rename columns if needed
        renamer = {
            self.id_col: "unique_id",
            self.time_col: "ds",
            self.target_col: "y",
        }
        df = self.df.copy()
        if not self.has_id_col:
            df[self.id_col] = "ts_0"
        df = df.rename(columns=renamer)
        for model in [SeasonalNaive(season_length=self.season_length), Naive()]:
            sf = StatsForecast(freq=self.freq, models=[model])
            init_time = time()
            cv_model_df = sf.cross_validation(
                df=df,
                h=self.h,
                n_windows=self.n_windows,
                step_size=self.h,
            )
            total_time = time() - init_time
            cv_model_df = cv_model_df.rename(
                columns={value: key for key, value in renamer.items()}
            )
            eval_model_df = self._evaluate_cv(cv_model_df, total_time, repr(model))
            eval_model_df = eval_model_df.set_index(self.eval_index)
            eval_df.append(eval_model_df)
            cv_df.append(cv_model_df.set_index([self.id_col, self.time_col, "cutoff"]))
        eval_df = pd.concat(eval_df, axis=1).reset_index()
        cv_df = pd.concat(cv_df, axis=1).reset_index()
        if not self.has_id_col:
            cv_df = cv_df.drop(columns=[self.id_col])
        return eval_df, cv_df.drop(columns=[self.target_col, "cutoff"])

    def plot_and_save_forecasts(self, cv_df: pd.DataFrame, plot_dir: str) -> str:
        """Plot ans saves forecasts, returns the path of the plot"""
        timegpt = NixtlaClient()
        df = self.df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        if not self.has_id_col:
            df[self.id_col] = "ts_0"
        cv_df[self.time_col] = pd.to_datetime(cv_df[self.time_col])
        fig = timegpt.plot(
            df[[self.id_col, self.time_col, self.target_col]],
            cv_df,
            max_insample_length=self.h * (self.n_windows + 4),
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
        )
        path = "plot"
        for attr in self.eval_index:
            if hasattr(self, attr):
                path += f"_{getattr(self, attr)}"
        plot_path = f"{plot_dir}/{path}.png"
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(plot_path, bbox_inches="tight")
        return plot_path


class ExperimentConfig:
    def __init__(
        self,
        config_path: str,
        plot_dir: str,
    ):
        self.config_path = config_path
        self.plot_dir = plot_dir
        self.default_models = ["timegpt-1", "timegpt-1-long-horizon"]

    def _parse_yaml(self):
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def run_experiments(self):
        config = self._parse_yaml()
        eval_df = []
        for experiment_dict in config["experiments"]:
            experiment_name = list(experiment_dict.keys())[0]
            experiment = {}
            for d in experiment_dict[experiment_name]:
                experiment.update(d)
            df_url = experiment["dataset_url"]
            df = pd.read_csv(df_url)
            id_col = experiment.get("id_col", "unique_id")
            time_col = experiment.get("time_col", "ds")
            target_col = experiment.get("target_col", "y")
            season_length = experiment["season_length"]
            df[time_col] = pd.to_datetime(df[time_col])
            # list parameters
            # we will iterate over this parameters
            horizons = experiment["h"]
            levels = experiment.get("level", [None])
            frequencies = experiment.get("freq", [None])
            for h in horizons:
                for level in levels:
                    for freq in frequencies:
                        logger.info(
                            f"Running experiment {experiment_name} with h={h}, level={level}, freq={freq}"
                        )
                        exp = Experiment(
                            df=df,
                            experiment_name=experiment_name,
                            id_col=id_col,
                            time_col=time_col,
                            target_col=target_col,
                            h=h,
                            freq=freq,
                            level=level,
                            season_length=season_length,
                        )
                        # Benchmark evaluation
                        logger.info("Running benchmark evaluation")
                        (
                            eval_bench_df,
                            cv_bench_df,
                        ) = exp.evaluate_benchmark_performace()
                        eval_bench_df = eval_bench_df.set_index(exp.eval_index)
                        cv_bench_df = cv_bench_df.set_index(exp.comb_cv)
                        eval_models_df = [eval_bench_df]
                        cv_models_df = [cv_bench_df]
                        # models evaluation
                        logger.info("Running TimeGPT evaluation")
                        for model in self.default_models:
                            (
                                eval_model_df,
                                cv_model_df,
                            ) = exp.evaluate_timegpt(model=model)
                            eval_model_df = eval_model_df.set_index(exp.eval_index)
                            eval_models_df.append(eval_model_df)
                            cv_model_df = cv_model_df.set_index(exp.comb_cv)
                            cv_models_df.append(cv_model_df)
                        cv_models_df = pd.concat(cv_models_df, axis=1).reset_index()
                        plot_path = exp.plot_and_save_forecasts(
                            cv_models_df, self.plot_dir
                        )
                        eval_models_df = pd.concat(eval_models_df, axis=1)
                        eval_models_df["plot_path"] = plot_path
                        eval_df.append(eval_models_df.reset_index())
        eval_df = pd.concat(eval_df)
        return eval_df, exp.benchmark_models

    def summary_performance(
        self, eval_df: pd.DataFrame, summary_path: str, benchmark_models: List[str]
    ):
        logger.info("Summarizing performance")
        models = self.default_models + benchmark_models
        with open(summary_path, "w") as f:
            results_comb = ["metric"] + models
            exp_config = [col for col in eval_df.columns if col not in results_comb]
            eval_df = eval_df.fillna("None")
            f.write("<details><summary>Experiment Results</summary>\n\n")
            for exp_number, (exp_desc, eval_exp_df) in enumerate(
                eval_df.groupby(exp_config), start=1
            ):
                exp_metadata = pd.DataFrame.from_dict(
                    {
                        "variable": exp_config,
                        "experiment": exp_desc,
                    }
                )
                experiment_name = exp_metadata.query("variable == 'experiment_name'")[
                    "experiment"
                ].iloc[0]
                exp_metadata.query(
                    "variable not in ['plot_path', 'experiment_name']", inplace=True
                )
                f.write(f"## Experiment {exp_number}: {experiment_name}\n\n")
                f.write("### Description:\n")
                f.write(f"{exp_metadata.to_markdown(index=False)}\n\n")
                f.write("### Results:\n")
                f.write(
                    f"{eval_exp_df[results_comb].round(4).to_markdown(index=False)}\n\n"
                )
                f.write("### Plot:\n")
                plot_path = eval_exp_df["plot_path"].iloc[0]
                if plot_path.startswith("."):
                    plot_path = plot_path[1:]
                if os.getenv("GITHUB_ACTIONS"):
                    plot_path = f"{os.getenv('PLOTS_REPO_URL')}/{plot_path}?raw=true"
                f.write(f"![]({plot_path})\n\n")
            f.write("</details>\n")


if __name__ == "__main__":
    exp_config = ExperimentConfig(
        config_path="./action_files/models_performance/experiments.yaml",
        plot_dir="./action_files/models_performance/plots",
    )
    eval_df, benchmark_models = exp_config.run_experiments()
    exp_config.summary_performance(
        eval_df, "./action_files/models_performance/summary.md", benchmark_models
    )
