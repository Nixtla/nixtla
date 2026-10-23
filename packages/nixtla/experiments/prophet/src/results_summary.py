from pathlib import Path

import fire
from numpy import column_stack
import pandas as pd


def read_kind_results(kind: str, dir: str):
    files = list(Path(dir).rglob(f"*{kind}.parquet"))
    df = pd.concat(
        [pd.read_parquet(file).assign(file=str(file).split("/")[-2]) for file in files],
        ignore_index=True,
    )
    return df


def summarize_results_per_file(metrics_df: pd.DataFrame):
    metrics_df_per_freq = metrics_df.groupby(["file", "metric", "model"]).mean(
        numeric_only=True
    )
    metrics_df_per_freq = metrics_df_per_freq.reset_index()
    metrics_df_per_freq = metrics_df_per_freq.query(
        "model in ['Prophet', 'SeasonalNaive', 'TimeGPT']"
    )
    models = metrics_df_per_freq["model"].unique()
    metrics_df_per_freq = pd.pivot(
        metrics_df_per_freq,
        index=["file", "metric"],
        columns="model",
        values="value",
    ).reset_index()
    for model in models:
        if model == "SeasonalNaive":
            continue
        metrics_df_per_freq[model] /= metrics_df_per_freq["SeasonalNaive"]
    metrics_df_per_freq["SeasonalNaive"] /= metrics_df_per_freq["SeasonalNaive"]
    return metrics_df_per_freq


def prepare_results(df: pd.DataFrame):
    def bold_best(row):
        row = row.round(3)
        models = row.drop(columns=["file", "metric"]).columns
        best_model = row[models].idxmin(axis=1).item()
        row[best_model] = "**" + str(row[best_model].item()) + "**"
        return row

    df_bolded = df.groupby(["file", "metric"]).apply(bold_best)
    df_bolded = df_bolded.reset_index(drop=True)
    return df_bolded


def write_to_readme(content: str):
    with open("README.md", "r") as file:
        readme_content = file.readlines()
    start_index = -1
    end_index = -1
    for i, line in enumerate(readme_content):
        if line.strip().lower() == "## results":
            start_index = i + 1
        if start_index != -1 and line.strip() == "<end>":
            end_index = i
            break

    if start_index != -1 and end_index != -1:
        readme_content = (
            readme_content[: start_index + 1]
            + [content + "\n"]
            + readme_content[end_index:]
        )
    else:
        print("Results section not found or improperly formatted")

    # Write the changes back to the README
    with open("README.md", "w") as file:
        file.writelines(readme_content)


def summarize_results(dir: str):
    metrics_df = read_kind_results("metrics", dir)
    summary_df = read_kind_results("summary", dir)
    summary_df = (
        summary_df.set_index(["file", "frequency"])
        .reset_index()
        .round(3)
        .sort_values("frequency")
    )
    no_int_cols = ["file", "frequency", "mean", "std"]
    for col in summary_df.columns:
        if col not in no_int_cols:
            summary_df[col] = summary_df[col].astype(int)
    summary_df = summary_df.to_markdown(index=False, intfmt=",", floatfmt=",.3f")
    time_df = read_kind_results("time", dir)
    time_df = time_df.assign(metric="time").rename(columns={"time": "value"})
    metrics_df_per_file = summarize_results_per_file(metrics_df)
    time_df = summarize_results_per_file(time_df)
    eval_df = pd.concat([metrics_df_per_file, time_df], ignore_index=True)
    eval_df = prepare_results(eval_df)[
        ["file", "metric", "TimeGPT", "Prophet", "SeasonalNaive"]
    ]
    n_files = eval_df["file"].nunique()
    eval_df = eval_df.to_markdown(
        index=False,
        colalign=2 * ["left"] + (eval_df.shape[1] - 2) * ["right"],
    )
    markdown_lines = eval_df.split("\n")
    custom_separator = markdown_lines[1].replace(":", "-")
    for i in range(4, len(markdown_lines) + n_files - 1, 4):
        markdown_lines.insert(i + 1, custom_separator)
    markdown_lines.insert(
        0,
        ("\n### Data Description\n\n" f"{summary_df}\n\n" "### Performance\n\n"),
    )
    eval_df = "\n".join(markdown_lines)
    write_to_readme(eval_df)


if __name__ == "__main__":
    fire.Fire(summarize_results)
