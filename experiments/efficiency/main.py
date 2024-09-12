import itertools
import logging
import time

import pandas as pd
from nixtla import NixtlaClient as V2Client
from nixtlats import NixtlaClient as V1Client
from utilsforecast.data import generate_series
from utilsforecast.feature_engineering import fourier

logging.getLogger("nixtla").setLevel(logging.ERROR)
logging.getLogger("nixtlats").setLevel(logging.ERROR)


def forecast(client, df, X_df, h, level):
    return client.forecast(df=df, X_df=X_df, h=h, level=level)


def cross_validation(client, df, X_df, h, level):
    return client.cross_validation(df=df, h=h, n_windows=4, level=level)


def anomaly_detection(client, df, X_df, h, level):
    if isinstance(level, list):
        level = level[0]
    return client.detect_anomalies(df=df, level=level)


v1_client = V1Client()
v2_client = V2Client()
n_series = 1_000
freq = "D"
h = 14
series = generate_series(n_series, freq=freq, min_length=200)
train, future = fourier(series, freq=freq, season_length=7, k=4, h=h)
features = ["none", "exog"]
level = [None, [80]]
clients = {"v1": v1_client, "v2": v2_client}
methods = {
    "forecast": forecast,
    "cross_validation": cross_validation,
    "anomaly_detection": anomaly_detection,
}
times = {version: {} for version in ("v1", "v2")}
for feats, lvl in itertools.product(features, level):
    if feats == "none":
        df = series
        X_df = None
    else:
        df = train
        X_df = future
    for name, method in methods.items():
        if name == "anomaly_detection" and lvl is None:
            continue
        for version, client in clients.items():
            start = time.perf_counter()
            combination = f"{version} {name}. Features: {feats}. Level: {lvl}"
            print(f"Running {combination}")
            res = method(client, df=df, X_df=X_df, h=h, level=lvl)
            time_taken = time.perf_counter() - start
            times[version][f"{name}-{feats}-{lvl}"] = time_taken
            print(f"{combination} took {time_taken:.1f} seconds.")

df = pd.DataFrame(times)
df.index = df.index.str.split("-", expand=True)
df.index.names = ["endpoint", "features", "level"]
df = df.sort_index()
df["speedup"] = df["v1"] / df["v2"]
df["speedup"] = df["speedup"].map("{:.0f}x".format)
for col in ("v1", "v2"):
    df[col] = df[col].map("{:.0f}s".format)
with open("endpoint_times.md", "wt") as f:
    f.write(df.reset_index().to_markdown(index=False))
