"""
Demo: Two-step replacement for forecast(add_history=True).

Step 1 — user calls forecast() as normal.
Step 2 — user calls forecast_history() to get in-sample fitted values.
          Concatenate the two to reproduce add_history=True output.

forecast_history() is designed to become NixtlaClient.forecast_history(self, df, ...)
— replace `client` with `self` to port it.
"""

import numpy as np
import pandas as pd
from nixtla import NixtlaClient


# ──────────────────────────────────────────────────────────────────────────────
# forecast_history — prototype for NixtlaClient.forecast_history()
# ──────────────────────────────────────────────────────────────────────────────
def forecast_history(
    client,
    df,
    freq=None,
    id_col="unique_id",
    time_col="ds",
    target_col="y",
    level=None,
    quantiles=None,
    finetuned_model_id=None,
    clean_ex_first=True,
    date_features=False,
    date_features_to_one_hot=False,
    model="timegpt-1",
    num_partitions=None,
    model_parameters=None,
    multivariate=False,
    categorical_exog_list=None,
    validate_api_key=False,
    batch_window_size=None,
    # Intentionally excluded vs forecast():
    #   h, X_df          — not applicable for in-sample
    #   finetune_steps   — forced to 0 by add_history (_forecast_payload_to_in_sample)
    #   finetune_depth   — passed through by add_history but has no effect when
    #                      finetune_steps=0; safe to omit here
    #   finetune_loss    — same as finetune_depth
    #   hist_exog_list   — add_history marks all exog as future-only internally;
    #                      omitting this replicates that behaviour
    #   feature_contributions — not supported by cross_validation endpoint
    # Note: finetuned_model_id IS passed through by add_history and is exposed
    #   here so the in-sample call uses the same model as the forecast call.
):
    """Return in-sample fitted values using the same logic as forecast(add_history=True).

    Typical usage::

        future = client.forecast(df, X_df=X_df, h=H, freq=FREQ, model=MODEL)
        history = client.forecast_history(df, freq=FREQ, model=MODEL)
        combined = (
            pd.concat([history, future])
            .sort_values(["unique_id", "ds"])
            .reset_index(drop=True)
        )

    Pass the same ``freq``, ``model``, ``level``, and ``clean_ex_first`` that you
    passed to ``forecast()`` so the cross-validation window parameters are computed
    consistently.

    Parameters
    ----------
    batch_window_size : int, optional
        Number of CV windows per cross-validation call.  Use this when a single
        series has enough data to produce many windows (large n_windows) and the
        resulting payload is too large.  Each batch trims trailing rows so the
        API sees only the windows for that batch; results are concatenated by
        timestamp.  Can be combined with ``batch_size`` — each series chunk is
        then also window-batched independently.
    """
    def _window_params(df_chunk):
        model_input_size, model_horizon = client._get_model_params(model, freq)
        min_series_len = (
            df_chunk.groupby(id_col, observed=True)[time_col].count().min()
        )
        insample_h = min(model_horizon, min_series_len - 1)
        if clean_ex_first:
            n_windows = max(
                (min_series_len - model_input_size) // model_horizon, 1
            )
        else:
            n_windows = max(
                (min_series_len - (model_input_size + model_horizon + 2 * insample_h))
                // model_horizon,
                1,
            )
        if level is not None and n_windows > 1:
            n_windows -= 1
        return insample_h, n_windows

    def _call_cv(df_chunk, n_windows_chunk, insample_h):
        cv = client.cross_validation(
            df=df_chunk,
            h=insample_h,
            freq=freq,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            level=level,
            quantiles=quantiles,
            finetune_steps=0,
            n_windows=n_windows_chunk,
            step_size=insample_h,
            clean_ex_first=clean_ex_first,
            finetuned_model_id=finetuned_model_id,
            date_features=date_features,
            date_features_to_one_hot=date_features_to_one_hot,
            model=model,
            num_partitions=num_partitions,
            model_parameters=model_parameters,
            multivariate=multivariate,
            categorical_exog_list=categorical_exog_list,
            validate_api_key=validate_api_key,
        )
        return cv.drop(columns=["cutoff", target_col])

    def _run_cv(df_chunk):
        insample_h, n_windows = _window_params(df_chunk)
        if batch_window_size is None:
            return _call_cv(df_chunk, n_windows, insample_h)
        # Window batching: trim the last (batch_start * insample_h) rows from
        # each series to shift the CV window backward in time, exposing earlier
        # windows without changing h, step_size, or any other parameter.
        parts = []
        for batch_start in range(0, n_windows, batch_window_size):
            n_windows_this = min(batch_window_size, n_windows - batch_start)
            trim_rows = batch_start * insample_h
            if trim_rows > 0:
                mask = (
                    df_chunk
                    .groupby(id_col, observed=True)
                    .cumcount(ascending=False) >= trim_rows
                )
                df_trimmed = df_chunk[mask].reset_index(drop=True)
            else:
                df_trimmed = df_chunk
            parts.append(_call_cv(df_trimmed, n_windows_this, insample_h))
        return (
            pd.concat(parts)
            .sort_values([id_col, time_col])
            .reset_index(drop=True)
        )

    return _run_cv(df)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset — 10 series × 1 200 points
#
# timegpt-1-long-horizon at 15min: input_size=840, horizon=168
#   n_windows = (1200 - 840) // 168 = 2  ← needed for window-batching demo
# ──────────────────────────────────────────────────────────────────────────────
client = NixtlaClient()

np.random.seed(42)
N_SERIES = 10
N_POINTS = 1200
H = 96
FREQ = "15min"
MODEL = "timegpt-1-long-horizon"

base_date = pd.Timestamp("2023-01-01")
dates = pd.date_range(base_date, periods=N_POINTS, freq=FREQ)
future_dates = pd.date_range(dates[-1] + pd.Timedelta(FREQ), periods=H, freq=FREQ)
exog_wave = np.sin(np.arange(N_POINTS + H) * 2 * np.pi / 96)

rows, x_rows = [], []
for i in range(N_SERIES):
    uid = f"series_{i}"
    y = np.random.randn(N_POINTS).cumsum() + 100 + exog_wave[:N_POINTS] * 5
    rows.append(pd.DataFrame({"unique_id": uid, "ds": dates, "y": y, "exog1": exog_wave[:N_POINTS]}))
    x_rows.append(pd.DataFrame({"unique_id": uid, "ds": future_dates, "exog1": exog_wave[N_POINTS:N_POINTS + H]}))

df = pd.concat(rows, ignore_index=True)
X_df = pd.concat(x_rows, ignore_index=True)
print(f"Dataset: {N_SERIES} series × {N_POINTS} points ({len(df):,} rows)\n")


def _sorted(frame):
    return frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Check 1 — forecast_history() matches add_history=True
# Run on 3 series to keep the baseline call fast.
# ──────────────────────────────────────────────────────────────────────────────
N_CHECK = 3
check_ids = df["unique_id"].unique()[:N_CHECK]
df_check = df[df["unique_id"].isin(check_ids)]
X_df_check = X_df[X_df["unique_id"].isin(check_ids)]

ref = client.forecast(df=df_check, X_df=X_df_check, h=H, freq=FREQ, add_history=True, model=MODEL)

future_check = client.forecast(df=df_check, X_df=X_df_check, h=H, freq=FREQ, model=MODEL)
history_check = forecast_history(client, df=df_check, freq=FREQ, model=MODEL)
combined_check = _sorted(pd.concat([history_check, future_check]))

pd.testing.assert_frame_equal(_sorted(ref), combined_check, check_exact=False, rtol=1e-5, atol=1e-5)
print(f"[1] ✓ forecast_history() matches add_history=True  ({len(combined_check)} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# Check 2 — trimming logic (no API call)
# insample_h=168 for timegpt-1-long-horizon; batch_start=1 trims 168 rows.
# Verify the cumcount mask drops exactly the right rows from each series.
# ──────────────────────────────────────────────────────────────────────────────
INSAMPLE_H = 168  # model_horizon for timegpt-1-long-horizon at 15min

trim_rows = INSAMPLE_H
mask = df.groupby("unique_id", observed=True).cumcount(ascending=False) >= trim_rows
df_trimmed = df[mask].reset_index(drop=True)

assert len(df_trimmed) == N_SERIES * (N_POINTS - trim_rows)
assert (df_trimmed.groupby("unique_id", observed=True)["ds"].first() == dates[0]).all()
assert (df_trimmed.groupby("unique_id", observed=True)["ds"].last() == dates[N_POINTS - trim_rows - 1]).all()
print(f"[2] ✓ trim_rows={trim_rows}: each series {N_POINTS}→{N_POINTS - trim_rows} rows, "
      f"correct start/end timestamps")


# ──────────────────────────────────────────────────────────────────────────────
# Check 3 — window batching: batch_window_size=1 gives consistent results
# n_windows=2 for this dataset → batch_window_size=1 produces 2 API calls:
#   batch 0 (trim_rows=0):   full series,   covers the most recent window
#   batch 1 (trim_rows=168): trimmed series, covers the older window
# The most recent window is produced from the same input in both cases, so
# those values must match exactly; the older window may differ slightly.
# ──────────────────────────────────────────────────────────────────────────────
future_all = client.forecast(df=df, X_df=X_df, h=H, freq=FREQ, model=MODEL)

history = forecast_history(client, df=df, freq=FREQ, model=MODEL)
history_win_batched = forecast_history(client, df=df, freq=FREQ, model=MODEL, batch_window_size=1)

assert history.shape == history_win_batched.shape
pd.testing.assert_frame_equal(
    _sorted(history)[["unique_id", "ds"]],
    _sorted(history_win_batched)[["unique_id", "ds"]],
)

# Most recent window: both calls use the full series → values must be identical.
recent_start = dates[N_POINTS - INSAMPLE_H]
h_recent = _sorted(history).query("ds >= @recent_start").reset_index(drop=True)
hb_recent = _sorted(history_win_batched).query("ds >= @recent_start").reset_index(drop=True)
pd.testing.assert_frame_equal(h_recent, hb_recent, check_exact=False, rtol=1e-5, atol=1e-5)

diff_w = (_sorted(history)["TimeGPT"] - _sorted(history_win_batched)["TimeGPT"]).abs().max()
print(f"[3] ✓ batch_window_size=1: timestamps match, recent window values identical; "
      f"max overall diff {diff_w:.4f}")