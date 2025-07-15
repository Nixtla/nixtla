### Generate synthetic data
from functools import partial
from pathlib import Path

import numpy as np

from utilsforecast.data import generate_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mape, mase
from utilsforecast.preprocessing import fill_gaps

ROOT_DIR = Path(__file__).resolve().parent.parent
IMG_PATH = ROOT_DIR / "nbs" / "imgs"


series = generate_series(3, with_trend=True, static_as_categorical=False)
series
### Plotting
from utilsforecast.plotting import plot_series

fig = plot_series(
    series, plot_random=False, max_insample_length=50, engine="matplotlib"
)
fig.savefig(IMG_PATH / "index.png", bbox_inches="tight")

### Preprocessing
serie = series[series["unique_id"].eq(0)].tail(10)
# drop some points
with_gaps = serie.sample(frac=0.5, random_state=0).sort_values("ds")

fill_gaps(with_gaps, freq="D")

### Evaluating
valid = series.groupby("unique_id").tail(7).copy()
train = series.drop(valid.index)
rng = np.random.RandomState(0)
valid["seas_naive"] = train.groupby("unique_id")["y"].tail(7).values
valid["rand_model"] = valid["y"] * rng.rand(valid["y"].shape[0])
daily_mase = partial(mase, seasonality=7)
evaluate(valid, metrics=[mape, daily_mase], train_df=train)
