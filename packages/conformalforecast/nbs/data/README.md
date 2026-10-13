# Documentation data

Small datasets committed so the documentation notebooks execute without network
access. CI runs every notebook on every pull request; a runtime download would
make the docs build fail on an unrelated outage.

## `air_passengers.csv`

Monthly totals of international airline passengers, January 1949 – December 1960
(144 observations, one series). The classic Box & Jenkins series, in Nixtla long
format (`unique_id`, `ds`, `y`).

- **Source:** `https://datasets-nixtla.s3.amazonaws.com/air-passengers.csv`
- **Origin:** Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976),
  *Time Series Analysis, Forecasting and Control*, Series G.
- **License:** public domain; widely redistributed as a reference dataset,
  including in R's `datasets` package.

The default dataset for the site. Strong trend and strong yearly seasonality in
144 points makes it fast to model and easy to read on a plot, so every
single-series page except SPCI's (which streams the M4 series `H1` below) uses
it. Pages that need exogenous features build them from
its own calendar (a linear trend and Fourier terms) rather than joining an
outside source.

## `m4_hourly_panel.csv`

Eight hourly series from the M4 competition (`H1`, `H10`, `H50`, `H100`, `H150`,
`H200`, `H300`, `H400`), 4 872 observations, in Nixtla long format.

- **Source:** `https://datasets-nixtla.s3.amazonaws.com/m4-hourly.csv`
- **Origin:** Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020),
  *The M4 Competition: 100,000 time series and 61 forecasting methods*,
  International Journal of Forecasting 36(1).
- **License:** CC BY 4.0. The same file backs the public statsforecast,
  mlforecast and neuralforecast documentation.

Derived from the source file by:

1. keeping the eight series above, chosen so their levels span three orders of
   magnitude (mean 19.5 for `H300` up to 33 466 for `H50`) — per-series
   calibration then visibly differs rather than looking like one shared constant;
2. keeping the **last** 504–672 observations of each, a different count per
   series, all multiples of the 24-hour season;
3. re-basing the integer `ds` index onto hourly timestamps ending
   `2015-01-28 23:00` **for every series**.

Step 3 is the point: because the series share an end date but not a length, the
panel is ragged, and `panel.long_to_panel` gives it a real left-NaN prefix. The
ragged-panels and transfer pages, the NexCP and SPCI method pages, and the
statsforecast, mlforecast and neuralforecast adapter pages all use it.
