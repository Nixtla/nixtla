# Auto Time Series
Reproduce M5 performance faster.

## Create environment

- `pip install virtualenv`
- `virtualenv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`

## Create datasets for training

- `source venv/bin/activate`
- `python -m src.download_data --directory [YOUR_DIRECTORY] --output-format [YOUR_FORMAT, csv or parquet]`

## Create s3 bucket

- Create s3 bucket: `aws s3api create-bucket --bucket [YOUR_BUCKET] --profile [YOUR_PROFILE]`

## Nixtla

- Nixtla can receive parquet files.
- So use `python -m src.download_data --directory [YOUR_DIRECTORY] --format parquet`

### Reproduce results using `autotimeseries`

- Open `jupyterlab`
- Execute each cell in `nbs/m5.ipynb`.

### Upload data

Optionally you can upload the m5 data directly using the terminal.
- Upload data: `aws s3 sync data/m5/parquet/. s3://[YOUR_BUCKET]/nixtla --exclude "datasets/*" --profile [YOUR_PROFILE]`

## Amazon Forecast

- Amazon Forecast needs csv files mandatorily.
- So use `python -m src.download_data --directory [YOUR_DIRECTORY] --format csv`

### Upload data

- Upload data: `aws s3 sync data/m5/csv/. s3://[YOUR_BUCKET]/amazon_forecast --exclude "datasets/*" --profile [YOUR_PROFILE]`

## Download forecasts

- `aws s3 sync s3://[YOUR_BUCKET]/[nixtla or amazon_forecast]/forecasts forecasts --profile [YOUR_PROFILE]`
