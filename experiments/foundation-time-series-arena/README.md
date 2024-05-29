# Instructions

## Download data

### Configure aws cli

Follow the instructions [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

### Download data from s3

```
make download_data
```


## Download lag llama code

```
make download_lag_llama_code
```


## Create environment

```
mamba create -n foundation-ts python=3.10
conda activate foundation-ts
pip install uv
uv pip install -r requirements.txt
```

## Run

```
python -m xiuhmolpilli.arena
```
