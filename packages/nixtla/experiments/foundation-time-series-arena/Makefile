download_lag_llama_code:
	@git clone https://github.com/time-series-foundation-models/lag-llama tempdir
	@cp -R tempdir/data/ .
	@cp -R tempdir/gluon_utils/ .
	@cp -R tempdir/lag_llama/ .
	@rm -rf tempdir

download_data:
	@aws s3 sync s3://nixtla-foundational-time-series/data nixtla-foundational-time-series/data --no-sign-request
