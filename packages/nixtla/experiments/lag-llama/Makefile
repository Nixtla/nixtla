download_lag_llama_code:
	@git clone https://github.com/time-series-foundation-models/lag-llama tempdir
	@cp -R tempdir/data/ .
	@cp -R tempdir/gluon_utils/ .
	@cp -R tempdir/lag_llama/ .
	@cp -R tempdir/requirements.txt lag-llama-requirements.txt
	@rm -rf tempdir

download_lag_llama_model:
	@huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir ./models/
