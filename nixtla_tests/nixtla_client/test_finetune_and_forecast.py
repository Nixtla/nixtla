import pytest
from nixtla.nixtla_client import ApiError
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse


class TestTimeSeriesDataSet1():
    def test_finetuning_and_forecasting(self, custom_client, ts_data_set1):
        # Finetune the model
        finetune_resp = custom_client.finetune(ts_data_set1.train, output_model_id=ts_data_set1.model_id1)
        assert finetune_resp == ts_data_set1.model_id1
        model_id2 = custom_client.finetune(ts_data_set1.train, finetuned_model_id=ts_data_set1.model_id1)

        # Forecast with fine-tuned models
        forecast_base = custom_client.forecast(ts_data_set1.train, h=ts_data_set1.h)
        forecast1 = custom_client.forecast(ts_data_set1.train, h=ts_data_set1.h, finetuned_model_id=ts_data_set1.model_id1)
        forecast2 = custom_client.forecast(ts_data_set1.train, h=ts_data_set1.h, finetuned_model_id=model_id2)
        all_fcsts = forecast_base.assign(ten_rounds=forecast1['TimeGPT'], twenty_rounds=forecast2['TimeGPT'])
        fcst_rmse = evaluate(
            all_fcsts.merge(ts_data_set1.valid),
            metrics=[rmse],
            agg_fn='mean',
        ).loc[0]
        # error was reduced over 40% by finetuning
        assert 1 - fcst_rmse['ten_rounds'] / fcst_rmse['TimeGPT'] > 0.4
        # error was reduced over 30% by further finetuning
        assert 1 - fcst_rmse['twenty_rounds'] / fcst_rmse['ten_rounds'] > 0.3

        # non-existent model returns 404
        with pytest.raises(ApiError) as excinfo:
            custom_client.forecast(ts_data_set1.train, h=ts_data_set1.h, finetuned_model_id='unexisting')
        assert getattr(excinfo.value, "status_code", None) == 404

    def test_cv_with_finetuned_model(self, custom_client, ts_data_set1):
        cv_base = custom_client.cross_validation(ts_data_set1.series, n_windows=2, h=ts_data_set1.h)
        cv_finetune = custom_client.cross_validation(ts_data_set1.series, n_windows=2, h=ts_data_set1.h, finetuned_model_id=ts_data_set1.model_id1)
        merged = cv_base.merge(
            cv_finetune,
            on=['unique_id', 'ds', 'cutoff', 'y'],
            suffixes=('_base', '_finetune')
        ).drop(columns='cutoff')
        cv_rmse = evaluate(
            merged,
            metrics=[rmse],
            agg_fn='mean',
        ).loc[0]
        # error was reduced over 40% by finetuning
        assert 1 - cv_rmse['TimeGPT_finetune'] / cv_rmse['TimeGPT_base'] > 0.4

        custom_client.delete_finetuned_model(ts_data_set1.model_id1)
