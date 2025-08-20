import pytest

from nixtla.nixtla_client import ApiError
from nixtla_tests.helpers.checks import check_finetuned_model
from nixtla_tests.helpers.states import model_ids_object
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse


class TestTimeSeriesDataSet1:
    def test_finetuning_and_forecasting(self, custom_client, ts_data_set1):
        # Finetune the model
        finetune_resp = custom_client.finetune(
            ts_data_set1.train, output_model_id=model_ids_object.model_id1
        )
        assert finetune_resp == model_ids_object.model_id1

        model_id2 = custom_client.finetune(
            ts_data_set1.train, finetuned_model_id=model_ids_object.model_id1
        )
        model_ids_object.model_id2 = model_id2  # store the model_id2 for later use

        # Forecast with fine-tuned models
        forecast_base = custom_client.forecast(ts_data_set1.train, h=ts_data_set1.h)
        forecast1 = custom_client.forecast(
            ts_data_set1.train,
            h=ts_data_set1.h,
            finetuned_model_id=model_ids_object.model_id1,
        )
        forecast2 = custom_client.forecast(
            ts_data_set1.train,
            h=ts_data_set1.h,
            finetuned_model_id=model_ids_object.model_id2,
        )
        all_fcsts = forecast_base.assign(
            ten_rounds=forecast1["TimeGPT"], twenty_rounds=forecast2["TimeGPT"]
        )
        fcst_rmse = evaluate(
            all_fcsts.merge(ts_data_set1.valid),
            metrics=[rmse],
            agg_fn="mean",
        ).loc[0]
        # error was reduced over 40% by finetuning
        assert 1 - fcst_rmse["ten_rounds"] / fcst_rmse["TimeGPT"] > 0.4
        # error was reduced over 30% by further finetuning
        assert 1 - fcst_rmse["twenty_rounds"] / fcst_rmse["ten_rounds"] > 0.3

        # non-existent model returns 404
        with pytest.raises(ApiError) as excinfo:
            custom_client.forecast(
                ts_data_set1.train, h=ts_data_set1.h, finetuned_model_id="unexisting"
            )
        assert getattr(excinfo.value, "status_code", None) == 404

        # Enough data to finetune
        _, model_horizon = custom_client._get_model_params("timegpt-1", ts_data_set1.freq)
        forecast3 = custom_client.forecast(ts_data_set1.train.tail(model_horizon + 1), h=ts_data_set1.h, finetune_steps=10)

        # Not enough data to finetune
        with pytest.raises(
            ValueError, match="Some series are too short. Please make sure that each series contains at least 8 observations."
        ):
            custom_client.forecast(ts_data_set1.train.tail(model_horizon), h=ts_data_set1.h, finetune_steps=10)

    def test_cv_with_finetuned_model(self, custom_client, ts_data_set1):
        cv_base = custom_client.cross_validation(
            ts_data_set1.series, n_windows=2, h=ts_data_set1.h
        )
        cv_finetune = custom_client.cross_validation(
            ts_data_set1.series,
            n_windows=2,
            h=ts_data_set1.h,
            finetuned_model_id=model_ids_object.model_id1,
        )
        merged = cv_base.merge(
            cv_finetune,
            on=["unique_id", "ds", "cutoff", "y"],
            suffixes=("_base", "_finetune"),
        ).drop(columns="cutoff")
        cv_rmse = evaluate(
            merged,
            metrics=[rmse],
            agg_fn="mean",
        ).loc[0]
        # error was reduced over 40% by finetuning
        assert 1 - cv_rmse["TimeGPT_finetune"] / cv_rmse["TimeGPT_base"] > 0.4

        custom_client.delete_finetuned_model(model_ids_object.model_id1)

    def test_anomaly_detection_with_finetuned_model(
        self, custom_client, ts_anomaly_data
    ):
        anomaly_base = custom_client.detect_anomalies(ts_anomaly_data.train_anomalies)
        anomaly_finetune = custom_client.detect_anomalies(
            ts_anomaly_data.train_anomalies,
            finetuned_model_id=model_ids_object.model_id2,
        )
        detected_anomalies_base = (
            anomaly_base.set_index("ds")
            .loc[ts_anomaly_data.anomaly_date, "anomaly"]
            .sum()
        )
        detected_anomalies_finetune = (
            anomaly_finetune.set_index("ds")
            .loc[ts_anomaly_data.anomaly_date, "anomaly"]
            .sum()
        )
        assert detected_anomalies_base < detected_anomalies_finetune

    def test_list_finetuned_models(self, custom_client):
        models = custom_client.finetuned_models()
        ids = {m.id for m in models}
        assert (
            model_ids_object.model_id1 not in ids and model_ids_object.model_id2 in ids
        )

    def test_get_single_finetuned_model(self, custom_client):
        single_model = custom_client.finetuned_model(model_ids_object.model_id2)
        assert single_model.id == model_ids_object.model_id2
        assert single_model.base_model_id == model_ids_object.model_id1

    def test_non_existing_model_returns_error(self, custom_client):
        with pytest.raises(ApiError, match="Model not found"):
            custom_client.finetuned_model("hi")

    @pytest.mark.distributed_run
    @pytest.mark.ray_run
    def test_ray_finetune_model(self, custom_client, ray_df):
        check_finetuned_model(custom_client, ray_df, model_ids_object.model_id2)

    @pytest.mark.distributed_run
    @pytest.mark.spark_run
    def test_spark_finetune_model(self, custom_client, spark_df):
        check_finetuned_model(custom_client, spark_df, model_ids_object.model_id2)

    @pytest.mark.distributed_run
    @pytest.mark.flaky(reruns=3, delay=10)
    def test_dask_finetune_model(self, custom_client, dask_df):
        check_finetuned_model(custom_client, dask_df, model_ids_object.model_id2)
