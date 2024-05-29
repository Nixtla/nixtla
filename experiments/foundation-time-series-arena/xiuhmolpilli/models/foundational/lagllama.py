from gluonts.torch.model.predictor import PyTorchPredictor
from lag_llama.gluon.estimator import LagLlamaEstimator

from ..utils.gluonts_forecaster import GluonTSForecaster


class LagLlama(GluonTSForecaster):
    def __init__(
        self,
        repo_id: str = "time-series-foundation-models/Lag-Llama",
        filename: str = "lag-llama.ckpt",
        alias: str = "LagLlama",
    ):
        super().__init__(
            repo_id=repo_id,
            filename=filename,
            alias=alias,
        )

    def get_predictor(self, prediction_length: int) -> PyTorchPredictor:
        ckpt = self.load()
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        # this context length is reported in the paper
        context_length = 32
        estimator = LagLlamaEstimator(
            ckpt_path=self.checkpoint_path,
            prediction_length=prediction_length,
            context_length=context_length,
            # estimator args
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
        )
        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)
        return predictor
