from gluonts.torch.model.predictor import PyTorchPredictor
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

from ..utils.gluonts_forecaster import GluonTSForecaster


class Moirai(GluonTSForecaster):
    def __init__(
        self,
        repo_id: str = "Salesforce/moirai-1.0-R-large",
        filename: str = "model.ckpt",
        alias: str = "Moirai",
    ):
        super().__init__(
            repo_id=repo_id,
            filename=filename,
            alias=alias,
        )

    def get_predictor(self, prediction_length: int) -> PyTorchPredictor:
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(self.repo_id),
            prediction_length=prediction_length,
            context_length=1000,
            patch_size=32,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=512)
        return predictor
