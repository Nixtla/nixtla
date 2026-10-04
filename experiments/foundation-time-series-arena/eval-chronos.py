import fire
import transformers
from xiuhmolpilli.arena import FoundationalTimeSeriesArena
from xiuhmolpilli.models.foundational import Chronos


if __name__ == "__main__":
    transformers.set_seed(42)  # for reproducibility

    frequencies = ["Hourly", "Daily", "Weekly", "Monthly"]
    files = [
        f"./nixtla-foundational-time-series/data/{freq}.parquet" for freq in frequencies
    ]
    arena = FoundationalTimeSeriesArena(
        models=[
            Chronos(
                repo_id="amazon/chronos-t5-large", batch_size=16, alias="Chronos-Large"
            ),
            Chronos(
                repo_id="amazon/chronos-t5-base", batch_size=40, alias="Chronos-Base"
            ),
            Chronos(
                repo_id="amazon/chronos-t5-small", batch_size=64, alias="Chronos-Small"
            ),
            Chronos(
                repo_id="amazon/chronos-t5-mini", batch_size=128, alias="Chronos-Mini"
            ),
            Chronos(
                repo_id="amazon/chronos-t5-tiny", batch_size=256, alias="Chronos-Tiny"
            ),
        ],
        parquet_data_paths=files,
    )
    fire.Fire(arena.compete)
