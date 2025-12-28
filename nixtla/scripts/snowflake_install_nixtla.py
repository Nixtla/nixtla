import os
import shutil
from tempfile import TemporaryDirectory

import pandas as pd
import pip
from fire import Fire
from rich import print
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt
from snowflake.snowpark import Session

"""
Example Snowflake credential file (~/.snowflake/config.toml):

```toml
[connections.default]
account = "..."
user = "..."
password = "..."
database = "..."
schema = "..."
```

Make sure to CHMOD 600 the file to protect your credentials.

```bash
chmod 600 ~/.snowflake/config.toml
```
"""

PACKAGES = [
    "annotated-types",
    "anyio",
    "certifi",
    "httpcore",
    "orjson",
    "pandas",
    "tenacity",
    "tqdm",
    "numpy",
    "zstandard",
    "httpx",
    "pydantic",
    "snowflake-snowpark-python",
    "requests",
]

TEMPLATE_ACCESS_INTEGRATION = """
CREATE OR REPLACE NETWORK RULE {ds_prefix}nixtla_network_rule
MODE = EGRESS
TYPE = HOST_PORT
VALUE_LIST = ('api.nixtla.io');

//<br>

CREATE OR REPLACE SECRET {ds_prefix}nixtla_api_key
  TYPE = GENERIC_STRING
  SECRET_STRING = '{nixtla_api_key}';

//<br>

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION nixtla_access_integration
  ALLOWED_NETWORK_RULES = ({ds_prefix}nixtla_network_rule)
  ALLOWED_AUTHENTICATION_SECRETS = ({ds_prefix}nixtla_api_key)
  ENABLED = true;
"""

TEMPLATE_SP = """
CREATE OR REPLACE PROCEDURE {ds_prefix}NIXTLA_FORECAST(
    "INPUT_DATA" VARCHAR,
    "PARAMS" OBJECT,
    "MAX_BATCHES" NUMBER(38,0) DEFAULT 1000
)
RETURNS TABLE (
  UNIQUE_ID VARCHAR, DS TIMESTAMP, FORECAST DOUBLE
)
LANGUAGE SQL AS
$$
DECLARE
  res RESULTSET;
BEGIN
  res := (SELECT b.* FROM 
    (SELECT
        MOD(HASH($1), :MAX_BATCHES) AS gp,
        TO_VARCHAR($1) AS unique_id,
        $2::TIMESTAMP_NTZ AS ds,
        TO_DOUBLE($3) AS y
     FROM TABLE(:INPUT_DATA)) a,
    TABLE(nixtla_forecast_batch(:PARAMS, unique_id, ds, y) OVER (PARTITION BY gp ORDER BY unique_id, ds)) b);
  RETURN TABLE(res);
END;
$$
;

//<br>

CREATE OR REPLACE PROCEDURE {ds_prefix}NIXTLA_EVALUATE(
    "INPUT_DATA" VARCHAR,
    "METRICS" ARRAY DEFAULT ['MAPE', 'MAE'],
    "MAX_BATCHES" NUMBER(38,0) DEFAULT 1000000
)
RETURNS TABLE (
  UNIQUE_ID VARCHAR, FORECASTER VARCHAR, METRIC VARCHAR, VALUE DOUBLE
)
LANGUAGE SQL AS
$$
DECLARE
  res RESULTSET;
BEGIN
  res := (SELECT b.* FROM 
    (SELECT MOD(HASH(unique_id), :MAX_BATCHES) AS gp, unique_id, ds, y, object_construct(*) AS obj FROM TABLE(:INPUT_DATA)) a,
    TABLE(nixtla_evaluate_batch(:METRICS, obj) OVER (PARTITION BY gp ORDER BY unique_id, ds)) b);
  RETURN TABLE(res);
END;
$$
;
"""

SECURITY_PARAMS = dict(
    secrets={"nixtla_api_key": "nixtla_api_key"},
    external_access_integrations=["nixtla_access_integration"],
)


def ask(
    question: str, *defaults, password: bool = False, skip_if_provided: bool = False
) -> str:
    dft = None
    for d in defaults:
        res = d()
        if res:
            dft = res
            break
    if dft and skip_if_provided:
        if password:
            print(question, " -- using default value")
        else:
            print(question, " -- using default value: ", dft)
        return dft
    if password:
        if dft:
            question += " (press enter to use the default value)"
        else:
            question += " (must provide, default value was not found)"
    while True:
        res = Prompt.ask(
            question,
            default=dft,
            password=password,
            show_default=not password and dft is not None,
        )
        if res:
            return res
        print("Please provide a valid answer.")


def make_udtf(stage_path: str, session: Session):
    import pandas as pd
    from snowflake.snowpark.functions import udtf
    from snowflake.snowpark.types import (
        ArrayType,
        DoubleType,
        MapType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    common_params = dict(
        session=session,
        packages=PACKAGES,
        imports=[f"@{stage_path}/nixtla.zip"],
        replace=True,
        immutable=True,
        is_permanent=True,
        stage_location=stage_path,
    )

    @udtf(
        input_types=[MapType(), StringType(), TimestampType(), DoubleType()],
        output_schema=StructType(
            [
                StructField("unique_id", StringType()),
                StructField("ds", TimestampType()),
                StructField("forecast", DoubleType()),
            ]
        ),
        name="nixtla_forecast_batch",
        **SECURITY_PARAMS,
        **common_params,
    )
    class forecast_udtf:
        def __init__(self):
            import _snowflake

            from nixtla import NixtlaClient

            token = _snowflake.get_generic_secret_string("nixtla_api_key")
            self.client = NixtlaClient(api_key=token)

        def end_partition(self, df: pd.DataFrame) -> pd.DataFrame:
            _df = pd.DataFrame(
                {"unique_id": df.iloc[:, 1], "ds": df.iloc[:, 2], "y": df.iloc[:, 3]}
            )
            conf = df.iloc[0, 0]
            if conf.get("finetune_steps", 0) > 0:
                raise ValueError("Finetuning is not allowed during forecasting")
            fcst = self.client.forecast(df=_df, **conf)
            vcol = None
            for col in fcst.columns:
                if col.lower() not in ["unique_id", "ds", "y"]:
                    vcol = col
                    break
            if vcol is None:
                raise ValueError("No valid column found for forecast")
            return fcst[["unique_id", "ds", vcol]].rename(columns={vcol: "forecast"})

        end_partition._sf_vectorized_input = pd.DataFrame  # type: ignore

    @udtf(
        input_types=[ArrayType(), MapType()],
        output_schema=StructType(
            [
                StructField("unique_id", StringType()),
                StructField("forecaster", StringType()),
                StructField("metric", StringType()),
                StructField("value", DoubleType()),
            ]
        ),
        name="nixtla_evaluate_batch",
        **common_params,
    )
    class evaluate_udtf:
        def end_partition(self, df: pd.DataFrame) -> pd.DataFrame:
            from utilsforecast.evaluation import evaluate

            metrics = list(self.parse_metrics(df.iloc[0, 0]))
            tdf = pd.DataFrame(df.iloc[:, 1].tolist())
            tdf.columns = tdf.columns.str.lower()
            tdf = tdf.sort_values(by=["unique_id", "ds"])
            tdf["ds"] = pd.to_datetime(tdf["ds"])
            forecasters = [
                k for k in tdf.columns if k.lower() not in ["unique_id", "ds", "y"]
            ]
            res = evaluate(tdf, metrics=metrics, models=forecasters, train_df=tdf)
            res = pd.melt(
                res,
                id_vars=["unique_id", "metric"],
                var_name="forecaster",
                value_name="_metric_value_",
            ).rename(columns={"_metric_value_": "value"})
            return res[["unique_id", "forecaster", "metric", "value"]]

        def parse_metrics(self, metrics):
            # TODO: implement in utilsforecast
            from utilsforecast import losses

            for k in metrics:
                if k.lower() == "mape":
                    yield losses.mape
                elif k.lower() == "mae":
                    yield losses.mae
                elif k.lower() == "mse":
                    yield losses.mse
                # elif k.lower().startswith("mase"):
                #    seasonality = int(k[4:].strip()[1:-1])
                #    yield partial(losses.mase, seasonality=seasonality)
                else:
                    raise NotImplementedError(f"Unsupported metric {k}")

        end_partition._sf_vectorized_input = pd.DataFrame  # type: ignore


def make_finetune_sproc(stage_path: str, session: Session):
    from snowflake.snowpark.functions import sproc

    @sproc(
        session=session,
        name="nixtla_finetune",
        packages=PACKAGES,
        imports=[f"@{stage_path}/nixtla.zip"],
        replace=True,
        is_permanent=True,
        stage_location=stage_path,
        **SECURITY_PARAMS,
    )
    def nixtla_finetune(
        session: Session,
        input_data: str,
        params: dict = {},
        max_series: int = 1000,
    ) -> str:
        import _snowflake
        from snowflake.snowpark import functions as F

        from nixtla import NixtlaClient

        token = _snowflake.get_generic_secret_string("nixtla_api_key")
        client = NixtlaClient(api_key=token)
        input = session.table(input_data)
        ids = (
            input.select("unique_id", F.hash("unique_id").alias("_od"))
            .distinct()
            .order_by("_od")
            .limit(max_series)
        )
        train_data = (
            input.join(ids, on="unique_id", how="inner", rsuffix="_")
            .select("unique_id", "ds", "y")
            .order_by("unique_id", "ds")
            .to_pandas()
        )
        train_data.columns = train_data.columns.str.lower()
        if "finetune_steps" not in params:
            params["finetune_steps"] = train_data["unique_id"].nunique()
        return client.finetune(train_data, **params)


def deploy_snowflake(
    connection_name: str = "default",
    database: str | None = None,
    schema: str | None = None,
    stage_path: str | None = None,
):
    with Session.builder.config("connection_name", connection_name).create() as session:
        _database = ask(
            "Snowflake database: ",
            lambda: database,
            lambda: session.get_current_database(),
        )
        session.use_database(_database)
        _schema = ask(
            "Snowflake schema: ",
            lambda: schema,
            lambda: session.get_current_schema(),
        )
        session.use_schema(_schema)
        stage = ask(
            "Stage path (without @) for the artifact: ",
            lambda: stage_path,
        )
        sf_prefix = f"{_database}.{_schema}."
        if Confirm.ask("Do you want to generate the security script?", default=False):
            nixtla_api_key = ask(
                "Nixtla API key: ",
                lambda: os.environ.get("NIXTLA_API_KEY"),
                password=True,
            )
            ai = TEMPLATE_ACCESS_INTEGRATION.format(
                ds_prefix=sf_prefix, nixtla_api_key=nixtla_api_key
            )
            print(Markdown(f"```sql\n{ai}\n```"))
            if Confirm.ask("Do you want to run the script now?", default=False):
                for query in ai.split("//<br>"):
                    query = query.strip().rstrip(";")
                    session.sql(query).collect()
                print("[green]Access integration created![/green]")

        if Confirm.ask("Do you want to (re)package the Nixtla client?", default=False):
            with TemporaryDirectory() as tmpdir:
                # Import version from nixtla package
                from nixtla import __version__ as nixtla_version

                pip.main(
                    f"install -t {tmpdir} nixtla=={nixtla_version} utilsforecast httpx --no-deps".split(
                        " "
                    )
                )
                shutil.make_archive(os.path.join(tmpdir, "nixtla"), "zip", tmpdir)
                fp = os.path.join(tmpdir, "nixtla.zip")
                res = session.sql(
                    f"PUT file://{fp} @{stage} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
                ).collect()
                if res[0]["status"].lower() != "uploaded":
                    raise ValueError("Upload failed " + str(res))
                print("[green]Nixtla client package uploaded[/green]")

        if Confirm.ask("Do you want to (re)create the UDTFs?", default=False):
            make_udtf(
                stage_path=stage,
                session=session,
            )
            print("[green]UDTF function created![/green]")

        if Confirm.ask(
            "Do you want to generate the stored procedures script for inference and evaluation?",
            default=False,
        ):
            sp = TEMPLATE_SP.format(ds_prefix=sf_prefix)
            print(Markdown(f"```sql\n{sp}\n```"))
            if Confirm.ask("Do you want to run the script now?", default=False):
                for query in sp.split("//<br>"):
                    query = query.strip().rstrip(";")
                    session.sql(query).collect()
                print("[green]Stored procedure created![/green]")

        if Confirm.ask(
            "Do you want to (re)create the stored procedure for finetuning?",
            default=False,
        ):
            make_finetune_sproc(
                stage_path=stage,
                session=session,
            )
            print("[green]Stored procedure created![/green]")

        if Confirm.ask(
            "Do you want to (re)create the example datasets?",
            default=False,
        ):
            all_data = pd.read_parquet("all_data.parquet")
            all_data.columns = all_data.columns.str.upper()
            session.write_pandas(
                all_data,
                "EXAMPLE_ALL_DATA",
                auto_create_table=True,
                overwrite=True,
                use_logical_type=True,
            )
            train = pd.read_parquet("train.parquet")
            train.columns = train.columns.str.upper()
            session.write_pandas(
                train,
                "EXAMPLE_TRAIN",
                auto_create_table=True,
                overwrite=True,
                use_logical_type=True,
            )
            print(
                "[green]Example datasets (example_all_data and example_train) created![/green]"
            )


def main():
    Fire(deploy_snowflake)
