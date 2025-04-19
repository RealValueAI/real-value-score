import logging
from pathlib import Path

import pandas as pd
from clickhouse_driver import Client

from src.learning.ml_model import RatingModel
from src.learning.preprocessing import RealValuePreprocessor
from src.utils.config import clickhouse_config

logger = logging.getLogger("inference_pipeline")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

RATING_BINS = 10


def run_inference_pipeline(
    data_path: str | Path = "x_inference.parquet",
    model_dir: str | Path = "models",
    output_table: str | None = None,
) -> pd.DataFrame:
    """
    Загрузка новых данных, предсказание вероятностей и рейтингов для трех аспектов,
    и запись результатов в ClickHouse.

    Возвращает DataFrame с колонками:
      listing_id, platform_id,
      flat_prob, flat_rating,
      building_prob, building_rating,
      location_prob, location_rating
    """
    data_path = Path(data_path)
    model_dir = Path(model_dir)
    output_table = output_table or clickhouse_config.output_table

    if not data_path.exists():
        raise FileNotFoundError(f"Inference data not found: {data_path}")
    if output_table is None:
        raise ValueError("Output table not specified")

    # 1) Load and preprocess
    logger.info(f"Loading inference data from {data_path}")
    prep = RealValuePreprocessor(
        array_cols=["subway_distances", "subway_names"],
        numeric_fillna={
            "price_per_sqm": 0,
            "area": 0,
            "rooms": 0,
            "floor": -1,
            "height": -1,
        },
        bool_cols=["placement_paid"],
    )
    df = prep.process(data_path, include_raw=True)

    key_cols = [c for c in ("listing_id", "platform_id") if c in df.columns]
    if "platform_id" not in key_cols:
        logger.warning("platform_id not found, using listing_id only")


    out = pd.DataFrame()
    out[key_cols] = df[key_cols]


    for aspect in ["flat_quality", "building_quality", "location_quality"]:
        model_path = model_dir / f"catboost_{aspect}.cbm"
        logger.info(f"Loading model for {aspect} from {model_path}")
        rm = RatingModel()
        rm.load(model_path)

        X, _, cat_idx = prep.split_xy(
            df,
            features=None,
            cat_features=None,
            target=aspect
        )

        X = df[X.columns]
        probs = rm.predict_proba(X)
        ratings = pd.qcut(probs, RATING_BINS, labels=False, duplicates='drop')

        out[f"{aspect}_prob"] = probs
        out[f"{aspect}_rating"] = ratings.astype(int)

    client = Client(
        host=clickhouse_config.host,
        port=clickhouse_config.port_native,
        user=clickhouse_config.user,
        password=clickhouse_config.password,
        database=clickhouse_config.database,
    )
    schema = ",\n        ".join(
        [
            "listing_id UInt64",
            "platform_id UInt64"
        ] + [
            f"{aspect}_prob Float64" for aspect in ["flat_quality", "building_quality", "location_quality"]
        ] + [
            f"{aspect}_rating UInt8" for aspect in ["flat_quality", "building_quality", "location_quality"]
        ]
    )

    client.execute(f"""
        CREATE TABLE IF NOT EXISTS {output_table} (
        {schema}
        ) ENGINE = MergeTree()
        ORDER BY ({', '.join(key_cols)})
    """
    )

    client.execute(f"TRUNCATE TABLE {output_table}")

    records = []
    for row in out.itertuples(index=False):
        records.append(tuple(getattr(row, col) for col in out.columns))
    client.execute(
        f"INSERT INTO {output_table} ({', '.join(out.columns)}) VALUES",
        records
    )
    logger.info(f"Inserted {len(records)} rows into {output_table}")
    return out
