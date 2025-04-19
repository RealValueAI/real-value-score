import logging
from typing import Literal

import pandas as pd

from src.utils.config import ClickHouseConfig, clickhouse_config
from sqlalchemy import create_engine, text


def create_clickhouse_engine(config: ClickHouseConfig):
    """
    Создает SQLAlchemy engine для подключения к ClickHouse.
    """
    connection_str = (
        f"clickhouse+native://{config.user}:{config.password}@"
        f"{config.host}:{config.port}/{config.database}"
    )
    engine = create_engine(connection_str)
    logging.info("ClickHouse engine создан успешно.")
    return engine


def read_table_to_dataframe(engine, table_full_name: str) -> pd.DataFrame:
    """
    Считывает данные из указанной таблицы ClickHouse и возвращает DataFrame.
    :param engine: SQLAlchemy engine для подключения к ClickHouse.
    :param table_full_name: Полное имя таблицы (например, dwh.listings_actual_c).
    :return: DataFrame с данными из таблицы.
    """
    query = f"SELECT * FROM {table_full_name}"
    logging.info(f"Выполняется запрос: {query}")
    df = pd.read_sql(query, engine)
    logging.info(f"Считано {len(df)} строк из таблицы {table_full_name}.")
    return df


def save_dataframe_to_parquet(df: pd.DataFrame, file_path: str) -> None:
    """
    Сохраняет DataFrame в файл формата Parquet.
    :param df: DataFrame для сохранения.
    :param file_path: Путь к выходному файлу Parquet.
    """
    df.to_parquet(file_path, index=False)
    logging.info(f"DataFrame сохранен в Parquet файл: {file_path}")


def refresh_data(
    epoch: Literal['learning', 'inference'],
    output_file: str = 'data.parquet'
) -> pd.DataFrame:
    """
    Загружает таблицу из ClickHouse по типу эпохи ('learning' or 'inference')
    и сохраняет результат в Parquet.
    """
    # Select table based on epoch
    if epoch == 'learning':
        table = clickhouse_config.learning_table
    elif epoch == 'inference':
        table = clickhouse_config.inference_table
    else:
        raise ValueError(f"Unsupported epoch: {epoch}")

    if not table:
        raise ValueError(f"Environment variable for {epoch}_table not set.")

    full_table = f"{clickhouse_config.database}.{table}"

    engine_url = (
        f"clickhouse://{clickhouse_config.user}:{clickhouse_config.password}@"
        f"{clickhouse_config.host}:{clickhouse_config.port_http}/"
        f"{clickhouse_config.database}?protocol=http"
    )
    engine = create_engine(engine_url)

    # Read full table into DataFrame
    query = f"SELECT * FROM {full_table}"
    df = pd.read_sql(query, engine)

    # Save to Parquet
    df.to_parquet(output_file, index=False)
    logging.info(f"Saved {len(df)} rows from {full_table} to {output_file}")
    return df




def upload_group_labels(parquet_path: str, table_full_name: str):
    """
    Загружает в ClickHouse содержимое parquet_path (listing_id, platform_id, group_id)
    в таблицу table_full_name: сначала дропаем, затем создаём и вставляем данные.
    """

    engine = create_clickhouse_engine(clickhouse_config)
    df = pd.read_parquet(parquet_path, columns=['listing_id', 'platform_id', 'group_id'])
    logging.info(f"Прочитано {len(df)} строк из {parquet_path}.")


    if '.' in table_full_name:
        schema, table = table_full_name.split('.', 1)
    else:
        schema, table = engine.url.database, table_full_name

    # 3) Дроп и создание новой таблицы
    ddl_drop = text(f"DROP TABLE IF EXISTS {table_full_name}")
    ddl_create = text(f"""
        CREATE TABLE {table_full_name} (
            listing_id   UInt64,
            platform_id  UInt8,
            group_id     Int64
        ) ENGINE = MergeTree()
        ORDER BY (listing_id)
    """)
    with engine.begin() as conn:
        conn.execute(ddl_drop)
        logging.info(f"Таблица {table_full_name} удалена (если существовала).")
        conn.execute(ddl_create)
        logging.info(f"Таблица {table_full_name} создана заново.")


    df = df.astype(
        {
            'listing_id': 'int64',
            'platform_id': 'int8',
            'group_id': 'int64'
        }
    )
    df.to_sql(
        name=table,
        con=engine,
        schema=schema,
        if_exists='append',
        index=False,
    )
    logging.info(f"Загружено {len(df)} записей в {table_full_name}.")