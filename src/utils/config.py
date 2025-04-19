import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()


@dataclass
class ClickHouseConfig:
    host: str = os.getenv("CLICKHOUSE_HOST")
    port_native: int = int(os.getenv("CLICKHOUSE_PORT", 9000))
    port_http: int = int(os.getenv("CLICKHOUSE_HTTP_PORT", 8123))
    user: str = os.getenv("CLICKHOUSE_USER")
    password: str = os.getenv("CLICKHOUSE_PASSWORD")
    database: str = os.getenv("CLICKHOUSE_DATABASE")
    learning_table: str = os.getenv("LEARNING_TABLE_NAME")
    inference_table: str = os.getenv("INFERENCE_TABLE_NAME")
    output_table: str = os.getenv("OUTPUT_TABLE_NAME")

clickhouse_config = ClickHouseConfig()
