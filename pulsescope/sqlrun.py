"""DuckDB access layer over the Parquet lake produced by Spark, plus a runner for the .sql library."""
from __future__ import annotations

from functools import lru_cache

import duckdb
import pandas as pd

from . import config as C

TABLES = ["events", "users", "sessions", "orders", "daily_kpis", "experiment_assignments",
          "experiment_users", "user_features"]


@lru_cache(maxsize=1)
def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET threads TO 8")
    for t in TABLES:
        path = C.LAKE / t
        if path.exists():
            con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM "
                        f"read_parquet('{path}/**/*.parquet', hive_partitioning = true)")
    return con


def query(sql: str, **params) -> pd.DataFrame:
    return connect().execute(sql.format(**params)).df()


def run_file(name: str, **params) -> pd.DataFrame:
    """Run sql/<name>.sql; `{param}` placeholders are filled from params (trusted, internal values only)."""
    return query((C.SQL_DIR / f"{name}.sql").read_text(), **params)
