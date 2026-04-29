import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


class SQLConnector:
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self.engine: Engine | None = None

    def connect(self) -> None:
        self.engine = create_engine(self.connection_url)
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def list_tables(self) -> list[str]:
        if self.engine is None:
            return []
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def query_dataframe(self, sql_query: str) -> pd.DataFrame:
        if self.engine is None:
            raise RuntimeError("Database is not connected.")
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql_query), conn)
