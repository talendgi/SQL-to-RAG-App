import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    embedding_provider: str
    embedding_model: str
    vector_db_path: str


def get_settings() -> Settings:
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
        embedding_model=os.getenv("EMBEDDING_MODEL", ""),
        vector_db_path=os.getenv("VECTOR_DB_PATH", "vector_store.sqlite"),
    )
