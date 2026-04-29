import pandas as pd

from rag.llm import GroqService
from rag.vector_store import SQLiteVectorStore


class RAGService:
    def __init__(self, vector_store: SQLiteVectorStore, llm_service: GroqService):
        self.vector_store = vector_store
        self.llm_service = llm_service

    @staticmethod
    def _row_to_text(row: pd.Series) -> str:
        return " | ".join([f"{col}: {row[col]}" for col in row.index])

    def ingest_dataframe(self, source_name: str, source_type: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        chunks = []
        for idx, row in df.fillna("").iterrows():
            row_text = self._row_to_text(row)
            chunks.append(
                {
                    "text": row_text,
                    "metadata": {"row_index": int(idx), "columns": list(df.columns)},
                }
            )
        embeddings = [self.llm_service.embed_text(c["text"]) for c in chunks]
        return self.vector_store.add_chunks(source_name, source_type, chunks, embeddings)

    def ask(self, question: str, top_k: int = 5) -> tuple[str, list[dict]]:
        query_embedding = self.llm_service.embed_text(question)
        hits = self.vector_store.search(query_embedding, top_k=top_k)
        contexts = [h["text"] for h in hits]
        answer = self.llm_service.answer_with_context(question, contexts)
        return answer, hits
