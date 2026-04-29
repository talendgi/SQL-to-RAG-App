import json
import sqlite3
import uuid

import numpy as np


class SQLiteVectorStore:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._setup()

    def _setup(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding_json TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def clear(self) -> None:
        self.conn.execute("DELETE FROM chunks")
        self.conn.commit()

    def add_chunks(self, source_name: str, source_type: str, chunks: list[dict], embeddings: list[list[float]]) -> int:
        rows = []
        for chunk, emb in zip(chunks, embeddings):
            rows.append(
                (
                    str(uuid.uuid4()),
                    source_name,
                    source_type,
                    chunk["text"],
                    json.dumps(chunk.get("metadata", {})),
                    json.dumps(emb),
                )
            )
        self.conn.executemany(
            """
            INSERT INTO chunks (id, source_name, source_type, text, metadata_json, embedding_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        return len(rows)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT source_name, source_type, text, metadata_json, embedding_json FROM chunks"
        )
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec) + 1e-8
        scored = []
        for source_name, source_type, text, metadata_json, embedding_json in cursor.fetchall():
            emb = np.array(json.loads(embedding_json), dtype=np.float32)
            denom = (np.linalg.norm(emb) + 1e-8) * query_norm
            score = float(np.dot(query_vec, emb) / denom)
            scored.append(
                {
                    "source_name": source_name,
                    "source_type": source_type,
                    "text": text,
                    "metadata": json.loads(metadata_json),
                    "score": score,
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def list_sources(self) -> list[dict]:
        cursor = self.conn.execute(
            """
            SELECT source_name, source_type, COUNT(*) AS chunk_count
            FROM chunks
            GROUP BY source_name, source_type
            ORDER BY source_name
            """
        )
        return [
            {"source_name": row[0], "source_type": row[1], "chunk_count": row[2]}
            for row in cursor.fetchall()
        ]

    def close(self) -> None:
        self.conn.close()
