import hashlib

import numpy as np
from groq import Groq


class GroqService:
    def __init__(
        self,
        api_key: str,
        embedding_provider: str,
        embedding_model: str,
        chat_model: str,
        local_embedding_dim: int = 1024,
    ):
        self.client = Groq(api_key=api_key)
        self.embedding_provider = embedding_provider.strip().lower()
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.local_embedding_dim = local_embedding_dim

    def _local_embed(self, text: str) -> list[float]:
        vec = np.zeros(self.local_embedding_dim, dtype=np.float32)
        for token in text.lower().split():
            h = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % self.local_embedding_dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_text(self, text: str) -> list[float]:
        if self.embedding_provider == "local":
            return self._local_embed(text)
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def answer_with_context(self, question: str, contexts: list[str]) -> str:
        context_block = "\n\n---\n\n".join(contexts) if contexts else "No context found."
        prompt = (
            "You are a helpful assistant for question answering over user-uploaded documents.\n"
            "Use only the provided context when possible. If context is insufficient, say so clearly.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}"
        )
        completion = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message.content or ""
