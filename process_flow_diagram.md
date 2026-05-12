# Process Flow  
---
*Professional SQL-to-Vector RAG to connect SQL databases, ingesting query results as chunks into a vector store, and supporting chat over retrieved context.*
---
##  Diagram
```mermaid
flowchart TD
    A[Start] --> B[Initialize RAGService]
    B --> C{Ingest Dataframe?}
    C -- Yes --> D[Iterate over DataFrame rows]
    D --> E[_row_to_text] --> F[Create chunk with text & metadata]
    F --> G[Collect all chunks]
    G --> H[Embed each chunk via GroqService]
    H --> I[Add chunks + embeddings to SQLiteVectorStore]
    I --> J[Ingestion Complete]
    C -- No (Ask) --> K[Receive Question]
    K --> L[Embed question via GroqService]
    L --> M[Search VectorStore for top_k similar chunks]
    M --> N[Extract contexts from hits]
    N --> O[Answer with context via GroqService]
    O --> P[Return answer and hits]
    J --> Q[Ready for queries]
    Q --> K
    P --> R[End]
```

---
### Execution Flow
1. **Initialization** – An `RAGService` instance is created with concrete `SQLiteVectorStore` and `GroqService` objects (usually in `app.py`).
2. **Ingestion** – `ingest_dataframe` transforms each DataFrame row into a textual chunk, enriches it with metadata, embeds it via the LLM, and stores both in the vector store.
3. **Querying** – `ask` embeds the incoming question, retrieves the most relevant chunks, builds a context list, and asks the LLM to generate an answer.
4. **Result** – The caller receives the answer string and the hit metadata for traceability.

