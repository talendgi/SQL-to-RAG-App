# Professional SQL-to-Vector RAG (Streamlit + Groq)

This app connects to SQL databases, ingests query results as chunks into a vector store, and supports chat over retrieved context.

## Supported DB Types

- MySQL
- SQL Server (MSSQL)
- Snowflake
- PostgreSQL
- SQLite

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `.env`:

- `GROQ_API_KEY=...`
- `GROQ_MODEL=llama-3.1-8b-instant`
- `EMBEDDING_PROVIDER=local`
- `EMBEDDING_MODEL=`
- `VECTOR_DB_PATH=vector_store.sqlite`

## Run

```bash
streamlit run app.py
```

## Workflow

1. Connect in **Database Connect + Ingest** tab.
2. Run SQL query to preview rows.
3. Ingest result set into vector DB.
4. Ask questions in **RAG Chat** tab.


## Demo
Watch the demo here:

[▶ Watch Demo Video](https://github.com/user-attachments/assets/SQL-RAG app.mp4)
(SQL-RAG app.mp4)

