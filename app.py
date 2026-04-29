from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

from rag.config import get_settings
from rag.db_connector import SQLConnector
from rag.llm import GroqService
from rag.rag_service import RAGService
from rag.vector_store import SQLiteVectorStore


st.set_page_config(page_title="RAG SQL Chat", page_icon="🧠", layout="wide")
st.markdown(
  """
   <style>
    :root {
      --bg: #000814;
      --panel: rgba(0, 29, 61, 0.75);
      --panel-2: #002855;
      --text: #ffffff;
      --muted: #b7c6d9;
      --green: #00c853;
      --border: #1f3b5b;
      --focus-white: #ffffff;
      --radius: 12px;
    }

    .stApp {
    background:
        linear-gradient(rgba(0, 8, 20, 0.85),
        rgba(0, 8, 20, 0.90)),
        url("https://plus.unsplash.com/premium_photo-1683880731785-e5b632e0ea13?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTN8fG9mZmljZSUyMHNwYWNlfGVufDB8fDB8fHww");    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: var(--text);
    }
    section[data-testid="stSidebar"] {
    background: rgba(0, 29, 61, 0.75);
    backdrop-filter: blur(10px);
    border-right: 1px solid var(--border);
    }
    h1, h2, h3, p, label, span { color: var(--text) ; }

    /* Inputs default */
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    .stSelectbox [data-baseweb="select"] > div {
      background: var(--panel-2) ;
      color: var(--text) ;
      border: 1px solid var(--border) ;
      border-radius: var(--radius) ;
      box-shadow: none ;
      outline: none ;
    }

    /* Focus state: single WHITE border only */
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stNumberInput input:focus,
    .stSelectbox [data-baseweb="select"] > div:focus-within,
    [data-baseweb="input"] input:focus,
    [data-baseweb="base-input"] input:focus {
      border: 1px solid var(--focus-white) ;
      box-shadow: none ;
      outline: none ;
      -webkit-appearance: none ;
      appearance: none ;
    }

    /* Remove extra wrapper rings from BaseWeb */
    [data-baseweb="input"],
    [data-baseweb="base-input"] {
      box-shadow: none ;
      outline: none ;
      border: none ;
    }

    .stButton button {
      background: var(--green) ;
      color: #001208 ;
      border: 0 ;
      border-radius: 10px ;
      font-weight: 700 ;
    }

    /* Let Streamlit theme control table colors (fixes light mode issues) */
    .stDataFrame, .stTable {
      background: transparent ;
      border: 1px solid rgba(127,127,127,0.25) ;
      border-radius: var(--radius) ;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.image("logo.png", width="content") 
# width="content"
st.title("🧠 SQL-to-Vector RAG")
st.caption("Connect databases from sidebar, ingest SQL results into vector DB, and chat over retrieved context.")


def get_rag_service() -> RAGService:
    settings = get_settings()
    if not settings.groq_api_key:
        st.error("Missing GROQ_API_KEY in .env")
        st.stop()
    llm = GroqService(
        api_key=settings.groq_api_key,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        chat_model=settings.groq_model,
    )
    vector_store = SQLiteVectorStore(settings.vector_db_path)
    return RAGService(vector_store=vector_store, llm_service=llm)


def build_connection_url(db_type: str, cfg: dict) -> str:
    if db_type == "sqlite":
        return f"sqlite:///{cfg.get('sqlite_path', 'local.db')}"

    user = quote_plus(cfg.get("username", ""))
    password = quote_plus(cfg.get("password", ""))

    if db_type == "mysql":
        return f"mysql+pymysql://{user}:{password}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    if db_type == "postgresql":
        return f"postgresql+psycopg2://{user}:{password}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    if db_type == "mssql":
        driver = quote_plus(cfg.get("driver", "ODBC Driver 17 for SQL Server"))
        return (
            f"mssql+pyodbc://{user}:{password}@{cfg['host']}:{cfg['port']}/"
            f"{cfg['database']}?driver={driver}"
        )
    if db_type == "snowflake":
        account = cfg["account"]
        database = cfg["database"]
        schema = cfg.get("schema", "")
        warehouse = quote_plus(cfg.get("warehouse", ""))
        role = quote_plus(cfg.get("role", "SYSADMIN"))
        return (
            f"snowflake://{user}:{password}@{account}/{database}/{schema}"
            f"?warehouse={warehouse}&role={role}"
        )
    raise ValueError(f"Unsupported db_type: {db_type}")


if "rag_service" not in st.session_state:
    st.session_state.rag_service = get_rag_service()
if "connector" not in st.session_state:
    st.session_state.connector = None
if "last_df" not in st.session_state:
    st.session_state.last_df = pd.DataFrame()
if "db_type" not in st.session_state:
    st.session_state.db_type = "mysql"

rag = st.session_state.rag_service

with st.sidebar:
    st.subheader("🗄️ Connection Setup")
    db_type = st.selectbox("Database Type", ["mysql", "mssql", "snowflake", "postgresql", "sqlite"], key="db_type")
    cfg: dict[str, str] = {}

    if db_type in ["mysql", "mssql", "postgresql"]:
        cfg["host"] = st.text_input("Host", value="localhost")
        cfg["port"] = st.text_input("Port", value={"mysql": "3306", "mssql": "1433", "postgresql": "5432"}[db_type])
        cfg["username"] = st.text_input("Username")
        cfg["password"] = st.text_input("Password", type="password")
        cfg["database"] = st.text_input("Database Name")
        if db_type == "mssql":
            cfg["driver"] = st.text_input("ODBC Driver", value="ODBC Driver 17 for SQL Server")
    elif db_type == "snowflake":
        cfg["account"] = st.text_input("Account")
        cfg["username"] = st.text_input("Username")
        cfg["password"] = st.text_input("Password", type="password")
        cfg["database"] = st.text_input("Database Name")
        cfg["schema"] = st.text_input("Schema (optional)")
        cfg["warehouse"] = st.text_input("Warehouse")
        cfg["role"] = st.text_input("Role (optional)", value="SYSADMIN")
    else:
        cfg["sqlite_path"] = st.text_input("SQLite File Path", value="local.db")

    if st.button("Connect", use_container_width=True):
        try:
            connection_url = build_connection_url(db_type, cfg)
            connector = SQLConnector(connection_url)
            connector.connect()
            st.session_state.connector = connector
            st.success(f"Connected - {db_type.upper()}")
        except Exception as exc:
            st.error(f"Connection failed: {exc}")

    if st.button("Disconnect", use_container_width=True):
        st.session_state.connector = None
        st.info("Disconnected.")

    if st.session_state.connector is not None:
        st.caption("Available tables")
        for table_name in st.session_state.connector.list_tables()[:25]:
            st.write(f"- {table_name}")
    else:
        st.caption("Not connected")

tab_data, tab_chat = st.tabs(["Query + Ingest", "RAG Chat"])

with tab_data:
    st.subheader("Run Query and Ingest")
    connector = st.session_state.connector
    if connector is None:
        st.info("Connect from sidebar first.")
    else:
        query = st.text_area("SQL Query", value="SELECT * FROM your_table LIMIT 100")
        if st.button("Run Query"):
            try:
                df = connector.query_dataframe(query)
                st.session_state.last_df = df
                st.dataframe(df, use_container_width=True)
            except Exception as exc:
                st.error(f"Query failed: {exc}")

        source_name = st.text_input("Source Name for Vector Store", value="db_result_set")
        if st.button("Ingest Query Result to Vector DB"):
            if st.session_state.last_df.empty:
                st.warning("Run a query first to load data.")
            else:
                count = rag.ingest_dataframe(
                    source_name=source_name,
                    source_type=st.session_state.db_type,
                    df=st.session_state.last_df,
                )
                st.success(f"Ingested {count} chunks into vector DB.")

    st.subheader("Stored Sources in Vector DB")
    st.dataframe(pd.DataFrame(rag.vector_store.list_sources()), use_container_width=True)
    if st.button("Clear Vector DB"):
        rag.vector_store.clear()
        st.success("Vector DB cleared.")

with tab_chat:
    st.subheader("Chat Over Retrieved Context")
    question = st.text_input("Ask a question", value="Which city has the highest number of order delivered and why?")
    # top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=20, value=6)
    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            answer, hits = rag.ask(question, top_k=10)
            st.markdown("### Answer")
            st.write(answer)
            # st.markdown("### Retrieved Context")
            # for i, hit in enumerate(hits, start=1):
            #     st.markdown(f"**{i}.** `{hit['source_name']}` [{hit['source_type']}] | score `{hit['score']:.4f}`")
            #     st.write(hit["text"])
