import os
from typing import Optional

import polars as pl
from rag_database.dataclasses import RAGIngestionPayload
from rag_database.rag_config import MODEL_CONFIG, DatabaseKeys
from rag_database.rag_database import RagDatabase, RAGQuery
import streamlit as st

from src.config import DEFAULT_EMBEDDING_MODEL, DIRECTORY_EMBEDDINGS, DIRECTORY_OBSIDIAN_DOCS, DIRECTORY_OBSIDIAN_VAULT, RAG_K_DOCS
from src.lib.streamlit_helper import nyan_cat_spinner

DATABASE_LABAL_OBSIDIAN = "obsidian"

def init_rag_workspace() -> None:
    """Initialize RAG workspace session state variables."""
    if "rag_databases" not in st.session_state:
        st.session_state.rag_databases = {}

def rag_sidebar() -> None:
    with st.sidebar:
        st.session_state.selected_embedding_model = st.selectbox(
            "Select Embedding Model",
            options=list(MODEL_CONFIG.keys()),
            index=list(MODEL_CONFIG.keys()).index(DEFAULT_EMBEDDING_MODEL),
        )

        if st.button("Initialize RAG Database", key="load_rag_db"):

            with nyan_cat_spinner():
                database = load_rag_database(
                    doc_path=f"{DIRECTORY_OBSIDIAN_VAULT}/{DIRECTORY_OBSIDIAN_DOCS}/",
                    model=st.session_state.selected_embedding_model,
                    label=DATABASE_LABAL_OBSIDIAN
                )
            st.session_state.rag_databases.setdefault(DATABASE_LABAL_OBSIDIAN, {})[st.session_state.selected_embedding_model] = database # noqa

        st.markdown("---")
        st.markdown("## RAG Databases in Memory")

        for label, models_dict in st.session_state.rag_databases.items():
            with st.expander(f"**Label**:{label}", expanded=False):
                for model, rag_db in models_dict.items():
                    with st.expander(f"**Model**:{model}", expanded=False), st.expander("Inspect Database", expanded=False):
                        st.dataframe(rag_db.vector_db.database)
                        if st.button("Store Database", key=f"store_rag_db_{label}_{model}"):
                            parquet_embeddings = f"{DIRECTORY_EMBEDDINGS}/{label}_{model}.parquet"
                            rag_db.vector_db.database.write_parquet(parquet_embeddings)
                            st.success(f"Stored RAG Database '{label}' to {parquet_embeddings}")

        st.markdown("---")

@st.cache_resource
def load_rag_database(doc_path: str, model: str, label: str, embedding_dimensions: Optional[int]=None) -> RagDatabase:
    """
    Initialize RAG Database with .md documents.
    Loads existing embeddings if available.
    Embed new documents & update database accordingly.
    """
    if embedding_dimensions is None:
        embedding_dimensions = MODEL_CONFIG[model]["dimensions"]

    parquet_embeddings = f"{DIRECTORY_EMBEDDINGS}/{label}_{model}.parquet"
    if os.path.exists(parquet_embeddings):
        # Load existing RAG database
        rag_dataframe = pl.read_parquet(parquet_embeddings)
        rag_db = RagDatabase(model=model, database=rag_dataframe)
    else:
        # Initialize empty RAG database
        rag_db = RagDatabase(model=model, embedding_dimensions=embedding_dimensions)

    titles = []
    texts = []
    documents = [f for f in os.listdir(doc_path) if f.endswith('.md')]

    for doc in documents:
        with open(f"{doc_path}/{doc}", "r", encoding="utf-8") as f:
            text = f.read()
            if not rag_db.is_document_in_database(doc):
                texts.append(text)
                titles.append(doc)

    if titles:

        metadata_template = [{} for _ in range(len(titles))]
        payload = RAGIngestionPayload.from_lists(titles=titles, texts=texts, metadata=metadata_template)
        kwargs = {}
        if "gemini" in model or "gemma" in model:
            kwargs["task_type"] = "RETRIEVAL_DOCUMENT"

        rag_db.add_documents(payload=payload, **kwargs)
    return rag_db

def rag_workspace_obsidian() -> None:
    """RAG Workspace for retrieval-augmented generation on specifiable obsidian vault."""

    rag_sidebar()

    with st._bottom:
        rag_query = st.chat_input("Send a message", key="rag_input")

    if rag_query:

        rag_database = st.session_state.rag_databases[DATABASE_LABAL_OBSIDIAN][st.session_state.selected_embedding_model]
        rag_query = RAGQuery(query=rag_query, k_documents=RAG_K_DOCS)

        model = st.session_state.selected_embedding_model
        kwargs = {}
        if "gemini" in model or "gemma" in model:
            kwargs["task_type"] = "RETRIEVAL_QUERY"

        rag_response = rag_database.rag_process_query(rag_query, **kwargs)

        with st.chat_message("user"):
            st.markdown(rag_query.query)

        with st.chat_message("assistant"):
            documents = rag_response.to_polars()
            for doc in documents.iter_rows(named=True):
                with st.expander(f"**Similarity**: {doc[DatabaseKeys.KEY_SIMILARITIES]:.2f}   -  **Title**: {doc[DatabaseKeys.KEY_TITLE]}"): # noqa
                    st.markdown(doc[DatabaseKeys.KEY_TXT])

if __name__ == "__main__":
    st.set_page_config(page_title="RAG Workspace", page_icon=":robot:", layout="wide")
    init_rag_workspace()
    rag_workspace_obsidian()