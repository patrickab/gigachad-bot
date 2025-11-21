import os

import polars as pl
from rag_database.rag_config import EMPTY_RAG_SCHEMA, MODEL_CONFIG, DatabaseKeys
from rag_database.rag_database import RagDatabase, RAGQuery
import streamlit as st

from src.config import DEFAULT_EMBEDDING_MODEL, DIRECTORY_EMBEDDINGS, DIRECTORY_OBSIDIAN_DOCS, DIRECTORY_OBSIDIAN_VAULT, RAG_K_DOCS


def rag_sidebar() -> None:
    with st.sidebar:
        st.session_state.selected_embedding_model = st.selectbox(
            "Select Embedding Model",
            options=list(MODEL_CONFIG.keys()),
            index=list(MODEL_CONFIG.keys()).index(DEFAULT_EMBEDDING_MODEL),
        )

@st.cache_resource
def load_rag_database(doc_path: str, model: str, label: str) -> RagDatabase:
    """
    Initialize RAG Database with .md documents.
    Loads existing embeddings if available.
    Embed new documents & update database accordingly.
    """

    parquet_embeddings = f"{DIRECTORY_EMBEDDINGS}/{label}_{model}.parquet"
    if os.path.exists(parquet_embeddings): # noqa
        rag_dataframe = pl.read_parquet(parquet_embeddings)
    else:
        rag_dataframe = EMPTY_RAG_SCHEMA

    rag_db = RagDatabase(model=model, database=rag_dataframe)
    titles = []
    texts = []
    documents = [f for f in os.listdir(doc_path) if f.endswith('.md')]

    for doc in documents:
        with open(f"{doc_path}/{doc}", "r") as f:
            text = f.read()
            if not rag_db.is_document_in_database(doc):
                texts.append(text)
                titles.append(doc)

    rag_db.add_documents(titles=titles, texts=texts)
    return rag_db

def rag_workspace_obsidian() -> None:
    """RAG Workspace for retrieval-augmented generation on specifiable obsidian vault."""
    if st.button("Initialize RAG Database", key="load_rag_db"):
        rag_database = load_rag_database(
            doc_path=f"{DIRECTORY_OBSIDIAN_VAULT}/{DIRECTORY_OBSIDIAN_DOCS}/",
            model=st.session_state.selected_embedding_model,
            label="rag_db_obsidian"
        )

        with st._bottom:
            rag_query = st.chat_input("Send a message", key="rag_input")

        if rag_query:

            rag_query = RAGQuery(query=rag_query, k_documents=RAG_K_DOCS)
            rag_response = rag_database.rag_process_query(rag_query)

            with st.chat_message("user"):
                st.markdown(rag_query.query)

            with st.chat_message("assistant"):
                documents = rag_response.to_polars()
                for doc in documents.iter_rows(named=True):
                    with st.expander(f"**Similarity**: {doc[DatabaseKeys.KEY_SIMILARITIES]:.2f}   -  **Title**: {doc[DatabaseKeys.KEY_TITLE]}"): # noqa
                        st.markdown(doc[DatabaseKeys.KEY_TXT])

if __name__ == "__main__":

    st.set_page_config(page_title="RAG Workspace", page_icon=":robot:", layout="wide")
    rag_sidebar()
    rag_workspace_obsidian()
