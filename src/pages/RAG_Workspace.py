import os
from pathlib import Path
from typing import Tuple

from rag_database.dataclasses import RAGIngestionPayload, RAGQuery
from rag_database.rag_config import MODEL_CONFIG, DatabaseKeys
from rag_database.rag_database import RagDatabase
import streamlit as st

from config import (
    DIRECTORY_EMBEDDINGS,
    DIRECTORY_OBSIDIAN_DOCS,
    DIRECTORY_OBSIDIAN_VAULT,
    DIRECTORY_RAG_INPUT,
)
from lib.streamlit_helper import nyan_cat_spinner
from llm_config import DEFAULT_EMBEDDING_MODEL

DATABASE_LABEL_OBSIDIAN = "obsidian"

def init_rag_workspace() -> None:
    """Initialize RAG workspace session state variables."""
    if "rag_databases" not in st.session_state:
        st.session_state.rag_databases = {}

def load_rag_database(rag_db: RagDatabase, payload: RAGIngestionPayload,) -> RagDatabase:
    """Generate a RAG database from a payload parquet file or RAGIngestionPayload."""
    # task type optimizes gemma/gemini embeddings
    # by litellm default ignored for all other models
    rag_db.add_documents(payload=payload,task_type="RETRIEVAL_DOCUMENT",)
    return rag_db

def load_parquet_data(payload_path: Path, embedding_path: Path, selection: str, model: str
    ) -> Tuple[RagDatabase, RAGIngestionPayload]:
    """
    Load RAG Database and RAGIngestionPayload from parquet files if they exist.
    Intentionally kept verbose instead of factorized to improve clarity.

    Logic separated from dataloads to allow API calls without .parquet files.

    1. If both payload and embeddings exist, load both.
    2. If only embeddings exist, load embeddings and create empty payload.
    3. If only payload exists, create empty database and load payload.
    4. If neither exist, return None and show error.
    """
    if payload_path.exists() and embedding_path.exists():
        # Load database & payload from parquet
        rag_db = RagDatabase.from_parquet(embedding_path, model=model)
        payload = RAGIngestionPayload.from_parquet(payload_path)
    elif embedding_path.exists():
        # Load database from parquet
        rag_db = RagDatabase.from_parquet(embedding_path, model=model)
        payload = RAGIngestionPayload.create_empty_payload()
    elif payload_path.exists():
        # Initialize empty database & load payload from parquet
        rag_db = RagDatabase(model=model, embedding_dimensions=MODEL_CONFIG[model]["dimensions"])
        payload = RAGIngestionPayload.from_parquet(payload_path)
    else:
        st.error(f"No payload or embeddings found for selection '{selection}'. Cannot initialize RAG Database.")
        return

    return rag_db, payload

def obsidian_dataloader(model: str) -> Tuple[RagDatabase, RAGIngestionPayload]:
    doc_path = Path(f"{DIRECTORY_OBSIDIAN_VAULT}/{DIRECTORY_OBSIDIAN_DOCS}/")
    embedding_path = Path(f"{DIRECTORY_EMBEDDINGS}/{DATABASE_LABEL_OBSIDIAN}_{model}_embeddings.parquet")
    titles = []
    texts = []
    documents = [f for f in os.listdir(doc_path) if f.endswith('.md')]

    if not documents:
        st.warning(f"No markdown documents found in {doc_path} for RAG ingestion.")
        raise ValueError("No documents found for RAG ingestion.")

    if not os.path.exists(embedding_path):
        rag_db = RagDatabase(model=model, embedding_dimensions=MODEL_CONFIG[model]["dimensions"])
    else:
        rag_db = RagDatabase.from_parquet(embedding_path, model=model)

    for doc in documents:
        with open(f"{doc_path}/{doc}", "r") as f:
            text = f.read()
            if not rag_db.is_document_in_database(doc):
                texts.append(text)
                titles.append(doc)

    metadata = []
    for i in range(len(titles)):
        meta = {"source": titles[i], "length": len(texts[i])}
        metadata.append(meta)

        payload = RAGIngestionPayload.from_lists(titles=titles, texts=texts, metadata=metadata)
        rag_db.add_documents(payload=payload, task_type="RETRIEVAL_DOCUMENT")

def rag_sidebar() -> None:
    """RAG Workspace Sidebar for RAG Database selection & initialization."""

    with st.sidebar, st.expander("RAG Workspace Options", expanded=False):

        st.session_state.selected_embedding_model = st.selectbox(
            "Select Embedding Model",
            options=list(MODEL_CONFIG.keys()),
            index=list(MODEL_CONFIG.keys()).index(DEFAULT_EMBEDDING_MODEL),
        )

        available_database_payloads = os.listdir(DIRECTORY_RAG_INPUT)
        available_database_embeddings = os.listdir(DIRECTORY_EMBEDDINGS)

        # for all available database embeddings check for all embedding models if they are contained as string & try to trunctate _<embedding_model>.parquet to receive the label # noqa
        unique_database_labels = set()
        for embedding_file in available_database_embeddings:
            for model_name in MODEL_CONFIG:
                suffix = f"_{model_name}.parquet"
                if embedding_file.endswith(suffix):
                    unique_database_labels.add(embedding_file.removesuffix(suffix))
                    break

        available_database_embeddings = sorted(unique_database_labels)
        options = set(available_database_payloads + available_database_embeddings)

        st.session_state.selected_rag_database = st.selectbox(
            "Select RAG Database",
            options=options,
            index=0,
        )
        if st.button("Initialize RAG Database", key="load_rag_db"):
            with nyan_cat_spinner():
                selection = st.session_state.selected_rag_database
                model = st.session_state.selected_embedding_model

                if selection == DATABASE_LABEL_OBSIDIAN:
                    rag_db, payload = obsidian_dataloader(model=model)
                else:
                    payload_path = Path(f"{DIRECTORY_RAG_INPUT}/{selection}/{selection}_ingestion_payload.parquet")
                    embedding_path = Path(f"{DIRECTORY_EMBEDDINGS}/{selection}_{model}.parquet")
                    rag_db, payload = load_parquet_data(
                        payload_path=payload_path,
                        embedding_path=embedding_path,
                        selection=selection,
                        model=model)

                rag_db = load_rag_database(rag_db=rag_db, payload=payload)

            # Create nested dictionary structure to allow different embeddings for the same documents - will be used for benchmarking
            st.session_state.rag_databases.setdefault(selection, {})
            st.session_state.rag_databases[selection][model] = rag_db

        with st.expander("RAG Databases in Memory", expanded=True):

            for label, models_dict in st.session_state.rag_databases.items():
                with st.expander(f"**Label**:{label}", expanded=True):
                    for model, rag_db in models_dict.items():
                        with st.expander(f"**Model**:{model}", expanded=False):
                            st.dataframe(rag_db.vector_db.database)
                            if st.button("Store Database", key=f"store_rag_db_{label}_{model}"):
                                parquet_embeddings = f"{DIRECTORY_EMBEDDINGS}/{label}_{model}.parquet"
                                rag_db.vector_db.database.write_parquet(parquet_embeddings) # noqa
                                st.success(f"Stored RAG Database '{label}' to {parquet_embeddings}")

        st.session_state.k_query_documents = st.slider(
            "Number of documents to retrieve per query", min_value=1, max_value=20, value=5, step=1, key="k_docs",
        )

def rag_workspace() -> None:
    """RAG Workspace main function."""
    st.title("RAG Workspace")
    with st._bottom:
        prompt = st.chat_input("Send a message", key="chat_input")

    if prompt:
        rag_db: RagDatabase = st.session_state.rag_databases[st.session_state.selected_rag_database][st.session_state.selected_embedding_model] # noqa
        query = RAGQuery(query=prompt, k_documents=st.session_state.k_query_documents)
        rag_response = rag_db.rag_process_query(rag_query=query)
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            documents = rag_response.to_polars()
            for doc in documents.iter_rows(named=True):
                with st.expander(f"**Similarity**: {doc[DatabaseKeys.KEY_SIMILARITIES]:.2f}   -  **Title**: {doc[DatabaseKeys.KEY_TITLE]}"): # noqa
                    st.markdown(doc[DatabaseKeys.KEY_TXT_RETRIEVAL])

if __name__ == "__main__":
    st.set_page_config(page_title="RAG Workspace", page_icon=":robot:", layout="wide")
    init_rag_workspace()
    rag_sidebar()
    rag_workspace()
