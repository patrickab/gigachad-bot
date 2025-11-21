import os

from rag_database.rag_config import MODEL_CONFIG, DatabaseKeys
from rag_database.rag_database import RagDatabase, RAGQuery
import streamlit as st

from src.config import OBSIDIAN_RAG, OBSIDIAN_VAULT, RAG_K_DOCS


@st.cache_resource
def load_rag_database(doc_path: str) -> RagDatabase:
    """Initialize RAG Database with .md documents."""
    rag_db = RagDatabase()
    titles = []
    texts = []
    documents = [f for f in os.listdir(doc_path) if f.endswith('.md')]

    for doc in documents:
        with open(f"{doc_path}/{doc}", "r") as f:
            text = f.read()
            texts.append(text)
            titles.append(doc)

    rag_db.add_documents(titles=titles, texts=texts)
    return rag_db

def rag_workspace_obsidian() -> None:
    """RAG Workspace for retrieval-augmented generation on specifiable obsidian vault."""
    rag_database = load_rag_database(f"{OBSIDIAN_VAULT}/{OBSIDIAN_RAG}/")

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
    rag_workspace_obsidian()
