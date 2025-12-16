import streamlit as st

from src.lib.streamlit_helper import (
    apply_custom_css,
    init_session_state,
)

PAGES = {
    "Chatbots": [
        st.Page("pages/Gigachad_Bot.py"),
        st.Page("pages/DataFrame_Bot.py")
    ],
    "Agents": [
        st.Page("pages/Code_Agent.py"),
    ],
    "Workspaces": [
        st.Page("pages/RAG_Workspace.py"),
        st.Page("pages/PDF_Workspace.py"),
        st.Page("pages/OCR_Workspace.py"),
        st.Page("pages/Code_Workspace.py"),
    ],
    "RAG Data Miner": [
        st.Page("pages/PDF_Preprocessor.py"),
        st.Page("pages/VLM_Markdown_Miner.py"),
        st.Page("pages/Markdown_Processor.py")
    ],
}

if __name__ == "__main__":
    init_session_state()
    apply_custom_css()
    st.set_page_config(page_title="Gigachad-Bot", page_icon=":robot:", layout="wide")

    pages = st.navigation(pages=PAGES, position="top")
    pages.run()
