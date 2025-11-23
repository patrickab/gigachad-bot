import streamlit as st

from src.lib.streamlit_helper import (
    apply_custom_css,
    init_session_state,
)

PAGES = {
    "Select Page": [
        st.Page("pages/Gigachad_Bot.py"),
        st.Page("pages/RAG_Workspace.py"),
        st.Page("pages/PDF_Workspace.py"),
        st.Page("pages/OCR_Workspace.py"),
    ],
}

if __name__ == "__main__":
    init_session_state()
    apply_custom_css()
    st.set_page_config(page_title="Gigachad-Bot", page_icon=":robot:", layout="wide")

    pages = st.navigation(pages=PAGES, position="top")
    pages.run()
