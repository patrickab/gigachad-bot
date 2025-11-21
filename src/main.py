import streamlit as st

from src.lib.streamlit_helper import (
    application_side_bar,
    apply_custom_css,
    chat_interface,
    init_session_state,
    pdf_workspace,
    rag_workspace,
)


def main() -> None:
    """Main function to run the Streamlit app."""

    st.set_page_config(page_title="Gigachad-Bot", page_icon=":robot:", layout="wide", initial_sidebar_state="collapsed")

    apply_custom_css()
    init_session_state()
    application_side_bar()

    tab_chat, tab_rag, tab_pdf = st.tabs(["Gigachad-Bot", "RAG Workspace", "PDF Workspace"])

    with tab_chat:
        chat_interface()
    with tab_rag:
        rag_workspace()
    with tab_pdf:
        pdf_workspace()

if __name__ == "__main__":
    main()
