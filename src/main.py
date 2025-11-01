from __future__ import annotations
import streamlit as st

from src.codebase_tokenizer import render_chat_with_your_codebase, render_code_graph, render_codebase_tokenizer
from src.lib.streamlit_helper import application_side_bar, apply_custom_style, chat_interface, init_session_state


def main() -> None:
    """Main function to run the Streamlit app."""

    st.set_page_config(page_title="OpenAI Chat", page_icon=":robot:", layout="wide", initial_sidebar_state="collapsed")

    apply_custom_style()
    init_session_state()
    application_side_bar()

    _chat_interface, work_in_progress = st.tabs(["Study Assistant", "Work in Progress"])

    with _chat_interface:
        chat_interface()

    with work_in_progress:
        tokenizer_tab, graph_tab, codebase_chat_tab = st.tabs(["Codebase Tokenizer", "Code Graph", "Chat With Your Codebase"])
        with tokenizer_tab:
            render_codebase_tokenizer()
        with graph_tab:
            render_code_graph()
        with codebase_chat_tab:
            render_chat_with_your_codebase()


if __name__ == "__main__":
    main()
