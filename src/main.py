from __future__ import annotations
import streamlit as st

from src.codebase_tokenizer import render_chat_with_your_codebase, render_code_graph, render_codebase_tokenizer
from src.lib.streamlit_helper import application_side_bar, apply_custom_style, init_session_state, render_messages


def _chat_interface() -> None:
    col_left, col_right = st.columns([0.382, 0.618])  # Golden ratio

    with col_left:
        with st.expander("Options", expanded=False):
            if st.button("Reset History", key="reset_history_main"):
                st.session_state.client.reset_history()
            with st.expander("Store answer", expanded=True):
                try:
                    idx_input = st.text_input("Index of message to save", key="index_input_main")
                    idx = int(idx_input) if idx_input.strip() else 0
                except ValueError:
                    st.error("Please enter a valid integer")
                    idx = 0
                filename = st.text_input("Filename", key="filename_input_main")
                if st.button("Save to Markdown", key="save_to_md_main"):
                    st.session_state.client.write_to_md(filename, idx)
                    st.success(f"Chat history saved to {filename}")

        prompt = st.text_area("Send a message", key="left_chat_input_main", height=200)
        send_btn = st.button("Send", key="send_btn_main")
        st.markdown("---")

    with col_right:
        st.subheader("Chat Interface")
        st.markdown("---")
        st.write("")  # Spacer
        message_container = st.container()
        render_messages(message_container)

        if send_btn and prompt:
            with st.chat_message("assistant"):
                st.session_state.client.chat(model=st.session_state.selected_model, user_message=prompt)
                st.rerun()


def main() -> None:
    """Main function to run the Streamlit app."""

    st.set_page_config(page_title="OpenAI Chat", page_icon=":robot:", layout="wide", initial_sidebar_state="collapsed")

    apply_custom_style()
    init_session_state()
    application_side_bar()

    chat_interface, work_in_progress = st.tabs(["Study Assistant", "Work in Progress"])

    with chat_interface:
        _chat_interface()

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
