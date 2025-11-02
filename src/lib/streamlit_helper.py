"""Streamlit helper functions."""

import io
import os
import tempfile

import pymupdf4llm
import streamlit as st

from src.lib.prompts import (
    SYS_ARTICLE,
    SYS_CONCEPT_IN_DEPTH,
    SYS_CONCEPTUAL_OVERVIEW,
    SYS_EMPTY_PROMPT,
    SYS_PDF_TO_LEARNING_GOALS,
    SYS_PRECISE_TASK_EXECUTION,
    SYS_PROMPT_ARCHITECT,
    SYS_SHORT_ANSWER,
)
from src.llm_client import MODELS_GEMINI, MODELS_OPENAI, LLMClient

AVAILABLE_MODELS = []

if os.getenv("OPENAI_API_KEY") is not None:
    AVAILABLE_MODELS += MODELS_OPENAI

if os.getenv("GEMINI_API_KEY") is not None:
    AVAILABLE_MODELS += MODELS_GEMINI

AVAILABLE_PROMPTS = {
    "Short Answer": SYS_SHORT_ANSWER,
    "Concept - High-Level": SYS_CONCEPTUAL_OVERVIEW,
    "Concept - In-Depth": SYS_CONCEPT_IN_DEPTH,
    "Concept - Article": SYS_ARTICLE,
    "Prompt Architect": SYS_PROMPT_ARCHITECT,
    "Precise Task Execution": SYS_PRECISE_TASK_EXECUTION,
    "PDF to Learning Goals": SYS_PDF_TO_LEARNING_GOALS,
    "<empty prompt>": SYS_EMPTY_PROMPT,
}


def init_session_state() -> None:
    if "client" not in st.session_state:
        st.session_state.file_context = ""
        st.session_state.system_prompts = AVAILABLE_PROMPTS
        st.session_state.selected_prompt = "<empty prompt>"
        st.session_state.selected_model = AVAILABLE_MODELS[0]
        st.session_state.client = LLMClient()
        st.session_state.client._set_system_prompt(AVAILABLE_PROMPTS["Short Answer"])
        st.session_state.rag_database_repo = ""


@st.cache_resource
def _extract_text_from_pdf(file: io.BytesIO) -> str:
    """Extract text from uploaded PDF file using pymupdf4llm."""
    # Create temporary file - pymupdf4llm requires a file path but Streamlit's doesnt support that directly
    with tempfile.TemporaryDirectory(delete=True) as tmpdir:

        # Preserve filename to allow correct naming of images extracted from PDFs (future proof)
        temp_file_path = os.path.join(tmpdir, file.name)
        with open(temp_file_path, "wb") as f:
            f.write(file.getvalue())
            text = pymupdf4llm.to_markdown(doc=f, write_images=True)

    return text


def application_side_bar() -> None:
    model = st.sidebar.selectbox(
        "Model",
        AVAILABLE_MODELS,
        key="model_select",
    )

    sys_prompt_name = st.sidebar.selectbox(
        "System prompt",
        list(st.session_state.system_prompts.keys()),
        key="prompt_select",
    )

    with st.sidebar:
        st.markdown("---")
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

            file = st.file_uploader(type=["pdf", "py", "md", "cpp", "txt"], label="fileloader_sidbar")
            if file is not None:
                text = _extract_text_from_pdf(file)
                st.session_state.file_context = text

    if sys_prompt_name != st.session_state.selected_prompt:
        st.session_state.client._set_system_prompt(st.session_state.system_prompts[sys_prompt_name])
        st.session_state.selected_prompt = sys_prompt_name

    if model != st.session_state.selected_model:
        st.session_state.selected_model = model

def chat_interface() -> None:
    _, col_center, _ = st.columns([0.025, 0.95, 0.025])

    with col_center:
        st.header("Learning Assistant")
        st.markdown("---")
        st.write("")  # Spacer
        message_container = st.container()
        render_messages(message_container)

        with st._bottom:
            prompt = st.chat_input("Send a message")

        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                prompt += st.session_state.file_context
                st.write_stream(st.session_state.client.chat(model=st.session_state.selected_model, user_message=prompt))
                st.rerun()

@st.cache_data
def extract_learning_goals(text: str) -> str:
    stream = st.session_state.client.api_query(
        model="gemini-2.5-flash-lite",
        user_message=text,
        system_prompt=SYS_PDF_TO_LEARNING_GOALS,
        chat_history=None
    )
    pdf_learning_goals = ""
    for chunk in stream:
        pdf_learning_goals += chunk

    return pdf_learning_goals


def pdf_workspace() -> None:
    """PDF Workspace for extracting learning goals and summary articles."""

    header, pdf_options = st.columns([0.8, 0.2])
    with header:
        st.header("PDF Workspace")

    with pdf_options, st.expander("Options", expanded=False):
        file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_workspace_uploader")

        if file is not None:
            pdf_text = _extract_text_from_pdf(file)
            learning_goals = extract_learning_goals(pdf_text)

    col_summary_article, col_learning_goals = st.columns([0.618, 0.382])
    with col_summary_article:
        st.subheader("Summary Article")
        st.info("To be implemented...")
    with col_learning_goals:
        st.subheader("Learning Goals")
        st.markdown(learning_goals if file is not None else "")
        st.markdown("---")
        st.subheader("Original PDF")
        st.pdf(file) if file is not None else None


def render_messages(message_container) -> None:  # noqa
    """Render chat messages from session state."""

    message_container.empty()  # Clear previous messages

    messages = st.session_state.client.messages

    if len(messages) == 0:
        return

    with message_container:
        for i in range(0, len(messages), 2):
            is_expanded = i == len(messages) - 2
            label = f"QA-Pair  {i // 2}: "
            _, user_msg = messages[i]
            _, assistant_msg = messages[i + 1]

            with st.expander(label=label, expanded=is_expanded):
                # Display user and assistant messages
                st.chat_message("user").markdown(user_msg)
                st.chat_message("assistant").markdown(assistant_msg)
