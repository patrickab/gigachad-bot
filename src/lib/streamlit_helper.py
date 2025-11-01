"""Streamlit helper functions."""

import os

import streamlit as st

from src.lib.prompts import (
    SYS_ARTICLE,
    SYS_CONCEPT_IN_DEPTH,
    SYS_CONCEPTUAL_OVERVIEW,
    SYS_EMPTY_PROMPT,
    SYS_SHORT_ANSWER,
)
from src.openai_client import MODELS_GEMINI, MODELS_OPENAI, LLMClient

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
    "<empty prompt>": SYS_EMPTY_PROMPT,
}


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>

        /* General text inherits Cascadia Code */
        p, div, span, h1, h2, h3, h4, h5, h6 {
            font-family: 'Cascadia Code', 'Georgia', 'Times New Roman', serif;
        }

        /* Code, pre, and LaTeX math uses Roboto Mono with default coloring */
        code, pre, .math {
            font-family: 'Roboto Mono', monospace;
            padding: 4px 6px;
            border-radius: 6px;
            line-height: 1.4;
            white-space: pre-wrap;
            word-break: break-word;
            user-select: text;
        }

        /* Increase default page padding */
        .block-container {
            padding-left: 3rem;
            padding-right: 3rem;
        }

        /* Card styling for graph area */
        div[data-testid="stAgraph"], .graph-card {
            background: #1a1a1a;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            position: relative;
        }

        /* Typography hierarchy */
        .graph-title {
            font-size: 32px;
            margin-bottom: 16px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    if "client" not in st.session_state:
        st.session_state.system_prompts = AVAILABLE_PROMPTS
        st.session_state.selected_prompt = "<empty prompt>"
        st.session_state.selected_model = AVAILABLE_MODELS[0]
        st.session_state.client = LLMClient()
        st.session_state.client._set_system_prompt(AVAILABLE_PROMPTS["Short Answer"])
        st.session_state.rag_database_repo = ""


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

    if sys_prompt_name != st.session_state.selected_prompt:
        st.session_state.client._set_system_prompt(st.session_state.system_prompts[sys_prompt_name])
        st.session_state.selected_prompt = sys_prompt_name

    if model != st.session_state.selected_model:
        st.session_state.selected_model = model


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
