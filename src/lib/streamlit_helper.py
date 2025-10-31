"""Streamlit helper functions."""

import os
from pathlib import Path

import streamlit as st

from src.lib.prompts import SYS_CONCEPTUAL_OVERVIEW, SYS_EMPTY_PROMPT, SYS_LEARNING_MATERIAL, SYS_PROFESSOR_EXPLAINS, SYS_SHORT_ANSWER
from src.openai_client import MODELS_GEMINI, MODELS_OPENAI, LLMClient

AVAILABLE_MODELS = []

if os.getenv("OPENAI_API_KEY") is not None:
    AVAILABLE_MODELS += MODELS_OPENAI

if os.getenv("GEMINI_API_KEY") is not None:
    AVAILABLE_MODELS += MODELS_GEMINI

AVAILABLE_PROMPTS = {
    "Short Answer": SYS_SHORT_ANSWER,
    "High-Level Concept": SYS_CONCEPTUAL_OVERVIEW,
    "In-Depth Concept": SYS_PROFESSOR_EXPLAINS,
    "Create Wiki Article": SYS_LEARNING_MATERIAL,
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
        st.session_state.selected_prompt = "Create Learning Material"
        st.session_state.selected_model = "gpt-4.1-mini"
        st.session_state.client = OpenAIBaseClient(st.session_state.selected_model)
        st.session_state.client.set_system_prompt(SYS_LEARNING_MATERIAL)
        st.session_state.rag_database_repo = ""


def application_side_bar() -> None:
    model = st.sidebar.selectbox(
        "Model",
        ["gpt-5", "gpt-4.1", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        key="model_select",
        help="Select Model",
    )

    sys_prompt_name = st.sidebar.selectbox(
        "System prompt",
        list(st.session_state.system_prompts.keys()),
        key="prompt_select",
        help="Select System Prompt",
    )

    if sys_prompt_name != st.session_state.selected_prompt:
        st.session_state.client.set_system_prompt(st.session_state.system_prompts[sys_prompt_name])
        st.session_state.selected_prompt = sys_prompt_name

    if model != st.session_state.selected_model:
        st.session_state.selected_model = model

    def _find_git_repos(base: Path) -> list[Path]:
        """Return directories in *base* that contain a .git folder."""
        return [p for p in base.iterdir() if (p / ".git").exists() and p.is_dir()]

    repos = _find_git_repos(Path.home())
    if repos:
        repo = st.sidebar.selectbox(
            "Repository",
            repos,
            format_func=lambda p: p.name,
            index=None,
            placeholder="Select a repository",
        )
        if repo is not None:
            selected = str(repo)
            if st.session_state.get("selected_repo") != selected:
                st.session_state.selected_repo = selected
    else:
        st.sidebar.info("No Git repositories found")


def render_messages(message_container) -> None:  # noqa
    """Render chat messages from session state."""

    message_container.empty()  # Clear previous messages

    messages = st.session_state.client.messages[1:][::-1]

    with message_container:
        for i in range(0, len(messages), 2):
            is_expanded = i == 0
            label = f"QA-Pair  {i // 2}: "
            user_msg = messages[i + 1]["content"][0]["text"]
            assistant_msg = messages[i]["content"][0]["text"]

            with st.expander(label + user_msg, expanded=is_expanded):
                # Display user and assistant messages
                st.chat_message("user").markdown(user_msg)
                st.chat_message("assistant").markdown(assistant_msg)
