"""Streamlit helper functions."""

from datetime import datetime
import io
import json
import os
import re
import tempfile

import fitz
import pandas as pd
import pymupdf4llm
from st_copy import copy_button
import streamlit as st

from src.config import (
    CHAT_HISTORY_FOLDER,
    MACROTASK_MODEL,
    MICROTASK_MODEL,
    MODELS_GEMINI,
    MODELS_OLLAMA,
    MODELS_OPENAI,
    NANOTASK_MODEL,
    OBSIDIAN_VAULT,
)
from src.lib.flashcards import DATE_ADDED, NEXT_APPEARANCE, render_flashcards
from src.lib.non_user_prompts import (
    SYS_IMAGE_IMPORTANCE,
    SYS_LEARNINGGOALS_TO_FLASHCARDS,
    SYS_NOTE_TO_OBSIDIAN_YAML,
    SYS_PDF_TO_ARTICLE,
    SYS_PDF_TO_LEARNING_GOALS,
)
from src.lib.prompts import (
    SYS_AI_TUTOR,
    SYS_ARTICLE,
    SYS_CONCEPT_IN_DEPTH,
    SYS_CONCEPTUAL_OVERVIEW,
    SYS_EMPTY_PROMPT,
    SYS_PRECISE_TASK_EXECUTION,
    SYS_PROMPT_ARCHITECT,
    SYS_QUICK_OVERVIEW,
)
from src.llm_client import LLMClient

AVAILABLE_MODELS = []

if os.getenv("GEMINI_API_KEY"):
    AVAILABLE_MODELS += MODELS_GEMINI

if os.getenv("OPENAI_API_KEY"):
    AVAILABLE_MODELS += MODELS_OPENAI

if MODELS_OLLAMA != []:
    AVAILABLE_MODELS += MODELS_OLLAMA

AVAILABLE_PROMPTS = {
    "Quick Overview": SYS_QUICK_OVERVIEW,
    "AI Tutor": SYS_AI_TUTOR,
    "Concept - High-Level": SYS_CONCEPTUAL_OVERVIEW,
    "Concept - In-Depth": SYS_CONCEPT_IN_DEPTH,
    "Concept - Article": SYS_ARTICLE,
    "Prompt Architect": SYS_PROMPT_ARCHITECT,
    "Precise Task Execution": SYS_PRECISE_TASK_EXECUTION,
    "<empty prompt>": SYS_EMPTY_PROMPT,
}


def init_session_state() -> None:
    if "client" not in st.session_state:
        st.session_state.file_context = ""
        st.session_state.system_prompts = AVAILABLE_PROMPTS
        st.session_state.selected_prompt = "<empty prompt>"
        st.session_state.selected_model = AVAILABLE_MODELS[0]
        st.session_state.client = LLMClient()
        st.session_state.client._set_system_prompt(next(iter(st.session_state.system_prompts.values()))) # set to first prompt
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
            text = pymupdf4llm.to_markdown(doc=f, write_images=False)

        # Get the height of first page
        doc = fitz.open(temp_file_path)
        doc_height = int(doc[0].rect.height * 1.5)  # Scale up for better visibility

    return text, doc_height


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

            file = st.file_uploader(type=["pdf", "py", "md", "cpp", "txt"], label="fileloader_sidbar")
            if file is not None:
                text = _extract_text_from_pdf(file)
                st.session_state.file_context = text

            if st.session_state.client.messages != []:
                st.markdown("---")
                with st.popover("Save History"):
                    filename = st.text_input("Filename", key="history_filename_input")
                    if st.button("Save Chat History", key="save_chat_history_button"):
                        if not os.path.exists(CHAT_HISTORY_FOLDER):
                            os.makedirs(CHAT_HISTORY_FOLDER)
                        st.session_state.client.store_history(CHAT_HISTORY_FOLDER + '/' + filename + '.csv')
                        st.success("Successfully saved chat")

        if os.path.exists(CHAT_HISTORY_FOLDER):
            chat_histories = [f.replace('.csv', '') for f in os.listdir(CHAT_HISTORY_FOLDER) if f.endswith('.csv')]
        else:
            chat_histories = []
        
        if chat_histories != []:
            st.markdown("---")
            with st.expander("Chat Histories", expanded=False):
                for history in chat_histories:
                    with st.expander(history, expanded=False):
                        col_load, col_delete, col_archive = st.columns(3)
                        with col_load:
                            if st.button("⟳", key=f"load_{history}"):
                                st.session_state.client.load_history(os.path.join(CHAT_HISTORY_FOLDER, history + '.csv'))
                        with col_delete:
                            if st.button("🗑", key=f"delete_{history}"):
                                os.remove(os.path.join(CHAT_HISTORY_FOLDER, history + '.csv'))
                                st.rerun()
                        with col_archive:
                            if st.button("⛁", key=f"archive_{history}"):
                                if not os.path.exists(CHAT_HISTORY_FOLDER + '/archived/'):
                                    os.makedirs(CHAT_HISTORY_FOLDER + '/archived/')
                                os.rename(
                                    os.path.join(CHAT_HISTORY_FOLDER, history + '.csv'),
                                    os.path.join(CHAT_HISTORY_FOLDER, 'archived', history + '.csv')
                                )
                                st.rerun()

    if sys_prompt_name != st.session_state.selected_prompt:
        st.session_state.client._set_system_prompt(st.session_state.system_prompts[sys_prompt_name])
        st.session_state.selected_prompt = sys_prompt_name

    if model != st.session_state.selected_model:
        st.session_state.selected_model = model


def chat_interface() -> None:
    col_left, _ = st.columns([0.9, 0.1])

    with col_left:
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


def _non_streaming_api_query(model: str, prompt: str, system_prompt: str) -> str:
    """
    Converts streaming response generator to generic string.
    Required for @st.cache_data compatibility.
    """
    stream = st.session_state.client.api_query(model=model, user_message=prompt, system_prompt=system_prompt, chat_history=None)
    response_text = ""
    for chunk in stream:
        response_text += chunk

    return response_text


@st.cache_data
def _generate_learning_goals(text: str) -> str:
    """Generate learning goals from PDF text."""
    print("Generating learning goals...")
    return _non_streaming_api_query(model=MACROTASK_MODEL, prompt=text, system_prompt=SYS_PDF_TO_LEARNING_GOALS)


@st.cache_data
def _generate_image_importance(pdf_text: str, learning_goals: str) -> str:
    """Generate image importance from PDF text and learning goals."""
    print("Generating image importance...")
    response = _non_streaming_api_query(
        model=MICROTASK_MODEL,
        prompt="## Learning Goals\n" + learning_goals + "\n\n## PDF Content\n" + pdf_text,
        system_prompt=SYS_IMAGE_IMPORTANCE,
    )
    return response

@st.cache_data
def _generate_flashcards(learning_goals: str) -> pd.DataFrame:
    """Generate flashcards from learning goals."""
    print("Generating flashcards...")
    response = _non_streaming_api_query(
        model=MACROTASK_MODEL,
        prompt=learning_goals,
        system_prompt=SYS_LEARNINGGOALS_TO_FLASHCARDS,
    )
    response = response.split("```json")[ -1].split("```")[0]  # Clean up response if necessary
    flashcards = json.loads(response)
    df_flashcards = pd.DataFrame(flashcards)
    df_flashcards[DATE_ADDED] = datetime.now()
    df_flashcards[NEXT_APPEARANCE] = datetime.now()
    return df_flashcards

@st.cache_data
def _generate_wiki_article(pdf_text: str, learning_goals: str) -> str:  # noqa
    print("Writing wiki article...")
    wiki_prompt = f"""
    Consider the following learning goals:
    
    {learning_goals}
    
    Dynamically adjust depth of explanation to the provided Bloom's taxonomy tags.
    Use the hierarchy of learning goals to structure the article.
    Generate a comprehensive study article based on the following PDF content.

    {pdf_text}
    """

    return _non_streaming_api_query(model=MACROTASK_MODEL, prompt=wiki_prompt, system_prompt=SYS_PDF_TO_ARTICLE)


def write_to_md(filename: str, message: str) -> None:
    """Write an assistant response to .md (idx: 0 = most recent)."""
    if not filename.endswith(".md"):
        filename += ".md"

    sys_prompt = SYS_NOTE_TO_OBSIDIAN_YAML.replace("{{file_name_no_ext}}", filename.split(".md")[0])
    yaml_header = _non_streaming_api_query(
        model=NANOTASK_MODEL,
        prompt=message,
        system_prompt=sys_prompt.replace("{{user_notes}}", message),
    )

    file_path = os.path.join(OBSIDIAN_VAULT, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(yaml_header + "\n" + message)

    os.makedirs("markdown", exist_ok=True)
    with open(os.path.join("markdown", filename), "w", encoding="utf-8") as f:
        f.write(yaml_header + "\n" + message)

def pdf_workspace() -> None:
    """PDF Workspace for extracting learning goals and summary articles."""

    tab_pdf, tab_summary, tab_flashcards = st.tabs(["PDF Viewer/Uploader", "PDF Summary", "PDF Flashcards"])

    with tab_pdf:

        with st.popover("Options"):
            file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_workspace_uploader")

        if file is not None:
            pdf_text, pdf_height = _extract_text_from_pdf(file)
            learning_goals = _generate_learning_goals(pdf_text)

            col_learning_goals, col_pdf = st.columns([0.5,0.5])

            with col_learning_goals:
                st.header("Learning Goals")

                if file is not None and learning_goals:

                    parts = re.split(r'(?m)^\#\s*(.*)\s*$', learning_goals)  # -> [before, h1, c1, h2, c2, ...]
                    for title, content in zip(parts[1::2], parts[2::2], strict=True):
                        if content == "\n": # Empty content = title of pdf without learning goals
                            st.markdown(f"##{title.strip()}")
                        else: # Actual learning goal section
                            with st.expander(title.strip()):
                                if content.strip():
                                    st.markdown(content.strip()) # noqa

                option_store_message(learning_goals, key_suffix="pdf_learning_goals") if file is not None else None

            with col_pdf:
                st.header("Original PDF")
                st.pdf(file, height=pdf_height) if file is not None else None

    with tab_summary:
        if file is not None:
            button = st.button("Generate Summary Article")
            if button:
                wiki_article = _generate_wiki_article(pdf_text=pdf_text, learning_goals=learning_goals)
                st.markdown(wiki_article if file is not None else "")
                option_store_message(wiki_article, key_suffix="pdf_wiki_article") if file is not None else None
            else:
                st.info("Click the button to generate the summary article.")
        else:
            st.info("Upload a PDF in the 'PDF Viewer/Uploader' tab to generate a summary article.")

    with tab_flashcards:
        if file is not None:
            button = st.button("Generate Flashcards", key="generate_flashcards_button")
            if button:
                flashcards_df = _generate_flashcards(learning_goals)
                render_flashcards(flashcards_df)
            else:
                st.info("Click the button to generate flashcards.")
        else:
            st.info("Upload a PDF in the 'PDF Viewer/Uploader' tab to generate flashcards.")

def option_store_message(message: str, key_suffix: str) -> None:
    """Uses st.popover for a less intrusive save option."""
    with st.popover("Store answer"):
        # Use the key_suffix to ensure widget keys are unique
        filename = st.text_input("Filename", key=f"filename_input_{key_suffix}")
        if st.button("Save to Markdown", key=f"save_to_md_{key_suffix}"):
            write_to_md(filename=filename, message=message)
            st.success(f"Answer saved to {filename}")


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
                with st.chat_message("user"):
                    st.markdown(user_msg)
                    copy_button(user_msg)

                with st.chat_message("assistant"):
                    st.markdown(assistant_msg)
                    copy_button(assistant_msg)

                option_store_message(assistant_msg, key_suffix=f"{i // 2}")

def apply_custom_css() -> None:
    """Apply custom CSS styles to Streamlit app."""
    st.markdown(
        """
        <style>
        /* Overall app background and text color with Times New Roman */
        .stApp {
            font-family: 'Times New Roman', serif;
        }

        /* Box shadow for code blocks */
        pre {
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
