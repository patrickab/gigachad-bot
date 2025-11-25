"""Streamlit helper functions."""

import base64
from contextlib import contextmanager
import io
import os
import tempfile
from typing import Iterator, Optional

import fitz
import pymupdf4llm
from st_copy import copy_button
import streamlit as st
from streamlit_paste_button import PasteResult, paste_image_button

from src.config import (
    DIRECTORY_CHAT_HISTORIES,
    DIRECTORY_OBSIDIAN_VAULT,
    DIRECTORY_VLM_OUTPUT,
    MODELS_GEMINI,
    MODELS_OCR_OLLAMA,
    MODELS_OLLAMA,
    MODELS_OPENAI,
    NANOTASK_MODEL,
    SERVER_STATIC_DIR,
)
from src.lib.non_user_prompts import SYS_NOTE_TO_OBSIDIAN_YAML
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

EMPTY_PASTE_RESULT = PasteResult(image_data=None)

AVAILABLE_LLM_MODELS = []

if os.getenv("GEMINI_API_KEY"):
    AVAILABLE_LLM_MODELS += MODELS_GEMINI

if os.getenv("OPENAI_API_KEY"):
    AVAILABLE_LLM_MODELS += MODELS_OPENAI

if MODELS_OLLAMA != []:
    ignore_models = ["embeddinggemma:300m", *MODELS_OCR_OLLAMA]
    AVAILABLE_LLM_MODELS += MODELS_OLLAMA
    AVAILABLE_LLM_MODELS = [model for model in AVAILABLE_LLM_MODELS if model not in ignore_models]

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
    """
    Initialize session state variables.
    Called within main directly after startup.

    Use for global session state variables.
    """
    # Create static directory for serving PDFs
    os.makedirs(SERVER_STATIC_DIR, exist_ok=True)
    os.makedirs(DIRECTORY_VLM_OUTPUT, exist_ok=True)

    if "client" not in st.session_state:
        st.session_state.workspace = "main"
        st.session_state.client = LLMClient()
        st.session_state.imgs_sent = [EMPTY_PASTE_RESULT]
        st.session_state.pasted_image = EMPTY_PASTE_RESULT

def init_chat_variables() -> None:
    """Initialize session state variables for chat."""
    if "system_prompts" not in st.session_state:
        st.session_state.file_context = ""
        st.session_state.selected_model = AVAILABLE_LLM_MODELS[0]
        st.session_state.selected_prompt = next(iter(AVAILABLE_PROMPTS.keys()))
        st.session_state.system_prompts = AVAILABLE_PROMPTS
        st.session_state.usr_msg_captions = []

def print_metrics(dict_metrics: dict[str,int|float], n_columns: Optional[int]=None) -> None:
    """Print metrics in Streamlit columns."""
    if n_columns is None:
        n_columns = len(dict_metrics)
    cols = st.columns(n_columns)
    for idx, (metric_name, metric_value) in enumerate(dict_metrics.items()):
        cols[idx % n_columns].metric(f"**{metric_name}:**", value=metric_value, border=True)

def paste_img_button() -> PasteResult:
    """Handle image pasting in Streamlit app."""

    if st.session_state.imgs_sent != [EMPTY_PASTE_RESULT]:
        with st.expander("Previously pasted images:"):
            for idx, img in enumerate(st.session_state.imgs_sent):
                if img != EMPTY_PASTE_RESULT:
                    with st.expander(f"Image {idx+1}"):
                        st.image(img.image_data, caption=f"Image {idx+1}")

    # check wether streamlit background is in dark mode or light mode
    bg_color = st.get_option("theme.base")  # 'light' or 'dark
    if bg_color == "dark":
        button_color_bg = "#34373E"
        button_color_txt = "#FFFFFF"
        button_color_hover = "#45494E"
    else:
        button_color_bg = "#E6E6E6"
        button_color_txt = "#000000"
        button_color_hover = "#CCCCCC"

    paste_result = paste_image_button("Paste from clipboard",
                background_color=button_color_bg,
                text_color=button_color_txt,
                hover_background_color=button_color_hover)

    if paste_result not in st.session_state.imgs_sent:

        st.session_state.pasted_image = paste_result
        st.image(paste_result.image_data)
        return paste_result

    else: # set pasted_image to None
        st.session_state.pasted_image = EMPTY_PASTE_RESULT
        return EMPTY_PASTE_RESULT

def default_sidebar_chat() -> None:
    """Render the default sidebar for chat applications."""
    init_chat_variables()

    with st.sidebar:

        model = st.selectbox(
            "Select LLM",
            AVAILABLE_LLM_MODELS,
            key="model_select",
        )

        sys_prompt_name = st.selectbox(
            "System prompt",
            list(st.session_state.system_prompts.keys()),
            key="prompt_select",
        )

        if sys_prompt_name != st.session_state.selected_prompt:
            st.session_state.selected_prompt = sys_prompt_name

        if model != st.session_state.selected_model:
            st.session_state.selected_model = model

        # -------------------------------------------------- Options & File Upload -------------------------------------------------- #
        st.markdown("---")
        with st.expander("Options", expanded=False):
            st.session_state.bool_caption_usr_msg = st.toggle("Caption User Messages", key="caption_toggle", value=False)
            st.markdown("---")
            if st.button("Reset History", key="reset_history_main"):
                st.session_state.client.reset_history()

            st.markdown("---")
            file = st.file_uploader(type=["pdf", "py", "md", "cpp", "txt"], label="Upload file context (.pdf/.txt/.py)")
            if file is not None:
                if file.type == "application/pdf":
                    text, _ = _extract_text_from_pdf(file)
                else:
                    text = file.getvalue().decode("utf-8")
                st.session_state.file_context = text

            if st.session_state.client.messages != []:
                st.markdown("---")
                with st.popover("Save History"):
                    filename = st.text_input("Filename", key="history_filename_input")
                    if st.button("Save Chat History", key="save_chat_history_button"):
                        if not os.path.exists(DIRECTORY_CHAT_HISTORIES):
                            os.makedirs(DIRECTORY_CHAT_HISTORIES)
                        st.session_state.client.store_history(DIRECTORY_CHAT_HISTORIES + '/' + filename + '.csv')
                        st.success("Successfully saved chat")

        # ---------------------------------------------- Paste Image & Chat Histories ---------------------------------------------- #
        st.markdown("---")
        with st.expander("Upload Image"):

            paste_img_button()

        if os.path.exists(DIRECTORY_CHAT_HISTORIES):
            chat_histories = [f.replace('.csv', '') for f in os.listdir(DIRECTORY_CHAT_HISTORIES) if f.endswith('.csv')]
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
                                st.session_state.client.load_history(os.path.join(DIRECTORY_CHAT_HISTORIES, history + '.csv'))
                        with col_delete:
                            if st.button("🗑", key=f"delete_{history}"):
                                os.remove(os.path.join(DIRECTORY_CHAT_HISTORIES, history + '.csv'))
                                st.rerun()
                        with col_archive:
                            if st.button("⛁", key=f"archive_{history}"):
                                if not os.path.exists(DIRECTORY_CHAT_HISTORIES + '/archived/'):
                                    os.makedirs(DIRECTORY_CHAT_HISTORIES + '/archived/')
                                os.rename(
                                    os.path.join(DIRECTORY_CHAT_HISTORIES, history + '.csv'),
                                    os.path.join(DIRECTORY_CHAT_HISTORIES, 'archived', history + '.csv')
                                )
                                st.rerun()

def _non_streaming_api_query(model: str, prompt: str, system_prompt: str, img:Optional[PasteResult] = EMPTY_PASTE_RESULT) -> str:
    """
    Converts streaming response generator to generic string.
    Required for @st.cache_data compatibility.
    """
    stream = st.session_state.client.api_query(
        model=model,
        user_message=prompt,
        system_prompt=system_prompt,
        chat_history=None, img=img)

    response_text = ""
    for chunk in stream:
        response_text += chunk

    return response_text

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

    file_path = os.path.join(DIRECTORY_OBSIDIAN_VAULT, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(yaml_header + "\n" + message)

    os.makedirs("markdown", exist_ok=True)
    with open(os.path.join("markdown", filename), "w", encoding="utf-8") as f:
        f.write(yaml_header + "\n" + message)

def options_message(assistant_message: str, button_key: str, user_message: str = None, index: int = None) -> None: # noqa
    """Uses st.popover for a less intrusive save option."""
    with st.popover("Options"):
        with st.popover("Store answer"):
            # Use the button_key to ensure widget keys are unique
            filename = st.text_input("Filename", key=f"filename_input_{button_key}")
            if st.button("Save to Markdown", key=f"save_to_md_{button_key}"):
                write_to_md(filename=filename, message=assistant_message)
                st.success(f"Answer saved to {filename}")

        with st.popover("Copy Messages"):
            if user_message is not None:
                st.markdown("**Copy User Message**")
                copy_button(text=user_message)

            st.markdown("**Asssistant Message**")
            copy_button(text=assistant_message)

        if index is not None and st.button("🗑", key=f"del_{index}"):
            del st.session_state.client.messages[index:index+2]
            st.rerun()

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

@st.cache_resource
def _extract_text_from_pdf(file: io.BytesIO) -> str:
    """Extract text from uploaded PDF file using pymupdf4llm."""
    # Create temporary file - pymupdf4llm requires a file path but Streamlit's doesnt support that directly
    # However, streamlits file uploader returns a BytesIO object which we can write to a temp file & read from there
    # Using with context manager ensures temp file is deleted after use
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

@contextmanager
def nyan_cat_spinner() -> Iterator:
    """Display nyan cat spinner animation."""
    file_path = "assets/nyan-cat.gif"
    placeholder = st.empty()

    with open(file_path, "rb") as f:
        contents = f.read()
        data_url = base64.b64encode(contents).decode("utf-8")

    try:
        with placeholder.container():
            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="nyan cat gif" width="150">',
                unsafe_allow_html=True,
            )
            yield

    finally:
        placeholder.empty()
