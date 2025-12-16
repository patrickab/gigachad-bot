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
from streamlit_ace import THEMES, st_ace
from streamlit_paste_button import PasteResult, paste_image_button

from config import (
    DIRECTORY_EMBEDDINGS,
    DIRECTORY_LLM_PREPROCESSING,
    DIRECTORY_OBSIDIAN_VAULT,
    DIRECTORY_RAG_INPUT,
    DIRECTORY_VLM_OUTPUT,
    SERVER_STATIC_DIR,
)
from lib.non_user_prompts import SYS_NOTE_TO_OBSIDIAN_YAML
from lib.prompts import (
    SYS_ADVISOR,
    SYS_ARTICLE,
    SYS_CODE_OPERATOR,
    SYS_CONCEPT_IN_DEPTH,
    SYS_CONCEPTUAL_OVERVIEW,
    SYS_EMPTY_PROMPT,
    SYS_MATH_PROOF,
    SYS_PROMPT_ENGINEER,
    SYS_QUICK_OVERVIEW,
    SYS_RAG_TUTOR,
)
from llm_client import LLMClient
from llm_config import MODELS_EXLLAMA, MODELS_GEMINI, MODELS_OLLAMA, MODELS_OPENAI, MODELS_VLLM, NANOTASK_MODEL

EMPTY_PASTE_RESULT = PasteResult(image_data=None)

AVAILABLE_LLM_MODELS = []

if os.getenv("GEMINI_API_KEY"):
    AVAILABLE_LLM_MODELS += MODELS_GEMINI

if os.getenv("OPENAI_API_KEY"):
    AVAILABLE_LLM_MODELS += MODELS_OPENAI

if MODELS_OLLAMA != []:
    ignore_models = ["embeddinggemma:300m"]
    AVAILABLE_LLM_MODELS += MODELS_OLLAMA
    AVAILABLE_LLM_MODELS = [model for model in AVAILABLE_LLM_MODELS if model not in ignore_models]

if MODELS_VLLM != []:
    AVAILABLE_LLM_MODELS += MODELS_VLLM

AVAILABLE_PROMPTS = {
    "Quick Overview": SYS_QUICK_OVERVIEW,
    "Code Assistant": SYS_CODE_OPERATOR,
    "Advisor": SYS_ADVISOR,
    "Tutor": SYS_RAG_TUTOR,
    "Math Proof": SYS_MATH_PROOF,
    "Concept - High-Level": SYS_CONCEPTUAL_OVERVIEW,
    "Concept - In-Depth": SYS_CONCEPT_IN_DEPTH,
    "Concept - Article": SYS_ARTICLE,
    "Prompt Engineer": SYS_PROMPT_ENGINEER,
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
    os.makedirs(DIRECTORY_RAG_INPUT, exist_ok=True)
    os.makedirs(DIRECTORY_EMBEDDINGS, exist_ok=True)
    os.makedirs(DIRECTORY_LLM_PREPROCESSING, exist_ok=True)

    if "client" not in st.session_state:
        st.session_state.workspace = "main"
        st.session_state.client = LLMClient()
        st.session_state.imgs_sent = [EMPTY_PASTE_RESULT]
        st.session_state.pasted_image = EMPTY_PASTE_RESULT

def model_selector(key: str) -> dict:
    """Create model selection dropdowns in Streamlit sidebar expanders."""
    model_options = []
    model_configs = {}

    if MODELS_GEMINI != []:
        model_options.append("Gemini")
        model_configs["Gemini"] = (MODELS_GEMINI, "gemini/")
    if MODELS_OPENAI != []:
        model_options.append("OpenAI")
        model_configs["OpenAI"] = (MODELS_OPENAI, "openai/")
    if MODELS_OLLAMA != []:
        model_options.append("Ollama")
        model_configs["Ollama"] = (MODELS_OLLAMA, "ollama/")
    if MODELS_VLLM != []:
        model_options.append("VLLM")
        model_configs["VLLM"] = (MODELS_VLLM, "hosted_vllm/")
    if MODELS_EXLLAMA != []:
        model_options.append("ExLlama")
        model_configs["ExLlama"] = (MODELS_EXLLAMA, "openai/")  # TabbyAPI uses OpenAI-convention

    selected_provider = st.radio(
        label="Model Provider",
        options=model_options,
        index=0,
        horizontal=True,
        key=f"model_provider_radio_{key}",
    )
    models_list, litellm_prefix = model_configs[selected_provider]

    st.session_state.selected_model = st.selectbox(
        label=f"Models ({selected_provider})",
        options=models_list,
        format_func=lambda model: model.replace(litellm_prefix, ""),
        key=f"{selected_provider}_model_select_{key}",
    )

def llm_params_sidebar()-> None:
    """Create LLM parameter sliders in Streamlit expander."""
    with st.expander("Model Configuration", expanded=False):

        if st.session_state.selected_prompt == "Code Assistant":
            st.session_state.refactor_code = st.toggle("Refactor code", value=False, key="refactor_code_toggle")

        st.session_state.llm_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.2,
            step=0.05,
            key="temperature",
        )
        st.session_state.llm_top_p = st.slider(
            "Top-p (nucleus sampling)",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05,
            key="top_p",
        )
        st.session_state.llm_reasoning_effort = st.selectbox(
            "Reasoning Effort",
            options=["none","low", "medium", "high"],
            key="reasoning_effort",
        )

def print_metrics(dict_metrics: dict[str,int|float], n_columns: Optional[int]=None) -> None:
    """Print metrics in Streamlit columns."""
    if n_columns is None:
        n_columns = len(dict_metrics)
    cols = st.columns(n_columns)
    for idx, (metric_name, metric_value) in enumerate(dict_metrics.items()):
        cols[idx % n_columns].metric(f"**{metric_name}:**", value=metric_value, border=True)

def streamlit_img_to_bytes(img: PasteResult) -> bytes:
    buffer = io.BytesIO()
    img.image_data.save(buffer, format="PNG")
    return buffer.getvalue()

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

def editor(text_to_edit: str, language: str, key: str, height: Optional[int] = None) -> str:
    """Create an ACE editor for displaying OCR extracted text."""
    default_theme = "chaos"
    selected_theme = st.selectbox(
        label="Editor Theme",
        options=THEMES,
        index=THEMES.index(default_theme),
        key=f"editor_theme_{key}"
    )

    line_count = text_to_edit.count("\n") + 1
    if height is None:
        height = line_count*15

    content = st_ace(value=text_to_edit, language=language, height=height, key=f"editor_{key}", theme=selected_theme) #noqa
    content # noqa
    return content

def _non_streaming_api_query(model: str, prompt: str, system_prompt: str, img:Optional[PasteResult] = EMPTY_PASTE_RESULT) -> str:
    """TODO: remove - Legacy helper - can be replaced by client api calls"""
    response = st.session_state.client.api_query(
        model=model,
        user_message=prompt,
        system_prompt=system_prompt,
        stream=False,
        chat_history=None, img=img.image_data)

    return response

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
