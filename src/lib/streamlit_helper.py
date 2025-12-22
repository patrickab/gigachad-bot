"""Streamlit helper functions."""

import base64
from contextlib import contextmanager
import hashlib
import io
import math
import os
from pathlib import Path
import tempfile
from typing import Iterator, Optional

import fitz
from PIL import Image
import pymupdf4llm
from rag_database.rag_database import DatabaseKeys
import requests
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

    if MODELS_OLLAMA != []:
        model_options.append("Ollama")
        model_configs["Ollama"] = (MODELS_OLLAMA, "ollama/")
    if MODELS_GEMINI != []:
        model_options.append("Gemini")
        model_configs["Gemini"] = (MODELS_GEMINI, "gemini/")
    if MODELS_OPENAI != []:
        model_options.append("OpenAI")
        model_configs["OpenAI"] = (MODELS_OPENAI, "openai/")
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

    return st.selectbox(
        label=f"Models ({selected_provider})",
        options=models_list,
        format_func=lambda model: model.replace(litellm_prefix, ""),
        key=f"{selected_provider}_model_select_{key}",
    )


def llm_params_sidebar() -> None:
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
            options=["none", "low", "medium", "high"],
            key="reasoning_effort",
        )


def render_messages(message_container, client: LLMClient) -> None:  # noqa
    """Render chat messages from session state."""

    message_container.empty()  # Clear previous messages

    messages = client.messages

    if len(messages) == 0:
        return

    with message_container:
        for i in range(0, len(messages), 2):
            is_last = i == len(messages) - 2  # expand only the last message / display RAG context
            label = f"QA-Pair {i // 2}: " if len(st.session_state.usr_msg_captions) == 0 else st.session_state.usr_msg_captions[i // 2]
            user_msg = messages[i]["content"]
            assistant_msg = messages[i + 1]["content"]

            with st.expander(label=label, expanded=is_last):
                # Display user and assistant messages
                with st.chat_message("user"):
                    st.markdown(user_msg)
                    # Copy button only works for expanded expanders
                    if is_last:
                        copy_button(user_msg)

                with st.chat_message("assistant"):
                    st.markdown(assistant_msg)
                    if is_last and st.session_state.is_rag_active:
                        documents = st.session_state.rag_response.to_polars()
                        for doc in documents.iter_rows(named=True):
                            with st.expander(
                                f"**Similarity**: {doc[DatabaseKeys.KEY_SIMILARITIES]:.2f}   -  **Title**: {doc[DatabaseKeys.KEY_TITLE]}"
                            ):  # noqa
                                st.markdown(doc[DatabaseKeys.KEY_TXT_RETRIEVAL])

                    if is_last:
                        copy_button(assistant_msg)

                options_message(assistant_message=assistant_msg, button_key=f"{i // 2}", user_message=user_msg, index=i)


def print_metrics(dict_metrics: dict[str, int | float], n_columns: Optional[int] = None) -> None:
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


def downscale_img(
    img: str | bytes | Path | Image.Image,
    max_tokens: int,
    grid_size: int = 28,
    tokens_per_patch: int = 1,
    row_overhead_tokens: int = 1,
    output_format: str = "JPEG",
    quality: int = 85,
) -> str:
    """
    Preserves aspect ratio (area-scaling) and aligns to ViT patches (grid_size) for spatial accuracy.
    Optimizes GPU inference efficiency within token budget.

    JPEG: Fastest TTFT (optimized CPU decoding).
    PNG: Max fidelity (lossless).
    """
    # 1. Normalize Input to PIL Image
    if isinstance(img, Image.Image):
        pass
    elif hasattr(img, "image_data"):  # Streamlit PasteResult
        img = Image.open(io.BytesIO(streamlit_img_to_bytes(img)))
    elif isinstance(img, (str, Path)):
        src = str(img)
        if src.startswith("http"):
            img = Image.open(io.BytesIO(requests.get(src, timeout=10).content))
        elif src.startswith("data:image"):
            img = Image.open(io.BytesIO(base64.b64decode(src.partition(",")[-1])))
        else:
            img = Image.open(Path(src))
    else:
        img = Image.open(io.BytesIO(img) if isinstance(img, bytes) else img)

    # 2. Universal Resizing Logic
    def get_tokens(w: int, h: int) -> int:
        pw, ph = math.ceil(w / grid_size), math.ceil(h / grid_size)
        return (pw * ph * tokens_per_patch) + (ph * row_overhead_tokens)

    w, h = img.size
    curr_tokens = get_tokens(w, h)

    scale = 1.0
    if curr_tokens > max_tokens:
        scale = math.sqrt(max_tokens / curr_tokens)

    # Snap to Grid (Preserving Aspect Ratio)
    fw = max(grid_size, round((w * scale) / grid_size) * grid_size)
    fh = max(grid_size, round((h * scale) / grid_size) * grid_size)

    # Iterative refinement to guarantee budget compliance
    while get_tokens(fw, fh) > max_tokens and (fw > grid_size or fh > grid_size):
        if fw > fh:
            fw -= grid_size
        else:
            fh -= grid_size

    if (fw, fh) != (w, h):
        img = img.resize((fw, fh), Image.Resampling.LANCZOS)

    # 3. Configurable Encoding (Optimized for Localhost)
    if output_format.upper() in ["JPEG", "JPG"]:
        if img.mode != "RGB":
            img = img.convert("RGB")
        save_params = {"quality": quality, "optimize": False}
    elif output_format.upper() == "PNG":
        save_params = {"optimize": False}
    else:
        save_params = {"quality": quality}

    buf = io.BytesIO()
    img.save(buf, format=output_format, **save_params)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{output_format.lower()};base64,{b64}"


def get_img_hash(img: PasteResult) -> str:
    """Generate SHA256 hash for a pasted image."""
    img_bytes = streamlit_img_to_bytes(img)
    return hashlib.sha256(img_bytes).hexdigest()

def paste_img_button() -> PasteResult:
    """Handle image pasting in Streamlit app with hashing and state control."""

    if "sent_hashes" not in st.session_state:
        st.session_state.sent_hashes = set()
    if "imgs_sent" not in st.session_state:
        st.session_state.imgs_sent = [EMPTY_PASTE_RESULT]

    # 1. Resizing Configuration UI
    st.session_state.use_resize = st.toggle("Enable Image Resizing", value=True, help="Downscale image to fit model token limits")

    params = {}
    if st.session_state.use_resize:
        with st.expander("Compression Configurations"):
            c1, c2 = st.columns(2)
            params["max_tokens"] = c1.number_input("Max Tokens", value=800, step=100)
            params["grid_size"] = c1.number_input("Grid Size", value=28, step=1)
            params["tokens_per_patch"] = c1.number_input("Tokens per Patch", value=1)
            params["output_format"] = c2.selectbox("Format", ["PNG", "JPEG"], index=0)
            params["quality"] = c2.slider("Quality", 1, 100, 85)
            params["row_overhead_tokens"] = c2.number_input("Row Overhead", value=1)
    else:
        # Defaults for raw conversion when resizing is disabled
        params = {"output_format": "PNG", "quality": 85}

    if st.session_state.imgs_sent != [EMPTY_PASTE_RESULT]:
        with st.expander("Previously pasted images:"):
            for idx, img in enumerate(st.session_state.imgs_sent):
                if img != EMPTY_PASTE_RESULT:
                    with st.expander(f"Image {idx + 1}"):
                        st.image(img.image_data, caption=f"Image {idx + 1}")

    # 2. Button Theme Logic
    bg_color = st.get_option("theme.base")
    if bg_color == "dark":
        button_color_bg, button_color_txt, button_color_hover = "#34373E", "#FFFFFF", "#45494E"
    else:
        button_color_bg, button_color_txt, button_color_hover = "#E6E6E6", "#000000", "#CCCCCC"

    paste_result = paste_image_button(
        "Paste from clipboard",
        background_color=button_color_bg,
        text_color=button_color_txt,
        hover_background_color=button_color_hover,
    )

    def return_empty() -> PasteResult:
        st.session_state.pasted_image = EMPTY_PASTE_RESULT
        st.session_state.api_img = None
        return EMPTY_PASTE_RESULT

    is_image_pasted = paste_result is not None and paste_result.image_data is not None
    if not is_image_pasted:
        return return_empty()

    img_hash = get_img_hash(paste_result)
    if img_hash in st.session_state.sent_hashes:
        return return_empty()

    # 3. Processing and State Control
    if st.session_state.use_resize:
        st.session_state.api_img = downscale_img(paste_result.image_data, **params)
        st.image(paste_result.image_data, caption="Pasted Image (original)")
        st.image(st.session_state.api_img, caption="Pasted Image (resized)")
    else:
        # Raw conversion to Base64 without scaling
        img = Image.open(io.BytesIO(streamlit_img_to_bytes(paste_result)))
        if img.mode != "RGB" and params["output_format"] == "JPEG":
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format=params["output_format"], quality=params["quality"])
        st.image(img, caption="Pasted Image")

        b64 = base64.b64encode(buf.getvalue()).decode()
        st.session_state.api_img = f"data:image/{params['output_format'].lower()};base64,{b64}"

    st.session_state.pasted_image = paste_result
    return paste_result

def editor(text_to_edit: str, language: str, key: str, height: Optional[int] = None) -> str:
    """Create an ACE editor for displaying OCR extracted text."""
    default_theme = "chaos"
    selected_theme = st.selectbox(label="Editor Theme", options=THEMES, index=THEMES.index(default_theme), key=f"editor_theme_{key}")

    line_count = text_to_edit.count("\n") + 1
    if height is None:
        height = line_count * 15

    content = st_ace(value=text_to_edit, language=language, height=height, key=f"editor_{key}", theme=selected_theme)  # noqa
    content  # noqa
    return content


def _non_streaming_api_query(model: str, prompt: str, system_prompt: str, img: Optional[PasteResult] = EMPTY_PASTE_RESULT) -> str:
    """TODO: remove - Legacy helper - can be replaced by client api calls"""
    response = st.session_state.client.api_query(
        model=model, user_message=prompt, system_prompt=system_prompt, stream=False, chat_history=None, img=img.image_data
    )

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


def options_message(assistant_message: str, button_key: str, user_message: str = None, index: int = None) -> None:  # noqa
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
            del st.session_state.client.messages[index : index + 2]
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
