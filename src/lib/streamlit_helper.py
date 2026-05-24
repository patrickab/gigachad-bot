"""Streamlit helper functions."""

import base64
import hashlib
import io
import math
import os
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
from PIL import Image
from st_copy import copy_button
from streamlit_ace import THEMES, st_ace
from streamlit_paste_button import PasteResult, paste_image_button

from config import (
    DIRECTORY_OBSIDIAN_VAULT,
)
from lib.non_user_prompts import SYS_NOTE_TO_OBSIDIAN_YAML
from lib.prompts import (
    SYS_ADVISOR,
    SYS_ARTICLE,
    SYS_CONCEPT_IN_DEPTH,
    SYS_CONCEPTUAL_OVERVIEW,
    SYS_EMPTY_PROMPT,
    SYS_QUICK_OVERVIEW,
    SYS_TUTOR,
)
from llm_client import LLMClient
from llm_config import (
    MODELS_GEMINI,
    MODELS_OLLAMA,
    MODELS_OPENAI,
    MODELS_VLLM,
    NANOTASK_MODEL,
)

EMPTY_PASTE_RESULT = PasteResult(image_data=None)

AVAILABLE_LLM_MODELS = []

if os.getenv("GEMINI_API_KEY"):
    AVAILABLE_LLM_MODELS += MODELS_GEMINI

if os.getenv("OPENAI_API_KEY"):
    AVAILABLE_LLM_MODELS += MODELS_OPENAI

if MODELS_OLLAMA != []:
    ignore_models = ["embeddinggemma:300m"]
    AVAILABLE_LLM_MODELS += MODELS_OLLAMA
    AVAILABLE_LLM_MODELS = [
        model for model in AVAILABLE_LLM_MODELS if model not in ignore_models
    ]

if MODELS_VLLM != []:
    AVAILABLE_LLM_MODELS += MODELS_VLLM

AVAILABLE_PROMPTS = {
    "Quick Overview": SYS_QUICK_OVERVIEW,
    "Advisor": SYS_ADVISOR,
    "Tutor": SYS_TUTOR,
    "Concept - High-Level": SYS_CONCEPTUAL_OVERVIEW,
    "Concept - In-Depth": SYS_CONCEPT_IN_DEPTH,
    "Concept - Article": SYS_ARTICLE,
    "<empty prompt>": SYS_EMPTY_PROMPT,
}


def init_session_state() -> None:
    """
    Initialize session state variables.
    Called within main directly after startup.

    Use for global session state variables.
    """

    if "client" not in st.session_state:
        st.session_state.workspace = "main"
        st.session_state.client = LLMClient()
        st.session_state.imgs_sent = [EMPTY_PASTE_RESULT]
        st.session_state.pasted_image = EMPTY_PASTE_RESULT
        st.session_state.selected_prompt = next(iter(AVAILABLE_PROMPTS.keys()))


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


def render_messages(message_container, client: LLMClient) -> None:  # noqa
    """Render chat messages from session state."""

    message_container.empty()  # Clear previous messages

    messages = client.messages

    if len(messages) == 0:
        return

    with message_container:
        for i in range(0, len(messages), 2):
            is_last = (
                i == len(messages) - 2
            )  # expand only the last message / display RAG context
            label = (
                f"QA-Pair {i // 2}: "
                if len(st.session_state.usr_msg_captions) == 0
                else st.session_state.usr_msg_captions[i // 2]
            )
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
                    if is_last:
                        copy_button(assistant_msg)

                options_message(
                    assistant_message=assistant_msg,
                    button_key=f"{i // 2}",
                    user_message=user_msg,
                    index=i,
                )


def editor(
    text_to_edit: str, language: str, key: str, height: Optional[int] = None
) -> str:
    """Create an ACE editor for displaying OCR extracted text."""
    default_theme = "chaos"
    selected_theme = st.selectbox(
        label="Editor Theme",
        options=THEMES,
        index=THEMES.index(default_theme),
        key=f"editor_theme_{key}",
    )

    line_count = text_to_edit.count("\n") + 1
    if height is None:
        height = line_count * 15

    content = st_ace(
        value=text_to_edit,
        language=language,
        height=height,
        key=f"editor_{key}",
        theme=selected_theme,
    )  # noqa
    content  # noqa
    return content


def streamlit_img_to_bytes(img: PasteResult) -> bytes:
    buffer = io.BytesIO()
    img.image_data.save(buffer, format="PNG")
    return buffer.getvalue()


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
    st.session_state.use_resize = st.toggle(
        "Enable Image Resizing",
        value=True,
        help="Downscale image to fit model token limits",
    )

    params = {}
    if st.session_state.use_resize:
        with st.expander("Compression Configurations"):
            c1, c2 = st.columns(2)
            params["max_tokens"] = c1.number_input("Max Tokens", value=300, step=100)
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
        button_color_bg, button_color_txt, button_color_hover = (
            "#34373E",
            "#FFFFFF",
            "#45494E",
        )
    else:
        button_color_bg, button_color_txt, button_color_hover = (
            "#E6E6E6",
            "#000000",
            "#CCCCCC",
        )

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
        st.session_state.api_img = st.session_state.client.downscale_img(
            paste_result.image_data, **params
        )
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
        st.session_state.api_img = (
            f"data:image/{params['output_format'].lower()};base64,{b64}"
        )

    st.session_state.pasted_image = paste_result
    return paste_result


def write_to_md(filename: str, message: str) -> None:
    """Write an assistant response to .md (idx: 0 = most recent)."""
    if not filename.endswith(".md"):
        filename += ".md"

    sys_prompt = SYS_NOTE_TO_OBSIDIAN_YAML.replace(
        "{{file_name_no_ext}}", filename.split(".md")[0]
    )
    llm_client = st.session_state.client
    response = llm_client.api_query(
        model=NANOTASK_MODEL,
        user_msg=message,
        system_prompt=sys_prompt.replace("{{user_notes}}", message),
        stream=False,
    )
    yaml_header = response.choices[0]["message"]["content"]

    file_path = DIRECTORY_OBSIDIAN_VAULT / filename
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(yaml_header + "\n" + message)

    markdown_dir = Path("markdown")
    markdown_dir.mkdir(exist_ok=True)
    with open(markdown_dir / filename, "w", encoding="utf-8") as f:
        f.write(yaml_header + "\n" + message)


def options_message(
    assistant_message: str, button_key: str, user_message: str = None, index: int = None
) -> None:  # noqa
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
