
import streamlit as st
from streamlit_ace import THEMES, st_ace

from lib.non_user_prompts import SYS_OCR_TEXT_EXTRACTION
from src.config import MODELS_OCR_OLLAMA
from src.lib.streamlit_helper import options_message, paste_img_button
from src.llm_client import EMPTY_PASTE_RESULT


def ocr_sidebar() -> None:
    with st.sidebar:

        st.session_state.selected_model_ocr = st.selectbox(
            label="Select OCR Model",
            options=MODELS_OCR_OLLAMA,
            index=0,
        )

        paste_img_button()

def ocr_workspace() -> None:
    """OCR Workspace page for uploading images and extracting text using OCR."""


    is_new_image = st.session_state.pasted_image != EMPTY_PASTE_RESULT and st.session_state.pasted_image not in st.session_state.imgs_sent # noqa
    if is_new_image:
        st.session_state.ocr_response = st.write_stream(
            st.session_state.client.api_query(
            model=st.session_state.selected_model_ocr,
            system_prompt=SYS_OCR_TEXT_EXTRACTION,
            img=st.session_state.pasted_image
            )
        )
        st.session_state.imgs_sent.append(st.session_state.pasted_image)
        st.rerun()

    if "ocr_response" in st.session_state:

        default = "chaos"
        with st.sidebar:
            selected_theme = st.selectbox(
                label="Editor Theme",
                options=THEMES,
                index=THEMES.index(default),
                key="ocr_editor_theme"
            )

        options_message(assistant_message=st.session_state.ocr_response, button_key="ocr_paste")
        line_count = st.session_state.ocr_response.count("\n") + 1
        adaptive_height = line_count*15
        content = st_ace(value=st.session_state.ocr_response, language="latex", height=adaptive_height, key="latex_editor", theme=selected_theme) # noqa
        content # noqa


def main() -> None:
    """Main function to run the OCR Workspace page."""
    ocr_sidebar()
    ocr_workspace()

if __name__ == "__main__":
    main()
