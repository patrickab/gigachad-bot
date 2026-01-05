import streamlit as st

from lib.non_user_prompts import SYS_OCR_TEXT_EXTRACTION
from lib.streamlit_helper import (
    PasteResult,
    editor,
    get_img_hash,
    model_selector,
    options_message,
    paste_img_button,
    EMPTY_PASTE_RESULT
)
from llm_client import LLMClient


def ocr_sidebar() -> None:
    with st.sidebar:
        st.session_state.selected_model_ocr = model_selector(key="ocr_workspace")

        paste_img_button()


def ocr_workspace() -> None:
    """OCR Workspace page for uploading images and extracting text using OCR."""

    # Defaults to empty image if not provided
    img: PasteResult = st.session_state.api_img
    model: str = st.session_state.selected_model_ocr
    client: LLMClient = st.session_state.client

    if img != EMPTY_PASTE_RESULT and img is not None:
        stream = st.write_stream(
            client.api_query(
                model=model,
                system_prompt=SYS_OCR_TEXT_EXTRACTION,
                img=img,
                stream=True,
            )
        )
        # Todo: Rewrite LLM Client to return only text instead of response objects
        st.session_state.ocr_response = stream

        img_hash = get_img_hash(st.session_state.pasted_image)
        st.session_state.sent_hashes.add(img_hash)
        st.session_state.imgs_sent.append(st.session_state.pasted_image)
        st.session_state.api_img = None
        st.session_state.pasted_image = EMPTY_PASTE_RESULT
        st.rerun()


    if "ocr_response" in st.session_state:
        options_message(
            assistant_message=st.session_state.ocr_response, button_key="ocr_paste"
        )
        edited_content = editor(
            text_to_edit=st.session_state.ocr_response, language="markdown", key="ocr"
        )
        st.markdown(edited_content)
        st.session_state.ocr_response = edited_content


def main() -> None:
    """Main function to run the OCR Workspace page."""
    ocr_sidebar()
    ocr_workspace()


if __name__ == "__main__":
    main()
