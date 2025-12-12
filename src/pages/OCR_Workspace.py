
import streamlit as st

from lib.non_user_prompts import SYS_OCR_TEXT_EXTRACTION
from lib.streamlit_helper import (
    EMPTY_PASTE_RESULT,
    PasteResult,
    editor,
    model_selector,
    options_message,
    paste_img_button,
    streamlit_img_to_bytes,
)


def ocr_sidebar() -> None:
    with st.sidebar:

        st.session_state.selected_model_ocr = model_selector(key="ocr_workspace")

        paste_img_button()

def ocr_workspace() -> None:
    """OCR Workspace page for uploading images and extracting text using OCR."""

    is_new_image = st.session_state.pasted_image != EMPTY_PASTE_RESULT and st.session_state.pasted_image not in st.session_state.imgs_sent # noqa
    if is_new_image:
        img: PasteResult = st.session_state.pasted_image
        st.session_state.ocr_response = st.write_stream(
            st.session_state.client.api_query(
            model=st.session_state.selected_model_ocr,
            system_prompt=SYS_OCR_TEXT_EXTRACTION,
            img=streamlit_img_to_bytes(img)
            )
        )
        st.session_state.imgs_sent.append(st.session_state.pasted_image)
        st.rerun()

    if "ocr_response" in st.session_state:
        options_message(assistant_message=st.session_state.ocr_response, button_key="ocr_paste")
        editor()


def main() -> None:
    """Main function to run the OCR Workspace page."""
    ocr_sidebar()
    ocr_workspace()

if __name__ == "__main__":
    main()
