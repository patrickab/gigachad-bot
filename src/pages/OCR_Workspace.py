import streamlit as st

from lib.non_user_prompts import SYS_OCR_TEXT_EXTRACTION
from src.config import MODELS_OCR_OLLAMA
from src.lib.streamlit_helper import options_message, paste_img_button
from src.llm_client import EMPTY_PASTE_RESULT


def init_ocr_workspace() -> None:
    """Initialize OCR workspace session state variables."""
    if "ocr_uploaded_images" not in st.session_state:
        st.session_state.ocr_uploaded_images = []
        st.session_state.selected_model = MODELS_OCR_OLLAMA[0]

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
    if st.session_state.pasted_image == EMPTY_PASTE_RESULT:
        st.markdown("Upload images to extract text using OCR.")
    else:
        response = st.write_stream(
            st.session_state.client.api_query(
            model=st.session_state.selected_model_ocr,
            system_prompt=SYS_OCR_TEXT_EXTRACTION,
            img=st.session_state.pasted_image
            )
        )
        st.session_state.imgs_sent.append(st.session_state.pasted_image)
        options_message(assistant_message=response, button_key="ocr_paste")

def main() -> None:
    """Main function to run the OCR Workspace page."""
    init_ocr_workspace()
    ocr_sidebar()
    ocr_workspace()

if __name__ == "__main__":
    main()
