import os

from st_copy import copy_button
import streamlit as st
from streamlit_paste_button import PasteResult

from config import DIRECTORY_CHAT_HISTORIES
from lib.streamlit_helper import (
    AVAILABLE_LLM_MODELS,
    AVAILABLE_PROMPTS,
    get_img_hash,
    llm_params_sidebar,
    model_selector,
    paste_img_button,
    render_messages,
)
from llm_client import LLMClient

EMPTY_PASTE_RESULT = PasteResult(image_data=None)


def init_chat_variables() -> None:
    """Initialize session state variables for chat."""
    if "system_prompts" not in st.session_state:
        st.session_state.file_context = ""
        st.session_state.selected_model = AVAILABLE_LLM_MODELS[0]
        st.session_state.selected_prompt = next(iter(AVAILABLE_PROMPTS.keys()))
        st.session_state.system_prompts = AVAILABLE_PROMPTS
        st.session_state.usr_msg_captions = []
        st.session_state.refactor_code = False


# ---------------------------------------------------- Chat Interface functions ---------------------------------------------------- #
def chat_interface() -> None:
    col_left, _ = st.columns([0.9, 0.1])

    with col_left:
        st.write("")  # Spacer
        message_container = st.container()
        render_messages(message_container, client=st.session_state.client)

        with st._bottom:
            prompt = st.chat_input("Send a message", key="chat_input")

        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
                copy_button(prompt)
            with st.chat_message("assistant"):
                # Can be toggled if "Code Assistant" prompt is selected
                if (
                    st.session_state.refactor_code
                    and st.session_state.selected_prompt == "Code Assistant"
                ):
                    prompt = f"Analyze this module for possibilities of more concise implementation - without loss of robustness, compatibility or understandability :\n\n <code>\n{prompt}\n</code>"  # noqa
                    st.session_state.refactor_code = False

                system_prompt = st.session_state.system_prompts[
                    st.session_state.selected_prompt
                ]

                # Defaults to empty image if not provided
                api_img: PasteResult = st.session_state.api_img
                client: LLMClient = st.session_state.client
                kwargs = {
                    "temperature": st.session_state.llm_temperature,
                    "top_p": st.session_state.llm_top_p,
                    "reasoning_effort": st.session_state.llm_reasoning_effort,
                }

                st.write_stream(
                    client.chat(
                        model=st.session_state.selected_model,
                        user_msg=prompt,
                        system_prompt=system_prompt,
                        img=api_img,
                        stream=True,
                        **kwargs,
                    )
                )

                # Clear pasted image after use
                if api_img:
                    img_hash = get_img_hash(st.session_state.pasted_image)
                    st.session_state.sent_hashes.add(img_hash)
                    st.session_state.imgs_sent.append(st.session_state.pasted_image)
                    st.session_state.api_img = None
                    st.session_state.pasted_image = EMPTY_PASTE_RESULT
                    st.rerun()

                st.rerun()


# ----------------------------------------------------------- Sidebar ----------------------------------------------------------- #
def gigachad_sidebar() -> None:
    """Render the sidebar for gigachad bot."""
    init_chat_variables()

    with st.sidebar:
        # ------------------------------------------------- Model & Prompt Selection ------------------------------------------------- #
        st.session_state.selected_model = model_selector(key="gigachad_bot")

        st.session_state.selected_prompt = st.selectbox(
            "System prompt",
            list(st.session_state.system_prompts.keys()),
            key="prompt_select",
        )
        st.markdown("---")

        # ------------------------------------------------------ Model Config ------------------------------------------------------ #
        llm_params_sidebar()

        # -------------------------------------------------- Options & File Upload -------------------------------------------------- #
        st.markdown("---")
        with st.expander("Options", expanded=False):
            st.markdown("---")
            if st.button("Reset History", key="reset_history_main"):
                st.session_state.client.reset_history()

            if st.session_state.client.messages != []:
                st.markdown("---")
                with st.popover("Save History"):
                    filename = st.text_input("Filename", key="history_filename_input")
                    if st.button("Save Chat History", key="save_chat_history_button"):
                        if not os.path.exists(DIRECTORY_CHAT_HISTORIES):
                            os.makedirs(DIRECTORY_CHAT_HISTORIES)
                        st.session_state.client.store_history(
                            DIRECTORY_CHAT_HISTORIES + "/" + filename + ".csv"
                        )
                        st.success("Successfully saved chat")

        # ---------------------------------------------- Paste Image & Chat Histories ---------------------------------------------- #
        st.markdown("---")
        with st.expander("Upload Image"):
            paste_img_button()

        if os.path.exists(DIRECTORY_CHAT_HISTORIES):
            chat_histories = [
                f.replace(".csv", "")
                for f in os.listdir(DIRECTORY_CHAT_HISTORIES)
                if f.endswith(".csv")
            ]
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
                                st.session_state.client.load_history(
                                    os.path.join(
                                        DIRECTORY_CHAT_HISTORIES, history + ".csv"
                                    )
                                )
                        with col_delete:
                            if st.button("🗑", key=f"delete_{history}"):
                                os.remove(
                                    os.path.join(
                                        DIRECTORY_CHAT_HISTORIES, history + ".csv"
                                    )
                                )
                                st.rerun()
                        with col_archive:
                            if st.button("⛁", key=f"archive_{history}"):
                                if not os.path.exists(
                                    DIRECTORY_CHAT_HISTORIES + "/archived/"
                                ):
                                    os.makedirs(DIRECTORY_CHAT_HISTORIES + "/archived/")
                                os.rename(
                                    os.path.join(
                                        DIRECTORY_CHAT_HISTORIES, history + ".csv"
                                    ),
                                    os.path.join(
                                        DIRECTORY_CHAT_HISTORIES,
                                        "archived",
                                        history + ".csv",
                                    ),
                                )
                                st.rerun()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Gigachad Bot",
        page_icon="🤖",
        layout="wide",
    )

    gigachad_sidebar()
    chat_interface()
