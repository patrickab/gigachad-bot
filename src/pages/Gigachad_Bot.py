import os

from llm_baseclient.client import LLMClient
from rag_database.rag_database import DatabaseKeys, RagDatabase, RAGQuery, RAGResponse
from st_copy import copy_button
import streamlit as st
from streamlit_paste_button import PasteResult

from config import DIRECTORY_CHAT_HISTORIES
from lib.non_user_prompts import SYS_CAPTION_GENERATOR, SYS_RAG
from lib.streamlit_helper import (
    AVAILABLE_LLM_MODELS,
    AVAILABLE_PROMPTS,
    _extract_text_from_pdf,
    _non_streaming_api_query,
    llm_params_sidebar,
    model_selector,
    options_message,
    paste_img_button,
    streamlit_img_to_bytes,
)
from llm_config import LOCAL_NANOTASK_MODEL, MODELS_OLLAMA
from pages.RAG_Workspace import init_rag_workspace, rag_sidebar

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
        render_messages(message_container)

        with st._bottom:
            prompt = st.chat_input("Send a message", key="chat_input")

        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
                copy_button(prompt)
            with st.chat_message("assistant"):

                # Can be toggled if "Code Assistant" prompt is selected
                if st.session_state.refactor_code and st.session_state.selected_prompt == "Code Assistant":
                    prompt = f"Analyze this module for possibilities of more concise implementation - without loss of robustness, compatibility or understandability :\n\n <code>\n{prompt}\n</code>" # noqa
                    st.session_state.refactor_code = False

                system_prompt = st.session_state.system_prompts[st.session_state.selected_prompt]

                if st.session_state.is_rag_active:
                    rag_db: RagDatabase = st.session_state.rag_databases[st.session_state.selected_rag_database][st.session_state.selected_embedding_model] # noqa
                    rag_query = RAGQuery(query=prompt, k_documents=st.session_state.k_query_documents)
                    rag_response: RAGResponse = rag_db.rag_process_query(rag_query=rag_query)
                    st.session_state.rag_response = rag_response

                    titles = rag_response.titles
                    texts = rag_response.texts

                    retrieved_information = "<context>"
                    for title, text in zip(titles, texts, strict=True):
                        retrieved_information += f"\n\n<doc title=\"{title}\">\n{text}</doc>\n" # noqa
                    retrieved_information += "</context>"

                    system_prompt = system_prompt + "\n\n" + SYS_RAG + "\n\n# RAG Retrieved Context:\n" + retrieved_information

                img: PasteResult = st.session_state.pasted_image # defaults to EMPTY_PASTE_RESULT if no image pasted
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
                        img=streamlit_img_to_bytes(img) if img.image_data is not None else None,
                        stream=True,
                        **kwargs,
                    )
                )

                # Clear pasted image after use
                st.session_state.last_sent_image = img
                st.session_state.pasted_image = EMPTY_PASTE_RESULT
                # Caption user message
                # Blocks new users without local ollama setup
                if LOCAL_NANOTASK_MODEL in MODELS_OLLAMA and st.session_state.bool_caption_usr_msg:
                    caption = _non_streaming_api_query(
                        model=LOCAL_NANOTASK_MODEL,
                        prompt=prompt[:80],  # limit prompt length for captioning to avoid long processing times
                        system_prompt=SYS_CAPTION_GENERATOR
                    )
                    st.session_state.usr_msg_captions += [caption]

                st.rerun()

def render_messages(message_container) -> None:  # noqa
    """Render chat messages from session state."""

    message_container.empty()  # Clear previous messages

    messages = st.session_state.client.messages

    if len(messages) == 0:
        return

    with message_container:
        for i in range(0, len(messages), 2):
            is_last = i == len(messages) - 2 # expand only the last message / display RAG context
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
                            with st.expander(f"**Similarity**: {doc[DatabaseKeys.KEY_SIMILARITIES]:.2f}   -  **Title**: {doc[DatabaseKeys.KEY_TITLE]}"): # noqa
                                st.markdown(doc[DatabaseKeys.KEY_TXT_RETRIEVAL])

                    if is_last:
                        copy_button(assistant_msg)

                options_message(assistant_message=assistant_msg, button_key=f"{i // 2}", user_message=user_msg, index=i)

# ----------------------------------------------------------- Sidebar ----------------------------------------------------------- #
def gigachad_sidebar() -> None:
    """Render the sidebar for gigachad bot."""
    init_chat_variables()

    with st.sidebar:

        #------------------------------------------------- Model & Prompt Selection ------------------------------------------------- #
        model_selector(key="gigachad_bot")

        sys_prompt_name = st.selectbox(
            "System prompt",
            list(st.session_state.system_prompts.keys()),
            key="prompt_select",
        )

        if sys_prompt_name != st.session_state.selected_prompt:
            st.session_state.selected_prompt = sys_prompt_name

        # ------------------------------------------------------ Model Config ------------------------------------------------------ #
        llm_params_sidebar()

        # -------------------------------------------------------- RAG Mode -------------------------------------------------------- #
        st.markdown("---")
        if st.toggle("Activate RAG Mode", key="gigachad_bot_rag_mode", value=False):
            st.session_state.is_rag_active = True
            init_rag_workspace()
            rag_sidebar()
        else:
            st.session_state.is_rag_active = False


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

if __name__ == "__main__":
    st.set_page_config(
        page_title="Gigachad Bot",
        page_icon="🤖",
        layout="wide",
    )

    gigachad_sidebar()
    chat_interface()