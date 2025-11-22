from st_copy import copy_button
import streamlit as st
from streamlit_paste_button import PasteResult

from config import LOCAL_NANOTASK_MODEL, MODELS_OLLAMA
from lib.non_user_prompts import SYS_CAPTION_GENERATOR
from lib.streamlit_helper import _non_streaming_api_query, default_sidebar_chat, options_message


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
                prompt += st.session_state.file_context

                st.write_stream(
                    st.session_state.client.chat(
                        model=st.session_state.selected_model,
                        user_message=prompt,
                        img=st.session_state.pasted_image))

                if st.session_state.client.messages[-1][1] == "":
                    st.error("An error occurred while processing your request. Please try again.", icon="🚨")
                    st.session_state.client.messages = st.session_state.client.messages[:-2]
                else:
                    # Clear pasted image after use
                    st.session_state.last_sent_image = st.session_state.pasted_image
                    st.session_state.pasted_image = PasteResult(image_data=None)
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
            is_expanded = i == len(messages) - 2 # expand only the latest message
            label = f"QA-Pair {i // 2}: " if len(st.session_state.usr_msg_captions) == 0 else st.session_state.usr_msg_captions[i // 2]
            _, user_msg = messages[i]
            _, assistant_msg = messages[i + 1]

            with st.expander(label=label, expanded=is_expanded):
                # Display user and assistant messages
                with st.chat_message("user"):
                    st.markdown(user_msg)
                    # Copy button only works for expanded expanders
                    if is_expanded:
                        copy_button(user_msg)

                with st.chat_message("assistant"):
                    st.markdown(assistant_msg)
                    if is_expanded:
                        copy_button(assistant_msg)

                options_message(assistant_message=assistant_msg, button_key=f"{i // 2}", user_message=user_msg, index=i)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Gigachad Bot",
        page_icon="🤖",
        layout="wide",
    )

    default_sidebar_chat()
    chat_interface()