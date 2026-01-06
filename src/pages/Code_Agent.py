from code_agents.app_ui import agent_controls, chat_interface, controller
import streamlit as st

if __name__ == "__main__":
    with st.sidebar:
        agent_controls(controller=controller)
    chat_interface(controller=controller)
