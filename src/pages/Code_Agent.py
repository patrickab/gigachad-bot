from code_agents.code_agents import agent_controls, chat_interface
import streamlit as st

if __name__ == "__main__":
    with st.sidebar:
        agent_controls()
    chat_interface()