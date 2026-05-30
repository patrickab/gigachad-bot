import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from streamlit_helper import apply_custom_css, init_session_state

PAGES = {
    "Workspaces": [
        st.Page("pages/Gigachad_Bot.py"),
        st.Page("pages/Deep_Research.py"),
        st.Page("pages/OCR_Workspace.py"),
    ],
}

if __name__ == "__main__":
    st.set_page_config(page_title="Gigachad-Bot", page_icon=":robot:", layout="wide")
    init_session_state()
    apply_custom_css()

    st.navigation(pages=PAGES, position="top").run()