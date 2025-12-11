from datetime import datetime
import json
import re

import pandas as pd
import streamlit as st

from lib.flashcards import DATE_ADDED, NEXT_APPEARANCE, render_flashcards
from lib.non_user_prompts import (
    SYS_LEARNINGGOALS_TO_FLASHCARDS,
    SYS_PDF_TO_LEARNING_GOALS,
)
from lib.streamlit_helper import _extract_text_from_pdf, _non_streaming_api_query, options_message
from llm_config import MACROTASK_MODEL


@st.cache_data
def _generate_learning_goals(text: str) -> str:
    """Generate learning goals from PDF text."""
    print("Generating learning goals...")
    return _non_streaming_api_query(model=MACROTASK_MODEL, prompt=text, system_prompt=SYS_PDF_TO_LEARNING_GOALS)

@st.cache_data
def _generate_flashcards(learning_goals: str) -> pd.DataFrame:
    """Generate flashcards from learning goals."""
    print("Generating flashcards...")
    response = _non_streaming_api_query(
        model=MACROTASK_MODEL,
        prompt=learning_goals,
        system_prompt=SYS_LEARNINGGOALS_TO_FLASHCARDS,
    )
    response = response.split("```json")[ -1].split("```")[0]  # Clean up response if necessary
    flashcards = json.loads(response)
    df_flashcards = pd.DataFrame(flashcards)
    df_flashcards[DATE_ADDED] = datetime.now()
    df_flashcards[NEXT_APPEARANCE] = datetime.now()
    return df_flashcards

@st.cache_data
def _generate_wiki_article(pdf_text: str, learning_goals: str) -> str:  # noqa
    # Currently unused - keep structure for future use
    print("Writing wiki article...")
    wiki_prompt = f"""
    Consider the following learning goals:
    
    {learning_goals}
    
    Dynamically adjust depth of explanation to the provided Bloom's taxonomy tags.
    Use the hierarchy of learning goals to structure the article.
    Generate a comprehensive study article based on the following PDF content.

    {pdf_text}
    """

    #return _non_streaming_api_query(model=MACROTASK_MODEL, prompt=wiki_prompt, system_prompt=SYS_PDF_TO_ARTICLE)


def pdf_workspace() -> None:
    """PDF Workspace for extracting learning goals and summary articles."""

    tab_pdf, tab_summary, tab_flashcards = st.tabs(["PDF Viewer/Uploader", "PDF Summary", "PDF Flashcards"])

    with tab_pdf:

        with st.popover("Options"):
            file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_workspace_uploader")

        if file is not None:
            pdf_text, pdf_height = _extract_text_from_pdf(file)
            learning_goals = _generate_learning_goals(pdf_text)

            col_learning_goals, col_pdf = st.columns([0.5,0.5])

            with col_learning_goals:
                st.header("Learning Goals")

                if file is not None and learning_goals:

                    parts = re.split(r'(?m)^\#\s*(.*)\s*$', learning_goals)  # -> [before, h1, c1, h2, c2, ...]
                    for title, content in zip(parts[1::2], parts[2::2], strict=True):
                        if content == "\n": # Empty content = title of pdf without learning goals
                            st.markdown(f"##{title.strip()}")
                        else: # Actual learning goal section
                            with st.expander(title.strip()):
                                if content.strip():
                                    st.markdown(content.strip()) # noqa

                options_message(assistant_message=learning_goals, button_key="pdf_learning_goals") if file is not None else None

            with col_pdf:
                st.header("Original PDF")
                st.pdf(file, height=pdf_height) if file is not None else None

    with tab_summary:
        if file is not None:
            button = st.button("Generate Summary Article")
            if button:
                wiki_article = _generate_wiki_article(pdf_text=pdf_text, learning_goals=learning_goals)
                st.markdown(wiki_article if file is not None else "")
                options_message(assistant_message=wiki_article, button_key="pdf_wiki_article") if file is not None else None
            else:
                st.info("Click the button to generate the summary article.")
        else:
            st.info("Upload a PDF in the 'PDF Viewer/Uploader' tab to generate a summary article.")

    with tab_flashcards:
        if file is not None:
            button = st.button("Generate Flashcards", key="generate_flashcards_button")
            if button:
                flashcards_df = _generate_flashcards(learning_goals)
                render_flashcards(flashcards_df)
            else:
                st.info("Click the button to generate flashcards.")
        else:
            st.info("Upload a PDF in the 'PDF Viewer/Uploader' tab to generate flashcards.")

if __name__ == "__main__":
    st.set_page_config(page_title="PDF Workspace", page_icon=":robot:", layout="wide")
    pdf_workspace()