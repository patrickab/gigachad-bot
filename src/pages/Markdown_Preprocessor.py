import os

import streamlit as st

from src.config import DIRECTORY_VLM_OUTPUT
from src.lib.streamlit_helper import editor


def markdown_preprocessor() -> None:
    """
    Markdown Preprocessor for Obsidian Notes.
    """
    st.title("Markdown Preprocessor")
    vlm_output = os.listdir(DIRECTORY_VLM_OUTPUT)
    vlm_output = [d.split("converted_")[1] for d in vlm_output]
    vlm_output = [d.split(".pdf")[0] for d in vlm_output]
    for output in vlm_output:
        with st.expander(output):
            content_path = f"./{DIRECTORY_VLM_OUTPUT}/converted_{output}.pdf/{output}/auto"
            contents = os.listdir(content_path)
            md_file = next(f for f in contents if f.endswith(".md"))

            imgs_path = content_path + "/images"
            imgs = os.listdir(imgs_path)
    
            imsgs_paths = [f"{imgs_path}/{img}" for img in imgs]
            md_path = f"{content_path}/{md_file}"

            with open(md_path, "r") as f:
                md_content = f.read()

            cols = st.columns(2)

            with cols[0]:
                edited_text = editor(language="latex", text_to_edit=md_content, key=output)
                edited_text
            with cols[1]:
                st.markdown(edited_text)

if __name__ == "__main__":
    markdown_preprocessor()