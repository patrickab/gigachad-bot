import os
import subprocess

import streamlit as st

from src.config import DIRECTORY_RAG_INPUT, DIRECTORY_VLM_OUTPUT
from src.lib.streamlit_helper import editor


def markdown_preprocessor() -> None:
    """Markdown Preprocessor for Obsidian Notes."""
    _,center, _ = st.columns([1,8,1])
    with center:
        st.title("Markdown Preprocessor")

        vlm_output = os.listdir(DIRECTORY_VLM_OUTPUT)
        vlm_output = [d.split("converted_")[1] for d in vlm_output]
        vlm_output = [d.split(".pdf")[0] for d in vlm_output]

        for output in vlm_output:

            content_path = f"./{DIRECTORY_VLM_OUTPUT}/converted_{output}.pdf/{output}/auto"
            contents = os.listdir(content_path)
            md_filepath = next(f for f in contents if f.endswith(".md"))
            imgs_path = content_path + "/images"

            with st.expander(output):

                cols_spacer = st.columns([0.1,0.9])

                imgs = os.listdir(imgs_path)
    
                imsgs_paths = [f"{imgs_path}/{img}" for img in imgs]
                md_path = f"{content_path}/{md_filepath}"

                with open(md_path, "r") as f:
                    md_content = f.read()

                cols = st.columns(2)

                with cols[0]:
                    edited_text = editor(language="latex", text_to_edit=md_content, key=output)
                with cols[1]:
                    st.markdown(edited_text)

                with cols_spacer[0]:
                    if st.button("Save Changes", key=f"save_md_{output}"):
                        with open(md_path, "w") as f:
                            f.write(edited_text)
                        st.success(f"Saved changes to {md_filepath}")


if __name__ == "__main__":
    markdown_preprocessor()