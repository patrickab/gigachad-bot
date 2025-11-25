import os
import re
import subprocess

import streamlit as st

from src.config import DIRECTORY_RAG_INPUT, DIRECTORY_VLM_OUTPUT, SERVER_APP_RAG_INPUT
from src.lib.streamlit_helper import editor


def init_session_state() -> None:
    """Initialize session state variables for the Markdown Preprocessor."""
    if "edited_markdown_files" not in st.session_state:
        st.session_state.moved_outputs = []

def fix_heading_levels(infile: iter, outfile: iter) -> None:
    """
    Streams file line-by-line to maintain O(1) memory usage regardless of file size.
    Uses regex to robustly identify variable-depth decimal numbering (e.g., '1.2.3')
    that simple string splitting cannot reliably distinguish from heading text.
    """    
    # Regex explanation:
    # ^#\s+          : Matches a line starting with one hash and whitespace
    # (?P<nums>...)  : Captures the numbering group
    # \d+            : Starts with digits
    # (?:\.\d+)*     : Followed by zero or more groups of (.digits)

    # Rest of line is purposefully not matched for efficiency.
    # Only numbering pattern at linestart is relevant.
    pattern = re.compile(r'^#\s+(?P<nums>\d+(?:\.\d+)*)')

    for line in infile:
        # Only check lines that actually start with "# " to save regex time
        if line.startswith('# '):
            match = pattern.match(line)
            if match:
                numbering = match.group('nums')

                # Calculate level based on dots. 
                # "1" -> 0 dots -> Level 1
                # "1.1" -> 1 dot -> Level 2
                # "1.1.1" -> 2 dots -> Level 3
                # We strip trailing dots just in case "1.1." appears
                level = numbering.strip('.').count('.') + 1

                # Cap the level at 6 (Standard Markdown limit), 
                # though you can remove this if your specific dialect supports deeper nesting.
                level = min(level, 6)

                # Replace the single '#' with the correct number of '#'
                # We use line[1:] to keep the original spacing and text
                new_line = ('#' * level) + line[1:]
                outfile.write(new_line)
            else:
                # Starts with # but no number pattern found (e.g., "# Introduction")
                outfile.write(line)
        else:
            # Not a heading line
            outfile.write(line)

def data_wrangler(vlm_output: list[str]) -> None:
    """
    1. Move VLM output files to RAG input directory.
    2. Fix image paths in markdown files.
    3. Fix heading levels in markdown files. 
        - # x.x -> ## x.x
        - # x.x.x -> ### x.x.x
    """
    for output in vlm_output:
        # Construct paths
        content_path = f"./{DIRECTORY_VLM_OUTPUT}/converted_{output}.pdf/{output}/auto"
        contents = os.listdir(content_path)

        # Identify file locations & copy to RAG input directory
        md_file = next(f for f in contents if f.endswith(".md"))
        md_filepath = f"{content_path}/{md_file}"
        imgs_path = content_path + "/images"
        os.makedirs(f"{DIRECTORY_RAG_INPUT}/{output}", exist_ok=True)
        subprocess.run(["cp", "-r", md_filepath, imgs_path, f"./{DIRECTORY_RAG_INPUT}/{output}"], check=True)

        # Convert ![](/images/<img-filename>) to ![](){DIRECTORY_RAG_INPUT/images/<img-filename>} image paths
        with open(f"{DIRECTORY_RAG_INPUT}/{output}/{md_file}", "r") as f:
            md_content = f.read()
            md_content = md_content.replace("![](images", f"![]({SERVER_APP_RAG_INPUT}/{output}/images")
            with open(f"{DIRECTORY_RAG_INPUT}/{output}/{md_file}", "w") as f:
                f.write(md_content)

        # Create a temporary file to write the fixed content
        temp_filepath = f"{content_path}/temp_{md_file}"

        with open(md_filepath, "r") as infile, open(temp_filepath, "w") as outfile:
            fix_heading_levels(infile, outfile)

        # Replace original file with fixed file
        os.replace(temp_filepath, md_filepath)
        print(f"Moved and processed files for {output}")

def markdown_preprocessor() -> None:
    """Markdown Preprocessor for Obsidian Notes."""
    _,center, _ = st.columns([1,8,1])
    with center:
        st.title("Markdown Preprocessor")

        vlm_output = os.listdir(DIRECTORY_VLM_OUTPUT)
        vlm_output = [d.split("converted_")[1] for d in vlm_output]
        vlm_output = [d.split(".pdf")[0] for d in vlm_output]

        # Move files only once per session
        if st.session_state.moved_outputs == []:
            data_wrangler(vlm_output)

        # Display editor & preview
        for output in vlm_output:

            # Update md_filepath to new location
            contents = os.listdir(f"{DIRECTORY_RAG_INPUT}/{output}")
            md_file = next(f for f in contents if f.endswith(".md"))
            md_filepath = f"{DIRECTORY_RAG_INPUT}/{output}/{md_file}"    

            with open(md_filepath, "r") as f:
                md_content = f.read()

            with st.expander(output):

                cols_spacer = st.columns([0.1,0.9])

                cols = st.columns(2)

                with cols[0]:
                    edited_text = editor(language="latex", text_to_edit=md_content, key=output)
                with cols[1]:
                    st.markdown(edited_text)

                with cols_spacer[0]:
                    if st.button("Save Changes", key=f"save_md_{output}"):
                        with open(md_filepath, "w") as f:
                            f.write(edited_text)
                        st.success(f"Saved changes to {md_filepath}")


if __name__ == "__main__":
    init_session_state()
    markdown_preprocessor()
