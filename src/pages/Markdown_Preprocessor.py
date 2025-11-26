import os
import re
import subprocess

import polars as pl
from rag_database.rag_config import DatabaseKeys
import streamlit as st

from src.config import DIRECTORY_RAG_INPUT, DIRECTORY_VLM_OUTPUT, SERVER_APP_RAG_INPUT
from src.lib.streamlit_helper import editor


def init_session_state() -> None:
    """Initialize session state variables for the Markdown Preprocessor."""
    if "edited_markdown_files" not in st.session_state:
        st.session_state.moved_outputs = []
        st.session_state.parsed_outputs = []

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
                # # Text -> no dots -> no heading
                line = line.replace('#', '', 1)
                # remove leading space & trailing \n
                line = line.lstrip().rstrip('\n') + '\n'
                line = "**" + line + "**"  # Bold the text instead
                outfile.write(line)
        else:
            # Not a heading line
            outfile.write(line)

def data_wrangler(vlm_output: list[str]) -> None:
    """
    1. Move VLM output_name files to RAG input directory.
    2. Fix image paths in markdown files.
    3. Fix heading levels in markdown files. 
        - # x.x -> ## x.x
        - # x.x.x -> ### x.x.x
    """
    for output_name in vlm_output:
        # Construct paths
        content_path = f"./{DIRECTORY_VLM_OUTPUT}/converted_{output_name}.pdf/{output_name}/auto"
        contents = os.listdir(content_path)

        # Identify file locations & copy to RAG input directory
        md_file = next(f for f in contents if f.endswith(".md"))
        md_filepath = f"{content_path}/{md_file}"
        imgs_path = content_path + "/images"
        os.makedirs(f"{DIRECTORY_RAG_INPUT}/{output_name}", exist_ok=True)
        subprocess.run(["cp", "-r", md_filepath, imgs_path, f"./{DIRECTORY_RAG_INPUT}/{output_name}"], check=True)

        # Convert ![](/images/<img-filename>) to ![](){DIRECTORY_RAG_INPUT/images/<img-filename>} image paths
        with open(f"{DIRECTORY_RAG_INPUT}/{output_name}/{md_file}", "r") as f:
            md_content = f.read()
            md_content = md_content.replace("![](images", f"![]({SERVER_APP_RAG_INPUT}/{output_name}/images")
            with open(f"{DIRECTORY_RAG_INPUT}/{output_name}/{md_file}", "w") as f:
                f.write(md_content)

        # Create a temporary file to write the fixed content
        temp_filepath = f"{content_path}/temp_{md_file}"

        with open(md_filepath, "r") as infile, open(temp_filepath, "w") as outfile:
            fix_heading_levels(infile, outfile)

        # Replace original file with fixed file
        os.replace(temp_filepath, md_filepath)
        print(f"Moved and processed files for {output_name}")

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
        for output_name in vlm_output:

            # Update md_filepath to new location
            contents = os.listdir(f"{DIRECTORY_RAG_INPUT}/{output_name}")
            md_file = next(f for f in contents if f.endswith(".md"))
            md_filepath = f"{DIRECTORY_RAG_INPUT}/{output_name}/{md_file}"

            with open(md_filepath, "r") as f:
                md_content = f.read()
                md_content = md_content[9000:] # Ignore first 9000 chars (bloat)

            with st.expander(output_name):

                cols_spacer = st.columns([0.1,0.9])

                cols = st.columns(2)

                with cols[0]:
                    edited_text = editor(language="latex", text_to_edit=md_content, key=output_name)
                with cols[1]:
                    st.markdown(edited_text)

                with cols_spacer[0]:
                    if st.button("Save Changes", key=f"save_md_preprocessor_{output_name}"):
                        # Replace edited content back to file
                        with open(md_filepath, "w") as f:
                            f.write(edited_text)
                        st.success(f"Saved changes to {md_filepath}")

def parse_markdown_to_chunks(markdown_text: str) -> list[dict]:
    lines = markdown_text.split('\n')

    # State tracking
    current_h1 = "General"
    current_h2 = "General"
    current_h3 = "General"

    chunks = []
    current_buffer = []

    def save_chunk() -> None:
        if not current_buffer:
            return

        text_content = "\n".join(current_buffer).strip()
        if not text_content:
            return

        # Determine the most specific title for the chunk
        if current_h3 != "General":
            title = current_h3
        elif current_h2 != "General":
            title = current_h2
        else:
            title = current_h1

        # Create the comprehensive context string for the embedding
        context_string = f"{current_h1} > {current_h2} > {current_h3}"

        # This is the object you send to your embedding function
        chunk_record = {
            "content": text_content,
            "title": title,
            "metadata": {
                "level": 3 if current_h3 != "General" else 2 if current_h2 != "General" else 1,
                "h1": current_h1,
                "h2": current_h2,
                "h3": current_h3,
                "context_path": context_string
            },
            # "embedding_text": f"{context_string}\n\n{text_content}" # Optional: Prepend context for better vectors
        }
        chunks.append(chunk_record)

    for line in lines:
        # Detect Headers
        if line.startswith("# "):
            save_chunk() # Save whatever we had before this new chapter
            current_buffer = [] 
            current_h1 = line.strip().replace("# ", "")
            current_h2 = "General" # Reset lower levels
            current_h3 = "General"

        elif line.startswith("## "):
            save_chunk()
            current_buffer = []
            current_h2 = line.strip().replace("## ", "")
            current_h3 = "General" # Reset lower levels

        elif line.startswith("### "):
            save_chunk()
            current_buffer = []
            current_h3 = line.strip().replace("### ", "")
        else:
            current_buffer.append(line)

    # Save the final buffer
    save_chunk()

    return chunks

def markdown_chunker() -> None:
    """Inspect preprocessed markdown chunks."""

    def render_chunks(chunks: list[dict], output_name:str) -> None:

        def render_chunk(chunks: list[dict], chunk: dict, output_name:str, i: str) -> None:
            cols_buttons = st.columns([1,1,8])
            cols_text = st.columns([1,1])

            with cols_text[0]:
                edited_text = editor(language="latex", text_to_edit=chunk['content'], key=f"editor_{output_name}_{i}") # noqa
            with cols_text[1]:
                st.markdown(edited_text)

            if cols_buttons[0].button("Save Chunk Changes", key=f"save_md_chunker_{output_name}_{i}"):
                chunks[i-1]['content'] = edited_text
                st.success(f"Saved changes to {md_filepath}")
                st.rerun()
            if cols_buttons[1].button("Delete Chunk", key=f"delete_md_chunker_{output_name}_{i}"):
                chunks.remove(chunk)
                st.rerun()
    
            return chunks, chunk

        level_1 = 1
        for i, chunk in enumerate([c for c in chunks if c["metadata"]["level"] == level_1], start=1):
            with st.expander(f"{chunk['title']}"):
                chunks, chunk = render_chunk(chunks, chunk, output_name=output_name, i=f"{level_1}_{i}")
                # while consecutive cunks of lower levels exist, render them too
                level_2 = 2
                for j, chunk in enumerate([c for c in chunks if c["metadata"]["level"] == level_2 and c["metadata"]["h1"] == chunk["metadata"]["h1"]], start=1): # noqa
                    with st.expander(f"{chunk['title']}"):
                        chunks, chunk = render_chunk(chunks, chunk, output_name=output_name, i=f"{chunk["title"]}")
                        level_3 = 3
                        for k, chunk in enumerate([c for c in chunks if c["metadata"]["level"] == level_3 and c["metadata"]["h2"] == chunk["metadata"]["h2"]], start=1): # noqa
                            with st.expander(f"{chunk['title']}"):
                                chunks, chunk = render_chunk(chunks, chunk, output_name=output_name, i=f"{chunk["title"]}") # noqa

        return chunks

    _,center, _ = st.columns([1,8,1])

    directory_preprocessed_output = os.listdir(DIRECTORY_RAG_INPUT)

    with center:

        for output_name in directory_preprocessed_output:

            md_filepath = f"{DIRECTORY_RAG_INPUT}/{output_name}/{output_name}.md"

            if output_name not in st.session_state.parsed_outputs:
                with open(md_filepath, "r") as f:
                    md_content = f.read()
                    chunks = parse_markdown_to_chunks(md_content)
                st.session_state.parsed_outputs.append(output_name)

            with st.expander(output_name):

                if st.button("Store chunks to Parquet", key=f"store_md_chunks_{output_name}", type="primary"):
                    titles = [chunk['title'] for chunk in chunks]
                    contents = [chunk['content'] for chunk in chunks]
                    metadata = [chunk['metadata'] for chunk in chunks]

                    chunked_dataframe = pl.DataFrame(

                        {
                            DatabaseKeys.KEY_TITLE: titles,
                            DatabaseKeys.KEY_TXT: contents,
                            DatabaseKeys.KEY_METADATA: metadata,
                        }
                    )
                    chunked_dataframe.write_parquet(f"{DIRECTORY_RAG_INPUT}/{output_name}/chunked_{output_name}.parquet")


                chunks = render_chunks(chunks=chunks, output_name=output_name)

if __name__ == "__main__":
    init_session_state()
    selection = st.sidebar.radio("Select Page", options=["Markdown Preprocessor", "Markdown Chunker", "Fufu"], index=0, key="markdown_page_selector")  # noqa
    if selection == "Markdown Preprocessor":
        markdown_preprocessor()
    elif selection == "Markdown Chunker":
        markdown_chunker()