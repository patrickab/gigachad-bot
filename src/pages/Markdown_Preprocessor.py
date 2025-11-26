"""
NOTE TO MY SELF: Every step in this pipeline shall seperate streamlit UI & data processing logic without user interaction.
Required for later building API endpoints for automation.
"""

import os
from pathlib import Path
import shutil

import polars as pl
from rag_database.rag_config import DatabaseKeys
import streamlit as st

from src.config import (
    DIRECTORY_MD_PREPROCESSING_1,
    DIRECTORY_RAG_INPUT,
    DIRECTORY_VLM_OUTPUT,
    SERVER_APP_RAG_INPUT,
)
from src.lib.streamlit_helper import editor


def init_session_state() -> None:
    """Initialize session state variables for the Markdown Preprocessor."""
    if "is_data_wrangling_needed" not in st.session_state:
        st.session_state.is_data_wrangling_needed = True
        st.session_state.parsed_outputs = []
        st.session_state.preprocessor_active = False
        st.session_state.chunker_active = False
        st.session_state.is_chunk_df_initialized = False

# ---------------------------- Preprocessing Step 1 - Move Paths / Fix Headings / Adjust MD Image Paths ---------------------------- #
def data_wrangler(vlm_output: list[str]) -> None:
    """
    Copies and processes VLM output files. For each file, it:
    1. Copies the markdown file and its images to DIRECTORY_MD_PREPROCESSING_1.
    2. Adjusts image paths in the copied markdown file to be server-accessible.
    3. Corrects heading levels in the copied file:
        - '# 1.2 Title' -> '## 1.2 Title'
        - '# Title'     -> '**Title**'
    """

    for output_name in vlm_output:

        # 1. Set up paths.
        content_path = Path(DIRECTORY_VLM_OUTPUT) / f"converted_{output_name}.pdf" / output_name / "auto"
        md_dest_path = Path(DIRECTORY_MD_PREPROCESSING_1) / output_name
        imgs_src_path = content_path / "images"
        app_imgs_dest_path = Path(SERVER_APP_RAG_INPUT) / output_name / "images" # app images need
        static_imgs_dest_path = Path(DIRECTORY_RAG_INPUT) / output_name / "images" # fileserver accessible images

        # Use glob to find the markdown file; more explicit and safer than list comprehension with next().
        try:
            md_file_path = next(content_path.glob("*.md"))
        except StopIteration:
            print(f"Warning: No markdown file found in {content_path} for {output_name}. Skipping.")
            continue

        md_dest_path.mkdir(parents=True, exist_ok=True)

        # Use shutil for platform-independent file operations.
        processed_md_filepath = md_dest_path / md_file_path.name
        shutil.copy(md_file_path, processed_md_filepath)
        if imgs_src_path.exists() and imgs_src_path.is_dir():
            shutil.copytree(imgs_src_path, static_imgs_dest_path, dirs_exist_ok=True) # 2. Copy folder

        content = processed_md_filepath.read_text(encoding="utf-8")
        content = content.replace("![](images", f"![]({app_imgs_dest_path}")
        lines = content.splitlines()
        processed_lines = []
        for line in lines:
            if line.startswith("# "):
                heading_content = line[2:].lstrip()
                first_word = heading_content.split(" ", 1)[0]
                numeric_part = first_word.rstrip(".")
                if numeric_part and numeric_part.replace(".", "").isdigit():
                    level = numeric_part.count(".") + 1
                    processed_lines.append(f"{'#' * min(level, 6)} {heading_content}")
                else:
                    processed_lines.append(f"**{heading_content.strip()}**")
            else:
                processed_lines.append(line)

        processed_md_filepath.write_text("\n".join(processed_lines), encoding="utf-8")


def markdown_preprocessor() -> None:
    """
    First level processing step:

    Markdown Preprocessor for Obsidian-Compatible Markdown Notes.
    1. Datawrangling
        - fixes heading levels
        - adjusts image paths to fileserver paths
        - moves VLM output files to 1st level MD preprocessing folder.
    2. Editor & Preview
        - Displays editor & markdown preview with images for each VLM output.
        - Allows to make manual adjustments before saving data for 2nd level preprocessing.
    """
    _,center, _ = st.columns([1,8,1])

    if st.sidebar.button("Exit Markdown Preprocessor"):
        st.session_state.preprocess_active = False
        return

    with center:
        st.title("Markdown Preprocessor")

        vlm_output = os.listdir(DIRECTORY_VLM_OUTPUT)
        vlm_output = [d.split("converted_")[1] for d in vlm_output]
        vlm_output = [d.split(".pdf")[0] for d in vlm_output]

        # Execute data wrangling only once per session
        if st.session_state.is_data_wrangling_needed is True:
            st.session_state.is_data_wrangling_needed = False
            data_wrangler(vlm_output)

        # Display editor & preview
        for output_name in vlm_output:

            output_path = Path(DIRECTORY_MD_PREPROCESSING_1) / output_name
            try:
                md_filepath = next(output_path.glob("*.md"))
            except StopIteration:
                st.warning(f"Markdown file for '{output_name}' not found. Skipping.")
                continue

            with open(md_filepath, "r") as f:
                md_content = f.read()

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


# ----------------------------- Preprocessing Step 2 - Chunking / Hierarchy / Parquet Storage ----------------------------- #
def parse_markdown_to_chunks(markdown_filepath: str) -> pl.DataFrame:
    """
    Helper function called in 2nd level markdown preprocessor
    Parses markdown text into hierarchical chunks based on heading levels.
    Each chunk is associated with its most specific heading (H1, H2, or H3).

    Returns a Polars DataFrame with columns for title, content, and metadata.
    """
    try:
        with open(markdown_filepath, "r", encoding="utf-8") as f:
            markdown_text = f.read()
    except FileNotFoundError:
        # Return an empty DataFrame with the correct schema if the file doesn't exist
        return pl.DataFrame({key: [] for key in [DatabaseKeys.KEY_TITLE, DatabaseKeys.KEY_TXT, DatabaseKeys.KEY_METADATA]})

    lines = markdown_text.split("\n")

    # State tracking
    current_h1 = "General"
    current_h2 = "General"
    current_h3 = "General"

    chunks = []
    current_buffer = []

    def save_chunk() -> None:
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

    if not chunks:
        return pl.DataFrame({key: [] for key in [DatabaseKeys.KEY_TITLE, DatabaseKeys.KEY_TXT, DatabaseKeys.KEY_METADATA]})

    # Create DataFrame directly from the list of dictionaries
    df = pl.DataFrame(chunks)

    # Rename columns to match the database schema
    df = df.rename({
        "title": DatabaseKeys.KEY_TITLE,
        "content": DatabaseKeys.KEY_TXT,
        "metadata": DatabaseKeys.KEY_METADATA
    })

    return df.select(DatabaseKeys.KEY_TITLE, DatabaseKeys.KEY_TXT, DatabaseKeys.KEY_METADATA)

def render_chunks(output_name: str) -> None:
    """
    Renders an interactive editor for a DataFrame of chunks stored in st.session_state.
    Modifications (save/delete) are performed directly on the DataFrame in session state.
    """
    # Get the DataFrame from session state
    chunks_df = st.session_state.chunk_dfs[output_name]

    # Ensure a unique, stable row identifier exists for editing/deleting
    if "chunk_id" not in chunks_df.columns:
        chunks_df = chunks_df.with_row_index("chunk_id")
        st.session_state.chunk_dfs[output_name] = chunks_df

    def render_chunk(row: dict) -> None:
        """Renders a single chunk (DataFrame row) with editing capabilities."""
        unique_key = f"{output_name}_{row['chunk_id']}"

        # Buttons to toggle between viewing and editing mode
        toggle_cols = st.columns([1, 1, 8])
        action_cols = st.columns([1, 1, 8])
        if toggle_cols[0].button("Edit Chunk", key=f"edit_md_chunker_{unique_key}"):
            st.session_state[f"edit_mode_{unique_key}"] = True
        if toggle_cols[1].button("Close Editor", key=f"close_md_chunker_{unique_key}"):
            st.session_state[f"edit_mode_{unique_key}"] = False

        # In edit mode, show the editor; otherwise, show the rendered markdown.
        if st.session_state.get(f"edit_mode_{unique_key}", False): # In edit mode
            cols_text = st.columns([1, 1])
            with cols_text[0]:
                editor_key = f"editor_{output_name}_{row[DatabaseKeys.KEY_TITLE]}_{row['chunk_id']}"
                edited_text = editor(language="latex", text_to_edit=row[DatabaseKeys.KEY_TXT], key=editor_key)

                if action_cols[0].button("Save Chunk Changes", key=f"save_md_chunker_{unique_key}"):
                    current_df = st.session_state.chunk_dfs[output_name]
                    updated_df = current_df.with_columns(
                        pl.when(pl.col("chunk_id") == row['chunk_id'])
                        .then(pl.lit(edited_text))
                        .otherwise(pl.col(DatabaseKeys.KEY_TXT))
                        .alias(DatabaseKeys.KEY_TXT)
                    )
                    st.session_state.chunk_dfs[output_name] = updated_df
                    st.session_state[f"edit_mode_{unique_key}"] = False # Exit edit mode
                    st.rerun()
                if action_cols[1].button("Delete Chunk", key=f"delete_md_chunker_{unique_key}"):
                    current_df = st.session_state.chunk_dfs[output_name]
                    updated_df = current_df.filter(pl.col("chunk_id") != row['chunk_id'])
                    st.session_state.chunk_dfs[output_name] = updated_df
                    st.rerun()

            with cols_text[1]:
                st.markdown(edited_text)
        else: # In view mode
            st.markdown(row[DatabaseKeys.KEY_TXT])

    # Hierarchy rendering using DataFrame filters
    level_1_df = chunks_df.filter(pl.col(DatabaseKeys.KEY_METADATA).struct.field("level") == 1)
    level_2_df = chunks_df.filter(pl.col(DatabaseKeys.KEY_METADATA).struct.field("level") == 2)
    level_3_df = chunks_df.filter(pl.col(DatabaseKeys.KEY_METADATA).struct.field("level") == 3)

    for l1_row in level_1_df.to_dicts():
        with st.expander(f"{l1_row[DatabaseKeys.KEY_TITLE]}"):
            render_chunk(l1_row)
            l2_children = level_2_df.filter(pl.col(DatabaseKeys.KEY_METADATA).struct.field("h1") == l1_row[DatabaseKeys.KEY_TITLE])
            for l2_row in l2_children.to_dicts():
                with st.expander(f"{l2_row[DatabaseKeys.KEY_TITLE]}"):
                    render_chunk(l2_row)
                    l3_children = level_3_df.filter(pl.col(DatabaseKeys.KEY_METADATA).struct.field("h2") == l2_row[DatabaseKeys.KEY_TITLE])
                    for l3_row in l3_children.to_dicts():
                        with st.expander(f"{l3_row[DatabaseKeys.KEY_TITLE]}"):
                            render_chunk(l3_row)

def markdown_chunker() -> None:
    """
    Second level processing step:
    1. Inspect preprocessed markdown chunks
        - allows manual editing
    2. Render hierarchy
    3. Store chunks to Parquet files for RAG ingestion.
    """

    if st.sidebar.button("Exit Markdown Chunker"):
        st.session_state.chunker_active = False
        return

    _, center, _ = st.columns([1, 8, 1])
    directory_preprocessed_output = os.listdir(DIRECTORY_MD_PREPROCESSING_1)

    # Initialize session state for holding DataFrames
    if 'chunk_dfs' not in st.session_state:
        st.session_state.chunk_dfs = {}

    with center:
        for output_name in directory_preprocessed_output:
            md_filepath = f"{DIRECTORY_MD_PREPROCESSING_1}/{output_name}/{output_name}.md"

            # Parse and store DataFrame in session state on first run for this file
            if not st.session_state.is_chunk_df_initialized:
                st.session_state.chunk_dfs[output_name] = parse_markdown_to_chunks(md_filepath)
                st.session_state.is_chunk_df_initialized = True

            with st.expander(output_name):
                if st.button("Store chunks to Parquet", key=f"store_md_chunks_{output_name}", type="primary"):
                    # On button click, write the current state of the DataFrame to Parquet
                    # Drop the temporary 'chunk_id' if it exists before saving
                    df_to_save = st.session_state.chunk_dfs[output_name]
                    if "chunk_id" in df_to_save.columns:
                        df_to_save = df_to_save.drop("chunk_id")
                    df_to_save.write_parquet(f"{DIRECTORY_RAG_INPUT}/{output_name}/chunked_{output_name}.parquet")

                # Render the interactive chunk editor, which operates on session_state directly
                render_chunks(output_name=output_name)

if __name__ == "__main__":

    init_session_state()
    selection = st.sidebar.radio("Select Page", options=["Markdown Preprocessor", "Markdown Chunker"], index=0, key="markdown_page_selector")  # noqa

    _, center, _ = st.columns([1, 8, 1])
    with center:
        if selection == "Markdown Preprocessor":

            if st.sidebar.button("Initialize Preprocessor"):
                st.session_state.preprocessor_active = True

            if st.session_state.preprocessor_active is True:
                markdown_preprocessor()

        elif selection == "Markdown Chunker":

            if st.sidebar.button("Initialize Chunker"):
                st.session_state.chunker_active = True

            if st.session_state.chunker_active is True:
                markdown_chunker()