import json
import os
import pathlib
from pathlib import Path
import re
import shutil

import polars as pl
from rag_database.dataclasses import RAGIngestionPayload
from rag_database.rag_config import DatabaseKeys
import streamlit as st

from lib.non_user_prompts import SYS_LECTURE_ENHENCER, SYS_LECTURE_SUMMARIZER
from src.config import (
    DIRECTORY_LLM_PREPROCESSING,
    DIRECTORY_MD_PREPROCESSING,
    DIRECTORY_RAG_INPUT,
    DIRECTORY_VLM_OUTPUT,
    SERVER_APP_RAG_INPUT,
)
from src.lib.streamlit_helper import editor, llm_params_sidebar, model_selector


def init_session_state() -> None:
    """Initialize session state variables for the Markdown Preprocessor."""
    if "is_data_wrangling_needed" not in st.session_state:
        st.session_state.is_data_wrangling_needed = True
        st.session_state.parsed_outputs = []

        st.session_state.preprocessor_active = False
        st.session_state.llm_preprocessor_active = False
        st.session_state.chunker_active = False

        st.session_state.is_doc_edit_mode_active = {} # per document
        st.session_state.is_chunk_edit_mode_active = {} # per chunk
        st.session_state.staging_complete = False
        st.session_state.is_payload_initialized = {} # boolean per document
        st.session_state.rag_ingestion_payload = {}

# ---------------------------- Preprocessing Step 1 - Move Paths / Fix Headings / Adjust MD Image Paths ---------------------------- #
def _transform_headings(lines: list[str]) -> list[str]:
    """
    Rewrites markdown heading lines based on specific formatting rules.
    - Converts numeric-prefixed H1s (e.g., '# 1.2 Title') to their correct level ('## 1.2 Title').
    - Converts non-numeric H1s (e.g., '# Conclusion') to bolded text.
    """
    processed_lines = []
    for line in lines:
        if not line.startswith("# "):
            processed_lines.append(line)
            continue

        heading_content = line[2:].strip()
        first_word = heading_content.split(" ", 1)[0]
        numeric_part = first_word.rstrip(".")

        if numeric_part.replace(".", "").isdigit():
            level = numeric_part.count(".") + 1
            processed_lines.append(f"{'#' * min(level, 6)} {heading_content}")
        else:
            processed_lines.append(f"**{heading_content}**")
    return processed_lines

IMAGE_PATH_PATTERN_SERVER = re.compile(r"!\[(.*?)\]\(images/") # used for mapping image paths to server URLs

def _preprocess_document(doc_id: str) -> None:
    """Copies, cleans, and restructures a single document and its assets."""
    try:
        source_base_path = Path(DIRECTORY_VLM_OUTPUT) / doc_id / doc_id / "auto"
        dest_base_path = Path(DIRECTORY_MD_PREPROCESSING) / doc_id
        source_imgs_path = source_base_path / "images"
        static_imgs_dest_path = Path(DIRECTORY_RAG_INPUT) / doc_id / "images"
        server_img_url_path = Path(SERVER_APP_RAG_INPUT) / doc_id / "images"

        try:
            source_md_path = next(source_base_path.glob("*.md"))
        except StopIteration:
            st.warning(f"No markdown file found for '{doc_id}'. Skipping.")
            return

        dest_base_path.mkdir(parents=True, exist_ok=True)
        dest_md_path = dest_base_path / source_md_path.name
        shutil.copy(source_md_path, dest_md_path)
        shutil.copytree(source_imgs_path, static_imgs_dest_path, dirs_exist_ok=True)

        original_content = dest_md_path.read_text(encoding="utf-8")
        content_with_updated_img_paths = IMAGE_PATH_PATTERN_SERVER.sub(rf"![\1]({server_img_url_path}/", original_content)

        lines = content_with_updated_img_paths.splitlines()
        processed_lines = _transform_headings(lines)
        final_content = "\n".join(processed_lines)
        dest_md_path.write_text(final_content, encoding="utf-8")

        if static_imgs_dest_path.is_dir():
            referenced_images = {Path(url).name for url in IMAGE_LINK_PATTERN_MD.findall(final_content)}
            referenced_images = {img_name.rstrip(')') for img_name in referenced_images}
            for img_file in static_imgs_dest_path.iterdir():
                if img_file.is_file() and img_file.name not in referenced_images:
                    img_file.unlink()


    except (IOError, OSError) as e:
        st.error(f"Failed to process '{doc_id}': {e}")

def _get_doc_ids(source_directory: str) -> list[str]:
    """Retrieves document IDs from the VLM output directory."""
    source_path = Path(source_directory)
    if not source_path.is_dir():
        return []
    doc_ids = [path.name for path in source_path.glob("*") if path.is_dir() and path.name != "archive"]
    doc_ids = sorted(doc_ids)
    return doc_ids

def stage_vlm_outputs(source_directory: str) -> None:
    """
    Copies and processes all VLM output files for preprocessing.

    For each document, this function:
    1. Copies the markdown file and its images to the preprocessing directory.
    2. Adjusts image paths within the markdown to be server-accessible URLs.
    3. Transforms markdown headings to a consistent, hierarchical format.
    """
    doc_ids = _get_doc_ids(source_directory)

    for doc_id in doc_ids:
        _preprocess_document(doc_id)
        st.session_state.is_doc_edit_mode_active[doc_id] = False

IMAGE_LINK_PATTERN_MD = re.compile(r"!\[.*?\]\s*\((?:.*?)\)") # used for extracting complete markdown image links

def _render_document_editor(doc_id: str, base_path: Path) -> None:
    """Displays a markdown editor and preview for a single document."""
    try:
        md_filepath = next(base_path.glob("*.md"))
    except StopIteration:
        st.warning(f"Markdown file for '{doc_id}' not found. Skipping.")
        return

    # Load the original content once per render
    if f"md_content_{doc_id}" not in st.session_state:
        st.session_state[f"md_content_{doc_id}"] = md_filepath.read_text(encoding="utf-8")
        st.session_state[f"md_images_{doc_id}"] = IMAGE_LINK_PATTERN_MD.findall(st.session_state[f"md_content_{doc_id}"])
        # Dynamically determine zero-padding based on the total number of images
        st.session_state[f"padding_width_{doc_id}"] = len(str(len(st.session_state[f"md_images_{doc_id}"])))
        st.session_state[f"n_kept_images_{doc_id}"] = 0

    selection = st.radio("Select Mode", options=["Image Review", "Document Edit/Preview"], key=f"doc_mode_{doc_id}")
    original_content = st.session_state[f"md_content_{doc_id}"]

    if selection == "Image Review":
        md_images = st.session_state[f"md_images_{doc_id}"]
        static_imgs_dest_path = Path(DIRECTORY_RAG_INPUT) / doc_id / "images"

        if md_images:
            col_buttons = st.columns([1,1,8])
            with col_buttons[0]:
                if st.button("Keep Image", key=f"store_img_{doc_id}"):

                    # 1. Increment counter and get current image link
                    st.session_state[f"n_kept_images_{doc_id}"] += 1
                    n_kept = st.session_state[f"n_kept_images_{doc_id}"]
                    padding = st.session_state[f"padding_width_{doc_id}"]
                    current_md_link = st.session_state[f"md_images_{doc_id}"].pop(0)

                    # 2. Extract path, generate new filename, rename file, and update content
                    if match := re.search(r"\((.*?)\)", current_md_link):
                        old_path = Path(match.group(1))
                        new_filename = f"{n_kept:0{padding}}{old_path.suffix}"

                        (static_imgs_dest_path / old_path.name).rename(static_imgs_dest_path / new_filename)

                        new_md_link = current_md_link.replace(old_path.name, new_filename)
                        st.session_state[f"md_content_{doc_id}"] = st.session_state[f"md_content_{doc_id}"].replace(current_md_link, new_md_link) # noqa

                    md_filepath.write_text(st.session_state[f"md_content_{doc_id}"], encoding="utf-8")
                    st.rerun()

            with col_buttons[1]:
                if st.button("Delete Image", key=f"delete_img_{doc_id}"):

                    # Remove image from the original content & list
                    st.session_state[f"md_content_{doc_id}"] = original_content.replace(md_images[0], "<removed image>")
                    img = st.session_state[f"md_images_{doc_id}"].pop(0)

                    # Also delete the image file from disk
                    if match := re.search(r"\((.*?)\)", img):
                        img_filename = Path(match.group(1)).name
                        (static_imgs_dest_path / img_filename).unlink(missing_ok=True)

                    md_filepath.write_text(st.session_state[f"md_content_{doc_id}"], encoding="utf-8")
                    st.rerun()

            st.markdown(md_images[0])

        else:
            st.info("No images found or all images processed.")
            md_images = IMAGE_LINK_PATTERN_MD.findall(st.session_state[f"md_content_{doc_id}"])


    if selection == "Document Edit/Preview":

        if not st.session_state.is_doc_edit_mode_active[doc_id]:

            if st.button("Edit Document", key=f"edit_{doc_id}"):
                st.session_state.is_doc_edit_mode_active[doc_id] = True
                st.rerun()

            st.subheader("Preview")
            st.markdown(original_content, unsafe_allow_html=True)

        else:

            button_col = st.columns([1, 8])

            st.subheader("Editor")
            text_cols = st.columns([1,1])
            with text_cols[0]:
                st.session_state[f"md_content_{doc_id}"] = editor(
                    text_to_edit=original_content,
                    language="markdown",
                    key=f"editor_{doc_id}"
                )
            with text_cols[1]:
                st.subheader("Preview")
                st.markdown(st.session_state[f"md_content_{doc_id}"], unsafe_allow_html=True)

            # Check for changes and save automatically
            if button_col[0].button("Exit Edit Mode", key=f"view_{doc_id}"):
                md_filepath.write_text(st.session_state[f"md_content_{doc_id}"], encoding="utf-8")
                st.session_state.is_doc_edit_mode_active[doc_id] = False

def render_preprocessor() -> None:
    """
    Renders the UI for the first-level markdown preprocessing step.

    This page allows users to trigger a one-time data staging process that
    copies and standardizes VLM outputs, then manually review, edit, and save
    each document before it proceeds to the next stage.
    """
    st.title("Markdown Preprocessing & Review")

    if st.sidebar.button("Exit Preprocessor"):
        st.session_state.preprocess_active = False
        st.rerun()

    if not st.session_state.staging_complete:
        stage_vlm_outputs(DIRECTORY_VLM_OUTPUT)
        st.session_state.staging_complete = True
        st.success("Document staging complete.")

    st.header("Document Editors")
    selected_doc = st.selectbox("Select Document to Edit", options=_get_doc_ids(DIRECTORY_VLM_OUTPUT))
    selected_doc_path = Path(DIRECTORY_MD_PREPROCESSING) / selected_doc
    _render_document_editor(selected_doc, selected_doc_path)


# -------------------------------------- Preprocessing Step 2 - LLM Formatting / Enhancement -------------------------------------- #
def render_llm_preprocessor() -> None:
    """Use LLM to structure & enhance markdown documents."""

    st.title("LLM Formatting/Enhancement")
    selected_document = st.selectbox("Select Document to Process", options=_get_doc_ids(DIRECTORY_MD_PREPROCESSING))
    source_doc_path = Path(DIRECTORY_MD_PREPROCESSING) / selected_document
    dest_doc_path = Path(DIRECTORY_LLM_PREPROCESSING) / selected_document

    # once initially, copy source doc to dest doc path
    if f"llm_staged_{selected_document}" not in st.session_state:
        dest_doc_path.mkdir(parents=True, exist_ok=True)
        source_md_filepath = next(source_doc_path.glob("*.md"))
        dest_md_filepath = dest_doc_path / source_md_filepath.name
        shutil.copy(source_md_filepath, dest_md_filepath)
        st.session_state[f"llm_staged_{selected_document}"] = True

    md_filepath = next(dest_doc_path.glob("*.md"))
    original_content = md_filepath.read_text(encoding="utf-8")

    # --- LLM Inputs Sidebar --- #
    llm_kwargs = {
        "temperature": st.session_state.llm_temperature,
        "top_p": st.session_state.llm_top_p,
        "reasoning_effort": st.session_state.llm_reasoning_effort,
    }
    lvl_1_heading = st.text_input("Provide Level 1 Heading (with number) for LLM Preprocessing", key=f"llm_heading_{selected_document}") # noqa

    if st.button("LLM Preprocess Document", key=f"llm_preprocess_{selected_document}"):
        user_message = (
            f"Enhance the following document for clarity, structure, and completeness while preserving all original information\n" # noqa
            f"Level 1 Heading: # {lvl_1_heading.strip()}\n---\n"
            f"<original content>\n{original_content}\n</original content>"
        )
        with st.spinner("...processing the document"):
            stream = st.session_state.client.api_query(
                model=st.session_state.md_model,
                user_message=user_message,
                system_prompt=SYS_LECTURE_SUMMARIZER,
                **llm_kwargs,
            )
            processed_content = "".join(chunk for chunk in stream)

        md_filepath.write_text(processed_content, encoding="utf-8")
        st.success(f"Document '{selected_document}' preprocessed and saved.")
        st.rerun()

    if st.button("LLM Enhance Document", key=f"llm_enhance_{selected_document}"):
        with st.spinner("...enhancing the document"):
            user_message = (
                f"<original content>\n{original_content}\n</original content>"
            )
            stream = st.session_state.client.api_query(
                model=st.session_state.md_model,
                user_message=user_message,
                system_prompt=SYS_LECTURE_ENHENCER,
                **llm_kwargs,
            )
            enhanced_content = "".join(chunk for chunk in stream)

        md_filepath.write_text(enhanced_content, encoding="utf-8")
        st.success(f"Document '{selected_document}' enhanced and saved.")
        st.rerun()

    st.subheader("Preview")
    st.markdown(original_content, unsafe_allow_html=True)

# ----------------------------- Preprocessing Step 3 - Chunking / Hierarchy / Parquet Storage ----------------------------- #
class MetadataKeys:
    LEVEL = "level"
    KEY_H1 = "h1"
    KEY_H2 = "h2"
    KEY_H3 = "h3"
    CONTEXT_PATH = "context_path"

METADATA_SCHEMA = {
    MetadataKeys.LEVEL: pl.Int8,
    MetadataKeys.KEY_H1: pl.String,
    MetadataKeys.KEY_H2: pl.String,
    MetadataKeys.KEY_H3: pl.String,
    MetadataKeys.CONTEXT_PATH: pl.String,
}

DEFAULT_HEADING = "General"

def create_ingestion_payload(markdown_filepath: str) -> RAGIngestionPayload:
    """
    Parses a Markdown file into hierarchical text chunks based on heading levels.

    The function reads a Markdown file and splits its content into chunks,
    where each chunk is demarcated by a heading (H1, H2, or H3). Each chunk
    is associated with its full hierarchical context (e.g., "H1 > H2 > H3").

    Args:
        markdown_filepath: The path to the input Markdown file.

    Returns:
        A RAGIngestionPayload object containing a Polars DataFrame. The DataFrame
        has columns for the chunk title, content for retrieval, content for
        embedding, and JSON-encoded metadata about the chunk's hierarchy.
        Returns an empty payload if the file is not found or is empty.
    """
    try:
        with open(markdown_filepath, "r", encoding="utf-8") as f:
            markdown_text = f.read()
    except FileNotFoundError:
        return RAGIngestionPayload.create_empty_payload()

    lines = markdown_text.split("\n")
    chunks = []
    current_buffer = []

    # State tracking for the current hierarchy
    current_h1 = DEFAULT_HEADING
    current_h2 = DEFAULT_HEADING
    current_h3 = DEFAULT_HEADING

    def save_current_chunk() -> None:
        """Saves the content in the buffer as a new chunk with current hierarchy."""
        text_content = "\n".join(current_buffer).strip()
        if not text_content:
            return

        # Determine the most specific heading to use as the chunk's title
        if current_h3 != DEFAULT_HEADING:
            title = current_h3
            level = 3
        elif current_h2 != DEFAULT_HEADING:
            title = current_h2
            level = 2
        else:
            title = current_h1
            level = 1

        # Create the full context path for better embedding and filtering
        context_path = f"{current_h1} > {current_h2} > {current_h3}"

        metadata = {
            MetadataKeys.LEVEL: level,
            MetadataKeys.KEY_H1: current_h1,
            MetadataKeys.KEY_H2: current_h2,
            MetadataKeys.KEY_H3: current_h3,
            MetadataKeys.CONTEXT_PATH: context_path,
        }

        text_without_image_links = re.sub(r"!\[.*?\]\(.*?\)", "", text_content).strip()

        # The record to be added to the DataFrame
        chunk_record = {
            DatabaseKeys.KEY_TITLE: title,
            DatabaseKeys.KEY_METADATA: json.dumps(metadata),
            DatabaseKeys.KEY_TXT_RETRIEVAL: text_content,
            DatabaseKeys.KEY_TXT_EMBEDDING: f"{context_path}\n\n{text_without_image_links}",
        }
        chunks.append(chunk_record)

    for line in lines:
        if line.startswith("# "):
            save_current_chunk()
            current_buffer = []
            current_h1 = line[2:].strip()
            current_h2 = DEFAULT_HEADING  # Reset sub-levels
            current_h3 = DEFAULT_HEADING
        elif line.startswith("## "):
            save_current_chunk()
            current_buffer = []
            current_h2 = line[3:].strip()
            current_h3 = DEFAULT_HEADING  # Reset sub-level
        elif line.startswith("### "):
            save_current_chunk()
            current_buffer = []
            current_h3 = line[4:].strip()
        else:
            current_buffer.append(line)

    save_current_chunk()  # Save the final chunk after the loop

    if not chunks:
        return RAGIngestionPayload.create_empty_payload()

    return RAGIngestionPayload(df=pl.DataFrame(chunks))


# ------ Rendering & Editing of Chunks in Hierarchical Format ------ #
def render_chunks(output_name: str) -> None:
    """
    Renders an interactive, hierarchical editor for chunks in a Streamlit app.

    The editor displays chunks in a nested, expandable format based on their
    H1/H2/H3 hierarchy. It allows users to view, edit, and delete each chunk.
    Modifications are applied directly to the DataFrame stored in
    `st.session_state.rag_ingestion_payload`.

    Args:
        output_name: The key to access the correct RAGIngestionPayload in
                     `st.session_state`.
    """
    payload = st.session_state.rag_ingestion_payload[output_name]
    if payload.df.is_empty():
        st.info("No chunks to display.")
        return

    # --- Logic for Merging Chunks (Absorbing Lowest Level) ---
    if st.button("Merge Chunks (Absorb Lowest Level)", key=f"merge_chunks_{output_name}"):
        # Keep index order to sort back correctly after split
        df = payload.df.with_row_index("index_order")

        # Parse metadata to determine hierarchy levels
        df_meta = df.with_columns(
            pl.col(DatabaseKeys.KEY_METADATA).str.json_decode(dtype=pl.Struct(METADATA_SCHEMA)).alias("meta")
        )

        max_level = df_meta.select(pl.col("meta").struct.field("level").max()).item()

        # Only proceed if we have levels deeper than 1
        if max_level is not None and max_level > 1:
            parent_level = max_level - 1

            # Define grouping keys (e.g., if merging L3->L2, group by H1 & H2)
            group_keys = ["h1"]
            if parent_level == 2:
                group_keys.append("h2")

            # Partition data
            grandparents = df_meta.filter(pl.col("meta").struct.field("level") < parent_level)
            parents = df_meta.filter(pl.col("meta").struct.field("level") == parent_level)
            children = df_meta.filter(pl.col("meta").struct.field("level") == max_level)

            if not children.is_empty():
                # Extract grouping keys for join
                children_keyed = children.with_columns([
                    pl.col("meta").struct.field(k).alias(k) for k in group_keys
                ])
                parents_keyed = parents.with_columns([
                    pl.col("meta").struct.field(k).alias(k) for k in group_keys
                ])

                # Aggregate children content (sort by index to maintain document flow)
                children_agg = (
                    children_keyed.sort("index_order")
                    .group_by(group_keys)
                    .agg([
                        pl.col(DatabaseKeys.KEY_TITLE).str.join(" > ").alias("c_titles"),
                        pl.col(DatabaseKeys.KEY_TXT_RETRIEVAL).str.join("\n\n").alias("c_texts"),
                        # Remove first line (old context) from child embeddings before joining
                        pl.col(DatabaseKeys.KEY_TXT_EMBEDDING)
                          .str.replace(r"^[^\n]*\n", "")
                          .str.strip_chars()
                          .str.join("\n\n")
                          .alias("c_embs")
                    ])
                )

                # Join parents with aggregated children and Update
                merged_parents = parents_keyed.join(children_agg, on=group_keys, how="left").with_columns([
                    # Append children titles
                    pl.when(pl.col("c_titles").is_not_null())
                      .then(pl.format("{} > {}", pl.col(DatabaseKeys.KEY_TITLE), pl.col("c_titles")))
                      .otherwise(pl.col(DatabaseKeys.KEY_TITLE))
                      .alias(DatabaseKeys.KEY_TITLE),

                    # Append children text
                    pl.when(pl.col("c_texts").is_not_null())
                      .then(pl.format("{}\n\n{}", pl.col(DatabaseKeys.KEY_TXT_RETRIEVAL), pl.col("c_texts")))
                      .otherwise(pl.col(DatabaseKeys.KEY_TXT_RETRIEVAL))
                      .alias(DatabaseKeys.KEY_TXT_RETRIEVAL),

                    # Append children embedding context (Use Merged Title as first line)
                    pl.when(pl.col("c_embs").is_not_null())
                      .then(pl.format(
                          "{}\n\n{}\n\n{}",
                          # 1. New Merged Title as First Line
                          pl.format("{} > {}", pl.col(DatabaseKeys.KEY_TITLE), pl.col("c_titles")),
                          # 2. Parent Body (Original Embedding minus first line)
                          pl.col(DatabaseKeys.KEY_TXT_EMBEDDING).str.replace(r"^[^\n]*\n", "").str.strip_chars(),
                          # 3. Aggregated Children Bodies
                          pl.col("c_embs")
                      ))
                      .otherwise(pl.col(DatabaseKeys.KEY_TXT_EMBEDDING))
                      .alias(DatabaseKeys.KEY_TXT_EMBEDDING),
                ])

                # Reassemble the dataframe (Grandparents + Merged Parents)
                final_df = (
                    pl.concat([
                        grandparents.select(df.columns),
                        merged_parents.select(df.columns)
                    ])
                    .sort("index_order")
                    .drop("index_order")
                )

                # Update State and Rerun
                st.session_state.rag_ingestion_payload[output_name].df = final_df
                st.rerun()

    # Add a unique row ID to prevent key collisions in Streamlit
    chunks_df_with_id = payload.df.with_row_index("unique_id")

    def render_chunk(row: dict) -> None:
        """Renders a single chunk (as a dict) with editing capabilities."""
        unique_key_suffix = f"{output_name}_{row['unique_id']}"

        if unique_key_suffix not in st.session_state.is_chunk_edit_mode_active:
            st.session_state.is_chunk_edit_mode_active[unique_key_suffix] = False

        # --- UI for Toggling Edit/View Mode ---
        toggle_cols = st.columns([1, 1, 8])
        if toggle_cols[0].button("Edit", key=f"edit_btn_{unique_key_suffix}"):
            st.session_state.is_chunk_edit_mode_active[unique_key_suffix] = True

        if st.session_state.is_chunk_edit_mode_active[unique_key_suffix] is True:
            # --- Edit Mode ---
            editor_key = f"editor_{unique_key_suffix}"
            original_text = row[DatabaseKeys.KEY_TXT_RETRIEVAL]
            edited_text = editor(text_to_edit=original_text, language="latex", key=editor_key)
            edited_text # noqa

            action_cols = st.columns([1, 1, 8])
            if action_cols[0].button("Save", key=f"save_btn_{unique_key_suffix}"):
                current_df = st.session_state.rag_ingestion_payload[output_name].df
                updated_df = current_df.with_columns(
                    pl.when(pl.col(DatabaseKeys.KEY_TITLE) == row[DatabaseKeys.KEY_TITLE])
                    .then(pl.lit(edited_text))
                    .otherwise(pl.col(DatabaseKeys.KEY_TXT_RETRIEVAL))
                    .alias(DatabaseKeys.KEY_TXT_RETRIEVAL)
                )
                st.session_state.rag_ingestion_payload[output_name].df = updated_df
                st.session_state.is_chunk_edit_mode_active[unique_key_suffix] = False
                st.rerun()  # Rerun to reflect changes and exit edit mode

            if action_cols[1].button("Delete", key=f"delete_btn_{unique_key_suffix}"):
                current_df = st.session_state.rag_ingestion_payload[output_name].df
                updated_df = current_df.filter(pl.col(DatabaseKeys.KEY_TITLE) != row[DatabaseKeys.KEY_TITLE])
                st.session_state.rag_ingestion_payload[output_name].df = updated_df
                st.rerun()  # Rerun to reflect the deletion
        else:
            # --- View Mode ---
            st.markdown(row[DatabaseKeys.KEY_TXT_RETRIEVAL])

    # --- Hierarchical Rendering Logic ---
    # Unpack metadata from JSON for efficient filtering
    chunks_with_metadata = chunks_df_with_id.with_columns(
        pl.col(DatabaseKeys.KEY_METADATA).str.json_decode(dtype=pl.Struct(METADATA_SCHEMA))
    ).unnest(DatabaseKeys.KEY_METADATA)

    level_1_df = chunks_with_metadata.filter(pl.col("level") == 1)
    level_2_df = chunks_with_metadata.filter(pl.col("level") == 2)
    level_3_df = chunks_with_metadata.filter(pl.col("level") == 3)

    for l1_row in level_1_df.to_dicts():
        with st.expander(f"{l1_row['h1']}"):
            render_chunk(l1_row)
            l2_children = level_2_df.filter(pl.col("h1") == l1_row["h1"])
            for l2_row in l2_children.to_dicts():
                with st.expander(f"{l2_row['h2']}"):
                    render_chunk(l2_row)
                    l3_children = level_3_df.filter(pl.col("h2") == l2_row["h2"])
                    for l3_row in l3_children.to_dicts():
                        with st.expander(f"{l3_row['h3']}"):
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
    directories_preprocessed_output = sorted(os.listdir(DIRECTORY_MD_PREPROCESSING))
    directories_preprocessed_output = [dir for dir in directories_preprocessed_output if dir != "archive"]

    # Set all payload initialization flags to False
    if st.session_state.is_payload_initialized == {}:
        for output_name in directories_preprocessed_output:
            st.session_state.is_payload_initialized[output_name] = False

    with center:
        selected_output = st.selectbox("Select Document to Chunk/Edit", options=directories_preprocessed_output)
        md_filepath = f"{DIRECTORY_MD_PREPROCESSING}/{selected_output}/{selected_output}.md"

        # Parse and store DataFrame in session state on first run for this file
        if not st.session_state.is_payload_initialized[selected_output]:
            st.session_state.rag_ingestion_payload[selected_output] = create_ingestion_payload(md_filepath)
            st.session_state.is_payload_initialized[selected_output] = True

        if st.button("Store chunks to Parquet", key=f"store_md_chunks_{selected_output}", type="primary"):
            payload = st.session_state.rag_ingestion_payload[selected_output]
            save_payload = RAGIngestionPayload(df=payload.df)
            save_payload.to_parquet(pathlib.Path(f"{DIRECTORY_RAG_INPUT}/{selected_output}/{selected_output}_ingestion_payload.parquet"))
            st.success(f"Stored chunked data for '{selected_output}' to Parquet.")

        # Render the interactive chunk editor, which operates on session_state directly
        render_chunks(output_name=selected_output)

def main() -> None:
    init_session_state()
    with st.sidebar:
        st.session_state.md_model = model_selector(key="markdown_preprocessor")
        llm_params_sidebar()
        st.markdown("---")
        selection = st.radio("Select Page", options=["Markdown Preprocessor","LLM Preprocessor", "Markdown Chunker"], index=0, key="markdown_page_selector")  # noqa

    _, center, _ = st.columns([1, 8, 1])
    with center:
        if selection == "Markdown Preprocessor":
            if st.sidebar.button("Initialize Preprocessor"):
                st.session_state.preprocessor_active = True

            if st.session_state.preprocessor_active is True:
                render_preprocessor()

        elif selection == "LLM Preprocessor":
            if st.sidebar.button("Initialize LLM Preprocessor"):
                st.session_state.llm_preprocessor_active = True

            if st.session_state.llm_preprocessor_active is True:
                render_llm_preprocessor()

        elif selection == "Markdown Chunker":
            if st.sidebar.button("Initialize Chunker"):
                st.session_state.chunker_active = True

            if st.session_state.chunker_active is True:
                markdown_chunker()

if __name__ == "__main__":
    main()