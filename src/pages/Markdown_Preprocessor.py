"""
NOTE TO MY SELF: Every step in this pipeline shall seperate streamlit UI & data processing logic without user interaction.
Required for later building API endpoints for automation.
"""
import json
import math
import os
import pathlib
from pathlib import Path
import re
import shutil

import networkx as nx
import polars as pl
from rag_database.dataclasses import RAGIngestionPayload
from rag_database.rag_config import DatabaseKeys
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph  # type: ignore

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
        st.session_state.is_rag_ingestion_payload_initialized = False
        st.session_state.is_edit_mode_active = False
        st.session_state.staging_complete = False
        st.session_state.rag_ingestion_payload = {}
        st.session_state.selected_graph_document = None

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

IMAGE_PATH_PATTERN = re.compile(r"!\[(.*?)\]\(images/")

def _process_document(doc_id: str) -> None:
    """Copies, cleans, and restructures a single document and its assets."""
    try:
        source_base_path = Path(DIRECTORY_VLM_OUTPUT) / f"converted_{doc_id}.pdf" / doc_id / "auto"
        dest_base_path = Path(DIRECTORY_MD_PREPROCESSING_1) / doc_id
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

        if source_imgs_path.is_dir():
            shutil.copytree(source_imgs_path, static_imgs_dest_path, dirs_exist_ok=True)

        original_content = dest_md_path.read_text(encoding="utf-8")
        content_with_updated_img_paths = IMAGE_PATH_PATTERN.sub(
            rf"![\1]({server_img_url_path}/", original_content
        )

        lines = content_with_updated_img_paths.splitlines()
        processed_lines = _transform_headings(lines)
        dest_md_path.write_text("\n".join(processed_lines), encoding="utf-8")

    except (IOError, OSError) as e:
        st.error(f"Failed to process '{doc_id}': {e}")

def _get_doc_ids(source_directory: str) -> list[str]:
    """Retrieves document IDs from the VLM output directory."""
    source_path = Path(source_directory)

    if not source_path.is_dir():
        return []

    doc_ids = [
        path.name.replace("converted_", "").replace(".pdf", "")
        for path in source_path.glob("converted_*.pdf")
        if path.is_dir()
    ]

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
        _process_document(doc_id)

def _render_document_editor(doc_id: str, base_path: Path) -> None:
    """Displays a markdown editor and preview for a single document."""
    try:
        md_filepath = next(base_path.glob("*.md"))
    except StopIteration:
        st.warning(f"Markdown file for '{doc_id}' not found. Skipping.")
        return

    with st.expander(doc_id):
        original_content = md_filepath.read_text(encoding="utf-8")
        editor_cols = st.columns(2)
        with editor_cols[0]:
            st.subheader("Editor")
            edited_text = editor(
                text_to_edit=original_content,
                language="markdown",
                key=f"editor_{doc_id}"
            )
        with editor_cols[1]:
            st.subheader("Preview")
            st.markdown(edited_text)

        if st.button("Save Changes", key=f"save_{doc_id}"):
            try:
                md_filepath.write_text(edited_text, encoding="utf-8")
                st.success(f"Saved changes to {md_filepath.name}")
            except (IOError, OSError) as e:
                st.error(f"Could not save file for '{doc_id}': {e}")

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
    for doc_id in _get_doc_ids(DIRECTORY_VLM_OUTPUT):
        doc_path = Path(DIRECTORY_MD_PREPROCESSING_1) / doc_id
        _render_document_editor(doc_id, doc_path)


# ----------------------------- Preprocessing Step 2 - Chunking / Hierarchy / Parquet Storage ----------------------------- #
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

def _create_empty_payload() -> RAGIngestionPayload:
    """Creates an empty RAGIngestionPayload with the correct schema."""
    schema = {
        DatabaseKeys.KEY_TITLE: pl.String,
        DatabaseKeys.KEY_TXT_RETRIEVAL: pl.String,
        DatabaseKeys.KEY_TXT_EMBEDDING: pl.String,
        DatabaseKeys.KEY_METADATA: pl.String,
    }
    return RAGIngestionPayload(df=pl.DataFrame(schema=schema))

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
        return _create_empty_payload()

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
            "level": level,
            "h1": current_h1,
            "h2": current_h2,
            "h3": current_h3,
            "context_path": context_path,
        }

        # The record to be added to the DataFrame
        chunk_record = {
            DatabaseKeys.KEY_TITLE: title,
            DatabaseKeys.KEY_METADATA: json.dumps(metadata),
            DatabaseKeys.KEY_TXT_RETRIEVAL: text_content,
            DatabaseKeys.KEY_TXT_EMBEDDING: f"{context_path}\n\n{text_content}",
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
        return _create_empty_payload()

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

    # Add a unique row ID to prevent key collisions in Streamlit
    chunks_df_with_id = payload.df.with_row_index("unique_id")

    def render_chunk(row: dict) -> None:
        """Renders a single chunk (as a dict) with editing capabilities."""
        unique_key_suffix = f"{output_name}_{row['unique_id']}"

        # --- UI for Toggling Edit/View Mode ---
        toggle_cols = st.columns([1, 1, 8])
        if toggle_cols[0].button("Edit", key=f"edit_btn_{unique_key_suffix}"):
            st.session_state.is_edit_mode_active = True

        if st.session_state.is_edit_mode_active is True:
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
                st.session_state.is_edit_mode_active = False
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
    directory_preprocessed_output = os.listdir(DIRECTORY_MD_PREPROCESSING_1)

    # Initialize session state for holding DataFrames
    if 'rag_ingestion_payload' not in st.session_state:
        st.session_state.rag_ingestion_payload = {}

    with center:
        for output_name in directory_preprocessed_output:
            md_filepath = f"{DIRECTORY_MD_PREPROCESSING_1}/{output_name}/{output_name}.md"

            # Parse and store DataFrame in session state on first run for this file
            if not st.session_state.is_rag_ingestion_payload_initialized:
                st.session_state.rag_ingestion_payload[output_name] = create_ingestion_payload(md_filepath)
                st.session_state.is_rag_ingestion_payload_initialized = True

            with st.expander(output_name):
                if st.button("Store chunks to Parquet", key=f"store_md_chunks_{output_name}", type="primary"):
                    payload = st.session_state.rag_ingestion_payload[output_name]
                    save_payload = RAGIngestionPayload(df=payload.df)
                    save_payload.to_parquet(pathlib.Path(f"{DIRECTORY_RAG_INPUT}/{output_name}/chunked_{output_name}.parquet"))
                    st.success(f"Stored chunked data for '{output_name}' to Parquet.")

                # Render the interactive chunk editor, which operates on session_state directly
                render_chunks(output_name=output_name)

def render_document_graph() -> None:
    """Render a hierarchical graph view of the document chunks."""

    st.header("Document Hierarchy Graph")

    if st.session_state.rag_ingestion_payload == {}:
        st.info("No processed document data available to build the graph.")
        return

    selected_graph_document = st.selectbox(
        "Select Document",
        options=st.session_state.rag_ingestion_payload.keys(),
        key="selected_document_for_graph"
    )

    if selected_graph_document is not None:
        st.session_state.selected_graph_document = selected_graph_document
    else:
        return

    processed_df = st.session_state.rag_ingestion_payload[selected_graph_document].df

    # --- 1. Build the Graph from DataFrame Hierarchy ---
    G = nx.DiGraph()
    # Add a root node for the entire document set
    document_root_node = "DOCUMENT ROOT"
    G.add_node(document_root_node, level=-1, label=document_root_node)

    # Use a dictionary to map node IDs to their corresponding text content
    node_id_to_content = {}

    # Unpack metadata for easier access
    df = processed_df.with_columns(
        pl.col("metadata").str.json_decode(dtype=pl.Struct(METADATA_SCHEMA))
    ).unnest("metadata")

    for row in df.iter_rows(named=True):
        # Create unique, path-like IDs for each node to avoid collisions
        h1_id = row[MetadataKeys.KEY_H1]
        h2_id = f"{h1_id} > {row[MetadataKeys.KEY_H2]}"
        h3_id = f"{h2_id} > {row[MetadataKeys.KEY_H3]}"
        chunk_id = f"{h3_id} > {row[DatabaseKeys.KEY_TITLE]}"

        # Add nodes and edges, ensuring parent nodes are created
        G.add_node(h1_id, level=1, label=row[MetadataKeys.KEY_H1])
        G.add_edge(document_root_node, h1_id)

        if row[MetadataKeys.KEY_H2] != "General":
            G.add_node(h2_id, level=2, label=row[MetadataKeys.KEY_H2])
            G.add_edge(h1_id, h2_id)

        if row[MetadataKeys.KEY_H3] != "General":
            G.add_node(h3_id, level=3, label=row[MetadataKeys.KEY_H3])
            G.add_edge(h2_id, h3_id)

        # The chunk itself is a leaf node
        parent_id = h3_id if row[MetadataKeys.KEY_H3] != "General" \
            else h2_id if row[MetadataKeys.KEY_H2] != "General" \
            else h1_id

        G.add_node(chunk_id, level=4, label=row[DatabaseKeys.KEY_TITLE])
        G.add_edge(parent_id, chunk_id)

        # Store the chunk's content for the click-to-view feature
        node_id_to_content[chunk_id] = row[DatabaseKeys.KEY_TXT_RETRIEVAL]

    # --- 2. Configure Node and Edge Styles ---
    nodes: list[Node] = []
    edges: list[Edge] = []

    level_colors = {
        -1: "#ff6b6b",  # Root
        1: "#ffd93d",   # H1
        2: "#6bcb77",   # H2
        3: "#4d96ff",   # H3
        4: "#f06595",   # Chunk
    }

    for node_id, attrs in G.nodes(data=True):
        level = attrs.get("level", 4)
        is_leaf = G.out_degree(node_id) == 0
        content_length = len(node_id_to_content.get(node_id, ""))

        nodes.append(Node(
            id=node_id,
            label=str(attrs.get("label", node_id)),
            size=max(int(math.log1p(content_length) * 3), 10) if is_leaf else 15,
            color=level_colors.get(level, "#AEC6CF"),
            title=node_id,  # Tooltip shows the full path
            shape="box" if is_leaf else "ellipse",
        ))

    for source, target in G.edges():
        edges.append(Edge(
            source=source,
            target=target,
            color="rgba(255,255,255,0.3)",
        ))

    # --- 3. Render the Graph and Interaction Expander ---
    config = Config(
        width="100%",
        height=700,
        directed=True,
        physics=False, # Physics is not ideal for tree structures
        hierarchical={
            "enabled": True,
            "sortMethod": "directed", # Sorts from the root
            "direction": "UD",       # Up-Down direction
            "levelSeparation": 150,
        },
        backgroundColor="#1a1a1a",
    )

    selected_node_id = agraph(nodes=nodes, edges=edges, config=config)

    with st.expander("Selected Document Chunk", expanded=True):
        if selected_node_id and selected_node_id in node_id_to_content:
            st.markdown(f"### {G.nodes[selected_node_id].get('label', 'Content')}")
            st.markdown(node_id_to_content[selected_node_id])
        elif selected_node_id:
            st.info("This is a heading node. Click on a rectangular document node to see its content.")
        else:
            st.info("Click a rectangular node in the graph to display its content here.")

def main() -> None:
    init_session_state()
    selection = st.sidebar.radio("Select Page", options=["Markdown Preprocessor", "Markdown Chunker"], index=0, key="markdown_page_selector")  # noqa

    _, center, _ = st.columns([1, 8, 1])
    with center:
        if selection == "Markdown Preprocessor":

            if st.sidebar.button("Initialize Preprocessor"):
                st.session_state.preprocessor_active = True

            if st.session_state.preprocessor_active is True:
                render_preprocessor()

        elif selection == "Markdown Chunker":

            if st.sidebar.button("Initialize Chunker"):
                st.session_state.chunker_active = True

            if st.session_state.chunker_active is True:
                tabs = st.tabs(["Chunk Editor", "Document Graph"])
                with tabs[0]:
                    markdown_chunker()
                with tabs[1]:
                    render_document_graph()

if __name__ == "__main__":
    main()