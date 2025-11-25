import os
import time

import fitz
import streamlit as st

from src.lib.streamlit_helper import print_metrics


def init_pdf_preprocessor() -> None:
    """Initialize PDF Preprocessor session state variables."""
    # Initialize session state for persistent storage across reruns
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
        st.session_state.preprocessed_filepaths = []
        st.session_state.deleted_filepaths = []
        st.session_state.total_pdfs_preprocessed = 0
        st.session_state.total_mb_preprocessed = 0.0
        st.session_state.total_pages_preprocessed = 0


def pdf_preprocessor() -> None:
    """
    PDF Viewer & Slicer.
    Upload large PDFs, view them inline, and slice specific page ranges for download.
    First preprocessing step for RAG mining.
    """
    st.set_page_config(layout="wide", page_title="PDF Preprocessor")
    st.header("PDF Preprocessor")

    cols = st.columns(2)

    with cols[0]:
        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, key="pdf_preprocessor_uploader")

        if not uploaded_files: # Block execution / notify user
            with cols[0]:
                st.info("Upload a file to begin preprocessing.")
            return

    # Convert single file to list for uniform processing
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files] if uploaded_files is not None else []

    # Store uploaded files in seperate list due to Streamlit uploader limitations
    processed_names = {f.name for f in st.session_state.uploaded_files}
    if st.session_state.uploaded_files == [] and st.session_state.total_pdfs_preprocessed == 0:
        for file in uploaded_files:
            if file.name not in processed_names:
                st.session_state.uploaded_files.append(file)

                # Store file on server for iframe rendering
                file_path = os.path.join(st.session_state.static_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

    processed_filepaths = st.session_state.preprocessed_filepaths + st.session_state.deleted_filepaths
    if len(processed_filepaths) == len(uploaded_files):
        with cols[0]:
            st.success("All uploaded PDFs have been preprocessed - proceed to VLM Extraction or upload new PDFs.")
        with cols[1]:
            if st.session_state.pdf_preprocessor_metrics:
                print_metrics(dict_metrics=st.session_state.pdf_preprocessor_metrics, n_columns=3)
            return

    with cols[1]:

        st.session_state.pdf_preprocessor_metrics = {
            "Total PDFs (in memory)": len(st.session_state.uploaded_files),
            "Total File Size MB (in memory)": round(sum(st.session_state.uploaded_files[i].size for i in range(len(st.session_state.uploaded_files)))/(1024*1024), 1), # noqa
            "Total Pages (in memory)": sum(fitz.open(stream=st.session_state.uploaded_files[i].getvalue(), filetype="pdf").page_count for i in range(len(st.session_state.uploaded_files))), # noqa
            "Total PDFs (preprocessed)": st.session_state.total_pdfs_preprocessed,
            "Total File Size MB (preprocessed)": f"{st.session_state.total_mb_preprocessed:.2f}",
            "Total Pages (preprocessed)": st.session_state.total_pages_preprocessed,
        }
        print_metrics(dict_metrics=st.session_state.pdf_preprocessor_metrics, n_columns=3)

    st.divider()

    relocated_filepaths = [os.path.join(st.session_state.static_dir, f.name) for f in st.session_state.uploaded_files]


    pdf_cols = st.columns([1,1])

    for i in range(len(st.session_state.uploaded_files[:2])): # Restrict to first 2 PDFs for memory efficiency
        with pdf_cols[i%2], st.expander(f"Preview: {st.session_state.uploaded_files[i].name}", expanded=True):

            file_bytes = st.session_state.uploaded_files[i].getvalue()
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            num_pages = doc.page_count

            metrics_dict = {
                "Pages": num_pages,
                "File Size (MB)": f"{st.session_state.uploaded_files[i].size/(1024*1024):.2f}",
            }
            print_metrics(dict_metrics=metrics_dict)

            # ... slicing logic
            st.divider()
            st.write("### Slice Range")
            c1, c2 = st.columns(2)
            start_p = c1.number_input("Start", 1, num_pages, 1, key=f"start_page_{i}")
            end_p = c2.number_input("End", 1, num_pages, num_pages, key=f"end_page_{i}")

            cols = st.columns(2)
            if cols[0].button("Delete", type="primary", use_container_width=True, key=f"delete_button_{i}"):
                st.session_state.uploaded_files.pop(i)
                st.session_state.deleted_filepaths.append(relocated_filepaths[i])
                os.remove(relocated_filepaths[i])

                if len(st.session_state.uploaded_files) == 0:
                    st.success("All uploaded PDFs have been preprocessed - proceed to VLM Extraction or upload new PDFs.") # noqa
                    st.session_state.all_pdfs_preprocessed = True
                    st.rerun()
                else:
                    st.success("Deleted!")
                st.rerun()


            if cols[1].button("Cut & Save", type="primary", use_container_width=True, key=f"slice_button_{i}"):
                if start_p <= end_p:
                    new_doc = fitz.open()
                    new_doc.insert_pdf(doc, from_page=start_p-1, to_page=end_p-1)

                    base_name, ext = os.path.splitext(st.session_state.uploaded_files[i].name)
                    sliced_filename = f"{base_name}_preprocessed_{start_p}-{end_p}{ext}"
                    sliced_filepath = os.path.join(st.session_state.static_dir, sliced_filename)

                    # Store sliced PDF & remove original server-file from static - PDF source location remains untouched
                    new_doc.save(sliced_filepath)
                    new_doc.close()
                    os.remove(relocated_filepaths[i])

                    st.session_state.preprocessed_filepaths.append(sliced_filepath)
                    st.session_state.uploaded_files.pop(i)
                    st.session_state.total_pdfs_preprocessed += 1
                    st.session_state.total_mb_preprocessed += os.path.getsize(sliced_filepath)/(1024*1024)
                    st.session_state.total_pages_preprocessed += (end_p - start_p + 1)

                    if len(st.session_state.uploaded_files) == 0:
                        st.success("All uploaded PDFs have been preprocessed - proceed to VLM Extraction or upload new PDFs.") # noqa
                        st.session_state.all_pdfs_preprocessed = True
                    else:
                        st.success(f"Preprocessed and saved: {sliced_filename}")
                    st.rerun()
                else:
                    st.error("Invalid Range")

            # Render the iframe pointing to the static file URL**
            # The URL will be `app/static/<selected_filename>.pdf?t=<timestamp>` to avoid caching issues
            pdf_url = f"{st.session_state.app_static_dir}/{st.session_state.uploaded_files[i].name}?t={int(time.time())}"

            st.markdown(
                f'''
                <iframe 
                    src="{pdf_url}"
                    width="100%" 
                    height="100%" 
                    style="min-height:85vh; border:none; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"
                ></iframe>
                ''',
                unsafe_allow_html=True
            )
    


if __name__ == "__main__":
    init_pdf_preprocessor()
    pdf_preprocessor()