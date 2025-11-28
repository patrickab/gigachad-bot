import pandas as pd # noqa
import numpy as np # noqa
import re # noqa
import plotly.graph_objects as go # noqa
import streamlit as st
from llm_client import LLMClient
import os
from src.lib.streamlit_helper import editor, AVAILABLE_LLM_MODELS


SYS_DATAFRAME_BOT ="""
    **Role:** Python Data Analysis Bot

    **Core Directive:**
    Translate natural language queries into a single, executable Python code block.
    The code must manipulate a pandas DataFrame or generate a Plotly figure by interacting directly with Streamlit session state variables (`st.session_state`).
    Success is measured by the code's correctness and strict adherence to all specified constraints.

    **Guiding Principles:**
    1.  **Stateful Architecture:** All operations are stateful.
        -   **Input:** Read the DataFrame exclusively from `st.session_state.display_dataframe`.
        -   **Output (Data):** Write the modified DataFrame back to `st.session_state.display_dataframe`.
        -   **Output (Plot):** Write Plotly figures (`go.Figure`) to `st.session_state.display_plot`.
    2.  **Modular Logic:** Encapsulate logic in specific, parameter-less functions that return `None`.
        -   Data manipulation: `def dataframe_operations() -> None:`
        -   Visualization: `def plotting_operations() -> None:`
        -   Generate only the function(s) required by the user's request.
    3.  **Code Clarity:** Prioritize readability and maintainability.
        -   **Docstrings:** Provide a concise, bullet-point explanation of the logic within each function's docstring.
        -   **No Inline Comments:** Do not use `#` comments for explanation.
        -   **Non-Destructive:** Do not drop columns unless explicitly instructed.

    **Constraints:**
    1.  **Library Usage:** Use only pre-imported libraries: `pandas as pd`, `numpy as np`, `plotly.graph_objects as go`, and `re`. Do not include `import` statements.
    2.  **State Access Pattern:** Functions must strictly adhere to this internal pattern:
        1.  `df = st.session_state.display_dataframe` (Start of function)
        2.  Perform all operations on the local `df` variable.
        3.  `st.session_state.display_dataframe = df` or `st.session_state.display_plot = fig` (End of function)
    3.  **Output Structure:**
        -   Produce a single, standalone Python code block.
        -   Include an `if __name__ == "__main__":` block to call the generated function(s).
        -   If both functions are generated, `dataframe_operations()` must be called before `plotting_operations()`.

    **Code Template:**
    ```python
    def dataframe_operations() -> None:
        \"\"\"
        Docstring with concise low-verbosity step-by-step explanation of steps
        \"\"\"
        df = st.session_state.display_dataframe
        # ... code to manipulate df ...
        st.session_state.display_dataframe = df

    def plotting_operations() -> None:
        \"\"\"
        Docstring with concise low-verbosity step-by-step explanation of steps
        \"\"\"
        df = st.session_state.display_dataframe
        # ... code to generate fig from df ...
        st.session_state.display_plot = fig

    if __name__ == "__main__":
        # Call only the function(s) you generate.
        # The order below is mandatory if both are generated.
        dataframe_operations()
        plotting_operations()
    ```
"""# noqa


def sidebar() -> None:
    """Sidebar for DataFrame Bot page."""
    with st.sidebar:

        st.session_state.selected_model = st.selectbox(
            "Select LLM",
            AVAILABLE_LLM_MODELS,
            key="model_select",
        )

        st.divider()

        parquet = st.file_uploader("Upload .parquet file", type=["parquet"], key="dataframe_uploader")
        if parquet is not None and st.session_state.is_df_load_required:
            st.session_state.is_df_load_required = False
            st.session_state.display_dataframe = pd.read_parquet(parquet)
        elif parquet is not None:
            pass
        else:
            return

        st.divider()

        with st.expander("DataFrame Operations", expanded=False):
            if st.button("Reset DataFrame"):
                st.session_state.is_df_reload_required = True
                st.rerun()

            if st.button("Store Dataframe"):
                path = st.text_input("Enter file path to store dataframe:")
                if path:
                    # get filename from path
                    filename = os.path.basename(path)
                    # backup
                    backup_dir = "src/static/DataFrame_Backup/"
                    os.makedirs(backup_dir, exist_ok=True)
                    backup_path = os.path.join(backup_dir, filename)
                    st.session_state.display_dataframe.to_parquet(backup_path)
                    st.success(f"Original DataFrame backed up to {backup_path}")
                    st.session_state.display_dataframe.to_parquet(path)
                    st.success(f"New DataFrame stored to {path}")


class DataFrame_Bot:
    """Client for interacting with the OpenAI to generate & execute code for DataFrame operations."""

    def __init__(self) -> None:
        """Initialize the Azure OpenAI client."""
        self.client = LLMClient()

    def llm_query_to_code(self, user_query: str) -> tuple[str, str]:
        """
        Convert a natural language query about dataframe operations into executable Python code.

        Parameters:
        -----------
            user_query: The user's natural language query about data analysis.
            dataframe: The pandas dataframe to be analyzed.

        Returns:
        --------
            tuple containing:
                executable Python code as string
                explanatory text/comments as markdown
        """
        column_info = str(st.session_state.display_dataframe.dtypes.to_dict())
        sample_data = str(st.session_state.display_dataframe.head(5))

        user_prompt = f"""
        # User Query Data
    
        DataFrame Information:
        Columns and types:
        {column_info}

        Sample data:
        {sample_data}

        User Query: {user_query}
        """

        try:
            stream = self.client.api_query(
                model=st.session_state.selected_model,
                user_message=user_prompt,
                system_prompt=SYS_DATAFRAME_BOT,
            )

            response = ""
            for chunk in stream:
                response += chunk

            if not response:
                return "No code generated", "No code generated"
            else:
                python_code_block = response.split("```python")[1].split("```")[0].strip()
                non_python_code = response.split("```python")[0].strip()
                return python_code_block, non_python_code

        except Exception as e:
            error = f"Error parsing response: {e}"
            st.error(error)
            return error, error

    def execute_generated_code(self) -> None:
        """
        Execute the generated Python code and return the result.

        Parameters:
            code: The Python code to execute.
            dataframe: The pandas DataFrame to be used in the code.

        Note:
            Not safe for productive execution; executes in python global namespace.
        """
        try:
            exec(st.session_state.generated_code, globals())
        except Exception as e:
            error = f"Error executing code: {e}"
            st.error(error)
            raise RuntimeError(error)

    def render_streamlit(self) -> None:
        """
        Wrap ChatilithYourDataClient into a Streamlit app interface.
        Can be integrated as tab, column or standalone app. Preserves state between interactions to avoid page reloads.
        """
        if "is_df_load_required" not in st.session_state:
            st.session_state.is_df_load_required = True
            st.session_state.generated_code = ""
            st.session_state.display_dataframe = pd.DataFrame()
            st.session_state.display_plot = go.Figure()
            st.session_state.code_explanation = ""
            st.session_state.is_editing_code = False           

        # Show dataframe (excluding embeddings)
        display_cols = [
            col
            for col in st.session_state.display_dataframe.columns
            if col != "embeddings"
        ]

        with st.expander("In-Memory DataFrame", expanded=True):
            st.dataframe(st.session_state.display_dataframe[display_cols])
            st.divider()

        with st._bottom:
            user_input = st.chat_input("Example: Visualize the distribution of text length", key="chat_input")
            if user_input:
                st.session_state.user_input = user_input
                st.session_state.generated_code, st.session_state.code_explanation = self.llm_query_to_code(user_query=user_input)

        col_code, col_plot = st.columns([1, 2])

        with col_code:
            if st.session_state.generated_code:
                st.markdown("### Chatbot Response")

                # Determine which code to display/edit
                current_code_content = st.session_state.generated_code

                if not st.session_state.is_editing_code:
                    st.code(current_code_content, height=300) # Display as non-editable codeblock
                    if st.button("Edit Code"):
                        st.session_state.is_editing_code = True
                        st.rerun() # Rerun to switch to editor mode
                else:
                    # Display the editable editor
                    edited_code = editor(current_code_content, height=300, key="edit_generated_code", language="python")
                    # Update session state with the content of the editor whenever a rerun happens if it changed
                    if st.session_state.generated_code != edited_code:
                        st.session_state.generated_code = edited_code
                    if st.button("Save & View Code"):
                        st.session_state.is_editing_code = False
                        st.rerun() # Rerun to switch back to static code view
                st.button("Execute Code", on_click=self.execute_generated_code)

        with col_plot:
            st.markdown("### Visualization Output")
            st.plotly_chart(st.session_state.display_plot)


if __name__ == "__main__":
    sidebar()
    df_bot = DataFrame_Bot()
    df_bot.render_streamlit()