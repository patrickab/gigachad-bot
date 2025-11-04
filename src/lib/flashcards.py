from datetime import datetime, timedelta
import os

import pandas as pd
import streamlit as st

# --- Constants ---
QUESTION = "Question"
ANSWER = "Answer"
DATE_ADDED = "Date Added"
NEXT_APPEARANCE = "Next Appearance"
TAG = "Tag"
DF_COLUMNS = [QUESTION, ANSWER, DATE_ADDED, NEXT_APPEARANCE, TAG]
N_CARDS_PER_ROW = 2
DEFAULT_TAGS = []
CSV_PATH = "data/flashcards.csv"

# --- Data Handling Functions ---
def get_empty_df() -> pd.DataFrame:
    """Creates an empty DataFrame with the required columns."""
    return pd.DataFrame(columns=DF_COLUMNS)

def df_to_flashcard(
    question: str,
    answer: str,
    date_added: datetime,
    next_appearance: datetime,
    tags: list,
) -> pd.DataFrame:
    """Creates a single-row DataFrame for a new flashcard."""
    return pd.DataFrame([{
        QUESTION: question,
        ANSWER: answer,
        DATE_ADDED: date_added,
        NEXT_APPEARANCE: next_appearance,
        TAG: tags,
    }])

def get_due_flashcards(df: pd.DataFrame) -> pd.DataFrame:
    """Filters a DataFrame to return only the cards that are due for review."""
    if df.empty:
        return get_empty_df()
    return df[df[NEXT_APPEARANCE] <= datetime.now()]

def store_flashcard_dataframe(df: pd.DataFrame, tag: str) -> None:
    """
    Appends the given DataFrame to the flashcards CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to append.
        tag (str): A tag for the batch of flashcards being stored.
    """
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # Check if the file exists to determine if the header should be written
    file_exists = os.path.isfile(CSV_PATH)

    df.to_csv(CSV_PATH, mode='a', header=not file_exists, index=False)


# --- UI & State Helper Functions ---
def local_css(file_name: str)-> None:
    """Injects local CSS file into the Streamlit app."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Styling will be default.")

def update_next_appearance(card_question: str, next_appearance: datetime) -> None:
    """Updates the next appearance date for a specific card in the session state."""
    if next_appearance is not None:
        df = st.session_state.flashcards_df
        df.loc[df[QUESTION] == card_question, NEXT_APPEARANCE] = next_appearance

def add_new_flashcard(new_card_df: pd.DataFrame) -> None:
    """Adds a new flashcard DataFrame to the session state."""
    if not new_card_df.empty:
        st.session_state.flashcards_df = pd.concat(
            [st.session_state.flashcards_df, new_card_df],
            ignore_index=True
        )

# --- Tab Rendering Functions ---
def render_review_tab() -> None:
    """Renders the UI for the 'Review' tab."""
    due_cards = get_due_flashcards(st.session_state.flashcards_df)

    if due_cards.empty:
        st.info("Hey! You have completed all the flashcards. Good Job!", icon="🙌")
        return

    row = due_cards.iloc[0]
    st.markdown(
        f"""
        <div class="blockquote-wrapper">
        <div class="blockquote">
        <h1><span style="color:#ffffff">{row[QUESTION]}</span></h1>
        """,
        unsafe_allow_html=True,
    )
    _, col_center, _ = st.columns([0.191, 0.618, 0.191])
    with col_center:
        st.markdown("---")
        with st.expander("Show Answer"):
            st.write(row[ANSWER])

        next_appearance = None
        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            if st.button(label="Easy", width=200):
                prev_time_diff = row[NEXT_APPEARANCE] - row[DATE_ADDED]
                next_appearance_days = min(prev_time_diff.days + 2, 60)
                next_appearance = datetime.now() + timedelta(days=next_appearance_days)
        with col2:
            if st.button(label="Medium", width=200):
                next_appearance = datetime.now() + timedelta(days=2)
        with col3:
            if st.button(label="Hard", width=200):
                next_appearance = datetime.now() + timedelta(days=1)

    if next_appearance:
        update_next_appearance(row[QUESTION], next_appearance)
        st.info(
            f"""Next appearance of this card will be {next_appearance.date().strftime("%d-%m-%Y")}!""",
            icon="🎉",
        )
        st.rerun()


def render_view_all_tab() -> None:
    """Renders the UI for the 'View All' tab."""
    df = st.session_state.flashcards_df
    options = st.multiselect("Tags", DEFAULT_TAGS)
    show_all = st.checkbox("Show All", value=True)

    df_to_show = df

    if not show_all and options:
        try:
            df_to_show = df[df[TAG].apply(lambda x: any(tag in x for tag in options))]
        except KeyError:
            st.warning("No flashcards found with the selected tags!")
            return
    elif not show_all and not options:
        st.info("Select tags or check 'Show All' to view flashcards.")
        return

    if not df_to_show.empty:
        st.dataframe(
            df_to_show,
            width=200,
            column_order=[QUESTION, ANSWER, DATE_ADDED, NEXT_APPEARANCE, TAG],
        )
    else:
        st.write("No flashcards available.")

    st.markdown("---")
    st.subheader("Store Flashcards")
    tag_input = st.text_input("Enter a tag for this save operation", "session_save")
    if st.button("Append current flashcards to CSV"):
        store_flashcard_dataframe(st.session_state.flashcards_df, tag_input)
        st.success(f"Appended flashcards to `{CSV_PATH}`.")


# --- Main Application ---
def render_flashcards(df: pd.DataFrame) -> None:
    """
    Initializes and runs the Streamlit flashcard application interface.

    This function provides a UI for reviewing flashcards and gives the option
    to append the current set of flashcards to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame containing flashcard data.
                           Must include the columns defined in DF_COLUMNS.
    """
    local_css("src/lib/flashcards_style.css")

    if "flashcards_df" not in st.session_state:
        # Use a copy to prevent modifying the original DataFrame outside this session
        st.session_state.flashcards_df = df.copy()

    tab1, tab2 = st.tabs(["Review", "View All"])

    with tab1:
        render_review_tab()
    with tab2:
        render_view_all_tab()

if __name__ == "__main__":
    # --- Example Usage ---
    # This block demonstrates how to use the module.
    # In a real application, you would load your DataFrame from a database, API, etc.

    # Create a sample DataFrame for demonstration
    now = datetime.now()
    sample_data = [
        {
            QUESTION: "Who is building the skibidiest study-chatbot of all?",
            ANSWER: "Patrick is building the skibidiest study-chatbot of all.",
            DATE_ADDED: now - timedelta(days=10),
            NEXT_APPEARANCE: now - timedelta(days=1), # Due for review
            TAG: ["personal", "other"],
        },
        {
            QUESTION: "What does SQL stand for?",
            ANSWER: "Structured Query Language",
            DATE_ADDED: now - timedelta(days=5),
            NEXT_APPEARANCE: now + timedelta(days=5), # Not due yet
            TAG: ["cs", "dbms"],
        },
        {
            QUESTION: "What is the time complexity of binary search?",
            ANSWER: "O(log n)",
            DATE_ADDED: now - timedelta(days=2),
            NEXT_APPEARANCE: now - timedelta(hours=1), # Due for review
            TAG: ["cs", "ds/algo"],
        },
    ]

    # Ensure the DataFrame has the correct data types
    sample_df = pd.DataFrame(sample_data)
    sample_df[DATE_ADDED] = pd.to_datetime(sample_df[DATE_ADDED])
    sample_df[NEXT_APPEARANCE] = pd.to_datetime(sample_df[NEXT_APPEARANCE])

    # Run the application with the sample data
    render_flashcards(sample_df)