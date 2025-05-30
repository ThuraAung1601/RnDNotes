import streamlit as st
import pandas as pd

CSV_PATH = "belebele_myanmar.csv"

# Load CSV as tab-separated file
df = pd.read_csv(CSV_PATH, encoding="utf-8", sep="\t")
df.columns = df.columns.str.strip()

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = 0
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False

# Get current row
row = df.iloc[st.session_state.index]

# Show ID (row number) and Question Number (from CSV)
st.markdown(f"## ID: {st.session_state.index + 1} | Question Number: {row['question_number']}")

if st.session_state.edit_mode:
    # Editable fields
    new_link = st.text_input("Link", row.get("link", ""))
    new_question = st.text_area("Question", row["question"])
    new_answers = [st.text_input(f"Answer {i}", row[f"mc_answer{i}"]) for i in range(1, 5)]
    new_correct = st.selectbox(
        "Correct Answer Number", options=[1, 2, 3, 4], index=int(row["correct_answer_num"]) - 1
    )

    if st.button("Save"):
        df.at[st.session_state.index, "link"] = new_link
        df.at[st.session_state.index, "question"] = new_question
        df.at[st.session_state.index, "correct_answer_num"] = new_correct
        for i in range(1, 5):
            df.at[st.session_state.index, f"mc_answer{i}"] = new_answers[i - 1]
        df.to_csv(CSV_PATH, index=False, sep="\t")  # Save again as TSV
        st.success("✅ Changes saved!")
        st.session_state.edit_mode = False
else:
    # Read-only display
    try:
        st.markdown(f"**Link:** [{row['link']}]({row['link']})")
    except KeyError:
        st.markdown("**Link:** Not available")

    st.markdown(f"**Flores Passage:**")
    st.markdown(f"{row['flores_passage']}")

    st.markdown(f"**Question:** {row['question']}")
    for i in range(1, 5):
        st.markdown(f"{i}. {row[f'mc_answer{i}']}")
    st.markdown(f"**Correct Answer:** {row['correct_answer_num']}")

    if st.button("Edit"):
        st.session_state.edit_mode = True

# Navigation buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("⬅️ Prev"):
        st.session_state.index = max(0, st.session_state.index - 1)
        st.session_state.edit_mode = False
with col2:
    if st.button("➡️ Next"):
        st.session_state.index = (st.session_state.index + 1) % len(df)
        st.session_state.edit_mode = False
