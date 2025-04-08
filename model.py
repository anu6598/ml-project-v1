import os
os.environ["OLLAMA_HOST"] = "http://localhost:11434"


# streamlit_app.py

import streamlit as st
import pandas as pd
from datetime import datetime
import ollama

st.set_page_config(page_title="Human vs Bot Q&A", page_icon="🤖")
st.title("🤖 Human vs Bot Signup Logs — Q&A Powered by Ollama")

uploaded_file = st.file_uploader("📂 Upload API Logs CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Display raw data preview
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Optional preprocessing (if needed by you or for context)
    try:
        df['first_request_time'] = pd.to_datetime(df['first_request_time'])
        df['last_request_time'] = pd.to_datetime(df['last_request_time'])
        df['session_duration'] = (df['last_request_time'] - df['first_request_time']).dt.total_seconds()
    except:
        st.warning("Couldn't parse datetime columns automatically.")

    # Ask Questions Section
    st.subheader("💬 Ask your data anything")

    user_input = st.text_input("🔎 E.g. 'Which IP had the highest total requests on April 3rd?'")

    if user_input:
        with st.spinner("Thinking with Ollama..."):
            response = ollama.chat(
                model="llama3",  # or any model you've downloaded locally
                messages=[
                    {"role": "system", "content": "You're a data analyst. Answer based only on the CSV provided."},
                    {"role": "user", "content": f"{user_input}\n\nHere is the CSV data:\n{df.to_csv(index=False)}"}
                ]
            )
            st.markdown("### 🧠 Answer")
            st.write(response['message']['content'])
