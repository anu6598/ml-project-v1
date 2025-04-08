import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama
from io import StringIO
from datetime import datetime

# Set page config
st.set_page_config(page_title="Bot Detection Dashboard", layout="wide")
st.title("🚨 Signup Bot Detection Dashboard")

# Sidebar summary and download
with st.sidebar:
    st.header("📊 Summary & Download")
    download_ready = False

# File uploader
uploaded_file = st.file_uploader("Upload your API logs CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    df['first_request_time'] = pd.to_datetime(df['first_request_time'], errors='coerce')
    df['last_request_time'] = pd.to_datetime(df['last_request_time'], errors='coerce')
    df['session_duration'] = (df['last_request_time'] - df['first_request_time']).dt.total_seconds()
    df['api_paths_called'] = df['api_paths_called'].fillna('').astype(str)
    df['unique_user_agents'] = df['unique_user_agents'].fillna('').astype(str)
    df['vpn_detection_flags'] = df['vpn_detection_flags'].fillna('').astype(str)

    # Feature Engineering
    df['num_user_agents'] = df['unique_user_agents'].apply(lambda x: len(set(x.split(','))))
    df['num_api_paths'] = df['api_paths_called'].apply(lambda x: len(set(x.split(','))))
    df['vpn_flag_count'] = df['vpn_detection_flags'].apply(lambda x: x.count('True'))

    # Rule-based Labeling
    def detect_bot(row):
        score = 0
        if row['total_requests'] > 100: score += 1
        if row['session_duration'] and row['session_duration'] < 10: score += 1
        if row['num_user_agents'] > 3: score += 1
        if row['num_api_paths'] > 10: score += 1
        if row['vpn_flag_count'] > 0: score += 1
        return 'Bot' if score >= 3 else 'Genuine'

    df['label'] = df.apply(detect_bot, axis=1)

    # Sidebar summary
    total = len(df)
    bots = (df['label'] == 'Bot').sum()
    genuine = total - bots
    bot_pct = round((bots / total) * 100, 2)

    with st.sidebar:
        st.markdown(f"**Total Signups:** {total}")
        st.markdown(f"**Bots Detected:** {bots} ({bot_pct}%)")
        st.markdown(f"**Genuine Signups:** {genuine}")
        csv_download = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Processed CSV", csv_download, file_name="processed_log.csv")

    st.subheader("🔍 Insights from the Data")

    # Visualization 1: Bot vs Genuine
    col1, col2 = st.columns(2)
    with col1:
        count_data = df['label'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=count_data.index, y=count_data.values, ax=ax, palette="Set2")
        ax.set_title("Bot vs Genuine Signups")
        ax.set_ylabel("Number of Unique IPs")
        st.pyplot(fig)
        st.caption("This chart shows the count of unique IP addresses identified as bots or genuine users.")

    # Visualization 2: Session duration by label
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x='label', y='session_duration', data=df, ax=ax, palette="coolwarm")
        ax.set_title("Session Duration by Label")
        ax.set_ylabel("Session Duration (seconds)")
        st.pyplot(fig)
        st.caption("Bot users often have very short session durations compared to genuine users.")

    # Additional insights
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots()
        sns.histplot(df[df['label'] == 'Bot']['total_requests'], bins=30, color='red', ax=ax)
        ax.set_title("Bot Users: Total Requests Distribution")
        st.pyplot(fig)
        st.caption("Bots usually make an unusually high number of requests in a short span of time.")

    with col4:
        fig, ax = plt.subplots()
        sns.histplot(df[df['label'] == 'Genuine']['total_requests'], bins=30, color='green', ax=ax)
        ax.set_title("Genuine Users: Total Requests Distribution")
        st.pyplot(fig)
        st.caption("Genuine users typically generate fewer total API requests.")

    # Ollama Chat Interface
    st.subheader("🧠 Ask Questions About Your Data")

    user_input = st.text_input("Ask a question about the signup data:")

    if user_input:
        # Convert dataframe to string
        df_csv_string = df.to_csv(index=False)

        prompt = f"""
        You are a data analyst. Here is the signup API log data:

        {df_csv_string}

        Now answer this question: {user_input}
        """

        try:
            response = ollama.chat(
                model="llama3",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            st.markdown(f"**Answer:** {response['message']['content']}")
        except Exception as e:
            st.error("Error using Ollama. Please ensure it's running and accessible.")
