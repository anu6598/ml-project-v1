import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import StringIO

# Set page config
st.set_page_config(page_title="Bot Detection Dashboard", layout="wide")
st.title("🚨 Signup Bot Detection Dashboard")

# Sidebar summary and download
with st.sidebar:
    st.header("📊 Summary & Download")

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

    # Graph 1: Bot vs Genuine
    col1, col2 = st.columns(2)
    with col1:
        count_data = df['label'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(x=count_data.index, y=count_data.values, ax=ax, palette="Set2")
        ax.set_title("Bot vs Genuine Signups")
        ax.set_ylabel("Number of Unique IPs")
        st.pyplot(fig)
        st.caption("This chart shows the count of IPs identified as bots or genuine users based on rule-based detection.")

    # Graph 2: Session Duration
    with col2:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.boxplot(x='label', y='session_duration', data=df, ax=ax, palette="coolwarm")
        ax.set_title("Session Duration by Label")
        ax.set_ylabel("Session Duration (seconds)")
        st.pyplot(fig)
        st.caption("Bots tend to have shorter session durations compared to genuine users.")

    # Graph 3: Bot Request Distribution
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(df[df['label'] == 'Bot']['total_requests'], bins=30, color='red', ax=ax)
        ax.set_title("Bot Users: Total Requests Distribution")
        st.pyplot(fig)
        st.caption("Bots usually make a high number of requests in a short time.")

    # Graph 4: Genuine Request Distribution
    with col4:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(df[df['label'] == 'Genuine']['total_requests'], bins=30, color='green', ax=ax)
        ax.set_title("Genuine Users: Total Requests Distribution")
        st.pyplot(fig)
        st.caption("Genuine users usually make fewer requests over a longer period.")

        # 📈 Line Graph: Daily Unique Signups
    st.subheader("📅 Daily Signup Spike Detection")

    # Convert timestamp to just the date
    df['signup_date'] = df['first_request_time'].dt.date

    # Count unique IPs per day
    daily_signups = df.groupby('signup_date')['x_real_ip'].nunique().reset_index()
    daily_signups.columns = ['Date', 'Unique_Signups']

    # Plot the line chart# 📈 Line Graph: Total Requests over Time
st.subheader("📅 Total Requests Over Time")

# Round start_time to hourly granularity for better chart readability
df['start_hour'] = df['first_request_time'].dt.floor('H')

# Group by start_hour and sum total requests
request_trend = df.groupby('start_hour')['total_requests'].sum().reset_index()

# Plot
fig = px.line(request_trend, x='start_hour', y='total_requests', markers=True,
              title="Total Requests Over Time")

fig.update_layout(
    xaxis_title="Start Time (Hourly)",
    yaxis_title="Total Requests",
    title_x=0.5
)

st.plotly_chart(fig, use_container_width=True)
st.caption("This chart shows total API requests over time. Spikes may indicate bot surges.")

