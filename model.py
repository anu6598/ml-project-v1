import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from datetime import datetime

st.set_page_config(page_title="Signup Bot Detector", layout="wide")
st.title("🤖 Signup Bot Detector")
st.write("Upload your API log CSV to identify bot-like signup activity and gain insights.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # Convert datetime columns
    df['first_request_time'] = pd.to_datetime(df['first_request_time'])
    df['last_request_time'] = pd.to_datetime(df['last_request_time'])
    df['duration'] = (df['last_request_time'] - df['first_request_time']).dt.total_seconds()
    df['requests_per_second'] = df['total_requests'] / df['duration'].replace(0, 1)

    # Fill NA values
    df.fillna({
        'unique_user_agents': 0,
        'api_paths_called': '',
        'vpn_detection_flags': ''
    }, inplace=True)

    # Feature Engineering
    df['api_path_count'] = df['api_paths_called'].apply(lambda x: len(str(x).split(',')))
    df['vpn_flagged'] = df['vpn_detection_flags'].apply(lambda x: 1 if 'vpn' in str(x).lower() else 0)

    # Model
    features = df[['total_requests', 'unique_user_agents', 'duration', 'requests_per_second', 'api_path_count', 'vpn_flagged']]
    model = IsolationForest(contamination=0.05, random_state=42)
    df['prediction'] = model.fit_predict(features)
    df['label'] = df['prediction'].apply(lambda x: 'Bot' if x == -1 else 'Genuine')

    # Display dataframe
    st.subheader("📊 Labeled Data")
    st.dataframe(df[['x_real_ip', 'label', 'total_requests', 'unique_user_agents', 'vpn_flagged']])

    # Download
    st.download_button("Download Result as CSV", df.to_csv(index=False), "bot_detection_output.csv", "text/csv")

    # Visual Insights
    st.subheader("🔍 Key Insights with Graphs")
    insights = {
        'Signups Over Time': ('first_request_time', 'count'),
        'Bot vs Genuine': ('label', 'count'),
        'VPN Usage Distribution': ('vpn_flagged', 'count'),
        'Requests per Second Distribution': ('requests_per_second', 'hist'),
        'Duration of Sessions': ('duration', 'hist'),
        'API Path Count Distribution': ('api_path_count', 'hist'),
        'Unique User Agents vs Requests': ('scatter', ('unique_user_agents', 'total_requests')),
        'Bot Detection by VPN': ('bar', ('vpn_flagged', 'label')),
        'Bot Count by API Path Count': ('bar', ('api_path_count', 'label')),
        'Daily Signup Spike': ('line', 'first_request_time')
    }

    for title, config in insights.items():
        st.markdown(f"### {title}")
        fig, ax = plt.subplots(figsize=(8, 4))

        if config[1] == 'count':
            sns.countplot(data=df, x=config[0], ax=ax)

        elif config[1] == 'hist':
            sns.histplot(data=df, x=config[0], bins=30, kde=True, ax=ax)

        elif config[0] == 'scatter':
            sns.scatterplot(data=df, x=config[1][0], y=config[1][1], hue='label', ax=ax)

        elif config[0] == 'bar':
            sns.barplot(data=df, x=config[1][0], y=config[1][1], estimator=lambda x: sum(pd.Series(x) == 'Bot'), ax=ax)

        elif config[1] == 'line':
            df_grouped = df.groupby(df[config[0]].dt.date).size()
            ax.plot(df_grouped.index, df_grouped.values, marker='o')
            ax.set_xlabel("Date")
            ax.set_ylabel("Signups")

        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to begin analysis.")
