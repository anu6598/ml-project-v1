import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bot Detection from Signup Logs", layout="wide")
st.title("🚨 Bot vs Human Signup Detection")

# File upload
uploaded_file = st.file_uploader("Upload CSV File with API Signup Logs", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Data Preprocessing
    df['first_request_time'] = pd.to_datetime(df['first_request_time'])
    df['last_request_time'] = pd.to_datetime(df['last_request_time'])
    df['duration'] = (df['last_request_time'] - df['first_request_time']).dt.total_seconds()

    # Create binary VPN flag
    df['vpn_flagged'] = df['vpn_detection_flags'].apply(lambda x: 1 if str(x).lower() in ['yes', 'true', '1'] else 0)

    # Features for ML
    features = df[['total_requests', 'duration', 'unique_user_agents', 'vpn_flagged']].copy()

    # Fill missing or invalid values
    features = features.fillna(0)
    clf = IsolationForest(contamination=0.1, random_state=42)
    df['bot_prediction'] = clf.fit_predict(features)
    df['label'] = df['bot_prediction'].apply(lambda x: 'Bot' if x == -1 else 'Genuine')

    st.success("✅ Bot Detection Complete")
    st.dataframe(df[['x_real_ip', 'total_requests', 'duration', 'unique_user_agents', 'vpn_flagged', 'label']])

    # Download option
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results as CSV", csv, "classified_signups.csv", "text/csv")

    # High-level insights
    st.subheader("📊 Top 10 Insights")
    insights = []

    insights.append(f"1. Total Records: {len(df)}")
    insights.append(f"2. Bots Detected: {sum(df['label'] == 'Bot')} ({sum(df['label'] == 'Bot') / len(df):.2%})")
    insights.append(f"3. Genuine Users: {sum(df['label'] == 'Genuine')} ({sum(df['label'] == 'Genuine') / len(df):.2%})")
    insights.append(f"4. Average Duration for Bots: {df[df['label'] == 'Bot']['duration'].mean():.2f} sec")
    insights.append(f"5. Average Duration for Humans: {df[df['label'] == 'Genuine']['duration'].mean():.2f} sec")
    insights.append(f"6. Most Common VPN Usage among Bots: {df[df['label'] == 'Bot']['vpn_flagged'].sum()} IPs")
    insights.append(f"7. Top Suspicious IP: {df[df['label'] == 'Bot'].sort_values('total_requests', ascending=False)['x_real_ip'].head(1).values[0]}")
    insights.append(f"8. IP with Most Unique User Agents: {df.sort_values('unique_user_agents', ascending=False)['x_real_ip'].head(1).values[0]}")
    insights.append(f"9. Average Requests by Bots: {df[df['label'] == 'Bot']['total_requests'].mean():.2f}")
    insights.append(f"10. VPN Usage Rate: {df['vpn_flagged'].mean():.2%}")

    for i in insights:
        st.markdown(f"- {i}")

    # Optional: Chart
    st.subheader("📈 Signup Volume by Type")
    chart_data = df.groupby('label')['x_real_ip'].count().reset_index(name='count')
    fig, ax = plt.subplots()
    sns.barplot(data=chart_data, x='label', y='count', ax=ax)
    st.pyplot(fig)

else:
    st.info("📂 Upload a CSV file to get started. Headers expected: x_real_ip, total_requests, first_request_time, last_request_time, unique_user_agents, api_paths_called, vpn_detection_flags")
