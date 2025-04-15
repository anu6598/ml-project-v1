import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="360° Attack Detection", layout="wide")

st.title("🚨 360° API Attack Detection System")
st.write("Upload your log file (1-day data) and detect suspicious activity across Web, iOS, and Android platforms.")

uploaded_file = st.file_uploader("📁 Upload 1-Day API Log CSV", type=["csv"])

@st.cache_data(show_spinner=False)
def read_data(file):
    df = pd.read_csv(file)
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['minute'] = df['start_time'].dt.floor('min')
    df['platform'] = df['dr_platform'].fillna('web')
    return df

def detect_brute_force(df):
    brute_df = df[df['request_path'].str.contains("login", case=False, na=False)]
    grouped = brute_df.groupby(['x_real_ip', 'minute']).size().reset_index(name='count')
    brute_ips = grouped[grouped['count'] > 10]['x_real_ip'].unique()
    return df[df['x_real_ip'].isin(brute_ips)]

def detect_vpn_geo(df):
    geo = df.groupby('dr_uid')['x_country_code'].nunique().reset_index()
    flagged = geo[geo['x_country_code'] > 2]['dr_uid']
    return df[df['dr_uid'].isin(flagged)]

def detect_bots(df):
    suspicious_ua = df['user_agent'].str.contains("bot|curl|python|scrapy|wget", case=False, na=False)
    low_duration = df['duration'].astype(float) < 0.3
    return df[suspicious_ua | low_duration]

def detect_ddos(df):
    volume = df.groupby(['x_real_ip', 'minute']).size().reset_index(name='count')
    high_vol_ips = volume[volume['count'] > 100]['x_real_ip'].unique()
    return df[df['x_real_ip'].isin(high_vol_ips)]

def summarize_detection(name, df, platform_col='platform'):
    summary = df.groupby(platform_col).size().reset_index(name='Suspicious Requests')
    summary['attack_type'] = name
    return summary

if uploaded_file:
    with st.spinner("Reading file..."):
        df = read_data(uploaded_file)
    st.success("✅ File loaded successfully!")

    st.subheader("🔍 Sample of Uploaded Data")
    st.dataframe(df.head(20))

    if st.button("🚨 Run Attack Detection"):
        with st.spinner("Detecting attack patterns..."):

            brute_df = detect_brute_force(df)
            vpn_df = detect_vpn_geo(df)
            bot_df = detect_bots(df)
            ddos_df = detect_ddos(df)

            attack_summary = pd.concat([
                summarize_detection("Brute Force", brute_df),
                summarize_detection("VPN/Geo Switch", vpn_df),
                summarize_detection("Bot-like", bot_df),
                summarize_detection("DDoS", ddos_df)
            ])

            st.success("Detection completed!")

            st.subheader("📊 Attack Summary by Platform")
            st.dataframe(attack_summary)

            st.subheader("🔐 Brute Force (top 10)")
            st.dataframe(brute_df[['start_time', 'x_real_ip', 'request_path', 'dr_uid']].head(10))

            st.subheader("🕵️ VPN/Geo Switch (top 10)")
            st.dataframe(vpn_df[['start_time', 'x_country_code', 'dr_uid', 'x_real_ip']].head(10))

            st.subheader("🤖 Bots Detected (top 10)")
            st.dataframe(bot_df[['start_time', 'user_agent', 'x_real_ip', 'duration']].head(10))

            st.subheader("🌊 DDoS Suspects (top 10)")
            st.dataframe(ddos_df[['start_time', 'x_real_ip', 'request_path']].head(10))
