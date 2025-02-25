import streamlit as st
import pandas as pd
import plotly.express as px

# Load datasets (replace with actual file paths or database connections)
@st.cache_data
def load_data():
    playback_data = pd.read_csv("video_playback_data.csv")
    license_data = pd.read_csv("license_consumption_data.csv")
    return playback_data, license_data

playback_data, license_data = load_data()

# Set custom background and page config
st.set_page_config(page_title="User Insights Dashboard", layout="wide")

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(to right, #1f4037, #99f2c8);
            color: white;
        }
        .stHeader, .stSubheader {
            color: #f0f0f0;
        }
        .stMetric {
            background-color: rgba(0, 0, 0, 0.2);
            padding: 10px;
            border-radius: 15px;
        }
        .css-1aumxhk {background-color: transparent !important;}
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Input
st.title("🌟 User Insights Dashboard: Playback & License Consumption")
user_id_input = st.text_input("🔍 Enter User ID to view details:")

if user_id_input:
    # Filter user data
    user_playback = playback_data[playback_data['user_id'] == user_id_input]
    user_license = license_data[license_data['user_id'] == user_id_input]

    if user_playback.empty and user_license.empty:
        st.warning("⚠️ No data found for the given User ID.")
    else:
        st.header(f"📊 Insights for User ID: {user_id_input}")

        # Playback summary
        st.subheader("🎬 Playback Summary")
        total_hours = user_playback['actual_hours'].sum()
        st.metric("⏱️ Total Hours Watched", f"{total_hours:.2f} hours")

        subject_wise = user_playback.groupby('_subject_title')['actual_hours'].sum().reset_index()
        fig_subject = px.bar(subject_wise, x='_subject_title', y='actual_hours', title='Hours Watched per Subject')
        fig_subject.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_subject, use_container_width=True)

        # Viewing Pattern
        st.subheader("📅 Viewing Pattern Over Time")
        user_playback['event_date'] = pd.to_datetime(user_playback['event_date'])
        daily_view = user_playback.groupby('event_date')['actual_hours'].sum().reset_index()
        fig_timeline = px.line(daily_view, x='event_date', y='actual_hours', title='Daily Viewing Trend')
        fig_timeline.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_timeline, use_container_width=True)

        # License consumption summary
        st.subheader("🔑 License Consumption Summary")
        license_count = user_license.shape[0]
        st.metric("🎟️ Total Licenses Consumed", f"{license_count}")

        platform_usage = user_license['platform'].value_counts().reset_index()
        platform_usage.columns = ['Platform', 'Count']
        fig_platform = px.pie(platform_usage, names='Platform', values='Count', title='Platform Usage Distribution')
        fig_platform.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_platform, use_container_width=True)

        # Device & Location Insights
        st.subheader("📍 Device and Region Insights")
        device_count = user_playback['_d_id'].nunique()
        region_count = user_playback['_region'].nunique()
        st.metric("📱 Unique Devices", device_count)
        st.metric("🌎 Regions Accessed", region_count)

        # User Journey Insights
        st.subheader("🛤️ User Journey Insights")
        subject_order = (
            user_playback.groupby(['_subject_title', '_lesson_id'])
            .agg({'event_date': 'min'})
            .reset_index()
            .sort_values('event_date')
        )
        st.dataframe(subject_order, use_container_width=True)

        # Completion Status
        st.subheader("✅ Completion Status")
        completion = user_playback.groupby('_subject_title')['percentage'].mean().reset_index()
        completion.columns = ['Subject', 'Avg Completion %']
        fig_completion = px.bar(completion, x='Subject', y='Avg Completion %', title='Subject Completion Status')
        fig_completion.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_completion, use_container_width=True)

        # Fast vs. Slow Finisher
        st.subheader("⏩ Finisher Category")
        total_lessons = user_playback['_lesson_id'].nunique()
        completed_lessons = user_playback[user_playback['percentage'] >= 85]['_lesson_id'].nunique()
        finisher_type = "⚡ Fast Finisher" if completed_lessons / total_lessons >= 0.8 else "🐢 Slow Finisher"
        st.success(f"This user is categorized as a: {finisher_type}")

        # Flexible Query Section
        st.subheader("💬 Ask a Custom Question")
        query = st.text_input("Type your question here:")

        def answer_query(q):
            q = q.lower()
            if "top most viewed subject" in q:
                top_subject = subject_wise.sort_values("actual_hours", ascending=False).iloc[0]['_subject_title']
                return f"🏆 Top most viewed subject: {top_subject}"
            elif "total hours" in q:
                return f"🕒 Total hours watched: {total_hours:.2f} hours"
            else:
                return "❓ Query not recognized. Please try a different question."

        if query:
            st.info(answer_query(query))

else:
    st.info("🔑 Please enter a User ID to load insights.")
