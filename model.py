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

# Streamlit App Design
st.set_page_config(page_title="User Insights Dashboard", layout="wide")
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f4f8;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Input
st.title("User Insights Dashboard: Playback & License Consumption")
user_id_input = st.text_input("Enter User ID to view details:")

if user_id_input:
    # Filter user data
    user_playback = playback_data[playback_data['user_id'] == user_id_input]
    user_license = license_data[license_data['user_id'] == user_id_input]

    if user_playback.empty and user_license.empty:
        st.warning("No data found for the given User ID.")
    else:
        st.header(f"Insights for User ID: {user_id_input}")

        # Playback summary
        st.subheader("Playback Summary")
        total_hours = user_playback['actual_hours'].sum()
        st.metric("Total Hours Watched", f"{total_hours:.2f} hours")

        subject_wise = user_playback.groupby('_subject_title')['actual_hours'].sum().reset_index()
        fig_subject = px.bar(subject_wise, x='_subject_title', y='actual_hours',
                             title='Hours Watched per Subject', template='plotly_dark')
        fig_subject.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_subject, use_container_width=True)

        # Viewing Pattern
        st.subheader("Viewing Pattern Over Time")
        user_playback['event_date'] = pd.to_datetime(user_playback['event_date'])
        daily_view = user_playback.groupby('event_date')['actual_hours'].sum().reset_index()
        fig_timeline = px.line(daily_view, x='event_date', y='actual_hours',
                               title='Daily Viewing Trend', template='plotly_dark')
        fig_timeline.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_timeline, use_container_width=True)

        # License consumption summary
        st.subheader("License Consumption Summary")
        license_count = user_license.shape[0]
        st.metric("Total Licenses Consumed", f"{license_count}")

        platform_usage = user_license['platform'].value_counts().reset_index()
        platform_usage.columns = ['Platform', 'Count']
        fig_platform = px.pie(platform_usage, names='Platform', values='Count',
                              title='Platform Usage Distribution', template='plotly_dark')
        fig_platform.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_platform, use_container_width=True)

        # License consumption patterns
        st.subheader("Detailed License Consumption Patterns")
        license_trend = user_license.groupby('date').size().reset_index(name='licenses_issued')
        fig_license_trend = px.line(license_trend, x='date', y='licenses_issued',
                                    title='License Issuance Over Time', template='plotly_dark')
        fig_license_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_license_trend, use_container_width=True)

        wv_levels = user_license['wv_level'].value_counts().reset_index()
        wv_levels.columns = ['WV Level', 'Count']
        fig_wv = px.bar(wv_levels, x='WV Level', y='Count',
                        title='WV Level Distribution', template='plotly_dark')
        fig_wv.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_wv, use_container_width=True)

        country_dist = user_license['country'].value_counts().reset_index()
        country_dist.columns = ['Country', 'Count']
        fig_country = px.pie(country_dist, names='Country', values='Count',
                             title='Country-wise License Consumption', template='plotly_dark')
        fig_country.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_country, use_container_width=True)

        # Completion Status
        st.subheader("Completion Status")
        completion = user_playback.groupby('_subject_title')['percentage'].mean().reset_index()
        completion.columns = ['Subject', 'Avg Completion %']
        fig_completion = px.bar(completion, x='Subject', y='Avg Completion %',
                                title='Subject Completion Status', template='plotly_dark')
        fig_completion.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_completion, use_container_width=True)

        # Fast vs. Slow Finisher
        st.subheader("Finisher Category")
        total_lessons = user_playback['_lesson_id'].nunique()
        completed_lessons = user_playback[user_playback['percentage'] >= 85]['_lesson_id'].nunique()
        finisher_type = "Fast Finisher" if completed_lessons / total_lessons >= 0.8 else "Slow Finisher"
        st.success(f"This user is categorized as a: {finisher_type}")

        # Flexible Query Section
        st.subheader("Ask a Custom Question")
        query = st.text_input("Type your question here:")

        def answer_query(q):
            q = q.lower()
            if "top most viewed subject" in q:
                top_subject = subject_wise.sort_values("actual_hours", ascending=False).iloc[0]['_subject_title']
                return f"Top most viewed subject: {top_subject}"
            elif "total hours" in q:
                return f"Total hours watched: {total_hours:.2f} hours"
            elif "licenses consumed" in q:
                return f"Total licenses consumed: {license_count}"
            else:
                return "Query not recognized. Please try a different question."

        if query:
            st.info(answer_query(query))

else:
    st.info("Please enter a User ID to load insights.")
