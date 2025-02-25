import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration with a modern theme
st.set_page_config(
    page_title="User Insights Dashboard",
    page_icon="📊",
    layout="wide",
)

# Custom CSS for background and section styling
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #1f1c2c, #928dab);
        color: white;
    }
    .stApp {
        background-color: rgba(0, 0, 0, 0.1);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);
    }
    h1, h2, h3 {
        color: #FFD700;
    }
    .metric-container {
        background-color: rgba(0, 0, 0, 0.3);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load datasets
@st.cache_data
def load_data():
    playback_data = pd.read_csv("video_playback_data.csv")
    license_data = pd.read_csv("license_consumption_data.csv")
    return playback_data, license_data

playback_data, license_data = load_data()

# Title and Input
st.title("📈 User Insights Dashboard: Playback & License Consumption")
st.info("Analyze playback trends and license consumption with interactive insights.")

user_id_input = st.text_input("🔍 Enter User ID to view details:")

if user_id_input:
    user_playback = playback_data[playback_data['user_id'] == user_id_input]
    user_license = license_data[license_data['user_id'] == user_id_input]

    if user_playback.empty and user_license.empty:
        st.warning("⚠️ No data found for the given User ID.")
    else:
        st.header(f"✨ Insights for User ID: {user_id_input}")

        # Playback summary
        st.subheader("🎬 Playback Summary")
        total_hours = user_playback['actual_hours'].sum()
        with st.container():
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Total Hours Watched", f"{total_hours:.2f} hours")
            st.markdown("</div>", unsafe_allow_html=True)

        subject_wise = user_playback.groupby('_subject_title')['actual_hours'].sum().reset_index()
        fig_subject = px.bar(subject_wise, x='_subject_title', y='actual_hours',
                             title='Hours Watched per Subject',
                             color='_subject_title',
                             color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_subject, use_container_width=True)

        # Viewing Pattern
        st.subheader("📅 Viewing Pattern Over Time")
        user_playback['event_date'] = pd.to_datetime(user_playback['event_date'])
        daily_view = user_playback.groupby('event_date')['actual_hours'].sum().reset_index()
        fig_timeline = px.line(daily_view, x='event_date', y='actual_hours',
                               title='Daily Viewing Trend',
                               markers=True, line_shape="spline",
                               color_discrete_sequence=['#FF5733'])
        st.plotly_chart(fig_timeline, use_container_width=True)

        # License consumption summary
        st.subheader("🔑 License Consumption Summary")
        license_count = user_license.shape[0]
        st.metric("Total Licenses Consumed", f"{license_count}")

        platform_usage = user_license['platform'].value_counts().reset_index()
        platform_usage.columns = ['Platform', 'Count']
        fig_platform = px.pie(platform_usage, names='Platform', values='Count',
                              title='Platform Usage Distribution',
                              color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_platform, use_container_width=True)

        # Device & Location Insights
        st.subheader("🖥️ Device and Region Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Unique Devices", user_playback['_d_id'].nunique())
        with col2:
            st.metric("Regions Accessed", user_playback['_region'].nunique())

        # User Journey Insights
        st.subheader("🚀 User Journey Insights")
        try:
            subject_order = (
                user_playback.groupby(['_subject_title', 'lesson_id'])
                .agg({'event_date': 'min'})
                .reset_index()
                .sort_values('event_date')
            )
            st.dataframe(subject_order, use_container_width=True)
        except KeyError:
            st.error("Required columns for User Journey Insights are missing.")

        # Completion Status
        st.subheader("✅ Completion Status")
        completion = user_playback.groupby('_subject_title')['percentage'].mean().reset_index()
        completion.columns = ['Subject', 'Avg Completion %']
        fig_completion = px.bar(completion, x='Subject', y='Avg Completion %',
                                title='Subject Completion Status',
                                color='Subject',
                                color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig_completion, use_container_width=True)

        # Finisher Category
        st.subheader("🏁 Finisher Category")
        total_lessons = user_playback['lesson_id'].nunique()
        completed_lessons = user_playback[user_playback['percentage'] >= 85]['lesson_id'].nunique()
        finisher_type = "Fast Finisher" if completed_lessons / total_lessons >= 0.8 else "Slow Finisher"
        st.success(f"This user is categorized as a: {finisher_type}")

        # Flexible Query Section
        st.subheader("💡 Ask a Custom Question")
        query = st.text_input("Type your question here:")

        def answer_query(q):
            q = q.lower()
            if "top most viewed subject" in q:
                top_subject = subject_wise.sort_values("actual_hours", ascending=False).iloc[0]['_subject_title']
                return f"\ud83d\udcdd Top most viewed subject: {top_subject}" 
            elif "total hours" in q:
                return f"\ud83d\udd50 Total hours watched: {total_hours:.2f} hours"
            else:
                return "⚡ Query not recognized. Please try a different question."

        if query:
            st.info(answer_query(query))

else:
    st.info("\ud83d\udcd6 Please enter a User ID to load insights.")
