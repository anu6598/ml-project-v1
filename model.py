import streamlit as st
import pandas as pd
import plotly.express as px

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Video Usage Trends", "Suspicious Users", "User Analysis"])

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if page == "Video Usage Trends":
        st.title("Video Usage Trends")
        
        # Top _subjects Consumption
        top__subjects = ['Anatomy', 'Surgery', 'ENT', 'Medicine', 'Community Medicine']
        _subject_hours = df[df['_subject_title'].isin(top__subjects)].groupby('_subject_title')['actual_hours'].sum().reset_index()
        fig = px.bar(_subject_hours, x='_subject_title', y='actual_hours', title="Hours Watched per _subject", color='_subject_title')
        st.plotly_chart(fig)
        
        # Top 50 Users by Watch Time
        top_users = df.groupby('user_id')['actual_hours'].sum().reset_index().sort_values(by='actual_hours', ascending=False).head(50)
        st.write("### Top 50 Users by Actual Hours")
        st.dataframe(top_users)
    
    elif page == "Suspicious Users":
        st.title("Suspicious Users Detection")
        # Suspicious user logic (Assuming the model is already integrated)
        st.write("Suspicious users will be displayed here.")
    
    elif page == "User Analysis":
        st.title("User Analysis")
        user_id = st.text_input("Enter User ID:")
        if user_id:
            user_data = df[df['user_id'] == user_id]
            st.write("### User Data")
            st.dataframe(user_data)
            # Model prediction logic (if needed)
