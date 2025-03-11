import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Load trained model
MODEL_PATH = "suspicious_model.pkl"
model = joblib.load(MODEL_PATH)

# Streamlit page configuration
st.set_page_config(page_title="User Analysis Dashboard", layout="wide")

# Custom CSS for blue ombre background and white text
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .css-1aumxhk {
        color: white !important;
    }
    .stTextInput, .stFileUploader {
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button {
        border-radius: 10px;
        background-color: #1f4e78;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Video Usage Trends", "🔍 Suspicious Users Detection"])

def load_and_preprocess(csv_file):
    """ Load and preprocess data for analysis """
    df = pd.read_csv(csv_file)
    df['user_id'] = df['user_id'].astype(str).str.strip().str.lower()
    
    # Aggregate user interactions
    features = df.groupby('user_id').agg({
        'actual_hours': 'sum',
        '_pause': 'sum',
        '_seek': 'sum',
        'lesson_id': 'nunique',
    }).rename(columns={'lesson_id': 'unique_lessons'})

    return df, features

# **PAGE 1: Video Usage Trends**
if page == "📊 Video Usage Trends":
    st.title("📊 Video Usage Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df, user_features = load_and_preprocess(uploaded_file)

        st.subheader("1️⃣ How users are navigating videos (Sankey Chart)")
        user_flow = df.groupby(['user_id', 'lesson_id']).size().reset_index(name='count')

        # Create node labels
        unique_users = user_flow['user_id'].unique().tolist()
        unique_lessons = user_flow['lesson_id'].unique().tolist()
        all_labels = unique_users + unique_lessons

        # Map users & lessons to indices
        node_indices = {label: i for i, label in enumerate(all_labels)}
        source = user_flow['user_id'].map(node_indices)
        target = user_flow['lesson_id'].map(node_indices)
        value = user_flow['count']

        fig_sankey = go.Figure(go.Sankey(
            node=dict(
                pad=15, thickness=20, line=dict(color="white", width=0.5),
                label=all_labels, color="blue"
            ),
            link=dict(
                source=source, target=target, value=value
            )
        ))
        fig_sankey.update_layout(title_text="User Video Flow", font=dict(color="white"))
        st.plotly_chart(fig_sankey)

        st.subheader("2️⃣ Power Users (High Watch Hours & Lessons)")
        top_users = user_features.nlargest(10, 'actual_hours')
        fig_bar = px.bar(
            top_users, x=top_users.index, y=['actual_hours', 'unique_lessons'],
            title="Top Users by Actual Hours & Unique Lessons",
            labels={"value": "Count", "variable": "Metric"},
            barmode="group", color_discrete_sequence=["blue", "cyan"]
        )
        fig_bar.update_layout(font=dict(color="white"), xaxis_title="User ID")
        st.plotly_chart(fig_bar)

# **PAGE 2: Suspicious Users Detection**
elif page == "🔍 Suspicious Users Detection":
    st.title("🔍 Detect Suspicious Users")
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df, features = load_and_preprocess(uploaded_file)
        
        # Predict suspicious users
        features['is_predicted_suspicious'] = model.predict(features)

        # Extract flagged users & filter top 50
        suspicious_users = features[features['is_predicted_suspicious'] == 1].nlargest(50, 'actual_hours')
        suspicious_users = suspicious_users.reset_index()
        
        # Display Results
        st.subheader("🚨 Top 50 Suspicious Users")
        st.dataframe(suspicious_users[['user_id', 'actual_hours', 'unique_lessons']])
        
        # Save results
        suspicious_users.to_csv("predicted_suspicious_users.csv", index=False)
        st.success("✅ Predictions saved to predicted_suspicious_users.csv")
