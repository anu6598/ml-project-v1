import streamlit as st
import pandas as pd
import joblib

# Load trained model
MODEL_PATH = "suspicious_model.pkl"
model = joblib.load(MODEL_PATH)

# Streamlit page configuration
st.set_page_config(page_title="User Analysis Dashboard", layout="wide")

# Custom CSS for styling
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
    .stTextInput, .stFileUploader, .stSelectbox {
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
page = st.sidebar.radio("Go to", ["📊 Video Usage Trends", "🔍 Suspicious Users Detection", "🔎 Check a User"])

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
        '_d_id': 'nunique'  # Ensure 'unique_devices' is derived
    }).rename(columns={'lesson_id': 'unique_lessons', '_d_id': 'unique_devices'})

    # Ensure all expected columns exist
    expected_features = ['actual_hours', '_pause', '_seek', 'unique_lessons', 'unique_devices']
    for col in expected_features:
        if col not in features.columns:
            features[col] = 0  # Fill missing columns with 0

    return df, features


# **PAGE 1: Video Usage Trends**
if page == "📊 Video Usage Trends":
    st.title("📊 Video Usage Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df, user_features = load_and_preprocess(uploaded_file)
        
        # Filter for specific subjects
        selected_subjects = ["ENT", "Surgery", "Community Medicine", "Anatomy"]
        filtered_df = df[df["_subject_title"].isin(selected_subjects)]
        
        # Count occurrences
        subject_counts = filtered_df["_subject_title"].value_counts()
        
        # Plot bar chart
        fig, ax = plt.subplots()
        subject_counts.plot(kind="bar", ax=ax, color=["blue", "red", "green", "purple"])
        ax.set_title("Video Count per Subject")
        ax.set_xlabel("Subjects")
        ax.set_ylabel("Number of Videos")
        
        # Show plot in Streamlit
        st.pyplot(fig)


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

# **PAGE 3: Check a User**
elif page == "🔎 Check a User":
    st.title("🔎 Check if a User is Suspicious")
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df, features = load_and_preprocess(uploaded_file)
        
        # User ID input
        user_id = st.text_input("Enter a User ID to Check", "").strip().lower()
        
        if user_id:
            if user_id in features.index:
                user_data = features.loc[user_id]
                prediction = model.predict([user_data])[0]
                
                # Display user data
                st.subheader("📊 User Data")
                st.dataframe(pd.DataFrame(user_data).T)

                # Display Result
                if prediction == 1:
                    st.error(f"🚨 User {user_id} is **SUSPICIOUS** 🚨")
                else:
                    st.success(f"✅ User {user_id} is **NOT Suspicious** ✅")

                # Reasoning
                st.subheader("🧐 Why is this user suspicious?")
                if prediction == 1:
                    reasons = []
                    if user_data['actual_hours'] > 10:
                        reasons.append("Unusually high watch hours.")
                    if user_data['_pause'] == 0 and user_data['_seek'] == 0:
                        reasons.append("No pauses or seeks detected.")
                    if user_data['unique_lessons'] < 2:
                        reasons.append("Very few unique lessons watched.")
                    
                    for reason in reasons:
                        st.write(f"🔸 {reason}")

            else:
                st.warning("⚠️ User ID not found in the uploaded data.")
