import streamlit as st
import pandas as pd
import joblib
import base64

def predict_suspicious_users(df, model_path='suspicious_model.pkl'):
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    if df.empty:
        return pd.DataFrame(columns=['user_id', 'reason', 'event_date'])

    features = df.groupby('user_id').agg({
        'actual_hours': 'sum',
        '_pause': 'sum',
        '_seek': 'sum',
        'lesson_id': 'nunique',
        '_d_id': 'nunique'
    }).rename(columns={'lesson_id': 'unique_lessons', '_d_id': 'unique_devices'})

    model = joblib.load(model_path)
    features['is_predicted_suspicious'] = model.predict(features)
    
    suspicious_users = features[features['is_predicted_suspicious'] == 1].reset_index()
    
    feature_importances = model.feature_importances_
    feature_names = features.columns[:-1]
    reason_map = {name: feature_importances[i] for i, name in enumerate(feature_names)}
    top_reason = max(reason_map, key=reason_map.get)
    suspicious_users['reason'] = f"High {top_reason} usage"
    suspicious_users['event_date'] = df['event_date'].max().date()
    
    return suspicious_users[['user_id', 'reason', 'event_date']]

# Streamlit UI
def main():
    st.set_page_config(page_title="Suspicious User Detector", layout="wide")
    
    # Background Style
    page_bg = """
    <style>
    body {
        background: linear-gradient(135deg, #0033cc, #66ccff);
        color: white;
    }
    .stApp {
        background: linear-gradient(135deg, #0033cc, #66ccff);
    }
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)
    
    st.title("📊 Suspicious User Detector")
    st.write("Upload a CSV file to identify suspicious users based on video interactions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        suspicious_users = predict_suspicious_users(df)
        
        if not suspicious_users.empty:
            st.success("Suspicious users identified!")
            st.dataframe(suspicious_users)
            
            # Download link
            csv = suspicious_users.to_csv(index=False).encode('utf-8')
            b64 = base64.b64encode(csv).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="suspicious_users.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("No suspicious users detected in this dataset.")

if __name__ == "__main__":
    main()
