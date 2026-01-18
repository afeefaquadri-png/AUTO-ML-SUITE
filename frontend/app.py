import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import json

API_BASE = "http://localhost:8000/api"

st.title("Auto ML Suite")

# Initialize session state
if 'upload_error' not in st.session_state:
    st.session_state.upload_error = None
if 'upload_success' not in st.session_state:
    st.session_state.upload_success = None

# File upload
st.header("1. Data Ingestion")

# Create a container for upload status
upload_status = st.container()

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    key="file_uploader"
)

if uploaded_file:
    # Reset previous status messages
    st.session_state.upload_error = None
    st.session_state.upload_success = None
    
    # Validate file is not empty before processing
    if uploaded_file.size == 0:
        st.session_state.upload_error = "The uploaded file is empty. Please select a valid file."
    elif not any(uploaded_file.name.endswith(ext) for ext in ['.csv', '.xlsx', '.xls']):
        st.session_state.upload_error = "Invalid file format. Please upload a CSV or Excel file."
    else:
        with st.spinner("Uploading and processing file..."):
            try:
                files = {'file': uploaded_file}
                response = requests.post(f"{API_BASE}/data/upload", files=files, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    st.session_state.upload_success = data['message']
                    st.session_state['columns'] = data['columns']
                    st.session_state['data'] = data['preview']
                    st.session_state['df'] = pd.DataFrame(data['data'])
                else:
                    error_msg = response.json().get('detail', 'Upload failed')
                    st.session_state.upload_error = error_msg
            except requests.exceptions.Timeout:
                st.session_state.upload_error = "File upload timed out. Please try with a smaller file."
            except requests.exceptions.ConnectionError:
                st.session_state.upload_error = "Cannot connect to the backend server. Please ensure the API is running at http://localhost:8000"
            except Exception as e:
                st.session_state.upload_error = f"Error during file upload: {str(e)}"

# Display status messages using containers to avoid persistence
with upload_status:
    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)
    elif st.session_state.upload_success:
        st.success(st.session_state.upload_success)
        if 'df' in st.session_state:
            st.write(f"Shape: {st.session_state['df'].shape}")
            st.dataframe(pd.DataFrame(st.session_state.get('data', [])))


# DB load
st.subheader("Or load from Database")
source = st.selectbox("Source", ["mongo", "postgres"])
if source == "mongo":
    collection = st.text_input("Collection name")
else:
    query = st.text_area("SQL Query")
db_name = st.text_input("DB Name", "auto_ml_db")
if st.button("Load from DB"):
    payload = {"source": source, "collection_or_query": collection if source=="mongo" else query, "db_name": db_name}
    response = requests.post(f"{API_BASE}/data/load_db", json=payload)
    if response.status_code == 200:
        data = response.json()
        st.success(data['message'])
        st.dataframe(pd.DataFrame(data['preview']))
        st.session_state['df'] = pd.DataFrame(data['data'])
    else:
        st.error("DB load failed")

# Preprocessing
if 'df' in st.session_state:
    st.header("2. Data Preprocessing")
    features = st.multiselect("Select Features (X)", st.session_state['df'].columns)
    target = st.selectbox("Select Target (Y)", st.session_state['df'].columns)
    if st.button("Preprocess"):
        payload = {
            "data": st.session_state['df'].to_dict('records'),
            "features": features,
            "target": target
        }
        response = requests.post(f"{API_BASE}/data/preprocess", json=payload)
        if response.status_code == 200:
            data = response.json()
            st.session_state['X'] = pd.DataFrame(data['X'])
            st.session_state['y'] = data['y']
            st.success("Data preprocessed")
            st.dataframe(st.session_state['X'].head())
        else:
            st.error("Preprocessing failed")

# Training
if 'X' in st.session_state:
    st.header("3. Model Training")
    if st.button("Train Model"):
        payload = {
            "X": st.session_state['X'].to_dict('records'),
            "y": st.session_state['y']
        }
        response = requests.post(f"{API_BASE}/model/train", json=payload)
        if response.status_code == 200:
            data = response.json()
            st.session_state['model_filename'] = data['model_filename']
            st.success(f"Model trained: {data['metrics']}")
            # Plot metrics
            fig, ax = plt.subplots()
            ax.bar(data['metrics'].keys(), data['metrics'].values())
            st.pyplot(fig)
        else:
            st.error("Training failed")

# Prediction
if 'model_filename' in st.session_state:
    st.header("4. Model Prediction")
    pred_data = st.text_area("Enter prediction data as JSON list of dicts")
    if st.button("Predict"):
        try:
            data = json.loads(pred_data)
            payload = {"model_filename": st.session_state['model_filename'], "data": data}
            response = requests.post(f"{API_BASE}/model/predict", json=payload)
            if response.status_code == 200:
                preds = response.json()['predictions']
                st.write("Predictions:", preds)
            else:
                st.error("Prediction failed")
        except json.JSONDecodeError:
            st.error("Invalid JSON format")
        except Exception as e:
            st.error(f"Error: {str(e)}")
