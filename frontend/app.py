import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

API_BASE = "http://localhost:8000/api"

st.title("Auto ML Suite")

# File upload
st.header("1. Data Ingestion")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
if uploaded_file:
    files = {'file': uploaded_file}
    response = requests.post(f"{API_BASE}/data/upload", files=files)

    if response.status_code == 200:
        data = response.json()
        st.success(data['message'])
        st.write(f"Shape: {data['shape']}")
        st.dataframe(pd.DataFrame(data['preview']))

        # 🔥 RESET POINTER HERE
        uploaded_file.seek(0)

        # Load locally for Streamlit session
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state['df'] = df
    else:
        st.error("Upload failed")

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
            data = eval(pred_data)  # dangerous, but for demo
            payload = {"model_filename": st.session_state['model_filename'], "data": data}
            response = requests.post(f"{API_BASE}/model/predict", json=payload)
            if response.status_code == 200:
                preds = response.json()['predictions']
                st.write("Predictions:", preds)
            else:
                st.error("Prediction failed")
        except:
            st.error("Invalid JSON")
