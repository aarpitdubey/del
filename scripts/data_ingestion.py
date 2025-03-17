import pandas as pd
import os
import streamlit as st

# Configurations
RAW_DATA_PATH = "data/raw_data.csv"
os.makedirs("data", exist_ok=True)

def load_data(uploaded_file):
    """Loads and saves raw data while handling dtype warnings."""
    try:
        df = pd.read_csv(uploaded_file, dtype=str, low_memory=False)  # Load all columns as strings
        df.to_csv(RAW_DATA_PATH, index=False)
        st.success("✅ Raw data saved successfully!")
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="📂 Data Ingestion", layout="wide")
st.title("📂 Data Ingestion")
st.write("Upload a CSV file to begin the AI pipeline.")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("### Preview of Raw Data:")
        st.dataframe(df.head())