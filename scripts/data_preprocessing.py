import pandas as pd
import os
import streamlit as st

# Configurations
RAW_DATA_PATH = "data/raw_data.csv"
CLEANED_DATA_PATH = "data/cleaned_data.csv"
os.makedirs("data", exist_ok=True)

def preprocess_data():
    """Loads, cleans, and saves processed data."""
    try:
        df = pd.read_csv(RAW_DATA_PATH, dtype=str, low_memory=False)
        
        # Handle missing values
        df.replace(to_replace=['NULL', 'null', '', ' '], value=pd.NA, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        df.to_csv(CLEANED_DATA_PATH, index=False)
        st.success("✅ Data Preprocessing Complete! Cleaned data saved.")
        return df
    except FileNotFoundError:
        st.error("⚠️ Raw data file not found. Please run data_ingestion.py first.")
        return None

# Streamlit UI
st.set_page_config(page_title="🛠 Data Preprocessing", layout="wide")
st.title("🛠 Data Preprocessing")
st.write("Preprocess the uploaded dataset by handling missing values.")
if st.button("Run Preprocessing"):
    df = preprocess_data()
    if df is not None:
        st.write("### Preview of Cleaned Data:")
        st.dataframe(df.head())
