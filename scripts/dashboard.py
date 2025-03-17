import pandas as pd
import os
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns

# Configurations
MODEL_RESULTS_PATH = "data/model_results.csv"
MODEL_PATH = "models/trained_model.pkl"

# Load Data
st.set_page_config(page_title="📊 Industry Dashboard", layout="wide")
st.title("📊 Industry Data Dashboard")

def load_data():
    try:
        df = pd.read_csv(MODEL_RESULTS_PATH, dtype=str, low_memory=False)
        df["latitude"] = pd.to_numeric(df["latitude"], errors='coerce')
        df["longitude"] = pd.to_numeric(df["longitude"], errors='coerce')
        st.success("✅ Model Results Loaded Successfully!")
        return df
    except FileNotFoundError:
        st.error("⚠️ Model results file not found. Please run data_modeling.py first.")
        return None

df = load_data()
if df is not None:
    # Industry Distribution Visualization
    st.subheader("📈 Industry Distribution by State")
    state_counts = df["state"].value_counts()
    st.bar_chart(state_counts)
    
    # Clustering Visualization
    st.subheader("🔵 Industry Clustering")
    if "cluster" in df.columns:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df["longitude"], y=df["latitude"], hue=df["cluster"], palette="viridis", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ Clustering data not found.")
    
    # Geospatial Map
    st.subheader("🗺️ Industry Locations on Map")
    if "latitude" in df.columns and "longitude" in df.columns:
        map_center = [df["latitude"].mean(), df["longitude"].mean()]
        m = folium.Map(location=map_center, zoom_start=5, tiles="CartoDB positron")
        for _, row in df.iterrows():
            folium.Marker([row["latitude"], row["longitude"]], popup=row.get("state", "Unknown")).add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.warning("⚠️ Latitude and Longitude data missing.")