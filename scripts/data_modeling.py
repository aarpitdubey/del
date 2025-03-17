import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st

# Configurations
CLEANED_DATA_PATH = "data/cleaned_data.csv"
MODEL_RESULTS_PATH = "data/model_results.csv"
MODEL_PATH = "models/trained_model.pkl"
os.makedirs("models", exist_ok=True)

def train_models():
    """Loads cleaned data, trains models, and saves results."""
    try:
        df = pd.read_csv(CLEANED_DATA_PATH, dtype=str, low_memory=False)
        df["latitude"] = pd.to_numeric(df["latitude"], errors='coerce')
        df["longitude"] = pd.to_numeric(df["longitude"], errors='coerce')
        df.dropna(subset=["latitude", "longitude"], inplace=True)
        df["geometry"] = df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        # Anomaly Detection with DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        gdf["anomaly"] = dbscan.fit_predict(gdf[["latitude", "longitude"]])
        
        # Clustering with K-Means
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        gdf["cluster"] = kmeans.fit_predict(gdf[["latitude", "longitude"]])

        # Industry Scale Prediction
        target_col = "scale"
        feature_cols = ["latitude", "longitude", "state"]
        df.dropna(subset=[target_col], inplace=True)
        
        label_encoders = {}
        for col in feature_cols:
            if df[col].dtype == "object":
                label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col])
        
        X, y = df[feature_cols], df[target_col]
        if len(X) > 5:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)
            
            df.to_csv(MODEL_RESULTS_PATH, index=False)
            st.success("✅ Model Training Complete! Results saved.")
        else:
            st.warning("⚠️ Not enough data for AI model training.")
    except FileNotFoundError:
        st.error("⚠️ Cleaned data file not found. Please run data_preprocessing.py first.")

# Streamlit UI
st.set_page_config(page_title="🤖 AI Modeling", layout="wide")
st.title("🤖 AI Modeling & Predictions")
st.write("Train machine learning models for anomaly detection, clustering, and predictions.")
if st.button("Train Models & Save Results"):
    train_models()