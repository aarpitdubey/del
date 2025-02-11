import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st

st.set_page_config(page_title="Industry Data Dashboard", layout="wide")
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Data Cleaning", "Geospatial Analysis", "Trends & Growth", "AI Predictions"])

latitude_col, longitude_col = "latitude", "longitude"

st.sidebar.subheader("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, dtype=str)
    st.sidebar.success("✅ Data Loaded!")
else:
    st.sidebar.warning("⚠️ Upload a dataset to continue.")
    st.stop()

# Home page
if page == "Home":
    st.title("📊 Industry Data Dashboard")
    st.subheader("Welcome to the Industry Data Insights & AI Predictions App!")
    st.write(
        "This application allows you to explore industry trends, visualize geospatial data, "
        "detect anomalies, and predict industrial hotspots using machine learning models."
    )

    st.markdown("---")

    st.markdown("## 👨‍💻 About the Author")
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("https://avatars.githubusercontent.com/u/31886268?v=4", width=150)  # Replace with actual author image URL

    with col2:
        st.markdown("### **Arpit Dubey**")
        st.write(
            "📌 **AI & Data Science Expert** | **Generative AI Developer**\n"
            "🔍 Specializing in AI/ML, NLP, and Data Analytics.\n"
            "💡 Passionate about building intelligent solutions for real-world problems."
        )
        st.markdown("📧 [Email](mailto:aarpitdubey@gmail.com)")
        st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/aarpitdubey)")
        st.markdown("🏆 [GitHub](https://github.com/aarpitdubey)")

    st.markdown("---")

    st.subheader("🔍 How to Use This App")
    st.write(
        "- **📂 Upload** a CSV file with industry data.\n"
        "- **🛠 Clean** the data by handling missing values.\n"
        "- **🗺️ Visualize** industry locations on an interactive map.\n"
        "- **📈 Analyze** trends, clustering, and anomalies.\n"
        "- **🤖 Predict** industry hotspots using AI/ML models."
    )

    st.success("✅ Get started by selecting an option from the sidebar!")

# Data Cleaning Page
elif page == "Data Cleaning":
    st.title("🛠 Data Cleaning & Missing Values")

    df.replace(to_replace=['NULL', 'null', '', ' '], value=np.nan, inplace=True)
    df = df.infer_objects(copy=False)

    missing_percent = df.isnull().mean() * 100
    if df.isnull().sum().sum() > 0:
        st.subheader("📊 Missing Values Heatmap")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='Reds', ax=ax)
        ax.set_title("Missing Values Heatmap")
        st.pyplot(fig)
    else:
        st.success("✅ No Missing Values Found!")

    missing_handling = st.radio("Handle Missing Values:", ["Fill with Mean/Mode", "Drop Rows", "Fill with Zero"])
    if missing_handling == "Fill with Mean/Mode":
        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
    elif missing_handling == "Fill with Zero":
        df.fillna(0, inplace=True)
    elif missing_handling == "Drop Rows":
        df.dropna(inplace=True)

    st.success("✅ Missing Values Handled!")

# Geospatial Analysis Page
elif page == "Geospatial Analysis":
    st.title("🗺️ Industry Geospatial Analysis")

    show_map = st.checkbox("Show Industry Map", value=True)
    if show_map:
        df[latitude_col] = pd.to_numeric(df[latitude_col], errors='coerce')
        df[longitude_col] = pd.to_numeric(df[longitude_col], errors='coerce')
        df_filtered = df.dropna(subset=[latitude_col, longitude_col])

        if not df_filtered.empty:
            max_points = st.slider("Select number of points for map", min_value=0, max_value=50, value=10)
            df_sample = df_filtered.sample(min(max_points, len(df_filtered)))

            m = folium.Map(location=[df_sample[latitude_col].mean(), df_sample[longitude_col].mean()], zoom_start=5, tiles="CartoDB positron")
            for _, row in df_sample.iterrows():
                folium.CircleMarker(location=[row[latitude_col], row[longitude_col]], radius=2, color="blue", fill=True, fill_color="blue", fill_opacity=0.6).add_to(m)
            st_folium(m, width=500, height=350)
        else:
            st.warning("⚠️ No valid geographic data available.")

# Trends & Growth Analysis Page
elif page == "Trends & Growth":
    st.title("📈 Industry Trends & Growth Analysis")

    state_counts = df["state"].value_counts()
    st.subheader("🏆 Top 5 States with Highest & Lowest Industries")
    col1, col2 = st.columns(2)
    col1.write("🔼 Highest:")
    col1.write(state_counts.head(5))
    col2.write("🔽 Lowest:")
    col2.write(state_counts.tail(5))

    st.subheader("📊 Industry Count by State")
    st.bar_chart(state_counts)

# AI Predictions Page
elif page == "AI Predictions":
    st.title("🤖 AI-Based Predictions")

    target_col = "scale"
    feature_cols = [latitude_col, longitude_col, "state"]
    
    if all(col in df.columns for col in feature_cols + [target_col]):
        df_filtered = df.dropna(subset=[target_col])
        
        if len(df_filtered) >= 5:
            X, y = df_filtered[feature_cols], df_filtered[target_col]
            X = X.copy()
            label_encoders = {}
            for col in feature_cols:
                if X[col].dtype == "object":
                    label_encoders[col] = LabelEncoder()
                    X[col] = label_encoders[col].fit_transform(X[col])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            st.write(f"✅ Model Accuracy: {model.score(X_test, y_test):.2f}")
            
            # Convert latitude & longitude to numeric before calling .median()
            df[latitude_col] = pd.to_numeric(df[latitude_col], errors='coerce')
            df[longitude_col] = pd.to_numeric(df[longitude_col], errors='coerce')

            lat_input = st.number_input("Enter Latitude", value=df[latitude_col].dropna().median())
            lon_input = st.number_input("Enter Longitude", value=df[longitude_col].dropna().median())
            state_input = st.selectbox("Select State", df["state"].unique())
            
            state_encoded = label_encoders["state"].transform([state_input])[0] if state_input in label_encoders["state"].classes_ else 0
            prediction = model.predict([[lat_input, lon_input, state_encoded]])[0]
            st.write(f"🏭 **Predicted Industry Scale:** {prediction}")
        else:
            st.warning("⚠️ Not enough data for AI model training.")