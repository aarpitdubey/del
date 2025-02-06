import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  
import geopandas as gpd  
import folium  
from shapely import wkb  
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN, KMeans 
from sklearn.ensemble import RandomForestClassifier  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import train_test_split  
import streamlit as st  


# Streamlit App
st.set_page_config(page_title="Industry Data Analysis", layout="wide")
st.title("Industry Data Visualization and Analysis")

# Step 1: Load the Dataset
@st.cache
def load_data():
    return pd.read_csv("other_industries_original_unaltered.csv", dtype=str)  # Load all columns as strings initially

df = load_data()

# Step 2: Data Inspection
st.subheader("Data Overview")
st.write(df.head())

# Step 3: Data Cleaning
## Handling Missing Values
df.replace(to_replace=['NULL', 'null', '', ' '], value=np.nan, inplace=True)
missing_data = df.isnull().sum() / len(df) * 100
st.subheader("Missing Values Percentage")
st.write(missing_data.sort_values(ascending=False))

## Trim whitespace from text columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

## Convert numeric columns
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['licenses_count'] = pd.to_numeric(df['licenses_count'], errors='coerce')
df['lab_count'] = pd.to_numeric(df['lab_count'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# Step 4: Data Visualization
st.subheader("Missing Values Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
st.pyplot(fig)

# Step 5: Geospatial Visualization
st.subheader("Industry Locations on Map")
def parse_geom(geom_str):
    try:
        return wkb.loads(geom_str, hex=True)
    except Exception:
        return None

df['geometry'] = df['geom'].apply(lambda x: parse_geom(x) if pd.notnull(x) else None)
gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.set_crs(epsg=4326, inplace=True)

m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
for _, row in gdf.iterrows():
    folium.Marker([row.latitude, row.longitude], popup=row['branch']).add_to(m)
st.components.v1.html(m._repr_html_(), height=600)

# Step 6: Industry Trends Analysis
st.subheader("Industry Count by State")
st.bar_chart(df['branch'].value_counts())

# Step 7: Anomaly Detection (DBSCAN for Latitude & Longitude)
st.subheader("Anomaly Detection (DBSCAN)")
coords = df[['latitude', 'longitude']].dropna().values
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(coords)
df['anomaly'] = dbscan.labels_
st.write("Anomalies Found:")
st.write(df[df['anomaly'] == -1])

# Step 8: Clustering (K-Means)
st.subheader("Industry Clusters (K-Means)")
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])
fig, ax = plt.subplots()
plt.scatter(df['longitude'], df['latitude'], c=df['cluster'], cmap='viridis')
st.pyplot(fig)

# Step 9: Predict Missing Data (Random Forest)
st.subheader("Predict Missing Scale Values")
df.dropna(subset=['scale'], inplace=True)
X = df[['latitude', 'longitude', 'branch']]
y = df['scale']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
st.write("Model Accuracy: ", model.score(X_test, y_test))

# Step 10: Industrial Hotspot Prediction (Decision Tree)
st.subheader("Industrial Hotspot Prediction")
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
prediction = model.predict([[12.9716, 77.5946, 'Karnataka']])
st.write("Predicted Industry Scale for Bangalore: ", prediction)

# Additional Feature: Data Completeness Dashboard
st.subheader("Data Completeness Dashboard")
st.bar_chart(missing_data)
st.write("Interactive Data Filtering")
selected_state = st.selectbox("Select State", df['branch'].unique())
st.write(df[df['branch'] == selected_state])
