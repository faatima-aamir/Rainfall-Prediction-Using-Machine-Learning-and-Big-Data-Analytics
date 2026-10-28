import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px  # For interactive time series

# ---------------------------
# Streamlit App Title
# ---------------------------
st.title("🌦️ Rainfall Prediction Dashboard")
st.markdown("Interactive dashboard using a LightGBM model for rainfall prediction.")

# ---------------------------
# Load Data
# ---------------------------
data = pd.read_csv("weather_ml.csv")
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date').reset_index(drop=True)

# ---------------------------
# Create Lag Features
# ---------------------------
data['rain_lag1'] = data['rainfall'].shift(1)
data['rain_lag7'] = data['rainfall'].shift(7)
data['rain_lag30'] = data['rainfall'].shift(30)
data = data.dropna().reset_index(drop=True)

# ---------------------------
# Define Feature Columns
# ---------------------------
feature_cols = ['T2M', 'RH2M', 'WS2M', 'solar',
                'rain_lag1', 'rain_lag7', 'rain_lag30']

X = data[feature_cols]
y = data['rainfall']

# ---------------------------
# Load Pre-trained Model
# ---------------------------
model = joblib.load("lgbm_model.pkl")
y_pred = model.predict(X)

# ---------------------------
# Sidebar: Feature Input Sliders
# ---------------------------
st.sidebar.header("🔧 Input Features for Prediction")
input_dict = {}
for col in feature_cols:
    min_val = float(data[col].min())
    max_val = float(data[col].max())
    mean_val = float(data[col].mean())
    input_dict[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

if st.sidebar.button("Predict Rainfall"):
    X_input = pd.DataFrame([input_dict])
    pred = model.predict(X_input)[0]
    st.sidebar.success(f"Predicted Rainfall: {pred:.2f} mm")

# ---------------------------
# Show Raw Data
# ---------------------------
if st.checkbox("Show Raw Data"):
    st.dataframe(data.head(20))

# ---------------------------
# Feature Importance
# ---------------------------
st.subheader("📊 Feature Importances")
feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp.index, y=feat_imp.values, palette="viridis")
plt.ylabel("Importance")
plt.xlabel("Features")
plt.xticks(rotation=45)
st.pyplot(plt)

# ---------------------------
# Correlation Heatmap
# ---------------------------
st.subheader("🧩 Correlation Heatmap")
numeric_data = data.select_dtypes(include='number')
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
st.pyplot(plt)

# ---------------------------
# Rainfall Distribution
# ---------------------------
st.subheader("🌧️ Rainfall Distribution")
plt.figure(figsize=(8,4))
sns.histplot(data['rainfall'], bins=30, kde=True, color='skyblue')
plt.xlabel("Rainfall (mm)")
plt.ylabel("Frequency")
st.pyplot(plt)

# ---------------------------
# Predicted vs Actual Scatter
# ---------------------------
st.subheader("Predicted vs Actual Rainfall")
plt.figure(figsize=(8,6))
plt.scatter(y, y_pred, alpha=0.6, color='orange', label='Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'b--', label='Perfect Fit')
plt.xlabel("Actual Rainfall")
plt.ylabel("Predicted Rainfall")
plt.legend()
plt.title(f"R²: {r2_score(y, y_pred):.3f}")
st.pyplot(plt)

# ---------------------------
# Interactive Time Series
# ---------------------------
st.subheader("📈 Interactive Time Series")

# Date filter
start_date = st.date_input("Start Date", value=data['date'].min(), min_value=data['date'].min(), max_value=data['date'].max())
end_date   = st.date_input("End Date", value=data['date'].max(), min_value=data['date'].min(), max_value=data['date'].max())

filtered_data = data[(data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))].copy()
filtered_data['Predicted'] = model.predict(filtered_data[feature_cols])

# Plotly interactive time series with distinct colors
fig = px.line(filtered_data, x='date', y=['rainfall', 'Predicted'],
              labels={'value':'Rainfall (mm)', 'date':'Date'},
              title="Actual vs Predicted Rainfall",
              color_discrete_map={'rainfall':'blue', 'Predicted':'orange'})
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Model Metrics
# ---------------------------
st.subheader("📐 Model Metrics")
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
st.write(f"R² Score: {r2:.3f}")
st.write(f"RMSE: {rmse:.3f}")

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# Load CSV with lat/lon
data = pd.read_csv("weather_ml.csv")

# Make sure your CSV has these columns
if 'lat_r' in data.columns and 'lon_r' in data.columns:

    st.subheader("Rainfall Map")

    # Center map on average coordinates
    m = folium.Map(location=[data['lat_r'].mean(), data['lon_r'].mean()],
                   zoom_start=5)

    # Add rainfall markers
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['lat_r'], row['lon_r']],
            radius=5 + row['rainfall']*0.1,  # scale radius with rainfall
            color='blue',
            fill=True,
            fill_opacity=0.6,
            popup=f"Rainfall: {row['rainfall']} mm"
        ).add_to(m)

    # Show map in Streamlit
    st_folium(m, width=700, height=500)
else:
    st.warning("Latitude/Longitude columns not found in CSV")
