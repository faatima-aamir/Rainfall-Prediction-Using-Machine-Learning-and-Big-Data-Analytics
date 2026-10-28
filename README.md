# Rainfall-Prediction-Using-Machine-Learning-and-Big-Data-Analytics
# 🌧️ Rainfall Prediction Using Machine Learning and GIS

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20LightGBM%20%7C%20LSTM-green)
![GIS](https://img.shields.io/badge/GIS-Google%20Earth%20Engine-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

Rainfall prediction is a critical task for disaster management, agriculture, water resource planning, and climate monitoring. Traditional forecasting approaches often struggle to capture the complex interactions between geographical and meteorological factors.

This project integrates **Machine Learning**, **Deep Learning**, and **Geographic Information Systems (GIS)** to predict rainfall occurrence and intensity using environmental and meteorological datasets. A two-stage modeling framework was developed to improve performance on zero-inflated rainfall data while leveraging Members

| Name | Roll Number |
|--------|------------|
| Faatima Aamir | 404214 |
| Hijab Zahra | 421132 |
| Noor-ul-Haya | 425722 |

**Course:** CS471 – Machine Learning  
**Institution:** National University of Sciences and Technology (NUST)  
**Department:** Institute of Geographical Information Systems (IGIS)

---

## 📑 Table of Contents

- #-project-objectives
- #-dataset-description
- #-methodology
- #-feature-engineering
- #-models-implemented
- #-training-strategy
- #-evaluation-metrics
- #-results
- #-technologies-used
- #-future-work
- #-conclusion

---

## 🎯 Project Objectives

### 1. Data Integration
- Collect and preprocess meteorological and geospatial datasets.
- Align datasets spatially and temporally.
- Generate a machine-learning-ready dataset.

### 2. Rainfall Prediction
Develop predictive models using:

- XGBoost
- LightGBM
- LSTM

### 3. Comparative Analysis
- Evaluate model performance.
- Identify the most suitable model for rainfall forecasting.

### 4. GIS-Based Visualization
- Visualize rainfall distribution and prediction outputs.
- Analyze spatial rainfall patterns.

---

## 📂 Dataset Description

### 🌧️ Rainfall Data
**CHIRPS (Climate Hazards Group InfraRed Precipitation with Station Data)**

- Daily precipitation data
- Satellite-based observations
- Available through Google Earth Engine

---

### 🌡️ Meteorological Data
**NASA POWER**

Variables collected:

- Air Temperature (T2M)
- Relative Humidity (RH2M)
- Wind Speed (WS2M)
- Solar Radiation

---

### 🌿 Vegetation Data
**NDVI (Normalized Difference Vegetation Index)**

Sources:

- MODIS
- Landsat

Used to capture vegetation dynamics and land surface conditions.

---

### 🗺️ Land Cover Data

**ESA WorldCover**

Provides high-resolution global land cover classifications.

---

### ⛰️ Elevation Data

**Digital Elevation Model (DEM)**

Sources:

- NASA SRTM
- USGS

Used to model topographic influences on rainfall.

---

## 🏗️ Methodology

The project follows a complete machine learning workflow:

```text
Data Collection
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Train/Test Split
       ↓
Model Training
       ↓
Prediction
       ↓
Performance Evaluation
```

---

## 🔧 Feature Engineering

### Meteorological Features

- Temperature
- Relative Humidity
- Wind Speed
- Solar Radiation

### Geospatial Features

- NDVI
- DEM
- Land Cover

### Temporal Features

Lagged rainfall values:

- Rainfall Lag-1
- Rainfall Lag-3
- Rainfall Lag-7

These features help capture temporal rainfall persistence and improve forecasting accuracy.

---

## ⚙️ Two-Stage Modeling Framework

Rainfall datasets contain a large number of zero-rainfall observations.

### Stage 1: Rainfall Classification

Predict whether rainfall occurs:

```python
1 = Rainfall > 0
0 = Rainfall = 0
```

### Stage 2: Rainfall Regression

For rainy days:

```python
Predict rainfall intensity (mm)
```

### Benefits

✅ Handles zero-inflated rainfall data

✅ Improves prediction stability

✅ Reduces regression bias

---

## 🤖 Models Implemented

### XGBoost

Extreme Gradient Boosting algorithm that builds decision-tree ensembles sequentially.

**Advantages**
- High predictive accuracy
- Handles nonlinear relationships
- Robust against overfitting

---

### LightGBM

Gradient boosting framework optimized for speed and memory efficiency.

**Advantages**
- Faster training
- Lower memory usage
- Competitive prediction accuracy

---

### LSTM

Long Short-Term Memory neural network developed for time-series forecasting.

**Advantages**
- Learns temporal dependencies
- Effective for sequential rainfall data
- Captures long-term trends

---

## 📊 Training Strategy

### Temporal Train-Test Split

To avoid data leakage:

- Earlier periods → Training Set
- Later periods → Testing Set

This approach simulates real-world forecasting scenarios.

---

## 📈 Evaluation Metrics

### Root Mean Squared Error (RMSE)

```math
RMSE = \sqrt{\frac{\sum (y - \hat{y})^2}{n}}
```

Measures average prediction error magnitude.

---

### Mean Absolute Error (MAE)

```math
MAE = \frac{\sum |y-\hat{y}|}{n}
```

Measures average prediction deviation.

---

### Coefficient of Determination (R²)

Measures the proportion of variance explained by a model.

---

## 📋 Results

### Model Comparison

| Model | Accuracy | Training Speed | Computational Cost |
|---------|----------|----------------|-------------------|
| XGBoost | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Medium |
| LightGBM | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Low |
| LSTM | ⭐⭐⭐⭐ | ⭐⭐ | High |

---

### Key Findings

✅ XGBoost achieved the highest prediction accuracy.

✅ LightGBM provided the best balance between speed and accuracy.

✅ LSTM effectively captured temporal rainfall patterns.

✅ Lag features significantly improved model performance.

✅ Two-stage modeling reduced bias caused by non-rainfall days.

---

## 🗺️ Visual Outputs

The following visualizations were generated:

- Actual vs Predicted Rainfall
- Time Series Forecasts
- Loss Curves
- Model Comparison Charts
- Rainfall Prediction Maps

---

## 🛠️ Technologies Used

### Programming Languages

- Python

### Machine Learning

- XGBoost
- LightGBM
- TensorFlow
- Keras

### Data Processing

- Pandas
- NumPy

### GIS & Remote Sensing

- Google Earth Engine
- CHIRPS
- ESA WorldCover
- NASA POWER
- SRTM DEM

### Cloud Computing

- AWS EC2

---

## 🚀 Future Work

- Incorporate additional atmospheric variables.
- Expand temporal coverage beyond 2025.
- Develop real-time rainfall forecasting systems.
- Explore Transformer-based architectures.
- Integrate deployment through a web dashboard.

---

## 📚 References

1. CHIRPS Rainfall Dataset
2. NASA POWER API
3. ESA WorldCover Dataset
4. Google Earth Engine
5. TensorFlow Documentation
6. XGBoost Documentation
7. LightGBM Documentation

---

## ✅ Conclusion

This project demonstrates the effectiveness of combining Machine Learning and GIS techniques for rainfall prediction. Among the tested approaches, **XGBoost achieved the highest predictive performance**, while **LightGBM provided the best computational efficiency**. The proposed two-stage modeling framework successfully handled zero-inflated rainfall datasets and improved forecasting robustness.

The project highlights the potential of data-driven geospatial analytics for supporting climate monitoring, disaster preparedness, and environmental decision-making.
