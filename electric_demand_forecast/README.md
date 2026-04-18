# Electricity Demand Forecasting Project (5-min Interval)

This project provides a complete end-to-end solution for short-term electricity demand forecasting using high-frequency time-series data.

## 🏗 Project Structure

- `electric_demand_forecast/`
    - `main.py`: The entry point for training and evaluation.
    - `app.py`: Streamlit dashboard for visualizing results.
    - `src/`
        - `preprocess.py`: Data cleaning, outlier detection, and feature engineering.
        - `models.py`: Definitions for ML (XGBoost, RF) and DL (LSTM, Seq2Seq) models.
        - `evaluate.py`: Performance metrics and premium visualizations.
    - `outputs/`: Directory for saved plots and results.

## 🛠 Features

1. **Preprocessing**:
   - Advanced outlier detection using **Z-score** and **Isolation Forest**.
   - Time-based feature generation (Hour, Day, Month, Weekend).
   - Multi-step lag features (24-48 steps).
   - Standard Scaling for both features and targets.

2. **Forecasting Models**:
   - **Baseline**: Naive persistence model.
   - **Machine Learning**: Linear Regression, Random Forest, and XGBoost.
   - **Deep Learning**: LSTM, GRU, and Seq2Seq Encoder-Decoder for multi-step forecasting.

3. **Multi-Step Strategy**:
   - Implements a Seq2Seq architecture to predict future demand (multi-step).
   - Configurable window size (default 12 steps/1 hour) and forecast horizon.

4. **Visualizations**:
   - Time-series demand trends.
   - Feature correlation heatmaps.
   - Anomaly detection scatter plots.
   - Actual vs. Predicted comparison plots.
   - Deep Learning loss curves.
   - Bar charts comparing MAE, RMSE, and MAPE across models.

## 🚀 How to Run

### 1. Train Models
Run the main script to process data and train all models:
```bash
python main.py
```

### 2. Launch Dashboard
Visualize the results and insights:
```bash
streamlit run app.py
```

## 📉 Insights & Observations

- **Temporal Patterns**: Demand exhibits high seasonality. Morning and evening peaks are prominent.
- **Weather Impact**: Temperature (`temp`) and Humidity (`rhum`) show significant correlation with demand (HVAC usage).
- **Model Efficiency**: XGBoost and Random Forest typically outperform Linear Regression due to their ability to capture non-linear relationships and interactions between weather and time.
- **DL Capabilities**: LSTM models better capture long-range dependencies than simple lag-based ML models.

---
*Created by Antigravity*
