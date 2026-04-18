# Electricity Demand Forecasting Project

## 1. Data Preprocessing

* **Data Loading**: We load high-resolution 5-minute interval electricity demand data along with weather variables (temperature, humidity, wind speed).
* **Handling Missing Values**: Temporal gaps are addressed using time-weighted interpolation, ensuring continuity in the sequence.
* **Time Sorting**: Data is strictly sorted by datetime to prevent look-ahead bias and ensure proper time-series sequences.
* **Scaling**: We apply global standard scaling (mean=0, std=1) to both features and the target variable to ensure optimal convergence for deep learning models.

---

## 2. Feature Engineering

* **Lag Features**: We incorporate historical demand lags (1, 6, 12, 24, 48 steps) to capture immediate and daily dependencies.
* **Rolling Statistics**: 6-step and 24-step moving averages are calculated to smooth noise and highlight local trends.
* **Time-based Features**: Extraction of hour, day of week, and seasonality (month). Cyclical encoding (sine/cosine transforms) is used for time features to preserve their periodic nature.
* **Weather Integration**: Atmospheric conditions are integrated as exogenous variables to account for their significant impact on heating and cooling demand.

---

## 3. Anomaly Detection & Handling

* **Method Used**: Z-score analysis (threshold > 3) was implemented to detect spikes and non-physical demand drops.
* **Treatment**: Detected anomalies were replaced using linear interpolation from neighboring data points, preserving the underlying seasonal signal while removing outliers.

---

## 4. Model Training (Multiple Models)

* **Naive Baseline**: A persistency model where future demand is predicted as the current value. Used as a reference for improvement.
* **Linear Regression**: A classic statistical approach to model linear relationships between features and demand.
* **XGBoost**: A powerful gradient-boosted tree ensemble capable of capturing complex non-linear interactions and feature importance.
* **LSTM**: A Long Short-Term Memory recurrent neural network designed to identify long-range temporal dependencies in sequence data.

---

## 5. Multi-step Forecasting

* **Sliding Window Approach**: A window of 4 hours (48 steps) of historical data is used to predict a horizon of 2 hours (24 steps) simultaneously.
* **Importance**: 24-step prediction (2 hours ahead) is critical for grid operators to manage supply-demand balance and prevent blackouts.
* **LSTM Performance**: The LSTM architecture uses its hidden states to maintain a representation of recent history, making it robust for sequential multi-output tasks.

---

## 6. Evaluation & Comparison

* **Metrics**: We use MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and MAPE (Mean Absolute Percentage Error).
* **Grid Stability**: RMSE is prioritized as it penalizes large errors more heavily, which is crucial for maintaining power grid frequency and stability.

| Model | MAE | RMSE | MAPE (%) |
| :--- | :--- | :--- | :--- |
| Naive | 232.22 | 379.18 | 5.53% |
| Linear Regression | 180.64 | 353.27 | 4.36% |
| XGBoost | 154.77 | 321.57 | 3.38% |
| **LSTM** | **163.50** | **306.53** | **3.93%** |

---

## 7. Key Insights

* **Peak Demand Hours**: Peak usage consistently occurs between 18:00 and 21:00, driven by residential activity.
* **Weather Influence**: Temperature and humidity show a high correlation with demand spikes during summer/winter months.
* **Model Comparison**: While XGBoost is competitive, LSTM shows superior performance in RMSE, effectively minimizing large prediction errors.

---

## 8. Conclusion

* **LSTM Advantage**: The LSTM model's ability to process sequential data and handle non-linear time dependencies makes it the most robust choice for this dataset.
* **Real-world Applicability**: This pipeline can be deployed in utility control rooms to assist in real-time load shedding decisions and renewable energy integration.
