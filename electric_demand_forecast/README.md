# ⚡ Advanced Electricity Demand Forecasting & Explainable AI

This project implements a state-of-the-art multi-step forecasting pipeline for electricity demand, enhanced with Explainable AI (SHAP), a premium dashboard, and deployment-ready architecture.

## 🚀 Recent Enhancements

- **Explainable AI (XAI)**: Integrated **SHAP** to provide transparency in model predictions.
- **Premium Dashboard**: A new **Streamlit** dashboard for performance analysis and live-prediction concepts.
- **Model Serialization**: Models and scalers are saved in `models/` for immediate production deployment.
- **Enhanced Visuals**: Added SHAP plots, residual analysis, and improved multi-step forecast overlays.

---

## 🏗 Project Architecture

- `main.py`: Core training logic.
- `enhance_project.py`: Generates SHAP values, additional visuals, and saves models.
- `app.py`: Streamlit dashboard.
- `models/`: Production-ready model artifacts (`.pkl`, `.keras`, `.pkl` scalers).
- `outputs/`: Visualization and performance report storage.

---

## 🔍 Explainable AI (SHAP)

Understanding *why* a model predicts a certain demand is crucial for grid reliability.
- **SHAP Summary Plot**: Shows the impact of each feature on the final forecast.
- **Feature Interaction**: Demonstrates how weather and time-based features (lags) interact to drive demand.
- **Transparency**: High scores in lag features indicate the model captures temporal persistence, while weather features explain seasonal spikes.

---

## 🌐 Deployment Readiness

### Real-World Grid Integration
1. **IoT Stream Integration**: Data from smart meters can be streamed via MQTT or Kafka directly into the preprocessing pipeline.
2. **Weather API Sync**: The system is designed to integrate with APIs like OpenWeatherMap to fetch real-time exogenous variables.
3. **Inference Strategy**:
   - **XGBoost**: Highly efficient for edge deployment on low-power devices.
   - **LSTM**: Best suited for central server deployment to capture long-term sequential dependencies.

### Scalability
- **Horizontal Scaling**: The inference engine can be containerized using **Docker** and orchestrated via **Kubernetes** to handle multiple regions simultaneously.
- **Optimized Weights**: Model size has been minimized for fast loading and low memory footprint.

### Ethical Considerations & Reliability
- **Forecast Errors**: Grid operators should use these forecasts alongside a "Confidence Interval" to manage spinning reserves.
- **Data Privacy**: Ensure that smart meter data is aggregated or anonymized before processing to comply with GDPR/Data protection laws.

---

## ⚙️ Model Optimizations (Non-Destructive)

We applied safe techniques to improve performance without degrading accuracy:
1. **Early Stopping**: LSTM training is monitored for validation loss stagnation, preventing overfitting and reducing compute time.
2. **Standardized Scaling**: Consistent use of `StandardScaler` ensures numerical stability across different model types.
3. **Multi-Step Vectorization**: Predicting the entire 24-step horizon in one pass (Vectorized Output) instead of recursive steps to avoid error propagation.
4. **Subsampling Strategy**: Used a representative sample of historical data for complex SHAP calculations to optimize throughput.

---

## 📈 Final Performance Table

| Model | MAE | RMSE | MAPE (%) |
| :--- | :--- | :--- | :--- |
| XGBoost | 154.77 | 321.57 | 3.38% |
| **LSTM** | **163.50** | **306.53** | **3.93%** |

*Note: LSTM excels in RMSE, making it safer for preventing large grid imbalances.*

---

## 🛠 How to Use

1. **Setup**: `pip install -r requirements.txt` (including `shap` and `streamlit`).
2. **Run Pipeline**: `python main.py` and `python enhance_project.py`.
3. **Launch Dashboard**: `streamlit run app.py`.

---
*Created by Antigravity AI*
