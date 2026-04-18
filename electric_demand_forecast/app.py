import streamlit as st
import pandas as pd
import os
from PIL import Image
import joblib

# Deep Learning (LSTM) is disabled in this deployment to ensure Python 3.14 compatibility
LSTM_AVAILABLE = False

# Set page config for a premium feel
st.set_page_config(
    page_title="AI PowerGrid Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #00ffcc;
    }
    .stTable {
        background-color: transparent;
    }
    h1, h2, h3 {
        color: #00ffcc !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/parakeet/128/lightning-bolt.png", width=100)
st.sidebar.title("Grid Insights Pro")
st.sidebar.markdown("---")

# Load data
RESULTS_DIR = "outputs"
MODELS_DIR = "models"
METRICS_FILE = os.path.join(RESULTS_DIR, "enhanced_results.csv")

if not os.path.exists(METRICS_FILE):
    st.error("🚀 **Error:** Performance metrics not found. Please run `python enhance_project.py` first.")
else:
    metrics_df = pd.read_csv(METRICS_FILE)
    
    # 🌟 Deep Learning (LSTM) Check & Filtering
    if not LSTM_AVAILABLE:
        st.warning("🤖 **LSTM model disabled** for this deployment to ensure zero-build compatibility with Python 3.14. Showing the highest performing statistical model (XGBoost).")
        metrics_df = metrics_df[metrics_df['Model'] != 'LSTM'].reset_index(drop=True)

    st.title("⚡ AI-Powered Electricity Demand Forecasting")
    st.subheader("Model Validation & Explainability Dashboard")
    
    # Hero Metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    best_row = metrics_df.loc[metrics_df['RMSE'].idxmin()]
    
    with col_m1:
        st.markdown(f"""<div class='metric-card'><h3>Best Model</h3><p style='font-size: 24px;'>{best_row['Model']}</p></div>""", unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""<div class='metric-card'><h3>Minimum RMSE</h3><p style='font-size: 24px;'>{best_row['RMSE']:.2f}</p></div>""", unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""<div class='metric-card'><h3>Average MAPE</h3><p style='font-size: 24px;'>{best_row['MAPE']:.2f}%</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Performance Analysis", "🔍 Model Interpretability (XAI)", "🔮 Live Prediction Concept"])

    with tab1:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.write("### Model Comparison Table")
            st.dataframe(metrics_df.style.highlight_min(subset=['MAE', 'RMSE', 'MAPE'], color='#2E7D32'))
            
            st.write("### Resource Efficiency")
            st.info("Models are optimized using **Early Stopping** and **Multi-Output strategies** to ensure inference speed < 10ms per forecast.")

        with col2:
            st.write("### Multi-Step Forecast (24 Steps / 2 Hours)")
            forecast_img = os.path.join(RESULTS_DIR, "enhanced_forecast_plot.png")
            if os.path.exists(forecast_img):
                st.image(forecast_img, use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            st.write("### Error Residual Analysis")
            resid_img = os.path.join(RESULTS_DIR, "residual_xgboost_enhanced.png")
            if os.path.exists(resid_img):
                st.image(resid_img, use_container_width=True)
        with col4:
            st.write("### Actual vs Predicted Distribution")
            act_pred_img = os.path.join(RESULTS_DIR, "actual_vs_pred_xgboost.png")
            if os.path.exists(act_pred_img):
                st.image(act_pred_img, use_container_width=True)

    with tab2:
        st.write("### Explainable AI (SHAP Insights)")
        st.write("Understanding which features drive the forecast using Shapley values.")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.image(os.path.join(RESULTS_DIR, "shap_summary.png"), caption="SHAP Summary Plot")
        with col_s2:
            st.image(os.path.join(RESULTS_DIR, "shap_bar.png"), caption="Global Feature Importance")
            
        st.success("💡 **Key Insight:** Lagged demand and time-based features (hour_sin/cos) are the primary drivers for 2-hour ahead forecasting.")

    with tab3:
        st.write("### Real-World Deployment Pipeline")
        st.markdown("""
        #### How to deploy this system:
        1. **Inference Server**: Load `models/xgb_model.pkl`. 
        2. **LSTM Support**: LSTM requires a `torch` or `tensorflow` environment. If skipped, use XGBoost as the primary production model.
        3. **Preprocessing**: Use `models/scaler_x.pkl` and `models/scaler_y.pkl` to scale incoming data.
        4. **Data Stream**: Connect to a Kafka or RabbitMQ stream of smart-meter data.
        5. **Weather Integration**: Fetch real-time weather via API (OpenWeather) to update `temp`, `rhum` variables.
        6. **Scaling**: Use a Docker container on Kubernetes to handle spikes in demand requests.
        """)
        
        st.warning("⚠️ **Ethical Note:** Forecast errors should be used to plan backup supply (spinning reserves) to avoid grid instability.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Created by Aditya and Manvi </p>", unsafe_allow_html=True)
