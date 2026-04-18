import streamlit as st
import pandas as pd
import os
from PIL import Image

# Set page config
st.set_page_config(page_title="Electric Demand Forecaster", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #00ffcc;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ Short-Term Electricity Demand Forecasting")
st.markdown("---")

# Load results
RESULTS_DIR = "outputs"

if not os.path.exists(os.path.join(RESULTS_DIR, "model_results.csv")):
    st.warning("⚠️ No results found. Please run main.py first to train models.")
else:
    results_df = pd.read_csv(os.path.join(RESULTS_DIR, "model_results.csv"))
    
    # Sidebar
    st.sidebar.title("Configuration")
    best_model = results_df.loc[results_df['RMSE'].idxmin()]['Model']
    st.sidebar.success(f"Best Model: {best_model}")
    
    # Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Model Performance")
        st.table(results_df)
        
        st.subheader("📈 Performance Metrics Comparison")
        st.image(os.path.join(RESULTS_DIR, "metrics_comparison.png"))

    with col2:
        st.subheader("📉 Actual vs Predicted")
        selected_model = st.selectbox("Select Model to View", results_df['Model'])
        img_path = os.path.join(RESULTS_DIR, f"actual_vs_pred_{selected_model}.png")
        if os.path.exists(img_path):
            st.image(img_path)
            
        # Feature Importance
        imp_path = os.path.join(RESULTS_DIR, f"importance_{selected_model}.png")
        if os.path.exists(imp_path):
            st.subheader("💡 Feature Importance")
            st.image(imp_path)

    st.markdown("---")
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("🕒 Demand Time Series")
        st.image(os.path.join(RESULTS_DIR, "demand_plot.png"))
        
        st.subheader("🔍 Anomaly Detection")
        st.image(os.path.join(RESULTS_DIR, "anomalies_plot.png"))

    with col4:
        st.subheader("🔮 Next 1-Hour Forecast")
        st.image(os.path.join(RESULTS_DIR, "multi_step_forecast.png"))
        
        st.subheader("🔥 Feature Correlations")
        st.image(os.path.join(RESULTS_DIR, "correlation_heatmap.png"))

    st.markdown("---")
    st.subheader("💡 Insights & Observations")
    st.info("""
    - **Trend**: Demand shows clear periodicity (daily/hourly).
    - **Anomalies**: Several spikes detected that don't follow the usual curve, possibly due to events or weather extremes.
    - **Models**: Tree-based models (Random Forest, XGBoost) tend to capture non-linear demand patterns better than Linear Regression.
    - **Deep Learning**: LSTM/GRU are effective at maintaining temporal context over sequences.
    """)
