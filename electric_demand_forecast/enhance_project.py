import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from src.preprocess import load_data, process_anomalies, add_comprehensive_features, get_train_test_split, scale_data_standard
from src.models import create_sequences_multistep, create_tabular_multistep
from src.evaluate import calculate_metrics, plot_residuals, plot_preds_vs_actual

# Set plotting style
plt.style.use('dark_background')
sns.set_palette("husl")

def main():
    print("Starting Post-Training Enhancement Pipeline (SHAP & Visuals)...")
    os.makedirs('outputs', exist_ok=True)
    
    HORIZON = 24 
    WINDOW = 48  

    # 1. Load trained models and data
    if not os.path.exists('models/xgb_model.pkl'):
        print("Error: Models not found. Run main.py first.")
        return

    xgb_multi = joblib.load('models/xgb_model.pkl')
    scaler_x = joblib.load('models/scaler_x.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')

    # Load Data (Same as main.py for consistent evaluation)
    df_raw = load_data('../powerdemand_5min_2021_to_2024_with weather.csv')
    df_raw = df_raw.iloc[-100000:] 
    df_clean, _ = process_anomalies(df_raw)
    df_feat = add_comprehensive_features(df_clean)
    _, test_df = get_train_test_split(df_feat)
    
    # Pre-scale for evaluation
    features = [c for c in test_df.columns if c != 'Power demand']
    X_ts = scaler_x.transform(test_df[features])
    y_ts = scaler_y.transform(test_df[['Power demand']]).flatten()
    
    X_test_df = pd.DataFrame(X_ts, columns=features, index=test_df.index)
    feat_names = features

    X_ts_multi, y_ts_multi = create_tabular_multistep(X_test_df, y_ts, HORIZON)
    
    # --- 2. XGBoost SHAP (Explainable AI) ---
    print("\n[1/4] Generating SHAP Explainability for XGBoost...")
    # Use a bigger sample for SHAP now that we have more data, but cap it for speed
    explainer = shap.TreeExplainer(xgb_multi.estimators_[0])
    shap_sample = X_ts_multi[:1000] 
    shap_values = explainer.shap_values(shap_sample)

    # SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, shap_sample, feature_names=feat_names, show=False)
    plt.title("SHAP Feature Importance (XGBoost - 1st Step)")
    plt.tight_layout()
    plt.savefig('outputs/shap_summary.png')
    plt.close()

    # SHAP Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, shap_sample, feature_names=feat_names, plot_type="bar", show=False)
    plt.title("Mean SHAP Values (Global Feature Importance)")
    plt.tight_layout()
    plt.savefig('outputs/shap_bar.png')
    plt.close()

    # --- 3. Predictions & Metrics ---
    print("\n[2/4] Re-evaluating Metrics...")
    y_pred_xgb = xgb_multi.predict(X_ts_multi)
    y_pred_xgb_inv = scaler_y.inverse_transform(y_pred_xgb)
    y_ts_multi_inv = scaler_y.inverse_transform(y_ts_multi)
    
    res_xgb = calculate_metrics(y_ts_multi_inv, y_pred_xgb_inv, 'XGBoost')
    results = [res_xgb]

    if os.path.exists('models/lstm_model.keras'):
        print("Evaluating LSTM...")
        import keras
        lstm = keras.models.load_model('models/lstm_model.keras')
        X_lstm_ts, y_lstm_ts = create_sequences_multistep(X_test_df, y_ts, WINDOW, HORIZON)
        
        y_pred_lstm = lstm.predict(X_lstm_ts, verbose=0)
        y_pred_lstm_inv = scaler_y.inverse_transform(y_pred_lstm)
        y_ts_lstm_inv = scaler_y.inverse_transform(y_lstm_ts)
        results.append(calculate_metrics(y_ts_lstm_inv, y_pred_lstm_inv, 'LSTM'))

    # --- 4. Additional Visualizations ---
    print("\n[3/4] Generating Additional Visualizations...")
    plot_residuals(y_ts_multi_inv[:, 0], y_pred_xgb_inv[:, 0], 'XGBoost_Enhanced')
    
    # Improved Multi-step Forecast Plot
    plt.figure(figsize=(12, 6))
    sample_idx = -1
    plt.plot(range(HORIZON), y_ts_multi_inv[sample_idx], 'o-', label='Actual Demand', color='#00ffcc', markersize=4)
    plt.plot(range(HORIZON), y_pred_xgb_inv[sample_idx], 'x--', label='XGBoost Prediction', color='#ff007f', markersize=4)
    plt.title(f'Enhanced Multi-step Forecast Comparison (Horizon={HORIZON})', fontsize=14)
    plt.xlabel('Future Time Steps (5-min intervals)')
    plt.ylabel('Electricity Demand (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/enhanced_forecast_plot.png')
    plt.close()

    # --- 5. Final Output Table ---
    print("\n[4/4] Final Results Summary...")
    metrics_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("RESTORED PERFORMANCE COMPARISON")
    print("="*50)
    print(metrics_df[['Model', 'MAE', 'RMSE', 'MAPE']].to_string(index=False))
    print("="*50)
    
    metrics_df.to_csv('outputs/enhanced_results.csv', index=False)
    print("\nPost-training tasks complete!")

if __name__ == "__main__":
    main()
