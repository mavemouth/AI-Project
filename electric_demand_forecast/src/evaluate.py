import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.style.use('dark_background')
sns.set_palette("husl")

def calculate_metrics(y_true, y_pred, model_name):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def plot_anomaly_comparison(df_orig, df_clean, anomaly_indices):
    plt.figure(figsize=(15, 6))
    plt.plot(df_orig.index[:5000], df_orig['Power demand'][:5000], color='red', alpha=0.5, label='Original')
    plt.plot(df_clean.index[:5000], df_clean['Power demand'][:5000], color='#00ffcc', alpha=0.8, label='Cleaned')
    plt.title('Anomaly Detection & Replacement (Z-Score)', fontsize=16)
    plt.legend()
    plt.savefig('outputs/anomalies_before_after.png')
    plt.close()

def plot_feature_importance_xgb(model, feature_names):
    plt.figure(figsize=(12, 8))
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sns.barplot(x=importances[sorted_idx[:15]], y=np.array(feature_names)[sorted_idx[:15]], palette='magma')
    plt.title('Top 15 Features - XGBoost Importance', fontsize=16)
    plt.savefig('outputs/importance_xgboost.png')
    plt.close()

def plot_multistep_forecast(y_actual, y_pred, model_name):
    # y_actual and y_pred expected to be (steps,)
    plt.figure(figsize=(12, 6))
    steps = range(1, len(y_actual) + 1)
    plt.plot(steps, y_actual, label='Actual Demand', color='#00ffcc', linewidth=2, linestyle='-')
    plt.plot(steps, y_pred, label='Predicted Demand', color='#ff007f', linewidth=2, linestyle='--')
    
    plt.title(f'24-Step Ahead Electricity Demand Forecast ({model_name})', fontsize=16)
    plt.xlabel('Steps Ahead (5-min intervals)', fontsize=12)
    plt.ylabel('Electricity Demand', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'outputs/multistep_forecast_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_comprehensive_metrics(results_df):
    plt.figure(figsize=(12, 6))
    melted = results_df.melt(id_vars='Model', value_vars=['MAE', 'RMSE'])
    sns.barplot(data=melted, x='Model', y='value', hue='variable')
    plt.title('Model Error Comparison', fontsize=16)
    plt.savefig('outputs/error_comparison.png')
    plt.close()

# Re-use standard EDA plots
def plot_basic_eda(df):
    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=False, cmap='mako')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()
    
    # Generic Time series
    plt.figure(figsize=(15, 5))
    plt.plot(df.index[-2000:], df['Power demand'][-2000:], color='#00ffcc')
    plt.title('Demand Time Series (Recent Trend)')
    plt.savefig('outputs/time_series_plot.png')
    plt.close()

def plot_loss_curve(history):
    if hasattr(history, 'history'):
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Train Loss', color='#00ffcc')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Val Loss', color='#ff007f')
        plt.title('LSTM Training Loss')
        plt.legend()
        plt.savefig('outputs/lstm_loss.png')
        plt.close()

def plot_preds_vs_actual(y_true, y_pred, model_name):
    plt.figure(figsize=(15, 6))
    plt.plot(y_true[-300:], label='Actual', color='#00ffcc', alpha=0.8, linewidth=1.5)
    plt.plot(y_pred[-300:], label='Predicted', color='#ff007f', linestyle='--', linewidth=1.5)
    plt.title(f'Actual vs Predicted Demand - {model_name}', fontsize=16)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Electricity Demand', fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.savefig(f'outputs/actual_vs_pred_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(15, 5))
    plt.plot(residuals[-500:], color='#00ffcc', alpha=0.7)
    plt.axhline(y=0, color='#ff007f', linestyle='--')
    plt.title(f'Residual Plot (Errors over Time) - {model_name}', fontsize=16)
    plt.xlabel('Samples', fontsize=12)
    plt.ylabel('Residual (Actual - Predicted)', fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.savefig(f'outputs/residual_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
