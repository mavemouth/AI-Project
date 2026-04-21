import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.style.use('dark_background')

def calculate_metrics_optimized(y_true, y_pred, model_name):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def plot_final_benchmark(results_df):
    plt.figure(figsize=(10, 6))
    melted = results_df.melt(id_vars='Model', value_vars=['MAPE'])
    sns.barplot(data=melted, x='Model', y='value', palette='magma')
    plt.title('Final Model Performance Comparison (MAPE)', fontsize=14)
    plt.ylabel('MAPE (%)')
    plt.savefig('outputs/error_comparison.png')
    plt.close()

def plot_lstm_loss(history):
    if hasattr(history, 'history'):
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Train Loss', color='#00ffcc')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Val Loss', color='#ff007f')
        plt.title('LSTM Training Stability')
        plt.legend()
        plt.savefig('outputs/lstm_loss.png')
        plt.close()

def plot_predictions(y_actual, y_pred, model_name):
    plt.figure(figsize=(12, 5))
    plt.plot(y_actual[:300], label='Actual', color='#00ffcc', alpha=0.8)
    plt.plot(y_pred[:300], label='Predicted', color='#ff007f', linestyle='--')
    plt.title(f'Demand Forecast: Actual vs {model_name}')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(f'outputs/actual_vs_pred_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
