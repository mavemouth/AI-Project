import os
import random
import numpy as np
import torch
os.environ["KERAS_BACKEND"] = "torch"

try:
    import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, Input
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ModelFactory:
    KERAS_AVAILABLE = KERAS_AVAILABLE
    
    @staticmethod
    def get_direct_xgboost():
        # Strong hyperparameters as requested
        base_xgb = XGBRegressor(
            n_estimators=800,
            max_depth=7, # Increased depth for more complexity
            learning_rate=0.03, # Slightly faster learning
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            reg_alpha=0.2,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        return MultiOutputRegressor(base_xgb)

    @staticmethod
    def get_linear_simple():
        return LinearRegression()

    @staticmethod
    def build_restored_lstm(input_shape, output_steps=24):
        set_seeds()
        model = Sequential([
            Input(shape=input_shape),
            # Robust LSTM architecture
            LSTM(128, return_sequences=False),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dense(output_steps)
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

def create_sequences_multistep(data, target, window_size, horizon=24):
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data.iloc[i : i + window_size].values)
        y.append(target[i + window_size : i + window_size + horizon])
    return np.array(X), np.array(y)

def create_tabular_direct(data, target, horizon=24):
    X, y = [], []
    for i in range(len(data) - horizon):
        X.append(data.iloc[i].values)
        y.append(target[i + 1 : i + 1 + horizon])
    return np.array(X), np.array(y)
