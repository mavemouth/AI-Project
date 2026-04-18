import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
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

class ModelFactory:
    KERAS_AVAILABLE = KERAS_AVAILABLE

    @staticmethod
    def get_multistep_xgboost():
        # High n_estimators and appropriate depth for multi-output
        base_xgb = XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        return MultiOutputRegressor(base_xgb)

    @staticmethod
    def get_linear_base():
        return LinearRegression()

    @staticmethod
    def build_multistep_lstm(input_shape, output_steps=24):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(output_steps)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

def create_sequences_multistep(data, target, window_size, horizon=24):
    X, y = [], []
    # target is the 1D array of power demand
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data.iloc[i : i + window_size].values)
        y.append(target[i + window_size : i + window_size + horizon])
    return np.array(X), np.array(y)

def create_tabular_multistep(data, target, horizon=24):
    # For ML models, we use current feature set to predict next N steps
    X, y = [], []
    for i in range(len(data) - horizon + 1):
        X.append(data.iloc[i].values) # Features at time T
        y.append(target[i : i + horizon]) # Targets for T to T+horizon-1
    return np.array(X), np.array(y)
