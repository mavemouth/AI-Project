import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    df = pd.read_csv(filepath)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
    elif df.columns[0] == '' or df.columns[0].startswith('Unnamed'):
        df = df.iloc[:, 1:]
        df['datetime'] = pd.to_datetime(df.iloc[:, 0]) # Assume first column is datetime
        df = df.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
    
    # Fill missing timestamps
    df = df.resample('5min').interpolate(method='time')
    return df

def add_comprehensive_features(df, target_col='Power demand'):
    df = df.copy()
    
    # 1. Temporal Features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    
    # Cyclical Encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # 2. Lag Features (Aggressive - Target 25+ features)
    for lag in [1, 2, 3, 4, 5, 12, 24, 48, 72, 168]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # 3. Rolling Statistics
    for window in [6, 12, 24, 48]:
        df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
    df['rolling_std_24'] = df[target_col].shift(1).rolling(window=24).std()
    df['rolling_std_6'] = df[target_col].shift(1).rolling(window=6).std()
    
    # 4. Weather (If available)
    weather_cols = ['temp', 'rhum', 'wspd']
    existing_weather = [c for c in weather_cols if c in df.columns]
    
    # Selection: Keep features between 20-30
    cols = [target_col, 'hour', 'dayofweek', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'] + \
           [f'lag_{l}' for l in [1, 2, 3, 4, 5, 12, 24, 48, 72, 168] if f'lag_{l}' in df.columns] + \
           [f'rolling_mean_{w}' for w in [6, 12, 24, 48] if f'rolling_mean_{w}' in df.columns] + \
           ['rolling_std_24', 'rolling_std_6'] + existing_weather
           
    df = df[cols].dropna()
    return df

def get_train_test_split(df, split_ratio=0.8):
    split = int(len(df) * split_ratio)
    return df.iloc[:split], df.iloc[split:]

def scale_data_pipeline(train, test, target_col='Power demand'):
    # User requested MinMaxScaler for LSTM
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    features = [c for c in train.columns if c != target_col]
    
    # Fit ONLY on train
    scaler_x.fit(train[features])
    scaler_y.fit(train[[target_col]])
    
    # Transform both
    X_train = scaler_x.transform(train[features])
    X_test = scaler_x.transform(test[features])
    y_train = scaler_y.transform(train[[target_col]]).flatten()
    y_test = scaler_y.transform(test[[target_col]]).flatten()
    
    X_train_df = pd.DataFrame(X_train, columns=features, index=train.index)
    X_test_df = pd.DataFrame(X_test, columns=features, index=test.index)
    
    return X_train_df, X_test_df, y_train, y_test, scaler_x, scaler_y
