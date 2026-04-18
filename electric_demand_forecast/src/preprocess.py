import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

def load_data(filepath):
    df = pd.read_csv(filepath)
    if df.columns[0] == '' or df.columns[0].startswith('Unnamed'):
        df = df.iloc[:, 1:]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').drop_duplicates('datetime')
    df = df.set_index('datetime')
    df = df.interpolate(method='time')
    return df

def process_anomalies(df, target_col='Power demand'):
    df_clean = df.copy()
    z_scores = np.abs(stats.zscore(df[target_col]))
    anomalies = z_scores > 3
    anomaly_indices = df_clean.index[anomalies]
    
    # Replace anomalies using interpolation
    df_clean.loc[anomalies, target_col] = np.nan
    df_clean[target_col] = df_clean[target_col].interpolate(method='linear').bfill().ffill()
    
    return df_clean, anomaly_indices

def add_comprehensive_features(df, target_col='Power demand'):
    df = df.copy()
    
    # Time Features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month # Added Month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical Encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Lag Features (Target based)
    for lag in [1, 6, 12, 24, 48]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling Statistics
    df['rolling_mean_6'] = df[target_col].rolling(window=6).mean()
    df['rolling_mean_24'] = df[target_col].rolling(window=24).mean()
    
    # Weather Features
    weather_cols = ['temp', 'rhum', 'wspd']
    existing_weather = [c for c in weather_cols if c in df.columns]
    
    # Target and Drop NaNs
    df = df.dropna()
    return df

def get_train_test_split(df, target_col='Power demand', split_ratio=0.8):
    # No shuffles for split
    split = int(len(df) * split_ratio)
    train, test = df.iloc[:split], df.iloc[split:]
    return train, test

def scale_data_standard(train, test, target_col='Power demand'):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    features = [c for c in train.columns if c != target_col]
    
    X_train = scaler_x.fit_transform(train[features])
    X_test = scaler_x.transform(test[features])
    
    y_train = scaler_y.fit_transform(train[[target_col]]).flatten()
    y_test = scaler_y.transform(test[[target_col]]).flatten()
    
    # Return dataframes with columns
    X_train_df = pd.DataFrame(X_train, columns=features, index=train.index)
    X_test_df = pd.DataFrame(X_test, columns=features, index=test.index)
    
    return X_train_df, X_test_df, y_train, y_test, scaler_x, scaler_y
