import numpy as np
import pandas as pd

def create_dataset_with_cyclic_features(ds, target_col='sales', lag=5):
    """
    Create a dataset with cyclic features for time series forecasting.
    
    Parameters:
    - ds: DataFrame containing the time series data.
    - target_col: Name of the target column to predict.
    - lag: Number of lagged observations to include as features.
    
    Returns:
    - X: Features array with cyclic features and lagged values.
    - y: Target values array.
    """
    df = df.copy()
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['day_of_week'] = df['sale_date'].dt.weekday
    df['day_of_month'] = df['sale_date'].dt.day
    df['month'] = df['sale_date'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    X, y = [], []
    for i in range(lag, len(df)):
        lagged = df[target_col].values[i - lag:i].tolist()
        row = df.iloc[i]
        features = lagged + [
            row['dow_sin'], row['dow_cos'],
            row['day_of_month'],
            row['month_sin'], row['month_cos'],
            row['is_weekend']
        ]
        X.append(features)
        y.append(row[target_col])
    return np.array(X), np.array(y)
