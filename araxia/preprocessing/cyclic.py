import numpy as np
# from atrax import Atrax as tx
from atrax import to_datetime

def create_dataset_with_cyclic_features(ds, date_col='sale_date', target_col='sales', lag=5):
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
    X, y = [], []
    try:
        ds[date_col] = to_datetime(ds[date_col])
        ds = ds.sort_values(by=date_col)


        ds['day_of_week'] = ds[date_col].dt.weekday
        ds['is_weekend'] = ds[date_col].dt.is_weekend.astype(int)
        ds['day_of_month'] = ds[date_col].dt.day
        ds['month'] = ds[date_col].dt.month 

        weekday = ds['day_of_week'].values[0]

        ds['dow_sin'] = np.sin(2 * np.pi * weekday / 7)
        ds['dow_cos'] = np.cos(2 * np.pi * weekday / 7)
        ds['month_sin'] = np.sin(2 * np.pi * ds['month'] / 12)
        ds['month_cos'] = np.cos(2 * np.pi * ds['month'] / 12)

        
        for i in range(lag, len(ds)):
            lagged_values = ds[target_col].values[i - lag:i]
            row = ds.iloc[i]
            features = lagged_values + [
                row['dow_sin'].values[0], 
                row['dow_cos'].values[0],
                row['month_cos'].values[0],
                row['month_sin'].values[0], 
                row['day_of_week'].values[0],
                row['day_of_month'].values[0],
                row['month'].values[0],
                row['is_weekend'].values[0]
            ]
            X.append(features)
            y.append(row[target_col].values[0])
        return np.array(X), np.array(y)
    except Exception as e:
        raise ValueError(f"Error processing dataset: {e}")
