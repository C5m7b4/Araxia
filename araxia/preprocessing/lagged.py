import numpy as np

def create_lagged_dataset(df, target_col='sales', lag=5):
    X, y = [], []
    for i in range(lag, len(df)):
        lagged_values = df[target_col].values[i - lag:i].tolist()
        row = df.iloc[i]
        X.append(lagged_values)
        y.append(row[target_col])
    return np.array(X), np.array(y)