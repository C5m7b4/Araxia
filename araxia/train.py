# train.py
import numpy as np
import matplotlib.pyplot as plt
from network import MLP

def normalize_features(X):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    return (X - X_mean) / X_std, X_mean, X_std

def normalize_targets(y):
    y_min = y.min()
    y_max = y.max()
    return (y - y_min) / (y_max - y_min), y_min, y_max

def denormalize_targets(y_norm, y_min, y_max):
    return y_norm * (y_max - y_min) + y_min

def train_model(X_raw, y_raw, hidden_size=10, learning_rate=0.001, epochs=300):
    # Normalize inputs and targets
    X_norm, X_mean, X_std = normalize_features(X_raw)
    y_norm, y_min, y_max = normalize_targets(y_raw)

    # Initialize and train the model
    model = MLP(input_size=X_norm.shape[1], hidden_size=hidden_size, learning_rate=learning_rate)
    model.train(X_norm, y_norm, epochs=epochs)

    return model, X_norm, X_mean, X_std, y_norm, y_min, y_max

def plot_predictions(dates, y_true, y_pred, title="Model Fit"):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="True Sales", marker='o')
    plt.plot(dates, y_pred, label="Predicted Sales", marker='x', linestyle='--')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
