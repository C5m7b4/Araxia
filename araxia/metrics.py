def MAPE(y_true, y_pred):
    """
    Calculate the mean absolute percentage error (MAPE) between true and predicted values.

    Args:
        y_true (list or np.array): True values.
        y_pred (list or np.array): Predicted values.

    Returns:
        float: The mean absolute percentage error.
    """
    n = len(y_true)
    return sum(abs((y - y_hat) / y) for y, y_hat in zip(y_true, y_pred) if y != 0) / n if n > 0 else float('inf')