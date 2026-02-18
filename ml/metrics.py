# Metrics
import numpy as np
from sklearn.metrics import mean_absolute_error


def normalized_mae(y_true, y_pred):
    """
    Calculate the normalized mean absolute error (MAE) between true and predicted values.

    The normalized MAE is computed as the mean absolute error divided by the mean 
    of the absolute true values. This normalization provides a relative measure of 
    the error compared to the scale of the true values.

    Args:
        y_true (array-like): Ground truth (true) values.
        y_pred (array-like): Predicted values.

    Returns:
        float: The normalized mean absolute error.
    """
    return mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true))


def within_band_accuracy(y_true, y_pred, band=0.3):
    """
    Calculates the proportion of predictions that fall within a specified 
    relative error band compared to the true values.
    Args:
        y_true (array-like): The ground truth values.
        y_pred (array-like): The predicted values.
        band (float, optional): The relative error tolerance band. Defaults to 0.3.
    Returns:
        float: The proportion of predictions within the specified relative error band.
    """
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-6) < band)


def quantile_loss(y_true, y_pred, q):
    """
    Calculates the quantile loss, a metric used to measure the accuracy of 
    quantile regression models. The quantile loss is asymmetric and penalizes 
    over-predictions and under-predictions differently based on the quantile value.

    Args:
        y_true (array-like): The true target values.
        y_pred (array-like): The predicted target values.
        q (float): The quantile to be evaluated, where 0 < q < 1. For example, 
                   q=0.5 corresponds to the median (50th percentile).

    Returns:
        float: The mean quantile loss value.

    Notes:
        - If q < 0.5, the loss penalizes under-predictions more heavily.
        - If q > 0.5, the loss penalizes over-predictions more heavily.
        - At q=0.5, the loss is equivalent to the mean absolute error (MAE).
    """
    diff = y_true - y_pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def coverage(y_true, y_pred):
    """
    Calculates the coverage metric, which measures the proportion of true values 
    that are less than or equal to the predicted values.

    Args:
        y_true (array-like): The ground truth values.
        y_pred (array-like): The predicted values.

    Returns:
        float: The mean proportion of true values that are less than or equal to 
        the predicted values.
    """
    return np.mean(y_true <= y_pred)