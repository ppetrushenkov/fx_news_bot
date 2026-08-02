import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    accuracy_score,
    precision_score, 
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    mean_pinball_loss,
    cohen_kappa_score
)
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import spearmanr


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


def ordinal_gradation_metrics(
    y_true,
    y_pred,
    y_min: int = 0,
    y_max: int = 7,
):
    """
    Metrics for ordered labels (e.g. future volatility gradation 0..7).

    Training should prefer RMSE so |0−7| is penalized much more than |0−3|.
    Here we also report agreement measures that respect ordering (Spearman,
    quadratic weighted kappa) and practical "off-by-one" accuracy.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    clipped = np.clip(y_pred, y_min, y_max)
    rounded = np.rint(clipped).astype(int)
    rounded = np.clip(rounded, y_min, y_max)
    y_true_int = np.rint(y_true).astype(int)

    rho, _ = spearmanr(y_true, clipped, nan_policy="omit")
    try:
        rho = float(rho)
        if not np.isfinite(rho):
            rho = 0.0
    except (TypeError, ValueError):
        rho = 0.0

    try:
        qwk = cohen_kappa_score(y_true_int, rounded, weights="quadratic")
    except ValueError:
        qwk = float("nan")

    return {
        "MAE": float(mean_absolute_error(y_true, clipped)),
        "RMSE": float(mean_squared_error(y_true, clipped) ** 0.5),
        "MAE_rounded": float(mean_absolute_error(y_true_int, rounded)),
        "Within1": float(np.mean(np.abs(y_true_int - rounded) <= 1)),
        "Spearman": float(rho),
        "QuadraticKappa": float(qwk),
    }
