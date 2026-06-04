import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)

    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    n = len(y_true)

    return float(-np.mean(np.log(y_pred[np.arange(n), y_true])))
    