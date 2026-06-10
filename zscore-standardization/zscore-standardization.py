import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    X = np.asarray(X, dtype=float)

    mean_value = np.mean(X, axis=axis, keepdims=True) 
    std_value = np.std(X, axis=axis, keepdims=True) 

    z_score = (X - mean_value) / (std_value + eps)
    return z_score