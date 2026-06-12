import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # if len(y_pred) != len(y_true): 
    #     return 0.0 
        
    # numerator = 0.0 
    # dominator = 0.0
    # for i in range(len(y_pred)): 
    #     numerator += (y_pred[i] - y_true[i])**2

    # mean_y = np.mean(y_true)
    # for i in range(len(y_true)): 
    #     dominator += (y_true[i] - mean_y) ** 2

    # return numerator / dominator 
    numerator = np.sum((y_true - y_pred) ** 2)
    mean_y = np.mean(y_true)
    dominator = np.sum((y_true - mean_y) ** 2)

    if dominator == 0: 
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0 

    return float(1.0 - numerator / dominator)

    