import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    result = 0 
    for i in range(len(x)): 
        result += np.abs(x[i] - y[i])
    return float(result)
        