import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    x = np.asarray(x, dtype=float)

    if x.ndim == 3: 
        return np.mean(x, axis=(1,2))

    if x.ndim == 4: 
        return np.mean(x, axis=(2,3))

    raise ValueError("Input must have shape (C,H,W) or (N,C,H,W)")