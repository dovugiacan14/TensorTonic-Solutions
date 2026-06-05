import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    x = np.asarray(x, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)

    # Convert parameters
    Wz = np.asarray(params["Wz"], dtype=float)
    Uz = np.asarray(params["Uz"], dtype=float)
    bz = np.asarray(params["bz"], dtype=float)

    Wr = np.asarray(params["Wr"], dtype=float)
    Ur = np.asarray(params["Ur"], dtype=float)
    br = np.asarray(params["br"], dtype=float)

    Wh = np.asarray(params["Wh"], dtype=float)
    Uh = np.asarray(params["Uh"], dtype=float)
    bh = np.asarray(params["bh"], dtype=float)

    D = Wz.shape[0]
    H = Wz.shape[1]

    x, x_was_1d = _as2d(x, D)
    h_prev, _ = _as2d(h_prev, H)

    # Update gate
    z = _sigmoid(
        x @ Wz +
        h_prev @ Uz +
        bz
    )

    # Reset gate
    r = _sigmoid(
        x @ Wr +
        h_prev @ Ur +
        br
    )

    # Candidate hidden state
    h_tilde = np.tanh(
        x @ Wh +
        (r * h_prev) @ Uh +
        bh
    )

    # New hidden state
    h = (1.0 - z) * h_prev + z * h_tilde
    return h[0] if x_was_1d else h 