import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    N, C, H, W_in = x.shape
    F, Cw, KH, KW = W.shape

    OH = H - KH + 1
    OW = W_in - KW + 1

    out = np.zeros((N, F, OH, OW), dtype=float)

    for n in range(N): 
        for f in range(F):
            for i in range(OH):
                for j in range(OW):
                    patch = x[n, :, i:i+KH, j:j+KW]
                    out[n, f, i, j] = np.sum(patch * W[f]) + b[f]

    return out
    