import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    m, n = X.shape 

    # intialize parameters 
    w = np.zeros(n)
    b = 0.0 

    for _ in range(steps):
        z = X @ w + b 
        p = _sigmoid(z)
        
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)

        loss = -np.mean(
            y * np.log(p) + (1 - y) * np.log(1 - p)
        )

        # gradients 
        error = p - y 
        dw = (X.T @ error) / m 
        db = np.mean(error)

        w -= lr * dw
        b -= lr * db 
    return w, b 
       
    