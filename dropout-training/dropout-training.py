import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.asarray(x)
    if rng is not None:
        rand = np.array([rng.random() for _ in range(x.size)]).reshape(x.shape)
    else:
        rand = np.random.random(x.shape)

    keep_mask = rand >= p

    dropout_pattern = keep_mask.astype(float) / (1 - p)

    output = x * dropout_pattern

    return output, dropout_pattern
    