import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    # Write code here
    n_samples = len(X) 
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2: 
        return 0.0 

    scores = []
    for i in range(n_samples): 
        current_label = labels[i]
        same_cluster = np.where(labels == current_label)[0]

        # mean intra-cluster distance 
        same_cluster = same_cluster[same_cluster != i] 
        if len(same_cluster) == 0: 
            scores.append(0.0)
            continue 
        
        a = np.mean(
            np.linalg.norm(X[i] - X[same_cluster], axis=1)
        )

        # b(i) smallest mean distance to another clutter 
        b = np.inf 
        for label in unique_labels: 
            if label == current_label: 
                continue 

            other_cluster = np.where(labels == label)[0]
            dist = np.mean(
                np.linalg.norm(X[i] - X[other_cluster], axis=1)
            )

            b = min(b, dist)
        s = (b - a) / max(a, b) 
        scores.append(s) 

    return np.mean(scores)
    

        