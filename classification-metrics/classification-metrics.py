import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(np.mean(y_true == y_pred))
    
    def compute_prf(label): 
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp) if (tp+fp) else 0.0 
        recall = tp / (tp + fn) if (tp + fn) else 0.0 

        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0 

        support = np.sum(y_true == label)
        return precision, recall, f1, support, tp, fp, fn

    labels = np.union1d(y_true, y_pred)

    if average == "binary":
        precision, recall, f1, _, _, _, _ = compute_prf(pos_label)
    else: 
        stats = [compute_prf(label) for label in labels]

        if average == "micro":
            tp = sum(s[4] for s in stats)
            fp = sum(s[5] for s in stats)
            fn = sum(s[6] for s in stats)

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0

            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )

        elif average == "macro":
            precision = np.mean([s[0] for s in stats])
            recall = np.mean([s[1] for s in stats])
            f1 = np.mean([s[2] for s in stats])

        elif average == "weighted":
            weights = np.array([s[3] for s in stats], dtype=float)

            precision = np.average(
                [s[0] for s in stats],
                weights=weights
            )

            recall = np.average(
                [s[1] for s in stats],
                weights=weights
            )

            f1 = np.average(
                [s[2] for s in stats],
                weights=weights
            )

        else:
            raise ValueError("Invalid average")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    
    