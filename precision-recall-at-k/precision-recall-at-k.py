def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # if k <= 0: 
    #     return 0.0, 0.0 
    top_k = set(recommended[:k])
    intersect = top_k & set(relevant)
    precision_k = len(intersect) / k 
    recall_k = len(intersect) / len(relevant)
    return [precision_k, recall_k] 
    