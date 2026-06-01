import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)

    single_point = points.ndim == 1 
    if single_point: 
        points = points[None, :]

    ones = np.ones((points.shape[0], 1))
    points_h = np.concatenate([points, ones], axis=1)
    transformed = points_h @ T.T 
    transformed = transformed[:, :3]
    if single_point: 
        return transformed[0]
    return transformed 
    