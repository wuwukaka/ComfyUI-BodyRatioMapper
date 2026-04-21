import numpy as np

"""
Vectorized geometry helpers for point-set operations.

These helpers are intentionally pure-numpy and side-effect-free except where
explicitly noted. They are used to reduce Python-loop overhead without changing
business-level control flow.
"""


def valid_point_mask(points, eps=0.01):
    """
    Return boolean mask for valid points by non-zero magnitude.

    points: (..., 2)
    returns: (...) bool
    """
    if not isinstance(points, np.ndarray):
        points = np.asarray(points)
    return np.sum(np.abs(points), axis=-1) > float(eps)


def masked_add(points, delta, eps=0.01):
    """
    In-place add delta to valid points only.

    Supports:
    - points shape (N, 2) with delta (2,) or (N, 2)
    - points shape (1, N, 2) with delta (2,) or (N, 2) or (1, N, 2)
    """
    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a numpy ndarray")

    mask = valid_point_mask(points, eps=eps)

    if isinstance(delta, np.ndarray):
        if delta.shape == points.shape:
            points[mask] += delta[mask]
            return
        if len(points.shape) == 3 and points.shape[0] == 1 and delta.shape == points.shape[1:]:
            points[mask] += delta[np.newaxis, ...][mask]
            return
        points[mask] += delta
        return

    points[mask] += delta


def scale_about_root(points, root, scale, eps=0.01):
    """
    In-place isotropic scaling around root for valid points.

    points: (N, 2)
    root: (2,)
    """
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
        return
    if not isinstance(root, np.ndarray):
        root = np.asarray(root, dtype=float)
    if np.sum(np.abs(root)) <= float(eps):
        return
    mask = valid_point_mask(points, eps=eps)
    points[mask] = root + (points[mask] - root) * float(scale)


def scale_about_center(points, center, scale_x, scale_y):
    """
    Return scaled points around a center (no in-place requirement).

    points: (..., 2)
    center: (2,)
    """
    if not isinstance(points, np.ndarray):
        points = np.asarray(points, dtype=float)
    if not isinstance(center, np.ndarray):
        center = np.asarray(center, dtype=float)
    centered = points - center
    centered[..., 0] *= float(scale_x)
    centered[..., 1] *= float(scale_y)
    return centered + center

