"""Helper functions for TNA package."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_matrix(x: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Convert input to numpy array if needed."""
    if isinstance(x, pd.DataFrame):
        return x.values
    return np.asarray(x)


def get_labels(x: np.ndarray | pd.DataFrame, labels: list[str] | None = None) -> list[str]:
    """Extract or generate labels for states."""
    if labels is not None:
        return list(labels)
    if isinstance(x, pd.DataFrame):
        return list(x.columns)
    n = x.shape[1] if x.ndim == 2 else len(np.unique(x[~pd.isna(x)]))
    return [f"S{i+1}" for i in range(n)]


def row_normalize(mat: np.ndarray) -> np.ndarray:
    """Normalize matrix rows to sum to 1 (transition probabilities)."""
    row_sums = mat.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return mat / row_sums


def minmax_scale(mat: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1]."""
    min_val = mat.min()
    max_val = mat.max()
    if max_val == min_val:
        return np.zeros_like(mat)
    return (mat - min_val) / (max_val - min_val)


def max_scale(mat: np.ndarray) -> np.ndarray:
    """Divide by maximum value."""
    max_val = mat.max()
    if max_val == 0:
        return mat.copy()
    return mat / max_val


def rank_scale(mat: np.ndarray) -> np.ndarray:
    """Convert to ranks (1-based, average ties)."""
    from scipy.stats import rankdata
    flat = mat.flatten()
    # Rank non-zero values, keep zeros as zero
    ranks = rankdata(flat, method='average')
    # Set zeros to zero rank
    ranks[flat == 0] = 0
    return ranks.reshape(mat.shape)


def apply_scaling(mat: np.ndarray, scaling: str | list[str] | None) -> tuple[np.ndarray, list[str]]:
    """Apply scaling method(s) to matrix.

    Parameters
    ----------
    mat : np.ndarray
        Input matrix
    scaling : str or list of str, optional
        Scaling method(s): 'minmax', 'max', 'rank', or None

    Returns
    -------
    tuple
        (scaled_matrix, list_of_applied_scalings)
    """
    if scaling is None:
        return mat.copy(), []

    if isinstance(scaling, str):
        scaling = [scaling]

    result = mat.copy()
    applied = []

    for method in scaling:
        method = method.lower()
        if method == 'minmax':
            result = minmax_scale(result)
            applied.append('minmax')
        elif method == 'max':
            result = max_scale(result)
            applied.append('max')
        elif method == 'rank':
            result = rank_scale(result)
            applied.append('rank')
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    return result, applied


def is_square_matrix(x: np.ndarray) -> bool:
    """Check if array is a square matrix."""
    return x.ndim == 2 and x.shape[0] == x.shape[1]


def is_sequence_data(x: np.ndarray | pd.DataFrame) -> bool:
    """Check if input appears to be sequence data (not a weight matrix)."""
    if isinstance(x, pd.DataFrame):
        # DataFrames with non-numeric columns are sequence data
        for col in x.columns:
            dtype = x[col].dtype
            # Check for string types (including pandas StringDtype)
            if dtype == object or dtype.name == 'string' or 'str' in str(dtype).lower():
                return True
            # Check for categorical
            if hasattr(dtype, 'name') and dtype.name == 'category':
                return True
            # Try to check if numeric
            try:
                if not np.issubdtype(dtype, np.number):
                    return True
            except TypeError:
                # If we can't determine, assume it's sequence data
                return True
        # Check if values look like categorical/sequence data
        # (small number of unique values relative to size)
        unique_vals = pd.unique(x.values.ravel())
        unique_vals = unique_vals[~pd.isna(unique_vals)]
        if len(unique_vals) < min(x.shape) and x.shape[0] > 1:
            return True

    arr = ensure_matrix(x)
    if not is_square_matrix(arr):
        return True
    # Square matrices that are not symmetric and have integer values might be sequences
    if arr.dtype in (np.int32, np.int64, np.object_):
        return True
    return False


def distance_from_weights(weights: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Convert weight matrix to distance matrix for path algorithms.

    Uses transformation: distance = -log(weight + epsilon)
    This makes high weights correspond to short distances.
    """
    # Add small epsilon to avoid log(0)
    safe_weights = np.maximum(weights, epsilon)
    # Where weights are 0, set distance to infinity
    distances = -np.log(safe_weights)
    distances[weights == 0] = np.inf
    return distances
