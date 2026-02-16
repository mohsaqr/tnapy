"""Sequence clustering functions for TNA package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform


_SENTINEL = "\x00__NA__"


@dataclass
class ClusterResult:
    """Result of sequence clustering.

    Attributes
    ----------
    data : pd.DataFrame
        Original sequence data.
    k : int
        Number of clusters.
    assignments : np.ndarray
        1-indexed cluster labels for each sequence.
    silhouette : float
        Mean silhouette score.
    sizes : np.ndarray
        Number of sequences in each cluster.
    method : str
        Clustering method used.
    distance : np.ndarray
        n x n distance matrix.
    dissimilarity : str
        Dissimilarity metric used.
    """

    data: pd.DataFrame
    k: int
    assignments: np.ndarray
    silhouette: float
    sizes: np.ndarray
    method: str
    distance: np.ndarray
    dissimilarity: str

    def __repr__(self) -> str:
        return (
            f"ClusterResult(k={self.k}, method={self.method!r}, "
            f"dissimilarity={self.dissimilarity!r}, silhouette={self.silhouette:.4f})"
        )


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------


def _to_token_lists(
    data: pd.DataFrame,
    na_syms: list[str] | None = None,
) -> list[list[str]]:
    """Convert DataFrame rows to lists of string tokens, replacing NA symbols."""
    if na_syms is None:
        na_syms = ["*", "%"]
    na_set = set(na_syms)

    sequences: list[list[str]] = []
    for _, row in data.iterrows():
        tokens: list[str] = []
        for val in row:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                tokens.append(_SENTINEL)
            elif str(val) in na_set:
                tokens.append(_SENTINEL)
            else:
                tokens.append(str(val))
        sequences.append(tokens)
    return sequences


def _hamming_distance(
    a: list[str],
    b: list[str],
    weighted: bool = False,
    lambda_: float = 1.0,
) -> float:
    """Hamming distance between two token sequences.

    Sequences are padded to equal length with sentinel values.
    With ``weighted=True``, mismatches are weighted by ``exp(-lambda_ * i)``.
    """
    max_len = max(len(a), len(b))
    a_padded = a + [_SENTINEL] * (max_len - len(a))
    b_padded = b + [_SENTINEL] * (max_len - len(b))

    dist = 0.0
    for i in range(max_len):
        if a_padded[i] != b_padded[i]:
            if weighted:
                dist += np.exp(-lambda_ * i)
            else:
                dist += 1.0
    return dist


def _levenshtein_distance(a: list[str], b: list[str]) -> float:
    """Levenshtein edit distance (insert, delete, substitute)."""
    m, n = len(a), len(b)
    # O(n) space DP
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost  # substitution
            )
        prev, curr = curr, prev
    return float(prev[n])


def _osa_distance(a: list[str], b: list[str]) -> float:
    """Optimal String Alignment distance (Levenshtein + adjacent transposition)."""
    m, n = len(a), len(b)
    # Need 3 rows for transposition check
    if m == 0:
        return float(n)
    if n == 0:
        return float(m)

    # Full matrix needed for transposition lookback
    d = np.zeros((m + 1, n + 1), dtype=float)
    for i in range(m + 1):
        d[i, 0] = i
    for j in range(n + 1):
        d[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i, j] = min(
                d[i - 1, j] + 1,       # deletion
                d[i, j - 1] + 1,       # insertion
                d[i - 1, j - 1] + cost  # substitution
            )
            if (
                i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                d[i, j] = min(d[i, j], d[i - 2, j - 2] + cost)
    return float(d[m, n])


def _lcs_distance(a: list[str], b: list[str]) -> float:
    """LCS-based distance = max(m, n) - LCS_length.

    Matches the R TNA package definition.
    """
    m, n = len(a), len(b)
    # O(n) space DP for LCS length
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    lcs_len = prev[n]
    return float(max(m, n) - lcs_len)


_DISTANCE_FUNCS = {
    "hamming": _hamming_distance,
    "lv": _levenshtein_distance,
    "osa": _osa_distance,
    "lcs": _lcs_distance,
}


def _compute_distance_matrix(
    sequences: list[list[str]],
    dissimilarity: str = "hamming",
    weighted: bool = False,
    lambda_: float = 1.0,
) -> np.ndarray:
    """Compute pairwise distance matrix."""
    n = len(sequences)
    dist = np.zeros((n, n))

    if dissimilarity == "hamming":
        for i in range(n):
            for j in range(i + 1, n):
                d = _hamming_distance(
                    sequences[i], sequences[j],
                    weighted=weighted, lambda_=lambda_,
                )
                dist[i, j] = d
                dist[j, i] = d
    else:
        func = _DISTANCE_FUNCS[dissimilarity]
        for i in range(n):
            for j in range(i + 1, n):
                d = func(sequences[i], sequences[j])
                dist[i, j] = d
                dist[j, i] = d

    return dist


# ---------------------------------------------------------------------------
# Silhouette
# ---------------------------------------------------------------------------


def _silhouette_score(dist: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette score from a distance matrix and cluster labels.

    ``labels`` are 1-indexed cluster assignments.
    """
    n = len(labels)
    unique_clusters = np.unique(labels)

    if len(unique_clusters) < 2:
        return 0.0

    scores = np.zeros(n)
    for i in range(n):
        cluster_i = labels[i]
        # a(i): mean distance to same-cluster members
        same_mask = labels == cluster_i
        same_mask[i] = False
        n_same = same_mask.sum()
        if n_same == 0:
            scores[i] = 0.0
            continue
        a_i = dist[i, same_mask].sum() / n_same

        # b(i): min over other clusters of mean distance
        b_i = np.inf
        for c in unique_clusters:
            if c == cluster_i:
                continue
            other_mask = labels == c
            b_i = min(b_i, dist[i, other_mask].mean())

        scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

    return float(scores.mean())


# ---------------------------------------------------------------------------
# PAM (Partitioning Around Medoids)
# ---------------------------------------------------------------------------


def _pam(dist: np.ndarray, k: int) -> np.ndarray:
    """PAM clustering: BUILD + SWAP.

    Returns 1-indexed cluster assignments.
    """
    n = dist.shape[0]

    # BUILD phase: greedily select k medoids
    medoids: list[int] = []
    # First medoid: minimizes total distance to all points
    total_dists = dist.sum(axis=1)
    medoids.append(int(np.argmin(total_dists)))

    # Track nearest medoid distance for each point
    nearest_dist = dist[:, medoids[0]].copy()

    for _ in range(1, k):
        # For each candidate, compute gain = sum of max(0, nearest_dist - d(j, candidate))
        best_gain = -np.inf
        best_candidate = -1
        for candidate in range(n):
            if candidate in medoids:
                continue
            gain = np.maximum(0, nearest_dist - dist[:, candidate]).sum()
            if gain > best_gain:
                best_gain = gain
                best_candidate = candidate
        medoids.append(best_candidate)
        nearest_dist = np.minimum(nearest_dist, dist[:, best_candidate])

    medoids_arr = np.array(medoids)

    # SWAP phase
    max_iter = 100
    for _ in range(max_iter):
        improved = False
        for m_idx in range(k):
            current_medoid = medoids_arr[m_idx]
            # Current total cost
            all_medoid_dists = dist[:, medoids_arr]
            current_cost = all_medoid_dists.min(axis=1).sum()

            best_swap_cost = current_cost
            best_swap = -1

            for candidate in range(n):
                if candidate in medoids_arr:
                    continue
                # Try swapping
                trial = medoids_arr.copy()
                trial[m_idx] = candidate
                trial_cost = dist[:, trial].min(axis=1).sum()
                if trial_cost < best_swap_cost:
                    best_swap_cost = trial_cost
                    best_swap = candidate

            if best_swap >= 0:
                medoids_arr[m_idx] = best_swap
                improved = True

        if not improved:
            break

    # Assign each point to its nearest medoid (1-indexed)
    assignments = np.argmin(dist[:, medoids_arr], axis=1) + 1
    return assignments


# ---------------------------------------------------------------------------
# Hierarchical clustering
# ---------------------------------------------------------------------------

# Mapping from R-style method names to scipy linkage methods
_LINKAGE_METHODS = {
    "ward.D": "ward",
    "ward.D2": "ward",
    "complete": "complete",
    "average": "average",
    "single": "single",
    "mcquitty": "weighted",
    "median": "median",
    "centroid": "centroid",
}


def _hierarchical(dist: np.ndarray, k: int, method: str) -> np.ndarray:
    """Hierarchical clustering using scipy.

    Returns 1-indexed cluster assignments.
    """
    scipy_method = _LINKAGE_METHODS[method]

    # Convert to condensed form for linkage
    condensed = squareform(dist)

    # ward.D uses raw distances; ward.D2 uses squared distances
    if method == "ward.D":
        condensed = condensed ** 2

    Z = linkage(condensed, method=scipy_method)
    # Use cut_tree instead of fcluster — fcluster fails when merge
    # heights are tied; cut_tree matches R's cutree behavior.
    labels = cut_tree(Z, n_clusters=k).flatten() + 1  # 1-indexed
    return labels


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def cluster_sequences(
    data: "pd.DataFrame | TNAData",
    k: int,
    dissimilarity: str = "hamming",
    method: str = "pam",
    na_syms: list[str] | None = None,
    weighted: bool = False,
    lambda_: float = 1.0,
) -> ClusterResult:
    """Cluster sequences using distance-based methods.

    Parameters
    ----------
    data : pd.DataFrame or TNAData
        Sequence data in wide format (rows = sequences, columns = time steps)
        or a TNAData object (uses its ``sequence_data``).
    k : int
        Number of clusters.
    dissimilarity : str
        Distance metric: ``'hamming'``, ``'lv'`` (Levenshtein), ``'osa'``
        (Optimal String Alignment), or ``'lcs'`` (Longest Common Subsequence).
    method : str
        Clustering method: ``'pam'`` or a hierarchical linkage method
        (``'ward.D'``, ``'ward.D2'``, ``'complete'``, ``'average'``,
        ``'single'``, ``'mcquitty'``, ``'median'``, ``'centroid'``).
    na_syms : list of str, optional
        Symbols to treat as missing values (default: ``['*', '%']``).
    weighted : bool
        If ``True`` and ``dissimilarity='hamming'``, apply exponential
        decay weighting to positions.
    lambda_ : float
        Decay parameter for weighted Hamming (higher = faster decay).

    Returns
    -------
    ClusterResult
        Clustering result with assignments, silhouette score, etc.
    """
    # Import here to avoid circular imports
    from .prepare import TNAData

    if isinstance(data, TNAData):
        df = data.sequence_data
    else:
        df = data

    if dissimilarity not in _DISTANCE_FUNCS:
        raise ValueError(
            f"Unknown dissimilarity {dissimilarity!r}. "
            f"Choose from: {list(_DISTANCE_FUNCS.keys())}"
        )

    if method != "pam" and method not in _LINKAGE_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. "
            f"Choose 'pam' or one of: {list(_LINKAGE_METHODS.keys())}"
        )

    if k < 2:
        raise ValueError("k must be >= 2")

    if k > len(df):
        raise ValueError(f"k={k} exceeds the number of sequences ({len(df)})")

    # Convert to token lists
    sequences = _to_token_lists(df, na_syms=na_syms)

    # Compute distance matrix
    dist = _compute_distance_matrix(
        sequences,
        dissimilarity=dissimilarity,
        weighted=weighted,
        lambda_=lambda_,
    )

    # Cluster
    if method == "pam":
        assignments = _pam(dist, k)
    else:
        assignments = _hierarchical(dist, k, method)

    # Silhouette
    sil = _silhouette_score(dist, assignments)

    # Cluster sizes
    sizes = np.array([np.sum(assignments == c) for c in range(1, k + 1)])

    return ClusterResult(
        data=df,
        k=k,
        assignments=assignments,
        silhouette=sil,
        sizes=sizes,
        method=method,
        distance=dist,
        dissimilarity=dissimilarity,
    )
