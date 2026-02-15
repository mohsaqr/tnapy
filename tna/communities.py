"""Community detection for TNA models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import networkx as nx

if TYPE_CHECKING:
    from .model import TNA


# Mapping from R method names to implementations
AVAILABLE_METHODS = [
    'fast_greedy',
    'louvain',
    'label_prop',
    'leading_eigen',
    'edge_betweenness',
    'walktrap',
]


@dataclass
class CommunityResult:
    """Result of community detection.

    Attributes
    ----------
    counts : dict of str to int
        Number of communities found by each method
    assignments : pd.DataFrame
        DataFrame with states as rows and methods as columns,
        containing community assignment integers
    model : TNA
        The original TNA model
    """

    counts: dict[str, int]
    assignments: pd.DataFrame
    model: 'TNA'

    def __repr__(self) -> str:
        methods = list(self.counts.keys())
        return (
            f"CommunityResult(methods={methods}, "
            f"counts={self.counts})"
        )

    def __str__(self) -> str:
        lines = ["Community Detection Results", ""]
        for method, n in self.counts.items():
            lines.append(f"  {method}: {n} communities")
        lines.append("")
        lines.append("Assignments:")
        lines.append(self.assignments.to_string())
        return "\n".join(lines)


def communities(
    model: 'TNA',
    methods: str | list[str] | None = None,
) -> CommunityResult | dict:
    """Detect communities in a TNA model.

    Applies one or more community detection algorithms to the
    transition network and returns the resulting community assignments.

    Parameters
    ----------
    model : TNA or GroupTNA
        The TNA model to analyze, or GroupTNA for per-group detection.
    methods : str, list of str, or None
        Community detection method(s) to use. If None, uses 'leading_eigen'.
        Available methods:
        - 'fast_greedy': Greedy modularity optimization
        - 'louvain': Louvain community detection
        - 'label_prop': Label propagation
        - 'leading_eigen': Leading eigenvector of modularity matrix
        - 'edge_betweenness': Girvan-Newman edge betweenness
        - 'walktrap': Random walk based (mapped to louvain)

    Returns
    -------
    CommunityResult or dict
        Object containing community assignments and counts.
        For GroupTNA input, returns ``dict[str, CommunityResult]``.
    """
    # Handle GroupTNA input
    from .group import _is_group_tna
    if _is_group_tna(model):
        return {name: communities(m, methods=methods) for name, m in model.items()}

    if methods is None:
        methods = ['leading_eigen']
    elif isinstance(methods, str):
        methods = [methods]

    # Validate methods
    invalid = set(methods) - set(AVAILABLE_METHODS)
    if invalid:
        raise ValueError(
            f"Unknown methods: {invalid}. Available: {AVAILABLE_METHODS}"
        )

    weights = model.weights.copy()
    n = weights.shape[0]
    labels = model.labels

    # Create undirected graph with symmetrized weights for community detection
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = weights[i, j] + weights[j, i]
            if w > 0:
                G.add_edge(i, j, weight=w)

    counts = {}
    assignments = {}

    for method in methods:
        comm = _detect_communities(G, n, method)
        counts[method] = len(set(comm))
        assignments[method] = comm

    # Build assignments DataFrame
    assign_df = pd.DataFrame(assignments, index=labels)

    return CommunityResult(
        counts=counts,
        assignments=assign_df,
        model=model,
    )


def _detect_communities(
    G: nx.Graph,
    n: int,
    method: str,
) -> list[int]:
    """Run a single community detection algorithm.

    Returns list of community assignments (0-indexed) for each node.
    """
    if method == 'fast_greedy':
        return _fast_greedy(G, n)
    elif method == 'louvain':
        return _louvain(G, n)
    elif method == 'label_prop':
        return _label_prop(G, n)
    elif method == 'leading_eigen':
        return _leading_eigen(G, n)
    elif method == 'edge_betweenness':
        return _edge_betweenness(G, n)
    elif method == 'walktrap':
        # Walktrap mapped to louvain as practical substitute
        return _louvain(G, n)
    else:
        raise ValueError(f"Unknown method: {method}")


def _fast_greedy(G: nx.Graph, n: int) -> list[int]:
    """Greedy modularity optimization."""
    comms = nx.community.greedy_modularity_communities(G, weight='weight')
    return _communities_to_labels(comms, n)


def _louvain(G: nx.Graph, n: int) -> list[int]:
    """Louvain community detection."""
    comms = nx.community.louvain_communities(G, weight='weight', seed=42)
    return _communities_to_labels(comms, n)


def _label_prop(G: nx.Graph, n: int) -> list[int]:
    """Label propagation."""
    comms = nx.community.label_propagation_communities(G)
    return _communities_to_labels(comms, n)


def _leading_eigen(G: nx.Graph, n: int) -> list[int]:
    """Leading eigenvector of modularity matrix, split by sign.

    Computes the modularity matrix B = A - k_i*k_j/(2m),
    finds the leading eigenvector, and splits nodes into two
    communities based on the sign of the eigenvector components.
    """
    if n <= 1:
        return [0] * n

    # Get adjacency matrix
    A = nx.to_numpy_array(G, weight='weight')

    # Degree vector and total weight
    k = A.sum(axis=1)
    m2 = k.sum()  # 2m

    if m2 == 0:
        return list(range(n))

    # Modularity matrix: B = A - k_i * k_j / (2m)
    B = A - np.outer(k, k) / m2

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Leading eigenvector (largest eigenvalue)
    leading_idx = np.argmax(eigenvalues)
    leading_vec = eigenvectors[:, leading_idx]

    # Split by sign
    labels = np.where(leading_vec >= 0, 0, 1).tolist()

    return labels


def _edge_betweenness(G: nx.Graph, n: int) -> list[int]:
    """Girvan-Newman edge betweenness, pick best modularity partition."""
    if G.number_of_edges() == 0:
        return list(range(n))

    comp = nx.community.girvan_newman(G)

    best_mod = -1.0
    best_partition = [{i} for i in range(n)]

    # Evaluate a limited number of partitions
    for i, partition in enumerate(comp):
        mod = nx.community.modularity(G, partition, weight='weight')
        if mod > best_mod:
            best_mod = mod
            best_partition = partition
        if i > n:
            break

    return _communities_to_labels(best_partition, n)


def _communities_to_labels(
    communities_iter,
    n: int,
) -> list[int]:
    """Convert an iterable of sets to a list of integer labels."""
    result = [0] * n
    for comm_id, comm in enumerate(sorted(communities_iter, key=min)):
        for node in comm:
            result[node] = comm_id
    return result
