"""Clique detection for TNA models."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model import TNA


@dataclass
class CliqueResult:
    """Result of clique detection.

    Attributes
    ----------
    weights : list of np.ndarray
        Weight sub-matrices for each clique
    indices : list of list of int
        Node indices for each clique
    labels : list of list of str]
        Node labels for each clique
    size : int
        Clique size searched for
    threshold : float
        Minimum edge weight threshold used
    """

    weights: list[np.ndarray]
    indices: list[list[int]]
    labels: list[list[str]]
    size: int
    threshold: float

    def __repr__(self) -> str:
        return (
            f"CliqueResult(n_cliques={len(self.indices)}, "
            f"size={self.size}, threshold={self.threshold})"
        )

    def __str__(self) -> str:
        lines = [
            f"Cliques of size {self.size} (threshold={self.threshold})",
            f"Number of cliques found: {len(self.indices)}",
            "",
        ]
        for i, (idx, lab, w) in enumerate(
            zip(self.indices, self.labels, self.weights)
        ):
            lines.append(f"Clique {i + 1}: {', '.join(lab)}")
            # Show weight sub-matrix
            for r, row_label in enumerate(lab):
                vals = "  ".join(f"{w[r, c]:.3f}" for c in range(len(lab)))
                lines.append(f"  {row_label}: {vals}")
            lines.append("")
        return "\n".join(lines)


def cliques(
    model: 'TNA',
    size: int = 2,
    threshold: float = 0,
    sum_weights: bool = False,
) -> CliqueResult:
    """Find directed cliques in a TNA model.

    A directed clique of size k is a set of k nodes where every pair
    has edges in BOTH directions (i->j AND j->i) above the threshold.

    The algorithm finds cliques by building undirected graphs from the
    upper and lower triangles of the weight matrix, finding cliques in
    each, and taking their intersection.

    Parameters
    ----------
    model : TNA
        The TNA model to analyze
    size : int
        Clique size to search for (default: 2 for dyads)
    threshold : float
        Minimum edge weight for an edge to be considered present
        (default: 0, meaning any positive weight counts)
    sum_weights : bool
        If True, store sum of weights instead of sub-matrix

    Returns
    -------
    CliqueResult
        Object containing detected cliques with their weights and labels
    """
    import networkx as nx

    weights = model.weights.copy()
    n = weights.shape[0]

    # Build upper-triangle undirected graph (edges where w[i,j] > threshold, i < j)
    G_upper = nx.Graph()
    G_upper.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if weights[i, j] > threshold:
                G_upper.add_edge(i, j, weight=weights[i, j])

    # Build lower-triangle undirected graph (edges where w[j,i] > threshold, i < j)
    G_lower = nx.Graph()
    G_lower.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if weights[j, i] > threshold:
                G_lower.add_edge(i, j, weight=weights[j, i])

    # Find cliques in each graph
    cliques_upper = {
        frozenset(c) for c in nx.find_cliques(G_upper) if len(c) >= size
    }
    cliques_lower = {
        frozenset(c) for c in nx.find_cliques(G_lower) if len(c) >= size
    }

    # For cliques of exact requested size, enumerate all sub-cliques of that size
    def _subcliques_of_size(clique_set, k):
        result = set()
        for clique in clique_set:
            if len(clique) == k:
                result.add(clique)
            elif len(clique) > k:
                for combo in combinations(clique, k):
                    result.add(frozenset(combo))
        return result

    subs_upper = _subcliques_of_size(cliques_upper, size)
    subs_lower = _subcliques_of_size(cliques_lower, size)

    # Intersection: cliques present in both directions
    mutual_cliques = sorted(subs_upper & subs_lower, key=lambda c: sorted(c))

    # Build results
    result_weights = []
    result_indices = []
    result_labels = []

    for clique in mutual_cliques:
        idx = sorted(clique)
        labs = [model.labels[i] for i in idx]
        k = len(idx)

        # Extract sub-matrix
        sub = np.zeros((k, k))
        for r, ri in enumerate(idx):
            for c, ci in enumerate(idx):
                sub[r, c] = weights[ri, ci]

        result_weights.append(sub)
        result_indices.append(idx)
        result_labels.append(labs)

    return CliqueResult(
        weights=result_weights,
        indices=result_indices,
        labels=result_labels,
        size=size,
        threshold=threshold,
    )
