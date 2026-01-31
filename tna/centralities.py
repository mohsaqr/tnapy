"""Centrality measures for TNA package.

Provides exact numerical equivalence to R TNA package centrality measures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import networkx as nx
from scipy import linalg

if TYPE_CHECKING:
    from .model import TNA


# Available centrality measures
AVAILABLE_MEASURES = [
    'OutStrength',
    'InStrength',
    'ClosenessIn',
    'ClosenessOut',
    'Closeness',
    'Betweenness',
    'BetweennessRSP',
    'Diffusion',
    'Clustering'
]


def centralities(
    model: 'TNA',
    loops: bool = False,
    normalize: bool = False,
    measures: list[str] | None = None
) -> pd.DataFrame:
    """Compute centrality measures for a TNA model.

    Parameters
    ----------
    model : TNA
        A TNA model object
    loops : bool
        If True, include self-loops in calculations
    normalize : bool
        If True, normalize measures to [0, 1] using min-max normalization
    measures : list of str, optional
        Which measures to compute. If None, computes all available measures.
        Available: OutStrength, InStrength, ClosenessIn, ClosenessOut,
        Closeness, Betweenness, BetweennessRSP, Diffusion, Clustering

    Returns
    -------
    pd.DataFrame
        DataFrame with states as rows and centrality measures as columns
    """
    if measures is None:
        measures = AVAILABLE_MEASURES.copy()

    # Validate measures
    invalid = set(measures) - set(AVAILABLE_MEASURES)
    if invalid:
        raise ValueError(f"Unknown measures: {invalid}. Available: {AVAILABLE_MEASURES}")

    weights = model.weights.copy()
    n = weights.shape[0]

    # Remove self-loops if requested (R: diag(x) <- ifelse_(loops, diag(x), 0))
    if not loops:
        np.fill_diagonal(weights, 0)

    results = {}

    # Create igraph-compatible graph for NetworkX
    # R uses igraph, we use networkx but match behavior
    G = _create_graph(weights)

    # Compute requested measures in order they appear in AVAILABLE_MEASURES
    for measure in AVAILABLE_MEASURES:
        if measure not in measures:
            continue
        if measure == 'OutStrength':
            results['OutStrength'] = _out_strength(weights)
        elif measure == 'InStrength':
            results['InStrength'] = _in_strength(weights)
        elif measure == 'ClosenessIn':
            results['ClosenessIn'] = _closeness_in(G, n)
        elif measure == 'ClosenessOut':
            results['ClosenessOut'] = _closeness_out(G, n)
        elif measure == 'Closeness':
            results['Closeness'] = _closeness_all(G, n)
        elif measure == 'Betweenness':
            results['Betweenness'] = _betweenness(G, n)
        elif measure == 'BetweennessRSP':
            results['BetweennessRSP'] = _betweenness_rsp(weights)
        elif measure == 'Diffusion':
            results['Diffusion'] = _diffusion(weights)
        elif measure == 'Clustering':
            results['Clustering'] = _clustering(weights)

    # Create DataFrame
    df = pd.DataFrame(results, index=model.labels)

    # Normalize if requested (R: ranger function - min-max normalization)
    if normalize:
        for col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[col] = 0.0

    return df


def _create_graph(weights: np.ndarray) -> nx.DiGraph:
    """Create NetworkX DiGraph from weight matrix."""
    n = weights.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if weights[i, j] > 0:
                G.add_edge(i, j, weight=weights[i, j])
    return G


def _out_strength(weights: np.ndarray) -> np.ndarray:
    """Compute out-strength (sum of outgoing edge weights).

    R equivalent: igraph::strength(g, mode = "out")
    """
    return weights.sum(axis=1)


def _in_strength(weights: np.ndarray) -> np.ndarray:
    """Compute in-strength (sum of incoming edge weights).

    R equivalent: igraph::strength(g, mode = "in")
    """
    return weights.sum(axis=0)


def _closeness_in(G: nx.DiGraph, n: int) -> np.ndarray:
    """Compute incoming closeness centrality.

    R equivalent: igraph::closeness(g, mode = "in")
    igraph uses: (n-1) / sum(distances_to_node)
    """
    result = np.zeros(n)
    # Reverse graph to compute paths TO each node
    rev_G = G.reverse()

    for i in range(n):
        try:
            lengths = nx.single_source_dijkstra_path_length(
                rev_G, i, weight=lambda u, v, d: 1.0 / d['weight']
            )
            # Sum of distances from all reachable nodes
            total_dist = sum(d for j, d in lengths.items() if j != i)
            n_reachable = len([j for j in lengths if j != i])
            if total_dist > 0:
                result[i] = n_reachable / total_dist
        except (nx.NetworkXError, nx.NodeNotFound):
            result[i] = 0.0

    return result


def _closeness_out(G: nx.DiGraph, n: int) -> np.ndarray:
    """Compute outgoing closeness centrality.

    R equivalent: igraph::closeness(g, mode = "out")
    """
    result = np.zeros(n)

    for i in range(n):
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, i, weight=lambda u, v, d: 1.0 / d['weight']
            )
            total_dist = sum(d for j, d in lengths.items() if j != i)
            n_reachable = len([j for j in lengths if j != i])
            if total_dist > 0:
                result[i] = n_reachable / total_dist
        except (nx.NetworkXError, nx.NodeNotFound):
            result[i] = 0.0

    return result


def _closeness_all(G: nx.DiGraph, n: int) -> np.ndarray:
    """Compute overall closeness centrality (mode = "all").

    R equivalent: igraph::closeness(g, mode = "all")
    Treats graph as undirected for distance calculation.
    """
    result = np.zeros(n)

    # Create undirected version
    U = G.to_undirected()

    for i in range(n):
        try:
            lengths = nx.single_source_dijkstra_path_length(
                U, i, weight=lambda u, v, d: 1.0 / d['weight']
            )
            total_dist = sum(d for j, d in lengths.items() if j != i)
            n_reachable = len([j for j in lengths if j != i])
            if total_dist > 0:
                result[i] = n_reachable / total_dist
        except (nx.NetworkXError, nx.NodeNotFound):
            result[i] = 0.0

    return result


def _betweenness(G: nx.DiGraph, n: int) -> np.ndarray:
    """Compute betweenness centrality.

    R equivalent: igraph::betweenness(g)
    Uses weighted shortest paths.
    """
    # igraph uses 1/weight as distance by default for weighted graphs
    bc = nx.betweenness_centrality(
        G,
        weight=lambda u, v, d: 1.0 / d['weight'] if d['weight'] > 0 else float('inf'),
        normalized=False
    )
    return np.array([bc.get(i, 0.0) for i in range(n)])


def _betweenness_rsp(weights: np.ndarray, beta: float = 0.01) -> np.ndarray:
    """Compute Randomized Shortest Path betweenness centrality.

    Exact port of R tna::rsp_bet function (Kivimäki et al. 2016).

    R code:
        n <- ncol(mat)
        D <- .rowSums(mat, m = n, n = n)
        if (any(D == 0)) return(NA)
        P_ref <- diag(D^-1) %*% mat
        C <- mat^-1
        C[is.infinite(C)] <- 0
        W <- P_ref * exp(-beta * C)
        Z <- solve(diag(1, n, n) - W)
        Z_recip <- Z^-1
        Z_recip[is.infinite(Z_recip)] <- 0
        Z_recip_diag <- diag(Z_recip) * diag(1, n, n)
        out <- diag(tcrossprod(Z, Z_recip - n * Z_recip_diag) %*% Z)
        out <- round(out)
        out <- out - min(out) + 1
    """
    n = weights.shape[0]
    mat = weights.copy()

    # D <- .rowSums(mat, m = n, n = n)
    D = mat.sum(axis=1)

    # if (any(D == 0)) return(NA)
    if np.any(D == 0):
        return np.full(n, np.nan)

    # P_ref <- diag(D^-1) %*% mat
    P_ref = np.diag(1.0 / D) @ mat

    # C <- mat^-1; C[is.infinite(C)] <- 0
    with np.errstate(divide='ignore'):
        C = 1.0 / mat
    C[np.isinf(C)] = 0

    # W <- P_ref * exp(-beta * C)
    W = P_ref * np.exp(-beta * C)

    # Z <- solve(diag(1, n, n) - W)
    try:
        Z = linalg.inv(np.eye(n) - W)
    except linalg.LinAlgError:
        return np.full(n, np.nan)

    # Z_recip <- Z^-1; Z_recip[is.infinite(Z_recip)] <- 0
    with np.errstate(divide='ignore'):
        Z_recip = 1.0 / Z
    Z_recip[np.isinf(Z_recip)] = 0

    # Z_recip_diag <- diag(Z_recip) * diag(1, n, n)
    Z_recip_diag = np.diag(np.diag(Z_recip))

    # out <- diag(tcrossprod(Z, Z_recip - n * Z_recip_diag) %*% Z)
    # tcrossprod(A, B) = A %*% t(B)
    out = np.diag(Z @ (Z_recip - n * Z_recip_diag).T @ Z)

    # out <- round(out)
    out = np.round(out)

    # out <- out - min(out) + 1
    out = out - out.min() + 1

    return out


def _diffusion(weights: np.ndarray) -> np.ndarray:
    """Compute diffusion centrality.

    Exact port of R tna::diffusion function (Banerjee et al. 2014).

    R code:
        s <- 0
        n <- ncol(mat)
        p <- diag(1, n, n)
        for (i in seq_len(n)) {
            p <- p %*% mat
            s <- s + p
        }
        .rowSums(s, n, n)
    """
    n = weights.shape[0]
    mat = weights.copy()

    s = np.zeros((n, n))
    p = np.eye(n)

    for _ in range(n):
        p = p @ mat
        s = s + p

    # .rowSums(s, n, n) - row sums
    return s.sum(axis=1)


def _clustering(weights: np.ndarray) -> np.ndarray:
    """Compute signed clustering coefficient.

    Exact port of R tna::wcc function (Zhang and Horvath 2005).
    Called on symmetric matrix: wcc(x + t(x))

    R code:
        wcc <- function(mat) {
            diag(mat) <- 0
            n <- ncol(mat)
            num <- diag(mat %*% mat %*% mat)
            den <- .colSums(mat, n, n)^2 - .colSums(mat^2, n, n)
            num / den
        }
        # Called as: wcc(x + t(x))
    """
    # Symmetrize: x + t(x)
    mat = weights + weights.T

    # diag(mat) <- 0
    np.fill_diagonal(mat, 0)

    n = mat.shape[0]

    # num <- diag(mat %*% mat %*% mat)
    num = np.diag(mat @ mat @ mat)

    # den <- .colSums(mat, n, n)^2 - .colSums(mat^2, n, n)
    col_sums = mat.sum(axis=0)
    col_sums_sq = (mat ** 2).sum(axis=0)
    den = col_sums ** 2 - col_sums_sq

    # num / den (handle division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = num / den
    result[~np.isfinite(result)] = 0.0

    return result
