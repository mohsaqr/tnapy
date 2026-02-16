"""TNA - Transition Network Analysis for Python.

A Python package providing exact numerical equivalence to the R TNA package
for analyzing sequential data as transition networks.

Example
-------
>>> import tna
>>> import pandas as pd
>>>
>>> # Prepare raw event data
>>> prepared = tna.prepare_data(
...     data=events_df,
...     actor="user_id",
...     time="timestamp",
...     action="event_type"
... )
>>>
>>> # Build a TNA model
>>> model = tna.tna(prepared)
>>>
>>> # Compute centralities
>>> cent = tna.centralities(model)
>>>
>>> # Visualize the network
>>> tna.plot_network(model)
>>> tna.plot_centralities(cent)
"""

from .model import TNA, build_model, tna, ftna, ctna, atna
from .prepare import TNAData, prepare_data, create_seqdata, import_onehot
from .centralities import centralities, betweenness_network, AVAILABLE_MEASURES
from .prune import prune
from .cliques import cliques, CliqueResult
from .communities import communities, CommunityResult
from .group import GroupTNA, group_tna, group_ftna, group_ctna, group_atna
from .utils import (
    row_normalize,
    minmax_scale,
    max_scale,
    rank_scale,
    apply_scaling,
)
from .cluster import cluster_sequences, ClusterResult
from .data import load_group_regulation, load_group_regulation_long
from .colors import color_palette, DEFAULT_COLORS, create_color_map
from .plot import (
    plot_network,
    plot_centralities,
    plot_heatmap,
    plot_comparison,
    plot_compare,
    plot_mosaic,
    plot_sequences,
    plot_frequencies,
    plot_histogram,
    plot_communities,
)
from .bootstrap import (
    BootstrapResult,
    PermutationResult,
    CentralityStabilityResult,
    bootstrap_tna,
    bootstrap_centralities,
    permutation_test,
    permutation_test_edges,
    estimate_cs,
    confidence_interval,
    bca_ci,
    plot_bootstrap,
    plot_permutation,
    plot_network_ci,
    plot_cs,
)

__version__ = "0.1.0"

__all__ = [
    # Main model class
    "TNA",
    # Model building functions
    "build_model",
    "tna",
    "ftna",
    "ctna",
    "atna",
    # Data preparation
    "TNAData",
    "prepare_data",
    "create_seqdata",
    "import_onehot",
    # Clustering
    "cluster_sequences",
    "ClusterResult",
    # Centralities
    "centralities",
    "betweenness_network",
    "AVAILABLE_MEASURES",
    # Pruning
    "prune",
    # Cliques
    "cliques",
    "CliqueResult",
    # Communities
    "communities",
    "CommunityResult",
    # Group models
    "GroupTNA",
    "group_tna",
    "group_ftna",
    "group_ctna",
    "group_atna",
    # Utilities
    "row_normalize",
    "minmax_scale",
    "max_scale",
    "rank_scale",
    "apply_scaling",
    # Data loaders
    "load_group_regulation",
    "load_group_regulation_long",
    # Color utilities
    "color_palette",
    "DEFAULT_COLORS",
    "create_color_map",
    # Plotting functions
    "plot_network",
    "plot_centralities",
    "plot_heatmap",
    "plot_comparison",
    "plot_compare",
    "plot_mosaic",
    "plot_sequences",
    "plot_frequencies",
    "plot_histogram",
    "plot_communities",
    # Statistical inference (bootstrap)
    "BootstrapResult",
    "PermutationResult",
    "CentralityStabilityResult",
    "bootstrap_tna",
    "bootstrap_centralities",
    "permutation_test",
    "permutation_test_edges",
    "estimate_cs",
    "confidence_interval",
    "bca_ci",
    "plot_bootstrap",
    "plot_permutation",
    "plot_network_ci",
    "plot_cs",
]
