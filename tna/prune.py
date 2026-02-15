"""Pruning functions for TNA models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model import TNA


def prune(model: 'TNA', threshold: float = 0.1) -> 'TNA':
    """Prune edges below a weight threshold.

    Sets all edges with weight below the threshold to zero and returns
    a new TNA model with the pruned weight matrix.

    Parameters
    ----------
    model : TNA or GroupTNA
        The TNA model to prune, or GroupTNA for per-group pruning.
    threshold : float
        Minimum edge weight to keep (default: 0.1).
        Edges with weight < threshold are set to 0.

    Returns
    -------
    TNA or GroupTNA
        New TNA model with pruned weights.
        For GroupTNA input, returns a GroupTNA.
    """
    # Handle GroupTNA input
    from .group import _is_group_tna, GroupTNA
    if _is_group_tna(model):
        return GroupTNA(
            models={name: prune(m, threshold=threshold) for name, m in model.items()}
        )

    from .model import TNA as TNAClass

    weights = model.weights.copy()
    weights[weights < threshold] = 0.0

    return TNAClass(
        weights=weights,
        inits=model.inits.copy(),
        labels=list(model.labels),
        data=model.data,
        type_=model.type_,
        scaling=list(model.scaling),
    )
