"""TNA model class and build functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .prepare import TNAData, create_seqdata
from .transitions import compute_transitions, compute_weights_from_matrix
from .utils import (
    apply_scaling,
    ensure_matrix,
    get_labels,
    is_sequence_data,
    is_square_matrix,
)


@dataclass
class TNA:
    """Transition Network Analysis model.

    Attributes
    ----------
    weights : np.ndarray
        Adjacency/transition matrix (n_states x n_states)
    inits : np.ndarray
        Initial state probabilities (n_states,)
    labels : list of str
        State labels
    data : np.ndarray or None
        Original sequence data (if built from sequences)
    type_ : str
        Model type ('relative', 'frequency', etc.)
    scaling : list of str
        Scaling methods applied to the weights
    """

    weights: np.ndarray
    inits: np.ndarray
    labels: list[str]
    data: np.ndarray | None = None
    type_: str = "relative"
    scaling: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        n = len(self.labels)
        return f"TNA(states={n}, type={self.type_!r}, scaling={self.scaling})"

    def __str__(self) -> str:
        lines = [
            f"TNA Model",
            f"  Type: {self.type_}",
            f"  States: {self.labels}",
            f"  Scaling: {self.scaling if self.scaling else 'none'}",
            "",
            "Transition Matrix:",
        ]

        # Format matrix with labels
        df = pd.DataFrame(self.weights, index=self.labels, columns=self.labels)
        lines.append(df.to_string())

        lines.append("")
        lines.append("Initial Probabilities:")
        init_df = pd.DataFrame({'prob': self.inits}, index=self.labels)
        lines.append(init_df.to_string())

        return '\n'.join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert weight matrix to labeled DataFrame."""
        return pd.DataFrame(self.weights, index=self.labels, columns=self.labels)

    def summary(self) -> dict:
        """Return summary statistics of the model."""
        return {
            'n_states': len(self.labels),
            'type': self.type_,
            'scaling': self.scaling,
            'n_edges': np.sum(self.weights > 0),
            'density': np.sum(self.weights > 0) / (len(self.labels) ** 2),
            'mean_weight': np.mean(self.weights[self.weights > 0]) if np.any(self.weights > 0) else 0,
            'max_weight': np.max(self.weights),
            'has_self_loops': np.any(np.diag(self.weights) > 0),
        }


def build_model(
    x: pd.DataFrame | np.ndarray | TNAData,
    type_: str = "relative",
    scaling: str | list[str] | None = None,
    cols: list[str] | None = None,
    labels: list[str] | None = None,
    begin_state: str | None = None,
    end_state: str | None = None,
    params: dict | None = None
) -> TNA:
    """Build a TNA model from data.

    Parameters
    ----------
    x : pd.DataFrame, np.ndarray, or TNAData
        Input data. Can be:
        - Wide-format DataFrame (rows=sequences, cols=time steps)
        - TNAData object from prepare_data()
        - numpy array of sequences
        - Square weight matrix (if type_='matrix')
    type_ : str
        Model type:
        - 'relative': Row-normalized transition probabilities
        - 'frequency': Raw transition counts
        - 'co-occurrence': Bidirectional co-occurrence
        - 'reverse': Reverse order transitions
        - 'n-gram': Higher-order n-gram transitions
        - 'gap': Non-adjacent transitions weighted by gap
        - 'window': Sliding window transitions
        - 'attention': Exponential decay weighted
    scaling : str or list of str, optional
        Scaling to apply: 'minmax', 'max', 'rank', or None
    cols : list of str, optional
        Column names to use (for DataFrame input)
    labels : list of str, optional
        State labels (auto-detected if not provided)
    begin_state : str, optional
        Add this state at the beginning of each sequence
    end_state : str, optional
        Add this state at the end of each sequence
    params : dict, optional
        Additional parameters for specific model types

    Returns
    -------
    TNA
        The built TNA model
    """
    # Handle TNAData input
    if isinstance(x, TNAData):
        x = x.sequence_data

    # Convert to array and get labels
    if isinstance(x, pd.DataFrame):
        mat = ensure_matrix(x)
        state_labels = labels
    else:
        mat = np.asarray(x)
        state_labels = labels

    # Check if input is a weight matrix or sequence data
    if is_square_matrix(mat) and not is_sequence_data(x):
        # Direct weight matrix input
        weights = compute_weights_from_matrix(mat, type_)
        n = weights.shape[0]
        if state_labels is None:
            state_labels = get_labels(x, labels)
        inits = np.ones(n) / n  # Uniform initial probabilities
        data = None
    else:
        # Sequence data input
        seq_data, detected_labels, _ = create_seqdata(
            x, cols=cols, begin_state=begin_state, end_state=end_state
        )

        if state_labels is None:
            state_labels = detected_labels

        weights, inits = compute_transitions(
            seq_data, state_labels, type_=type_, params=params
        )
        data = seq_data

    # Apply scaling
    weights, applied_scaling = apply_scaling(weights, scaling)

    return TNA(
        weights=weights,
        inits=inits,
        labels=state_labels,
        data=data,
        type_=type_,
        scaling=applied_scaling
    )


def tna(
    x: pd.DataFrame | np.ndarray | TNAData,
    scaling: str | list[str] | None = None,
    cols: list[str] | None = None,
    labels: list[str] | None = None,
    begin_state: str | None = None,
    end_state: str | None = None
) -> TNA:
    """Build a relative transition probability model.

    This is the standard TNA model with row-normalized transition probabilities.

    Parameters
    ----------
    x : pd.DataFrame, np.ndarray, or TNAData
        Input data (sequences or weight matrix)
    scaling : str or list of str, optional
        Scaling to apply: 'minmax', 'max', 'rank', or None
    cols : list of str, optional
        Column names to use (for DataFrame input)
    labels : list of str, optional
        State labels
    begin_state : str, optional
        Add this state at the beginning of each sequence
    end_state : str, optional
        Add this state at the end of each sequence

    Returns
    -------
    TNA
        The built TNA model

    Examples
    --------
    >>> import pandas as pd
    >>> import tna
    >>> df = pd.DataFrame({
    ...     'step1': ['A', 'B', 'A'],
    ...     'step2': ['B', 'C', 'C'],
    ...     'step3': ['C', 'A', 'B']
    ... })
    >>> model = tna.tna(df)
    >>> model.weights
    """
    return build_model(
        x, type_="relative", scaling=scaling, cols=cols,
        labels=labels, begin_state=begin_state, end_state=end_state
    )


def ftna(
    x: pd.DataFrame | np.ndarray | TNAData,
    scaling: str | list[str] | None = None,
    cols: list[str] | None = None,
    labels: list[str] | None = None,
    begin_state: str | None = None,
    end_state: str | None = None
) -> TNA:
    """Build a frequency-based transition model.

    Returns raw transition counts without normalization.

    Parameters
    ----------
    x : pd.DataFrame, np.ndarray, or TNAData
        Input data (sequences or weight matrix)
    scaling : str or list of str, optional
        Scaling to apply: 'minmax', 'max', 'rank', or None
    cols : list of str, optional
        Column names to use (for DataFrame input)
    labels : list of str, optional
        State labels
    begin_state : str, optional
        Add this state at the beginning of each sequence
    end_state : str, optional
        Add this state at the end of each sequence

    Returns
    -------
    TNA
        The built TNA model
    """
    return build_model(
        x, type_="frequency", scaling=scaling, cols=cols,
        labels=labels, begin_state=begin_state, end_state=end_state
    )


def ctna(
    x: pd.DataFrame | np.ndarray | TNAData,
    scaling: str | list[str] | None = None,
    cols: list[str] | None = None,
    labels: list[str] | None = None,
    begin_state: str | None = None,
    end_state: str | None = None
) -> TNA:
    """Build a co-occurrence transition model.

    Counts bidirectional adjacent co-occurrences.

    Parameters
    ----------
    x : pd.DataFrame, np.ndarray, or TNAData
        Input data (sequences or weight matrix)
    scaling : str or list of str, optional
        Scaling to apply: 'minmax', 'max', 'rank', or None
    cols : list of str, optional
        Column names to use (for DataFrame input)
    labels : list of str, optional
        State labels
    begin_state : str, optional
        Add this state at the beginning of each sequence
    end_state : str, optional
        Add this state at the end of each sequence

    Returns
    -------
    TNA
        The built TNA model
    """
    return build_model(
        x, type_="co-occurrence", scaling=scaling, cols=cols,
        labels=labels, begin_state=begin_state, end_state=end_state
    )


def atna(
    x: pd.DataFrame | np.ndarray | TNAData,
    beta: float = 0.1,
    scaling: str | list[str] | None = None,
    cols: list[str] | None = None,
    labels: list[str] | None = None,
    begin_state: str | None = None,
    end_state: str | None = None
) -> TNA:
    """Build an attention-weighted transition model.

    Uses exponential decay weighting based on distance.

    Parameters
    ----------
    x : pd.DataFrame, np.ndarray, or TNAData
        Input data (sequences or weight matrix)
    beta : float
        Decay parameter (higher = faster decay with distance)
    scaling : str or list of str, optional
        Scaling to apply: 'minmax', 'max', 'rank', or None
    cols : list of str, optional
        Column names to use (for DataFrame input)
    labels : list of str, optional
        State labels
    begin_state : str, optional
        Add this state at the beginning of each sequence
    end_state : str, optional
        Add this state at the end of each sequence

    Returns
    -------
    TNA
        The built TNA model
    """
    return build_model(
        x, type_="attention", scaling=scaling, cols=cols,
        labels=labels, begin_state=begin_state, end_state=end_state,
        params={'beta': beta}
    )
