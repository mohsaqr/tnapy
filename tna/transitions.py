"""Transition computation algorithms for TNA package."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import ensure_matrix, row_normalize


def compute_transitions(
    data: np.ndarray,
    states: list[str],
    type_: str = "relative",
    params: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute transition matrix and initial probabilities from sequence data.

    Parameters
    ----------
    data : np.ndarray
        Sequence data (rows are sequences, columns are time steps)
    states : list of str
        List of state labels
    type_ : str
        Type of transitions to compute:
        - 'relative': Row-normalized transition probabilities
        - 'frequency': Raw transition counts
        - 'co-occurrence': Bidirectional co-occurrence
        - 'reverse': Reverse order transitions
        - 'n-gram': Higher-order n-gram transitions
        - 'gap': Non-adjacent transitions weighted by gap
        - 'window': Sliding window transitions
        - 'attention': Exponential decay weighted
    params : dict, optional
        Additional parameters for specific types:
        - n_gram: {'n': int} - order of n-gram (default: 2)
        - gap: {'max_gap': int, 'decay': float} - max gap and decay factor
        - window: {'size': int} - window size
        - attention: {'beta': float} - decay parameter

    Returns
    -------
    tuple
        (transition_matrix, initial_probabilities)
    """
    params = params or {}
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    if type_ == "relative":
        weights, inits = _transitions_relative(data, state_to_idx, n_states)
    elif type_ == "frequency":
        weights, inits = _transitions_frequency(data, state_to_idx, n_states)
    elif type_ == "co-occurrence":
        weights, inits = _transitions_cooccurrence(data, state_to_idx, n_states)
    elif type_ == "reverse":
        weights, inits = _transitions_reverse(data, state_to_idx, n_states)
    elif type_ == "n-gram":
        n = params.get('n', 2)
        weights, inits = _transitions_ngram(data, state_to_idx, n_states, n)
    elif type_ == "gap":
        max_gap = params.get('max_gap', 5)
        decay = params.get('decay', 0.5)
        weights, inits = _transitions_gap(data, state_to_idx, n_states, max_gap, decay)
    elif type_ == "window":
        size = params.get('size', 3)
        weights, inits = _transitions_window(data, state_to_idx, n_states, size)
    elif type_ == "attention":
        beta = params.get('beta', 0.1)
        weights, inits = _transitions_attention(data, state_to_idx, n_states, beta)
    else:
        raise ValueError(f"Unknown transition type: {type_}")

    return weights, inits


def _get_valid_transitions(row: np.ndarray) -> list[tuple[int, str]]:
    """Get list of (position, state) for non-NA values in a row."""
    result = []
    for i, val in enumerate(row):
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            result.append((i, str(val)))
    return result


def _transitions_relative(
    data: np.ndarray,
    state_to_idx: dict,
    n_states: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute relative (row-normalized) transition probabilities."""
    counts = np.zeros((n_states, n_states))
    inits = np.zeros(n_states)

    for row in data:
        valid = _get_valid_transitions(row)
        if len(valid) == 0:
            continue

        # Count initial state
        first_state = valid[0][1]
        if first_state in state_to_idx:
            inits[state_to_idx[first_state]] += 1

        # Count transitions
        for i in range(len(valid) - 1):
            from_state = valid[i][1]
            to_state = valid[i + 1][1]
            if from_state in state_to_idx and to_state in state_to_idx:
                counts[state_to_idx[from_state], state_to_idx[to_state]] += 1

    # Normalize
    weights = row_normalize(counts)
    inits = inits / inits.sum() if inits.sum() > 0 else inits

    return weights, inits


def _transitions_frequency(
    data: np.ndarray,
    state_to_idx: dict,
    n_states: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute raw transition counts (frequency model)."""
    counts = np.zeros((n_states, n_states))
    inits = np.zeros(n_states)

    for row in data:
        valid = _get_valid_transitions(row)
        if len(valid) == 0:
            continue

        # Count initial state
        first_state = valid[0][1]
        if first_state in state_to_idx:
            inits[state_to_idx[first_state]] += 1

        # Count transitions
        for i in range(len(valid) - 1):
            from_state = valid[i][1]
            to_state = valid[i + 1][1]
            if from_state in state_to_idx and to_state in state_to_idx:
                counts[state_to_idx[from_state], state_to_idx[to_state]] += 1

    # Normalize inits only
    inits = inits / inits.sum() if inits.sum() > 0 else inits

    return counts, inits


def _transitions_cooccurrence(
    data: np.ndarray,
    state_to_idx: dict,
    n_states: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute bidirectional co-occurrence matrix."""
    counts = np.zeros((n_states, n_states))
    inits = np.zeros(n_states)

    for row in data:
        valid = _get_valid_transitions(row)
        if len(valid) == 0:
            continue

        # Count initial state
        first_state = valid[0][1]
        if first_state in state_to_idx:
            inits[state_to_idx[first_state]] += 1

        # Count co-occurrences (bidirectional adjacent pairs)
        for i in range(len(valid) - 1):
            state1 = valid[i][1]
            state2 = valid[i + 1][1]
            if state1 in state_to_idx and state2 in state_to_idx:
                idx1, idx2 = state_to_idx[state1], state_to_idx[state2]
                counts[idx1, idx2] += 1
                counts[idx2, idx1] += 1

    # Normalize
    weights = row_normalize(counts)
    inits = inits / inits.sum() if inits.sum() > 0 else inits

    return weights, inits


def _transitions_reverse(
    data: np.ndarray,
    state_to_idx: dict,
    n_states: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute transitions in reverse order."""
    counts = np.zeros((n_states, n_states))
    inits = np.zeros(n_states)

    for row in data:
        valid = _get_valid_transitions(row)
        if len(valid) == 0:
            continue

        # Count initial state (last in original sequence)
        last_state = valid[-1][1]
        if last_state in state_to_idx:
            inits[state_to_idx[last_state]] += 1

        # Count reverse transitions
        for i in range(len(valid) - 1, 0, -1):
            from_state = valid[i][1]
            to_state = valid[i - 1][1]
            if from_state in state_to_idx and to_state in state_to_idx:
                counts[state_to_idx[from_state], state_to_idx[to_state]] += 1

    # Normalize
    weights = row_normalize(counts)
    inits = inits / inits.sum() if inits.sum() > 0 else inits

    return weights, inits


def _transitions_ngram(
    data: np.ndarray,
    state_to_idx: dict,
    n_states: int,
    n: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Compute n-gram transitions (skip n-1 steps)."""
    counts = np.zeros((n_states, n_states))
    inits = np.zeros(n_states)

    for row in data:
        valid = _get_valid_transitions(row)
        if len(valid) == 0:
            continue

        # Count initial state
        first_state = valid[0][1]
        if first_state in state_to_idx:
            inits[state_to_idx[first_state]] += 1

        # Count n-gram transitions (pairs separated by n-1 positions)
        for i in range(len(valid) - n + 1):
            from_state = valid[i][1]
            to_state = valid[i + n - 1][1]
            if from_state in state_to_idx and to_state in state_to_idx:
                counts[state_to_idx[from_state], state_to_idx[to_state]] += 1

    # Normalize
    weights = row_normalize(counts)
    inits = inits / inits.sum() if inits.sum() > 0 else inits

    return weights, inits


def _transitions_gap(
    data: np.ndarray,
    state_to_idx: dict,
    n_states: int,
    max_gap: int = 5,
    decay: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Compute transitions with gap-based decay weighting."""
    counts = np.zeros((n_states, n_states))
    inits = np.zeros(n_states)

    for row in data:
        valid = _get_valid_transitions(row)
        if len(valid) == 0:
            continue

        # Count initial state
        first_state = valid[0][1]
        if first_state in state_to_idx:
            inits[state_to_idx[first_state]] += 1

        # Count weighted transitions for all gaps up to max_gap
        for i in range(len(valid)):
            from_state = valid[i][1]
            if from_state not in state_to_idx:
                continue
            for j in range(i + 1, min(i + max_gap + 1, len(valid))):
                to_state = valid[j][1]
                if to_state not in state_to_idx:
                    continue
                gap = j - i
                weight = decay ** (gap - 1)  # gap=1 has weight 1
                counts[state_to_idx[from_state], state_to_idx[to_state]] += weight

    # Normalize
    weights = row_normalize(counts)
    inits = inits / inits.sum() if inits.sum() > 0 else inits

    return weights, inits


def _transitions_window(
    data: np.ndarray,
    state_to_idx: dict,
    n_states: int,
    size: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Compute transitions within sliding windows."""
    counts = np.zeros((n_states, n_states))
    inits = np.zeros(n_states)

    for row in data:
        valid = _get_valid_transitions(row)
        if len(valid) == 0:
            continue

        # Count initial state
        first_state = valid[0][1]
        if first_state in state_to_idx:
            inits[state_to_idx[first_state]] += 1

        # Count all pairs within each window
        for window_start in range(len(valid) - size + 1):
            window = valid[window_start:window_start + size]
            for i in range(len(window)):
                for j in range(i + 1, len(window)):
                    state1 = window[i][1]
                    state2 = window[j][1]
                    if state1 in state_to_idx and state2 in state_to_idx:
                        counts[state_to_idx[state1], state_to_idx[state2]] += 1

    # Normalize
    weights = row_normalize(counts)
    inits = inits / inits.sum() if inits.sum() > 0 else inits

    return weights, inits


def _transitions_attention(
    data: np.ndarray,
    state_to_idx: dict,
    n_states: int,
    beta: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Compute attention-weighted transitions with exponential decay."""
    counts = np.zeros((n_states, n_states))
    inits = np.zeros(n_states)

    for row in data:
        valid = _get_valid_transitions(row)
        if len(valid) == 0:
            continue

        # Count initial state
        first_state = valid[0][1]
        if first_state in state_to_idx:
            inits[state_to_idx[first_state]] += 1

        # Count attention-weighted transitions
        for i in range(len(valid)):
            from_state = valid[i][1]
            if from_state not in state_to_idx:
                continue
            for j in range(i + 1, len(valid)):
                to_state = valid[j][1]
                if to_state not in state_to_idx:
                    continue
                distance = j - i
                weight = np.exp(-beta * distance)
                counts[state_to_idx[from_state], state_to_idx[to_state]] += weight

    # Normalize
    weights = row_normalize(counts)
    inits = inits / inits.sum() if inits.sum() > 0 else inits

    return weights, inits


def _is_na(val) -> bool:
    """Check if a value is NA/None/NaN."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    return False


def compute_transitions_3d(
    data: np.ndarray,
    states: list[str],
    type_: str = "relative",
    params: dict | None = None
) -> np.ndarray:
    """Compute per-sequence transition counts as a 3D array.

    Matches R TNA's compute_transitions function exactly.
    Returns array of shape (n_sequences, n_states, n_states) where
    trans[k, i, j] = number of i->j transitions in sequence k.

    Parameters
    ----------
    data : np.ndarray
        Sequence data (rows are sequences, columns are time steps)
    states : list of str
        List of state labels
    type_ : str
        Transition type ('relative', 'frequency', 'co-occurrence', 'reverse')
    params : dict, optional
        Additional parameters

    Returns
    -------
    np.ndarray
        3D array of shape (n_sequences, n_states, n_states)
    """
    params = params or {}
    n_sequences = data.shape[0]
    n_steps = data.shape[1]
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    trans = np.zeros((n_sequences, n_states, n_states))

    if type_ in ("relative", "frequency"):
        for col in range(n_steps - 1):
            for row in range(n_sequences):
                from_val = data[row, col]
                to_val = data[row, col + 1]
                if _is_na(from_val) or _is_na(to_val):
                    continue
                from_str = str(from_val)
                to_str = str(to_val)
                if from_str in state_to_idx and to_str in state_to_idx:
                    trans[row, state_to_idx[from_str], state_to_idx[to_str]] += 1

    elif type_ == "reverse":
        for col in range(n_steps - 1):
            for row in range(n_sequences):
                from_val = data[row, col + 1]
                to_val = data[row, col]
                if _is_na(from_val) or _is_na(to_val):
                    continue
                from_str = str(from_val)
                to_str = str(to_val)
                if from_str in state_to_idx and to_str in state_to_idx:
                    trans[row, state_to_idx[from_str], state_to_idx[to_str]] += 1

    elif type_ == "co-occurrence":
        for i in range(n_steps - 1):
            for j in range(i + 1, n_steps):
                for row in range(n_sequences):
                    from_val = data[row, i]
                    to_val = data[row, j]
                    if _is_na(from_val) or _is_na(to_val):
                        continue
                    from_str = str(from_val)
                    to_str = str(to_val)
                    if from_str in state_to_idx and to_str in state_to_idx:
                        fi = state_to_idx[from_str]
                        ti = state_to_idx[to_str]
                        trans[row, fi, ti] += 1
                        trans[row, ti, fi] += 1

    return trans


def compute_weights_from_3d(
    transitions: np.ndarray,
    type_: str = "relative",
    scaling: str | list[str] | None = None,
) -> np.ndarray:
    """Compute weight matrix from 3D per-sequence transitions.

    Matches R TNA's compute_weights + scale_weights:
        weights <- apply(transitions, c(2, 3), sum)
        scale_weights(weights, type, scaling, a)

    Parameters
    ----------
    transitions : np.ndarray
        3D array (n_sequences, n_states, n_states)
    type_ : str
        Model type ('relative' for row-normalization)
    scaling : str or list, optional
        Additional scaling to apply

    Returns
    -------
    np.ndarray
        2D weight matrix (n_states, n_states)
    """
    # apply(transitions, c(2, 3), sum) - sum over sequences
    weights = transitions.sum(axis=0)

    # scale_weights: row-normalize for "relative" type
    if type_ == "relative":
        weights = row_normalize(weights)

    # Apply additional scaling if requested
    if scaling:
        from .utils import apply_scaling as _apply_scaling
        weights, _ = _apply_scaling(weights, scaling)

    return weights


def compute_weights_from_matrix(
    mat: np.ndarray,
    type_: str = "relative"
) -> np.ndarray:
    """Process an existing weight/count matrix.

    Parameters
    ----------
    mat : np.ndarray
        Input matrix (counts or weights)
    type_ : str
        How to process:
        - 'relative': Row-normalize
        - 'frequency': Keep as-is

    Returns
    -------
    np.ndarray
        Processed weight matrix
    """
    if type_ == "relative":
        return row_normalize(mat)
    return mat.copy()
