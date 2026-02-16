"""Compare sequential patterns across groups."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _combine_data(group_tna: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Combine sequence data from all groups.

    Returns
    -------
    data : np.ndarray
        Combined sequence matrix (n_total × n_cols), dtype object.
    group : np.ndarray
        Group label for each row.
    group_names : list of str
        Ordered group names.
    """
    arrays = []
    groups = []
    group_names = list(group_tna.keys())

    for name in group_names:
        model = group_tna[name]
        if model.data is None:
            raise ValueError(
                f"Group '{name}' has no sequence data. "
                "compare_sequences requires models built from sequences."
            )
        d = np.asarray(model.data, dtype=object)
        arrays.append(d)
        groups.append(np.full(d.shape[0], name, dtype=object))

    # Pad to same number of columns
    max_cols = max(a.shape[1] for a in arrays)
    padded = []
    for a in arrays:
        if a.shape[1] < max_cols:
            pad = np.full((a.shape[0], max_cols - a.shape[1]), None, dtype=object)
            a = np.hstack([a, pad])
        padded.append(a)

    data = np.vstack(padded)
    group = np.concatenate(groups)
    return data, group, group_names


def _extract_patterns(
    data: np.ndarray, lengths: list[int]
) -> list[np.ndarray]:
    """Extract subsequence patterns of given lengths.

    Parameters
    ----------
    data : np.ndarray
        Sequence matrix (n_rows × n_cols), dtype object.
    lengths : list of int
        Subsequence lengths to extract.

    Returns
    -------
    list of np.ndarray
        One pattern matrix per length. Each matrix has shape
        (n_rows, n_positions) with string pattern values or None.
    """
    n_rows, n_cols = data.shape
    pattern_matrices = []

    for length in lengths:
        if length > n_cols:
            break
        n_positions = n_cols - length + 1
        patterns = np.full((n_rows, n_positions), None, dtype=object)

        for i in range(n_rows):
            for j in range(n_positions):
                subseq = data[i, j : j + length]
                # Skip if any element is None/NaN
                if any(s is None or (isinstance(s, float) and np.isnan(s)) for s in subseq):
                    continue
                patterns[i, j] = "->".join(str(s) for s in subseq)

        pattern_matrices.append(patterns)

    return pattern_matrices


def _factorize_patterns(
    pattern_matrices: list[np.ndarray],
    group: np.ndarray,
    group_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Count pattern occurrences per group.

    Returns
    -------
    freq : np.ndarray
        Frequency matrix (n_patterns × n_groups).
    pattern_labels : list of str
        Pattern names corresponding to rows.
    """
    all_patterns: list[str] = []
    pattern_to_idx: dict[str, int] = {}

    # Collect all unique patterns across all lengths
    for pm in pattern_matrices:
        for val in pm.flat:
            if val is not None and val not in pattern_to_idx:
                pattern_to_idx[val] = len(all_patterns)
                all_patterns.append(val)

    n_patterns = len(all_patterns)
    n_groups = len(group_names)
    group_to_idx = {g: i for i, g in enumerate(group_names)}

    freq = np.zeros((n_patterns, n_groups), dtype=int)

    for pm in pattern_matrices:
        n_rows = pm.shape[0]
        for i in range(n_rows):
            g_idx = group_to_idx[group[i]]
            for val in pm[i]:
                if val is not None:
                    freq[pattern_to_idx[val], g_idx] += 1

    return freq, all_patterns


def _pattern_statistic(freq: np.ndarray) -> np.ndarray:
    """Compute chi-squared-like statistic per pattern.

    statistic = sqrt(sum_j((observed_ij - expected_ij)^2))

    where expected_ij = row_total_i * col_total_j / grand_total
    """
    row_sums = freq.sum(axis=1, keepdims=True).astype(float)
    col_sums = freq.sum(axis=0, keepdims=True).astype(float)
    total = freq.sum().astype(float)

    if total == 0:
        return np.zeros(freq.shape[0])

    expected = row_sums * col_sums / total
    diff = freq.astype(float) - expected
    return np.sqrt((diff ** 2).sum(axis=1))


def _p_adjust(p: np.ndarray, method: str = "bonferroni") -> np.ndarray:
    """Adjust p-values for multiple comparisons.

    Supports 'bonferroni', 'holm', 'fdr'/'BH', and 'none'.
    """
    n = len(p)
    if n == 0:
        return p.copy()

    if method == "none":
        return p.copy()
    elif method == "bonferroni":
        return np.minimum(p * n, 1.0)
    elif method == "holm":
        order = np.argsort(p)
        sorted_p = p[order]
        adjusted = np.empty(n)
        cummax = 0.0
        for i in range(n):
            val = sorted_p[i] * (n - i)
            cummax = max(cummax, val)
            adjusted[order[i]] = min(cummax, 1.0)
        return adjusted
    elif method in ("fdr", "BH"):
        order = np.argsort(p)[::-1]
        sorted_p = p[order]
        adjusted = np.empty(n)
        cummin = 1.0
        for i in range(n):
            val = sorted_p[i] * n / (n - i)
            cummin = min(cummin, val)
            adjusted[order[i]] = min(cummin, 1.0)
        return adjusted
    else:
        raise ValueError(f"Unknown adjustment method: {method!r}")


def _precompute_row_patterns(
    data: np.ndarray,
    lengths: list[int],
    pattern_to_idx: dict[str, int],
) -> list[list[int]]:
    """Pre-compute which patterns each row contributes.

    Returns a list of lists: row_patterns[i] contains the pattern
    indices observed in row i (with repeats for multiple occurrences).
    """
    n_rows, n_cols = data.shape
    row_patterns: list[list[int]] = [[] for _ in range(n_rows)]

    for length in lengths:
        if length > n_cols:
            break
        n_positions = n_cols - length + 1
        for i in range(n_rows):
            for j in range(n_positions):
                subseq = data[i, j : j + length]
                if any(
                    s is None or (isinstance(s, float) and np.isnan(s))
                    for s in subseq
                ):
                    continue
                pat = "->".join(str(s) for s in subseq)
                if pat in pattern_to_idx:
                    row_patterns[i].append(pattern_to_idx[pat])

    return row_patterns


def _permutation_test_patterns(
    data: np.ndarray,
    group: np.ndarray,
    group_names: list[str],
    lengths: list[int],
    freq: np.ndarray,
    pattern_labels: list[str],
    iter_: int,
    adjust: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Permutation test for pattern frequency differences.

    Returns
    -------
    effect_size : np.ndarray
        Standardised effect size per pattern.
    p_value : np.ndarray
        Adjusted p-values.
    """
    true_stat = _pattern_statistic(freq)
    n_patterns = len(pattern_labels)
    n_groups = len(group_names)
    perm_stats = np.zeros((iter_, n_patterns))

    pattern_to_idx = {p: i for i, p in enumerate(pattern_labels)}
    group_to_idx = {g: i for i, g in enumerate(group_names)}

    # Pre-compute row-level group indices
    group_indices = np.array([group_to_idx[g] for g in group])

    # Pre-compute which patterns each row contributes
    row_patterns = _precompute_row_patterns(data, lengths, pattern_to_idx)

    for it in range(iter_):
        perm_group_idx = rng.permutation(group_indices)

        # Recount patterns under permuted labels
        perm_freq = np.zeros((n_patterns, n_groups), dtype=int)
        for i, pat_list in enumerate(row_patterns):
            g_idx = perm_group_idx[i]
            for pat_idx in pat_list:
                perm_freq[pat_idx, g_idx] += 1

        perm_stats[it] = _pattern_statistic(perm_freq)

    # Effect size: (true - mean) / sd
    mean_perm = perm_stats.mean(axis=0)
    sd_perm = perm_stats.std(axis=0, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        effect_size = np.where(sd_perm > 0, (true_stat - mean_perm) / sd_perm, 0.0)

    # P-value per pattern: done per subsequence length group
    # R groups patterns by their subsequence length for separate adjustment
    raw_p = np.ones(n_patterns)
    for pat_idx in range(n_patterns):
        count_ge = np.sum(perm_stats[:, pat_idx] >= true_stat[pat_idx])
        raw_p[pat_idx] = (count_ge + 1) / (iter_ + 1)

    # Adjust per subsequence length group (matching R behavior)
    pattern_length = np.array([p.count("->") + 1 for p in pattern_labels])
    p_value = np.ones(n_patterns)
    for length in lengths:
        mask = pattern_length == length
        if mask.any():
            p_value[mask] = _p_adjust(raw_p[mask], method=adjust)

    return effect_size, p_value


def compare_sequences(
    x: Any,
    sub: list[int] | range | None = None,
    min_freq: int = 5,
    test: bool = False,
    iter_: int = 1000,
    adjust: str = "bonferroni",
    seed: int | None = None,
) -> pd.DataFrame:
    """Compare subsequence patterns across groups.

    Extracts n-gram subsequence patterns from a grouped TNA model and
    compares their frequencies across groups. Optionally performs a
    permutation test for statistical significance.

    Parameters
    ----------
    x : GroupTNA
        A grouped TNA model (from ``group_tna`` or similar).
    sub : list of int or range, optional
        Subsequence lengths to examine. Default is ``range(1, 6)``
        (unigrams through 5-grams), capped at the number of columns.
    min_freq : int
        Minimum frequency a pattern must have in *every* group to be
        included. Default 5.
    test : bool
        If ``True``, run a permutation test. Default ``False``.
    iter_ : int
        Number of permutation iterations (only used when ``test=True``).
    adjust : str
        P-value adjustment method: ``'bonferroni'``, ``'holm'``,
        ``'fdr'``/``'BH'``, or ``'none'``.
    seed : int, optional
        Random seed for reproducibility of permutation test.

    Returns
    -------
    pd.DataFrame
        One row per pattern with columns:

        - ``pattern`` — the subsequence (e.g. ``"adapt->cohesion"``)
        - ``freq_<group>`` — frequency in each group
        - ``prop_<group>`` — proportion in each group
        - ``effect_size`` — standardised effect size (if ``test=True``)
        - ``p_value`` — adjusted p-value (if ``test=True``)

    Examples
    --------
    >>> import tna
    >>> prep = tna.prepare_data(long_df, actor="Actor", action="Action", time="Time")
    >>> gm = tna.group_tna(prep, group="Achiever")
    >>> res = tna.compare_sequences(gm)
    >>> res.head()
    """
    # Validate input
    if not (hasattr(x, "models") and hasattr(x, "items") and callable(getattr(x, "items", None))):
        raise TypeError(
            "compare_sequences requires a GroupTNA object. "
            "Use group_tna() or similar to create one."
        )

    if len(x) < 2:
        raise ValueError("compare_sequences requires at least 2 groups.")

    # Combine data from all groups
    data, group, group_names = _combine_data(x)
    n_rows, n_cols = data.shape

    # Default subsequence lengths
    if sub is None:
        sub = list(range(1, min(6, n_cols + 1)))
    else:
        sub = list(sub)
        sub = [s for s in sub if s <= n_cols]

    if not sub:
        raise ValueError("No valid subsequence lengths for the data.")

    # Extract patterns and count per group
    pattern_matrices = _extract_patterns(data, sub)
    freq, pattern_labels = _factorize_patterns(
        pattern_matrices, group, group_names
    )

    # Compute proportions per subsequence length group BEFORE filtering
    # (R uses all patterns as denominator, not just filtered ones)
    pattern_length_all = np.array([p.count("->") + 1 for p in pattern_labels])
    props_all = np.zeros_like(freq, dtype=float)
    for length in sub:
        mask = pattern_length_all == length
        if mask.any():
            length_totals = freq[mask].sum(axis=0).astype(float)
            for j in range(len(group_names)):
                if length_totals[j] > 0:
                    props_all[mask, j] = freq[mask, j] / length_totals[j]

    # Run permutation test on ALL patterns BEFORE min_freq filtering
    # (R adjusts p-values using the full pattern count per length)
    effect_size_all = None
    p_value_all = None
    if test:
        rng = np.random.default_rng(seed)
        effect_size_all, p_value_all = _permutation_test_patterns(
            data, group, group_names, sub, freq, pattern_labels,
            iter_, adjust, rng
        )

    # Filter by min_freq (minimum across ALL groups)
    min_per_group = freq.min(axis=1)
    keep = min_per_group >= min_freq
    freq = freq[keep]
    props = props_all[keep]
    pattern_labels = [p for p, k in zip(pattern_labels, keep) if k]
    if test:
        effect_size_all = effect_size_all[keep]
        p_value_all = p_value_all[keep]

    if len(pattern_labels) == 0:
        cols = ["pattern"]
        for g in group_names:
            cols.extend([f"freq_{g}", f"prop_{g}"])
        if test:
            cols.extend(["effect_size", "p_value"])
        return pd.DataFrame(columns=cols)

    # Sort by pattern length then alphabetically (matching R output order)
    pattern_length = np.array([p.count("->") + 1 for p in pattern_labels])
    sort_order = np.lexsort(
        (np.array(pattern_labels), pattern_length)
    )
    freq = freq[sort_order]
    props = props[sort_order]
    pattern_labels = [pattern_labels[i] for i in sort_order]
    if test:
        effect_size_all = effect_size_all[sort_order]
        p_value_all = p_value_all[sort_order]

    # Build output DataFrame
    out = {"pattern": pattern_labels}
    for j, g in enumerate(group_names):
        out[f"freq_{g}"] = freq[:, j]
    for j, g in enumerate(group_names):
        out[f"prop_{g}"] = props[:, j]
    if test:
        out["effect_size"] = effect_size_all
        out["p_value"] = p_value_all

    result = pd.DataFrame(out)

    # Sort by p_value if test was run
    if test:
        result = result.sort_values("p_value", kind="stable").reset_index(drop=True)

    return result
