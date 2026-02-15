"""Statistical inference for TNA: bootstrap and permutation tests.

Provides bootstrap resampling and permutation testing functionality
matching the R TNA package algorithms exactly.

R TNA bootstrap uses per-sequence transition resampling with a stability
method. R TNA permutation test uses edge-wise comparisons with effect sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .model import TNA, build_model
from .prepare import TNAData, create_seqdata
from .centralities import centralities as compute_centralities, AVAILABLE_MEASURES
from .transitions import compute_transitions_3d, compute_weights_from_3d


# -----------------------------------------------------------------------------
# Result Classes
# -----------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """Result of bootstrap analysis for a TNA model.

    Matches R TNA's tna_bootstrap class structure exactly.

    Attributes
    ----------
    weights_orig : np.ndarray
        Original weight matrix
    weights_sig : np.ndarray
        Significant weights: (p < level) * weights
    weights_mean : np.ndarray
        Mean of bootstrap weight matrices
    weights_sd : np.ndarray
        SD of bootstrap weight matrices
    p_values : np.ndarray
        P-value matrix
    cr_lower : np.ndarray
        Consistency range lower bound (weights * consistency_range[0])
    cr_upper : np.ndarray
        Consistency range upper bound (weights * consistency_range[1])
    ci_lower : np.ndarray
        CI lower bound (quantile at level/2)
    ci_upper : np.ndarray
        CI upper bound (quantile at 1 - level/2)
    boot_summary : pd.DataFrame
        Summary DataFrame of non-zero edges with stats
    model : TNA
        TNA model with pruned weights (only significant edges)
    labels : list[str]
        State labels
    method : str
        Bootstrap method ('stability' or 'threshold')
    iter : int
        Number of bootstrap iterations
    level : float
        Significance level
    """

    weights_orig: np.ndarray
    weights_sig: np.ndarray
    weights_mean: np.ndarray
    weights_sd: np.ndarray
    p_values: np.ndarray
    cr_lower: np.ndarray
    cr_upper: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    boot_summary: pd.DataFrame
    model: TNA
    labels: list[str]
    method: str = "stability"
    iter: int = 1000
    level: float = 0.05

    # --- Backward compatibility properties ---

    @property
    def estimate(self) -> TNA:
        """Original model (backward compatibility)."""
        return TNA(
            weights=self.weights_orig,
            inits=self.model.inits,
            labels=list(self.labels),
            data=self.model.data,
            type_=self.model.type_,
            scaling=self.model.scaling,
        )

    @property
    def replicates(self) -> list:
        """Not stored in R-compatible mode."""
        return []

    @property
    def weights_ci(self) -> tuple[np.ndarray, np.ndarray]:
        """CI bounds as tuple (backward compatibility)."""
        return (self.ci_lower, self.ci_upper)

    @property
    def inits_ci(self) -> tuple[np.ndarray, np.ndarray]:
        """Not available in R-compatible mode."""
        n = len(self.labels)
        return (np.zeros(n), np.ones(n))

    @property
    def n_boot(self) -> int:
        """Number of bootstrap iterations (backward compatibility)."""
        return self.iter

    @property
    def ci_level(self) -> float:
        """Confidence interval level (backward compatibility)."""
        return 1 - self.level

    def summary(self) -> pd.DataFrame:
        """Return the bootstrap summary DataFrame.

        Returns
        -------
        pd.DataFrame
            Summary with edges, weights, p-values, CIs, and significance
        """
        return self.boot_summary

    def significant_edges(self, threshold: float = 0) -> list[tuple[str, str, float]]:
        """Find edges with significant weights (p < level).

        Returns
        -------
        list of tuples
            List of (from_state, to_state, weight) for significant edges
        """
        significant = []
        n = len(self.labels)
        for i in range(n):
            for j in range(n):
                if self.weights_sig[i, j] != 0:
                    significant.append((
                        self.labels[i],
                        self.labels[j],
                        self.weights_orig[i, j]
                    ))
        return significant


@dataclass
class PermutationResult:
    """Result of permutation test for comparing TNA models.

    Matches R TNA's tna_permutation class structure exactly.

    Attributes
    ----------
    edges : dict
        Edge-wise results with keys:
        - 'stats': DataFrame with edge_name, diff_true, effect_size, p_value
        - 'diffs_true': np.ndarray of true differences
        - 'diffs_sig': np.ndarray of significant differences
    centralities : dict or None
        Centrality-wise results (if measures were specified) with keys:
        - 'stats': DataFrame with state, centrality, diff_true, effect_size, p_value
        - 'diffs_true': pd.DataFrame
        - 'diffs_sig': pd.DataFrame
    labels : list[str]
        State labels
    """

    edges: dict
    centralities: dict | None = None
    labels: list[str] = field(default_factory=list)

    # --- Backward compatibility properties ---

    @property
    def observed(self) -> float:
        """Frobenius norm of edge differences (backward compatibility)."""
        return float(np.linalg.norm(self.edges['diffs_true'], 'fro'))

    @property
    def null_distribution(self) -> np.ndarray:
        """Not stored in R-compatible mode."""
        return np.array([])

    @property
    def p_value(self) -> float:
        """Minimum edge p-value (backward compatibility)."""
        return float(self.edges['stats']['p_value'].min())

    @property
    def n_perm(self) -> int:
        """Number of edges tested (backward compatibility)."""
        return len(self.edges['stats'])

    @property
    def alternative(self) -> str:
        """Alternative hypothesis (backward compatibility)."""
        return 'two-sided'

    def is_significant(self, alpha: float = 0.05) -> bool | np.ndarray:
        """Check if any edge is significant at given alpha level."""
        return bool(np.any(self.edges['stats']['p_value'].values < alpha))

    def plot(self, ax: Axes | None = None, **kwargs) -> Axes:
        """Plot the permutation results."""
        return plot_permutation(self, ax=ax, **kwargs)


# -----------------------------------------------------------------------------
# Confidence Interval Methods
# -----------------------------------------------------------------------------


def confidence_interval(
    bootstrap_values: np.ndarray,
    ci: float = 0.95,
    method: str = "percentile"
) -> tuple[float, float]:
    """Calculate confidence interval from bootstrap distribution.

    Parameters
    ----------
    bootstrap_values : np.ndarray
        Array of bootstrap estimates
    ci : float
        Confidence level (default: 0.95)
    method : str
        CI method: 'percentile', 'basic', or 'bca'

    Returns
    -------
    tuple
        (lower, upper) confidence interval bounds
    """
    alpha = 1 - ci

    if method == "percentile":
        lower = np.percentile(bootstrap_values, 100 * alpha / 2)
        upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
        return (lower, upper)

    elif method == "basic":
        # Basic bootstrap: 2*theta_hat - percentile
        theta_hat = np.mean(bootstrap_values)
        lower_pct = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
        upper_pct = np.percentile(bootstrap_values, 100 * alpha / 2)
        return (2 * theta_hat - lower_pct, 2 * theta_hat - upper_pct)

    else:
        raise ValueError(f"Unknown CI method: {method}. Use 'percentile' or 'basic'.")


def bca_ci(
    data: np.ndarray,
    bootstrap_values: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    ci: float = 0.95
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) confidence interval.

    More accurate than percentile method for skewed distributions.

    Parameters
    ----------
    data : np.ndarray
        Original data
    bootstrap_values : np.ndarray
        Array of bootstrap estimates
    statistic_func : callable
        Function that computes the statistic from data
    ci : float
        Confidence level (default: 0.95)

    Returns
    -------
    tuple
        (lower, upper) confidence interval bounds
    """
    from scipy import stats

    alpha = 1 - ci
    n = len(data)
    theta_hat = statistic_func(data)

    # Bias correction factor (z0)
    prop_below = np.mean(bootstrap_values < theta_hat)
    # Avoid extreme values
    prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop_below)

    # Acceleration factor (a) using jackknife
    theta_jack = np.zeros(n)
    for i in range(n):
        jack_sample = np.delete(data, i)
        theta_jack[i] = statistic_func(jack_sample)

    theta_jack_mean = np.mean(theta_jack)
    num = np.sum((theta_jack_mean - theta_jack) ** 3)
    den = 6 * (np.sum((theta_jack_mean - theta_jack) ** 2) ** 1.5)

    if den == 0:
        a = 0
    else:
        a = num / den

    # Adjusted percentiles
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

    # BCa adjustment
    def adjusted_percentile(z_alpha):
        num = z0 + z_alpha
        den = 1 - a * num
        if den == 0:
            return 0.5
        return stats.norm.cdf(z0 + num / den)

    pct_lower = adjusted_percentile(z_alpha_lower) * 100
    pct_upper = adjusted_percentile(z_alpha_upper) * 100

    # Clip to valid percentile range
    pct_lower = np.clip(pct_lower, 0, 100)
    pct_upper = np.clip(pct_upper, 0, 100)

    lower = np.percentile(bootstrap_values, pct_lower)
    upper = np.percentile(bootstrap_values, pct_upper)

    return (lower, upper)


# -----------------------------------------------------------------------------
# Bootstrap Functions
# -----------------------------------------------------------------------------


def bootstrap_tna(
    x: TNA | pd.DataFrame | TNAData,
    iter: int = 1000,
    level: float = 0.05,
    method: str = "stability",
    threshold: float | None = None,
    consistency_range: tuple[float, float] = (0.75, 1.25),
    seed: int | None = None,
    type_: str = "relative",
    scaling: str | list[str] | None = None,
) -> BootstrapResult:
    """Bootstrap resampling for TNA model stability testing.

    Matches R TNA's bootstrap.tna function exactly. Uses per-sequence
    transition resampling (not raw sequence resampling) with a stability
    method that tests whether bootstrap weights stay within a consistency
    range of the original weights.

    R equivalent: bootstrap(model, iter, level, method, threshold, consistency_range)

    Parameters
    ----------
    x : TNA, pd.DataFrame, or TNAData
        A TNA model object (preferred, matches R), or sequence data
        from which a model will be built
    iter : int
        Number of bootstrap iterations (default: 1000)
    level : float
        Significance level (default: 0.05). CI is at level/2 and 1-level/2.
    method : str
        Bootstrap method: 'stability' or 'threshold'
    threshold : float, optional
        For threshold method. Default: 10th percentile of weights.
    consistency_range : tuple
        For stability method: (lower, upper) multipliers (default: (0.75, 1.25))
    seed : int, optional
        Random seed for reproducibility
    type_ : str
        Model type when building from data (default: 'relative')
    scaling : str or list, optional
        Scaling when building from data

    Returns
    -------
    BootstrapResult
        Object matching R's tna_bootstrap structure

    Examples
    --------
    >>> import tna
    >>> df = tna.load_group_regulation()
    >>> model = tna.tna(df)
    >>> boot = tna.bootstrap_tna(model, iter=500, seed=42)
    >>> print(boot.summary())
    """
    # Handle input: TNA model or raw data
    if isinstance(x, TNA):
        model = x
        if model.data is None:
            raise ValueError("TNA model must have sequence data for bootstrap")
        seq_data = model.data
        labels = model.labels
        model_type = model.type_
        model_scaling = model.scaling if model.scaling else None
    else:
        # Build model from data
        if isinstance(x, TNAData):
            df = x.sequence_data
        else:
            df = x
        model = build_model(df, type_=type_, scaling=scaling)
        seq_data = model.data
        labels = model.labels
        model_type = type_
        model_scaling = scaling

    if seq_data is None:
        raise ValueError("Cannot bootstrap: no sequence data available")

    n = seq_data.shape[0]  # number of sequences
    a = len(labels)         # number of states

    # Set random seed
    rng = np.random.default_rng(seed)

    # Compute per-sequence 3D transitions (R: compute_transitions)
    trans = compute_transitions_3d(seq_data, labels, type_=model_type)

    # Compute original weights (R: compute_weights)
    weights = compute_weights_from_3d(trans, type_=model_type, scaling=model_scaling)

    # Default threshold (R: quantile(x$weights, probs = 0.1))
    if threshold is None:
        threshold = float(np.quantile(weights, 0.1))

    # Bootstrap loop
    weights_boot = np.zeros((iter, a, a))
    p_values = np.zeros((a, a))
    idx = np.arange(n)

    if method == "stability":
        for i in range(iter):
            # Resample per-sequence transitions (R: trans[sample(idx, n, replace=TRUE), , ])
            boot_idx = rng.choice(idx, size=n, replace=True)
            trans_boot = trans[boot_idx]
            weights_boot[i] = compute_weights_from_3d(
                trans_boot, type_=model_type, scaling=model_scaling
            )
            # Count exceedances outside consistency range
            # R: p_values + 1L * (wb <= w * cr[1]) + 1L * (wb >= w * cr[2])
            p_values += (
                (weights_boot[i] <= weights * consistency_range[0]).astype(int) +
                (weights_boot[i] >= weights * consistency_range[1]).astype(int)
            )
    elif method == "threshold":
        for i in range(iter):
            boot_idx = rng.choice(idx, size=n, replace=True)
            trans_boot = trans[boot_idx]
            weights_boot[i] = compute_weights_from_3d(
                trans_boot, type_=model_type, scaling=model_scaling
            )
            # R: p_values + 1L * (weights_boot[i,,] < threshold)
            p_values += (weights_boot[i] < threshold).astype(int)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'stability' or 'threshold'.")

    # P-values: (count + 1) / (iter + 1)  (R formula)
    p_values = (p_values + 1) / (iter + 1)

    # Bootstrap statistics
    weights_mean = np.nanmean(weights_boot, axis=0)
    weights_sd = np.nanstd(weights_boot, axis=0, ddof=1)  # R's sd uses ddof=1

    # Confidence intervals (quantiles at level/2 and 1-level/2)
    ci_lower = np.quantile(weights_boot, level / 2, axis=0)
    ci_upper = np.quantile(weights_boot, 1 - level / 2, axis=0)

    # Significant weights: (p < level) * weights
    weights_sig = (p_values < level) * weights

    # Consistency range bounds
    cr_lower = weights * consistency_range[0]
    cr_upper = weights * consistency_range[1]

    # Build summary DataFrame (R: only non-zero weight edges)
    weights_vec = weights.flatten(order='F')  # column-major to match R
    p_vec = p_values.flatten(order='F')
    sig_vec = (p_values < level).flatten(order='F')
    cr_lo_vec = cr_lower.flatten(order='F')
    cr_up_vec = cr_upper.flatten(order='F')
    ci_lo_vec = ci_lower.flatten(order='F')
    ci_up_vec = ci_upper.flatten(order='F')

    # Edge names matching R: from = rep(alphabet, times = a), to = rep(alphabet, each = a)
    from_labels = labels * a
    to_labels = [l for l in labels for _ in range(a)]

    combined = pd.DataFrame({
        'from': from_labels,
        'to': to_labels,
        'weight': weights_vec,
        'p_value': p_vec,
        'sig': sig_vec,
        'cr_lower': cr_lo_vec,
        'cr_upper': cr_up_vec,
        'ci_lower': ci_lo_vec,
        'ci_upper': ci_up_vec,
    })
    # Filter to non-zero weights (R: combined[weights_vec > 0, ])
    combined = combined[combined['weight'] > 0].reset_index(drop=True)

    # Build pruned model (R: model$weights <- weights_sig)
    pruned_model = TNA(
        weights=weights_sig,
        inits=model.inits.copy(),
        labels=list(labels),
        data=seq_data,
        type_=model_type,
        scaling=model.scaling if hasattr(model, 'scaling') else [],
    )

    return BootstrapResult(
        weights_orig=weights,
        weights_sig=weights_sig,
        weights_mean=weights_mean,
        weights_sd=weights_sd,
        p_values=p_values,
        cr_lower=cr_lower,
        cr_upper=cr_upper,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        boot_summary=combined,
        model=pruned_model,
        labels=list(labels),
        method=method,
        iter=iter,
        level=level,
    )


def bootstrap_centralities(
    x: TNA | pd.DataFrame | TNAData,
    measures: list[str] | None = None,
    iter: int = 1000,
    level: float = 0.05,
    seed: int | None = None,
    type_: str = "relative",
    scaling: str | list[str] | None = None,
    loops: bool = False,
    normalize: bool = False
) -> pd.DataFrame:
    """Bootstrap confidence intervals for centrality measures.

    Uses per-sequence transition resampling (matching R algorithm).

    Parameters
    ----------
    x : TNA, pd.DataFrame, or TNAData
        TNA model or sequence data
    measures : list of str, optional
        Centrality measures to compute. If None, computes all.
    iter : int
        Number of bootstrap iterations (default: 1000)
    level : float
        Significance level (default: 0.05)
    seed : int, optional
        Random seed for reproducibility
    type_ : str
        Model type when building from data (default: 'relative')
    scaling : str or list, optional
        Scaling when building from data
    loops : bool
        Whether to include self-loops in centrality calculations
    normalize : bool
        Whether to normalize centrality measures

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: measure, state, estimate, ci_lower, ci_upper, se
    """
    if measures is None:
        measures = AVAILABLE_MEASURES.copy()

    # Handle input
    if isinstance(x, TNA):
        model = x
        seq_data = model.data
        labels = model.labels
        model_type = model.type_
        model_scaling = model.scaling if model.scaling else None
    else:
        if isinstance(x, TNAData):
            df = x.sequence_data
        else:
            df = x
        model = build_model(df, type_=type_, scaling=scaling)
        seq_data = model.data
        labels = model.labels
        model_type = type_
        model_scaling = scaling

    if seq_data is None:
        raise ValueError("Cannot bootstrap: no sequence data available")

    n_sequences = seq_data.shape[0]
    n_states = len(labels)
    rng = np.random.default_rng(seed)

    # Compute per-sequence 3D transitions
    trans = compute_transitions_3d(seq_data, labels, type_=model_type)

    # Original centralities
    estimate_cent = compute_centralities(
        model, measures=measures, loops=loops, normalize=normalize
    )

    # Bootstrap
    n_measures = len(measures)
    cent_samples = np.zeros((iter, n_states, n_measures))

    for b in range(iter):
        boot_idx = rng.choice(n_sequences, size=n_sequences, replace=True)
        trans_boot = trans[boot_idx]
        boot_weights = compute_weights_from_3d(
            trans_boot, type_=model_type, scaling=model_scaling
        )
        # Create temporary model for centrality computation
        boot_model = TNA(
            weights=boot_weights,
            inits=model.inits.copy(),
            labels=list(labels),
            type_=model_type,
        )
        boot_cent = compute_centralities(
            boot_model, measures=measures, loops=loops, normalize=normalize
        )
        for m_idx, measure in enumerate(measures):
            cent_samples[b, :, m_idx] = boot_cent[measure].values

    # Compute CIs and standard errors
    rows = []
    for m_idx, measure in enumerate(measures):
        for s_idx, state in enumerate(labels):
            values = cent_samples[:, s_idx, m_idx]
            rows.append({
                'measure': measure,
                'state': state,
                'estimate': estimate_cent.loc[state, measure],
                'ci_lower': np.quantile(values, level / 2),
                'ci_upper': np.quantile(values, 1 - level / 2),
                'se': np.std(values, ddof=1)
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Permutation Tests
# -----------------------------------------------------------------------------


def permutation_test(
    x: TNA,
    y: TNA,
    iter: int = 1000,
    adjust: str = "none",
    paired: bool = False,
    level: float = 0.05,
    measures: list[str] | None = None,
    seed: int | None = None,
    **kwargs
) -> PermutationResult:
    """Permutation test for comparing two TNA models.

    Matches R TNA's permutation_test function exactly. Tests edge-wise
    differences between two groups using per-sequence transition permutation.
    Optionally tests centrality differences.

    R equivalent: permutation_test(x, y, adjust, iter, paired, level, measures)

    Parameters
    ----------
    x : TNA
        First group's TNA model (must have sequence data)
    y : TNA
        Second group's TNA model (must have sequence data)
    iter : int
        Number of permutation iterations (default: 1000)
    adjust : str
        P-value adjustment method: 'none', 'bonferroni', 'fdr'/'BH', 'holm'
    paired : bool
        Whether to use paired permutation test
    level : float
        Significance level (default: 0.05)
    measures : list of str, optional
        Centrality measures to test. If None or empty, only edges are tested.
    seed : int, optional
        Random seed for reproducibility
    **kwargs
        Additional arguments passed to centralities()

    Returns
    -------
    PermutationResult
        Object matching R's tna_permutation structure

    Examples
    --------
    >>> import tna
    >>> df = tna.load_group_regulation()
    >>> model1 = tna.tna(df.iloc[:1000])
    >>> model2 = tna.tna(df.iloc[1000:])
    >>> result = tna.permutation_test(model1, model2, iter=500, seed=42)
    >>> print(result.edges['stats'])
    """
    if measures is None:
        measures = []

    # Handle backward compat: accept DataFrames as first two args
    if isinstance(x, (pd.DataFrame, TNAData)):
        if isinstance(x, TNAData):
            x = build_model(x.sequence_data, type_="relative")
        else:
            x = build_model(x, type_="relative")
    if isinstance(y, (pd.DataFrame, TNAData)):
        if isinstance(y, TNAData):
            y = build_model(y.sequence_data, type_="relative")
        else:
            y = build_model(y, type_="relative")

    if x.data is None or y.data is None:
        raise ValueError("Both TNA models must have sequence data for permutation test")

    data_x = x.data
    data_y = y.data
    n_x = data_x.shape[0]
    n_y = data_y.shape[0]

    if paired and n_x != n_y:
        raise ValueError("For paired test, both groups must have the same number of sequences")

    labels = x.labels
    a = len(labels)

    # Verify same labels
    if len(labels) != len(y.labels) or not all(a == b for a, b in zip(labels, y.labels)):
        raise ValueError("Both models must have the same state labels in the same order")

    model_type = x.type_
    model_scaling = x.scaling if x.scaling else None

    # Combine data
    combined_data = np.vstack([data_x, data_y])
    n_xy = n_x + n_y

    weights_x = x.weights
    weights_y = y.weights

    rng = np.random.default_rng(seed)

    n_measures = len(measures)
    include_centralities = n_measures > 0

    # Compute centrality differences if requested
    if include_centralities:
        cent_x = compute_centralities(x, measures=measures, **kwargs)
        cent_y = compute_centralities(y, measures=measures, **kwargs)
        cent_diffs_true = cent_x[measures].values - cent_y[measures].values
        cent_diffs_true_abs = np.abs(cent_diffs_true)

    # True edge differences
    edge_diffs_true = weights_x - weights_y
    edge_diffs_true_abs = np.abs(edge_diffs_true)

    # Edge names (R format: from -> to)
    edge_names = []
    for j in range(a):
        for i in range(a):
            edge_names.append(f"{labels[i]} -> {labels[j]}")

    # Compute per-sequence transitions for combined data
    combined_trans = compute_transitions_3d(combined_data, labels, type_=model_type)

    idx_x = np.arange(n_x)
    idx_y = np.arange(n_x, n_xy)

    # Permutation loop
    edge_diffs_perm = np.zeros((iter, a, a))
    cent_diffs_perm = np.zeros((iter, a, n_measures)) if include_centralities else None
    edge_p_values = np.zeros((a, a), dtype=int)
    cent_p_values = np.zeros((a, n_measures), dtype=int) if include_centralities else None

    for i in range(iter):
        if paired:
            # Paired permutation: for each pair, randomly swap
            pair_idx = np.arange(n_xy).reshape(-1, 2)
            perm_pairs = np.array([rng.permutation(pair) for pair in pair_idx])
            perm_idx = perm_pairs.flatten()
        else:
            perm_idx = rng.permutation(n_xy)

        trans_perm_x = combined_trans[perm_idx[idx_x]]
        trans_perm_y = combined_trans[perm_idx[idx_y]]

        weights_perm_x = compute_weights_from_3d(
            trans_perm_x, type_=model_type, scaling=model_scaling
        )
        weights_perm_y = compute_weights_from_3d(
            trans_perm_y, type_=model_type, scaling=model_scaling
        )

        if include_centralities:
            model_perm_x = TNA(
                weights=weights_perm_x, inits=x.inits.copy(),
                labels=list(labels), type_=model_type,
            )
            model_perm_y = TNA(
                weights=weights_perm_y, inits=y.inits.copy(),
                labels=list(labels), type_=model_type,
            )
            cent_perm_x = compute_centralities(model_perm_x, measures=measures, **kwargs)
            cent_perm_y = compute_centralities(model_perm_y, measures=measures, **kwargs)
            cent_diffs_perm[i] = cent_perm_x[measures].values - cent_perm_y[measures].values
            cent_p_values += (np.abs(cent_diffs_perm[i]) >= cent_diffs_true_abs).astype(int)

        edge_diffs_perm[i] = weights_perm_x - weights_perm_y
        edge_p_values += (np.abs(edge_diffs_perm[i]) >= edge_diffs_true_abs).astype(int)

    # P-values: (count + 1) / (iter + 1)
    edge_p_values_float = (edge_p_values + 1) / (iter + 1)

    # Apply p-value adjustment (R: p.adjust)
    edge_p_adjusted = _p_adjust(edge_p_values_float.flatten(), method=adjust)
    edge_p_values_float = edge_p_adjusted.reshape((a, a))

    # Effect sizes: diff_true / sd(perm_diffs)
    edge_diffs_sd = np.std(edge_diffs_perm, axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        edge_effect_size = edge_diffs_true / edge_diffs_sd
    edge_effect_size[~np.isfinite(edge_effect_size)] = np.nan

    # Significant differences
    edge_diffs_sig = edge_diffs_true * (edge_p_values_float < level)

    # Build edge stats DataFrame (column-major order to match R)
    edge_stats = pd.DataFrame({
        'edge_name': edge_names,
        'diff_true': edge_diffs_true.flatten(order='F'),
        'effect_size': edge_effect_size.flatten(order='F'),
        'p_value': edge_p_values_float.flatten(order='F'),
    })

    out_edges = {
        'stats': edge_stats,
        'diffs_true': edge_diffs_true,
        'diffs_sig': edge_diffs_sig,
    }

    out_centralities = None
    if include_centralities:
        cent_p_values_float = (cent_p_values + 1) / (iter + 1)
        cent_p_adjusted = _p_adjust(cent_p_values_float.flatten(), method=adjust)
        cent_p_values_float = cent_p_adjusted.reshape((a, n_measures))

        cent_diffs_sd = np.std(cent_diffs_perm, axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            cent_effect_size = cent_diffs_true / cent_diffs_sd
        cent_effect_size[~np.isfinite(cent_effect_size)] = np.nan

        cent_diffs_sig = cent_diffs_true * (cent_p_values_float < level)

        # Build centrality stats DataFrame
        cent_stats_rows = []
        for m_idx, measure in enumerate(measures):
            for s_idx in range(a):
                cent_stats_rows.append({
                    'state': labels[s_idx],
                    'centrality': measure,
                    'diff_true': cent_diffs_true[s_idx, m_idx],
                    'effect_size': cent_effect_size[s_idx, m_idx],
                    'p_value': cent_p_values_float[s_idx, m_idx],
                })
        cent_stats = pd.DataFrame(cent_stats_rows)

        # DataFrames for diffs
        cent_diffs_true_df = pd.DataFrame(
            cent_diffs_true, index=labels, columns=measures
        )
        cent_diffs_true_df.insert(0, 'state', labels)

        cent_diffs_sig_df = pd.DataFrame(
            cent_diffs_sig, index=labels, columns=measures
        )
        cent_diffs_sig_df.insert(0, 'state', labels)

        out_centralities = {
            'stats': cent_stats,
            'diffs_true': cent_diffs_true_df,
            'diffs_sig': cent_diffs_sig_df,
        }

    return PermutationResult(
        edges=out_edges,
        centralities=out_centralities,
        labels=list(labels),
    )


def permutation_test_edges(
    data1: pd.DataFrame | TNAData,
    data2: pd.DataFrame | TNAData,
    n_perm: int = 1000,
    correction: str = "none",
    seed: int | None = None,
    type_: str = "relative"
) -> pd.DataFrame:
    """Test each edge for significant difference between groups.

    Convenience wrapper around permutation_test() that accepts raw data
    and returns a simple DataFrame.

    Parameters
    ----------
    data1 : pd.DataFrame or TNAData
        First group's sequence data
    data2 : pd.DataFrame or TNAData
        Second group's sequence data
    n_perm : int
        Number of permutations (default: 1000)
    correction : str
        Multiple testing correction: 'bonferroni', 'fdr', 'none'
    seed : int, optional
        Random seed for reproducibility
    type_ : str
        Model type for build_model (default: 'relative')

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: from, to, diff, p_value, p_adjusted, significant
    """
    # Build models
    if isinstance(data1, TNAData):
        df1 = data1.sequence_data
    else:
        df1 = data1
    if isinstance(data2, TNAData):
        df2 = data2.sequence_data
    else:
        df2 = data2

    # Get common labels
    model1_temp = build_model(df1, type_=type_)
    model2_temp = build_model(df2, type_=type_)
    all_labels = sorted(set(model1_temp.labels) | set(model2_temp.labels))

    # Build models with common labels
    model1 = build_model(df1, type_=type_, labels=all_labels)
    model2 = build_model(df2, type_=type_, labels=all_labels)

    # Map correction name to p.adjust name
    adjust_map = {'fdr': 'fdr', 'bonferroni': 'bonferroni', 'none': 'none'}
    adjust = adjust_map.get(correction, correction)

    # Run permutation test
    result = permutation_test(
        model1, model2, iter=n_perm, adjust=adjust, seed=seed
    )

    # Convert to legacy format
    n_states = len(all_labels)
    rows = []
    for i in range(n_states):
        for j in range(n_states):
            rows.append({
                'from': all_labels[i],
                'to': all_labels[j],
                'diff': result.edges['diffs_true'][i, j],
                'p_value': result.edges['stats'].iloc[i * n_states + j]['p_value']
                if len(result.edges['stats']) == n_states * n_states else 0,
                'p_adjusted': result.edges['stats'].iloc[i * n_states + j]['p_value']
                if len(result.edges['stats']) == n_states * n_states else 0,
                'significant': False,
            })

    df_result = pd.DataFrame(rows)
    # The edge stats are in column-major order; rebuild p_values in row-major
    p_matrix = np.zeros((n_states, n_states))
    for idx, row in result.edges['stats'].iterrows():
        # Parse edge name to get indices
        parts = row['edge_name'].split(' -> ')
        from_label = parts[0]
        to_label = parts[1]
        i = all_labels.index(from_label)
        j = all_labels.index(to_label)
        p_matrix[i, j] = row['p_value']

    rows = []
    for i in range(n_states):
        for j in range(n_states):
            rows.append({
                'from': all_labels[i],
                'to': all_labels[j],
                'diff': result.edges['diffs_true'][i, j],
                'p_value': p_matrix[i, j],
                'p_adjusted': p_matrix[i, j],
                'significant': p_matrix[i, j] < 0.05,
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# P-value adjustment
# -----------------------------------------------------------------------------


def _p_adjust(p_values: np.ndarray, method: str = "none") -> np.ndarray:
    """Adjust p-values for multiple testing.

    Matches R's p.adjust() function.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
    method : str
        Adjustment method: 'none', 'bonferroni', 'holm', 'fdr'/'BH'

    Returns
    -------
    np.ndarray
        Adjusted p-values
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)

    if method == "none":
        return p.copy()

    elif method == "bonferroni":
        return np.minimum(p * n, 1.0)

    elif method == "holm":
        # Holm step-down
        sorted_idx = np.argsort(p)
        sorted_p = p[sorted_idx]
        adjusted = np.zeros(n)
        cummax = 0.0
        for i in range(n):
            val = sorted_p[i] * (n - i)
            cummax = max(cummax, val)
            adjusted[sorted_idx[i]] = cummax
        return np.minimum(adjusted, 1.0)

    elif method in ("fdr", "BH"):
        return _fdr_correction(p)

    else:
        raise ValueError(f"Unknown p.adjust method: {method}")


def _fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Matches R's p.adjust(method = "BH").

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
    alpha : float
        Significance level (not used for adjusted p-values)

    Returns
    -------
    np.ndarray
        Adjusted p-values
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH procedure: cumulative minimum of n/i * p[i] from largest to smallest
    adjusted = np.zeros(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]

    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(
            sorted_p[i] * n / (i + 1),
            adjusted[sorted_idx[i + 1]]
        )

    return np.minimum(adjusted, 1.0)


# -----------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------


def plot_bootstrap(
    result: BootstrapResult,
    plot_type: str = "weights",
    measure: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "RdBu_r"
) -> Figure:
    """Visualize bootstrap results.

    Parameters
    ----------
    result : BootstrapResult
        Result from bootstrap_tna()
    plot_type : str
        What to plot: 'weights', 'pvalues', or 'significance'
    measure : str, optional
        Centrality measure for centrality plots
    figsize : tuple
        Figure size
    cmap : str
        Colormap for heatmap

    Returns
    -------
    matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Plotting requires matplotlib. Install with: pip install matplotlib")

    labels = result.labels
    n = len(labels)

    if plot_type == "weights":
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Point estimate
        im0 = axes[0].imshow(result.weights_orig, cmap=cmap, aspect='equal')
        axes[0].set_xticks(range(n))
        axes[0].set_yticks(range(n))
        axes[0].set_xticklabels(labels, rotation=45, ha='right')
        axes[0].set_yticklabels(labels)
        axes[0].set_title('Original Weights')
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        # CI width
        ci_width = result.ci_upper - result.ci_lower
        im1 = axes[1].imshow(ci_width, cmap='YlOrRd', aspect='equal')
        axes[1].set_xticks(range(n))
        axes[1].set_yticks(range(n))
        axes[1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1].set_yticklabels(labels)
        axes[1].set_title(f'{int((1 - result.level) * 100)}% CI Width')
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        # Significance
        sig = (result.p_values < result.level).astype(float)
        sig[result.weights_orig == 0] = 0.5
        im2 = axes[2].imshow(sig, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
        axes[2].set_xticks(range(n))
        axes[2].set_yticks(range(n))
        axes[2].set_xticklabels(labels, rotation=45, ha='right')
        axes[2].set_yticklabels(labels)
        axes[2].set_title(f'Significant (p < {result.level})')
        cbar = plt.colorbar(im2, ax=axes[2], shrink=0.8)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['No', 'N/A', 'Yes'])

        plt.tight_layout()
        return fig

    elif plot_type == "centrality":
        if measure is None:
            raise ValueError("measure must be specified for centrality plot")

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(n), result.weights_mean.sum(axis=1))
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title(f'Bootstrap Mean Weights ({measure})')
        plt.tight_layout()
        return fig

    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")


def plot_permutation(
    result: PermutationResult,
    figsize: tuple[float, float] = (8, 6),
    ax: Axes | None = None,
    bins: int = 30,
    color: str = "lightblue"
) -> Axes:
    """Plot permutation test results.

    Parameters
    ----------
    result : PermutationResult
        Result from permutation_test()
    figsize : tuple
        Figure size (used if ax is None)
    ax : matplotlib Axes, optional
        Axes to plot on
    bins : int
        Number of histogram bins
    color : str
        Histogram color

    Returns
    -------
    matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Plotting requires matplotlib. Install with: pip install matplotlib")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot p-value distribution
    p_vals = result.edges['stats']['p_value'].values
    ax.hist(p_vals, bins=bins, color=color,
            edgecolor='black', alpha=0.7, label='Edge p-values')

    # Add significance threshold line
    ax.axvline(0.05, color='red', linewidth=2,
               linestyle='--', label='p = 0.05')

    n_sig = np.sum(p_vals < 0.05)
    n_total = len(p_vals)
    ax.text(0.95, 0.95, f'{n_sig}/{n_total} significant edges',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('P-value')
    ax.set_ylabel('Frequency')
    ax.set_title('Permutation Test Edge P-values')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax


def plot_network_ci(
    bootstrap_result: BootstrapResult,
    edge_alpha: str = "significance",
    figsize: tuple[float, float] = (10, 10),
    layout: str = "circular",
    **kwargs
) -> Axes:
    """Network plot with confidence interval information.

    Edge width represents point estimate, alpha represents significance.

    Parameters
    ----------
    bootstrap_result : BootstrapResult
        Result from bootstrap_tna()
    edge_alpha : str
        How to set edge transparency:
        - 'significance': fade non-significant edges
        - 'ci_width': fade edges with wide CIs
    figsize : tuple
        Figure size
    layout : str
        Network layout algorithm
    **kwargs
        Additional arguments

    Returns
    -------
    matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError("Plotting requires matplotlib. Install with: pip install matplotlib")

    import networkx as nx
    from .plot import _get_layout
    from .colors import color_palette

    labels = bootstrap_result.labels
    n = len(labels)
    weights = bootstrap_result.weights_orig

    fig, ax = plt.subplots(figsize=figsize)

    # Create graph
    G = nx.DiGraph()
    G.add_nodes_from(labels)

    for i, from_label in enumerate(labels):
        for j, to_label in enumerate(labels):
            weight = weights[i, j]
            if weight > 0:
                G.add_edge(from_label, to_label, weight=weight)

    # Get layout
    pos = _get_layout(G, layout)

    # Get colors
    colors = color_palette(n)
    node_colors = colors

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    # Calculate edge properties
    edges = list(G.edges(data=True))
    edge_widths = []
    edge_alphas = []

    for u, v, data in edges:
        i = labels.index(u)
        j = labels.index(v)
        weight = data['weight']

        edge_widths.append(0.5 + 4.5 * weight / weights.max())

        if edge_alpha == "significance":
            is_sig = bootstrap_result.p_values[i, j] < bootstrap_result.level
            edge_alphas.append(0.9 if is_sig else 0.2)
        elif edge_alpha == "ci_width":
            ci_w = bootstrap_result.ci_upper[i, j] - bootstrap_result.ci_lower[i, j]
            max_w = np.max(bootstrap_result.ci_upper - bootstrap_result.ci_lower)
            if max_w > 0:
                edge_alphas.append(0.9 - 0.7 * ci_w / max_w)
            else:
                edge_alphas.append(0.9)
        else:
            edge_alphas.append(0.7)

    for idx, (u, v, data) in enumerate(edges):
        if u != v:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=edge_widths[idx],
                edge_color='gray',
                alpha=edge_alphas[idx],
                arrows=True,
                arrowsize=15,
                connectionstyle="arc3,rad=0.1",
                ax=ax
            )

    sig_patch = mpatches.Patch(color='gray', alpha=0.9, label='Significant')
    nonsig_patch = mpatches.Patch(color='gray', alpha=0.2, label='Non-significant')
    ax.legend(handles=[sig_patch, nonsig_patch], loc='upper left')

    ax.set_title(f'TNA Network (p < {bootstrap_result.level})', fontweight='bold')
    ax.set_axis_off()
    ax.margins(0.1)

    return ax


# -----------------------------------------------------------------------------
# Module exports
# -----------------------------------------------------------------------------

__all__ = [
    # Result classes
    'BootstrapResult',
    'PermutationResult',
    # Bootstrap functions
    'bootstrap_tna',
    'bootstrap_centralities',
    # Permutation tests
    'permutation_test',
    'permutation_test_edges',
    # CI methods
    'confidence_interval',
    'bca_ci',
    # Plotting
    'plot_bootstrap',
    'plot_permutation',
    'plot_network_ci',
]
