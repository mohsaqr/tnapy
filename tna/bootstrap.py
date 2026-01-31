"""Statistical inference for TNA: bootstrap and permutation tests.

Provides bootstrap resampling and permutation testing functionality for
statistical significance analysis of TNA models and centrality measures.
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
from .prepare import TNAData
from .centralities import centralities as compute_centralities, AVAILABLE_MEASURES


# -----------------------------------------------------------------------------
# Result Classes
# -----------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """Result of bootstrap analysis for a TNA model.

    Attributes
    ----------
    estimate : TNA
        Point estimate (original model)
    replicates : list[TNA]
        All bootstrap models
    weights_ci : tuple[np.ndarray, np.ndarray]
        (lower, upper) confidence intervals for weights
    inits_ci : tuple[np.ndarray, np.ndarray]
        (lower, upper) confidence intervals for initial probabilities
    n_boot : int
        Number of bootstrap replicates
    ci_level : float
        Confidence interval level (e.g., 0.95)
    """

    estimate: TNA
    replicates: list[TNA]
    weights_ci: tuple[np.ndarray, np.ndarray]
    inits_ci: tuple[np.ndarray, np.ndarray]
    n_boot: int
    ci_level: float

    def summary(self) -> pd.DataFrame:
        """Generate summary statistics for bootstrap results.

        Returns
        -------
        pd.DataFrame
            Summary with edges, estimates, CIs, and standard errors
        """
        labels = self.estimate.labels
        n = len(labels)

        rows = []
        for i in range(n):
            for j in range(n):
                rows.append({
                    'from': labels[i],
                    'to': labels[j],
                    'estimate': self.estimate.weights[i, j],
                    'ci_lower': self.weights_ci[0][i, j],
                    'ci_upper': self.weights_ci[1][i, j],
                    'se': np.std([r.weights[i, j] for r in self.replicates]),
                })

        return pd.DataFrame(rows)

    def significant_edges(self, threshold: float = 0) -> list[tuple[str, str, float]]:
        """Find edges significantly different from threshold.

        Parameters
        ----------
        threshold : float
            Value to test against (default: 0)

        Returns
        -------
        list of tuples
            List of (from_state, to_state, estimate) for significant edges
        """
        labels = self.estimate.labels
        n = len(labels)
        significant = []

        for i in range(n):
            for j in range(n):
                lower = self.weights_ci[0][i, j]
                upper = self.weights_ci[1][i, j]
                # Edge is significant if CI doesn't contain threshold
                if lower > threshold or upper < threshold:
                    significant.append((
                        labels[i],
                        labels[j],
                        self.estimate.weights[i, j]
                    ))

        return significant


@dataclass
class PermutationResult:
    """Result of permutation test for comparing TNA models.

    Attributes
    ----------
    observed : float | np.ndarray
        Observed test statistic
    null_distribution : np.ndarray
        Permutation distribution
    p_value : float | np.ndarray
        P-value(s)
    n_perm : int
        Number of permutations
    alternative : str
        Alternative hypothesis ('two-sided', 'greater', 'less')
    """

    observed: float | np.ndarray
    null_distribution: np.ndarray
    p_value: float | np.ndarray
    n_perm: int
    alternative: str

    def is_significant(self, alpha: float = 0.05) -> bool | np.ndarray:
        """Check if result is significant at given alpha level.

        Parameters
        ----------
        alpha : float
            Significance level (default: 0.05)

        Returns
        -------
        bool or np.ndarray
            True if significant
        """
        if isinstance(self.p_value, np.ndarray):
            return self.p_value < alpha
        return self.p_value < alpha

    def plot(self, ax: Axes | None = None, **kwargs) -> Axes:
        """Plot the null distribution with observed statistic.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to plot on
        **kwargs
            Additional arguments passed to hist()

        Returns
        -------
        matplotlib Axes
        """
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
    data: pd.DataFrame | TNAData,
    n_boot: int = 1000,
    type_: str = "relative",
    ci: float = 0.95,
    seed: int | None = None,
    parallel: bool = False
) -> BootstrapResult:
    """Bootstrap resampling of sequence data to estimate confidence intervals.

    Resamples sequences (rows) with replacement and rebuilds TNA model each time
    to estimate sampling variability of model parameters.

    Parameters
    ----------
    data : pd.DataFrame or TNAData
        Sequence data in wide format (rows = sequences, cols = time steps)
    n_boot : int
        Number of bootstrap replicates (default: 1000)
    type_ : str
        Model type for build_model (default: 'relative')
    ci : float
        Confidence interval level (default: 0.95)
    seed : int, optional
        Random seed for reproducibility
    parallel : bool
        Whether to use parallel processing (not yet implemented)

    Returns
    -------
    BootstrapResult
        Object containing bootstrap estimates and confidence intervals

    Examples
    --------
    >>> import tna
    >>> df = tna.load_group_regulation()
    >>> boot = tna.bootstrap_tna(df, n_boot=500)
    >>> print(boot.summary())
    >>> sig_edges = boot.significant_edges(threshold=0.05)
    """
    # Handle TNAData input
    if isinstance(data, TNAData):
        df = data.sequence_data
    else:
        df = data

    # Convert to numpy for efficient resampling
    arr = df.values
    n_sequences = len(arr)

    # Set random seed
    rng = np.random.default_rng(seed)

    # Build original model (point estimate)
    estimate = build_model(df, type_=type_)
    labels = estimate.labels
    n_states = len(labels)

    # Bootstrap resampling
    replicates = []
    weights_samples = np.zeros((n_boot, n_states, n_states))
    inits_samples = np.zeros((n_boot, n_states))

    for b in range(n_boot):
        # Resample sequences with replacement
        indices = rng.choice(n_sequences, size=n_sequences, replace=True)
        boot_data = pd.DataFrame(arr[indices], columns=df.columns)

        # Build model from bootstrap sample
        boot_model = build_model(boot_data, type_=type_, labels=labels)
        replicates.append(boot_model)
        weights_samples[b] = boot_model.weights
        inits_samples[b] = boot_model.inits

    # Compute confidence intervals
    alpha = 1 - ci
    weights_lower = np.percentile(weights_samples, 100 * alpha / 2, axis=0)
    weights_upper = np.percentile(weights_samples, 100 * (1 - alpha / 2), axis=0)
    inits_lower = np.percentile(inits_samples, 100 * alpha / 2, axis=0)
    inits_upper = np.percentile(inits_samples, 100 * (1 - alpha / 2), axis=0)

    return BootstrapResult(
        estimate=estimate,
        replicates=replicates,
        weights_ci=(weights_lower, weights_upper),
        inits_ci=(inits_lower, inits_upper),
        n_boot=n_boot,
        ci_level=ci
    )


def bootstrap_centralities(
    data: pd.DataFrame | TNAData,
    measures: list[str] | None = None,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
    type_: str = "relative",
    loops: bool = False,
    normalize: bool = False
) -> pd.DataFrame:
    """Bootstrap confidence intervals for centrality measures.

    Parameters
    ----------
    data : pd.DataFrame or TNAData
        Sequence data in wide format
    measures : list of str, optional
        Centrality measures to compute. If None, computes all.
    n_boot : int
        Number of bootstrap replicates (default: 1000)
    ci : float
        Confidence interval level (default: 0.95)
    seed : int, optional
        Random seed for reproducibility
    type_ : str
        Model type for build_model (default: 'relative')
    loops : bool
        Whether to include self-loops in centrality calculations
    normalize : bool
        Whether to normalize centrality measures

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: measure, state, estimate, ci_lower, ci_upper, se

    Examples
    --------
    >>> import tna
    >>> df = tna.load_group_regulation()
    >>> cent_ci = tna.bootstrap_centralities(df, measures=['OutStrength', 'Betweenness'])
    >>> print(cent_ci[cent_ci['measure'] == 'OutStrength'])
    """
    if measures is None:
        measures = AVAILABLE_MEASURES.copy()

    # Handle TNAData input
    if isinstance(data, TNAData):
        df = data.sequence_data
    else:
        df = data

    arr = df.values
    n_sequences = len(arr)
    rng = np.random.default_rng(seed)

    # Build original model and compute centralities
    estimate_model = build_model(df, type_=type_)
    labels = estimate_model.labels
    estimate_cent = compute_centralities(
        estimate_model, measures=measures, loops=loops, normalize=normalize
    )

    # Bootstrap centralities
    n_states = len(labels)
    n_measures = len(measures)
    cent_samples = np.zeros((n_boot, n_states, n_measures))

    for b in range(n_boot):
        indices = rng.choice(n_sequences, size=n_sequences, replace=True)
        boot_data = pd.DataFrame(arr[indices], columns=df.columns)
        boot_model = build_model(boot_data, type_=type_, labels=labels)
        boot_cent = compute_centralities(
            boot_model, measures=measures, loops=loops, normalize=normalize
        )
        for m_idx, measure in enumerate(measures):
            cent_samples[b, :, m_idx] = boot_cent[measure].values

    # Compute CIs and standard errors
    alpha = 1 - ci
    rows = []
    for m_idx, measure in enumerate(measures):
        for s_idx, state in enumerate(labels):
            values = cent_samples[:, s_idx, m_idx]
            rows.append({
                'measure': measure,
                'state': state,
                'estimate': estimate_cent.loc[state, measure],
                'ci_lower': np.percentile(values, 100 * alpha / 2),
                'ci_upper': np.percentile(values, 100 * (1 - alpha / 2)),
                'se': np.std(values)
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Permutation Tests
# -----------------------------------------------------------------------------


def permutation_test(
    data1: pd.DataFrame | TNAData,
    data2: pd.DataFrame | TNAData,
    n_perm: int = 1000,
    statistic: str = "weights",
    measure: str | None = None,
    alternative: str = "two-sided",
    seed: int | None = None,
    type_: str = "relative"
) -> PermutationResult:
    """Permutation test for comparing two TNA models.

    Tests the null hypothesis that there is no difference between the two groups.

    Parameters
    ----------
    data1 : pd.DataFrame or TNAData
        First group's sequence data
    data2 : pd.DataFrame or TNAData
        Second group's sequence data
    n_perm : int
        Number of permutations (default: 1000)
    statistic : str
        Test statistic: 'weights' (Frobenius norm of difference),
        'centrality' (difference in specific centrality measure),
        'density' (difference in network density)
    measure : str, optional
        Centrality measure name when statistic='centrality'
    alternative : str
        Alternative hypothesis: 'two-sided', 'greater', 'less'
    seed : int, optional
        Random seed for reproducibility
    type_ : str
        Model type for build_model (default: 'relative')

    Returns
    -------
    PermutationResult
        Object containing observed statistic, null distribution, and p-value

    Examples
    --------
    >>> import tna
    >>> df = tna.load_group_regulation()
    >>> # Split data (example)
    >>> df1 = df.iloc[:1000]
    >>> df2 = df.iloc[1000:]
    >>> result = tna.permutation_test(df1, df2, n_perm=500)
    >>> print(f"p-value: {result.p_value:.4f}")
    """
    # Handle TNAData input
    if isinstance(data1, TNAData):
        df1 = data1.sequence_data
    else:
        df1 = data1

    if isinstance(data2, TNAData):
        df2 = data2.sequence_data
    else:
        df2 = data2

    # Combine data
    arr1 = df1.values
    arr2 = df2.values
    n1, n2 = len(arr1), len(arr2)
    combined = np.vstack([arr1, arr2])

    rng = np.random.default_rng(seed)

    # Get common labels
    model1 = build_model(df1, type_=type_)
    model2 = build_model(df2, type_=type_)
    all_labels = sorted(set(model1.labels) | set(model2.labels))

    # Define test statistic function
    def compute_statistic(group1_data, group2_data):
        m1 = build_model(
            pd.DataFrame(group1_data, columns=df1.columns),
            type_=type_, labels=all_labels
        )
        m2 = build_model(
            pd.DataFrame(group2_data, columns=df2.columns),
            type_=type_, labels=all_labels
        )

        if statistic == "weights":
            # Frobenius norm of weight difference
            return np.linalg.norm(m1.weights - m2.weights, 'fro')
        elif statistic == "density":
            # Difference in network density
            d1 = np.sum(m1.weights > 0) / (len(all_labels) ** 2)
            d2 = np.sum(m2.weights > 0) / (len(all_labels) ** 2)
            return d1 - d2
        elif statistic == "centrality":
            if measure is None:
                raise ValueError("measure must be specified for centrality statistic")
            cent1 = compute_centralities(m1, measures=[measure])
            cent2 = compute_centralities(m2, measures=[measure])
            # Mean absolute difference in centrality
            return np.mean(np.abs(cent1[measure].values - cent2[measure].values))
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

    # Observed statistic
    observed = compute_statistic(arr1, arr2)

    # Permutation distribution
    null_dist = np.zeros(n_perm)
    for p in range(n_perm):
        # Shuffle group labels
        perm_indices = rng.permutation(n1 + n2)
        perm_group1 = combined[perm_indices[:n1]]
        perm_group2 = combined[perm_indices[n1:]]
        null_dist[p] = compute_statistic(perm_group1, perm_group2)

    # Compute p-value
    if alternative == "two-sided":
        p_value = np.mean(np.abs(null_dist) >= np.abs(observed))
    elif alternative == "greater":
        p_value = np.mean(null_dist >= observed)
    elif alternative == "less":
        p_value = np.mean(null_dist <= observed)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Ensure p-value is at least 1/(n_perm+1) for finite samples
    p_value = max(p_value, 1 / (n_perm + 1))

    return PermutationResult(
        observed=observed,
        null_distribution=null_dist,
        p_value=p_value,
        n_perm=n_perm,
        alternative=alternative
    )


def permutation_test_edges(
    data1: pd.DataFrame | TNAData,
    data2: pd.DataFrame | TNAData,
    n_perm: int = 1000,
    correction: str = "fdr",
    seed: int | None = None,
    type_: str = "relative"
) -> pd.DataFrame:
    """Test each edge for significant difference between groups.

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

    Examples
    --------
    >>> import tna
    >>> df = tna.load_group_regulation()
    >>> df1 = df.iloc[:1000]
    >>> df2 = df.iloc[1000:]
    >>> edges = tna.permutation_test_edges(df1, df2, correction='fdr')
    >>> sig_edges = edges[edges['significant']]
    """
    # Handle TNAData input
    if isinstance(data1, TNAData):
        df1 = data1.sequence_data
    else:
        df1 = data1

    if isinstance(data2, TNAData):
        df2 = data2.sequence_data
    else:
        df2 = data2

    arr1 = df1.values
    arr2 = df2.values
    n1, n2 = len(arr1), len(arr2)
    combined = np.vstack([arr1, arr2])

    rng = np.random.default_rng(seed)

    # Get common labels
    model1 = build_model(df1, type_=type_)
    model2 = build_model(df2, type_=type_)
    all_labels = sorted(set(model1.labels) | set(model2.labels))
    n_states = len(all_labels)

    # Rebuild models with common labels
    model1 = build_model(df1, type_=type_, labels=all_labels)
    model2 = build_model(df2, type_=type_, labels=all_labels)

    # Observed differences
    observed_diff = model1.weights - model2.weights

    # Permutation distribution for each edge
    perm_diffs = np.zeros((n_perm, n_states, n_states))
    for p in range(n_perm):
        perm_indices = rng.permutation(n1 + n2)
        perm_data1 = pd.DataFrame(combined[perm_indices[:n1]], columns=df1.columns)
        perm_data2 = pd.DataFrame(combined[perm_indices[n1:]], columns=df1.columns)
        m1 = build_model(perm_data1, type_=type_, labels=all_labels)
        m2 = build_model(perm_data2, type_=type_, labels=all_labels)
        perm_diffs[p] = m1.weights - m2.weights

    # Compute p-values for each edge (two-sided)
    p_values = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            obs = np.abs(observed_diff[i, j])
            null = np.abs(perm_diffs[:, i, j])
            p_values[i, j] = max(np.mean(null >= obs), 1 / (n_perm + 1))

    # Apply multiple testing correction
    p_flat = p_values.flatten()
    if correction == "bonferroni":
        p_adjusted = np.minimum(p_flat * len(p_flat), 1.0)
    elif correction == "fdr":
        p_adjusted = _fdr_correction(p_flat)
    elif correction == "none":
        p_adjusted = p_flat
    else:
        raise ValueError(f"Unknown correction: {correction}")

    p_adjusted = p_adjusted.reshape((n_states, n_states))

    # Build result DataFrame
    rows = []
    for i in range(n_states):
        for j in range(n_states):
            rows.append({
                'from': all_labels[i],
                'to': all_labels[j],
                'diff': observed_diff[i, j],
                'p_value': p_values[i, j],
                'p_adjusted': p_adjusted[i, j],
                'significant': p_adjusted[i, j] < 0.05
            })

    return pd.DataFrame(rows)


def _fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

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

    # BH procedure
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
    """Visualize bootstrap distributions and confidence intervals.

    Parameters
    ----------
    result : BootstrapResult
        Result from bootstrap_tna()
    plot_type : str
        What to plot: 'weights' or 'centrality'
    measure : str, optional
        Centrality measure when plot_type='centrality'
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

    if plot_type == "weights":
        # Plot weight estimates with CIs as error bars or heatmap
        labels = result.estimate.labels
        n = len(labels)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Point estimate
        im0 = axes[0].imshow(result.estimate.weights, cmap=cmap, aspect='equal')
        axes[0].set_xticks(range(n))
        axes[0].set_yticks(range(n))
        axes[0].set_xticklabels(labels, rotation=45, ha='right')
        axes[0].set_yticklabels(labels)
        axes[0].set_title('Point Estimate')
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        # CI width (uncertainty)
        ci_width = result.weights_ci[1] - result.weights_ci[0]
        im1 = axes[1].imshow(ci_width, cmap='YlOrRd', aspect='equal')
        axes[1].set_xticks(range(n))
        axes[1].set_yticks(range(n))
        axes[1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1].set_yticklabels(labels)
        axes[1].set_title(f'{int(result.ci_level * 100)}% CI Width')
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        # Significance (CI excludes 0)
        sig = ((result.weights_ci[0] > 0) | (result.weights_ci[1] < 0)).astype(float)
        sig[result.estimate.weights == 0] = 0.5  # Gray for zero weights
        im2 = axes[2].imshow(sig, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
        axes[2].set_xticks(range(n))
        axes[2].set_yticks(range(n))
        axes[2].set_xticklabels(labels, rotation=45, ha='right')
        axes[2].set_yticklabels(labels)
        axes[2].set_title('Significant (CI excludes 0)')
        cbar = plt.colorbar(im2, ax=axes[2], shrink=0.8)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['No', 'N/A', 'Yes'])

        plt.tight_layout()
        return fig

    elif plot_type == "centrality":
        if measure is None:
            raise ValueError("measure must be specified for centrality plot")

        # Compute centralities for all replicates
        labels = result.estimate.labels
        n = len(labels)

        cent_values = np.zeros((result.n_boot, n))
        for b, rep in enumerate(result.replicates):
            cent = compute_centralities(rep, measures=[measure])
            cent_values[b] = cent[measure].values

        # Point estimate
        orig_cent = compute_centralities(result.estimate, measures=[measure])

        fig, ax = plt.subplots(figsize=figsize)

        # Box plots
        positions = np.arange(n)
        bp = ax.boxplot([cent_values[:, i] for i in range(n)],
                       positions=positions, widths=0.6, patch_artist=True)

        # Overlay point estimates
        ax.scatter(positions, orig_cent[measure].values,
                  color='red', s=100, zorder=5, label='Point estimate')

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(measure)
        ax.set_title(f'Bootstrap Distribution: {measure}')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

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
    """Plot null distribution with observed statistic.

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

    # Plot null distribution
    ax.hist(result.null_distribution, bins=bins, color=color,
            edgecolor='black', alpha=0.7, label='Null distribution')

    # Add observed statistic line
    ax.axvline(result.observed, color='red', linewidth=2,
               linestyle='--', label=f'Observed = {result.observed:.4f}')

    # Add p-value annotation
    if isinstance(result.p_value, (float, np.floating)):
        ax.text(0.95, 0.95, f'p = {result.p_value:.4f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Test Statistic')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Permutation Test (n={result.n_perm})')
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
        Additional arguments passed to plot_network()

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

    model = bootstrap_result.estimate
    labels = model.labels
    n = len(labels)

    fig, ax = plt.subplots(figsize=figsize)

    # Create graph
    G = nx.DiGraph()
    G.add_nodes_from(labels)

    for i, from_label in enumerate(labels):
        for j, to_label in enumerate(labels):
            weight = model.weights[i, j]
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

        # Width based on weight
        edge_widths.append(0.5 + 4.5 * weight / model.weights.max())

        # Alpha based on significance or CI width
        if edge_alpha == "significance":
            lower = bootstrap_result.weights_ci[0][i, j]
            upper = bootstrap_result.weights_ci[1][i, j]
            # Significant if CI doesn't include 0
            is_sig = lower > 0 or upper < 0
            edge_alphas.append(0.9 if is_sig else 0.2)
        elif edge_alpha == "ci_width":
            ci_width = bootstrap_result.weights_ci[1][i, j] - bootstrap_result.weights_ci[0][i, j]
            max_width = np.max(bootstrap_result.weights_ci[1] - bootstrap_result.weights_ci[0])
            # Narrower CI = more certain = higher alpha
            if max_width > 0:
                edge_alphas.append(0.9 - 0.7 * ci_width / max_width)
            else:
                edge_alphas.append(0.9)
        else:
            edge_alphas.append(0.7)

    # Draw edges with varying alpha (need to draw individually)
    for idx, (u, v, data) in enumerate(edges):
        if u != v:  # Skip self-loops for now
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

    # Add legend
    sig_patch = mpatches.Patch(color='gray', alpha=0.9, label='Significant')
    nonsig_patch = mpatches.Patch(color='gray', alpha=0.2, label='Non-significant')
    ax.legend(handles=[sig_patch, nonsig_patch], loc='upper left')

    ax.set_title(f'TNA Network with {int(bootstrap_result.ci_level * 100)}% CI', fontweight='bold')
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
