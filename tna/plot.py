"""Plotting functions for TNA package.

Provides visualization capabilities matching the R TNA package using
matplotlib and networkx.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import networkx as nx

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from .colors import color_palette, create_color_map, DEFAULT_COLORS

if TYPE_CHECKING:
    from .model import TNA
    from .prepare import TNAData


def _check_matplotlib():
    """Raise error if matplotlib not available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Plotting requires matplotlib. Install with: pip install matplotlib"
        )


def _create_networkx_graph(model: 'TNA') -> nx.DiGraph:
    """Create a NetworkX DiGraph from a TNA model."""
    G = nx.DiGraph()
    n = len(model.labels)

    # Add nodes
    for i, label in enumerate(model.labels):
        G.add_node(label, index=i)

    # Add edges
    for i in range(n):
        for j in range(n):
            weight = model.weights[i, j]
            if weight > 0:
                G.add_edge(model.labels[i], model.labels[j], weight=weight)

    return G


def _get_layout(
    G: nx.DiGraph,
    layout: str,
    seed: int | None = None
) -> dict[str, tuple[float, float]]:
    """Get node positions based on layout algorithm."""
    if layout == "circular" or layout == "circle":
        return nx.circular_layout(G)
    elif layout == "spring" or layout == "fruchterman_reingold":
        return nx.spring_layout(G, seed=seed, k=2/np.sqrt(len(G)), iterations=50)
    elif layout == "kamada_kawai" or layout == "kk":
        try:
            return nx.kamada_kawai_layout(G)
        except nx.NetworkXException:
            # Fall back to spring if kamada_kawai fails
            return nx.spring_layout(G, seed=seed)
    elif layout == "shell":
        return nx.shell_layout(G)
    elif layout == "spectral":
        try:
            return nx.spectral_layout(G)
        except nx.NetworkXException:
            return nx.spring_layout(G, seed=seed)
    elif layout == "random":
        return nx.random_layout(G, seed=seed)
    else:
        raise ValueError(
            f"Unknown layout: {layout}. "
            "Available: circular, spring, kamada_kawai, shell, spectral, random"
        )


def plot_network(
    model: 'TNA',
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    layout: str = "circular",
    node_size: str | float | None = None,
    node_size_range: tuple[float, float] = (800, 3000),
    edge_labels: bool = True,
    edge_threshold: float | None = None,
    edge_width_range: tuple[float, float] = (0.5, 5.0),
    show_self_loops: bool = True,
    self_loop_scale: float = 0.15,
    curved_edges: bool = True,
    arrow_size: float = 15,
    font_size: float = 10,
    edge_font_size: float = 8,
    figsize: tuple[float, float] = (10, 10),
    title: str | None = None,
    ax: Any | None = None,
    seed: int | None = 42,
    **kwargs
) -> Any:
    """Plot a TNA model as a network graph.

    Parameters
    ----------
    model : TNA
        The TNA model to visualize
    labels : list of str, optional
        Custom node labels. If None, uses model.labels.
    colors : list of str, optional
        Custom node colors. If None, generates automatically.
    layout : str
        Layout algorithm: 'circular', 'spring', 'kamada_kawai', 'shell',
        'spectral', 'random'
    node_size : str or float, optional
        If str, name of centrality measure to scale nodes by
        ('OutStrength', 'InStrength', 'Betweenness', etc.).
        If float, fixed size for all nodes.
        If None, uses default size (1500).
    node_size_range : tuple
        Min and max node sizes when scaling by centrality
    edge_labels : bool
        Whether to show edge weight labels
    edge_threshold : float, optional
        Hide edges with weight below this threshold
    edge_width_range : tuple
        Min and max edge widths
    show_self_loops : bool
        Whether to show self-loops
    self_loop_scale : float
        Size scale for self-loop arcs
    curved_edges : bool
        Whether to use curved edges for directed graphs
    arrow_size : float
        Size of arrow heads
    font_size : float
        Font size for node labels
    edge_font_size : float
        Font size for edge labels
    figsize : tuple
        Figure size (width, height)
    title : str, optional
        Plot title
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    seed : int, optional
        Random seed for reproducible layouts
    **kwargs
        Additional arguments passed to networkx drawing functions

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot

    Examples
    --------
    >>> import tna
    >>> df = tna.load_group_regulation()
    >>> model = tna.tna(df)
    >>> tna.plot_network(model)
    >>> plt.show()

    >>> # With centrality-based node sizing
    >>> tna.plot_network(model, node_size='OutStrength', layout='spring')
    """
    _check_matplotlib()

    # Create graph
    G = _create_networkx_graph(model)

    # Get layout positions
    pos = _get_layout(G, layout, seed=seed)

    # Setup figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Get labels
    if labels is None:
        labels = model.labels

    # Get colors
    if colors is None:
        colors = color_palette(len(model.labels))

    color_map = {label: colors[i] for i, label in enumerate(model.labels)}
    node_colors = [color_map[node] for node in G.nodes()]

    # Calculate node sizes
    if node_size is None:
        sizes = [1500] * len(G.nodes())
    elif isinstance(node_size, (int, float)):
        sizes = [float(node_size)] * len(G.nodes())
    else:
        # node_size is a centrality measure name
        from .centralities import centralities
        cent = centralities(model, measures=[node_size])
        values = cent[node_size].values
        # Normalize to size range
        if values.max() > values.min():
            normalized = (values - values.min()) / (values.max() - values.min())
        else:
            normalized = np.ones(len(values)) * 0.5
        sizes = node_size_range[0] + normalized * (node_size_range[1] - node_size_range[0])
        sizes = [sizes[model.labels.index(node)] for node in G.nodes()]

    # Filter edges by threshold
    edges_to_draw = []
    edge_weights = []
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        if edge_threshold is not None and weight < edge_threshold:
            continue
        if not show_self_loops and u == v:
            continue
        edges_to_draw.append((u, v))
        edge_weights.append(weight)

    # Calculate edge widths
    if edge_weights:
        w_min, w_max = min(edge_weights), max(edge_weights)
        if w_max > w_min:
            edge_widths = [
                edge_width_range[0] + (w - w_min) / (w_max - w_min) * (edge_width_range[1] - edge_width_range[0])
                for w in edge_weights
            ]
        else:
            edge_widths = [(edge_width_range[0] + edge_width_range[1]) / 2] * len(edge_weights)
    else:
        edge_widths = []

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=sizes,
        ax=ax,
        **{k: v for k, v in kwargs.items() if k.startswith('node_')}
    )

    # Draw edges
    if edges_to_draw:
        # Separate self-loops from other edges
        regular_edges = [(u, v) for u, v in edges_to_draw if u != v]
        regular_widths = [edge_widths[i] for i, (u, v) in enumerate(edges_to_draw) if u != v]

        self_loops = [(u, v) for u, v in edges_to_draw if u == v]
        self_loop_widths = [edge_widths[i] for i, (u, v) in enumerate(edges_to_draw) if u == v]

        # Draw regular edges
        if regular_edges:
            connection_style = "arc3,rad=0.1" if curved_edges else "arc3,rad=0"
            nx.draw_networkx_edges(
                G, pos,
                edgelist=regular_edges,
                width=regular_widths,
                edge_color='gray',
                alpha=0.7,
                arrows=True,
                arrowsize=arrow_size,
                connectionstyle=connection_style,
                ax=ax,
                **{k: v for k, v in kwargs.items() if k.startswith('edge_')}
            )

        # Draw self-loops as arcs above nodes
        if self_loops and show_self_loops:
            for (u, v), width in zip(self_loops, self_loop_widths):
                x, y = pos[u]
                # Draw arc above the node
                node_idx = list(G.nodes()).index(u)
                node_radius = np.sqrt(sizes[node_idx]) / 150  # Approximate radius
                loop_radius = node_radius * self_loop_scale * 5

                arc = mpatches.FancyArrowPatch(
                    (x - loop_radius * 0.3, y + node_radius),
                    (x + loop_radius * 0.3, y + node_radius),
                    connectionstyle=f"arc3,rad=-1.5",
                    arrowstyle=f"->,head_length={arrow_size/15},head_width={arrow_size/20}",
                    color='gray',
                    alpha=0.7,
                    linewidth=width,
                    mutation_scale=15
                )
                ax.add_patch(arc)

    # Draw node labels
    label_map = {model.labels[i]: labels[i] for i in range(len(model.labels))}
    nx.draw_networkx_labels(
        G, pos,
        labels=label_map,
        font_size=font_size,
        font_weight='bold',
        ax=ax
    )

    # Draw edge labels
    if edge_labels and edges_to_draw:
        edge_label_dict = {}
        for (u, v), w in zip(edges_to_draw, edge_weights):
            if u != v:  # Skip self-loop labels (hard to position)
                edge_label_dict[(u, v)] = f'{w:.2f}'

        if edge_label_dict:
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_label_dict,
                font_size=edge_font_size,
                ax=ax
            )

    # Set title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_axis_off()
    ax.margins(0.1)

    return ax


def plot_centralities(
    centralities: pd.DataFrame,
    measures: list[str] | None = None,
    colors: list[str] | None = None,
    ncol: int = 3,
    figsize: tuple[float, float] | None = None,
    normalize: bool = False,
    sort_values: bool = True,
    title: str | None = None,
    **kwargs
) -> Any:
    """Plot centrality measures as faceted bar charts.

    Parameters
    ----------
    centralities : pd.DataFrame
        Centrality DataFrame from tna.centralities()
    measures : list of str, optional
        Which measures to plot. If None, plots all columns.
    colors : list of str, optional
        Colors for each state. If None, generates automatically.
    ncol : int
        Number of columns in subplot grid
    figsize : tuple, optional
        Figure size. Auto-calculated if None.
    normalize : bool
        Whether to normalize values to [0, 1] for display
    sort_values : bool
        Whether to sort bars by value within each subplot
    title : str, optional
        Overall figure title
    **kwargs
        Additional arguments passed to plt.barh()

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plots

    Examples
    --------
    >>> import tna
    >>> df = tna.load_group_regulation()
    >>> model = tna.tna(df)
    >>> cent = tna.centralities(model)
    >>> tna.plot_centralities(cent, ncol=3)
    >>> plt.show()
    """
    _check_matplotlib()

    if measures is None:
        measures = list(centralities.columns)
    else:
        # Validate measures
        invalid = set(measures) - set(centralities.columns)
        if invalid:
            raise ValueError(f"Unknown measures: {invalid}")

    n_measures = len(measures)
    nrow = (n_measures + ncol - 1) // ncol

    if figsize is None:
        figsize = (4 * ncol, 3 * nrow)

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    if n_measures == 1:
        axes = np.array([[axes]])
    elif nrow == 1:
        axes = axes.reshape(1, -1)
    elif ncol == 1:
        axes = axes.reshape(-1, 1)

    # Get colors
    if colors is None:
        colors = color_palette(len(centralities))

    color_map = {label: colors[i] for i, label in enumerate(centralities.index)}

    for idx, measure in enumerate(measures):
        row = idx // ncol
        col = idx % ncol
        ax = axes[row, col]

        values = centralities[measure].copy()

        if normalize:
            v_min, v_max = values.min(), values.max()
            if v_max > v_min:
                values = (values - v_min) / (v_max - v_min)

        if sort_values:
            values = values.sort_values()

        bar_colors = [color_map[label] for label in values.index]

        ax.barh(range(len(values)), values.values, color=bar_colors, **kwargs)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(values.index)
        ax.set_xlabel('Value')
        ax.set_title(measure, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused subplots
    for idx in range(n_measures, nrow * ncol):
        row = idx // ncol
        col = idx % ncol
        axes[row, col].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_heatmap(
    model: 'TNA',
    cmap: str = "RdBu_r",
    annot: bool = True,
    fmt: str = ".2f",
    figsize: tuple[float, float] = (8, 8),
    title: str | None = None,
    ax: Any | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs
) -> Any:
    """Plot transition matrix as a heatmap.

    Parameters
    ----------
    model : TNA
        The TNA model to visualize
    cmap : str
        Matplotlib colormap name
    annot : bool
        Whether to annotate cells with values
    fmt : str
        Format string for annotations
    figsize : tuple
        Figure size (width, height)
    title : str, optional
        Plot title
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    vmin, vmax : float, optional
        Min and max values for colormap
    **kwargs
        Additional arguments passed to plt.imshow() or sns.heatmap()

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot

    Examples
    --------
    >>> import tna
    >>> df = tna.load_group_regulation()
    >>> model = tna.tna(df)
    >>> tna.plot_heatmap(model)
    >>> plt.show()
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Use seaborn if available for nicer heatmaps
    if HAS_SEABORN:
        df = pd.DataFrame(model.weights, index=model.labels, columns=model.labels)
        sns.heatmap(
            df,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            square=True,
            **kwargs
        )
    else:
        # Fallback to matplotlib
        im = ax.imshow(model.weights, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

        # Set ticks and labels
        ax.set_xticks(range(len(model.labels)))
        ax.set_yticks(range(len(model.labels)))
        ax.set_xticklabels(model.labels, rotation=45, ha='right')
        ax.set_yticklabels(model.labels)

        # Add annotations
        if annot:
            for i in range(len(model.labels)):
                for j in range(len(model.labels)):
                    value = model.weights[i, j]
                    text_color = 'white' if value > (model.weights.max() + model.weights.min()) / 2 else 'black'
                    ax.text(j, i, f'{value:{fmt[1:]}}',
                           ha='center', va='center', color=text_color)

        plt.colorbar(im, ax=ax)

    ax.set_xlabel('To', fontweight='bold')
    ax.set_ylabel('From', fontweight='bold')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    elif title is None:
        ax.set_title('Transition Matrix', fontsize=14, fontweight='bold')

    return ax


def plot_comparison(
    model1: 'TNA',
    model2: 'TNA',
    plot_type: Literal["heatmap", "scatter", "network"] = "heatmap",
    labels: tuple[str, str] = ("Model 1", "Model 2"),
    figsize: tuple[float, float] | None = None,
    **kwargs
) -> Any:
    """Plot comparison of two TNA models.

    Parameters
    ----------
    model1 : TNA
        First TNA model
    model2 : TNA
        Second TNA model
    plot_type : str
        Type of comparison plot:
        - 'heatmap': Side-by-side heatmaps with difference
        - 'scatter': Scatter plot of corresponding weights
        - 'network': Side-by-side network plots
    labels : tuple of str
        Labels for the two models
    figsize : tuple, optional
        Figure size
    **kwargs
        Additional arguments passed to plotting functions

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the comparison

    Examples
    --------
    >>> import tna
    >>> df = tna.load_group_regulation()
    >>> model1 = tna.tna(df)
    >>> model2 = tna.ftna(df)
    >>> tna.plot_comparison(model1, model2, plot_type='heatmap')
    >>> plt.show()
    """
    _check_matplotlib()

    # Verify models have same labels
    if set(model1.labels) != set(model2.labels):
        raise ValueError("Models must have the same state labels for comparison")

    # Reorder model2 weights to match model1 label order
    label_order = model1.labels
    idx_map = {label: model2.labels.index(label) for label in label_order}
    weights2_reordered = np.zeros_like(model2.weights)
    for i, li in enumerate(label_order):
        for j, lj in enumerate(label_order):
            weights2_reordered[i, j] = model2.weights[idx_map[li], idx_map[lj]]

    if plot_type == "heatmap":
        if figsize is None:
            figsize = (16, 5)
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Model 1 heatmap
        plot_heatmap(model1, ax=axes[0], title=labels[0], **kwargs)

        # Model 2 heatmap (with reordered weights)
        model2_reordered = type(model2)(
            weights=weights2_reordered,
            inits=model2.inits,
            labels=label_order,
            data=model2.data,
            type_=model2.type_,
            scaling=model2.scaling
        )
        plot_heatmap(model2_reordered, ax=axes[1], title=labels[1], **kwargs)

        # Difference heatmap
        diff = model1.weights - weights2_reordered
        diff_max = max(abs(diff.min()), abs(diff.max()))
        im = axes[2].imshow(diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
        axes[2].set_xticks(range(len(label_order)))
        axes[2].set_yticks(range(len(label_order)))
        axes[2].set_xticklabels(label_order, rotation=45, ha='right')
        axes[2].set_yticklabels(label_order)
        axes[2].set_title(f'Difference ({labels[0]} - {labels[1]})', fontweight='bold')
        axes[2].set_xlabel('To', fontweight='bold')
        axes[2].set_ylabel('From', fontweight='bold')
        plt.colorbar(im, ax=axes[2])

        plt.tight_layout()
        return fig

    elif plot_type == "scatter":
        if figsize is None:
            figsize = (8, 8)
        fig, ax = plt.subplots(figsize=figsize)

        w1 = model1.weights.flatten()
        w2 = weights2_reordered.flatten()

        ax.scatter(w1, w2, alpha=0.6, edgecolors='black', linewidth=0.5)

        # Add diagonal reference line
        max_val = max(w1.max(), w2.max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')

        ax.set_xlabel(labels[0], fontweight='bold')
        ax.set_ylabel(labels[1], fontweight='bold')
        ax.set_title('Weight Comparison', fontsize=14, fontweight='bold')
        ax.legend()

        # Add correlation coefficient
        corr = np.corrcoef(w1, w2)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
               fontsize=12, verticalalignment='top')

        plt.tight_layout()
        return fig

    elif plot_type == "network":
        if figsize is None:
            figsize = (16, 8)
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        plot_network(model1, ax=axes[0], title=labels[0], **kwargs)

        model2_reordered = type(model2)(
            weights=weights2_reordered,
            inits=model2.inits,
            labels=label_order,
            data=model2.data,
            type_=model2.type_,
            scaling=model2.scaling
        )
        plot_network(model2_reordered, ax=axes[1], title=labels[1], **kwargs)

        plt.tight_layout()
        return fig

    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'heatmap', 'scatter', or 'network'.")


def plot_sequences(
    data: pd.DataFrame | 'TNAData',
    plot_type: Literal["index", "distribution"] = "index",
    colors: list[str] | None = None,
    max_sequences: int | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    **kwargs
) -> Any:
    """Plot sequence data visualization.

    Parameters
    ----------
    data : pd.DataFrame or TNAData
        Sequence data in wide format or TNAData object
    plot_type : str
        Type of sequence plot:
        - 'index': Each sequence as a row with colored states
        - 'distribution': Stacked bar chart showing state distribution over time
    colors : list of str, optional
        Colors for each state. If None, generates automatically.
    max_sequences : int, optional
        Maximum number of sequences to show in index plot
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    **kwargs
        Additional arguments passed to plotting functions

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot

    Examples
    --------
    >>> import tna
    >>> df = tna.load_group_regulation()
    >>> tna.plot_sequences(df, plot_type='index', max_sequences=100)
    >>> plt.show()
    """
    _check_matplotlib()

    # Handle TNAData input
    from .prepare import TNAData
    if isinstance(data, TNAData):
        df = data.sequence_data
    else:
        df = data

    # Get all unique states
    values = df.values.flatten()
    unique_states = pd.unique(values[~pd.isna(values)])
    unique_states = sorted([str(s) for s in unique_states])

    # Create state to integer mapping
    state_to_int = {state: i for i, state in enumerate(unique_states)}

    # Get colors
    if colors is None:
        colors = color_palette(len(unique_states))

    color_map = {state: colors[i] for i, state in enumerate(unique_states)}

    if plot_type == "index":
        # Index plot: each sequence as a row
        if max_sequences is not None and len(df) > max_sequences:
            df_plot = df.head(max_sequences)
        else:
            df_plot = df

        n_sequences = len(df_plot)
        n_timesteps = len(df_plot.columns)

        if figsize is None:
            figsize = (max(8, n_timesteps * 0.3), max(6, n_sequences * 0.1))

        fig, ax = plt.subplots(figsize=figsize)

        # Create numeric matrix for plotting
        matrix = np.full((n_sequences, n_timesteps), np.nan)
        for i, (_, row) in enumerate(df_plot.iterrows()):
            for j, val in enumerate(row):
                if pd.notna(val):
                    matrix[i, j] = state_to_int.get(str(val), -1)

        # Create custom colormap
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap(colors + ['white'])  # Add white for NaN
        bounds = list(range(len(unique_states) + 2))
        norm = BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(matrix, aspect='auto', cmap=cmap, norm=norm,
                      interpolation='nearest')

        ax.set_xlabel('Time Step', fontweight='bold')
        ax.set_ylabel('Sequence', fontweight='bold')

        # Create legend
        patches = [mpatches.Patch(color=color_map[state], label=state)
                  for state in unique_states]
        ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5))

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        elif title is None:
            ax.set_title('Sequence Index Plot', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    elif plot_type == "distribution":
        # Distribution plot: stacked bar chart over time
        n_timesteps = len(df.columns)

        # Count state frequencies at each time step
        freq_data = []
        for col in df.columns:
            counts = df[col].value_counts()
            total = counts.sum()
            freq_row = {str(state): counts.get(state, 0) / total if total > 0 else 0
                       for state in unique_states}
            freq_data.append(freq_row)

        freq_df = pd.DataFrame(freq_data, index=range(1, n_timesteps + 1))

        if figsize is None:
            figsize = (max(8, n_timesteps * 0.5), 6)

        fig, ax = plt.subplots(figsize=figsize)

        # Create stacked bar chart
        bottom = np.zeros(n_timesteps)
        for state in unique_states:
            values = freq_df[state].values
            ax.bar(freq_df.index, values, bottom=bottom,
                  color=color_map[state], label=state, **kwargs)
            bottom += values

        ax.set_xlabel('Time Step', fontweight='bold')
        ax.set_ylabel('Proportion', fontweight='bold')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylim(0, 1)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        elif title is None:
            ax.set_title('State Distribution Over Time', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'index' or 'distribution'.")


def plot_frequencies(
    model: 'TNA',
    colors: list[str] | None = None,
    horizontal: bool = True,
    sort_values: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    ax: Any | None = None,
    **kwargs
) -> Any:
    """Plot state frequencies from initial probabilities.

    Parameters
    ----------
    model : TNA
        The TNA model
    colors : list of str, optional
        Colors for each state. If None, generates automatically.
    horizontal : bool
        Whether to use horizontal bars
    sort_values : bool
        Whether to sort bars by frequency
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    ax : matplotlib Axes, optional
        Axes to plot on
    **kwargs
        Additional arguments passed to bar plot

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot

    Examples
    --------
    >>> import tna
    >>> df = tna.load_group_regulation()
    >>> model = tna.tna(df)
    >>> tna.plot_frequencies(model)
    >>> plt.show()
    """
    _check_matplotlib()

    if ax is None:
        if figsize is None:
            figsize = (8, 6)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Get colors
    if colors is None:
        colors = color_palette(len(model.labels))

    # Create frequency series
    freq = pd.Series(model.inits, index=model.labels)

    if sort_values:
        freq = freq.sort_values()

    bar_colors = [colors[model.labels.index(label)] for label in freq.index]

    if horizontal:
        ax.barh(range(len(freq)), freq.values, color=bar_colors, **kwargs)
        ax.set_yticks(range(len(freq)))
        ax.set_yticklabels(freq.index)
        ax.set_xlabel('Frequency', fontweight='bold')
    else:
        ax.bar(range(len(freq)), freq.values, color=bar_colors, **kwargs)
        ax.set_xticks(range(len(freq)))
        ax.set_xticklabels(freq.index, rotation=45, ha='right')
        ax.set_ylabel('Frequency', fontweight='bold')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    elif title is None:
        ax.set_title('State Frequencies', fontsize=14, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax


def plot_histogram(
    model: 'TNA',
    bins: int = 20,
    color: str = "lightblue",
    include_zeros: bool = False,
    figsize: tuple[float, float] = (8, 6),
    title: str | None = None,
    ax: Any | None = None,
    **kwargs
) -> Any:
    """Plot histogram of transition weights.

    Parameters
    ----------
    model : TNA
        The TNA model
    bins : int
        Number of histogram bins
    color : str
        Bar color
    include_zeros : bool
        Whether to include zero weights
    figsize : tuple
        Figure size
    title : str, optional
        Plot title
    ax : matplotlib Axes, optional
        Axes to plot on
    **kwargs
        Additional arguments passed to plt.hist()

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot

    Examples
    --------
    >>> import tna
    >>> df = tna.load_group_regulation()
    >>> model = tna.tna(df)
    >>> tna.plot_histogram(model)
    >>> plt.show()
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    weights = model.weights.flatten()

    if not include_zeros:
        weights = weights[weights > 0]

    ax.hist(weights, bins=bins, color=color, edgecolor='black', alpha=0.7, **kwargs)

    ax.set_xlabel('Transition Weight', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    elif title is None:
        ax.set_title('Distribution of Transition Weights', fontsize=14, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax
