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


# ---------------------------------------------------------------------------
# Helper functions for qgraph-style network drawing
# ---------------------------------------------------------------------------


def _compute_graph_center(
    pos: dict[str, tuple[float, float]]
) -> tuple[float, float]:
    """Compute the centroid of all node positions."""
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    return (np.mean(xs), np.mean(ys))


def _node_radius_approx(node_size: float, ax) -> float:
    """Convert scatter *point* size to approximate data-coordinate radius.

    matplotlib scatter sizes are in *points²*.  We convert to data
    coordinates using the figure size and axis limits directly, avoiding
    ax.get_window_extent() which is unreliable on non-interactive backends
    (e.g. Google Colab's agg backend).
    """
    fig = ax.figure
    dpi = fig.dpi
    radius_pts = np.sqrt(node_size) / 2          # half-side in points
    radius_inches = radius_pts / 72               # points -> inches
    radius_disp = radius_inches * dpi             # inches -> display px

    # Compute data-per-pixel from figure size (inches) and axis limits
    # This avoids get_window_extent() which fails on headless backends
    fig_w, fig_h = fig.get_size_inches()
    ax_pos = ax.get_position()  # fractional position within figure
    ax_w_px = ax_pos.width * fig_w * dpi
    ax_h_px = ax_pos.height * fig_h * dpi

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    sx = (xlim[1] - xlim[0]) / ax_w_px if ax_w_px else 1
    sy = (ylim[1] - ylim[0]) / ax_h_px if ax_h_px else 1
    scale = (sx + sy) / 2
    return radius_disp * scale


def _identify_bidirectional_pairs(
    edges: list[tuple[str, str]]
) -> set[tuple[str, str]]:
    """Return set of (u, v) edges that have a reverse (v, u) in *edges*.

    Only non-self-loop edges are considered.
    """
    edge_set = {(u, v) for u, v in edges if u != v}
    return {(u, v) for u, v in edge_set if (v, u) in edge_set}


def _compute_edge_alphas(
    weights: list[float],
    alpha_range: tuple[float, float] = (0.25, 0.95)
) -> list[float]:
    """Map edge weights to alpha (opacity) values."""
    if not weights:
        return []
    w_min, w_max = min(weights), max(weights)
    if w_max > w_min:
        return [
            alpha_range[0] + (w - w_min) / (w_max - w_min) * (alpha_range[1] - alpha_range[0])
            for w in weights
        ]
    return [(alpha_range[0] + alpha_range[1]) / 2] * len(weights)


def _draw_curved_edge(
    ax,
    p1: tuple[float, float],
    p2: tuple[float, float],
    rad: float = 0.0,
    width: float = 1.0,
    alpha: float = 0.7,
    color: str = '#444444',
    arrow_size: float = 15,
    shrink_a: float = 0.0,
    shrink_b: float = 0.0,
    zorder: int = 1,
):
    """Draw a single curved directed edge using FancyArrowPatch."""
    arrow = mpatches.FancyArrowPatch(
        p1, p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=f"->,head_length={arrow_size / 15:.2f},head_width={arrow_size / 25:.2f}",
        color=color,
        alpha=alpha,
        linewidth=width,
        mutation_scale=15,
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def _draw_self_loop(
    ax,
    node_pos: tuple[float, float],
    center: tuple[float, float],
    node_radius: float,
    width: float = 1.0,
    alpha: float = 0.7,
    color: str = '#444444',
    arrow_size: float = 15,
    loop_scale: float = 0.15,
    zorder: int = 1,
):
    """Draw a self-loop as a circular arc sitting outside the node.

    Draws an almost-complete circle (300 deg arc) whose center is placed
    outward from the node, tangent to the node perimeter — similar to
    qgraph's self-loop style.
    """
    from matplotlib.patches import Arc, FancyArrow
    nx_pos, ny_pos = node_pos
    cx, cy = center

    # Direction from graph center to node (outward)
    dx = nx_pos - cx
    dy = ny_pos - cy
    dist = np.hypot(dx, dy)
    if dist < 1e-9:
        dx, dy = 0.0, 1.0  # default: upward
    else:
        dx, dy = dx / dist, dy / dist

    # Loop circle: sits outside the node, tangent to the perimeter
    loop_r = node_radius * (0.6 + loop_scale * 2)
    loop_cx = nx_pos + dx * (node_radius + loop_r * 0.85)
    loop_cy = ny_pos + dy * (node_radius + loop_r * 0.85)

    # Arc opening faces the node; rotate so the gap is toward the node
    outward_angle = np.degrees(np.arctan2(dy, dx))
    # The arc spans 300 degrees, gap of 60 degrees facing the node
    # theta1/theta2 are relative to the `angle` rotation
    arc = Arc(
        (loop_cx, loop_cy),
        2 * loop_r, 2 * loop_r,
        angle=outward_angle + 180,  # rotate so gap faces node
        theta1=30, theta2=330,
        color=color,
        linewidth=width,
        alpha=alpha,
        zorder=zorder + 1,
    )
    ax.add_patch(arc)

    # Add arrowhead at the end of the arc (theta2=330 deg)
    # Compute the endpoint in data coordinates
    end_angle_rad = np.radians(outward_angle + 180 + 330)
    ex = loop_cx + loop_r * np.cos(end_angle_rad)
    ey = loop_cy + loop_r * np.sin(end_angle_rad)

    # Tangent direction at the endpoint (perpendicular to radius, clockwise)
    tangent_angle = end_angle_rad + np.pi / 2
    head_len = loop_r * 0.4
    arrow_dx = head_len * np.cos(tangent_angle)
    arrow_dy = head_len * np.sin(tangent_angle)

    arrowhead = FancyArrow(
        ex - arrow_dx, ey - arrow_dy,
        arrow_dx, arrow_dy,
        head_width=loop_r * 0.35,
        head_length=loop_r * 0.25,
        length_includes_head=True,
        fc=color, ec=color,
        alpha=alpha,
        zorder=zorder + 1,
        linewidth=0,
    )
    ax.add_patch(arrowhead)
    return arc


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
    minimum: float | None = None,
    cut: float | None = None,
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
    minimum : float, optional
        Alias for edge_threshold. Hide edges with weight below this value.
        If both minimum and edge_threshold are set, minimum takes precedence.
    cut : float, optional
        Fade edges below this weight (reduced alpha ~0.15) but still show them.
        Edges above this value are drawn normally.
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
    >>> tna.plot_network(model, minimum=0.05, cut=0.1)
    >>> plt.show()

    >>> # With centrality-based node sizing
    >>> tna.plot_network(model, node_size='OutStrength', layout='spring')
    """
    # Handle minimum as alias for edge_threshold
    if minimum is not None:
        edge_threshold = minimum
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

    # ------------------------------------------------------------------
    # Draw nodes (ax.scatter for white edge borders + proper zorder)
    # ------------------------------------------------------------------
    node_list = list(G.nodes())
    xs = [pos[n][0] for n in node_list]
    ys = [pos[n][1] for n in node_list]
    ax.scatter(
        xs, ys,
        s=sizes,
        c=node_colors,
        edgecolors='white',
        linewidths=2,
        zorder=3,
    )

    # Force axis limits before computing radii (scatter auto-scales)
    pad = 0.25
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    fig.canvas.draw()  # flush layout so axis positions are valid

    # Compute per-edge alphas proportional to weight
    edge_alphas = _compute_edge_alphas(edge_weights)

    # Apply cut parameter: fade edges below cut threshold
    if cut is not None:
        edge_alphas = [
            0.15 if w < cut else a
            for w, a in zip(edge_weights, edge_alphas)
        ]

    # ------------------------------------------------------------------
    # Draw edges
    # ------------------------------------------------------------------
    if edges_to_draw:
        # Identify bidirectional pairs for opposite curvature
        bidir = _identify_bidirectional_pairs(edges_to_draw)

        # Separate self-loops from regular edges
        regular_edges = [(u, v) for u, v in edges_to_draw if u != v]
        regular_widths = [edge_widths[i] for i, (u, v) in enumerate(edges_to_draw) if u != v]
        regular_alphas = [edge_alphas[i] for i, (u, v) in enumerate(edges_to_draw) if u != v]

        self_loops = [(u, v) for u, v in edges_to_draw if u == v]
        self_loop_widths = [edge_widths[i] for i, (u, v) in enumerate(edges_to_draw) if u == v]
        self_loop_alphas = [edge_alphas[i] for i, (u, v) in enumerate(edges_to_draw) if u == v]

        # Per-node shrink in points: scatter size s is in points²,
        # so radius in points = sqrt(s) / 2
        node_idx_map = {n: i for i, n in enumerate(node_list)}
        node_shrink_pts = {
            n: np.sqrt(sizes[node_idx_map[n]]) / 2 for n in node_list
        }
        # Data-coordinate radius for self-loop anchor placement
        node_radii_data = {
            n: _node_radius_approx(sizes[node_idx_map[n]], ax) for n in node_list
        }

        # Draw regular edges with FancyArrowPatch
        for (u, v), w, a in zip(regular_edges, regular_widths, regular_alphas):
            if (u, v) in bidir:
                # First direction (canonical pair gets +rad, reverse gets -rad)
                rad = 0.2 if u < v else -0.2
            else:
                rad = 0.0

            if not curved_edges:
                rad = 0.0

            _draw_curved_edge(
                ax, pos[u], pos[v],
                rad=rad,
                width=w,
                alpha=a,
                color='#444444',
                arrow_size=arrow_size,
                shrink_a=node_shrink_pts[u],
                shrink_b=node_shrink_pts[v],
                zorder=1,
            )

        # Draw self-loops
        if self_loops and show_self_loops:
            graph_center = _compute_graph_center(pos)
            for (u, _v), w, a in zip(self_loops, self_loop_widths, self_loop_alphas):
                _draw_self_loop(
                    ax, pos[u], graph_center,
                    node_radius=node_radii_data[u],
                    width=w,
                    alpha=a,
                    color='#444444',
                    arrow_size=arrow_size,
                    loop_scale=self_loop_scale,
                    zorder=1,
                )

    # ------------------------------------------------------------------
    # Draw node labels (ax.text — centered, bold, above edges)
    # ------------------------------------------------------------------
    label_map = {model.labels[i]: labels[i] for i in range(len(model.labels))}
    for node in G.nodes():
        x, y = pos[node]
        ax.text(
            x, y, label_map[node],
            fontsize=font_size,
            fontweight='bold',
            ha='center', va='center',
            zorder=4,
        )

    # ------------------------------------------------------------------
    # Draw edge labels (manual ax.text with smart positioning)
    # ------------------------------------------------------------------
    if edge_labels and edges_to_draw:
        bidir = _identify_bidirectional_pairs(edges_to_draw)
        graph_center = _compute_graph_center(pos)
        for (u, v), w in zip(edges_to_draw, edge_weights):
            if u == v:
                # Place label outside the self-loop arc
                nx_pos, ny_pos = pos[u]
                cx, cy = graph_center
                ddx = nx_pos - cx
                ddy = ny_pos - cy
                dd = np.hypot(ddx, ddy)
                if dd < 1e-9:
                    ddx, ddy = 0.0, 1.0
                else:
                    ddx, ddy = ddx / dd, ddy / dd
                node_idx = node_idx_map[u]
                nr = _node_radius_approx(sizes[node_idx], ax)
                loop_r = nr * (0.6 + self_loop_scale * 2)
                label_dist = nr + loop_r * 2.3
                lx = nx_pos + ddx * label_dist
                ly = ny_pos + ddy * label_dist
                ax.text(
                    lx, ly, f'{w:.2f}',
                    fontsize=edge_font_size,
                    ha='center', va='center',
                    zorder=5,
                    bbox=dict(
                        boxstyle='round,pad=0.15',
                        facecolor='white',
                        edgecolor='none',
                        alpha=0.8,
                    ),
                )
                continue
            # Midpoint
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2

            # Perpendicular offset for bidirectional edges
            if (u, v) in bidir:
                dx = x2 - x1
                dy = y2 - y1
                length = np.hypot(dx, dy) or 1.0
                # Normal vector
                nx_off = -dy / length
                ny_off = dx / length
                offset = 0.06 if u < v else -0.06
                mx += nx_off * offset
                my += ny_off * offset

            ax.text(
                mx, my, f'{w:.2f}',
                fontsize=edge_font_size,
                ha='center', va='center',
                zorder=5,
                bbox=dict(
                    boxstyle='round,pad=0.15',
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.8,
                ),
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
            figsize = (max(8, n_timesteps * 0.3), max(6, min(n_sequences * 0.03, 12)))

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
            ax.bar(freq_df.index, values, width=1.0, bottom=bottom,
                  color=color_map[state], label=state, edgecolor='white',
                  linewidth=0.3, **kwargs)
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


def plot_communities(
    result: Any,
    model: 'TNA | None' = None,
    method: str | None = None,
    layout: str = "circular",
    minimum: float | None = None,
    cut: float | None = None,
    figsize: tuple[float, float] = (10, 10),
    title: str | None = None,
    ax: Any | None = None,
    seed: int | None = 42,
    **kwargs
) -> Any:
    """Plot network with nodes colored by community assignment.

    Parameters
    ----------
    result : CommunityResult
        Result from communities()
    model : TNA, optional
        TNA model to use for the network. If None, uses result.model.
    method : str, optional
        Which community detection method to use for coloring.
        If None, uses the first method in the result.
    layout : str
        Layout algorithm for network positioning
    minimum : float, optional
        Hide edges with weight below this value
    cut : float, optional
        Fade edges below this weight
    figsize : tuple
        Figure size
    title : str, optional
        Plot title
    ax : matplotlib Axes, optional
        Axes to plot on
    seed : int, optional
        Random seed for reproducible layouts
    **kwargs
        Additional arguments passed to plot_network

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    _check_matplotlib()

    if model is None:
        model = result.model

    # Determine which method to use
    if method is None:
        method = list(result.counts.keys())[0]

    if method not in result.assignments.columns:
        raise ValueError(
            f"Method '{method}' not found. Available: {list(result.assignments.columns)}"
        )

    # Get community assignments
    assignments = result.assignments[method].values
    n_communities = result.counts[method]

    # Generate colors per community
    comm_colors = color_palette(n_communities)

    # Map node colors based on community assignment
    node_colors = [comm_colors[assignments[i]] for i in range(len(model.labels))]

    # Build title
    if title is None:
        title = f"Communities ({method}, k={n_communities})"

    return plot_network(
        model,
        colors=node_colors,
        layout=layout,
        minimum=minimum,
        cut=cut,
        figsize=figsize,
        title=title,
        ax=ax,
        seed=seed,
        **kwargs
    )
