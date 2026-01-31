"""Color palette utilities for TNA package.

Provides color palettes matching R TNA/TraMineR package styling.
"""

from __future__ import annotations

import colorsys
from typing import Sequence

# Default 9-color palette matching R TNA package
DEFAULT_COLORS = [
    '#d1ea2c',  # lime green
    '#fd5306',  # orange
    '#68b033',  # green
    '#4a9dcc',  # blue
    '#8601b0',  # purple
    '#a60d35',  # dark red
    '#0392ce',  # cyan
    '#f97b72',  # salmon
    '#7f7f7f',  # gray
]

# Accent palette (8 colors) - matches matplotlib/colorbrewer Accent
ACCENT_PALETTE = [
    '#7fc97f',  # green
    '#beaed4',  # purple
    '#fdc086',  # orange
    '#ffff99',  # yellow
    '#386cb0',  # blue
    '#f0027f',  # pink
    '#bf5b17',  # brown
    '#666666',  # gray
]

# Set3 palette (12 colors) - matches matplotlib/colorbrewer Set3
SET3_PALETTE = [
    '#8dd3c7',  # teal
    '#ffffb3',  # yellow
    '#bebada',  # lavender
    '#fb8072',  # salmon
    '#80b1d3',  # blue
    '#fdb462',  # orange
    '#b3de69',  # lime
    '#fccde5',  # pink
    '#d9d9d9',  # gray
    '#bc80bd',  # purple
    '#ccebc5',  # mint
    '#ffed6f',  # gold
]


def _hcl_to_rgb(h: float, c: float, l: float) -> tuple[float, float, float]:
    """Convert HCL (Hue-Chroma-Luminance) to RGB.

    This is a simplified conversion that approximates the HCL color space
    used in R's colorspace package.

    Parameters
    ----------
    h : float
        Hue in degrees (0-360)
    c : float
        Chroma (saturation-like, 0-100+)
    l : float
        Luminance (lightness, 0-100)

    Returns
    -------
    tuple of float
        RGB values in range [0, 1]
    """
    # Convert HCL to LAB (approximate)
    h_rad = h * 3.14159265 / 180.0
    a = c * (h_rad / (2 * 3.14159265) if h_rad > 0 else 0)
    b = c * (1 - h_rad / (2 * 3.14159265) if h_rad > 0 else 0)

    # Simplified: use HSL as approximation
    # Map HCL approximately to HSL
    h_norm = h / 360.0
    s = min(1.0, c / 100.0)
    l_norm = l / 100.0

    # Adjust saturation based on luminance for better colors
    if l_norm > 0.5:
        s = s * (1.0 - (l_norm - 0.5) * 0.5)

    return colorsys.hls_to_rgb(h_norm, l_norm, s)


def _generate_hcl_palette(n: int, h_start: float = 15, h_end: float = 375,
                          c: float = 60, l: float = 65) -> list[str]:
    """Generate a qualitative HCL color palette.

    Parameters
    ----------
    n : int
        Number of colors to generate
    h_start : float
        Starting hue (degrees)
    h_end : float
        Ending hue (degrees)
    c : float
        Chroma (saturation-like)
    l : float
        Luminance (lightness)

    Returns
    -------
    list of str
        Hex color codes
    """
    colors = []
    h_range = h_end - h_start

    for i in range(n):
        h = (h_start + (i * h_range / n)) % 360
        r, g, b = _hcl_to_rgb(h, c, l)
        # Clamp to valid range
        r = max(0, min(1, r))
        g = max(0, min(1, g))
        b = max(0, min(1, b))
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        colors.append(hex_color)

    return colors


def color_palette(n_states: int, palette: str | None = None) -> list[str]:
    """Generate a color palette for TNA visualization.

    Matches the R TNA/TraMineR package's color selection behavior.

    Parameters
    ----------
    n_states : int
        Number of states/colors needed
    palette : str, optional
        Force a specific palette:
        - 'default': Use DEFAULT_COLORS
        - 'accent': Use Accent palette
        - 'set3': Use Set3 palette
        - 'hcl': Generate HCL qualitative palette
        If None, automatically selects based on n_states.

    Returns
    -------
    list of str
        List of hex color codes

    Examples
    --------
    >>> colors = color_palette(5)
    >>> len(colors)
    5
    >>> colors[0]
    '#7fc97f'

    >>> colors = color_palette(15)  # More than 12, uses HCL
    >>> len(colors)
    15
    """
    if n_states <= 0:
        return []

    if palette is not None:
        palette = palette.lower()
        if palette == 'default':
            base = DEFAULT_COLORS
        elif palette == 'accent':
            base = ACCENT_PALETTE
        elif palette == 'set3':
            base = SET3_PALETTE
        elif palette == 'hcl':
            return _generate_hcl_palette(n_states)
        else:
            raise ValueError(f"Unknown palette: {palette}. "
                           "Use 'default', 'accent', 'set3', or 'hcl'.")

        if n_states <= len(base):
            return base[:n_states]
        else:
            # Extend with HCL colors
            extra = _generate_hcl_palette(n_states - len(base))
            return base + extra

    # Auto-select based on n_states (matching R TNA behavior)
    if n_states <= 8:
        return ACCENT_PALETTE[:n_states]
    elif n_states <= 12:
        return SET3_PALETTE[:n_states]
    else:
        return _generate_hcl_palette(n_states)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple.

    Parameters
    ----------
    hex_color : str
        Hex color code (with or without #)

    Returns
    -------
    tuple of int
        RGB values (0-255)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color.

    Parameters
    ----------
    r, g, b : int
        RGB values (0-255)

    Returns
    -------
    str
        Hex color code with #
    """
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def lighten_color(hex_color: str, amount: float = 0.3) -> str:
    """Lighten a color by a given amount.

    Parameters
    ----------
    hex_color : str
        Hex color code
    amount : float
        Amount to lighten (0-1)

    Returns
    -------
    str
        Lightened hex color
    """
    r, g, b = hex_to_rgb(hex_color)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return rgb_to_hex(r, g, b)


def darken_color(hex_color: str, amount: float = 0.3) -> str:
    """Darken a color by a given amount.

    Parameters
    ----------
    hex_color : str
        Hex color code
    amount : float
        Amount to darken (0-1)

    Returns
    -------
    str
        Darkened hex color
    """
    r, g, b = hex_to_rgb(hex_color)
    r = int(r * (1 - amount))
    g = int(g * (1 - amount))
    b = int(b * (1 - amount))
    return rgb_to_hex(r, g, b)


def create_color_map(labels: Sequence[str], colors: list[str] | None = None) -> dict[str, str]:
    """Create a mapping from labels to colors.

    Parameters
    ----------
    labels : sequence of str
        State labels
    colors : list of str, optional
        Colors to use. If None, generates automatically.

    Returns
    -------
    dict
        Mapping from label to hex color
    """
    if colors is None:
        colors = color_palette(len(labels))

    if len(colors) < len(labels):
        # Extend colors if needed
        extra = color_palette(len(labels) - len(colors), palette='hcl')
        colors = list(colors) + extra

    return {label: colors[i] for i, label in enumerate(labels)}
