/**
 * Color palette utilities for TNA visualization.
 * Port of Python tna/colors.py
 */

/** Default 9-color palette matching R TNA package. */
export const DEFAULT_COLORS = [
  '#d1ea2c', '#fd5306', '#68b033', '#4a9dcc', '#8601b0',
  '#a60d35', '#0392ce', '#f97b72', '#7f7f7f',
];

/** Accent palette (8 colors). */
export const ACCENT_PALETTE = [
  '#7fc97f', '#beaed4', '#fdc086', '#ffff99',
  '#386cb0', '#f0027f', '#bf5b17', '#666666',
];

/** Set3 palette (12 colors). */
export const SET3_PALETTE = [
  '#8dd3c7', '#ffffb3', '#bebada', '#fb8072',
  '#80b1d3', '#fdb462', '#b3de69', '#fccde5',
  '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f',
];

/** Convert HCL to RGB (simplified). */
function hclToRgb(h: number, c: number, l: number): [number, number, number] {
  const hNorm = h / 360;
  const s = Math.min(1, c / 100);
  const lNorm = l / 100;

  const adjS = lNorm > 0.5 ? s * (1 - (lNorm - 0.5) * 0.5) : s;

  // HLS to RGB
  const hue2rgb = (p: number, q: number, t: number): number => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };

  const q = lNorm < 0.5
    ? lNorm * (1 + adjS)
    : lNorm + adjS - lNorm * adjS;
  const p = 2 * lNorm - q;

  return [
    Math.max(0, Math.min(1, hue2rgb(p, q, hNorm + 1 / 3))),
    Math.max(0, Math.min(1, hue2rgb(p, q, hNorm))),
    Math.max(0, Math.min(1, hue2rgb(p, q, hNorm - 1 / 3))),
  ];
}

/** Generate a qualitative HCL color palette. */
function generateHclPalette(
  n: number,
  hStart = 15,
  hEnd = 375,
  c = 60,
  l = 65,
): string[] {
  const colors: string[] = [];
  const hRange = hEnd - hStart;

  for (let i = 0; i < n; i++) {
    const h = (hStart + (i * hRange) / n) % 360;
    const [r, g, b] = hclToRgb(h, c, l);
    colors.push(
      `#${Math.round(r * 255).toString(16).padStart(2, '0')}${Math.round(g * 255).toString(16).padStart(2, '0')}${Math.round(b * 255).toString(16).padStart(2, '0')}`,
    );
  }

  return colors;
}

/**
 * Generate a color palette for TNA visualization.
 *
 * @param nStates - Number of colors needed
 * @param palette - Force a specific palette: 'default', 'accent', 'set3', 'hcl'
 */
export function colorPalette(
  nStates: number,
  palette?: 'default' | 'accent' | 'set3' | 'hcl',
): string[] {
  if (nStates <= 0) return [];

  if (palette) {
    let base: string[];
    switch (palette) {
      case 'default':
        base = DEFAULT_COLORS;
        break;
      case 'accent':
        base = ACCENT_PALETTE;
        break;
      case 'set3':
        base = SET3_PALETTE;
        break;
      case 'hcl':
        return generateHclPalette(nStates);
      default:
        throw new Error(`Unknown palette: ${palette}`);
    }
    if (nStates <= base.length) return base.slice(0, nStates);
    return [...base, ...generateHclPalette(nStates - base.length)];
  }

  // Auto-select based on nStates
  if (nStates <= 8) return ACCENT_PALETTE.slice(0, nStates);
  if (nStates <= 12) return SET3_PALETTE.slice(0, nStates);
  return generateHclPalette(nStates);
}

/** Convert hex to [r, g, b] (0-255). */
export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [
    parseInt(h.substring(0, 2), 16),
    parseInt(h.substring(2, 4), 16),
    parseInt(h.substring(4, 6), 16),
  ];
}

/** Convert RGB to hex. */
export function rgbToHex(r: number, g: number, b: number): string {
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

/** Lighten a color. */
export function lightenColor(hex: string, amount = 0.3): string {
  const [r, g, b] = hexToRgb(hex);
  return rgbToHex(
    Math.round(r + (255 - r) * amount),
    Math.round(g + (255 - g) * amount),
    Math.round(b + (255 - b) * amount),
  );
}

/** Darken a color. */
export function darkenColor(hex: string, amount = 0.3): string {
  const [r, g, b] = hexToRgb(hex);
  return rgbToHex(
    Math.round(r * (1 - amount)),
    Math.round(g * (1 - amount)),
    Math.round(b * (1 - amount)),
  );
}

/** Create a mapping from labels to colors. */
export function createColorMap(labels: string[], colors?: string[]): Record<string, string> {
  const c = colors ?? colorPalette(labels.length);
  const result: Record<string, string> = {};
  for (let i = 0; i < labels.length; i++) {
    result[labels[i]!] = c[i % c.length]!;
  }
  return result;
}
