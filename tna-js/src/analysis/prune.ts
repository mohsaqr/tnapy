/**
 * Pruning functions for TNA models.
 * Port of Python tna/prune.py
 */
import type { TNA, GroupTNA } from '../core/types.js';
import { isGroupTNA, groupEntries } from '../core/group.js';

/**
 * Prune edges below a weight threshold.
 * Sets all edges with weight below the threshold to zero.
 *
 * @param model - TNA model (or GroupTNA for per-group pruning)
 * @param threshold - Minimum edge weight to keep (default: 0.1)
 */
export function prune(model: TNA | GroupTNA, threshold = 0.1): TNA | Record<string, TNA> {
  if (isGroupTNA(model)) {
    const result: Record<string, TNA> = {};
    for (const [name, m] of groupEntries(model)) {
      result[name] = prune(m, threshold) as TNA;
    }
    return result;
  }

  const tnaModel = model as TNA;
  const weights = tnaModel.weights.map((v) => (v < threshold ? 0 : v));

  return {
    weights,
    inits: new Float64Array(tnaModel.inits),
    labels: [...tnaModel.labels],
    data: tnaModel.data,
    type: tnaModel.type,
    scaling: [...tnaModel.scaling],
  };
}
