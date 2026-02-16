/**
 * Clique detection for TNA models.
 * Port of Python tna/cliques.py
 */
import { Matrix } from '../core/matrix.js';
import type { TNA, GroupTNA, CliqueResult } from '../core/types.js';
import { isGroupTNA, groupEntries } from '../core/group.js';

/**
 * Find directed cliques in a TNA model.
 * A directed clique of size k is a set of k nodes where every pair
 * has edges in BOTH directions above the threshold.
 */
export function cliques(
  model: TNA | GroupTNA,
  options?: { size?: number; threshold?: number },
): CliqueResult | Record<string, CliqueResult> {
  if (isGroupTNA(model)) {
    const result: Record<string, CliqueResult> = {};
    for (const [name, m] of groupEntries(model)) {
      result[name] = cliques(m, options) as CliqueResult;
    }
    return result;
  }

  const tnaModel = model as TNA;
  const size = options?.size ?? 2;
  const threshold = options?.threshold ?? 0;
  const weights = tnaModel.weights;
  const n = weights.rows;

  // Build adjacency for upper triangle (i->j, i<j)
  const upperAdj: Set<number>[] = Array.from({ length: n }, () => new Set<number>());
  const lowerAdj: Set<number>[] = Array.from({ length: n }, () => new Set<number>());

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (weights.get(i, j) > threshold) {
        upperAdj[i]!.add(j);
        upperAdj[j]!.add(i);
      }
      if (weights.get(j, i) > threshold) {
        lowerAdj[i]!.add(j);
        lowerAdj[j]!.add(i);
      }
    }
  }

  // Find all cliques using Bron-Kerbosch
  const upperCliques = findAllCliques(upperAdj, n);
  const lowerCliques = findAllCliques(lowerAdj, n);

  // Get sub-cliques of exact size
  const subsUpper = subcliquesOfSize(upperCliques, size);
  const subsLower = subcliquesOfSize(lowerCliques, size);

  // Intersection
  const mutualCliques: number[][] = [];
  for (const clique of subsUpper) {
    const key = clique.join(',');
    for (const other of subsLower) {
      if (other.join(',') === key) {
        mutualCliques.push(clique);
        break;
      }
    }
  }

  // Sort and deduplicate
  const seen = new Set<string>();
  const uniqueCliques: number[][] = [];
  for (const c of mutualCliques) {
    const key = c.join(',');
    if (!seen.has(key)) {
      seen.add(key);
      uniqueCliques.push(c);
    }
  }
  uniqueCliques.sort((a, b) => {
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      if (a[i] !== b[i]) return a[i]! - b[i]!;
    }
    return a.length - b.length;
  });

  // Build results
  const resultWeights: Matrix[] = [];
  const resultIndices: number[][] = [];
  const resultLabels: string[][] = [];

  for (const idx of uniqueCliques) {
    const k = idx.length;
    const sub = Matrix.zeros(k, k);
    for (let r = 0; r < k; r++) {
      for (let c = 0; c < k; c++) {
        sub.set(r, c, weights.get(idx[r]!, idx[c]!));
      }
    }
    resultWeights.push(sub);
    resultIndices.push(idx);
    resultLabels.push(idx.map((i) => tnaModel.labels[i]!));
  }

  return {
    weights: resultWeights,
    indices: resultIndices,
    labels: resultLabels,
    size,
    threshold,
  };
}

/** Bron-Kerbosch algorithm with pivoting. */
function findAllCliques(adj: Set<number>[], n: number): number[][] {
  const cliques: number[][] = [];

  function bronKerbosch(R: number[], P: number[], X: number[]): void {
    if (P.length === 0 && X.length === 0) {
      if (R.length >= 2) cliques.push([...R]);
      return;
    }

    // Choose pivot with most connections in P
    let pivot = -1;
    let maxConn = -1;
    for (const u of [...P, ...X]) {
      let conn = 0;
      for (const v of P) {
        if (adj[u]!.has(v)) conn++;
      }
      if (conn > maxConn) {
        maxConn = conn;
        pivot = u;
      }
    }

    // Nodes in P not adjacent to pivot
    const candidates = pivot >= 0
      ? P.filter((v) => !adj[pivot]!.has(v))
      : [...P];

    for (const v of candidates) {
      const newR = [...R, v];
      const newP = P.filter((u) => adj[v]!.has(u));
      const newX = X.filter((u) => adj[v]!.has(u));
      bronKerbosch(newR, newP, newX);
      P = P.filter((u) => u !== v);
      X.push(v);
    }
  }

  const allNodes = Array.from({ length: n }, (_, i) => i);
  bronKerbosch([], allNodes, []);
  return cliques;
}

/** Get all sub-cliques of exact size from a set of cliques. */
function subcliquesOfSize(allCliques: number[][], k: number): number[][] {
  const result: number[][] = [];
  const seen = new Set<string>();

  for (const clique of allCliques) {
    if (clique.length === k) {
      const sorted = [...clique].sort((a, b) => a - b);
      const key = sorted.join(',');
      if (!seen.has(key)) {
        seen.add(key);
        result.push(sorted);
      }
    } else if (clique.length > k) {
      // Generate all combinations of size k
      const combos = combinations(clique.sort((a, b) => a - b), k);
      for (const combo of combos) {
        const key = combo.join(',');
        if (!seen.has(key)) {
          seen.add(key);
          result.push(combo);
        }
      }
    }
  }

  return result;
}

/** Generate all combinations of size k from arr. */
function combinations(arr: number[], k: number): number[][] {
  if (k === 0) return [[]];
  if (arr.length < k) return [];
  const result: number[][] = [];

  function recurse(start: number, current: number[]): void {
    if (current.length === k) {
      result.push([...current]);
      return;
    }
    for (let i = start; i < arr.length; i++) {
      current.push(arr[i]!);
      recurse(i + 1, current);
      current.pop();
    }
  }

  recurse(0, []);
  return result;
}
