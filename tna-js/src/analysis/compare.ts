/**
 * Compare sequential patterns across groups.
 * Port of Python tna/compare.py
 */
import type { GroupTNA, CompareRow, SequenceData } from '../core/types.js';
import { SeededRNG } from '../core/rng.js';

/**
 * Compare subsequence patterns across groups.
 */
export function compareSequences(
  x: GroupTNA,
  options?: {
    sub?: number[];
    minFreq?: number;
    test?: boolean;
    iter?: number;
    adjust?: 'bonferroni' | 'holm' | 'fdr' | 'BH' | 'none';
    seed?: number;
  },
): CompareRow[] {
  const groupNames = Object.keys(x.models);
  if (groupNames.length < 2) {
    throw new Error('compare_sequences requires at least 2 groups');
  }

  // Combine data from all groups
  const { data, group } = combineData(x, groupNames);
  const nCols = data[0]?.length ?? 0;

  // Default subsequence lengths
  let sub = options?.sub ?? Array.from({ length: Math.min(5, nCols) }, (_, i) => i + 1);
  sub = sub.filter((s) => s <= nCols);
  if (sub.length === 0) throw new Error('No valid subsequence lengths');

  const minFreq = options?.minFreq ?? 5;
  const test = options?.test ?? false;
  const iter = options?.iter ?? 1000;
  const adjust = options?.adjust ?? 'bonferroni';

  // Extract patterns
  const patternMatrices = extractPatterns(data, sub);

  // Count per group
  const { freq, patternLabels } = factorizePatterns(patternMatrices, group, groupNames);

  // Compute proportions BEFORE filtering (using all patterns per length as denominator)
  const nPatterns = patternLabels.length;
  const nGroups = groupNames.length;
  const props = new Float64Array(nPatterns * nGroups);

  for (const length of sub) {
    for (let p = 0; p < nPatterns; p++) {
      const pLen = patternLabels[p]!.split('->').length;
      if (pLen !== length) continue;

      for (let g = 0; g < nGroups; g++) {
        // Sum all patterns of this length in this group
        let total = 0;
        for (let pp = 0; pp < nPatterns; pp++) {
          if (patternLabels[pp]!.split('->').length === length) {
            total += freq[pp * nGroups + g]!;
          }
        }
        if (total > 0) {
          props[p * nGroups + g] = freq[p * nGroups + g]! / total;
        }
      }
    }
  }

  // Permutation test BEFORE filtering
  let effectSizes: Float64Array | null = null;
  let pValues: Float64Array | null = null;

  if (test) {
    const rng = new SeededRNG(options?.seed ?? 42);
    const result = permutationTestPatterns(
      data, group, groupNames, sub, freq, patternLabels, iter, adjust, rng,
    );
    effectSizes = result.effectSize;
    pValues = result.pValue;
  }

  // Filter by min_freq (minimum across ALL groups)
  const keep: boolean[] = [];
  for (let p = 0; p < nPatterns; p++) {
    let minCount = Infinity;
    for (let g = 0; g < nGroups; g++) {
      const count = freq[p * nGroups + g]!;
      if (count < minCount) minCount = count;
    }
    keep.push(minCount >= minFreq);
  }

  // Build result rows
  const rows: CompareRow[] = [];
  for (let p = 0; p < nPatterns; p++) {
    if (!keep[p]) continue;

    const frequencies: Record<string, number> = {};
    const proportions: Record<string, number> = {};
    for (let g = 0; g < nGroups; g++) {
      frequencies[groupNames[g]!] = freq[p * nGroups + g]!;
      proportions[groupNames[g]!] = props[p * nGroups + g]!;
    }

    const row: CompareRow = {
      pattern: patternLabels[p]!,
      frequencies,
      proportions,
    };

    if (test && effectSizes && pValues) {
      row.effectSize = effectSizes[p]!;
      row.pValue = pValues[p]!;
    }

    rows.push(row);
  }

  // Sort by pattern length then alphabetically
  rows.sort((a, b) => {
    const lenA = a.pattern.split('->').length;
    const lenB = b.pattern.split('->').length;
    if (lenA !== lenB) return lenA - lenB;
    return a.pattern.localeCompare(b.pattern);
  });

  // Sort by p_value if test
  if (test) {
    rows.sort((a, b) => (a.pValue ?? 1) - (b.pValue ?? 1));
  }

  return rows;
}

function combineData(
  g: GroupTNA,
  groupNames: string[],
): { data: SequenceData; group: string[] } {
  const data: SequenceData = [];
  const group: string[] = [];

  let maxCols = 0;
  for (const name of groupNames) {
    const model = g.models[name]!;
    if (!model.data) {
      throw new Error(`Group '${name}' has no sequence data`);
    }
    for (const row of model.data) {
      if (row.length > maxCols) maxCols = row.length;
    }
  }

  for (const name of groupNames) {
    const model = g.models[name]!;
    for (const row of model.data!) {
      // Pad to max length
      const padded = [...row];
      while (padded.length < maxCols) padded.push(null);
      data.push(padded);
      group.push(name);
    }
  }

  return { data, group };
}

function extractPatterns(
  data: SequenceData,
  lengths: number[],
): (string | null)[][] {
  const allPatterns: (string | null)[][] = [];
  const nRows = data.length;
  const nCols = data[0]?.length ?? 0;

  for (const length of lengths) {
    if (length > nCols) break;
    const nPositions = nCols - length + 1;
    const patterns: (string | null)[] = new Array(nRows * nPositions).fill(null);

    for (let i = 0; i < nRows; i++) {
      for (let j = 0; j < nPositions; j++) {
        const subseq = data[i]!.slice(j, j + length);
        if (subseq.some((s) => s === null || s === undefined)) continue;
        patterns[i * nPositions + j] = subseq.join('->');
      }
    }

    allPatterns.push(patterns);
  }

  return allPatterns;
}

function factorizePatterns(
  patternMatrices: (string | null)[][],
  group: string[],
  groupNames: string[],
): { freq: Float64Array; patternLabels: string[] } {
  const patternToIdx = new Map<string, number>();
  const allPatterns: string[] = [];

  for (const pm of patternMatrices) {
    for (const val of pm) {
      if (val !== null && !patternToIdx.has(val)) {
        patternToIdx.set(val, allPatterns.length);
        allPatterns.push(val);
      }
    }
  }

  const nPatterns = allPatterns.length;
  const nGroups = groupNames.length;
  const groupToIdx = new Map<string, number>();
  groupNames.forEach((g, i) => groupToIdx.set(g, i));

  // Flat array: freq[patternIdx * nGroups + groupIdx]
  const freq = new Float64Array(nPatterns * nGroups);
  const nRows = group.length;

  for (const pm of patternMatrices) {
    // Figure out positions per row
    const nPositions = pm.length / nRows;
    for (let i = 0; i < nRows; i++) {
      const gIdx = groupToIdx.get(group[i]!)!;
      for (let j = 0; j < nPositions; j++) {
        const val = pm[i * nPositions + j];
        if (val !== null && val !== undefined) {
          const idx = patternToIdx.get(val)! * nGroups + gIdx;
          freq[idx] = freq[idx]! + 1;
        }
      }
    }
  }

  return { freq, patternLabels: allPatterns };
}

function patternStatistic(freq: Float64Array, nPatterns: number, nGroups: number): Float64Array {
  const stat = new Float64Array(nPatterns);

  // Row sums, col sums, total
  const rowSums = new Float64Array(nPatterns);
  const colSums = new Float64Array(nGroups);
  let total = 0;
  for (let p = 0; p < nPatterns; p++) {
    for (let g = 0; g < nGroups; g++) {
      const v = freq[p * nGroups + g]!;
      rowSums[p] = rowSums[p]! + v;
      colSums[g] = colSums[g]! + v;
      total += v;
    }
  }

  if (total === 0) return stat;

  for (let p = 0; p < nPatterns; p++) {
    let sumSq = 0;
    for (let g = 0; g < nGroups; g++) {
      const expected = (rowSums[p]! * colSums[g]!) / total;
      const diff = freq[p * nGroups + g]! - expected;
      sumSq += diff * diff;
    }
    stat[p] = Math.sqrt(sumSq);
  }

  return stat;
}

function permutationTestPatterns(
  data: SequenceData,
  group: string[],
  groupNames: string[],
  lengths: number[],
  freq: Float64Array,
  patternLabels: string[],
  iter: number,
  adjust: string,
  rng: SeededRNG,
): { effectSize: Float64Array; pValue: Float64Array } {
  const nPatterns = patternLabels.length;
  const nGroups = groupNames.length;
  const groupToIdx = new Map<string, number>();
  groupNames.forEach((g, i) => groupToIdx.set(g, i));

  const trueStat = patternStatistic(freq, nPatterns, nGroups);

  // Pre-compute row patterns
  const nRows = group.length;
  const patternToIdx = new Map<string, number>();
  patternLabels.forEach((p, i) => patternToIdx.set(p, i));

  const rowPatterns: number[][] = [];
  const nCols = data[0]?.length ?? 0;

  for (let i = 0; i < nRows; i++) {
    const pats: number[] = [];
    for (const length of lengths) {
      if (length > nCols) break;
      const nPositions = nCols - length + 1;
      for (let j = 0; j < nPositions; j++) {
        const subseq = data[i]!.slice(j, j + length);
        if (subseq.some((s) => s === null || s === undefined)) continue;
        const pat = subseq.join('->');
        const idx = patternToIdx.get(pat);
        if (idx !== undefined) pats.push(idx);
      }
    }
    rowPatterns.push(pats);
  }

  // Group indices
  const groupIndices = group.map((g) => groupToIdx.get(g)!);

  // Permutation loop
  const permMean = new Float64Array(nPatterns);
  const permM2 = new Float64Array(nPatterns);
  const countGe = new Float64Array(nPatterns);

  for (let it = 0; it < iter; it++) {
    // Permute group indices
    const permGroupIdx = [...groupIndices];
    rng.shuffle(permGroupIdx);

    // Recount
    const permFreq = new Float64Array(nPatterns * nGroups);
    for (let i = 0; i < nRows; i++) {
      const gIdx = permGroupIdx[i]!;
      for (const patIdx of rowPatterns[i]!) {
        const idx = patIdx * nGroups + gIdx;
        permFreq[idx] = permFreq[idx]! + 1;
      }
    }

    const permStat = patternStatistic(permFreq, nPatterns, nGroups);

    // Online mean/variance
    for (let p = 0; p < nPatterns; p++) {
      const delta = permStat[p]! - permMean[p]!;
      permMean[p] = permMean[p]! + delta / (it + 1);
      permM2[p] = permM2[p]! + delta * (permStat[p]! - permMean[p]!);

      if (permStat[p]! >= trueStat[p]!) countGe[p] = countGe[p]! + 1;
    }
  }

  // Effect size
  const effectSize = new Float64Array(nPatterns);
  for (let p = 0; p < nPatterns; p++) {
    const sd = iter > 1 ? Math.sqrt(permM2[p]! / iter) : 0;
    effectSize[p] = sd > 0 ? (trueStat[p]! - permMean[p]!) / sd : 0;
  }

  // P-values per pattern
  const rawP = new Float64Array(nPatterns);
  for (let p = 0; p < nPatterns; p++) {
    rawP[p] = (countGe[p]! + 1) / (iter + 1);
  }

  // Adjust per subsequence length
  const patternLength = patternLabels.map((p) => p.split('->').length);
  const pValue = new Float64Array(nPatterns).fill(1);
  for (const length of lengths) {
    const indices: number[] = [];
    for (let p = 0; p < nPatterns; p++) {
      if (patternLength[p] === length) indices.push(p);
    }
    if (indices.length > 0) {
      const subP = new Float64Array(indices.length);
      for (let i = 0; i < indices.length; i++) subP[i] = rawP[indices[i]!]!;
      const adjusted = pAdjust(subP, adjust);
      for (let i = 0; i < indices.length; i++) pValue[indices[i]!] = adjusted[i]!;
    }
  }

  return { effectSize, pValue };
}

function pAdjust(p: Float64Array, method: string): Float64Array {
  const n = p.length;
  if (n === 0) return new Float64Array(0);

  if (method === 'none') return new Float64Array(p);

  if (method === 'bonferroni') {
    return new Float64Array(p.map((v) => Math.min(v * n, 1)));
  }

  if (method === 'holm') {
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[a]! - p[b]!);
    const adjusted = new Float64Array(n);
    let cummax = 0;
    for (let i = 0; i < n; i++) {
      const val = p[order[i]!]! * (n - i);
      cummax = Math.max(cummax, val);
      adjusted[order[i]!] = Math.min(cummax, 1);
    }
    return adjusted;
  }

  if (method === 'fdr' || method === 'BH') {
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[b]! - p[a]!);
    const adjusted = new Float64Array(n);
    let cummin = 1;
    for (let i = 0; i < n; i++) {
      const val = (p[order[i]!]! * n) / (n - i);
      cummin = Math.min(cummin, val);
      adjusted[order[i]!] = Math.min(cummin, 1);
    }
    return adjusted;
  }

  throw new Error(`Unknown p-value adjustment method: ${method}`);
}
