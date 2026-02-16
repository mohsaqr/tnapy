/**
 * Sequence clustering functions.
 * Port of Python tna/cluster.py
 */
import { Matrix } from '../core/matrix.js';
import type { SequenceData, ClusterResult, TNAData } from '../core/types.js';

const SENTINEL = '\0__NA__';

// ---- Distance functions ----

function toTokenLists(data: SequenceData, naSyms = ['*', '%']): string[][] {
  const naSet = new Set(naSyms);
  return data.map((row) =>
    row.map((val) => {
      if (val === null || val === undefined || val === '') return SENTINEL;
      if (naSet.has(val)) return SENTINEL;
      return val;
    }),
  );
}

function hammingDistance(
  a: string[], b: string[],
  weighted = false, lambda_ = 1,
): number {
  const maxLen = Math.max(a.length, b.length);
  const aPad = [...a, ...new Array(maxLen - a.length).fill(SENTINEL)];
  const bPad = [...b, ...new Array(maxLen - b.length).fill(SENTINEL)];

  let dist = 0;
  for (let i = 0; i < maxLen; i++) {
    if (aPad[i] !== bPad[i]) {
      dist += weighted ? Math.exp(-lambda_ * i) : 1;
    }
  }
  return dist;
}

function levenshteinDistance(a: string[], b: string[]): number {
  const m = a.length;
  const n = b.length;
  let prev = Array.from({ length: n + 1 }, (_, i) => i);
  let curr = new Array(n + 1).fill(0);

  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(
        prev[j]! + 1,
        curr[j - 1]! + 1,
        prev[j - 1]! + cost,
      );
    }
    [prev, curr] = [curr, prev];
  }
  return prev[n]!;
}

function osaDistance(a: string[], b: string[]): number {
  const m = a.length;
  const n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;

  const d: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) d[i]![0] = i;
  for (let j = 0; j <= n; j++) d[0]![j] = j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      d[i]![j] = Math.min(
        d[i - 1]![j]! + 1,
        d[i]![j - 1]! + 1,
        d[i - 1]![j - 1]! + cost,
      );
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) {
        d[i]![j] = Math.min(d[i]![j]!, d[i - 2]![j - 2]! + cost);
      }
    }
  }
  return d[m]![n]!;
}

function lcsDistance(a: string[], b: string[]): number {
  const m = a.length;
  const n = b.length;
  let prev = new Array(n + 1).fill(0);
  let curr = new Array(n + 1).fill(0);

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        curr[j] = prev[j - 1]! + 1;
      } else {
        curr[j] = Math.max(prev[j]!, curr[j - 1]!);
      }
    }
    [prev, curr] = [curr, new Array(n + 1).fill(0)];
  }
  return Math.max(m, n) - prev[n]!;
}

type DistFunc = (a: string[], b: string[]) => number;

const DISTANCE_FUNCS: Record<string, DistFunc> = {
  hamming: (a, b) => hammingDistance(a, b),
  lv: levenshteinDistance,
  osa: osaDistance,
  lcs: lcsDistance,
};

function computeDistanceMatrix(
  sequences: string[][],
  dissimilarity: string,
  weighted = false,
  lambda_ = 1,
): Matrix {
  const n = sequences.length;
  const dist = Matrix.zeros(n, n);

  if (dissimilarity === 'hamming') {
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = hammingDistance(sequences[i]!, sequences[j]!, weighted, lambda_);
        dist.set(i, j, d);
        dist.set(j, i, d);
      }
    }
  } else {
    const func = DISTANCE_FUNCS[dissimilarity];
    if (!func) throw new Error(`Unknown dissimilarity: ${dissimilarity}`);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = func(sequences[i]!, sequences[j]!);
        dist.set(i, j, d);
        dist.set(j, i, d);
      }
    }
  }

  return dist;
}

// ---- Silhouette ----

function silhouetteScore(dist: Matrix, labels: number[]): number {
  const n = labels.length;
  const uniqueClusters = [...new Set(labels)];
  if (uniqueClusters.length < 2) return 0;

  let totalScore = 0;
  for (let i = 0; i < n; i++) {
    const ci = labels[i]!;

    // a(i): mean distance to same-cluster members
    let sumSame = 0;
    let countSame = 0;
    for (let j = 0; j < n; j++) {
      if (j !== i && labels[j] === ci) {
        sumSame += dist.get(i, j);
        countSame++;
      }
    }
    if (countSame === 0) continue;
    const ai = sumSame / countSame;

    // b(i): min over other clusters of mean distance
    let bi = Infinity;
    for (const c of uniqueClusters) {
      if (c === ci) continue;
      let sumOther = 0;
      let countOther = 0;
      for (let j = 0; j < n; j++) {
        if (labels[j] === c) {
          sumOther += dist.get(i, j);
          countOther++;
        }
      }
      if (countOther > 0) {
        bi = Math.min(bi, sumOther / countOther);
      }
    }

    const maxAB = Math.max(ai, bi);
    totalScore += maxAB > 0 ? (bi - ai) / maxAB : 0;
  }

  return totalScore / n;
}

// ---- PAM (Partitioning Around Medoids) ----

function pam(dist: Matrix, k: number): number[] {
  const n = dist.rows;

  // BUILD phase
  const medoids: number[] = [];
  const totalDists = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) totalDists[i] = totalDists[i]! + dist.get(i, j);
  }

  let bestIdx = 0;
  for (let i = 1; i < n; i++) {
    if (totalDists[i]! < totalDists[bestIdx]!) bestIdx = i;
  }
  medoids.push(bestIdx);

  const nearestDist = new Float64Array(n);
  for (let i = 0; i < n; i++) nearestDist[i] = dist.get(i, medoids[0]!);

  for (let m = 1; m < k; m++) {
    let bestGain = -Infinity;
    let bestCandidate = -1;
    for (let c = 0; c < n; c++) {
      if (medoids.includes(c)) continue;
      let gain = 0;
      for (let i = 0; i < n; i++) {
        gain += Math.max(0, nearestDist[i]! - dist.get(i, c));
      }
      if (gain > bestGain) {
        bestGain = gain;
        bestCandidate = c;
      }
    }
    medoids.push(bestCandidate);
    for (let i = 0; i < n; i++) {
      nearestDist[i] = Math.min(nearestDist[i]!, dist.get(i, bestCandidate));
    }
  }

  // SWAP phase
  const medoidsArr = [...medoids];
  for (let iter = 0; iter < 100; iter++) {
    let improved = false;
    for (let mIdx = 0; mIdx < k; mIdx++) {
      // Current cost
      let currentCost = 0;
      for (let i = 0; i < n; i++) {
        let minD = Infinity;
        for (const m of medoidsArr) minD = Math.min(minD, dist.get(i, m));
        currentCost += minD;
      }

      let bestSwapCost = currentCost;
      let bestSwap = -1;

      for (let c = 0; c < n; c++) {
        if (medoidsArr.includes(c)) continue;
        const trial = [...medoidsArr];
        trial[mIdx] = c;
        let trialCost = 0;
        for (let i = 0; i < n; i++) {
          let minD = Infinity;
          for (const m of trial) minD = Math.min(minD, dist.get(i, m));
          trialCost += minD;
        }
        if (trialCost < bestSwapCost) {
          bestSwapCost = trialCost;
          bestSwap = c;
        }
      }

      if (bestSwap >= 0) {
        medoidsArr[mIdx] = bestSwap;
        improved = true;
      }
    }
    if (!improved) break;
  }

  // Assign (1-indexed)
  return Array.from({ length: n }, (_, i) => {
    let minD = Infinity;
    let bestM = 0;
    for (let m = 0; m < k; m++) {
      const d = dist.get(i, medoidsArr[m]!);
      if (d < minD) {
        minD = d;
        bestM = m;
      }
    }
    return bestM + 1;
  });
}

// ---- Hierarchical clustering ----

/** Simple hierarchical clustering (Ward's method). */
function hierarchical(dist: Matrix, k: number, _method: string): number[] {
  const n = dist.rows;

  // Initialize: each point is its own cluster
  const clusters: number[][] = Array.from({ length: n }, (_, i) => [i]);
  const active = new Set<number>(Array.from({ length: n }, (_, i) => i));

  while (active.size > k) {
    // Find closest pair of clusters
    let bestDist = Infinity;
    let bestI = -1;
    let bestJ = -1;

    const activeArr = [...active];
    for (let a = 0; a < activeArr.length; a++) {
      for (let b = a + 1; b < activeArr.length; b++) {
        const ci = activeArr[a]!;
        const cj = activeArr[b]!;
        // Ward's distance: increase in total within-cluster variance
        const d = clusterDistance(dist, clusters[ci]!, clusters[cj]!);
        if (d < bestDist) {
          bestDist = d;
          bestI = ci;
          bestJ = cj;
        }
      }
    }

    // Merge bestJ into bestI
    clusters[bestI] = [...clusters[bestI]!, ...clusters[bestJ]!];
    active.delete(bestJ);
  }

  // Assign cluster labels (1-indexed)
  const assignments = new Array(n).fill(0);
  let clusterIdx = 1;
  for (const ci of active) {
    for (const point of clusters[ci]!) {
      assignments[point] = clusterIdx;
    }
    clusterIdx++;
  }

  return assignments;
}

function clusterDistance(dist: Matrix, a: number[], b: number[]): number {
  // Average linkage distance between clusters
  let sum = 0;
  for (const i of a) {
    for (const j of b) {
      sum += dist.get(i, j);
    }
  }
  return sum / (a.length * b.length);
}

// ---- Main function ----

/**
 * Cluster sequences using distance-based methods.
 */
export function clusterSequences(
  data: SequenceData | TNAData,
  k: number,
  options?: {
    dissimilarity?: 'hamming' | 'lv' | 'osa' | 'lcs';
    method?: string;
    naSyms?: string[];
    weighted?: boolean;
    lambda?: number;
  },
): ClusterResult {
  // Handle TNAData
  const seqData = 'sequenceData' in (data as TNAData) ? (data as TNAData).sequenceData : data as SequenceData;

  const dissimilarity = options?.dissimilarity ?? 'hamming';
  const method = options?.method ?? 'pam';
  const naSyms = options?.naSyms ?? ['*', '%'];
  const weighted = options?.weighted ?? false;
  const lambda_ = options?.lambda ?? 1;

  if (k < 2) throw new Error('k must be >= 2');
  if (k > seqData.length) throw new Error(`k=${k} exceeds number of sequences (${seqData.length})`);

  // Convert to token lists
  const sequences = toTokenLists(seqData, naSyms);

  // Compute distance matrix
  const dist = computeDistanceMatrix(sequences, dissimilarity, weighted, lambda_);

  // Cluster
  let assignments: number[];
  if (method === 'pam') {
    assignments = pam(dist, k);
  } else {
    assignments = hierarchical(dist, k, method);
  }

  // Silhouette
  const sil = silhouetteScore(dist, assignments);

  // Cluster sizes
  const sizes: number[] = [];
  for (let c = 1; c <= k; c++) {
    sizes.push(assignments.filter((a) => a === c).length);
  }

  return {
    data: seqData,
    k,
    assignments,
    silhouette: sil,
    sizes,
    method,
    distance: dist,
    dissimilarity,
  };
}
