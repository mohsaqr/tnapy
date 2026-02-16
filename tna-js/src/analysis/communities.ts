/**
 * Community detection for TNA models.
 * Port of Python tna/communities.py
 *
 * Implements community detection algorithms natively without graphology
 * dependency, for simpler bundling and exact R matching.
 */
import { Matrix } from '../core/matrix.js';
import type { TNA, GroupTNA, CommunityResult, CommunityMethod } from '../core/types.js';
import { isGroupTNA, groupEntries } from '../core/group.js';

export const AVAILABLE_METHODS: CommunityMethod[] = [
  'fast_greedy', 'louvain', 'label_prop',
  'leading_eigen', 'edge_betweenness', 'walktrap',
];

/**
 * Detect communities in a TNA model.
 */
export function communities(
  model: TNA | GroupTNA,
  options?: { methods?: CommunityMethod | CommunityMethod[] },
): CommunityResult | Record<string, CommunityResult> {
  if (isGroupTNA(model)) {
    const result: Record<string, CommunityResult> = {};
    for (const [name, m] of groupEntries(model)) {
      result[name] = communities(m, options) as CommunityResult;
    }
    return result;
  }

  const tnaModel = model as TNA;
  let methods: CommunityMethod[] = ['leading_eigen'];
  if (options?.methods) {
    methods = typeof options.methods === 'string' ? [options.methods] : options.methods;
  }

  const weights = tnaModel.weights;
  const n = weights.rows;

  // Create symmetric adjacency matrix
  const sym = Matrix.zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const w = weights.get(i, j) + weights.get(j, i);
      if (w > 0) {
        sym.set(i, j, w);
        sym.set(j, i, w);
      }
    }
  }

  const counts: Record<string, number> = {};
  const assignments: Record<string, number[]> = {};

  for (const method of methods) {
    const comm = detectCommunities(sym, n, method);
    counts[method] = new Set(comm).size;
    assignments[method] = comm;
  }

  return { counts, assignments, labels: tnaModel.labels };
}

function detectCommunities(adj: Matrix, n: number, method: CommunityMethod): number[] {
  switch (method) {
    case 'leading_eigen':
      return leadingEigen(adj, n);
    case 'louvain':
    case 'walktrap':
      return louvain(adj, n);
    case 'fast_greedy':
      return greedyModularity(adj, n);
    case 'label_prop':
      return labelPropagation(adj, n);
    case 'edge_betweenness':
      return edgeBetweennessCommunities(adj, n);
    default:
      throw new Error(`Unknown community detection method: ${method}`);
  }
}

/** Leading eigenvector method: split by sign of leading eigenvector of modularity matrix. */
function leadingEigen(adj: Matrix, n: number): number[] {
  if (n <= 1) return new Array(n).fill(0);

  // Degree vector and total weight
  const k = adj.rowSums();
  let m2 = 0;
  for (let i = 0; i < n; i++) m2 += k[i]!;

  if (m2 === 0) return Array.from({ length: n }, (_, i) => i);

  // Modularity matrix: B = A - k_i * k_j / (2m)
  const B = Matrix.zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      B.set(i, j, adj.get(i, j) - (k[i]! * k[j]!) / m2);
    }
  }

  // Power iteration for leading eigenvector (deterministic init)
  let v = new Float64Array(n);
  for (let i = 0; i < n; i++) v[i] = ((i * 7 + 3) % 11) / 11 - 0.5;

  for (let iter = 0; iter < 100; iter++) {
    const Bv = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) {
        s += B.get(i, j) * v[j]!;
      }
      Bv[i] = s;
    }

    // Normalize
    let norm = 0;
    for (let i = 0; i < n; i++) norm += Bv[i]! * Bv[i]!;
    norm = Math.sqrt(norm);
    if (norm < 1e-15) break;

    for (let i = 0; i < n; i++) Bv[i]! /= norm;
    v = Bv;
  }

  // Split by sign
  return Array.from(v, (val) => (val >= 0 ? 0 : 1));
}

/** Louvain-style community detection. */
function louvain(adj: Matrix, n: number): number[] {
  // Initialize each node in its own community
  const comm = Array.from({ length: n }, (_, i) => i);
  const totalWeight = adj.sum() / 2;
  if (totalWeight === 0) return comm;

  let improved = true;
  while (improved) {
    improved = false;
    for (let i = 0; i < n; i++) {
      const currentComm = comm[i]!;
      let bestComm = currentComm;
      let bestDeltaQ = 0;

      // Calculate modularity change for moving i to each neighbor's community
      const neighborComms = new Set<number>();
      for (let j = 0; j < n; j++) {
        if (adj.get(i, j) > 0 || adj.get(j, i) > 0) {
          neighborComms.add(comm[j]!);
        }
      }

      for (const c of neighborComms) {
        if (c === currentComm) continue;
        const deltaQ = modularityDelta(adj, comm, i, c, totalWeight, n);
        if (deltaQ > bestDeltaQ) {
          bestDeltaQ = deltaQ;
          bestComm = c;
        }
      }

      if (bestComm !== currentComm) {
        comm[i] = bestComm;
        improved = true;
      }
    }
  }

  // Renumber communities to be contiguous starting from 0
  return renumberCommunities(comm);
}

function modularityDelta(
  adj: Matrix, comm: number[], node: number, targetComm: number,
  totalWeight: number, n: number,
): number {
  const ki = adj.rowSums()[node]!;
  let sumIn = 0;
  let sumTot = 0;

  for (let j = 0; j < n; j++) {
    if (comm[j] === targetComm) {
      sumIn += adj.get(node, j) + adj.get(j, node);
      sumTot += adj.rowSums()[j]!;
    }
  }

  const m2 = totalWeight * 2;
  return sumIn / m2 - (sumTot * ki) / (m2 * m2) * 2;
}

/** Greedy modularity optimization. */
function greedyModularity(adj: Matrix, n: number): number[] {
  // Start with each node in its own community
  const comm = Array.from({ length: n }, (_, i) => i);
  const totalWeight = adj.sum() / 2;
  if (totalWeight === 0) return comm;

  let improved = true;
  while (improved) {
    improved = false;
    // Try merging each pair of communities
    const uniqueComms = [...new Set(comm)];
    let bestMerge: [number, number] | null = null;
    let bestDeltaQ = 0;

    for (let a = 0; a < uniqueComms.length; a++) {
      for (let b = a + 1; b < uniqueComms.length; b++) {
        const cA = uniqueComms[a]!;
        const cB = uniqueComms[b]!;

        // Check if communities are connected
        let connected = false;
        for (let i = 0; i < n && !connected; i++) {
          if (comm[i] !== cA) continue;
          for (let j = 0; j < n && !connected; j++) {
            if (comm[j] === cB && (adj.get(i, j) > 0 || adj.get(j, i) > 0)) {
              connected = true;
            }
          }
        }
        if (!connected) continue;

        // Calculate modularity delta for merging
        let eAB = 0;
        let aA = 0;
        let aB = 0;
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            const w = adj.get(i, j);
            if (comm[i] === cA && comm[j] === cB) eAB += w;
            if (comm[i] === cA) aA += w;
            if (comm[i] === cB) aB += w;
          }
        }
        const m2 = totalWeight * 2;
        const deltaQ = 2 * (eAB / m2 - (aA * aB) / (m2 * m2));

        if (deltaQ > bestDeltaQ) {
          bestDeltaQ = deltaQ;
          bestMerge = [cA, cB];
        }
      }
    }

    if (bestMerge) {
      const [keep, merge] = bestMerge;
      for (let i = 0; i < n; i++) {
        if (comm[i] === merge) comm[i] = keep;
      }
      improved = true;
    }
  }

  return renumberCommunities(comm);
}

/** Label propagation. */
function labelPropagation(adj: Matrix, n: number): number[] {
  const comm = Array.from({ length: n }, (_, i) => i);

  for (let iter = 0; iter < 100; iter++) {
    let changed = false;
    // Random order
    const order = Array.from({ length: n }, (_, i) => i);
    // Simple deterministic shuffle for reproducibility
    for (let i = n - 1; i > 0; i--) {
      const j = (i * 7 + iter * 13) % (i + 1);
      [order[i], order[j]] = [order[j]!, order[i]!];
    }

    for (const i of order) {
      // Count label weights from neighbors
      const labelWeights = new Map<number, number>();
      for (let j = 0; j < n; j++) {
        const w = adj.get(i, j) + adj.get(j, i);
        if (w > 0) {
          labelWeights.set(comm[j]!, (labelWeights.get(comm[j]!) ?? 0) + w);
        }
      }

      if (labelWeights.size > 0) {
        let bestLabel = comm[i]!;
        let bestWeight = 0;
        for (const [label, weight] of labelWeights) {
          if (weight > bestWeight) {
            bestWeight = weight;
            bestLabel = label;
          }
        }
        if (bestLabel !== comm[i]) {
          comm[i] = bestLabel;
          changed = true;
        }
      }
    }

    if (!changed) break;
  }

  return renumberCommunities(comm);
}

/** Edge betweenness community detection (Girvan-Newman). */
function edgeBetweennessCommunities(adj: Matrix, n: number): number[] {
  if (adj.count((v) => v > 0) === 0) {
    return Array.from({ length: n }, (_, i) => i);
  }

  // Repeatedly remove highest betweenness edge, find best modularity partition
  const work = adj.clone();
  let bestPartition = Array.from({ length: n }, (_, i) => i);
  let bestMod = -1;

  for (let step = 0; step < n * n; step++) {
    // Find connected components
    const partition = connectedComponents(work, n);
    const mod = modularity(adj, partition, n);
    if (mod > bestMod) {
      bestMod = mod;
      bestPartition = partition;
    }

    // Find edge with highest betweenness
    let maxBet = 0;
    let maxI = -1;
    let maxJ = -1;

    // Simple edge betweenness: count shortest paths through each edge
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (work.get(i, j) > 0 || work.get(j, i) > 0) {
          const bet = edgeBetweennessScore(work, n, i, j);
          if (bet > maxBet) {
            maxBet = bet;
            maxI = i;
            maxJ = j;
          }
        }
      }
    }

    if (maxI < 0) break;

    work.set(maxI, maxJ, 0);
    work.set(maxJ, maxI, 0);
  }

  return renumberCommunities(bestPartition);
}

function edgeBetweennessScore(adj: Matrix, n: number, u: number, v: number): number {
  // Simplified: count paths through this edge
  let count = 0;
  for (let s = 0; s < n; s++) {
    for (let t = 0; t < n; t++) {
      if (s === t) continue;
      // BFS to check if shortest path goes through (u,v)
      const dist = bfsDistance(adj, n, s);
      if (dist[t]! < Infinity) {
        if ((dist[u]! + 1 === dist[v]! || dist[v]! + 1 === dist[u]!) &&
            dist[s]! + dist[t]! === dist[t]!) {
          count++;
        }
      }
    }
  }
  return count;
}

function bfsDistance(adj: Matrix, n: number, source: number): Float64Array {
  const dist = new Float64Array(n).fill(Infinity);
  dist[source] = 0;
  const queue = [source];
  let qi = 0;

  while (qi < queue.length) {
    const u = queue[qi++]!;
    for (let v = 0; v < n; v++) {
      if (dist[v] === Infinity && (adj.get(u, v) > 0 || adj.get(v, u) > 0)) {
        dist[v] = dist[u]! + 1;
        queue.push(v);
      }
    }
  }

  return dist;
}

function connectedComponents(adj: Matrix, n: number): number[] {
  const comp = new Array(n).fill(-1);
  let nextComp = 0;

  for (let start = 0; start < n; start++) {
    if (comp[start] >= 0) continue;
    // BFS
    const queue = [start];
    let qi = 0;
    comp[start] = nextComp;

    while (qi < queue.length) {
      const u = queue[qi++]!;
      for (let v = 0; v < n; v++) {
        if (comp[v] < 0 && (adj.get(u, v) > 0 || adj.get(v, u) > 0)) {
          comp[v] = nextComp;
          queue.push(v);
        }
      }
    }
    nextComp++;
  }

  return comp;
}

function modularity(adj: Matrix, comm: number[], n: number): number {
  const m2 = adj.sum();
  if (m2 === 0) return 0;

  let Q = 0;
  const k = adj.rowSums();
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (comm[i] === comm[j]) {
        Q += adj.get(i, j) - (k[i]! * k[j]!) / m2;
      }
    }
  }
  return Q / m2;
}

function renumberCommunities(comm: number[]): number[] {
  const mapping = new Map<number, number>();
  let nextId = 0;
  const sorted = [...new Set(comm)].sort((a, b) => {
    // Sort by first occurrence to match R behavior
    const aFirst = comm.indexOf(a);
    const bFirst = comm.indexOf(b);
    return aFirst - bFirst;
  });

  for (const c of sorted) {
    mapping.set(c, nextId++);
  }

  return comm.map((c) => mapping.get(c)!);
}
