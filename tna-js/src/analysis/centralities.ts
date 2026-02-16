/**
 * Centrality measures for TNA.
 * Port of Python tna/centralities.py
 *
 * Uses hand-rolled graph algorithms instead of graphology to match
 * the exact R TNA behavior (igraph with 1/weight distances).
 */
import { Matrix } from '../core/matrix.js';
import type { TNA, GroupTNA, CentralityMeasure, CentralityResult } from '../core/types.js';
import { isGroupTNA, groupEntries } from '../core/group.js';

export const AVAILABLE_MEASURES: CentralityMeasure[] = [
  'OutStrength', 'InStrength', 'ClosenessIn', 'ClosenessOut',
  'Closeness', 'Betweenness', 'BetweennessRSP', 'Diffusion', 'Clustering',
];

/**
 * Compute centrality measures for a TNA model.
 */
export function centralities(
  model: TNA | GroupTNA,
  options?: {
    loops?: boolean;
    normalize?: boolean;
    measures?: CentralityMeasure[];
  },
): CentralityResult {
  // Handle GroupTNA
  if (isGroupTNA(model)) {
    const allLabels: string[] = [];
    const allGroups: string[] = [];
    const allMeasures: Record<CentralityMeasure, number[]> = {} as Record<CentralityMeasure, number[]>;

    for (const [name, m] of groupEntries(model)) {
      const result = centralities(m, options);
      for (let i = 0; i < result.labels.length; i++) {
        allLabels.push(result.labels[i]!);
        allGroups.push(name);
      }
      for (const [measure, values] of Object.entries(result.measures) as [CentralityMeasure, Float64Array][]) {
        if (!allMeasures[measure]) allMeasures[measure] = [];
        for (let i = 0; i < values.length; i++) {
          allMeasures[measure]!.push(values[i]!);
        }
      }
    }

    const measures: Record<CentralityMeasure, Float64Array> = {} as Record<CentralityMeasure, Float64Array>;
    for (const [m, vals] of Object.entries(allMeasures) as [CentralityMeasure, number[]][]) {
      measures[m] = new Float64Array(vals);
    }

    return { labels: allLabels, measures, groups: allGroups };
  }

  const tnaModel = model as TNA;
  const requestedMeasures = options?.measures ?? [...AVAILABLE_MEASURES];
  const loops = options?.loops ?? false;
  const normalize = options?.normalize ?? false;

  const weights = tnaModel.weights.clone();
  const n = weights.rows;

  // Remove self-loops if not requested
  if (!loops) {
    for (let i = 0; i < n; i++) weights.set(i, i, 0);
  }

  const measures: Record<string, Float64Array> = {};

  for (const measure of AVAILABLE_MEASURES) {
    if (!requestedMeasures.includes(measure)) continue;

    switch (measure) {
      case 'OutStrength':
        measures.OutStrength = outStrength(weights);
        break;
      case 'InStrength':
        measures.InStrength = inStrength(weights);
        break;
      case 'ClosenessIn':
        measures.ClosenessIn = closenessIn(weights, n);
        break;
      case 'ClosenessOut':
        measures.ClosenessOut = closenessOut(weights, n);
        break;
      case 'Closeness':
        measures.Closeness = closenessAll(weights, n);
        break;
      case 'Betweenness':
        measures.Betweenness = betweenness(weights, n);
        break;
      case 'BetweennessRSP':
        measures.BetweennessRSP = betweennessRSP(weights);
        break;
      case 'Diffusion':
        measures.Diffusion = diffusion(weights);
        break;
      case 'Clustering':
        measures.Clustering = clustering(weights);
        break;
    }
  }

  // Normalize if requested (min-max per measure)
  if (normalize) {
    for (const key of Object.keys(measures)) {
      const vals = measures[key]!;
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < vals.length; i++) {
        if (vals[i]! < min) min = vals[i]!;
        if (vals[i]! > max) max = vals[i]!;
      }
      if (max > min) {
        for (let i = 0; i < vals.length; i++) {
          vals[i] = (vals[i]! - min) / (max - min);
        }
      } else {
        vals.fill(0);
      }
    }
  }

  return {
    labels: tnaModel.labels,
    measures: measures as Record<CentralityMeasure, Float64Array>,
  };
}

// ---- Strength ----

function outStrength(weights: Matrix): Float64Array {
  return weights.rowSums();
}

function inStrength(weights: Matrix): Float64Array {
  return weights.colSums();
}

// ---- Dijkstra shortest paths with 1/weight distances ----

/** Dijkstra from source, using 1/weight as distance. Returns distances to all nodes. */
function dijkstra(n: number, getWeight: (from: number, to: number) => number, source: number): Float64Array {
  const dist = new Float64Array(n).fill(Infinity);
  const visited = new Uint8Array(n);
  dist[source] = 0;

  for (let step = 0; step < n; step++) {
    // Find unvisited node with min distance
    let u = -1;
    let minDist = Infinity;
    for (let i = 0; i < n; i++) {
      if (!visited[i] && dist[i]! < minDist) {
        minDist = dist[i]!;
        u = i;
      }
    }
    if (u === -1) break;
    visited[u] = 1;

    for (let v = 0; v < n; v++) {
      if (visited[v]) continue;
      const w = getWeight(u, v);
      if (w > 0) {
        const d = 1 / w; // inverse weight as distance
        const newDist = dist[u]! + d;
        if (newDist < dist[v]!) {
          dist[v] = newDist;
        }
      }
    }
  }

  return dist;
}

// ---- Closeness ----

function closenessIn(weights: Matrix, n: number): Float64Array {
  const result = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    // Reverse graph: paths TO node i
    const dist = dijkstra(n, (from, to) => weights.get(to, from), i);

    let totalDist = 0;
    for (let j = 0; j < n; j++) {
      if (j !== i && isFinite(dist[j]!)) totalDist += dist[j]!;
    }
    result[i] = totalDist > 0 ? 1 / totalDist : 0;
  }

  return result;
}

function closenessOut(weights: Matrix, n: number): Float64Array {
  const result = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const dist = dijkstra(n, (from, to) => weights.get(from, to), i);

    let totalDist = 0;
    for (let j = 0; j < n; j++) {
      if (j !== i && isFinite(dist[j]!)) totalDist += dist[j]!;
    }
    result[i] = totalDist > 0 ? 1 / totalDist : 0;
  }

  return result;
}

function closenessAll(weights: Matrix, n: number): Float64Array {
  // Mode="all" — undirected with max weight per edge pair
  const symWeights = Matrix.zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const w = Math.max(weights.get(i, j), weights.get(j, i));
      symWeights.set(i, j, w);
      symWeights.set(j, i, w);
    }
  }

  const result = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const dist = dijkstra(n, (from, to) => symWeights.get(from, to), i);

    let totalDist = 0;
    for (let j = 0; j < n; j++) {
      if (j !== i && isFinite(dist[j]!)) totalDist += dist[j]!;
    }
    result[i] = totalDist > 0 ? 1 / totalDist : 0;
  }

  return result;
}

// ---- Betweenness (Brandes' algorithm with 1/weight distances) ----

function betweenness(weights: Matrix, n: number): Float64Array {
  const CB = new Float64Array(n);

  for (let s = 0; s < n; s++) {
    // Modified Dijkstra
    const stack: number[] = [];
    const pred: number[][] = Array.from({ length: n }, () => []);
    const sigma = new Float64Array(n);
    const dist = new Float64Array(n).fill(Infinity);

    sigma[s] = 1;
    dist[s] = 0;

    // Priority queue (simple linear scan for small n)
    const visited = new Uint8Array(n);

    for (let step = 0; step < n; step++) {
      let u = -1;
      let minDist = Infinity;
      for (let i = 0; i < n; i++) {
        if (!visited[i] && dist[i]! < minDist) {
          minDist = dist[i]!;
          u = i;
        }
      }
      if (u === -1) break;
      visited[u] = 1;
      stack.push(u);

      for (let v = 0; v < n; v++) {
        if (visited[v]) continue;
        const w = weights.get(u, v);
        if (w <= 0) continue;
        const d = 1 / w;
        const newDist = dist[u]! + d;

        if (newDist < dist[v]! - 1e-15) {
          dist[v] = newDist;
          sigma[v] = sigma[u]!;
          pred[v] = [u];
        } else if (Math.abs(newDist - dist[v]!) < 1e-15) {
          sigma[v] = sigma[v]! + sigma[u]!;
          pred[v]!.push(u);
        }
      }
    }

    // Back-propagation
    const delta = new Float64Array(n);
    while (stack.length > 0) {
      const w = stack.pop()!;
      for (const v of pred[w]!) {
        const frac = (sigma[v]! / sigma[w]!) * (1 + delta[w]!);
        delta[v] = delta[v]! + frac;
      }
      if (w !== s) {
        CB[w] = CB[w]! + delta[w]!;
      }
    }
  }

  return CB;
}

// ---- Randomized Shortest Path Betweenness (Kivimaki et al. 2016) ----

function betweennessRSP(weights: Matrix, beta = 0.01): Float64Array {
  const n = weights.rows;
  const mat = weights.clone();

  // D <- rowSums(mat)
  const D = mat.rowSums();

  // if (any(D == 0)) return NA
  for (let i = 0; i < n; i++) {
    if (D[i] === 0) return new Float64Array(n).fill(NaN);
  }

  // P_ref <- diag(D^-1) %*% mat
  const Pref = Matrix.zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      Pref.set(i, j, mat.get(i, j) / D[i]!);
    }
  }

  // C <- mat^-1; C[is.infinite(C)] <- 0
  const C = mat.map((v) => (v === 0 ? 0 : 1 / v));

  // W <- P_ref * exp(-beta * C)
  const W = Matrix.zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      W.set(i, j, Pref.get(i, j) * Math.exp(-beta * C.get(i, j)));
    }
  }

  // Z <- solve(I - W)
  const IminusW = Matrix.eye(n).sub(W);
  let Z: Matrix;
  try {
    Z = IminusW.inverse();
  } catch {
    return new Float64Array(n).fill(NaN);
  }

  // Z_recip <- Z^-1; Z_recip[is.infinite(Z_recip)] <- 0
  const Zrecip = Z.map((v) => {
    if (v === 0) return 0;
    const inv = 1 / v;
    return isFinite(inv) ? inv : 0;
  });

  // Z_recip_diag <- diag(Z_recip) * I
  const ZrecipDiag = Matrix.diag(Zrecip.diag());

  // out <- diag(Z %*% (Z_recip - n * Z_recip_diag)^T %*% Z)
  // = diag(Z @ (Zrecip - n*ZrecipDiag).T @ Z)
  const inner = Zrecip.sub(ZrecipDiag.scale(n));
  const step1 = Z.matmul(inner.transpose());
  const step2 = step1.matmul(Z);

  const out = step2.diag();

  // Round
  for (let i = 0; i < n; i++) {
    out[i] = Math.round(out[i]!);
  }

  // out = out - min(out) + 1
  let minVal = Infinity;
  for (let i = 0; i < n; i++) {
    if (out[i]! < minVal) minVal = out[i]!;
  }
  for (let i = 0; i < n; i++) {
    out[i] = out[i]! - minVal + 1;
  }

  return out;
}

// ---- Diffusion centrality (Banerjee et al. 2014) ----

function diffusion(weights: Matrix): Float64Array {
  const n = weights.rows;
  let s = Matrix.zeros(n, n);
  let p = Matrix.eye(n);

  for (let i = 0; i < n; i++) {
    p = p.matmul(weights);
    s = s.add(p);
  }

  return s.rowSums();
}

// ---- Clustering coefficient (Zhang and Horvath 2005) ----

function clustering(weights: Matrix): Float64Array {
  // Symmetrize: mat = weights + weights.T
  const mat = weights.add(weights.transpose());

  // diag(mat) <- 0
  for (let i = 0; i < mat.rows; i++) mat.set(i, i, 0);

  const n = mat.rows;

  // num = diag(mat %*% mat %*% mat)
  const mat2 = mat.matmul(mat);
  const mat3 = mat2.matmul(mat);
  const num = mat3.diag();

  // den = colSums(mat)^2 - colSums(mat^2)
  const colSums = mat.colSums();
  const matSq = mat.map((v) => v * v);
  const colSumsSq = matSq.colSums();

  const result = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const den = colSums[i]! * colSums[i]! - colSumsSq[i]!;
    result[i] = den !== 0 ? num[i]! / den : 0;
  }

  return result;
}

/**
 * Compute edge betweenness centrality and return as a new TNA model.
 */
export function betweennessNetwork(model: TNA | GroupTNA): TNA | Record<string, TNA> {
  if (isGroupTNA(model)) {
    const result: Record<string, TNA> = {};
    for (const [name, m] of groupEntries(model)) {
      result[name] = betweennessNetwork(m) as TNA;
    }
    return result;
  }

  const tnaModel = model as TNA;
  const weights = tnaModel.weights;
  const n = weights.rows;

  // Use Brandes' algorithm but collect edge betweenness
  const edgeBet = Matrix.zeros(n, n);

  for (let s = 0; s < n; s++) {
    const stack: number[] = [];
    const pred: number[][] = Array.from({ length: n }, () => []);
    const sigma = new Float64Array(n);
    const dist = new Float64Array(n).fill(Infinity);

    sigma[s] = 1;
    dist[s] = 0;

    const visited = new Uint8Array(n);

    for (let step = 0; step < n; step++) {
      let u = -1;
      let minDist = Infinity;
      for (let i = 0; i < n; i++) {
        if (!visited[i] && dist[i]! < minDist) {
          minDist = dist[i]!;
          u = i;
        }
      }
      if (u === -1) break;
      visited[u] = 1;
      stack.push(u);

      for (let v = 0; v < n; v++) {
        if (visited[v]) continue;
        const w = weights.get(u, v);
        if (w <= 0) continue;
        const d = 1 / w;
        const newDist = dist[u]! + d;

        if (newDist < dist[v]! - 1e-15) {
          dist[v] = newDist;
          sigma[v] = sigma[u]!;
          pred[v] = [u];
        } else if (Math.abs(newDist - dist[v]!) < 1e-15) {
          sigma[v] = sigma[v]! + sigma[u]!;
          pred[v]!.push(u);
        }
      }
    }

    const delta = new Float64Array(n);
    while (stack.length > 0) {
      const w = stack.pop()!;
      for (const v of pred[w]!) {
        const frac = (sigma[v]! / sigma[w]!) * (1 + delta[w]!);
        delta[v] = delta[v]! + frac;
        edgeBet.set(v, w, edgeBet.get(v, w) + sigma[v]! / sigma[w]! * (1 + delta[w]!));
      }
    }
  }

  return {
    weights: edgeBet,
    inits: new Float64Array(tnaModel.inits),
    labels: [...tnaModel.labels],
    data: tnaModel.data,
    type: 'betweenness',
    scaling: [...tnaModel.scaling],
  };
}
