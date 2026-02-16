/**
 * Bootstrap resampling and permutation tests for TNA.
 * Port of Python tna/bootstrap.py
 */
import { Matrix, arrayStd, arrayQuantile } from '../core/matrix.js';
import { SeededRNG } from '../core/rng.js';
import { computeTransitions3D, computeWeightsFrom3D } from '../core/transitions.js';
import { centralities } from '../analysis/centralities.js';
import type {
  TNA, GroupTNA, BootstrapResult, BootstrapEdge,
  PermutationResult, PermutationEdgeStat,
  CentralityMeasure, CentralityStabilityResult,
} from '../core/types.js';
import { isGroupTNA, groupEntries } from '../core/group.js';

// ---- Bootstrap ----

/**
 * Bootstrap resampling for TNA model stability testing.
 * Matches R TNA's bootstrap.tna function.
 */
export function bootstrapTna(
  model: TNA | GroupTNA,
  options?: {
    iter?: number;
    level?: number;
    method?: 'stability' | 'threshold';
    threshold?: number;
    consistencyRange?: [number, number];
    seed?: number;
  },
): BootstrapResult | Record<string, BootstrapResult> {
  if (isGroupTNA(model)) {
    const result: Record<string, BootstrapResult> = {};
    for (const [name, m] of groupEntries(model)) {
      result[name] = bootstrapTna(m, options) as BootstrapResult;
    }
    return result;
  }

  const tnaModel = model as TNA;
  if (!tnaModel.data) throw new Error('TNA model must have sequence data for bootstrap');

  const iter = options?.iter ?? 1000;
  const level = options?.level ?? 0.05;
  const method = options?.method ?? 'stability';
  const consistencyRange = options?.consistencyRange ?? [0.75, 1.25];
  const rng = new SeededRNG(options?.seed ?? 42);

  const seqData = tnaModel.data;
  const labels = tnaModel.labels;
  const a = labels.length;
  const n = seqData.length;

  // Per-sequence 3D transitions
  const trans = computeTransitions3D(seqData, labels, tnaModel.type);
  const weights = computeWeightsFrom3D(trans, tnaModel.type, tnaModel.scaling.length > 0 ? tnaModel.scaling : undefined);

  // Default threshold
  let threshold = options?.threshold ?? weights.quantile(0.1);

  // Bootstrap loop
  const pValues = Matrix.zeros(a, a);
  const bootWeights: Matrix[] = [];

  for (let i = 0; i < iter; i++) {
    const bootIdx = rng.choice(n, n);
    const transBoot = bootIdx.map((idx) => trans[idx]!);
    const wb = computeWeightsFrom3D(transBoot, tnaModel.type, tnaModel.scaling.length > 0 ? tnaModel.scaling : undefined);
    bootWeights.push(wb);

    if (method === 'stability') {
      for (let r = 0; r < a; r++) {
        for (let c = 0; c < a; c++) {
          if (wb.get(r, c) <= weights.get(r, c) * consistencyRange[0] ||
              wb.get(r, c) >= weights.get(r, c) * consistencyRange[1]) {
            pValues.set(r, c, pValues.get(r, c) + 1);
          }
        }
      }
    } else {
      for (let r = 0; r < a; r++) {
        for (let c = 0; c < a; c++) {
          if (wb.get(r, c) < threshold) {
            pValues.set(r, c, pValues.get(r, c) + 1);
          }
        }
      }
    }
  }

  // P-values: (count + 1) / (iter + 1)
  const pValuesResult = pValues.map((v) => (v + 1) / (iter + 1));

  // Bootstrap statistics
  const weightsMean = Matrix.zeros(a, a);
  const weightsSd = Matrix.zeros(a, a);

  for (let r = 0; r < a; r++) {
    for (let c = 0; c < a; c++) {
      const vals = new Float64Array(iter);
      for (let i = 0; i < iter; i++) vals[i] = bootWeights[i]!.get(r, c);

      let mean = 0;
      for (let i = 0; i < iter; i++) mean += vals[i]!;
      mean /= iter;
      weightsMean.set(r, c, mean);
      weightsSd.set(r, c, arrayStd(vals, 1));
    }
  }

  // Confidence intervals
  const ciLower = Matrix.zeros(a, a);
  const ciUpper = Matrix.zeros(a, a);
  for (let r = 0; r < a; r++) {
    for (let c = 0; c < a; c++) {
      const vals = new Float64Array(iter);
      for (let i = 0; i < iter; i++) vals[i] = bootWeights[i]!.get(r, c);
      ciLower.set(r, c, arrayQuantile(vals, level / 2));
      ciUpper.set(r, c, arrayQuantile(vals, 1 - level / 2));
    }
  }

  // Significant weights
  const weightsSig = Matrix.zeros(a, a);
  for (let r = 0; r < a; r++) {
    for (let c = 0; c < a; c++) {
      if (pValuesResult.get(r, c) < level) {
        weightsSig.set(r, c, weights.get(r, c));
      }
    }
  }

  // Consistency range bounds
  const crLower = weights.scale(consistencyRange[0]);
  const crUpper = weights.scale(consistencyRange[1]);

  // Build summary (column-major order to match R)
  const bootSummary: BootstrapEdge[] = [];
  for (let c = 0; c < a; c++) {
    for (let r = 0; r < a; r++) {
      if (weights.get(r, c) > 0) {
        bootSummary.push({
          from: labels[r]!,
          to: labels[c]!,
          weight: weights.get(r, c),
          pValue: pValuesResult.get(r, c),
          sig: pValuesResult.get(r, c) < level,
          crLower: crLower.get(r, c),
          crUpper: crUpper.get(r, c),
          ciLower: ciLower.get(r, c),
          ciUpper: ciUpper.get(r, c),
        });
      }
    }
  }

  // Pruned model
  const prunedModel: TNA = {
    weights: weightsSig,
    inits: new Float64Array(tnaModel.inits),
    labels: [...labels],
    data: seqData,
    type: tnaModel.type,
    scaling: [...tnaModel.scaling],
  };

  return {
    weightsOrig: weights,
    weightsSig,
    weightsMean,
    weightsSd,
    pValues: pValuesResult,
    crLower,
    crUpper,
    ciLower,
    ciUpper,
    bootSummary,
    model: prunedModel,
    labels: [...labels],
    method,
    iter,
    level,
  };
}

// ---- Permutation Test ----

/**
 * Permutation test for comparing two TNA models.
 * Matches R TNA's permutation_test function.
 */
export function permutationTest(
  x: TNA,
  y: TNA,
  options?: {
    iter?: number;
    adjust?: string;
    paired?: boolean;
    level?: number;
    measures?: CentralityMeasure[];
    seed?: number;
  },
): PermutationResult {
  if (!x.data || !y.data) {
    throw new Error('Both TNA models must have sequence data');
  }

  const iter = options?.iter ?? 1000;
  const adjust = options?.adjust ?? 'none';
  const level = options?.level ?? 0.05;
  const rng = new SeededRNG(options?.seed ?? 42);

  const dataX = x.data;
  const dataY = y.data;
  const nX = dataX.length;
  const nY = dataY.length;
  const labels = x.labels;
  const a = labels.length;

  // Combine data
  const combinedData = [...dataX, ...dataY];
  const nXY = nX + nY;

  const weightsX = x.weights;
  const weightsY = y.weights;

  // True edge differences
  const edgeDiffsTrue = weightsX.sub(weightsY);
  const edgeDiffsTrueAbs = edgeDiffsTrue.map(Math.abs);

  // Combined per-sequence transitions
  const combinedTrans = computeTransitions3D(combinedData, labels, x.type);

  // Permutation loop
  const edgePValues = Matrix.zeros(a, a);
  const edgeDiffsSumSq = Matrix.zeros(a, a);
  const edgeDiffsMean = Matrix.zeros(a, a);

  for (let i = 0; i < iter; i++) {
    const permIdx = rng.permutation(nXY);
    const transPermX = permIdx.slice(0, nX).map((idx) => combinedTrans[idx]!);
    const transPermY = permIdx.slice(nX).map((idx) => combinedTrans[idx]!);

    const wPermX = computeWeightsFrom3D(transPermX, x.type, x.scaling.length > 0 ? x.scaling : undefined);
    const wPermY = computeWeightsFrom3D(transPermY, y.type, y.scaling.length > 0 ? y.scaling : undefined);

    const permDiff = wPermX.sub(wPermY);

    for (let r = 0; r < a; r++) {
      for (let c = 0; c < a; c++) {
        if (Math.abs(permDiff.get(r, c)) >= edgeDiffsTrueAbs.get(r, c)) {
          edgePValues.set(r, c, edgePValues.get(r, c) + 1);
        }

        // Online mean/variance
        const delta = permDiff.get(r, c) - edgeDiffsMean.get(r, c);
        edgeDiffsMean.set(r, c, edgeDiffsMean.get(r, c) + delta / (i + 1));
        edgeDiffsSumSq.set(r, c, edgeDiffsSumSq.get(r, c) + delta * (permDiff.get(r, c) - edgeDiffsMean.get(r, c)));
      }
    }
  }

  // P-values
  const edgePValuesFloat = edgePValues.map((v) => (v + 1) / (iter + 1));

  // Apply adjustment
  const flatP = edgePValuesFloat.flattenColMajor();
  const adjustedP = pAdjust(flatP, adjust);
  const edgePAdjusted = Matrix.zeros(a, a);
  let idx = 0;
  for (let c = 0; c < a; c++) {
    for (let r = 0; r < a; r++) {
      edgePAdjusted.set(r, c, adjustedP[idx++]!);
    }
  }

  // Effect sizes
  const edgeEffectSize = Matrix.zeros(a, a);
  for (let r = 0; r < a; r++) {
    for (let c = 0; c < a; c++) {
      const sd = iter > 1 ? Math.sqrt(edgeDiffsSumSq.get(r, c) / (iter - 1)) : 0;
      edgeEffectSize.set(r, c, sd > 0 ? edgeDiffsTrue.get(r, c) / sd : NaN);
    }
  }

  // Significant differences
  const edgeDiffsSig = Matrix.zeros(a, a);
  for (let r = 0; r < a; r++) {
    for (let c = 0; c < a; c++) {
      if (edgePAdjusted.get(r, c) < level) {
        edgeDiffsSig.set(r, c, edgeDiffsTrue.get(r, c));
      }
    }
  }

  // Build edge stats (column-major to match R)
  const edgeStats: PermutationEdgeStat[] = [];
  for (let c = 0; c < a; c++) {
    for (let r = 0; r < a; r++) {
      edgeStats.push({
        edgeName: `${labels[r]} -> ${labels[c]}`,
        diffTrue: edgeDiffsTrue.get(r, c),
        effectSize: edgeEffectSize.get(r, c),
        pValue: edgePAdjusted.get(r, c),
      });
    }
  }

  return {
    edges: {
      stats: edgeStats,
      diffsTrue: edgeDiffsTrue,
      diffsSig: edgeDiffsSig,
    },
    labels: [...labels],
  };
}

// ---- Centrality Stability ----

/**
 * Estimate centrality stability using case-dropping bootstrap.
 */
export function estimateCs(
  model: TNA | GroupTNA,
  options?: {
    loops?: boolean;
    normalize?: boolean;
    measures?: CentralityMeasure[];
    iter?: number;
    method?: 'pearson' | 'spearman';
    dropProp?: number[];
    threshold?: number;
    certainty?: number;
    seed?: number;
  },
): CentralityStabilityResult | Record<string, CentralityStabilityResult> {
  if (isGroupTNA(model)) {
    const result: Record<string, CentralityStabilityResult> = {};
    for (const [name, m] of groupEntries(model)) {
      result[name] = estimateCs(m, options) as CentralityStabilityResult;
    }
    return result;
  }

  const tnaModel = model as TNA;
  if (!tnaModel.data) throw new Error('TNA model must have sequence data');

  const measuresToUse = options?.measures ?? ['InStrength', 'OutStrength', 'Betweenness'] as CentralityMeasure[];
  const iter = options?.iter ?? 1000;
  const dropProp = options?.dropProp ?? [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
  const corThreshold = options?.threshold ?? 0.7;
  const certainty = options?.certainty ?? 0.95;
  const rng = new SeededRNG(options?.seed ?? 42);

  const seqData = tnaModel.data;
  const labels = tnaModel.labels;
  const n = seqData.length;

  // Per-sequence 3D transitions
  const trans = computeTransitions3D(seqData, labels, tnaModel.type);

  // Original centralities
  const origCent = centralities(tnaModel, {
    measures: measuresToUse,
    loops: options?.loops,
    normalize: options?.normalize,
  });

  // Case-dropping bootstrap
  const correlations: Record<string, Float64Array[]> = {};
  for (const m of measuresToUse) {
    correlations[m] = dropProp.map(() => new Float64Array(iter));
  }

  for (let j = 0; j < dropProp.length; j++) {
    const dp = dropProp[j]!;
    const nDrop = Math.floor(n * dp);
    const nKeep = n - nDrop;

    if (nDrop === 0) {
      for (const m of measuresToUse) {
        correlations[m]![j]!.fill(NaN);
      }
      continue;
    }

    for (let i = 0; i < iter; i++) {
      const keepIdx = rng.choiceWithoutReplacement(n, nKeep);
      const transSub = keepIdx.map((idx) => trans[idx]!);
      const weightsSub = computeWeightsFrom3D(transSub, tnaModel.type, tnaModel.scaling.length > 0 ? tnaModel.scaling : undefined);

      const subModel: TNA = {
        weights: weightsSub,
        inits: new Float64Array(tnaModel.inits),
        labels: [...labels],
        data: null,
        type: tnaModel.type,
        scaling: [...tnaModel.scaling],
      };

      const subCent = centralities(subModel, {
        measures: measuresToUse,
        loops: options?.loops,
        normalize: options?.normalize,
      });

      for (const m of measuresToUse) {
        const origVals = origCent.measures[m]!;
        const subVals = subCent.measures[m]!;
        correlations[m]![j]![i] = pearsonCorrArr(origVals, subVals);
      }
    }
  }

  // Calculate CS coefficients
  const csCoefficients: Record<string, number> = {};
  for (const m of measuresToUse) {
    csCoefficients[m] = calculateCs(correlations[m]!, dropProp, corThreshold, certainty);
  }

  return {
    csCoefficients,
    correlations,
    dropProp,
    threshold: corThreshold,
    certainty,
  };
}

function calculateCs(
  correlations: Float64Array[],
  dropProp: number[],
  threshold: number,
  certainty: number,
): number {
  let lastValid = 0;
  for (let j = 0; j < dropProp.length; j++) {
    const corrs = correlations[j]!;
    let countAbove = 0;
    let countValid = 0;
    for (let i = 0; i < corrs.length; i++) {
      if (!isNaN(corrs[i]!)) {
        countValid++;
        if (corrs[i]! >= threshold) countAbove++;
      }
    }
    const propAbove = countValid > 0 ? countAbove / countValid : 0;
    if (propAbove >= certainty) {
      lastValid = dropProp[j]!;
    }
  }
  return lastValid;
}

function pearsonCorrArr(a: Float64Array, b: Float64Array): number {
  if (a.length !== b.length || a.length < 2) return NaN;
  let meanA = 0, meanB = 0;
  for (let i = 0; i < a.length; i++) {
    meanA += a[i]!;
    meanB += b[i]!;
  }
  meanA /= a.length;
  meanB /= b.length;

  let num = 0, denA = 0, denB = 0;
  for (let i = 0; i < a.length; i++) {
    const da = a[i]! - meanA;
    const db = b[i]! - meanB;
    num += da * db;
    denA += da * da;
    denB += db * db;
  }
  const den = Math.sqrt(denA * denB);
  return den === 0 ? NaN : num / den;
}

// ---- P-value adjustment ----

function pAdjust(p: Float64Array, method: string): Float64Array {
  const n = p.length;
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

  return new Float64Array(p);
}

// ---- Confidence Interval methods ----

/**
 * Calculate confidence interval from bootstrap distribution.
 */
export function confidenceInterval(
  values: Float64Array,
  ci = 0.95,
  method: 'percentile' | 'basic' = 'percentile',
): [number, number] {
  const alpha = 1 - ci;

  if (method === 'percentile') {
    return [
      arrayQuantile(values, alpha / 2),
      arrayQuantile(values, 1 - alpha / 2),
    ];
  }

  // Basic bootstrap
  let mean = 0;
  for (let i = 0; i < values.length; i++) mean += values[i]!;
  mean /= values.length;

  const lowerPct = arrayQuantile(values, 1 - alpha / 2);
  const upperPct = arrayQuantile(values, alpha / 2);

  return [2 * mean - lowerPct, 2 * mean - upperPct];
}
