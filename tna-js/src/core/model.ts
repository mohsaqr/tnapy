/**
 * TNA model class and build functions.
 * Port of Python tna/model.py
 */
import { Matrix, applyScaling } from './matrix.js';
import { createSeqdata } from './prepare.js';
import { computeTransitions, computeWeightsFromMatrix } from './transitions.js';
import type {
  TNA,
  TNAData,
  ModelType,
  SequenceData,
  BuildModelOptions,
} from './types.js';

/** Create a TNA model object. */
export function createTNA(
  weights: Matrix,
  inits: Float64Array,
  labels: string[],
  data: SequenceData | null = null,
  type: ModelType = 'relative',
  scaling: string[] = [],
): TNA {
  return { weights, inits, labels, data, type, scaling };
}

/** Check if input is a square numeric matrix (not sequence data). */
function isSquareMatrix(data: number[][]): boolean {
  if (data.length === 0) return false;
  return data.length === data[0]!.length;
}

/**
 * Build a TNA model from data.
 *
 * @param x - Input data: sequence data (SequenceData), TNAData, or a square weight matrix (number[][])
 * @param options - Build options
 */
export function buildModel(
  x: SequenceData | TNAData | number[][],
  options?: BuildModelOptions,
): TNA {
  const type = options?.type ?? 'relative';
  const scaling = options?.scaling ?? null;
  let labels = options?.labels;
  const beginState = options?.beginState;
  const endState = options?.endState;
  const params = options?.params;

  // Handle TNAData input
  if (isTNAData(x)) {
    return buildModel(x.sequenceData, { ...options, labels: labels ?? x.labels });
  }

  // Handle direct weight matrix input
  if (isNumericMatrix(x)) {
    if (isSquareMatrix(x)) {
      const mat = Matrix.from2D(x);
      const weights = computeWeightsFromMatrix(mat, type);
      const n = weights.rows;
      const stateLabels = labels ?? Array.from({ length: n }, (_, i) => `S${i + 1}`);
      const inits = new Float64Array(n).fill(1 / n);

      const { weights: scaled, applied } = applyScaling(weights, scaling);

      return createTNA(scaled, inits, stateLabels, null, type, applied);
    }
  }

  // Sequence data input
  const seqData = x as SequenceData;
  const { data: processedData, labels: detectedLabels } = createSeqdata(seqData, {
    beginState,
    endState,
  });

  const stateLabels = labels ?? detectedLabels;

  const { weights, inits } = computeTransitions(processedData, stateLabels, type, params);
  const { weights: scaled, applied } = applyScaling(weights, scaling);

  return createTNA(scaled, inits, stateLabels, processedData, type, applied);
}

/** Build a relative transition probability model. */
export function tna(
  x: SequenceData | TNAData | number[][],
  options?: Omit<BuildModelOptions, 'type' | 'params'>,
): TNA {
  return buildModel(x, { ...options, type: 'relative' });
}

/** Build a frequency-based transition model. */
export function ftna(
  x: SequenceData | TNAData | number[][],
  options?: Omit<BuildModelOptions, 'type' | 'params'>,
): TNA {
  return buildModel(x, { ...options, type: 'frequency' });
}

/** Build a co-occurrence transition model. */
export function ctna(
  x: SequenceData | TNAData | number[][],
  options?: Omit<BuildModelOptions, 'type' | 'params'>,
): TNA {
  return buildModel(x, { ...options, type: 'co-occurrence' });
}

/** Build an attention-weighted transition model. */
export function atna(
  x: SequenceData | TNAData | number[][],
  options?: Omit<BuildModelOptions, 'type'> & { beta?: number },
): TNA {
  return buildModel(x, {
    ...options,
    type: 'attention',
    params: { beta: options?.beta ?? 0.1 },
  });
}

// ---- Helper type guards ----

function isTNAData(x: unknown): x is TNAData {
  return (
    typeof x === 'object' &&
    x !== null &&
    'sequenceData' in x &&
    'labels' in x &&
    'statistics' in x
  );
}

function isNumericMatrix(x: unknown): x is number[][] {
  if (!Array.isArray(x) || x.length === 0) return false;
  const first = x[0];
  if (!Array.isArray(first) || first.length === 0) return false;
  return typeof first[0] === 'number';
}

/** Get a summary of the TNA model. */
export function summary(model: TNA): Record<string, unknown> {
  return {
    nStates: model.labels.length,
    type: model.type,
    scaling: model.scaling,
    nEdges: model.weights.count((v) => v > 0),
    density: model.weights.count((v) => v > 0) / (model.labels.length ** 2),
    meanWeight: model.weights.meanNonZero(),
    maxWeight: model.weights.max(),
    hasSelfLoops: model.weights.diag().some((v) => v > 0),
  };
}
