/**
 * Transition computation algorithms.
 * Port of Python tna/transitions.py
 */
import { Matrix, rowNormalize, applyScaling } from './matrix.js';
import type { ModelType, SequenceData, TransitionParams } from './types.js';

/** Check if a value is null/undefined/NaN. */
function isNA(val: string | null | undefined): boolean {
  return val === null || val === undefined || val === '';
}

/** Get list of (position, state) for non-NA values in a sequence row. */
function getValidTransitions(row: (string | null)[]): { pos: number; state: string }[] {
  const result: { pos: number; state: string }[] = [];
  for (let i = 0; i < row.length; i++) {
    const val = row[i];
    if (!isNA(val)) {
      result.push({ pos: i, state: val! });
    }
  }
  return result;
}

/**
 * Compute transition matrix and initial probabilities from sequence data.
 */
export function computeTransitions(
  data: SequenceData,
  states: string[],
  type: ModelType = 'relative',
  params?: TransitionParams,
): { weights: Matrix; inits: Float64Array } {
  const nStates = states.length;
  const stateToIdx = new Map<string, number>();
  states.forEach((s, i) => stateToIdx.set(s, i));

  switch (type) {
    case 'relative':
      return transitionsRelative(data, stateToIdx, nStates);
    case 'frequency':
      return transitionsFrequency(data, stateToIdx, nStates);
    case 'co-occurrence':
      return transitionsCooccurrence(data, stateToIdx, nStates);
    case 'reverse':
      return transitionsReverse(data, stateToIdx, nStates);
    case 'n-gram':
      return transitionsNgram(data, stateToIdx, nStates, params?.n ?? 2);
    case 'gap':
      return transitionsGap(data, stateToIdx, nStates, params?.maxGap ?? 5, params?.decay ?? 0.5);
    case 'window':
      return transitionsWindow(data, stateToIdx, nStates, params?.size ?? 3);
    case 'attention':
      return transitionsAttention(data, stateToIdx, nStates, params?.beta ?? 0.1);
    default:
      throw new Error(`Unknown transition type: ${type}`);
  }
}

function transitionsRelative(
  data: SequenceData,
  stateToIdx: Map<string, number>,
  nStates: number,
): { weights: Matrix; inits: Float64Array } {
  const counts = Matrix.zeros(nStates, nStates);
  const inits = new Float64Array(nStates);

  for (const row of data) {
    const valid = getValidTransitions(row);
    if (valid.length === 0) continue;

    const firstIdx = stateToIdx.get(valid[0]!.state);
    if (firstIdx !== undefined) inits[firstIdx]!++;

    for (let i = 0; i < valid.length - 1; i++) {
      const fromIdx = stateToIdx.get(valid[i]!.state);
      const toIdx = stateToIdx.get(valid[i + 1]!.state);
      if (fromIdx !== undefined && toIdx !== undefined) {
        counts.set(fromIdx, toIdx, counts.get(fromIdx, toIdx) + 1);
      }
    }
  }

  const weights = rowNormalize(counts);
  const initSum = inits.reduce((a, b) => a + b, 0);
  if (initSum > 0) {
    for (let i = 0; i < inits.length; i++) inits[i]! /= initSum;
  }

  return { weights, inits };
}

function transitionsFrequency(
  data: SequenceData,
  stateToIdx: Map<string, number>,
  nStates: number,
): { weights: Matrix; inits: Float64Array } {
  const counts = Matrix.zeros(nStates, nStates);
  const inits = new Float64Array(nStates);

  for (const row of data) {
    const valid = getValidTransitions(row);
    if (valid.length === 0) continue;

    const firstIdx = stateToIdx.get(valid[0]!.state);
    if (firstIdx !== undefined) inits[firstIdx]!++;

    for (let i = 0; i < valid.length - 1; i++) {
      const fromIdx = stateToIdx.get(valid[i]!.state);
      const toIdx = stateToIdx.get(valid[i + 1]!.state);
      if (fromIdx !== undefined && toIdx !== undefined) {
        counts.set(fromIdx, toIdx, counts.get(fromIdx, toIdx) + 1);
      }
    }
  }

  const initSum = inits.reduce((a, b) => a + b, 0);
  if (initSum > 0) {
    for (let i = 0; i < inits.length; i++) inits[i]! /= initSum;
  }

  return { weights: counts, inits };
}

function transitionsCooccurrence(
  data: SequenceData,
  stateToIdx: Map<string, number>,
  nStates: number,
): { weights: Matrix; inits: Float64Array } {
  const counts = Matrix.zeros(nStates, nStates);
  const inits = new Float64Array(nStates);

  for (const row of data) {
    const valid = getValidTransitions(row);
    if (valid.length === 0) continue;

    const firstIdx = stateToIdx.get(valid[0]!.state);
    if (firstIdx !== undefined) inits[firstIdx]!++;

    for (let i = 0; i < valid.length - 1; i++) {
      const idx1 = stateToIdx.get(valid[i]!.state);
      const idx2 = stateToIdx.get(valid[i + 1]!.state);
      if (idx1 !== undefined && idx2 !== undefined) {
        counts.set(idx1, idx2, counts.get(idx1, idx2) + 1);
        counts.set(idx2, idx1, counts.get(idx2, idx1) + 1);
      }
    }
  }

  const weights = rowNormalize(counts);
  const initSum = inits.reduce((a, b) => a + b, 0);
  if (initSum > 0) {
    for (let i = 0; i < inits.length; i++) inits[i]! /= initSum;
  }

  return { weights, inits };
}

function transitionsReverse(
  data: SequenceData,
  stateToIdx: Map<string, number>,
  nStates: number,
): { weights: Matrix; inits: Float64Array } {
  const counts = Matrix.zeros(nStates, nStates);
  const inits = new Float64Array(nStates);

  for (const row of data) {
    const valid = getValidTransitions(row);
    if (valid.length === 0) continue;

    const lastIdx = stateToIdx.get(valid[valid.length - 1]!.state);
    if (lastIdx !== undefined) inits[lastIdx]!++;

    for (let i = valid.length - 1; i > 0; i--) {
      const fromIdx = stateToIdx.get(valid[i]!.state);
      const toIdx = stateToIdx.get(valid[i - 1]!.state);
      if (fromIdx !== undefined && toIdx !== undefined) {
        counts.set(fromIdx, toIdx, counts.get(fromIdx, toIdx) + 1);
      }
    }
  }

  const weights = rowNormalize(counts);
  const initSum = inits.reduce((a, b) => a + b, 0);
  if (initSum > 0) {
    for (let i = 0; i < inits.length; i++) inits[i]! /= initSum;
  }

  return { weights, inits };
}

function transitionsNgram(
  data: SequenceData,
  stateToIdx: Map<string, number>,
  nStates: number,
  n: number,
): { weights: Matrix; inits: Float64Array } {
  const counts = Matrix.zeros(nStates, nStates);
  const inits = new Float64Array(nStates);

  for (const row of data) {
    const valid = getValidTransitions(row);
    if (valid.length === 0) continue;

    const firstIdx = stateToIdx.get(valid[0]!.state);
    if (firstIdx !== undefined) inits[firstIdx]!++;

    for (let i = 0; i <= valid.length - n; i++) {
      const fromIdx = stateToIdx.get(valid[i]!.state);
      const toIdx = stateToIdx.get(valid[i + n - 1]!.state);
      if (fromIdx !== undefined && toIdx !== undefined) {
        counts.set(fromIdx, toIdx, counts.get(fromIdx, toIdx) + 1);
      }
    }
  }

  const weights = rowNormalize(counts);
  const initSum = inits.reduce((a, b) => a + b, 0);
  if (initSum > 0) {
    for (let i = 0; i < inits.length; i++) inits[i]! /= initSum;
  }

  return { weights, inits };
}

function transitionsGap(
  data: SequenceData,
  stateToIdx: Map<string, number>,
  nStates: number,
  maxGap: number,
  decay: number,
): { weights: Matrix; inits: Float64Array } {
  const counts = Matrix.zeros(nStates, nStates);
  const inits = new Float64Array(nStates);

  for (const row of data) {
    const valid = getValidTransitions(row);
    if (valid.length === 0) continue;

    const firstIdx = stateToIdx.get(valid[0]!.state);
    if (firstIdx !== undefined) inits[firstIdx]!++;

    for (let i = 0; i < valid.length; i++) {
      const fromIdx = stateToIdx.get(valid[i]!.state);
      if (fromIdx === undefined) continue;
      for (let j = i + 1; j < Math.min(i + maxGap + 1, valid.length); j++) {
        const toIdx = stateToIdx.get(valid[j]!.state);
        if (toIdx === undefined) continue;
        const gap = j - i;
        const weight = Math.pow(decay, gap - 1);
        counts.set(fromIdx, toIdx, counts.get(fromIdx, toIdx) + weight);
      }
    }
  }

  const weights = rowNormalize(counts);
  const initSum = inits.reduce((a, b) => a + b, 0);
  if (initSum > 0) {
    for (let i = 0; i < inits.length; i++) inits[i]! /= initSum;
  }

  return { weights, inits };
}

function transitionsWindow(
  data: SequenceData,
  stateToIdx: Map<string, number>,
  nStates: number,
  size: number,
): { weights: Matrix; inits: Float64Array } {
  const counts = Matrix.zeros(nStates, nStates);
  const inits = new Float64Array(nStates);

  for (const row of data) {
    const valid = getValidTransitions(row);
    if (valid.length === 0) continue;

    const firstIdx = stateToIdx.get(valid[0]!.state);
    if (firstIdx !== undefined) inits[firstIdx]!++;

    for (let wStart = 0; wStart <= valid.length - size; wStart++) {
      const window = valid.slice(wStart, wStart + size);
      for (let i = 0; i < window.length; i++) {
        for (let j = i + 1; j < window.length; j++) {
          const idx1 = stateToIdx.get(window[i]!.state);
          const idx2 = stateToIdx.get(window[j]!.state);
          if (idx1 !== undefined && idx2 !== undefined) {
            counts.set(idx1, idx2, counts.get(idx1, idx2) + 1);
          }
        }
      }
    }
  }

  const weights = rowNormalize(counts);
  const initSum = inits.reduce((a, b) => a + b, 0);
  if (initSum > 0) {
    for (let i = 0; i < inits.length; i++) inits[i]! /= initSum;
  }

  return { weights, inits };
}

function transitionsAttention(
  data: SequenceData,
  stateToIdx: Map<string, number>,
  nStates: number,
  beta: number,
): { weights: Matrix; inits: Float64Array } {
  const counts = Matrix.zeros(nStates, nStates);
  const inits = new Float64Array(nStates);

  for (const row of data) {
    const valid = getValidTransitions(row);
    if (valid.length === 0) continue;

    const firstIdx = stateToIdx.get(valid[0]!.state);
    if (firstIdx !== undefined) inits[firstIdx]!++;

    for (let i = 0; i < valid.length; i++) {
      const fromIdx = stateToIdx.get(valid[i]!.state);
      if (fromIdx === undefined) continue;
      for (let j = i + 1; j < valid.length; j++) {
        const toIdx = stateToIdx.get(valid[j]!.state);
        if (toIdx === undefined) continue;
        const distance = j - i;
        const weight = Math.exp(-beta * distance);
        counts.set(fromIdx, toIdx, counts.get(fromIdx, toIdx) + weight);
      }
    }
  }

  const weights = rowNormalize(counts);
  const initSum = inits.reduce((a, b) => a + b, 0);
  if (initSum > 0) {
    for (let i = 0; i < inits.length; i++) inits[i]! /= initSum;
  }

  return { weights, inits };
}

/**
 * Compute per-sequence transition counts as a 3D array.
 * Returns array of matrices of shape (nStates x nStates),
 * one per sequence, where mat[i][j] = count of i->j transitions in that sequence.
 *
 * Matches R TNA's compute_transitions function.
 */
export function computeTransitions3D(
  data: SequenceData,
  states: string[],
  type: ModelType = 'relative',
): Matrix[] {
  const nSequences = data.length;
  const nStates = states.length;
  const stateToIdx = new Map<string, number>();
  states.forEach((s, i) => stateToIdx.set(s, i));

  const trans: Matrix[] = [];
  for (let r = 0; r < nSequences; r++) {
    trans.push(Matrix.zeros(nStates, nStates));
  }

  const nCols = data.length > 0 ? data[0]!.length : 0;

  if (type === 'relative' || type === 'frequency') {
    for (let col = 0; col < nCols - 1; col++) {
      for (let row = 0; row < nSequences; row++) {
        const fromVal = data[row]![col];
        const toVal = data[row]![col + 1];
        if (isNA(fromVal) || isNA(toVal)) continue;
        const fromIdx = stateToIdx.get(fromVal!);
        const toIdx = stateToIdx.get(toVal!);
        if (fromIdx !== undefined && toIdx !== undefined) {
          trans[row]!.set(fromIdx, toIdx, trans[row]!.get(fromIdx, toIdx) + 1);
        }
      }
    }
  } else if (type === 'reverse') {
    for (let col = 0; col < nCols - 1; col++) {
      for (let row = 0; row < nSequences; row++) {
        const fromVal = data[row]![col + 1];
        const toVal = data[row]![col];
        if (isNA(fromVal) || isNA(toVal)) continue;
        const fromIdx = stateToIdx.get(fromVal!);
        const toIdx = stateToIdx.get(toVal!);
        if (fromIdx !== undefined && toIdx !== undefined) {
          trans[row]!.set(fromIdx, toIdx, trans[row]!.get(fromIdx, toIdx) + 1);
        }
      }
    }
  } else if (type === 'co-occurrence') {
    for (let i = 0; i < nCols - 1; i++) {
      for (let j = i + 1; j < nCols; j++) {
        for (let row = 0; row < nSequences; row++) {
          const fromVal = data[row]![i];
          const toVal = data[row]![j];
          if (isNA(fromVal) || isNA(toVal)) continue;
          const fi = stateToIdx.get(fromVal!);
          const ti = stateToIdx.get(toVal!);
          if (fi !== undefined && ti !== undefined) {
            trans[row]!.set(fi, ti, trans[row]!.get(fi, ti) + 1);
            trans[row]!.set(ti, fi, trans[row]!.get(ti, fi) + 1);
          }
        }
      }
    }
  }

  return trans;
}

/**
 * Compute weight matrix from array of per-sequence transition matrices.
 * Sums over sequences, then row-normalizes for 'relative' type.
 */
export function computeWeightsFrom3D(
  transitions: Matrix[],
  type: ModelType = 'relative',
  scaling?: string | string[] | null,
): Matrix {
  if (transitions.length === 0) {
    throw new Error('No transition matrices provided');
  }

  const n = transitions[0]!.rows;
  const weights = Matrix.zeros(n, n);

  // Sum over sequences
  for (const t of transitions) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        weights.set(i, j, weights.get(i, j) + t.get(i, j));
      }
    }
  }

  // Row normalize for relative type
  let result = type === 'relative' ? rowNormalize(weights) : weights;

  // Apply additional scaling
  if (scaling) {
    const scaled = applyScaling(result, scaling);
    result = scaled.weights;
  }

  return result;
}

/** Process an existing weight/count matrix. */
export function computeWeightsFromMatrix(
  mat: Matrix,
  type: ModelType = 'relative',
): Matrix {
  if (type === 'relative') return rowNormalize(mat);
  return mat.clone();
}
