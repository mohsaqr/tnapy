import { describe, it, expect } from 'vitest';
import { buildModel, tna, ftna, ctna, atna, summary } from '../src/core/model.js';
import { createSeqdata, prepareData } from '../src/core/prepare.js';
import { groupTna, groupFtna, isGroupTNA, groupNames } from '../src/core/group.js';
import type { SequenceData } from '../src/core/types.js';
import fixture from './fixtures/ground_truth.json';

const smallData: SequenceData = fixture.small_data;
const expectedLabels = fixture.labels;
const expectedWeights = fixture.tna_weights;
const expectedInits = fixture.tna_inits;
const expectedFtnaWeights = fixture.ftna_weights;

const TOL = 1e-10;

function matClose(actual: number[][], expected: number[][], tol = TOL): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]!.length).toBe(expected[i]!.length);
    for (let j = 0; j < actual[i]!.length; j++) {
      expect(actual[i]![j]).toBeCloseTo(expected[i]![j]!, -Math.log10(tol));
    }
  }
}

describe('createSeqdata', () => {
  it('extracts unique sorted labels', () => {
    const data: SequenceData = [
      ['B', 'A', 'C'],
      ['A', 'C', null],
    ];
    const { labels } = createSeqdata(data);
    expect(labels).toEqual(['A', 'B', 'C']);
  });

  it('handles begin/end states', () => {
    const data: SequenceData = [['A', 'B']];
    const { data: result, labels } = createSeqdata(data, {
      beginState: 'START',
      endState: 'END',
    });
    expect(labels).toEqual(['START', 'A', 'B', 'END']);
    expect(result[0]).toEqual(['START', 'A', 'B', 'END']);
  });
});

describe('prepareData', () => {
  it('computes statistics', () => {
    const data: SequenceData = [
      ['A', 'B', 'C'],
      ['B', 'C', null],
    ];
    const prepared = prepareData(data);
    expect(prepared.labels).toEqual(['A', 'B', 'C']);
    expect(prepared.statistics.nSessions).toBe(2);
    expect(prepared.statistics.nUniqueActions).toBe(3);
    expect(prepared.statistics.maxSequenceLength).toBe(3);
  });
});

describe('TNA model building', () => {
  it('builds TNA model with correct labels', () => {
    const model = tna(smallData);
    expect(model.labels).toEqual(expectedLabels);
    expect(model.type).toBe('relative');
  });

  it('produces correct transition weights (R ground truth)', () => {
    const model = tna(smallData);
    matClose(model.weights.to2D(), expectedWeights);
  });

  it('produces correct initial probabilities (R ground truth)', () => {
    const model = tna(smallData);
    for (let i = 0; i < expectedInits.length; i++) {
      expect(model.inits[i]).toBeCloseTo(expectedInits[i]!, 10);
    }
  });

  it('builds FTNA model with correct weights (R ground truth)', () => {
    const model = ftna(smallData);
    matClose(model.weights.to2D(), expectedFtnaWeights);
    expect(model.type).toBe('frequency');
  });

  it('builds from explicit matrix', () => {
    const mat = fixture.mat_input;
    const model = tna(mat);
    matClose(model.weights.to2D(), fixture.mat_model_weights);
  });

  it('builds with minmax scaling', () => {
    const model = buildModel(smallData, { scaling: ['minmax'] });
    matClose(model.weights.to2D(), fixture.scaled_minmax_weights);
    expect(model.scaling).toEqual(['minmax']);
  });

  it('model row sums equal 1 for relative type', () => {
    const model = tna(smallData);
    const rowSums = model.weights.rowSums();
    for (let i = 0; i < rowSums.length; i++) {
      // Row sums should be ~1.0 (or 0 for zero rows)
      if (rowSums[i]! > 0) {
        expect(rowSums[i]).toBeCloseTo(1.0, 10);
      }
    }
  });
});

describe('CTNA model', () => {
  it('builds co-occurrence model', () => {
    const model = ctna(smallData);
    expect(model.type).toBe('co-occurrence');
    // Symmetric: w(i,j) should relate to w(j,i)
    const w = model.weights;
    // Co-occurrence is row-normalized, so not directly symmetric
    // but the underlying counts are symmetric
    expect(w.rows).toBe(expectedLabels.length);
  });
});

describe('ATNA model', () => {
  it('builds attention model', () => {
    const model = atna(smallData);
    expect(model.type).toBe('attention');
    expect(model.weights.rows).toBe(expectedLabels.length);
    // Row sums should be ~1.0
    const rowSums = model.weights.rowSums();
    for (let i = 0; i < rowSums.length; i++) {
      if (rowSums[i]! > 0) {
        expect(rowSums[i]).toBeCloseTo(1.0, 8);
      }
    }
  });
});

describe('summary', () => {
  it('returns model summary', () => {
    const model = tna(smallData);
    const s = summary(model);
    expect(s.nStates).toBe(9);
    expect(s.type).toBe('relative');
    expect(typeof s.nEdges).toBe('number');
    expect(typeof s.density).toBe('number');
    expect(typeof s.meanWeight).toBe('number');
    expect(typeof s.maxWeight).toBe('number');
    expect(typeof s.hasSelfLoops).toBe('boolean');
  });
});

describe('GroupTNA', () => {
  it('builds group models', () => {
    // Split data into two groups
    const groups = smallData.map((_, i) => (i < 10 ? 'A' : 'B'));
    const gmodel = groupTna(smallData, groups);
    expect(isGroupTNA(gmodel)).toBe(true);
    expect(groupNames(gmodel)).toEqual(['A', 'B']);
  });

  it('group models have same labels', () => {
    const groups = smallData.map((_, i) => (i < 10 ? 'A' : 'B'));
    const gmodel = groupTna(smallData, groups);
    const modelA = gmodel.models['A']!;
    const modelB = gmodel.models['B']!;
    expect(modelA.labels).toEqual(modelB.labels);
  });

  it('builds group FTNA models', () => {
    const groups = smallData.map((_, i) => (i < 10 ? 'A' : 'B'));
    const gmodel = groupFtna(smallData, groups);
    expect(gmodel.models['A']!.type).toBe('frequency');
    expect(gmodel.models['B']!.type).toBe('frequency');
  });
});
