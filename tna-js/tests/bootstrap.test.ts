import { describe, it, expect } from 'vitest';
import { bootstrapTna, permutationTest, confidenceInterval } from '../src/stats/bootstrap.js';
import { tna } from '../src/core/model.js';
import { groupTna } from '../src/core/group.js';
import type { SequenceData, BootstrapResult } from '../src/core/types.js';
import fixture from './fixtures/ground_truth.json';

// Use only first 8 sequences to keep bootstrap fast
const bootData: SequenceData = fixture.small_data.slice(0, 8);

describe('bootstrapTna', () => {
  it('runs stability bootstrap with correct structure', () => {
    const model = tna(bootData);
    const result = bootstrapTna(model, {
      iter: 20,
      seed: 42,
      method: 'stability',
    }) as BootstrapResult;

    expect(result.iter).toBe(20);
    expect(result.method).toBe('stability');
    expect(result.bootSummary.length).toBeGreaterThan(0);

    for (const edge of result.bootSummary) {
      expect(typeof edge.from).toBe('string');
      expect(typeof edge.to).toBe('string');
      expect(typeof edge.weight).toBe('number');
      expect(typeof edge.pValue).toBe('number');
      expect(typeof edge.ciLower).toBe('number');
      expect(typeof edge.ciUpper).toBe('number');
      expect(edge.ciLower).toBeLessThanOrEqual(edge.ciUpper);
    }
  });

  it('runs threshold bootstrap', () => {
    const model = tna(bootData);
    const result = bootstrapTna(model, {
      iter: 20,
      seed: 42,
      method: 'threshold',
    }) as BootstrapResult;

    expect(result.iter).toBe(20);
    expect(result.method).toBe('threshold');
    expect(result.bootSummary.length).toBeGreaterThan(0);
  });

  it('is deterministic with same seed', () => {
    const model = tna(bootData);
    const r1 = bootstrapTna(model, { iter: 10, seed: 42 }) as BootstrapResult;
    const r2 = bootstrapTna(model, { iter: 10, seed: 42 }) as BootstrapResult;

    expect(r1.bootSummary.length).toBe(r2.bootSummary.length);
    for (let i = 0; i < r1.bootSummary.length; i++) {
      expect(r1.bootSummary[i]!.weight).toBe(r2.bootSummary[i]!.weight);
      expect(r1.bootSummary[i]!.pValue).toBe(r2.bootSummary[i]!.pValue);
    }
  });

  it('produces matrices of correct dimensions', () => {
    const model = tna(bootData);
    const n = model.labels.length;
    const result = bootstrapTna(model, { iter: 10, seed: 42 }) as BootstrapResult;

    expect(result.weightsMean.rows).toBe(n);
    expect(result.weightsMean.cols).toBe(n);
    expect(result.weightsSd.rows).toBe(n);
    expect(result.pValues.rows).toBe(n);
    expect(result.ciLower.rows).toBe(n);
    expect(result.ciUpper.rows).toBe(n);
  });
});

describe('permutationTest', () => {
  it('runs edge-wise permutation test', () => {
    const groups = bootData.map((_, i) => (i < 4 ? 'High' : 'Low'));
    const gmodel = groupTna(bootData, groups);
    const modelA = gmodel.models['High']!;
    const modelB = gmodel.models['Low']!;

    const result = permutationTest(modelA, modelB, {
      iter: 20,
      seed: 42,
    });

    expect(result.edges.stats.length).toBeGreaterThan(0);

    for (const stat of result.edges.stats) {
      expect(typeof stat.edgeName).toBe('string');
      expect(typeof stat.pValue).toBe('number');
      expect(stat.pValue).toBeGreaterThanOrEqual(0);
      expect(stat.pValue).toBeLessThanOrEqual(1);
    }
  });

  it('is deterministic with same seed', () => {
    const groups = bootData.map((_, i) => (i < 4 ? 'High' : 'Low'));
    const gmodel = groupTna(bootData, groups);
    const modelA = gmodel.models['High']!;
    const modelB = gmodel.models['Low']!;

    const r1 = permutationTest(modelA, modelB, { iter: 10, seed: 42 });
    const r2 = permutationTest(modelA, modelB, { iter: 10, seed: 42 });

    expect(r1.edges.stats.length).toBe(r2.edges.stats.length);
    for (let i = 0; i < r1.edges.stats.length; i++) {
      expect(r1.edges.stats[i]!.pValue).toBe(r2.edges.stats[i]!.pValue);
    }
  });
});

describe('confidenceInterval', () => {
  it('computes percentile confidence interval', () => {
    const values = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const [lower, upper] = confidenceInterval(values, 0.9, 'percentile');
    expect(lower).toBeLessThan(upper);
    expect(lower).toBeCloseTo(1.45, 1);
    expect(upper).toBeCloseTo(9.55, 1);
  });

  it('computes basic confidence interval', () => {
    const values = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const [lower, upper] = confidenceInterval(values, 0.9, 'basic');
    expect(lower).toBeLessThan(upper);
  });
});
