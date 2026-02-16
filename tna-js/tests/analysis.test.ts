import { describe, it, expect } from 'vitest';
import { prune } from '../src/analysis/prune.js';
import { cliques } from '../src/analysis/cliques.js';
import { communities } from '../src/analysis/communities.js';
import { compareSequences } from '../src/analysis/compare.js';
import { clusterSequences } from '../src/analysis/cluster.js';
import { tna } from '../src/core/model.js';
import { groupTna } from '../src/core/group.js';
import type { SequenceData, CommunityResult } from '../src/core/types.js';
import fixture from './fixtures/ground_truth.json';

const smallData: SequenceData = fixture.small_data;

describe('prune', () => {
  it('removes edges below threshold (R ground truth)', () => {
    const model = tna(smallData);
    const pruned = prune(model, 0.05);
    if ('weights' in pruned) {
      const actual = pruned.weights.to2D();
      const expected = fixture.pruned_weights_005;
      for (let i = 0; i < actual.length; i++) {
        for (let j = 0; j < actual[i]!.length; j++) {
          expect(actual[i]![j]).toBeCloseTo(expected[i]![j]!, 10);
        }
      }
    }
  });

  it('sets sub-threshold edges to zero', () => {
    const model = tna(smallData);
    const pruned = prune(model, 0.3);
    if ('weights' in pruned) {
      const w = pruned.weights;
      for (let i = 0; i < w.rows; i++) {
        for (let j = 0; j < w.cols; j++) {
          expect(w.get(i, j) === 0 || w.get(i, j) >= 0.3).toBe(true);
        }
      }
    }
  });
});

describe('cliques', () => {
  it('finds directed cliques (R ground truth)', () => {
    const model = tna(smallData);
    const result = cliques(model);
    if ('labels' in result) {
      const foundLabels = result.labels.map((c) => [...c].sort());
      const expectedLabels = fixture.clique_labels.map((c: string[]) => [...c].sort());
      expect(foundLabels.length).toBe(expectedLabels.length);

      for (const expected of expectedLabels) {
        const found = foundLabels.some(
          (f) => f.length === expected.length && f.every((v, i) => v === expected[i]),
        );
        expect(found).toBe(true);
      }
    }
  });

  it('returns correct number of cliques', () => {
    const model = tna(smallData);
    const result = cliques(model);
    if ('labels' in result) {
      expect(result.labels.length).toBe(14);
    }
  });
});

describe('communities', () => {
  it('detects communities with leading_eigen', () => {
    const model = tna(smallData);
    const result = communities(model) as CommunityResult;

    const commAssign = result.assignments['leading_eigen']!;
    expect(commAssign.length).toBe(fixture.labels.length);
    // Should find at least 2 communities
    const numComms = new Set(commAssign).size;
    expect(numComms).toBeGreaterThanOrEqual(2);
    // All assignments should be valid non-negative integers
    for (const v of commAssign) {
      expect(typeof v).toBe('number');
      expect(v).toBeGreaterThanOrEqual(0);
    }
  });

  it('detects communities with louvain', () => {
    const model = tna(smallData);
    const result = communities(model, { methods: 'louvain' }) as CommunityResult;
    const commAssign = result.assignments['louvain']!;
    expect(commAssign.length).toBe(fixture.labels.length);
    for (const v of commAssign) {
      expect(typeof v).toBe('number');
    }
  });

  it('detects communities with label_prop', () => {
    const model = tna(smallData);
    const result = communities(model, { methods: 'label_prop' }) as CommunityResult;
    const commAssign = result.assignments['label_prop']!;
    expect(commAssign.length).toBe(fixture.labels.length);
    for (const v of commAssign) {
      expect(typeof v).toBe('number');
    }
  });

  it('reports community counts', () => {
    const model = tna(smallData);
    const result = communities(model) as CommunityResult;
    expect(result.counts['leading_eigen']).toBeGreaterThanOrEqual(2);
  });
});

describe('clusterSequences', () => {
  const clusterData: SequenceData = smallData.slice(0, 10);

  it('clusters with hamming distance', () => {
    const result = clusterSequences(clusterData, 2);
    expect(result.k).toBe(2);
    expect(result.assignments.length).toBe(10);
    expect(result.sizes.length).toBe(2);
    expect(result.sizes.reduce((a, b) => a + b, 0)).toBe(10);
    expect(typeof result.silhouette).toBe('number');
  });

  it('clusters with levenshtein distance', () => {
    const result = clusterSequences(clusterData, 2, { dissimilarity: 'lv' });
    expect(result.assignments.length).toBe(10);
    expect(result.dissimilarity).toBe('lv');
  });

  it('clusters with osa distance', () => {
    const result = clusterSequences(clusterData, 2, { dissimilarity: 'osa' });
    expect(result.assignments.length).toBe(10);
  });

  it('clusters with lcs distance', () => {
    const result = clusterSequences(clusterData, 2, { dissimilarity: 'lcs' });
    expect(result.assignments.length).toBe(10);
  });

  it('clusters with hierarchical method', () => {
    const result = clusterSequences(clusterData, 3, { method: 'hierarchical' });
    expect(result.k).toBe(3);
    expect(result.sizes.length).toBe(3);
    expect(result.method).toBe('hierarchical');
  });

  it('throws for k < 2', () => {
    expect(() => clusterSequences(clusterData, 1)).toThrow();
  });

  it('throws for k > n', () => {
    expect(() => clusterSequences(clusterData, 100)).toThrow();
  });
});

describe('compareSequences', () => {
  it('compares patterns across groups', () => {
    const groups = smallData.map((_, i) => (i < 10 ? 'High' : 'Low'));
    const gmodel = groupTna(smallData, groups);
    const result = compareSequences(gmodel, { sub: [1, 2], minFreq: 1 });

    expect(result.length).toBeGreaterThan(0);
    for (const row of result) {
      expect(typeof row.pattern).toBe('string');
      expect(row.frequencies['High']).toBeDefined();
      expect(row.frequencies['Low']).toBeDefined();
      expect(row.proportions['High']).toBeDefined();
      expect(row.proportions['Low']).toBeDefined();
    }
  });

  it('runs permutation test', () => {
    const groups = smallData.map((_, i) => (i < 10 ? 'High' : 'Low'));
    const gmodel = groupTna(smallData, groups);
    const result = compareSequences(gmodel, {
      sub: [1],
      minFreq: 1,
      test: true,
      iter: 100,
      seed: 42,
    });

    expect(result.length).toBeGreaterThan(0);
    for (const row of result) {
      expect(typeof row.effectSize).toBe('number');
      expect(typeof row.pValue).toBe('number');
      expect(row.pValue!).toBeGreaterThanOrEqual(0);
      expect(row.pValue!).toBeLessThanOrEqual(1);
    }
  });
});
