import { describe, it, expect } from 'vitest';
import { centralities, betweennessNetwork, AVAILABLE_MEASURES } from '../src/analysis/centralities.js';
import { tna } from '../src/core/model.js';
import { groupTna } from '../src/core/group.js';
import type { SequenceData } from '../src/core/types.js';
import fixture from './fixtures/ground_truth.json';

const smallData: SequenceData = fixture.small_data;
const expectedCent = fixture.centralities;

const TOL = 1e-4; // centrality algorithms may have minor floating point diffs

describe('centralities', () => {
  const model = tna(smallData);
  const result = centralities(model);

  it('returns all 9 measures by default', () => {
    expect(Object.keys(result.measures).length).toBe(9);
    for (const m of AVAILABLE_MEASURES) {
      expect(result.measures[m]).toBeDefined();
    }
  });

  it('returns correct labels', () => {
    expect(result.labels).toEqual(fixture.labels);
  });

  it('OutStrength matches R ground truth', () => {
    const vals = result.measures.OutStrength!;
    for (let i = 0; i < fixture.labels.length; i++) {
      const label = fixture.labels[i]!;
      expect(vals[i]).toBeCloseTo(expectedCent.OutStrength[label]!, 4);
    }
  });

  it('InStrength matches R ground truth', () => {
    const vals = result.measures.InStrength!;
    for (let i = 0; i < fixture.labels.length; i++) {
      const label = fixture.labels[i]!;
      expect(vals[i]).toBeCloseTo(expectedCent.InStrength[label]!, 4);
    }
  });

  it('ClosenessIn matches R ground truth', () => {
    const vals = result.measures.ClosenessIn!;
    for (let i = 0; i < fixture.labels.length; i++) {
      const label = fixture.labels[i]!;
      expect(vals[i]).toBeCloseTo(expectedCent.ClosenessIn[label]!, 3);
    }
  });

  it('ClosenessOut matches R ground truth', () => {
    const vals = result.measures.ClosenessOut!;
    for (let i = 0; i < fixture.labels.length; i++) {
      const label = fixture.labels[i]!;
      expect(vals[i]).toBeCloseTo(expectedCent.ClosenessOut[label]!, 3);
    }
  });

  it('Closeness matches R ground truth', () => {
    const vals = result.measures.Closeness!;
    for (let i = 0; i < fixture.labels.length; i++) {
      const label = fixture.labels[i]!;
      expect(vals[i]).toBeCloseTo(expectedCent.Closeness[label]!, 3);
    }
  });

  it('Betweenness matches R ground truth', () => {
    const vals = result.measures.Betweenness!;
    for (let i = 0; i < fixture.labels.length; i++) {
      const label = fixture.labels[i]!;
      expect(vals[i]).toBeCloseTo(expectedCent.Betweenness[label]!, 0);
    }
  });

  it('BetweennessRSP matches R ground truth', () => {
    const vals = result.measures.BetweennessRSP!;
    for (let i = 0; i < fixture.labels.length; i++) {
      const label = fixture.labels[i]!;
      expect(vals[i]).toBeCloseTo(expectedCent.BetweennessRSP[label]!, 0);
    }
  });

  it('Diffusion matches R ground truth', () => {
    const vals = result.measures.Diffusion!;
    for (let i = 0; i < fixture.labels.length; i++) {
      const label = fixture.labels[i]!;
      expect(vals[i]).toBeCloseTo(expectedCent.Diffusion[label]!, 3);
    }
  });

  it('Clustering matches R ground truth', () => {
    const vals = result.measures.Clustering!;
    for (let i = 0; i < fixture.labels.length; i++) {
      const label = fixture.labels[i]!;
      expect(vals[i]).toBeCloseTo(expectedCent.Clustering[label]!, 3);
    }
  });

  it('computes subset of measures', () => {
    const result = centralities(model, {
      measures: ['OutStrength', 'InStrength'],
    });
    expect(Object.keys(result.measures)).toEqual(['OutStrength', 'InStrength']);
  });

  it('normalizes measures to [0,1]', () => {
    const result = centralities(model, { normalize: true });
    for (const [, vals] of Object.entries(result.measures)) {
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < vals.length; i++) {
        if (vals[i]! < min) min = vals[i]!;
        if (vals[i]! > max) max = vals[i]!;
      }
      expect(min).toBeGreaterThanOrEqual(-1e-10);
      expect(max).toBeLessThanOrEqual(1 + 1e-10);
    }
  });
});

describe('betweennessNetwork', () => {
  it('computes edge betweenness (R ground truth)', () => {
    const model = tna(smallData);
    const bn = betweennessNetwork(model);
    if ('weights' in bn) {
      const actual = bn.weights.to2D();
      const expected = fixture.betweenness_network_weights;
      for (let i = 0; i < actual.length; i++) {
        for (let j = 0; j < actual[i]!.length; j++) {
          expect(actual[i]![j]).toBeCloseTo(expected[i]![j]!, 0);
        }
      }
    }
  });
});

describe('group centralities', () => {
  it('computes centralities for grouped models', () => {
    const groups = smallData.map((_, i) => (i < 10 ? 'A' : 'B'));
    const gmodel = groupTna(smallData, groups);
    const result = centralities(gmodel);
    expect(result.groups).toBeDefined();
    expect(result.labels.length).toBe(fixture.labels.length * 2);
    expect(result.groups!.filter((g) => g === 'A').length).toBe(fixture.labels.length);
  });
});
