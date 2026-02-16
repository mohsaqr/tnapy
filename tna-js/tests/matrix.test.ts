import { describe, it, expect } from 'vitest';
import {
  Matrix,
  rowNormalize,
  minmaxScale,
  maxScale,
  rankScale,
  applyScaling,
  arrayMean,
  arrayStd,
  pearsonCorr,
  arrayQuantile,
} from '../src/core/matrix.js';

describe('Matrix', () => {
  it('creates from 2D array and converts back', () => {
    const arr = [[1, 2, 3], [4, 5, 6]];
    const m = Matrix.from2D(arr);
    expect(m.rows).toBe(2);
    expect(m.cols).toBe(3);
    expect(m.to2D()).toEqual(arr);
  });

  it('creates identity matrix', () => {
    const I = Matrix.eye(3);
    expect(I.get(0, 0)).toBe(1);
    expect(I.get(0, 1)).toBe(0);
    expect(I.get(1, 1)).toBe(1);
    expect(I.get(2, 2)).toBe(1);
  });

  it('creates zero matrix', () => {
    const m = Matrix.zeros(2, 3);
    expect(m.sum()).toBe(0);
    expect(m.rows).toBe(2);
    expect(m.cols).toBe(3);
  });

  it('creates filled matrix', () => {
    const m = Matrix.fill(2, 2, 5);
    expect(m.sum()).toBe(20);
  });

  it('get/set elements', () => {
    const m = Matrix.zeros(3, 3);
    m.set(1, 2, 42);
    expect(m.get(1, 2)).toBe(42);
    expect(m.get(0, 0)).toBe(0);
  });

  it('clones deeply', () => {
    const m = Matrix.from2D([[1, 2], [3, 4]]);
    const c = m.clone();
    c.set(0, 0, 99);
    expect(m.get(0, 0)).toBe(1);
    expect(c.get(0, 0)).toBe(99);
  });

  it('transposes correctly', () => {
    const m = Matrix.from2D([[1, 2, 3], [4, 5, 6]]);
    const t = m.transpose();
    expect(t.rows).toBe(3);
    expect(t.cols).toBe(2);
    expect(t.get(0, 0)).toBe(1);
    expect(t.get(0, 1)).toBe(4);
    expect(t.get(2, 0)).toBe(3);
    expect(t.get(2, 1)).toBe(6);
  });

  it('multiplies matrices', () => {
    const a = Matrix.from2D([[1, 2], [3, 4]]);
    const b = Matrix.from2D([[5, 6], [7, 8]]);
    const c = a.matmul(b);
    expect(c.to2D()).toEqual([[19, 22], [43, 50]]);
  });

  it('adds element-wise', () => {
    const a = Matrix.from2D([[1, 2], [3, 4]]);
    const b = Matrix.from2D([[10, 20], [30, 40]]);
    const c = a.add(b);
    expect(c.to2D()).toEqual([[11, 22], [33, 44]]);
  });

  it('subtracts element-wise', () => {
    const a = Matrix.from2D([[10, 20], [30, 40]]);
    const b = Matrix.from2D([[1, 2], [3, 4]]);
    const c = a.sub(b);
    expect(c.to2D()).toEqual([[9, 18], [27, 36]]);
  });

  it('multiplies element-wise', () => {
    const a = Matrix.from2D([[1, 2], [3, 4]]);
    const b = Matrix.from2D([[2, 3], [4, 5]]);
    const c = a.mul(b);
    expect(c.to2D()).toEqual([[2, 6], [12, 20]]);
  });

  it('scales by scalar', () => {
    const m = Matrix.from2D([[1, 2], [3, 4]]);
    const s = m.scale(3);
    expect(s.to2D()).toEqual([[3, 6], [9, 12]]);
  });

  it('maps elements', () => {
    const m = Matrix.from2D([[1, 4], [9, 16]]);
    const r = m.map((v) => Math.sqrt(v));
    expect(r.to2D()).toEqual([[1, 2], [3, 4]]);
  });

  it('computes sum', () => {
    const m = Matrix.from2D([[1, 2], [3, 4]]);
    expect(m.sum()).toBe(10);
  });

  it('computes row sums', () => {
    const m = Matrix.from2D([[1, 2, 3], [4, 5, 6]]);
    const rs = m.rowSums();
    expect(Array.from(rs)).toEqual([6, 15]);
  });

  it('computes column sums', () => {
    const m = Matrix.from2D([[1, 2, 3], [4, 5, 6]]);
    const cs = m.colSums();
    expect(Array.from(cs)).toEqual([5, 7, 9]);
  });

  it('extracts diagonal', () => {
    const m = Matrix.from2D([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    expect(Array.from(m.diag())).toEqual([1, 5, 9]);
  });

  it('sets diagonal', () => {
    const m = Matrix.from2D([[1, 2], [3, 4]]);
    const r = m.setDiag(0);
    expect(r.to2D()).toEqual([[0, 2], [3, 0]]);
  });

  it('computes max and min', () => {
    const m = Matrix.from2D([[3, 1], [4, 2]]);
    expect(m.max()).toBe(4);
    expect(m.min()).toBe(1);
  });

  it('counts elements matching predicate', () => {
    const m = Matrix.from2D([[0, 1, 2], [3, 0, 5]]);
    expect(m.count((v) => v > 0)).toBe(4);
    expect(m.count((v) => v === 0)).toBe(2);
  });

  it('checks any predicate', () => {
    const m = Matrix.from2D([[0, 0], [0, 1]]);
    expect(m.any((v) => v > 0)).toBe(true);
    expect(m.any((v) => v > 1)).toBe(false);
  });

  it('extracts row and column', () => {
    const m = Matrix.from2D([[1, 2, 3], [4, 5, 6]]);
    expect(Array.from(m.row(0))).toEqual([1, 2, 3]);
    expect(Array.from(m.row(1))).toEqual([4, 5, 6]);
    expect(Array.from(m.col(0))).toEqual([1, 4]);
    expect(Array.from(m.col(2))).toEqual([3, 6]);
  });

  it('extracts sub-matrix', () => {
    const m = Matrix.from2D([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    const sub = m.subMatrix([0, 2], [1, 2]);
    expect(sub.to2D()).toEqual([[2, 3], [8, 9]]);
  });

  it('computes quantile', () => {
    const m = Matrix.from2D([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);
    expect(m.quantile(0)).toBe(1);
    expect(m.quantile(1)).toBe(10);
    expect(m.quantile(0.5)).toBeCloseTo(5.5);
  });

  it('checks isSquare', () => {
    expect(Matrix.zeros(3, 3).isSquare).toBe(true);
    expect(Matrix.zeros(2, 3).isSquare).toBe(false);
  });

  it('computes meanNonZero', () => {
    const m = Matrix.from2D([[0, 2, 4], [0, 0, 6]]);
    expect(m.meanNonZero()).toBe(4); // (2+4+6)/3
  });

  it('inverts a matrix', () => {
    const m = Matrix.from2D([[4, 7], [2, 6]]);
    const inv = m.inverse();
    // A * A^-1 should be identity
    const product = m.matmul(inv);
    expect(product.get(0, 0)).toBeCloseTo(1, 10);
    expect(product.get(0, 1)).toBeCloseTo(0, 10);
    expect(product.get(1, 0)).toBeCloseTo(0, 10);
    expect(product.get(1, 1)).toBeCloseTo(1, 10);
  });

  it('computes outer product', () => {
    const a = new Float64Array([1, 2, 3]);
    const b = new Float64Array([4, 5]);
    const r = Matrix.outer(a, b);
    expect(r.to2D()).toEqual([[4, 5], [8, 10], [12, 15]]);
  });

  it('flattens column-major', () => {
    const m = Matrix.from2D([[1, 2], [3, 4]]);
    expect(Array.from(m.flattenColMajor())).toEqual([1, 3, 2, 4]);
  });

  it('creates diagonal matrix from vector', () => {
    const d = Matrix.diag(new Float64Array([1, 2, 3]));
    expect(d.get(0, 0)).toBe(1);
    expect(d.get(1, 1)).toBe(2);
    expect(d.get(2, 2)).toBe(3);
    expect(d.get(0, 1)).toBe(0);
  });
});

describe('rowNormalize', () => {
  it('normalizes rows to sum to 1', () => {
    const m = Matrix.from2D([[2, 3, 5], [1, 1, 2]]);
    const n = rowNormalize(m);
    expect(n.get(0, 0)).toBeCloseTo(0.2);
    expect(n.get(0, 1)).toBeCloseTo(0.3);
    expect(n.get(0, 2)).toBeCloseTo(0.5);
    expect(n.get(1, 0)).toBeCloseTo(0.25);
    expect(n.get(1, 1)).toBeCloseTo(0.25);
    expect(n.get(1, 2)).toBeCloseTo(0.5);
  });

  it('handles zero rows', () => {
    const m = Matrix.from2D([[0, 0, 0], [1, 1, 2]]);
    const n = rowNormalize(m);
    // zero row stays zero (divided by 1 to avoid div-by-zero)
    expect(n.get(0, 0)).toBe(0);
  });
});

describe('scaling functions', () => {
  it('minmaxScale normalizes to [0,1]', () => {
    const m = Matrix.from2D([[2, 4], [6, 8]]);
    const s = minmaxScale(m);
    expect(s.get(0, 0)).toBeCloseTo(0);
    expect(s.get(1, 1)).toBeCloseTo(1);
    expect(s.get(0, 1)).toBeCloseTo(1 / 3);
  });

  it('maxScale divides by max', () => {
    const m = Matrix.from2D([[2, 4], [6, 8]]);
    const s = maxScale(m);
    expect(s.get(1, 1)).toBeCloseTo(1);
    expect(s.get(0, 0)).toBeCloseTo(0.25);
  });

  it('rankScale converts to ranks', () => {
    const m = Matrix.from2D([[3, 1], [2, 4]]);
    const r = rankScale(m);
    expect(r.get(0, 1)).toBe(1); // smallest
    expect(r.get(1, 1)).toBe(4); // largest
  });

  it('applyScaling chains methods', () => {
    const m = Matrix.from2D([[2, 4], [6, 8]]);
    const { weights, applied } = applyScaling(m, ['minmax']);
    expect(applied).toEqual(['minmax']);
    expect(weights.get(0, 0)).toBeCloseTo(0);
    expect(weights.get(1, 1)).toBeCloseTo(1);
  });

  it('applyScaling with null returns clone', () => {
    const m = Matrix.from2D([[1, 2], [3, 4]]);
    const { weights, applied } = applyScaling(m, null);
    expect(applied).toEqual([]);
    expect(weights.to2D()).toEqual(m.to2D());
  });
});

describe('array utilities', () => {
  it('arrayMean computes mean', () => {
    expect(arrayMean(new Float64Array([1, 2, 3, 4]))).toBe(2.5);
    expect(arrayMean(new Float64Array([]))).toBe(0);
  });

  it('arrayStd computes standard deviation', () => {
    const arr = new Float64Array([2, 4, 4, 4, 5, 5, 7, 9]);
    expect(arrayStd(arr, 0)).toBeCloseTo(2, 1);
  });

  it('pearsonCorr computes correlation', () => {
    const a = new Float64Array([1, 2, 3, 4, 5]);
    const b = new Float64Array([2, 4, 6, 8, 10]);
    expect(pearsonCorr(a, b)).toBeCloseTo(1, 10);

    const c = new Float64Array([5, 4, 3, 2, 1]);
    expect(pearsonCorr(a, c)).toBeCloseTo(-1, 10);
  });

  it('arrayQuantile computes quantiles', () => {
    const arr = new Float64Array([1, 2, 3, 4, 5]);
    expect(arrayQuantile(arr, 0)).toBe(1);
    expect(arrayQuantile(arr, 1)).toBe(5);
    expect(arrayQuantile(arr, 0.5)).toBe(3);
  });
});
