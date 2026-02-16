import { describe, it, expect } from 'vitest';
import { SeededRNG } from '../src/core/rng.js';

describe('SeededRNG', () => {
  it('produces deterministic output for same seed', () => {
    const rng1 = new SeededRNG(42);
    const rng2 = new SeededRNG(42);

    const vals1 = Array.from({ length: 10 }, () => rng1.random());
    const vals2 = Array.from({ length: 10 }, () => rng2.random());
    expect(vals1).toEqual(vals2);
  });

  it('produces different output for different seeds', () => {
    const rng1 = new SeededRNG(42);
    const rng2 = new SeededRNG(123);

    const vals1 = Array.from({ length: 10 }, () => rng1.random());
    const vals2 = Array.from({ length: 10 }, () => rng2.random());
    expect(vals1).not.toEqual(vals2);
  });

  it('random() produces values in [0, 1)', () => {
    const rng = new SeededRNG(42);
    for (let i = 0; i < 1000; i++) {
      const v = rng.random();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('randInt() produces values in [0, max)', () => {
    const rng = new SeededRNG(42);
    for (let i = 0; i < 1000; i++) {
      const v = rng.randInt(10);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(10);
      expect(Number.isInteger(v)).toBe(true);
    }
  });

  it('shuffle produces a permutation', () => {
    const rng = new SeededRNG(42);
    const arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const original = [...arr];
    rng.shuffle(arr);
    expect(arr.sort((a, b) => a - b)).toEqual(original);
  });

  it('shuffle is deterministic', () => {
    const rng1 = new SeededRNG(42);
    const rng2 = new SeededRNG(42);
    const a = [1, 2, 3, 4, 5];
    const b = [1, 2, 3, 4, 5];
    rng1.shuffle(a);
    rng2.shuffle(b);
    expect(a).toEqual(b);
  });

  it('permutation generates valid permutation', () => {
    const rng = new SeededRNG(42);
    const perm = rng.permutation(10);
    expect(perm.length).toBe(10);
    expect([...perm].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('choice generates correct number of samples', () => {
    const rng = new SeededRNG(42);
    const result = rng.choice(100, 20);
    expect(result.length).toBe(20);
    for (const v of result) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(100);
    }
  });

  it('choiceWithoutReplacement generates unique samples', () => {
    const rng = new SeededRNG(42);
    const result = rng.choiceWithoutReplacement(10, 5);
    expect(result.length).toBe(5);
    expect(new Set(result).size).toBe(5);
    for (const v of result) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(10);
    }
  });

  it('choiceWithoutReplacement throws for size > n', () => {
    const rng = new SeededRNG(42);
    expect(() => rng.choiceWithoutReplacement(3, 5)).toThrow();
  });
});
