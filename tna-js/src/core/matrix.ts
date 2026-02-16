/**
 * Matrix class wrapping Float64Array with row-major layout.
 * Designed for small matrices (typically 9x9 to ~30x30) used in TNA.
 */
export class Matrix {
  readonly data: Float64Array;
  readonly rows: number;
  readonly cols: number;

  constructor(rows: number, cols: number, data?: Float64Array | number[]) {
    this.rows = rows;
    this.cols = cols;
    if (data) {
      this.data = data instanceof Float64Array ? data : new Float64Array(data);
      if (this.data.length !== rows * cols) {
        throw new Error(
          `Data length ${this.data.length} doesn't match ${rows}x${cols}=${rows * cols}`,
        );
      }
    } else {
      this.data = new Float64Array(rows * cols);
    }
  }

  /** Create from a 2D array. */
  static from2D(arr: number[][]): Matrix {
    const rows = arr.length;
    if (rows === 0) return new Matrix(0, 0);
    const cols = arr[0]!.length;
    const data = new Float64Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        data[i * cols + j] = arr[i]![j]!;
      }
    }
    return new Matrix(rows, cols, data);
  }

  /** Create an identity matrix. */
  static eye(n: number): Matrix {
    const m = new Matrix(n, n);
    for (let i = 0; i < n; i++) {
      m.data[i * n + i] = 1;
    }
    return m;
  }

  /** Create a matrix filled with a value. */
  static fill(rows: number, cols: number, value: number): Matrix {
    const data = new Float64Array(rows * cols);
    data.fill(value);
    return new Matrix(rows, cols, data);
  }

  /** Create a zero matrix. */
  static zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols);
  }

  /** Get element at (i, j). */
  get(i: number, j: number): number {
    return this.data[i * this.cols + j]!;
  }

  /** Set element at (i, j). */
  set(i: number, j: number, value: number): void {
    this.data[i * this.cols + j] = value;
  }

  /** Deep copy. */
  clone(): Matrix {
    return new Matrix(this.rows, this.cols, new Float64Array(this.data));
  }

  /** Convert to 2D array. */
  to2D(): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < this.rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.cols; j++) {
        row.push(this.get(i, j));
      }
      result.push(row);
    }
    return result;
  }

  /** Transpose. */
  transpose(): Matrix {
    const result = new Matrix(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.set(j, i, this.get(i, j));
      }
    }
    return result;
  }

  /** Matrix multiply: this @ other. */
  matmul(other: Matrix): Matrix {
    if (this.cols !== other.rows) {
      throw new Error(
        `Cannot multiply ${this.rows}x${this.cols} by ${other.rows}x${other.cols}`,
      );
    }
    const result = new Matrix(this.rows, other.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < other.cols; j++) {
        let sum = 0;
        for (let k = 0; k < this.cols; k++) {
          sum += this.get(i, k) * other.get(k, j);
        }
        result.set(i, j, sum);
      }
    }
    return result;
  }

  /** Element-wise addition. */
  add(other: Matrix): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = this.data[i]! + other.data[i]!;
    }
    return result;
  }

  /** Element-wise subtraction. */
  sub(other: Matrix): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = this.data[i]! - other.data[i]!;
    }
    return result;
  }

  /** Element-wise multiplication. */
  mul(other: Matrix): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = this.data[i]! * other.data[i]!;
    }
    return result;
  }

  /** Scalar multiply. */
  scale(s: number): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = this.data[i]! * s;
    }
    return result;
  }

  /** Element-wise apply. */
  map(fn: (value: number, i: number, j: number) => number): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.set(i, j, fn(this.get(i, j), i, j));
      }
    }
    return result;
  }

  /** Sum of all elements. */
  sum(): number {
    let s = 0;
    for (let i = 0; i < this.data.length; i++) {
      s += this.data[i]!;
    }
    return s;
  }

  /** Row sums as array. */
  rowSums(): Float64Array {
    const sums = new Float64Array(this.rows);
    for (let i = 0; i < this.rows; i++) {
      let s = 0;
      for (let j = 0; j < this.cols; j++) {
        s += this.get(i, j);
      }
      sums[i] = s;
    }
    return sums;
  }

  /** Column sums as array. */
  colSums(): Float64Array {
    const sums = new Float64Array(this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        sums[j]! += this.get(i, j);
      }
    }
    return sums;
  }

  /** Get diagonal as array. */
  diag(): Float64Array {
    const n = Math.min(this.rows, this.cols);
    const d = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      d[i] = this.get(i, i);
    }
    return d;
  }

  /** Set diagonal values. */
  setDiag(value: number): Matrix {
    const result = this.clone();
    const n = Math.min(this.rows, this.cols);
    for (let i = 0; i < n; i++) {
      result.set(i, i, value);
    }
    return result;
  }

  /** Fill diagonal with values from array. */
  setDiagFrom(values: Float64Array | number[]): Matrix {
    const result = this.clone();
    const n = Math.min(this.rows, this.cols, values.length);
    for (let i = 0; i < n; i++) {
      result.set(i, i, values[i]!);
    }
    return result;
  }

  /** Create a diagonal matrix from a vector. */
  static diag(values: Float64Array | number[]): Matrix {
    const n = values.length;
    const result = new Matrix(n, n);
    for (let i = 0; i < n; i++) {
      result.set(i, i, values[i]!);
    }
    return result;
  }

  /** Max element. */
  max(): number {
    let m = -Infinity;
    for (let i = 0; i < this.data.length; i++) {
      if (this.data[i]! > m) m = this.data[i]!;
    }
    return m;
  }

  /** Min element. */
  min(): number {
    let m = Infinity;
    for (let i = 0; i < this.data.length; i++) {
      if (this.data[i]! < m) m = this.data[i]!;
    }
    return m;
  }

  /** Count elements matching a predicate. */
  count(predicate: (v: number) => boolean): number {
    let c = 0;
    for (let i = 0; i < this.data.length; i++) {
      if (predicate(this.data[i]!)) c++;
    }
    return c;
  }

  /** Check if any element satisfies predicate. */
  any(predicate: (v: number) => boolean): boolean {
    for (let i = 0; i < this.data.length; i++) {
      if (predicate(this.data[i]!)) return true;
    }
    return false;
  }

  /** Flatten to array in column-major order (matching R's as.vector). */
  flattenColMajor(): Float64Array {
    const result = new Float64Array(this.rows * this.cols);
    let idx = 0;
    for (let j = 0; j < this.cols; j++) {
      for (let i = 0; i < this.rows; i++) {
        result[idx++] = this.get(i, j);
      }
    }
    return result;
  }

  /** Flatten to array in row-major order. */
  flatten(): Float64Array {
    return new Float64Array(this.data);
  }

  /** Get a row as array. */
  row(i: number): Float64Array {
    const result = new Float64Array(this.cols);
    for (let j = 0; j < this.cols; j++) {
      result[j] = this.get(i, j);
    }
    return result;
  }

  /** Get a column as array. */
  col(j: number): Float64Array {
    const result = new Float64Array(this.rows);
    for (let i = 0; i < this.rows; i++) {
      result[i] = this.get(i, j);
    }
    return result;
  }

  /** Extract a sub-matrix given row and column indices. */
  subMatrix(rowIndices: number[], colIndices: number[]): Matrix {
    const result = new Matrix(rowIndices.length, colIndices.length);
    for (let i = 0; i < rowIndices.length; i++) {
      for (let j = 0; j < colIndices.length; j++) {
        result.set(i, j, this.get(rowIndices[i]!, colIndices[j]!));
      }
    }
    return result;
  }

  /** Quantile of all elements. */
  quantile(p: number): number {
    const sorted = Array.from(this.data).sort((a, b) => a - b);
    const idx = p * (sorted.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo]!;
    const frac = idx - lo;
    return sorted[lo]! * (1 - frac) + sorted[hi]! * frac;
  }

  /** Is square? */
  get isSquare(): boolean {
    return this.rows === this.cols;
  }

  /** Mean of non-zero elements. */
  meanNonZero(): number {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < this.data.length; i++) {
      if (this.data[i]! > 0) {
        sum += this.data[i]!;
        count++;
      }
    }
    return count > 0 ? sum / count : 0;
  }

  /** Invert matrix using Gauss-Jordan elimination. */
  inverse(): Matrix {
    if (!this.isSquare) {
      throw new Error('Cannot invert non-square matrix');
    }
    const n = this.rows;
    // Augmented matrix [A | I]
    const aug = new Matrix(n, 2 * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        aug.set(i, j, this.get(i, j));
      }
      aug.set(i, n + i, 1);
    }

    for (let col = 0; col < n; col++) {
      // Find pivot
      let maxVal = Math.abs(aug.get(col, col));
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        const val = Math.abs(aug.get(row, col));
        if (val > maxVal) {
          maxVal = val;
          maxRow = row;
        }
      }

      if (maxVal < 1e-15) {
        throw new Error('Matrix is singular');
      }

      // Swap rows
      if (maxRow !== col) {
        for (let j = 0; j < 2 * n; j++) {
          const tmp = aug.get(col, j);
          aug.set(col, j, aug.get(maxRow, j));
          aug.set(maxRow, j, tmp);
        }
      }

      // Scale pivot row
      const pivot = aug.get(col, col);
      for (let j = 0; j < 2 * n; j++) {
        aug.set(col, j, aug.get(col, j) / pivot);
      }

      // Eliminate column
      for (let row = 0; row < n; row++) {
        if (row !== col) {
          const factor = aug.get(row, col);
          for (let j = 0; j < 2 * n; j++) {
            aug.set(row, j, aug.get(row, j) - factor * aug.get(col, j));
          }
        }
      }
    }

    // Extract inverse
    const result = new Matrix(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        result.set(i, j, aug.get(i, n + j));
      }
    }
    return result;
  }

  /** Outer product of two vectors. */
  static outer(a: Float64Array | number[], b: Float64Array | number[]): Matrix {
    const m = a.length;
    const n = b.length;
    const result = new Matrix(m, n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        result.set(i, j, a[i]! * b[j]!);
      }
    }
    return result;
  }
}

// ---- Utility functions on arrays ----

/** Row normalize a matrix (each row sums to 1). */
export function rowNormalize(mat: Matrix): Matrix {
  const result = mat.clone();
  for (let i = 0; i < mat.rows; i++) {
    let rowSum = 0;
    for (let j = 0; j < mat.cols; j++) {
      rowSum += mat.get(i, j);
    }
    if (rowSum === 0) rowSum = 1; // avoid division by zero
    for (let j = 0; j < mat.cols; j++) {
      result.set(i, j, mat.get(i, j) / rowSum);
    }
  }
  return result;
}

/** Min-max normalization to [0, 1]. */
export function minmaxScale(mat: Matrix): Matrix {
  const minVal = mat.min();
  const maxVal = mat.max();
  if (maxVal === minVal) return Matrix.zeros(mat.rows, mat.cols);
  const range = maxVal - minVal;
  return mat.map((v) => (v - minVal) / range);
}

/** Divide by maximum value. */
export function maxScale(mat: Matrix): Matrix {
  const maxVal = mat.max();
  if (maxVal === 0) return mat.clone();
  return mat.map((v) => v / maxVal);
}

/** Convert to ranks (1-based, average ties). */
export function rankScale(mat: Matrix): Matrix {
  const flat = Array.from(mat.data);
  const n = flat.length;

  // Create (value, index) pairs
  const indexed = flat.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);

  const ranks = new Float64Array(n);

  let i = 0;
  while (i < n) {
    // Find group of ties
    let j = i;
    while (j < n && indexed[j]!.v === indexed[i]!.v) j++;
    // Average rank for ties
    const avgRank = (i + 1 + j) / 2;
    for (let k = i; k < j; k++) {
      const idx = indexed[k]!.i;
      // Set zeros to zero rank
      if (indexed[k]!.v === 0) {
        ranks[idx] = 0;
      } else {
        ranks[idx] = avgRank;
      }
    }
    i = j;
  }

  return new Matrix(mat.rows, mat.cols, ranks);
}

/** Apply one or more scaling methods to a matrix. */
export function applyScaling(
  mat: Matrix,
  scaling: string | string[] | null | undefined,
): { weights: Matrix; applied: string[] } {
  if (!scaling) return { weights: mat.clone(), applied: [] };

  const methods = typeof scaling === 'string' ? [scaling] : scaling;
  let result = mat.clone();
  const applied: string[] = [];

  for (const method of methods) {
    const m = method.toLowerCase();
    switch (m) {
      case 'minmax':
        result = minmaxScale(result);
        applied.push('minmax');
        break;
      case 'max':
        result = maxScale(result);
        applied.push('max');
        break;
      case 'rank':
        result = rankScale(result);
        applied.push('rank');
        break;
      default:
        throw new Error(`Unknown scaling method: ${method}`);
    }
  }

  return { weights: result, applied };
}

/** Compute mean of a Float64Array. */
export function arrayMean(arr: Float64Array): number {
  if (arr.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < arr.length; i++) sum += arr[i]!;
  return sum / arr.length;
}

/** Compute standard deviation of a Float64Array (ddof=1). */
export function arrayStd(arr: Float64Array, ddof = 1): number {
  if (arr.length <= ddof) return 0;
  const mean = arrayMean(arr);
  let sumSq = 0;
  for (let i = 0; i < arr.length; i++) {
    const diff = arr[i]! - mean;
    sumSq += diff * diff;
  }
  return Math.sqrt(sumSq / (arr.length - ddof));
}

/** Pearson correlation between two Float64Arrays. */
export function pearsonCorr(a: Float64Array, b: Float64Array): number {
  if (a.length !== b.length || a.length < 2) return NaN;
  const meanA = arrayMean(a);
  const meanB = arrayMean(b);
  let num = 0;
  let denA = 0;
  let denB = 0;
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

/** Quantile of a Float64Array. */
export function arrayQuantile(arr: Float64Array, p: number): number {
  const sorted = Array.from(arr).sort((a, b) => a - b);
  const idx = p * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo]!;
  const frac = idx - lo;
  return sorted[lo]! * (1 - frac) + sorted[hi]! * frac;
}
