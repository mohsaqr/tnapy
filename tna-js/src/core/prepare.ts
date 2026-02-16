/**
 * Data preparation functions.
 * Port of Python tna/prepare.py
 */
import type { SequenceData, TNAData } from './types.js';

/**
 * Create sequence data from a 2D string array (wide format).
 * Extracts unique state labels and optionally adds begin/end states.
 */
export function createSeqdata(
  data: SequenceData,
  options?: {
    beginState?: string;
    endState?: string;
  },
): { data: SequenceData; labels: string[] } {
  // Get unique states
  const stateSet = new Set<string>();
  for (const row of data) {
    for (const val of row) {
      if (val !== null && val !== undefined && val !== '') {
        stateSet.add(val);
      }
    }
  }
  const labels = Array.from(stateSet).sort();

  // Add begin/end states
  if (options?.beginState && !labels.includes(options.beginState)) {
    labels.unshift(options.beginState);
  }
  if (options?.endState && !labels.includes(options.endState)) {
    labels.push(options.endState);
  }

  let result = data;

  // Prepend begin state
  if (options?.beginState) {
    result = result.map((row) => [options.beginState!, ...row]);
  }

  // Append end state
  if (options?.endState) {
    result = result.map((row) => [...row, options.endState!]);
  }

  return { data: result, labels };
}

/**
 * Parse wide-format data into a TNAData object.
 * Input: array of arrays where each inner array is a sequence.
 */
export function prepareData(
  data: SequenceData,
  options?: {
    beginState?: string;
    endState?: string;
  },
): TNAData {
  const { data: seqData, labels } = createSeqdata(data, options);

  // Compute statistics
  const actionCounts = new Map<string, number>();
  let totalActions = 0;
  let totalLength = 0;
  let maxLen = 0;

  for (const row of seqData) {
    let rowLen = 0;
    for (const val of row) {
      if (val !== null && val !== undefined && val !== '') {
        actionCounts.set(val, (actionCounts.get(val) ?? 0) + 1);
        totalActions++;
        rowLen++;
      }
    }
    totalLength += rowLen;
    if (rowLen > maxLen) maxLen = rowLen;
  }

  return {
    sequenceData: seqData,
    labels,
    statistics: {
      nSessions: seqData.length,
      nUniqueActions: labels.length,
      uniqueActions: labels,
      maxSequenceLength: maxLen,
      meanSequenceLength: seqData.length > 0 ? totalLength / seqData.length : 0,
    },
  };
}

/**
 * Convert one-hot encoded data into wide-format sequence data.
 *
 * @param data - Array of records with 0/1 values for each column
 * @param cols - Column names that are one-hot encoded state indicators
 * @param options - windowing options
 */
export function importOnehot(
  data: Record<string, number>[],
  cols: string[],
  options?: {
    actor?: string;
    session?: string;
    windowSize?: number;
    windowType?: 'tumbling' | 'sliding';
    aggregate?: boolean;
  },
): SequenceData {
  const windowSize = options?.windowSize ?? 1;
  const windowType = options?.windowType ?? 'tumbling';
  const aggregate = options?.aggregate ?? false;

  // Decode: 1 -> column name, 0 -> null
  const decoded: (string | null)[][] = data.map((row) =>
    cols.map((col) => (row[col] === 1 ? col : null)),
  );

  // Group by actor/session if provided
  const groups: (string | null)[][][] = [];
  if (options?.actor || options?.session) {
    const groupMap = new Map<string, (string | null)[][]>();
    for (let i = 0; i < data.length; i++) {
      const parts: string[] = [];
      if (options?.actor) parts.push(String(data[i]![options.actor] ?? ''));
      if (options?.session) parts.push(String(data[i]![options.session] ?? ''));
      const key = parts.join('_');
      if (!groupMap.has(key)) groupMap.set(key, []);
      groupMap.get(key)!.push(decoded[i]!);
    }
    for (const rows of groupMap.values()) {
      groups.push(rows);
    }
  } else {
    groups.push(decoded);
  }

  // Process each group into one flattened row
  const result: (string | null)[][] = [];

  for (const groupRows of groups) {
    const nRows = groupRows.length;

    // Generate window boundaries
    const windows: [number, number][] = [];
    if (windowType === 'tumbling') {
      for (let start = 0; start < nRows; start += windowSize) {
        windows.push([start, Math.min(start + windowSize, nRows)]);
      }
    } else {
      const nWindows = Math.max(1, nRows - windowSize + 1);
      for (let start = 0; start < nWindows; start++) {
        windows.push([start, Math.min(start + windowSize, nRows)]);
      }
    }

    const rowValues: (string | null)[] = [];

    for (const [start, end] of windows) {
      const windowRows = groupRows.slice(start, end);

      if (windowType === 'sliding' || aggregate) {
        // One slot per column: first non-null value
        for (let c = 0; c < cols.length; c++) {
          let firstVal: string | null = null;
          for (const r of windowRows) {
            if (r[c] !== null) {
              firstVal = r[c]!;
              break;
            }
          }
          rowValues.push(firstVal);
        }
      } else {
        // Tumbling without aggregate: expand all rows x cols
        for (const r of windowRows) {
          for (let c = 0; c < cols.length; c++) {
            rowValues.push(r[c]!);
          }
        }
      }
    }

    result.push(rowValues);
  }

  return result;
}
