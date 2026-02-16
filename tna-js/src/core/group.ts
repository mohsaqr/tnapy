/**
 * Group TNA models — build and manage per-group transition networks.
 * Port of Python tna/group.py
 */
import { buildModel } from './model.js';
import type { TNA, GroupTNA, SequenceData, BuildModelOptions, ModelType } from './types.js';

/** Check if an object is a GroupTNA (duck typing). */
export function isGroupTNA(x: unknown): x is GroupTNA {
  return typeof x === 'object' && x !== null && 'models' in x;
}

/** Create a GroupTNA from a models record. */
export function createGroupTNA(models: Record<string, TNA>): GroupTNA {
  return { models };
}

/** Get group names. */
export function groupNames(g: GroupTNA): string[] {
  return Object.keys(g.models);
}

/** Iterate over groups. */
export function groupEntries(g: GroupTNA): [string, TNA][] {
  return Object.entries(g.models);
}

/** Apply a function to each group model. */
export function groupApply<T>(g: GroupTNA, fn: (model: TNA, name: string) => T): Record<string, T> {
  const result: Record<string, T> = {};
  for (const [name, model] of Object.entries(g.models)) {
    result[name] = fn(model, name);
  }
  return result;
}

/** Rename groups. */
export function renameGroups(g: GroupTNA, newNames: string[]): GroupTNA {
  const oldNames = Object.keys(g.models);
  if (newNames.length !== oldNames.length) {
    throw new Error(`Expected ${oldNames.length} names, got ${newNames.length}`);
  }
  const models: Record<string, TNA> = {};
  for (let i = 0; i < oldNames.length; i++) {
    models[newNames[i]!] = g.models[oldNames[i]!]!;
  }
  return { models };
}

/**
 * Build one TNA model per group.
 *
 * @param data - Sequence data for all groups
 * @param groups - Group label for each sequence (same length as data)
 * @param options - Build options (type, scaling, labels, etc.)
 */
function buildGroupModels(
  data: SequenceData,
  groups: string[],
  options?: BuildModelOptions & { type?: ModelType },
): GroupTNA {
  if (data.length !== groups.length) {
    throw new Error(`Data length ${data.length} doesn't match groups length ${groups.length}`);
  }

  // Detect shared labels across all groups
  let labels = options?.labels;
  if (!labels) {
    const stateSet = new Set<string>();
    for (const row of data) {
      for (const val of row) {
        if (val !== null && val !== undefined && val !== '') {
          stateSet.add(val);
        }
      }
    }
    labels = Array.from(stateSet).sort();
  }

  // Get unique groups in order of appearance
  const uniqueGroups: string[] = [];
  const seen = new Set<string>();
  for (const g of groups) {
    if (!seen.has(g)) {
      uniqueGroups.push(g);
      seen.add(g);
    }
  }

  // Build one model per group
  const models: Record<string, TNA> = {};
  for (const grp of uniqueGroups) {
    const grpData: SequenceData = [];
    for (let i = 0; i < data.length; i++) {
      if (groups[i] === grp) {
        grpData.push(data[i]!);
      }
    }
    models[grp] = buildModel(grpData, { ...options, labels });
  }

  return { models };
}

/** Build grouped relative transition probability models. */
export function groupTna(
  data: SequenceData,
  groups: string[],
  options?: Omit<BuildModelOptions, 'type' | 'params'>,
): GroupTNA {
  return buildGroupModels(data, groups, { ...options, type: 'relative' });
}

/** Build grouped frequency-based transition models. */
export function groupFtna(
  data: SequenceData,
  groups: string[],
  options?: Omit<BuildModelOptions, 'type' | 'params'>,
): GroupTNA {
  return buildGroupModels(data, groups, { ...options, type: 'frequency' });
}

/** Build grouped co-occurrence transition models. */
export function groupCtna(
  data: SequenceData,
  groups: string[],
  options?: Omit<BuildModelOptions, 'type' | 'params'>,
): GroupTNA {
  return buildGroupModels(data, groups, { ...options, type: 'co-occurrence' });
}

/** Build grouped attention-weighted transition models. */
export function groupAtna(
  data: SequenceData,
  groups: string[],
  options?: Omit<BuildModelOptions, 'type'> & { beta?: number },
): GroupTNA {
  return buildGroupModels(data, groups, {
    ...options,
    type: 'attention',
    params: { beta: options?.beta ?? 0.1 },
  });
}
