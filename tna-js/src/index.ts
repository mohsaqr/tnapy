/**
 * TNA - Transition Network Analysis for JavaScript/TypeScript.
 *
 * @example
 * ```ts
 * import { tna, centralities, prune } from 'tna';
 *
 * const data = [
 *   ['A', 'B', 'C', 'A'],
 *   ['B', 'C', 'A', 'B'],
 *   ['A', 'C', 'B', 'A'],
 * ];
 *
 * const model = tna(data);
 * const cent = centralities(model);
 * const pruned = prune(model, 0.1);
 * ```
 */

// Core
export {
  // Matrix
  Matrix, rowNormalize, minmaxScale, maxScale, rankScale, applyScaling,
  arrayMean, arrayStd, pearsonCorr, arrayQuantile,
  // Model
  createTNA, buildModel, tna, ftna, ctna, atna, summary,
  // Prepare
  createSeqdata, prepareData, importOnehot,
  // Transitions
  computeTransitions, computeTransitions3D, computeWeightsFrom3D, computeWeightsFromMatrix,
  // Group
  isGroupTNA, createGroupTNA, groupNames, groupEntries, groupApply, renameGroups,
  groupTna, groupFtna, groupCtna, groupAtna,
  // Colors
  colorPalette, DEFAULT_COLORS, ACCENT_PALETTE, SET3_PALETTE,
  hexToRgb, rgbToHex, lightenColor, darkenColor, createColorMap,
  // RNG
  SeededRNG,
} from './core/index.js';

// Analysis
export {
  centralities, betweennessNetwork, AVAILABLE_MEASURES,
  prune,
  cliques,
  communities, AVAILABLE_METHODS,
  compareSequences,
  clusterSequences,
} from './analysis/index.js';

// Stats
export {
  bootstrapTna,
  permutationTest,
  estimateCs,
  confidenceInterval,
} from './stats/index.js';

// Types
export type {
  TNA, GroupTNA, TNAData, SequenceData, Sequence, ModelType, TransitionParams, BuildModelOptions,
  CentralityMeasure, CentralityResult, CliqueResult, CommunityResult, CommunityMethod,
  BootstrapResult, BootstrapEdge, PermutationResult, PermutationEdgeStat,
  ClusterResult, CentralityStabilityResult, CompareRow,
} from './core/types.js';
