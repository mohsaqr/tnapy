/**
 * Core module: TNA model, matrix, preparation, transitions, colors, RNG.
 */

// Matrix and utilities
export { Matrix, rowNormalize, minmaxScale, maxScale, rankScale, applyScaling, arrayMean, arrayStd, pearsonCorr, arrayQuantile } from './matrix.js';

// Types
export type {
  TNA, GroupTNA, TNAData, SequenceData, Sequence, ModelType, TransitionParams, BuildModelOptions,
  CentralityMeasure, CentralityResult, CliqueResult, CommunityResult, CommunityMethod,
  BootstrapResult, BootstrapEdge, PermutationResult, PermutationEdgeStat,
  ClusterResult, CentralityStabilityResult, CompareRow,
} from './types.js';

// Model
export { createTNA, buildModel, tna, ftna, ctna, atna, summary } from './model.js';

// Data preparation
export { createSeqdata, prepareData, importOnehot } from './prepare.js';

// Transitions
export { computeTransitions, computeTransitions3D, computeWeightsFrom3D, computeWeightsFromMatrix } from './transitions.js';

// Group models
export { isGroupTNA, createGroupTNA, groupNames, groupEntries, groupApply, renameGroups, groupTna, groupFtna, groupCtna, groupAtna } from './group.js';

// Colors
export { colorPalette, DEFAULT_COLORS, ACCENT_PALETTE, SET3_PALETTE, hexToRgb, rgbToHex, lightenColor, darkenColor, createColorMap } from './colors.js';

// RNG
export { SeededRNG } from './rng.js';
