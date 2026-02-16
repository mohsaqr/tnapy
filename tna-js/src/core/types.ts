/**
 * Core type definitions for TNA.
 */
import type { Matrix } from './matrix.js';

/** A sequence is a row of string tokens (states), possibly with null for missing. */
export type Sequence = (string | null)[];

/** A sequence dataset: array of sequences. */
export type SequenceData = Sequence[];

/**
 * TNA model type identifiers.
 * - 'relative': Row-normalized transition probabilities
 * - 'frequency': Raw transition counts
 * - 'co-occurrence': Bidirectional co-occurrence
 * - 'reverse': Reverse order transitions
 * - 'n-gram': Higher-order n-gram transitions
 * - 'gap': Non-adjacent transitions weighted by gap
 * - 'window': Sliding window transitions
 * - 'attention': Exponential decay weighted
 * - 'betweenness': Edge betweenness weights
 * - 'matrix': Direct matrix input
 */
export type ModelType =
  | 'relative'
  | 'frequency'
  | 'co-occurrence'
  | 'reverse'
  | 'n-gram'
  | 'gap'
  | 'window'
  | 'attention'
  | 'betweenness'
  | 'matrix';

/** Parameters for specific model types. */
export interface TransitionParams {
  /** Order for n-gram transitions. Default 2. */
  n?: number;
  /** Max gap for gap model. Default 5. */
  maxGap?: number;
  /** Decay factor for gap model. Default 0.5. */
  decay?: number;
  /** Window size for window model. Default 3. */
  size?: number;
  /** Decay parameter for attention model. Default 0.1. */
  beta?: number;
}

/** Options for building a TNA model. */
export interface BuildModelOptions {
  type?: ModelType;
  scaling?: string | string[] | null;
  labels?: string[];
  beginState?: string;
  endState?: string;
  params?: TransitionParams;
}

/** TNA model. */
export interface TNA {
  /** Adjacency/transition matrix (n_states x n_states). */
  weights: Matrix;
  /** Initial state probabilities (n_states). */
  inits: Float64Array;
  /** State labels. */
  labels: string[];
  /** Original sequence data (if built from sequences). */
  data: SequenceData | null;
  /** Model type. */
  type: ModelType;
  /** Scaling methods applied. */
  scaling: string[];
}

/** GroupTNA: mapping from group name to TNA model. */
export interface GroupTNA {
  models: Record<string, TNA>;
}

/** Centrality measure names. */
export type CentralityMeasure =
  | 'OutStrength'
  | 'InStrength'
  | 'ClosenessIn'
  | 'ClosenessOut'
  | 'Closeness'
  | 'Betweenness'
  | 'BetweennessRSP'
  | 'Diffusion'
  | 'Clustering';

/** Centrality result: map from state label to measure values. */
export interface CentralityResult {
  labels: string[];
  measures: Record<CentralityMeasure, Float64Array>;
  /** Optional group column for GroupTNA results. */
  groups?: string[];
}

/** Clique detection result. */
export interface CliqueResult {
  weights: Matrix[];
  indices: number[][];
  labels: string[][];
  size: number;
  threshold: number;
}

/** Community detection result. */
export interface CommunityResult {
  counts: Record<string, number>;
  assignments: Record<string, number[]>;
  labels: string[];
}

/** Community detection method. */
export type CommunityMethod =
  | 'fast_greedy'
  | 'louvain'
  | 'label_prop'
  | 'leading_eigen'
  | 'edge_betweenness'
  | 'walktrap';

/** Bootstrap result. */
export interface BootstrapResult {
  weightsOrig: Matrix;
  weightsSig: Matrix;
  weightsMean: Matrix;
  weightsSd: Matrix;
  pValues: Matrix;
  crLower: Matrix;
  crUpper: Matrix;
  ciLower: Matrix;
  ciUpper: Matrix;
  bootSummary: BootstrapEdge[];
  model: TNA;
  labels: string[];
  method: string;
  iter: number;
  level: number;
}

/** A single edge row in the bootstrap summary. */
export interface BootstrapEdge {
  from: string;
  to: string;
  weight: number;
  pValue: number;
  sig: boolean;
  crLower: number;
  crUpper: number;
  ciLower: number;
  ciUpper: number;
}

/** Permutation test edge statistics. */
export interface PermutationEdgeStat {
  edgeName: string;
  diffTrue: number;
  effectSize: number;
  pValue: number;
}

/** Permutation test result. */
export interface PermutationResult {
  edges: {
    stats: PermutationEdgeStat[];
    diffsTrue: Matrix;
    diffsSig: Matrix;
  };
  centralities?: {
    stats: { state: string; centrality: string; diffTrue: number; effectSize: number; pValue: number }[];
    diffsTrue: Matrix;
    diffsSig: Matrix;
  };
  labels: string[];
}

/** Cluster result. */
export interface ClusterResult {
  data: SequenceData;
  k: number;
  assignments: number[];
  silhouette: number;
  sizes: number[];
  method: string;
  distance: Matrix;
  dissimilarity: string;
}

/** Centrality stability result. */
export interface CentralityStabilityResult {
  csCoefficients: Record<string, number>;
  correlations: Record<string, Float64Array[]>;
  dropProp: number[];
  threshold: number;
  certainty: number;
}

/** Prepared data container (analogous to Python TNAData). */
export interface TNAData {
  sequenceData: SequenceData;
  labels: string[];
  statistics: {
    nSessions: number;
    nUniqueActions: number;
    uniqueActions: string[];
    maxSequenceLength: number;
    meanSequenceLength: number;
  };
}

/** Compare sequences result row. */
export interface CompareRow {
  pattern: string;
  frequencies: Record<string, number>;
  proportions: Record<string, number>;
  effectSize?: number;
  pValue?: number;
}
