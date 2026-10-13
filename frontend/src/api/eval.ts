import { apiClient, type ApiRequestOptions } from './client';

export type EvalLabel = 'SAME_STORY' | 'EVOLVED' | 'DISTINCT';
export type EvalStrategy = 'coverage' | 'uncertainty';
export type EvalSplit = 'unassigned' | 'train' | 'dev' | 'test';
export type ContextRelation = 'PARAPHRASE' | 'UPDATE_OR_CORRECTION' | 'INCOMPATIBLE_ASSERTION' | 'INSUFFICIENT_CONTEXT' | 'UNRELATED';
export interface EvalChangedSpan {
  side: 'a' | 'b'; start: number; end: number; text: string; mutation_type: string;
}
export interface EvalRubric {
  version: string;
  labels: Record<EvalLabel, string>;
  context_relations: Record<ContextRelation, string>;
  mutation_types: string[];
  span_offsets: string;
  guidance: string[];
}

export interface EvalClaimSide { text: string; domain: string | null }
export interface EvalPair { pair_id: string; a: EvalClaimSide; b: EvalClaimSide }
export interface EvalNext { done: boolean; pair: EvalPair | null }
export interface EvalBucketProgress { bucket: string; pairs: number; labeled: number; consensus?: number }
export interface EvalOriginProgress { votes: number; pairs: number; labels?: Record<string, number> }
export interface EvalProgress {
  total_pairs: number;
  labeled_votes: number;
  labeled_pairs_distinct: number;
  by_bucket: EvalBucketProgress[];
  per_annotator: Record<string, { labels: Record<string, number>; total: number }>;
  inter_annotator: Array<{
    annotators: string[]; pairs: number; kappa: number | null;
    raw_agreement?: number; origin?: string; warning?: string;
  }>;
  complete: boolean;
  origins?: Record<string, EvalOriginProgress>;
  human_votes?: number;
  consensus_pairs?: number;
  disagreement_pairs?: number;
  invalid_human_votes?: number;
  by_split?: Array<{ split: EvalSplit; pairs: number; labeled: number; consensus?: number }>;
  warning?: string;
  warnings?: string[];
}
export interface EvalAnnotation {
  mutation_types?: string[]; notes?: string;
  changed_spans?: EvalChangedSpan[];
  context_relation?: ContextRelation;
  rubric_version?: string;
}
export interface EvalLabelResult {
  status: string; pair_id: string; label: string;
  annotator?: string; origin?: string; mutation_types?: string[] | null; notes?: string | null;
}

export const evalApi = {
  rubric: (options?: ApiRequestOptions) => apiClient.get<EvalRubric>('/eval/rubric', { ...options, authenticated: true }),
  reviewNext: (split: EvalSplit = 'unassigned', options?: ApiRequestOptions) =>
    apiClient.get<EvalNext>(`/eval/review/next?${new URLSearchParams({ split })}`, { ...options, authenticated: true }),
  next: (strategy: EvalStrategy, split: EvalSplit = 'unassigned', options?: ApiRequestOptions) =>
    apiClient.get<EvalNext>(`/eval/next?${new URLSearchParams({ strategy, split })}`, { ...options, authenticated: true }),
  label: (pairId: string, label: EvalLabel, annotation: EvalAnnotation = {}) =>
    apiClient.post<EvalLabelResult>('/eval/label', { pair_id: pairId, label, ...annotation }, { authenticated: true }),
  progress: (options?: ApiRequestOptions) => apiClient.get<EvalProgress>('/eval/progress', { ...options, authenticated: true }),
  export: (datasetVersion: string, publication = false) =>
    apiClient.download(`/eval/export?${new URLSearchParams({ dataset_version: datasetVersion, publication: String(publication) })}`, { authenticated: true, timeoutMs: 45_000 }),
};
