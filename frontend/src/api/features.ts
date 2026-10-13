import { apiClient, type ApiRequestOptions } from './client';

export interface MutationComparisonInput {
  older_text: string;
  newer_text: string;
  older_timestamp?: string;
  newer_timestamp?: string;
  context_note?: string;
  provenance_url?: string;
}
export interface MutationComparison {
  analysis: { mutation_types: string[]; [key: string]: unknown };
  llm_summary?: string;
  observed_propagation?: boolean;
  warnings?: string[];
}

export type CorpusAssessment = 'SUPPORTED_BY_CORPUS' | 'CONTRADICTED_BY_CORPUS' | 'MIXED_EVIDENCE' | 'INSUFFICIENT_EVIDENCE';
export interface ClaimCheckInput { claim_text: string; as_of?: string; limit: number }
export interface CorpusEvidence {
  claim_id: string;
  article_id: string;
  text: string;
  passage: string | null;
  passage_source?: 'cleaned_content' | 'content' | 'extracted_claim';
  url: string | null;
  title: string | null;
  domain: string | null;
  published_at: string | null;
  similarity: number | null;
  retrieval_method?: 'vector' | 'keyword' | 'hybrid';
  collected_at?: string | null;
  extracted_at?: string | null;
  passage_start?: number | null;
  passage_end?: number | null;
  passage_sha256?: string | null;
  abstention_reason?: string | null;
  stance: 'entailment' | 'contradiction' | 'neutral';
  syndication_group: string | null;
}
export interface ClaimCheckResult {
  claim_text: string;
  assessment: CorpusAssessment;
  evidence: CorpusEvidence[];
  warnings: string[];
  score_kind: 'uncalibrated_heuristic';
  observed_at: string;
  method_version: string;
}

export interface ExternalCheckResult {
  claim: string;
  verdict: 'SUPPORTED' | 'DISPUTED' | 'SUPPORTED_BY_SOURCES' | 'CONTRADICTED_BY_SOURCES' | 'MIXED_EVIDENCE' | 'UNVERIFIABLE';
  explanation: string;
  observed_at?: string;
  method_version?: string;
  warnings?: string[];
  assessment?: string;
  sources: {
    title: string; url: string; passage?: string;
    stance?: string; abstention_reason?: string | null;
    revision_id?: number | null; published_at?: string | null;
    retrieved_at?: string; passage_sha256?: string;
  }[];
}

export const featuresApi = {
  compare: (input: MutationComparisonInput, options?: ApiRequestOptions) =>
    apiClient.post<MutationComparison>('/graph/mutation/compare', input, { timeoutMs: 45_000, ...options, authenticated: true }),
  check: (input: ClaimCheckInput, options?: ApiRequestOptions) =>
    apiClient.post<ClaimCheckResult>('/verdicts/check', input, { timeoutMs: 45_000, ...options, authenticated: true }),
  checkExternal: (input: ClaimCheckInput, options?: ApiRequestOptions) =>
    apiClient.post<ExternalCheckResult>('/verdicts/check-external', input, { timeoutMs: 60_000, ...options, authenticated: true }),
};

