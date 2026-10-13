import { useEffect, useId, useRef, useState, type FormEvent } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Dna } from 'lucide-react';
import { featuresApi, type MutationComparisonInput } from '../api/features';
import { useSession } from '../lib/session';
import { QueryError } from './QueryError';
import { SignInNotice } from './SignInNotice';
import { safeHttpUrl } from '../lib/urls';

const RELATIONS: Record<string, string> = {
  EQUIVALENT_QUANTITY: 'Equivalent quantities',
  NO_LEXICAL_CHANGE: 'No wording change',
  REPORTED_CORRECTION: 'Reported correction cue',
  REPORTED_UPDATE: 'Reported update cue',
  CONTEXT_REQUIRED: 'More context needed',
  TEXTUAL_CHANGE: 'Wording changed',
};

function returnedSpanText(value: unknown, side: 'older' | 'newer'): string | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const record = value as Record<string, unknown>;
  const span = record[`${side}_span`];
  if (span && typeof span === 'object' && 'text' in span && typeof span.text === 'string') return span.text;
  const legacy = record[side === 'older' ? 'old' : 'new'];
  return typeof legacy === 'string' ? legacy : undefined;
}

function ComparisonForm() {
  const [older, setOlder] = useState('');
  const [newer, setNewer] = useState('');
  const [olderTimestamp, setOlderTimestamp] = useState('');
  const [newerTimestamp, setNewerTimestamp] = useState('');
  const [contextNote, setContextNote] = useState('');
  const [provenanceUrl, setProvenanceUrl] = useState('');
  const controller = useRef<AbortController | null>(null);
  const id = useId();
  useEffect(() => () => controller.current?.abort(), []);
  const comparison = useMutation({
    mutationFn: (input: MutationComparisonInput) => {
      controller.current = new AbortController();
      return featuresApi.compare(input, { signal: controller.current.signal });
    },
  });
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!comparison.isPending && older.trim() && newer.trim() && validContext) comparison.mutate(input());
  };
  const validContext = !provenanceUrl.trim() || !!safeHttpUrl(provenanceUrl.trim());
  const input = (): MutationComparisonInput => ({
    older_text: older, newer_text: newer,
    ...(olderTimestamp ? { older_timestamp: new Date(olderTimestamp).toISOString() } : {}),
    ...(newerTimestamp ? { newer_timestamp: new Date(newerTimestamp).toISOString() } : {}),
    ...(contextNote.trim() ? { context_note: contextNote.trim() } : {}),
    ...(provenanceUrl.trim() ? { provenance_url: provenanceUrl.trim() } : {}),
  });
  const result = comparison.data;
  const observedPropagation = result?.observed_propagation ?? result?.analysis.observed_propagation;
  const limitations = Array.isArray(result?.analysis.limitations) ? result.analysis.limitations.filter((value: unknown): value is string => typeof value === 'string') : [];
  const spans = result && (Array.isArray(result.analysis.changed_spans) ? result.analysis.changed_spans
    : Array.isArray(result.analysis.segments) ? result.analysis.segments : []);

  return (
    <div className="space-y-4">
      <form onSubmit={submit} className="space-y-4">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div>
            <label htmlFor={`${id}-older`} className="block text-xs font-mono mb-2">Older text</label>
            <textarea id={`${id}-older`} required maxLength={2000} rows={5} value={older} disabled={comparison.isPending} onChange={(event) => { setOlder(event.target.value); comparison.reset(); }} className="field-brutal w-full" />
          </div>
          <div>
            <label htmlFor={`${id}-newer`} className="block text-xs font-mono mb-2">Newer text</label>
            <textarea id={`${id}-newer`} required maxLength={2000} rows={5} value={newer} disabled={comparison.isPending} onChange={(event) => { setNewer(event.target.value); comparison.reset(); }} className="field-brutal w-full" />
          </div>
        </div>
        <details className="border border-hermes-bone/20 p-3 space-y-3">
          <summary className="text-xs font-mono cursor-pointer">Temporal context and source reference (optional)</summary>
          <p className="text-xs text-hermes-bone/60">Supply publication times and context for a possible update or correction. A supplied link is a reference to review; it does not prove copying or establish accuracy.</p>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            <div><label htmlFor={`${id}-older-time`} className="block text-xs font-mono mb-1">Older publication time (local)</label>
              <input id={`${id}-older-time`} type="datetime-local" value={olderTimestamp} disabled={comparison.isPending} onChange={(event) => { setOlderTimestamp(event.target.value); comparison.reset(); }} className="field-brutal w-full" /></div>
            <div><label htmlFor={`${id}-newer-time`} className="block text-xs font-mono mb-1">Newer publication time (local)</label>
              <input id={`${id}-newer-time`} type="datetime-local" value={newerTimestamp} disabled={comparison.isPending} onChange={(event) => { setNewerTimestamp(event.target.value); comparison.reset(); }} className="field-brutal w-full" /></div>
          </div>
          <div><label htmlFor={`${id}-context`} className="block text-xs font-mono mb-1">Context note</label>
            <textarea id={`${id}-context`} value={contextNote} rows={3} maxLength={2000} disabled={comparison.isPending} onChange={(event) => { setContextNote(event.target.value); comparison.reset(); }} className="field-brutal w-full" /></div>
          <div><label htmlFor={`${id}-provenance`} className="block text-xs font-mono mb-1">Source reference URL</label>
            <input id={`${id}-provenance`} type="url" value={provenanceUrl} maxLength={2000} disabled={comparison.isPending} onChange={(event) => { setProvenanceUrl(event.target.value); comparison.reset(); }} className="field-brutal w-full" /></div>
          {!validContext && <p role="alert" className="text-xs text-hermes-red-bright">Use an HTTP/HTTPS source URL.</p>}
          {olderTimestamp && newerTimestamp && Date.parse(newerTimestamp) <= Date.parse(olderTimestamp) && <p role="note" className="text-xs text-amber-300">These times do not establish a later version. The comparison will report this ordering limitation.</p>}
        </details>
        <div className="flex flex-wrap gap-3">
          <button type="submit" disabled={comparison.isPending || !older.trim() || !newer.trim() || !validContext} className="btn-brutal">{comparison.isPending ? 'Comparing texts…' : 'Compare mutations'}</button>
          {comparison.isPending && <button type="button" onClick={() => controller.current?.abort()} className="btn-brutal">Cancel comparison</button>}
        </div>
      </form>
      {comparison.isIdle && <p className="text-xs font-mono text-hermes-bone/55">Enter two texts to request an analysis. No comparison has been run.</p>}
      {comparison.isPending && <p role="status" className="text-xs font-mono">Waiting for mutation analysis…</p>}
      {comparison.isError && <QueryError title="Mutation comparison unavailable" error={comparison.error} onRetry={() => { if (validContext) comparison.mutate(input()); }} />}
      {result && <div aria-live="polite" className="border border-hermes-bone/20 bg-hermes-panel-deep p-4 space-y-4">
        <h3 className="font-display text-xl">Returned mutation analysis</h3>
        <p className="text-xs font-mono text-amber-300">{observedPropagation === false ? 'No observed propagation — this compares wording only, not how a story spread.' : 'Propagation observation is not confirmed by this text comparator.'}</p>
        {typeof result.analysis.relation_category === 'string' && <div className="border border-hermes-bone/20 p-3 space-y-2">
          <h4 className="font-display text-lg">{RELATIONS[result.analysis.relation_category] ?? result.analysis.relation_category}</h4>
          {typeof result.analysis.relation_explanation === 'string' && <p className="text-sm">{result.analysis.relation_explanation}</p>}
          {result.analysis.quantity_equivalent === true && <p className="text-xs">The normalized quantities are equivalent despite different surface forms or units.</p>}
          <p className="text-xs text-hermes-bone/60">Correction and update cues describe the supplied text; independent evidence is needed to verify the account.</p>
        </div>}
        <div className="flex flex-wrap gap-2">
          {result.llm_summary && (
            <div className="w-full border border-emerald-500/30 bg-emerald-500/10 p-3 mb-2">
              <span className="font-mono text-[10px] uppercase tracking-widest text-emerald-400/80 mb-1 block">Generated wording summary</span>
              <p className="text-sm text-emerald-100">{result.llm_summary}</p>
            </div>
          )}
          {(result.analysis.mutation_types ?? []).map((type) => <span key={type} className="chip-brutal border-hermes-red-bright/50 text-hermes-red-bright">{type}</span>)}
          {result.analysis.mutation_types?.length === 0 && <p className="text-xs text-hermes-bone/60">No mutation types reported. This does not establish factual equivalence.</p>}
        </div>
        {spans && spans.length > 0 && <div className="space-y-2">
          <h4 className="text-xs font-mono">Changed spans / diff returned by the API</h4>
          {spans.map((span: unknown, index: number) => {
            const oldText = returnedSpanText(span, 'older');
            const newText = returnedSpanText(span, 'newer');
            return <div key={index} className="text-sm border border-hermes-bone/15 p-3 space-y-2">
              {oldText !== undefined || newText !== undefined ? <>
                <p>Older: <del className="text-hermes-red-bright whitespace-pre-wrap">{oldText === undefined ? 'not supplied' : `“${oldText}”`}</del></p>
                <p>Newer: <ins className="text-emerald-300 whitespace-pre-wrap">{newText === undefined ? 'not supplied' : `“${newText}”`}</ins></p>
                <details><summary className="text-xs font-mono cursor-pointer">Returned span metadata</summary><pre className="whitespace-pre-wrap break-words text-xs">{JSON.stringify(span, null, 2)}</pre></details>
              </> : <pre className="whitespace-pre-wrap break-words text-xs">{typeof span === 'string' ? span : JSON.stringify(span, null, 2)}</pre>}
            </div>;
          })}
        </div>}
        {typeof result.analysis.similarity === 'number' && <p className="text-xs font-mono">Text similarity: {result.analysis.similarity.toFixed(3)} (uncalibrated matching signal)</p>}
        {typeof result.analysis.algorithm_version === 'string' && <p className="text-xs font-mono">Algorithm: {result.analysis.algorithm_version}</p>}
        {limitations.length > 0 && <ul className="list-disc pl-5 text-xs text-amber-300">{limitations.map((limitation, index) => <li key={index}>{limitation}</li>)}</ul>}
        {result.warnings?.length ? <ul className="list-disc pl-5 text-xs text-amber-300">{result.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul> : null}
        <p className="text-xs font-mono text-hermes-bone/55">Any similarity or confidence fields below are unvalidated heuristic signals, not calibrated probabilities.</p>
        <details><summary className="cursor-pointer text-xs font-mono">Full API analysis (including additive diff fields)</summary><pre className="mt-2 whitespace-pre-wrap break-words text-xs text-hermes-bone/75">{JSON.stringify(result.analysis, null, 2)}</pre></details>
      </div>}
    </div>
  );
}

export function MutationComparator() {
  const { user, revision } = useSession();
  return (
    <section aria-labelledby="mutation-comparator-title" className="card-brutal-dark p-6 space-y-4">
      <h2 id="mutation-comparator-title" className="text-2xl font-display flex items-center gap-2"><Dna className="w-5 h-5" /> Mutation Comparator</h2>
      <p className="text-sm text-hermes-bone/65">Compare two supplied versions for changed numbers, entities and wording. Text changes do not prove misinformation, direction of transmission, or observed propagation.</p>
      {user ? <ComparisonForm key={revision} /> : <SignInNotice feature="the mutation comparator" />}
    </section>
  );
}
