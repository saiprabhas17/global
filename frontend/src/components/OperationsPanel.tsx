import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { apiClient } from '../api/client';
import { useSession } from '../lib/session';
import { QueryError } from './QueryError';

interface Job {
  id: string;
  job_type: string;
  status: string;
  started_at: string;
  error: string | null;
}
interface Recovery { stage: string; requeued: number; failed: number; dry_run: boolean }

const actions = [
  ['Collect sources', '/collection/trigger'],
  ['Process articles', '/preprocessing/trigger'],
  ['Extract claims', '/nlp/trigger'],
  ['Analyze evolution', '/evolution/trigger'],
  ['Sync graph', '/evolution/graph/sync'],
] as const;

function OperatorControls() {
  const queryClient = useQueryClient();
  const [stage, setStage] = useState<'nlp' | 'preprocessing'>('nlp');
  const [preview, setPreview] = useState<Recovery | null>(null);
  const [notice, setNotice] = useState('');
  const jobs = useQuery({
    queryKey: ['operations-jobs'],
    queryFn: ({ signal }) => apiClient.get<{ items: Job[] }>('/operations/jobs', { authenticated: true, signal }),
    refetchInterval: 10_000,
  });
  const action = useMutation({
    mutationFn: ({ path }: { path: string; label: string }) => apiClient.post(path, {}, { authenticated: true, timeoutMs: 600_000 }),
    onSuccess: (_data, variables) => {
      setNotice(variables.path === '/evolution/graph/sync'
        ? 'Sync graph completed. Graph views refreshed.'
        : `${variables.label} completed. Inspect the recorded run for results.`);
      void queryClient.invalidateQueries();
    },
  });
  const recovery = useMutation({
    mutationFn: (dry_run: boolean) => apiClient.post<Recovery>('/operations/recover', { stage, dry_run }, { authenticated: true }),
    onSuccess: (result) => {
      setPreview(result.dry_run ? result : null);
      setNotice(result.dry_run ? `${result.requeued} expired articles can retry; ${result.failed} have exhausted retries.` : `${result.requeued} articles requeued; ${result.failed} marked failed.`);
      void queryClient.invalidateQueries({ queryKey: ['operations-jobs'] });
    },
  });
  const busy = action.isPending || recovery.isPending;
  return <div className="space-y-4 mt-4">
    <p className="text-sm text-hermes-bone/65">Run pipeline stages and inspect their recorded results. Collection also starts preprocessing and NLP. Long jobs may take several minutes.</p>
    <div className="flex flex-wrap gap-2">{actions.map(([label, path]) => <button key={path} type="button" className="btn-brutal" disabled={busy} onClick={() => { setNotice(''); action.mutate({ path, label }); }}>{label}</button>)}</div>
    {action.isPending && <p role="status">{action.variables?.label} is running…</p>}
    {notice && <p role="status" className="text-xs font-mono">{notice}</p>}
    {action.error && <QueryError title="Pipeline action failed" error={action.error} />}
    {recovery.error && <QueryError title="Recovery failed" error={recovery.error} />}
    <div className="flex flex-wrap items-center gap-2">
      <label className="text-xs font-mono">Recovery stage <select aria-label="Recovery stage" className="field-brutal ml-2" value={stage} disabled={busy} onChange={(event) => { setStage(event.target.value as typeof stage); setPreview(null); }}><option value="nlp">Claim extraction</option><option value="preprocessing">Preprocessing</option></select></label>
      <button type="button" className="btn-brutal" disabled={busy} onClick={() => recovery.mutate(true)}>Preview expired work</button>
      <button type="button" className="btn-brutal" disabled={busy || !preview || !(preview.requeued + preview.failed)} onClick={() => recovery.mutate(false)}>Recover expired work</button>
    </div>
    <p className="text-xs text-hermes-bone/55">Recovery preserves active leases. It retries expired work within the attempt limit.</p>
    {jobs.isLoading && <p role="status">Loading recent runs…</p>}
    {jobs.error && <QueryError title="Recent runs unavailable" error={jobs.error} onRetry={() => void jobs.refetch()} />}
    {jobs.data && <div className="overflow-x-auto"><table className="w-full text-xs font-mono"><caption className="text-left mb-2">Recent pipeline runs</caption><thead><tr><th className="text-left">Stage</th><th className="text-left">Status</th><th className="text-left">Started</th></tr></thead><tbody>{jobs.data.items.map((job) => <tr key={job.id}><td className="py-2">{job.job_type}</td><td>{job.status}{job.error && <p className="text-hermes-orange">{job.error}</p>}</td><td>{new Date(job.started_at).toLocaleString()}</td></tr>)}</tbody></table>{jobs.data.items.length === 0 && <p className="text-xs">No pipeline runs recorded yet.</p>}</div>}
  </div>;
}

export function OperationsPanel() {
  const { user, revision } = useSession();
  const [open, setOpen] = useState(false);
  if (user?.role.toLowerCase() !== 'admin') return null;
  return <section className="card-brutal-dark p-4" aria-label="Pipeline operations">
    <button type="button" className="btn-brutal" aria-expanded={open} onClick={() => setOpen((value) => !value)}>Pipeline operations</button>
    {open && <OperatorControls key={revision} />}
  </section>;
}
