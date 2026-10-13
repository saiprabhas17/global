import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Newspaper,
  Search,
  ExternalLink,
  Network,
  Radio,
  RefreshCw,
} from 'lucide-react';
import { corpusApi } from '../api/corpus';
import { usePollingPolicy } from '../lib/polling';
import { safeHttpUrl } from '../lib/urls';
import { QueryError } from './QueryError';

/**
 * CorpusExplorer — searchable real corpus listing,
 * live story clusters from Neo4j, and the true ingest stream.
 */

const PAGE = 25;

const CorpusExplorer: React.FC = () => {
  const [q, setQ] = useState('');
  const [query, setQuery] = useState('');
  const [domain, setDomain] = useState('');
  const [offset, setOffset] = useState(0);
  const polling = usePollingPolicy(15000);

  const list = useQuery({
    queryKey: ['corpus', query, domain, offset],
    queryFn: ({ signal }) => corpusApi.list({ q: query, domain, limit: PAGE, offset }, { signal }),
    ...polling,
  });

  const clusters = useQuery({
    queryKey: ['clusters'],
    queryFn: ({ signal }) => corpusApi.storyClusters(3, 4, { signal }),
    ...polling,
    refetchInterval: polling.refetchInterval === false ? false : Math.max(30000, polling.refetchInterval),
  });

  const recent = useQuery({
    queryKey: ['recent-articles'],
    queryFn: ({ signal }) => corpusApi.recent(10, { signal }),
    ...polling,
    refetchInterval: polling.refetchInterval === false ? false : Math.max(20000, polling.refetchInterval),
  });

  const submitSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setQuery(q.trim());
    setOffset(0);
  };

  return (
    <div className="space-y-8">
      {/* ------------------------- Story Clusters ------------------------- */}
      <section className="card-brutal-dark p-6">
        <div className="flex items-center justify-between pb-3 border-b border-hermes-bone/12 mb-5">
          <div className="flex items-center gap-2">
            <Network className="w-5 h-5" />
            <h2 className="text-2xl font-display">Story Clusters</h2>
          </div>
          <span className="font-mono text-[10px] uppercase tracking-widest text-hermes-bone/45">
            most-linked stories · neo4j
          </span>
        </div>

        {clusters.isLoading ? (
          <div role="status" aria-label="Loading story clusters" className="shimmer h-40" />
        ) : clusters.isError ? (
          <QueryError title="Story clusters unavailable" error={clusters.error} onRetry={() => clusters.refetch()} retrying={clusters.isFetching} />
        ) : clusters.data?.clusters.length === 0 ? (
          <p className="text-xs font-mono text-hermes-bone/55">No story clusters returned.</p>
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
            {clusters.data?.clusters.map((c) => (
              <div
                key={c.id}
                className="border border-hermes-bone/15 bg-hermes-panel-deep p-4 flex flex-col gap-3"
              >
                <div className="flex items-start justify-between gap-3">
                  <a
                    href={safeHttpUrl(c.url)}
                    target="_blank"
                    rel="noreferrer"
                    className="font-display text-lg leading-snug hover:text-hermes-red-bright transition-colors"
                  >
                    {c.title}
                  </a>
                  <span className="chip-brutal bg-hermes-red text-hermes-bone border-hermes-red shrink-0">
                    {c.deg} links
                  </span>
                </div>
                <div className="flex flex-col gap-1.5">
                  {c.neighbors.map((nb) => (
                    <a
                      key={nb.id}
                      href={safeHttpUrl(nb.url)}
                      target="_blank"
                      rel="noreferrer"
                      className="group flex items-center gap-2 font-mono text-[11px] text-hermes-bone/70 hover:text-hermes-bone"
                    >
                      <span className="w-9 shrink-0 text-right tabular-nums text-hermes-red-bright">
                        {nb.score.toFixed(2)}
                      </span>
                      <span className="w-32 shrink-0 truncate uppercase tracking-wider text-hermes-bone/40 group-hover:text-hermes-bone/70">
                        {nb.domain?.replace('www.', '')}
                      </span>
                      <span className="truncate">{nb.title}</span>
                    </a>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* -------------------------- Corpus search ------------------------- */}
      <section className="card-brutal-dark p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-hermes-bone/12 mb-5">
          <div className="flex items-center gap-2">
            <Newspaper className="w-5 h-5" />
            <h2 className="text-2xl font-display">Article Corpus</h2>
            {!list.isLoading && list.data && (
              <span className="chip-brutal border-hermes-bone/30 text-hermes-bone/60">
                {list.data.total.toLocaleString()} articles
              </span>
            )}
          </div>

          <form onSubmit={submitSearch} className="flex flex-col sm:flex-row sm:items-center gap-2 min-w-0">
            <select
              aria-label="Corpus outlet filter"
              value={domain}
              onChange={(e) => {
                setDomain(e.target.value);
                setOffset(0);
              }}
              className="bg-hermes-panel-deep border border-hermes-bone/25 text-hermes-bone font-mono text-xs px-2 py-1.5 outline-none cursor-pointer"
            >
              <option value="">all outlets</option>
              {list.data?.domains.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name.replace('www.', '')} ({d.count})
                </option>
              ))}
            </select>
            <div className="flex items-center min-w-0 border border-hermes-bone/25 bg-hermes-panel-deep">
              <input
                aria-label="Corpus search query"
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="search titles & authors…"
                className="bg-transparent text-hermes-bone font-mono text-xs px-3 py-1.5 outline-none w-full min-w-0 sm:w-48 md:w-64 placeholder:text-hermes-bone/35"
              />
              <button type="submit" aria-label="Search corpus" className="btn-brutal !py-1.5 !px-2.5 m-0.5">
                <Search className="w-3.5 h-3.5" />
              </button>
            </div>
          </form>
        </div>

        {list.isLoading ? (
          <div role="status" aria-label="Loading corpus articles" className="space-y-2">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="shimmer h-12" />
            ))}
          </div>
        ) : list.isError ? (
          <QueryError title="Corpus articles unavailable" error={list.error} onRetry={() => list.refetch()} retrying={list.isFetching} />
        ) : (
          <>
            {list.data?.items.length === 0 && <p className="text-xs font-mono text-hermes-bone/55 py-4">No articles match this search.</p>}
            <div className="divide-y divide-hermes-bone/8">
              {list.data?.items.map((a) => (
                <a
                  key={a.id}
                  href={safeHttpUrl(a.url)}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center justify-between gap-4 py-2.5 group"
                >
                  <div className="min-w-0 flex flex-col gap-0.5">
                    <span className="truncate text-sm font-medium text-hermes-bone/90 group-hover:text-white">
                      {a.title}
                    </span>
                    <span className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-hermes-bone/40">
                      <span className="text-hermes-red-bright/80">
                        {a.domain?.replace('www.', '')}
                      </span>
                      {a.author && <span>· {a.author}</span>}
                      {a.published_at && (
                        <span>· {new Date(a.published_at).toLocaleDateString()}</span>
                      )}
                      {a.word_count != null ? <span>· {a.word_count}w</span> : null}
                    </span>
                  </div>
                  <ExternalLink className="w-3.5 h-3.5 shrink-0 text-hermes-bone/30 group-hover:text-hermes-red-bright" />
                </a>
              ))}
            </div>

            <div className="flex items-center justify-between pt-4 mt-2 border-t border-hermes-bone/12">
              <span className="font-mono text-[10px] uppercase tracking-widest text-hermes-bone/40">
                offset {offset} · showing {list.data?.items.length ?? 0}
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setOffset(Math.max(0, offset - PAGE))}
                  disabled={offset === 0}
                  className="btn-ghost-brutal !py-1 !px-3 !text-hermes-bone !border-hermes-bone/30 disabled:opacity-30"
                >
                  Prev
                </button>
                <button
                  onClick={() => setOffset(offset + PAGE)}
                  disabled={(list.data?.items.length ?? 0) < PAGE}
                  className="btn-ghost-brutal !py-1 !px-3 !text-hermes-bone !border-hermes-bone/30 disabled:opacity-30"
                >
                  Next
                </button>
              </div>
            </div>
          </>
        )}
      </section>

      {/* ------------------------ Live ingest stream ----------------------- */}
      <section className="card-brutal-dark p-6">
        <div className="flex items-center justify-between pb-3 border-b border-hermes-bone/12 mb-4">
          <div className="flex items-center gap-2">
            <Radio className="w-5 h-5" />
            <h2 className="text-2xl font-display">Ingest Stream</h2>
          </div>
          <span className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-widest text-hermes-bone/45">
            <RefreshCw className={`w-3 h-3 ${recent.isFetching ? 'animate-spin' : ''}`} aria-hidden="true" /> newest collections · postgres
          </span>
        </div>

        {recent.isLoading ? (
          <div role="status" aria-label="Loading ingest stream" className="shimmer h-32" />
        ) : recent.isError ? (
          <QueryError title="Ingest stream unavailable" error={recent.error} onRetry={() => recent.refetch()} retrying={recent.isFetching} />
        ) : recent.data?.items.length === 0 ? (
          <p className="text-xs font-mono text-hermes-bone/55">No collected articles returned.</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 divide-y md:divide-y-0 divide-hermes-bone/8">
            {recent.data?.items.map((r) => (
              <a
                key={r.id}
                href={safeHttpUrl(r.url)}
                target="_blank"
                rel="noreferrer"
                className="group py-2 flex flex-col gap-0.5 border-b border-hermes-bone/8 last:border-0"
              >
                <span className="truncate text-xs font-medium text-hermes-bone/85 group-hover:text-white">
                  {r.title}
                </span>
                <span className="font-mono text-[10px] uppercase tracking-wider text-hermes-bone/40">
                  <span className="text-hermes-red-bright/80">
                    {(r.source_name || r.domain || 'unknown').replace('www.', '')}
                  </span>
                  {r.collected_at && (
                    <> · collected {new Date(r.collected_at).toLocaleTimeString()}</>
                  )}
                </span>
              </a>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};

export default CorpusExplorer;
