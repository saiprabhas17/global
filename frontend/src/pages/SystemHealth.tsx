import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchHealth } from '../api/health';
import { useDashboardTab } from '../lib/routes';
import { useDocumentVisible, usePollingControl, usePollingPolicy } from '../lib/polling';
import { PollingProvider } from '../components/PollingProvider';
import { QueryError } from '../components/QueryError';
import { fetchDashboard } from '../api/dashboard';
import { Header } from '../components/Header';
import { MetricCard } from '../components/MetricCard';
import { GraphVisualizer } from '../components/GraphVisualizer';
import { VectorTelemetry } from '../components/VectorTelemetry';
import { LiveAuditFeed } from '../components/LiveAuditFeed';
import { EarlyWarning } from '../components/EarlyWarning';
import { CacheTelemetry } from '../components/CacheTelemetry';
import CorpusExplorer from '../components/CorpusExplorer';
import { FactCheck } from '../components/FactCheck';
import { EvalLab } from '../components/EvalLab';
import { OperationsPanel } from '../components/OperationsPanel';
import FullCorpusGraph from '../components/FullCorpusGraph';
import TimelineTunnel from '../components/TimelineTunnel';
import MasalaLab from '../components/MasalaLab';
import AsciiEqualizer from '../components/AsciiEqualizer';
import {
  Database,
  Network,
  HardDrive,
  ShieldCheck,
  FlaskConical,
  X,
  Play
} from 'lucide-react';


/* ------------------------------ Demo telemetry ----------------------------- */

interface DemoTelemetry {
  claimsIndexed: number;
  graphNodes: number;
  graphEdges: number;
  queueDepth: number;
  vectorDrift: number;
  latencyMs: number;
  hitRatio: number;
  revokedTokens: number;
  memoryMb: number;
}

const DEMO_BASE: DemoTelemetry = {
  claimsIndexed: 148920,
  graphNodes: 42118,
  graphEdges: 137540,
  queueDepth: 231,
  vectorDrift: 0.031,
  latencyMs: 4.2,
  hitRatio: 98.4,
  revokedTokens: 1409,
  memoryMb: 128.4
};

const jitter = (base: number, pct: number, decimals = 0): number => {
  const delta = base * pct;
  const next = base + (Math.random() * 2 - 1) * delta;
  return Number(next.toFixed(decimals));
};

/* ------------------------------- Skeleton UI ------------------------------- */

const SkeletonCard: React.FC = () => (
  <div className="card-brutal-dark p-5">
    <div className="flex items-center justify-between mb-4">
      <div className="shimmer w-11 h-11" />
      <div className="shimmer w-14 h-5" />
    </div>
    <div className="shimmer h-3 w-24 mb-2" />
    <div className="shimmer h-9 w-32" />
    <div className="shimmer h-3 w-full mt-4 pt-3 border-t border-hermes-bone/10" />
  </div>
);

const LoadingGrid: React.FC = () => (
  <div className="space-y-8" data-testid="loading-grid">
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
      {[0, 1, 2, 3].map(i => <SkeletonCard key={i} />)}
    </div>
    <div className="card-brutal-dark p-6">
      <div className="shimmer h-6 w-72 mb-2" />
      <div className="shimmer h-3 w-96 max-w-full mb-6" />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 shimmer h-64" />
        <div className="shimmer h-64" />
      </div>
    </div>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <div className="card-brutal-dark p-6 shimmer h-56" />
      <div className="card-brutal-dark p-6 shimmer h-56" />
    </div>
  </div>
);

/* --------------------------------- Component -------------------------------- */

const SystemHealthContent: React.FC = () => {
  const { interval: refetchInterval, setInterval: setRefetchInterval } = usePollingControl();
  const polling = usePollingPolicy(0);
  const visible = useDocumentVisible();
  const [activeTab, setActiveTab] = useDashboardTab();
  const [demoMode, setDemoMode] = useState(false);
  const [telemetry, setTelemetry] = useState<DemoTelemetry>(DEMO_BASE);

  const { data, error, isLoading, isFetching, refetch } = useQuery({
    queryKey: ['health'],
    queryFn: ({ signal }) => fetchHealth({ signal }),
    ...polling,
  });

  // Each dependency reports independently; readiness never gates usable tabs.
  const dashboard = useQuery({
    queryKey: ['dashboard'],
    queryFn: ({ signal }) => fetchDashboard({ signal }),
    ...polling,
  });
  const dash = dashboard.data;

  // Slowly drift the simulated metrics so the dashboard feels alive
  useEffect(() => {
    if (!demoMode || refetchInterval === 0 || !visible || !['overview', 'cache'].includes(activeTab)) return;
    const id = setInterval(() => {
      setTelemetry(prev => ({
        claimsIndexed: prev.claimsIndexed + Math.floor(Math.random() * 3),
        graphNodes: prev.graphNodes + (Math.random() > 0.6 ? 1 : 0),
        graphEdges: prev.graphEdges + Math.floor(Math.random() * 3),
        queueDepth: Math.max(12, jitter(prev.queueDepth, 0.06)),
        vectorDrift: Math.max(0.005, jitter(prev.vectorDrift, 0.08, 4)),
        latencyMs: Math.max(1.2, jitter(prev.latencyMs, 0.08, 1)),
        hitRatio: Math.min(99.9, Math.max(94, jitter(prev.hitRatio, 0.004, 1))),
        revokedTokens: prev.revokedTokens + (Math.random() > 0.85 ? 1 : 0),
        memoryMb: Math.max(96, jitter(prev.memoryMb, 0.015, 1))
      }));
    }, 5000); // gentle drift every 5s — calm, not twitchy
    return () => clearInterval(id);
  }, [demoMode, refetchInterval, visible, activeTab]);

  const isPostgresOk = data?.postgres === 'ok';
  const isNeo4jOk = data?.neo4j === 'ok';
  const isRedisOk = data?.redis === 'ok';
  const isAllOk = data?.status === 'ready';
  const headerStatus = error ? 'error' : data?.status ?? 'connecting';
  const servicesUp = [isPostgresOk, isNeo4jOk, isRedisOk].filter(Boolean).length;
  const count = (value: number | undefined) => value === undefined ? 'unavailable' : value.toLocaleString();

  return (
    <div className="min-h-screen flex flex-col items-center p-4 pb-0 md:p-8 md:pb-0 lg:p-12 lg:pb-0 relative bg-hermes-ink text-hermes-bone overflow-x-clip">
      {/* Paper grain texture — Hermes signature */}
      <div className="fixed inset-0 z-0 pointer-events-none grain-overlay" />

      <div className="w-full max-w-7xl z-10">
        {/* Header & Controls */}
        <Header
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          refetchInterval={refetchInterval}
          setRefetchInterval={setRefetchInterval}
          isFetching={isFetching}
          onManualRefresh={() => { void refetch(); void dashboard.refetch(); }}
          status={headerStatus}
          demoMode={demoMode}
        />

        {/* Decorative equalizer, never presented as measured telemetry. */}
        <div className="mt-6 mb-8 border border-hermes-bone/15 bg-hermes-panel px-4 py-3">
          <AsciiEqualizer count={56} className="text-lg" label="Decorative equalizer — not live telemetry" />
        </div>

        {/* DEMO MODE banner */}
        {demoMode && (
          <div className="mb-8 flex items-center justify-between gap-4 px-4 py-3 border border-hermes-ink/90 bg-hermes-red-bright text-hermes-bone shadow-[4px_4px_0_0_rgba(26,6,8,1)]">
            <div className="flex items-center gap-3 min-w-0">
              <span className="p-2 border border-hermes-ink/90 bg-hermes-bone shrink-0">
                <FlaskConical className="w-4 h-4 text-hermes-red-deep" />
              </span>
              <div className="min-w-0">
                <div className="text-xs font-mono font-bold text-hermes-bone tracking-wider uppercase">
                  Demo Telemetry — simulated data stream
                </div>
                <div className="text-[11px] font-mono text-hermes-bone/85 truncate">
                  Opt-in infrastructure simulator only · all corpus, graph and feature requests still use real API data
                </div>
              </div>
            </div>
            <button
              onClick={() => setDemoMode(false)}
              className="btn-ghost-brutal shrink-0 !py-1.5 !px-3"
              title="Hide simulated data and return to real metrics"
            >
              <X className="w-3.5 h-3.5" /> Exit Demo
            </button>
          </div>
        )}

        <section aria-label="Dependency readiness" className="mb-6 card-brutal-dark p-4 space-y-3">
          {isLoading && <p role="status" className="font-mono text-xs">Connecting to backend readiness… Tabs can fetch independently.</p>}
          {error && <QueryError title="Readiness unavailable" error={error} onRetry={refetch} retrying={isFetching} />}
          {data && <>
            <p className="font-mono text-xs text-hermes-bone/65">
              {error ? 'Last reported readiness' : 'Readiness'}: {data.status}.
              {data.status === 'not_ready' && ' A dependency is unavailable; usable tabs remain open.'}
            </p>
            <dl className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-xs">
              {(['postgres', 'neo4j', 'redis'] as const).map((service) => <div key={service}>
                <dt className="uppercase text-hermes-bone/50">{service}</dt>
                <dd className="break-words">{data[service] ?? 'not reported'}</dd>
              </div>)}
            </dl>
          </>}
          {!demoMode && <button type="button" onClick={() => setDemoMode(true)} className="btn-brutal">
            <Play className="w-3.5 h-3.5" /> Launch Demo Telemetry
          </button>}
        </section>
        {dashboard.isError && <div className="mb-6"><QueryError title="Dashboard metrics unavailable" error={dashboard.error} onRetry={() => dashboard.refetch()} retrying={dashboard.isFetching} />
          {dash && <p className="font-mono text-xs mt-2">Showing the last real snapshot from {dash.generated_at}; values may be stale.</p>}
        </div>}
        <OperationsPanel />

        {activeTab === 'overview' && isLoading && dashboard.isLoading && !demoMode ? <LoadingGrid /> : (
          /* Active Content Views Based on Tab */
          <div className="space-y-8">
            {activeTab === 'overview' && (
              <>
                {/* Metric Summary Banner */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
                  <MetricCard
                    title="PostgreSQL Vector"
                    value={demoMode ? `${telemetry.claimsIndexed.toLocaleString()} simulated claims` : dash?.claims_total !== undefined ? `${count(dash.claims_total)} claims` : isPostgresOk ? 'Ready · counts unavailable' : data ? 'Unavailable' : 'Unknown'}
                    subtitle={demoMode ? 'Simulated preview — not corpus data' : `${count(dash?.embedded_claims)} embedded · ${count(dash?.entities_total)} entities`} 
                    icon={Database}
                    accentColor={isPostgresOk ? "violet" : "rose"}
                    trend={isPostgresOk ? 'READY' : data?.postgres === undefined ? 'UNKNOWN' : 'NOT READY'}
                    trendPositive={isPostgresOk}
                  />
                  <MetricCard
                    title="Neo4j Graph Engine"
                    value={demoMode ? 'Simulated graph' : isNeo4jOk ? 'Ready' : data ? 'Unavailable' : 'Unknown'}
                    subtitle={demoMode
                      ? `${telemetry.graphNodes.toLocaleString()} simulated nodes · ${telemetry.graphEdges.toLocaleString()} simulated edges`
                      : dash?.graph?.available
                        ? `${Object.values(dash.graph.nodes).reduce((a, b) => a + b, 0).toLocaleString()} nodes · ${dash.graph.relationships.toLocaleString()} edges`
                        : 'Topology counts unavailable'}
                    icon={Network}
                    accentColor={isNeo4jOk ? "cyan" : "rose"}
                    trend={isNeo4jOk ? 'READY' : data?.neo4j === undefined ? 'UNKNOWN' : 'NOT READY'}
                    trendPositive={isNeo4jOk}
                  />
                  <MetricCard
                    title="Redis Cache & Queue"
                    value={demoMode ? 'Simulated cache' : dash?.cache?.available ? dash.cache.used_memory_human : isRedisOk ? 'Ready · metrics unavailable' : data ? 'Unavailable' : 'Unknown'}
                    subtitle={demoMode
                      ? `Simulated queue depth ${telemetry.queueDepth.toLocaleString()} jobs`
                      : dash?.cache?.available ? `In-memory store · ${dash.cache.used_memory_mb} MB allocated` : 'Queue depth and memory not reported'}
                    icon={HardDrive}
                    accentColor={isRedisOk ? "amber" : "rose"}
                    trend={isRedisOk ? 'READY' : data?.redis === undefined ? 'UNKNOWN' : 'NOT READY'}
                    trendPositive={isRedisOk}
                  />
                  <MetricCard
                    title="Infrastructure Readiness"
                    value={data ? `${servicesUp} / 3 ready` : 'Unknown'}
                    subtitle="Real dependency readiness · not a quality score"
                    icon={ShieldCheck}
                    accentColor={isAllOk ? "emerald" : "amber"}
                    trend={isAllOk ? 'READY' : data ? 'NOT READY' : 'CONNECTING'}
                    trendPositive={isAllOk}
                  />
                </div>

                {/* Early-warning board — persisted alerts + 24h spike snapshot */}
                <EarlyWarning />

                {/* Interactive Neo4j Topology Graph */}
                <GraphVisualizer setActiveTab={setActiveTab} />

                {/* Postgres Vector Telemetry & Audit Stream */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <VectorTelemetry />
                  <LiveAuditFeed />
                </div>
              </>
            )}

            {activeTab === 'graph' && (
              <div className="space-y-6">
                <GraphVisualizer setActiveTab={setActiveTab} />
                <LiveAuditFeed />
              </div>
            )}

            {activeTab === 'vector' && (
              <div className="space-y-6">
                <VectorTelemetry />
                <LiveAuditFeed />
              </div>
            )}

            {activeTab === 'timeline' && <TimelineTunnel />}
            {activeTab === 'masala' && <MasalaLab />}
            {activeTab === 'fullgraph' && <FullCorpusGraph setActiveTab={setActiveTab} />}
            {activeTab === 'factcheck' && <FactCheck />}
            {activeTab === 'eval' && <EvalLab />}
            {activeTab === 'corpus' && <CorpusExplorer />}

            {activeTab === 'cache' && (
              <CacheTelemetry
                hitRatio={demoMode ? `${telemetry.hitRatio.toFixed(1)}%` : undefined}
                latencyMs={demoMode ? `${telemetry.latencyMs.toFixed(1)}ms` : undefined}
                revokedTokens={demoMode ? `${telemetry.revokedTokens.toLocaleString()} simulated tokens` : undefined}
                memoryMb={demoMode ? `${telemetry.memoryMb.toFixed(1)} MB` : dash?.cache?.available ? `${dash.cache.used_memory_mb} MB` : undefined}
                simulated={demoMode}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export const SystemHealth: React.FC = () => <PollingProvider><SystemHealthContent /></PollingProvider>;
