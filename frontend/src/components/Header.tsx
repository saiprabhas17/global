import React from 'react';
import { Activity, RefreshCw, Layers, Database, Network, HardDrive, FlaskConical, Newspaper, ShieldCheck, Share2, Orbit, Sparkles, ClipboardCheck } from 'lucide-react';
import { AsciiSpinner } from './AsciiSpinner';
import { AccountPanel } from './AccountPanel';
import { useDocumentVisible } from '../lib/polling';

export type TabType = 'overview' | 'factcheck' | 'fullgraph' | 'timeline' | 'masala' | 'graph' | 'vector' | 'cache' | 'corpus' | 'eval';

interface HeaderProps {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
  refetchInterval: number;
  setRefetchInterval: (interval: number) => void;
  isFetching: boolean;
  onManualRefresh: () => void;
  status: 'connecting' | 'ready' | 'not_ready' | 'error';
  demoMode?: boolean;
}

const STATUS_STYLES = {
  connecting: {
    spinMs: 120,
    spinClass: 'text-hermes-bone/60',
    badge: 'border-hermes-bone/25 bg-hermes-panel text-hermes-bone/70',
    label: 'Connecting'
  },
  ready: {
    spinMs: 120,
    spinClass: 'text-emerald-300',
    badge: 'border-hermes-bone/25 bg-hermes-panel text-emerald-300',
    label: 'Systems Ready'
  },
  not_ready: {
    spinMs: 120,
    spinClass: 'text-amber-300',
    badge: 'border-hermes-bone/25 bg-hermes-panel text-amber-300',
    label: 'Not Ready · Partial Outage'
  },
  error: {
    spinMs: 90,
    spinClass: 'text-hermes-red',
    badge: 'border-hermes-bone/25 bg-hermes-panel text-hermes-red-bright',
    label: 'Connection Failed'
  }
} as const;

const TICKER_ITEMS = [
  'GMEE // GLOBAL MISINFORMATION EVOLUTION ENGINE',
  'LIVE CORPUS & CLAIM CHANGE TELEMETRY',
  'PGVECTOR · NEO4J · REDIS',
  'V1.0'
];

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  refetchInterval,
  setRefetchInterval,
  isFetching,
  onManualRefresh,
  status,
  demoMode = false
}) => {
  const statusStyle = STATUS_STYLES[status];
  const visible = useDocumentVisible();

  return (
    <header className="sticky top-0 z-40 -mx-4 md:-mx-8 lg:-mx-12 mb-8">
      <div className="bg-hermes-ink/95 backdrop-blur-sm border-b border-hermes-bone/15">
        {/* Marquee ticker — Hermes-site signature */}
        <div className={`overflow-hidden border-b border-hermes-ink/20 bg-hermes-red text-hermes-bone ${visible ? '' : 'marquee-paused'}`}>
          <div className="marquee-track py-1">
            {[0, 1].map(dup => (
              <div key={dup} className="flex shrink-0 items-center" aria-hidden={dup === 1}>
                {TICKER_ITEMS.concat(TICKER_ITEMS).map((item, i) => (
                  <span
                    key={`${dup}-${i}`}
                    className="px-6 text-[10px] font-mono font-bold uppercase tracking-[0.2em] whitespace-nowrap"
                  >
                    {item} <span className="ml-6 opacity-60">◆</span>
                  </span>
                ))}
              </div>
            ))}
          </div>
        </div>

        <div className="px-4 md:px-8 lg:px-12 pt-5 flex flex-col gap-4">
          {/* Top Banner */}
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-5 border-b border-hermes-bone/12">
            <div className="flex items-center gap-4">
              <div className={`flex items-center justify-center w-12 h-12 border border-hermes-bone/30 shadow-[3px_3px_0_0_rgba(26,6,8,1)] ${
                demoMode ? 'bg-hermes-red-bright' : 'bg-hermes-red'
              }`}>
                <Activity className="w-6 h-6 text-hermes-bone" />
              </div>
              <div>
                <div className="flex items-center gap-3">
                  <h1 className="text-3xl md:text-4xl font-display leading-none">
                    GMEE
                  </h1>
                  {demoMode ? (
                    <span className="chip-brutal bg-hermes-red-bright text-hermes-bone animate-pulse">
                      <FlaskConical className="w-3 h-3" />
                      Demo Telemetry
                    </span>
                  ) : (
                    <span className="chip-brutal bg-hermes-red text-hermes-bone">v1.0 telemetry</span>
                  )}
                </div>
                <p className="text-xs font-mono uppercase tracking-widest text-hermes-bone/60 mt-1">
                  Global Misinformation Evolution Engine
                </p>
              </div>
            </div>

            {/* Status + Controls */}
            <div className="flex flex-wrap items-center gap-3">
              <div className={`flex items-center gap-2 px-3 py-1.5 border text-xs font-semibold tracking-wide uppercase font-mono ${statusStyle.badge}`}>
                {status === 'connecting' || isFetching ? <AsciiSpinner advanceMs={statusStyle.spinMs} className={statusStyle.spinClass} label={statusStyle.label} /> : <span aria-hidden="true" className={statusStyle.spinClass}>|</span>}
                {statusStyle.label}
                <span className="ml-1 normal-case text-hermes-bone/50">(readiness)</span>
              </div>

              <AccountPanel />

              {/* Pauses background polling across all mounted dashboard panels. */}
              <div className="flex items-center border border-hermes-bone/25 bg-hermes-panel p-0.5 text-xs">
                <select
                  aria-label="Background polling interval"
                  value={refetchInterval}
                  onChange={(e) => setRefetchInterval(Number(e.target.value))}
                  className="bg-transparent text-hermes-bone border-none outline-none py-1 pl-2 pr-1 font-mono cursor-pointer"
                >
                  <option value={2000}>2s live</option>
                  <option value={5000}>5s standard</option>
                  <option value={15000}>15s slow</option>
                  <option value={0}>Paused</option>
                </select>
              </div>

              {/* Manual refresh */}
              <button
                onClick={onManualRefresh}
                disabled={isFetching}
                className="btn-brutal !px-2.5 !py-2"
                title="Force Telemetry Sync"
                                aria-label="Refresh telemetry"
              >
                <RefreshCw className={`w-4 h-4 ${isFetching ? 'animate-spin' : ''}`} />
              </button>
            </div>
          </div>

          {/* Navigation Tabs — flat segmented control */}
          <nav aria-label="Dashboard sections" className="flex items-center gap-0 overflow-x-auto">
            <TabButton id="overview" label="Overview Vitals" icon={Layers} activeTab={activeTab} setActiveTab={setActiveTab} />
            <TabButton id="graph" label="Neo4j Graph Engine" icon={Network} badge="7687" activeTab={activeTab} setActiveTab={setActiveTab} />
            <TabButton id="vector" label="Postgres Vector" icon={Database} badge="pgvector" activeTab={activeTab} setActiveTab={setActiveTab} />
            <TabButton id="cache" label="Redis Queue" icon={HardDrive} badge="6379" activeTab={activeTab} setActiveTab={setActiveTab} />
            <TabButton id="fullgraph" label="Graph" icon={Share2} activeTab={activeTab} setActiveTab={setActiveTab} />
            <TabButton id="timeline" label="Timeline" icon={Orbit} badge="3D" activeTab={activeTab} setActiveTab={setActiveTab} />
            <TabButton id="masala" label="Masala Lab" icon={Sparkles} badge="NEW" activeTab={activeTab} setActiveTab={setActiveTab} />
            <TabButton id="factcheck" label="FactCheck" icon={ShieldCheck} badge="VERDICTS" activeTab={activeTab} setActiveTab={setActiveTab} />
            <TabButton id="eval" label="Eval Lab" icon={ClipboardCheck} badge="REVIEW" activeTab={activeTab} setActiveTab={setActiveTab} />
            <TabButton id="corpus" label="Corpus" icon={Newspaper} activeTab={activeTab} setActiveTab={setActiveTab} />
          </nav>
        </div>
      </div>
    </header>
  );
};

const TabButton: React.FC<{
  id: TabType;
  label: string;
  icon: React.ElementType;
  badge?: string;
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
}> = ({ id, label, icon: Icon, badge, activeTab, setActiveTab }) => {
  const isActive = activeTab === id;

  return (
    <button
      onClick={() => setActiveTab(id)}
      aria-current={isActive ? 'page' : undefined}
      className={`flex items-center gap-2.5 px-4 py-2.5 text-xs md:text-sm font-mono font-bold uppercase tracking-wide transition-colors whitespace-nowrap cursor-pointer border-b-2 -mb-px ${
        isActive
          ? 'border-hermes-red text-hermes-red bg-hermes-panel'
          : 'border-transparent text-hermes-bone/50 hover:text-hermes-bone'
      }`}
    >
      <Icon className="w-4 h-4" />
      <span>{label}</span>
      {badge && (
        <span className={`px-1.5 py-0.5 text-[10px] border ${
          isActive ? 'bg-hermes-red text-hermes-bone border-hermes-red' : 'bg-hermes-bone/5 text-hermes-bone/50 border-hermes-bone/20'
        }`}>
          {badge}
        </span>
      )}
    </button>
  );
};
