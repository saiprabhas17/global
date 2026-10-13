import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Network, ExternalLink, Layers, Zap, Search, RotateCcw, Orbit } from 'lucide-react';
import { graphApi, FullGraph, GraphNode } from '../api/graph';
import { setSelectedArticle } from '../lib/useSelectedArticle';
import type { TabType } from './Header';
import { usePollingPolicy } from '../lib/polling';
import { useAnimationActivity } from '../lib/useAnimationActivity';
import { safeHttpUrl } from '../lib/urls';
import { QueryError } from './QueryError';

/**
 * FullCorpusGraph — THE ENTIRE corpus as one interactive force-directed graph.
 *
 * Canvas-rendered (SVG would die at 6.4k nodes). Custom lightweight force
 * sim: repulsion via spatial grid, spring edges, centering. 120Hz-capable
 * stepping with calm damping per GMEE motion rules.
 *
 * Interactions: drag nodes · pan canvas · ctrl/⌘+wheel zoom · click to inspect ·
 * outlet filter · search highlight.
 */

const OUTLET_COLORS: Record<string, string> = {
  'www.thehindu.com': '#E11D2E',
  'www.aljazeera.com': '#FF8A5C',
  'www.theguardian.com': '#7FB4FF',
  'www.ndtv.com': '#9BE38A',
  'timesofindia.indiatimes.com': '#E3C567',
  'www.bbc.co.uk': '#C99CFF',
  'www.wired.com': '#6FE0D2',
  'www.nytimes.com': '#F2F2F2',
  'techcrunch.com': '#8FD18F',
  'www.npr.org': '#FFB3C7',
  'news.sky.com': '#79A8FF',
  'theverge.com': '#FF7AC8',
  'www.cnn.com': '#FF6B6B',
};
const colorFor = (d?: string | null): string => (d && OUTLET_COLORS[d]) || '#B08A85';

interface SimNode extends GraphNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
  fixed?: boolean;
}

function buildSim(graph: FullGraph): { nodes: SimNode[]; index: Map<string, SimNode> } {
  const R = 520;
  const nodes: SimNode[] = graph.nodes.map((n, i) => {
    // golden-angle spiral init — deterministic, well-spread
    const a = i * 2.39996;
    const r = R * Math.sqrt((i + 1) / graph.nodes.length);
    return { ...n, x: Math.cos(a) * r + Math.sin(i * 17) * 15, y: Math.sin(a) * r * 0.72 + Math.cos(i * 31) * 15, vx: 0, vy: 0 };
  });
  const index = new Map(nodes.map((n) => [n.id, n]));
  return { nodes, index };
}

const FullCorpusGraph: React.FC<{ setActiveTab?: (tab: TabType) => void }> = ({ setActiveTab }) => {
  const [mode, setMode] = useState<'articles' | 'claims'>('articles');
  const [selected, setSelected] = useState<SimNode | null>(null);
  const [outletFilter, setOutletFilter] = useState<string>('');
  const [search, setSearch] = useState('');
  const [query, setQuery] = useState('');

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const { animate, visible } = useAnimationActivity(wrapRef);
  const [renderVersion, setRenderVersion] = useState(0);
  const polling = usePollingPolicy(60000);
  const simRef = useRef<{ nodes: SimNode[]; edges: { a: SimNode; b: SimNode; s: number }[] }>({ nodes: [], edges: [] });
  const viewRef = useRef({ ox: 0, oy: 0, scale: 1 });
  const alphaRef = useRef(1);
  const dragRef = useRef<{ node: SimNode | null; panning: boolean; lx: number; ly: number }>({ node: null, panning: false, lx: 0, ly: 0 });

  const g = useQuery({
    queryKey: ['full-graph', mode],
    queryFn: ({ signal }) => (mode === 'articles' ? graphApi.full({ signal }) : graphApi.claims({ signal })),
    ...polling,
    staleTime: 30000,
  });

  const outlets = useMemo(() => {
    const m = new Map<string, number>();
    (g.data?.nodes ?? []).forEach((n) => {
      if (n.domain) m.set(n.domain, (m.get(n.domain) ?? 0) + 1);
    });
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, [g.data]);

  // rebuild sim when data or filters change
  useEffect(() => {
    if (!g.data) return;
    let nodesSrc = g.data.nodes;
    let edgesSrc = g.data.edges;

    if (outletFilter) {
      const keep = new Set(nodesSrc.filter((n) => n.domain === outletFilter).map((n) => n.id));
      nodesSrc = nodesSrc.filter((n) => keep.has(n.id));
      edgesSrc = edgesSrc.filter((e) => keep.has(e.src) && keep.has(e.dst));
    }
    if (query.trim()) {
      const q = query.toLowerCase();
      const matches = new Set(nodesSrc.filter((n) => n.title.toLowerCase().includes(q)).map((n) => n.id));
      const keep = new Set(matches);
      // Only expand one hop from the original matches, independent of edge ordering.
      edgesSrc.forEach((e) => {
        if (matches.has(e.src)) keep.add(e.dst);
        if (matches.has(e.dst)) keep.add(e.src);
      });
      nodesSrc = nodesSrc.filter((n) => keep.has(n.id));
      edgesSrc = edgesSrc.filter((e) => keep.has(e.src) && keep.has(e.dst));
    }

    const { nodes, index } = buildSim({ nodes: nodesSrc, edges: edgesSrc, counts: { nodes: nodesSrc.length, edges: edgesSrc.length } });
    const edges = edgesSrc
      .map((e) => ({ a: index.get(e.src), b: index.get(e.dst), s: e.score }))
      .filter((e): e is { a: SimNode; b: SimNode; s: number } => !!e.a && !!e.b);

    simRef.current = { nodes, edges };
    alphaRef.current = 1;
    viewRef.current = { ox: 0, oy: 0, scale: 1 };
    setSelected(null);
  }, [g.data, outletFilter, query]);

  // physics step + render at display refresh
  useEffect(() => {
    let raf = 0;

    const step = () => {
      if (!visible || !canvasRef.current) return;
      const { nodes, edges } = simRef.current;
      const alpha = alphaRef.current;
      if (animate && alpha > 0.005) {
        // repulsion (grid-accelerated approximation)
        const CELL = 90;
        const grid = new Map<string, SimNode[]>();
        for (const n of nodes) {
          const k = `${Math.round(n.x / CELL)},${Math.round(n.y / CELL)}`;
          const arr = grid.get(k);
          if (arr) arr.push(n);
          else grid.set(k, [n]);
        }
        for (const n of nodes) {
          const gx = Math.round(n.x / CELL);
          const gy = Math.round(n.y / CELL);
          for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
              const cell = grid.get(`${gx + dx},${gy + dy}`);
              if (!cell) continue;
              for (const o of cell) {
                if (o === n) continue;
                const ddx = n.x - o.x;
                const ddy = n.y - o.y;
                const d2 = ddx * ddx + ddy * ddy + 40;
                if (d2 < 36000) {
                  const f = (1400 * alpha) / d2;
                  const d = Math.sqrt(d2);
                  n.vx += (ddx / d) * f;
                  n.vy += (ddy / d) * f;
                }
              }
            }
          }
        }
        // springs
        for (const e of edges) {
          const dx = e.b.x - e.a.x;
          const dy = e.b.y - e.a.y;
          const d = Math.sqrt(dx * dx + dy * dy) || 1;
          const target = 95;
          const f = ((d - target) / d) * 0.06 * alpha;
          e.a.vx += dx * f;
          e.a.vy += dy * f;
          e.b.vx -= dx * f;
          e.b.vy -= dy * f;
        }
        // integrate + gentle centering
        for (const n of nodes) {
          if (!n.fixed) {
            n.vx *= 0.82;
            n.vy *= 0.82;
            n.x += Math.max(-14, Math.min(14, n.vx));
            n.y += Math.max(-14, Math.min(14, n.vy));
            n.x *= 0.9995;
            n.y *= 0.9995;
          }
        }
        alphaRef.current = alpha * 0.994;
      }
      if (render() && animate && nodes.length > 0) raf = requestAnimationFrame(step);
    };

    const render = () => {
      const cv = canvasRef.current;
      const wrap = wrapRef.current;
      if (!cv || !wrap) return false;
      const dpr = window.devicePixelRatio || 1;
      const W = wrap.clientWidth;
      const H = wrap.clientHeight;
      if (cv.width !== W * dpr || cv.height !== H * dpr) {
        cv.width = W * dpr;
        cv.height = H * dpr;
        cv.style.width = `${W}px`;
        cv.style.height = `${H}px`;
      }
      const ctx = cv.getContext('2d');
      if (!ctx) return false;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      const { ox, oy, scale } = viewRef.current;
      ctx.save();
      ctx.translate(W / 2 + ox, H / 2 + oy);
      ctx.scale(scale, scale);

      const { nodes, edges } = simRef.current;
      const q = query.trim().toLowerCase();

      // edges
      ctx.lineWidth = 1;
      for (const e of edges) {
        const hot = selected && (e.a === selected || e.b === selected);
        ctx.strokeStyle = hot ? '#E11D2E' : 'rgba(255,247,242,0.16)';
        ctx.globalAlpha = hot ? 0.9 : selected ? 0.25 : 1;
        ctx.beginPath();
        ctx.moveTo(e.a.x, e.a.y);
        ctx.lineTo(e.b.x, e.b.y);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      // nodes
      for (const n of nodes) {
        const isSel = selected === n;
        const match = q ? n.title.toLowerCase().includes(q) : false;
        const r = mode === 'articles'
          ? (n.deg != null && n.deg > 0 ? 3.4 + Math.min(7, n.deg * 1.15) : 2.4)
          : 3.2;
        if (match) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, r + 3.5, 0, Math.PI * 2);
          ctx.strokeStyle = '#FFF7F2';
          ctx.lineWidth = 1.6;
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fillStyle = colorFor(n.domain);
        ctx.globalAlpha = selected && !isSel ? 0.35 : match ? 1 : 0.88;
        ctx.fill();
        if (isSel) {
          ctx.globalAlpha = 1;
          ctx.strokeStyle = '#FFF7F2';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
      ctx.restore();

      // scale badge
      ctx.fillStyle = 'rgba(255,247,242,0.45)';
      ctx.font = '10px monospace';
      ctx.fillText(`zoom ${(scale * 100).toFixed(0)}% · ${simRef.current.nodes.length.toLocaleString()} nodes`, 12, H - 12);
      return true;
    };

    if (visible) raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [selected, query, mode, animate, visible, renderVersion, g.data, outletFilter]);

  // ------- interactions -------
  const pickNode = (mx: number, my: number): SimNode | null => {
    const { ox, oy, scale } = viewRef.current;
    const wrap = wrapRef.current;
    if (!wrap) return null;
    const wx = (mx - wrap.clientWidth / 2 - ox) / scale;
    const wy = (my - wrap.clientHeight / 2 - oy) / scale;
    let best: SimNode | null = null;
    let bd = 12 * 12;
    for (const n of simRef.current.nodes) {
      const dx = n.x - wx;
      const dy = n.y - wy;
      const d2 = dx * dx + dy * dy;
      if (d2 < bd) {
        bd = d2;
        best = n;
      }
    }
    return best;
  };

  const onMouseDown = (e: React.MouseEvent) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const n = pickNode(mx, my);
    if (n) {
      n.fixed = true;
      dragRef.current = { node: n, panning: false, lx: mx, ly: my };
      alphaRef.current = Math.max(alphaRef.current, 0.5);
    } else {
      dragRef.current = { node: null, panning: true, lx: mx, ly: my };
    }
  };

  const onMouseMove = (e: React.MouseEvent) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const dr = dragRef.current;
    if (dr.node) {
      const { ox, oy, scale } = viewRef.current;
      dr.node.x = (mx - wrapRef.current!.clientWidth / 2 - ox) / scale;
      dr.node.y = (my - wrapRef.current!.clientHeight / 2 - oy) / scale;
      dr.node.vx = dr.node.vy = 0;
    } else if (dr.panning) {
      viewRef.current.ox += mx - dr.lx;
      viewRef.current.oy += my - dr.ly;
      dr.lx = mx;
      dr.ly = my;
    }
    if (!animate && (dr.node || dr.panning)) setRenderVersion((value) => value + 1);
  };

  const onMouseUp = () => {
    if (dragRef.current.node) dragRef.current.node.fixed = false;
    dragRef.current = { node: null, panning: false, lx: 0, ly: 0 };
  };

  const onClick = (e: React.MouseEvent) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect) return;
    const n = pickNode(e.clientX - rect.left, e.clientY - rect.top);
    setSelected(n);
  };

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const handleNativeWheel = (e: WheelEvent) => {
      // Plain wheel belongs to the page: the corpus canvas sits in a long
      // scrolling tab, so it must not fight vertical scroll. Zoom is
      // an explicit gesture: ctrl/⌘ + wheel, exactly like map embeds.
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      const v = viewRef.current;
      v.scale = Math.min(4, Math.max(0.25, v.scale * (e.deltaY < 0 ? 1.12 : 0.89)));
      if (!animate) setRenderVersion((value) => value + 1);
    };
    el.addEventListener('wheel', handleNativeWheel, { passive: false });
    return () => el.removeEventListener('wheel', handleNativeWheel);
  }, [animate]);

  const resetView = () => {
    viewRef.current = { ox: 0, oy: 0, scale: 1 };
    alphaRef.current = 1;
    setRenderVersion((value) => value + 1);
  };
  const timelineArticleId = selected?.article_id || (mode === 'articles' ? selected?.id : undefined);

  return (
    <div className="card-brutal-dark p-6 flex flex-col gap-5">
      {/* header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-hermes-bone/12">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Network className="w-5 h-5" />
            <h2 className="text-2xl font-display">Whole-Corpus Graph</h2>
            {!g.isLoading && g.data && (
              <span className="chip-brutal border-hermes-red text-hermes-red-bright">
                {mode === 'articles'
                  ? `${g.data.counts.nodes.toLocaleString()} articles · ${g.data.counts.edges.toLocaleString()} links`
                  : `${g.data.counts.nodes.toLocaleString()} claims · ${g.data.counts.edges.toLocaleString()} near-dupes`}
              </span>
            )}
          </div>
          <p className="text-xs font-mono text-hermes-bone/55 uppercase tracking-wider">
            every node is real · drag · pan · ctrl+scroll zoom · click inspect
          </p>
        </div>

        {/* controls row — always visible, wraps predictably */}
        <div className="flex flex-wrap items-center gap-2 pb-4 border-b border-hermes-bone/12 -mt-1">
          {/* mode switch */}
          <div className="flex border border-hermes-bone/25 shrink-0">
            <button
              onClick={() => { setMode('articles'); setSelected(null); }}
                            aria-pressed={mode === 'articles'}
              className={`px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider ${mode === 'articles' ? 'bg-hermes-red text-white' : 'text-hermes-bone/60 hover:text-hermes-bone'}`}
            >
              <Layers className="w-3 h-3 inline mr-1" /> Articles
            </button>
            <button
              onClick={() => { setMode('claims'); setSelected(null); }}
                            aria-pressed={mode === 'claims'}
              className={`px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider ${mode === 'claims' ? 'bg-hermes-red text-white' : 'text-hermes-bone/60 hover:text-hermes-bone'}`}
            >
              <Zap className="w-3 h-3 inline mr-1" /> Claims
            </button>
          </div>

          {/* search */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              setQuery(search.trim());
            }}
            className="flex items-center border border-hermes-bone/25 bg-hermes-panel-deep shrink-0"
          >
            <input
              aria-label="Graph title search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder='try "earthquake", "election"…'
              className="bg-transparent text-hermes-bone font-mono text-xs px-3 py-1.5 outline-none w-52 placeholder:text-hermes-bone/35"
            />
            <button type="submit" aria-label="Search graph titles" className="btn-brutal !py-1 !px-2 m-0.5">
              <Search className="w-3.5 h-3.5" />
            </button>
          </form>

          <select
            aria-label="Graph outlet filter"
            value={outletFilter}
            onChange={(e) => setOutletFilter(e.target.value)}
            className="bg-hermes-panel-deep border border-hermes-bone/25 text-hermes-bone font-mono text-xs px-2 py-1.5 outline-none cursor-pointer shrink-0 max-w-[190px]"
          >
            <option value="">all outlets</option>
            {outlets.map(([name, count]) => (
              <option key={name} value={name}>
                {name.replace('www.', '')} ({count})
              </option>
            ))}
          </select>

          <button onClick={resetView} aria-label="Reset graph view" className="btn-ghost-brutal !py-1.5 !px-2.5 !text-hermes-bone !border-hermes-bone/30">
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {g.data && <div className="flex flex-wrap items-center gap-3">
        <label htmlFor="graph-node-select" className="text-xs font-mono">Inspect a node (keyboard alternative)</label>
        <select id="graph-node-select" value={selected?.id ?? ''} onChange={(event) => setSelected(simRef.current.nodes.find((node) => node.id === event.target.value) ?? null)} className="field-brutal max-w-full">
          <option value="">Choose a real graph node</option>
          {g.data.nodes.filter((node) => (!outletFilter || node.domain === outletFilter) && (!query || node.title.toLowerCase().includes(query.toLowerCase()))).map((node) => <option key={node.id} value={node.id}>{node.title}</option>)}
        </select>
      </div>}
      {/* canvas */}
      <div
        ref={wrapRef}
        className="relative bg-hermes-panel-deep border border-hermes-bone/20 h-[600px] overflow-hidden cursor-grab active:cursor-grabbing"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onClick={onClick}
      >
        {g.isLoading ? (
          <div role="status" aria-label="Loading corpus graph" className="shimmer h-full w-full" />
        ) : g.isError ? (
          <div className="p-6"><QueryError title="Corpus graph unavailable" error={g.error} onRetry={() => g.refetch()} retrying={g.isFetching} /></div>
        ) : (
          <>
            <canvas ref={canvasRef} className="block" aria-hidden="true" />
            {g.data?.nodes.length === 0 && <p className="absolute inset-0 flex items-center justify-center text-xs font-mono">No graph nodes returned.</p>}
            {/* legend */}
            <div className="absolute top-3 left-3 bg-hermes-panel/90 border border-hermes-bone/20 px-3 py-2 space-y-1 pointer-events-none">
              {outlets.slice(0, 13).map(([name]) => (
                <div key={name} className="flex items-center gap-2 font-mono text-[9px] uppercase tracking-wider text-hermes-bone/70">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: colorFor(name) }} />
                  {name.replace('www.', '')}
                </div>
              ))}
            </div>
            {query && (
              <div className="absolute top-3 right-3 chip-brutal border-hermes-red-bright text-hermes-red-bright bg-hermes-panel/90 pointer-events-none">
                highlighting “{query}”
              </div>
            )}
          </>
        )}
      </div>

      {/* inspector */}
      {selected && (
        <div className="border border-hermes-bone/20 bg-hermes-panel-deep p-5 flex items-start justify-between gap-6">
          <div className="min-w-0">
            <div className="font-mono text-[10px] uppercase tracking-widest text-hermes-bone/45 mb-1">
              Node Inspector
              {selected.deg != null && ` · ${selected.deg} similarity links`}
            </div>
            <h3 className="font-display text-lg leading-snug">{selected.title}</h3>
            <span className="chip-brutal mt-2 w-fit" style={{ borderColor: colorFor(selected.domain), color: colorFor(selected.domain) }}>
              {selected.domain?.replace('www.', '')}
            </span>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <button
              disabled={!timelineArticleId || !setActiveTab}
              title={timelineArticleId ? 'Open this article’s story timeline' : 'This claim has no article_id; timeline drill-down is unsupported'}
              onClick={() => {
                if (!timelineArticleId) return;
                setSelectedArticle({
                  id: timelineArticleId,
                  title: selected.title,
                  domain: selected.domain,
                });
                setActiveTab?.('timeline');
              }}
              className="btn-brutal shrink-0"
            >
              View in Timeline <Orbit className="w-3.5 h-3.5" />
            </button>
            {safeHttpUrl(selected.url) && (
              <a href={safeHttpUrl(selected.url)} target="_blank" rel="noreferrer" className="btn-ghost-brutal shrink-0 !text-hermes-bone !border-hermes-bone/30">
                Read original <ExternalLink className="w-3.5 h-3.5" />
              </a>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default FullCorpusGraph;
