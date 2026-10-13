/**
 * Contract tests for the telemetry presentational components:
 * LiveAuditFeed (real ingest rows + error state), CacheTelemetry
 * (props pass-through) and VectorTelemetry (stats + semantic search).
 */
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { CacheTelemetry } from '../src/components/CacheTelemetry';
import { VectorTelemetry } from '../src/components/VectorTelemetry';
import { LiveAuditFeed } from '../src/components/LiveAuditFeed';
import { corpusApi } from '../src/api/corpus';

const recentItem = {
  id: 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
  title: 'Coordinated bot network amplified story X',
  url: 'https://example.com/botnet',
  domain: 'bots.example',
  source_name: 'RSS Bot',
  collected_at: '2026-01-05T10:00:00Z',
  published_at: null,
  word_count: 512,
};

vi.mock('../src/api/corpus', () => ({
  corpusApi: {
    recent: vi.fn(),
    stats: vi.fn(),
    search: vi.fn(),
    list: vi.fn(async () => ({ total: 0, limit: 25, offset: 0, items: [], domains: [] })),
    storyClusters: vi.fn(async () => ({ clusters: [] })),
  },
}));

const wrap = (ui: React.ReactElement) => {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
};

describe('<LiveAuditFeed/>', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(corpusApi.recent).mockResolvedValue({ items: [recentItem] });
  });

  it('renders real ingest rows', async () => {
    wrap(<LiveAuditFeed />);
    expect(await screen.findByText('Live Ingest Stream')).toBeTruthy();
    expect(await screen.findByText(/Coordinated bot network/i)).toBeTruthy();
  });

  it('shows the unavailable-state on API failure', async () => {
    vi.mocked(corpusApi.recent).mockRejectedValueOnce(new Error('down'));
    wrap(<LiveAuditFeed />);
    expect(await screen.findByText('Stream unavailable')).toBeTruthy();
  });
});

describe('<CacheTelemetry/>', () => {
  it('renders supplied cache metrics verbatim', () => {
    vi.mocked(corpusApi.recent).mockResolvedValue({ items: [] });
    wrap(
      <CacheTelemetry hitRatio="98.4%" latencyMs="4.2ms" revokedTokens="1,409" memoryMb="128.4 MB" />,
    );
    expect(screen.getByText('98.4%')).toBeTruthy();
    expect(screen.getByText('1,409')).toBeTruthy();
    expect(screen.getByText('128.4 MB')).toBeTruthy();
    expect(screen.getByText(/Redis Cache & Token Blocklist Telemetry/)).toBeTruthy();
  });
});

describe('<VectorTelemetry/> — semantic search surface', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(corpusApi.stats).mockResolvedValue({
      total: 6436,
      embedded: 6436,
      domains: 42,
      earliest: '2025-01-01',
      latest: '2026-01-01',
      nlp_counts: {},
    });
    vi.mocked(corpusApi.search).mockResolvedValue({
      query: 'election map',
      count: 0,
      items: [],
    });
  });

  it('displays corpus stats and runs a semantic search query', async () => {
    wrap(<VectorTelemetry />);
    expect(await screen.findByText('Semantic Corpus Search')).toBeTruthy();
    const statCells = await screen.findAllByText(/6[,.]?436/);
    expect(statCells.length).toBeGreaterThanOrEqual(2);

    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: 'election map' } });
    fireEvent.submit(input.closest('form')!);

    await waitFor(() => {
      expect(corpusApi.search).toHaveBeenCalledWith('election map', 8, expect.objectContaining({ signal: expect.any(AbortSignal) }));
    });
  });
});
