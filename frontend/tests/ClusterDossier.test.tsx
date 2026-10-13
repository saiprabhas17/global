import { describe, expect, it, vi } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithClient } from './helpers';
import { ClusterDossier } from '../src/components/ClusterDossier';
import type { TimelineCluster } from '../src/api/timeline';

const hubAt = '2026-08-15T08:00:00Z';
const cluster: TimelineCluster = {
  id: '11111111-1111-1111-1111-111111111111',
  title: 'Powerful 7.7-magnitude earthquake kills at least 38',
  domain: 'www.bbc.co.uk',
  url: 'https://example.com/hub',
  published_at: hubAt,
  newest_member_at: '2026-08-15T12:00:00Z',
  deg: 4,
  members: [
    {
      id: 'm1',
      title: 'Magnitude 7.7 earthquake strikes off coast',
      domain: 'www.npr.org',
      url: 'https://example.com/a',
      published_at: '2026-08-15T08:30:00Z', // 30m after hub
      score: 0.83,
    },
    {
      id: 'm2',
      title: 'Aftermath of deadly earthquake',
      domain: 'www.aljazeera.com',
      url: 'https://example.com/b',
      published_at: '2026-08-15T07:15:00Z', // 45m BEFORE hub → scoop
      score: 0.86,
    },
  ],
};

const renderDossier = (isFocused = false) =>
  renderWithClient(
    <ClusterDossier cluster={cluster} isFocused={isFocused} onClose={() => vi.fn()} />,
  );

describe('<ClusterDossier/> — rich but honest cluster inspector', () => {
  it('renders the hub head with outlet color dot, hub chip and coverage span', () => {
    renderDossier();
    expect(screen.getByText(/Powerful 7\.7-magnitude earthquake/)).toBeTruthy();
    expect(screen.getByText('hub')).toBeTruthy();
    expect(screen.getByText(/coverage span/i)).toBeTruthy();
    expect(screen.getByText(/first reported/i)).toBeTruthy();
  });

  it('computes publish-time lag vs the hub, including scoop badges', () => {
    renderDossier();
    expect(screen.getByText('+30m later')).toBeTruthy();
    expect(screen.getByText(/scoop 45m before/i)).toBeTruthy();
  });

  it('shows similarity meters and the match-range footer chip', () => {
    renderDossier();
    expect(screen.getByText('83')).toBeTruthy();
    expect(screen.getByText('86')).toBeTruthy();
    expect(screen.getByText(/match 83–86%/i)).toBeTruthy();
    expect(screen.getByText(/4 similarity links/i)).toBeTruthy();
  });
});
