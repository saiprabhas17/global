import { fireEvent, screen, waitFor } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { OperationsPanel } from '../src/components/OperationsPanel';
import { jsonResponse, mockApi, renderWithClient, signInFixture } from './helpers';

describe('Pipeline operations', () => {
  it('shows controls only to administrators', () => {
    signInFixture();
    renderWithClient(<OperationsPanel />);
    expect(screen.queryByRole('button', { name: 'Pipeline operations' })).toBeNull();
  });

  it('requires preview before recovery and sends the selected stage', async () => {
    signInFixture({ role: 'admin' });
    const calls: unknown[] = [];
    mockApi((path, init) => {
      if (path === '/operations/jobs') return jsonResponse({ items: [] });
      if (path === '/operations/recover') {
        const body = JSON.parse(String(init?.body)) as { stage: string; dry_run: boolean };
        calls.push(body);
        return jsonResponse({ ...body, requeued: 2, failed: 0 });
      }
      return jsonResponse({}, 404);
    });
    renderWithClient(<OperationsPanel />);
    fireEvent.click(screen.getByRole('button', { name: 'Pipeline operations' }));
    const recover = screen.getByRole('button', { name: 'Recover expired work' }) as HTMLButtonElement;
    expect(recover.disabled).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: 'Preview expired work' }));
    await waitFor(() => expect(recover.disabled).toBe(false));
    expect(calls).toEqual([{ stage: 'nlp', dry_run: true }]);
    fireEvent.click(recover);
    await waitFor(() => expect(calls).toHaveLength(2));
    expect(calls[1]).toEqual({ stage: 'nlp', dry_run: false });
  });
});
