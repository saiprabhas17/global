import { describe, expect, it, vi } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { EvalLab } from '../src/components/EvalLab';
import { accountFixture, jsonResponse, mockApi, renderWithClient, signInFixture } from './helpers';

const pairPayload = {
  done: false,
  pair: { pair_id: '11111111-1111-1111-1111-111111111111', a: { text: 'Claim A text here', domain: 'a.example' }, b: { text: 'Claim B text here', domain: 'b.example' } },
};
const progressPayload = {
  total_pairs: 8, labeled_votes: 2, labeled_pairs_distinct: 1,
  by_bucket: [{ bucket: 'b00', pairs: 8, labeled: 1 }], per_annotator: {}, inter_annotator: [], complete: false,
  origins: { human: { votes: 2, pairs: 1 }, automatic: { votes: 40, pairs: 8 }, legacy: { votes: 0, pairs: 0 }, test: { votes: 0, pairs: 0 } },
  warning: 'Complete means current consensus from authenticated humans, not publication validity.',
};

const rubricPayload = {
  version: 'claim-pairs-v2', labels: { SAME_STORY: 'Equivalent proposition', EVOLVED: 'Changed detail', DISTINCT: 'Unrelated propositions' },
  context_relations: { PARAPHRASE: 'Equivalent meaning', UPDATE_OR_CORRECTION: 'Explicit update or correction', INCOMPATIBLE_ASSERTION: 'Incompatible assertions', INSUFFICIENT_CONTEXT: 'More context needed', UNRELATED: 'No shared proposition' },
  mutation_types: ['NUMERIC_DRIFT', 'HEDGING_SHIFT', 'WORDING_DRIFT'], span_offsets: 'Unicode code points', guidance: ['Judge independently.'],
};

function evalResponses(endpoint: string): Response {
  if (endpoint === '/eval/rubric') return jsonResponse(rubricPayload);
  if (endpoint.startsWith('/eval/next')) return jsonResponse(pairPayload);
  if (endpoint === '/eval/progress') return jsonResponse(progressPayload);
  return jsonResponse({ status: 'recorded', pair_id: pairPayload.pair.pair_id, label: 'DISTINCT', origin: 'human' }, 201);
}

describe('Eval Lab authenticated labeling', () => {
  it('requires an account, never a locally stored annotator handle', () => {
    window.localStorage.setItem('gmee-eval-annotator', 'pretend-admin');
    const fetch = mockApi(evalResponses);
    renderWithClient(<EvalLab />);
    expect(screen.getByText(/Sign in to use Eval Lab/)).toBeTruthy();
    expect(screen.queryByLabelText(/^Annotator$/)).toBeNull();
    expect(fetch).not.toHaveBeenCalled();
  });

  it('requests a blinded pair using the authenticated identity and no annotator parameter', async () => {
    signInFixture();
    window.localStorage.setItem('gmee-eval-annotator', 'spoofed-user');
    const fetch = mockApi(evalResponses);
    renderWithClient(<EvalLab />);
    expect(await screen.findByText(/Claim A text here/)).toBeTruthy();
    expect(screen.getByText(/Claim B text here/)).toBeTruthy();
    expect(screen.getByText(/Authenticated annotator: Corpus Reviewer/)).toBeTruthy();
    const next = fetch.mock.calls.find(([url]) => String(url).includes('/eval/next'));
    expect(String(next?.[0])).toContain('strategy=coverage');
    expect(String(next?.[0])).toContain('split=unassigned');
    expect(String(next?.[0])).not.toContain('annotator');
    expect(new Headers(next?.[1]?.headers).get('Authorization')).toBe('Bearer test-access');
    expect(screen.queryByText('spoofed-user')).toBeNull();
    expect(Object.keys(pairPayload.pair).sort()).toEqual(['a', 'b', 'pair_id']);
  });

  it('records labels with optional mutation types and notes, then advances', async () => {
    signInFixture();
    let recorded = false;
    const fetch = mockApi((endpoint) => {
      if (endpoint === '/eval/label') { recorded = true; return evalResponses(endpoint); }
      if (endpoint.startsWith('/eval/next') && recorded) return jsonResponse({ done: true, pair: null });
      return evalResponses(endpoint);
    });
    renderWithClient(<EvalLab />);
    await screen.findByText(/Claim A text here/);
    fireEvent.change(screen.getByLabelText(/Mutation types/), { target: { value: 'NUMERIC_DRIFT, HEDGING_SHIFT, NUMERIC_DRIFT' } });
    fireEvent.change(screen.getByLabelText('Notes (optional)'), { target: { value: 'Number shifted; retain uncertainty.' } });
    fireEvent.click(screen.getByRole('button', { name: /Distinct/ }));
    expect(await screen.findByText(/No remaining pairs in the selected split/)).toBeTruthy();
    const vote = fetch.mock.calls.find(([url]) => String(url).endsWith('/eval/label'));
    expect(JSON.parse(String(vote?.[1]?.body))).toEqual({ pair_id: pairPayload.pair.pair_id, label: 'DISTINCT', mutation_types: ['NUMERIC_DRIFT', 'HEDGING_SHIFT'], notes: 'Number shifted; retain uncertainty.', rubric_version: 'claim-pairs-v2' });
    expect(new Headers(vote?.[1]?.headers).get('Authorization')).toBe('Bearer test-access');
  });

  it('selects uncertainty only on the training split', async () => {
    signInFixture();
    const fetch = mockApi(evalResponses);
    renderWithClient(<EvalLab />);
    await screen.findByText(/Claim A text here/);
    fireEvent.change(screen.getByLabelText('Sampling strategy'), { target: { value: 'uncertainty' } });
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => String(url).includes('strategy=uncertainty&split=train'))).toBe(true));
    expect((screen.getByLabelText('Dataset split') as HTMLSelectElement).value).toBe('train');
    expect(screen.getByLabelText('Dataset split').hasAttribute('disabled')).toBe(true);
    expect(screen.getByText(/not calibrated probability entropy/)).toBeTruthy();
  });

  it('ignores vote hotkeys while typing notes and prevents duplicate pending votes', async () => {
    signInFixture();
    const fetch = mockApi((endpoint) => endpoint === '/eval/label' ? new Promise<Response>(() => {}) : evalResponses(endpoint));
    const page = renderWithClient(<EvalLab />);
    await screen.findByText(/Claim A text here/);
    fireEvent.keyDown(screen.getByLabelText('Notes (optional)'), { key: '1' });
    expect(fetch.mock.calls.filter(([url]) => String(url).endsWith('/eval/label'))).toHaveLength(0);
    // Prefix-anchored: the EVOLVED card's hint also contains "Same story".
    fireEvent.click(screen.getByRole('button', { name: /^Same story/ }));
    await waitFor(() => expect(screen.getByRole('button', { name: /Distinct/ }).hasAttribute('disabled')).toBe(true));
    fireEvent.keyDown(window, { key: '3' });
    expect(fetch.mock.calls.filter(([url]) => String(url).endsWith('/eval/label'))).toHaveLength(1);
    page.unmount();
  });

  it('distinguishes pair/progress failure from an empty or completed dataset and supports retry', async () => {
    signInFixture();
    let failure = true;
    mockApi((endpoint) => failure ? jsonResponse({ detail: 'Eval store unavailable' }, 503) : evalResponses(endpoint));
    renderWithClient(<EvalLab />);
    expect(await screen.findByText('Eval pairs unavailable')).toBeTruthy();
    expect(screen.getByText('Eval progress unavailable')).toBeTruthy();
    expect(screen.queryByText(/No remaining pairs/)).toBeNull();
    expect(screen.queryByText(/0 \/ 0 labeled/)).toBeNull();
    failure = false;
    fireEvent.click(screen.getByRole('button', { name: /Retry eval pairs/ }));
    expect(await screen.findByText(/Claim A text here/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /Retry eval progress/ }));
    expect(await screen.findByText('1 / 8 labeled')).toBeTruthy();
  });

  it('shows origin counts and returned warnings, and tolerates null kappa', async () => {
    signInFixture();
    mockApi((endpoint) => endpoint === '/eval/progress' ? jsonResponse({ ...progressPayload, inter_annotator: [{ annotators: ['user:a', 'user:b'], pairs: 2, kappa: null, warning: 'Shared-pair agreement is descriptive only.' }] }) : evalResponses(endpoint));
    renderWithClient(<EvalLab />);
    expect(await screen.findByText('Label origins')).toBeTruthy();
    expect(screen.getByText('40 votes · 8 pairs')).toBeTruthy();
    expect(screen.getByText('Complete means current consensus from authenticated humans, not publication validity.')).toBeTruthy();
    expect(screen.getByText(/not independent human gold/)).toBeTruthy();
    expect(screen.getByText(/not estimable/)).toBeTruthy();
    expect(screen.getByText('Shared-pair agreement is descriptive only.')).toBeTruthy();
    expect(screen.queryByText(/NaN/)).toBeNull();
  });

  it('renders legitimate zero coverage without claiming every gold pair was labeled', async () => {
    signInFixture();
    mockApi((endpoint) => endpoint === '/eval/rubric' ? jsonResponse(rubricPayload) : endpoint.startsWith('/eval/next') ? jsonResponse({ done: true, pair: null }) : jsonResponse({ ...progressPayload, total_pairs: 0, labeled_votes: 0, labeled_pairs_distinct: 0, by_bucket: [], origins: {} }));
    renderWithClient(<EvalLab />);
    expect(await screen.findByText('0 / 0 labeled')).toBeTruthy();
    expect(screen.getByText(/No evaluation pairs available/)).toBeTruthy();
    expect(screen.queryByText(/every pair in the gold set/)).toBeNull();
  });

  it('preserves a pair on label failure and offers a real retry', async () => {
    signInFixture();
    let failure = true;
    mockApi((endpoint) => endpoint === '/eval/label' && failure ? jsonResponse({ detail: 'Vote transaction failed' }, 503) : evalResponses(endpoint));
    renderWithClient(<EvalLab />);
    await screen.findByText(/Claim A text here/);
    fireEvent.click(screen.getByRole('button', { name: /Evolved/ }));
    expect(await screen.findByText('Vote transaction failed')).toBeTruthy();
    expect(screen.getByText(/Claim A text here/)).toBeTruthy();
    failure = false;
    fireEvent.click(screen.getByRole('button', { name: /Retry vote failed/ }));
    expect(await screen.findByText(/Recorded EVOLVED/)).toBeTruthy();
  });

  it('does not expose admin export controls to ordinary accounts', async () => {
    signInFixture();
    mockApi(evalResponses);
    renderWithClient(<EvalLab />);
    await screen.findByText(/Claim A text here/);
    expect(screen.queryByRole('button', { name: /Download actual export/ })).toBeNull();
    expect(screen.queryByLabelText('Independent review of disagreements')).toBeNull();
  });

  it('anchors selected spans to Unicode codepoints and includes their type and context', async () => {
    signInFixture();
    const unicodePair = { ...pairPayload, pair: { ...pairPayload.pair, a: { text: '🌊 River rose 12 metres.', domain: 'a.example' } } };
    const fetch = mockApi((endpoint) => endpoint.startsWith('/eval/next') ? jsonResponse(unicodePair) : evalResponses(endpoint));
    renderWithClient(<EvalLab />);
    const claim = await screen.findByLabelText('Claim A text; select a changed span') as HTMLTextAreaElement;
    await screen.findByLabelText('Changed span type');
    fireEvent.change(screen.getByLabelText('Changed span type'), { target: { value: 'NUMERIC_DRIFT' } });
    claim.focus();
    claim.setSelectionRange(14, 16);
    fireEvent.select(claim);
    expect(screen.getByText('Selected claim A: “12”')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Add selected span' }));
    fireEvent.change(screen.getByLabelText('Context relation (optional)'), { target: { value: 'INSUFFICIENT_CONTEXT' } });
    fireEvent.click(screen.getByRole('button', { name: /Evolved/ }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/eval/label'))).toBe(true));
    const vote = fetch.mock.calls.find(([url]) => String(url).endsWith('/eval/label'));
    expect(JSON.parse(String(vote?.[1]?.body))).toMatchObject({
      rubric_version: 'claim-pairs-v2', context_relation: 'INSUFFICIENT_CONTEXT', mutation_types: ['NUMERIC_DRIFT'],
      changed_spans: [{ side: 'a', start: 13, end: 15, text: '12', mutation_type: 'NUMERIC_DRIFT' }],
    });
  });

  it('distinguishes an explicitly empty span annotation from an unannotated field', async () => {
    signInFixture();
    const fetch = mockApi(evalResponses);
    renderWithClient(<EvalLab />);
    await screen.findByLabelText(/Changed spans reviewed/);
    fireEvent.click(screen.getByLabelText(/Changed spans reviewed/));
    fireEvent.click(screen.getByRole('button', { name: /^Same story/ }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/eval/label'))).toBe(true));
    const vote = fetch.mock.calls.find(([url]) => String(url).endsWith('/eval/label'));
    expect(JSON.parse(String(vote?.[1]?.body))).toMatchObject({ changed_spans: [] });
  });

  it('loads the separate blinded disagreement queue for an administrator', async () => {
    signInFixture({ role: 'admin' });
    const fetch = mockApi((endpoint) => endpoint.startsWith('/eval/review/next') ? jsonResponse(pairPayload) : evalResponses(endpoint));
    renderWithClient(<EvalLab />);
    await screen.findByText(/Claim A text here/);
    fireEvent.click(screen.getByLabelText('Independent review of disagreements'));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/eval/review/next?split=unassigned'))).toBe(true));
    expect(screen.getByLabelText('Sampling strategy').hasAttribute('disabled')).toBe(true);
    expect(screen.getByText(/preserves earlier dissent/)).toBeTruthy();
  });

  it('requires the current rubric before accepting a vote and offers retry', async () => {
    signInFixture();
    let unavailable = true;
    mockApi((endpoint) => endpoint === '/eval/rubric' && unavailable ? jsonResponse({ detail: 'Rubric unavailable' }, 503) : evalResponses(endpoint));
    renderWithClient(<EvalLab />);
    await screen.findByText('Annotation rubric unavailable');
    expect(screen.getByRole('button', { name: /^Same story/ }).hasAttribute('disabled')).toBe(true);
    unavailable = false;
    fireEvent.click(screen.getByRole('button', { name: /Retry annotation rubric/ }));
    await screen.findByText('Annotation rubric: claim-pairs-v2');
    expect(screen.getByRole('button', { name: /^Same story/ }).hasAttribute('disabled')).toBe(false);
  });

  it('downloads the actual admin export response bytes with the required dataset_version', async () => {
    signInFixture({ role: 'admin' });
    const bytes = '{"dataset_version":"review-v1","records":[{"pair_id":"returned-pair","origin":"human"}]}';
    const fetch = mockApi((endpoint) => endpoint.startsWith('/eval/export')
      ? new Response(bytes, { headers: { 'Content-Type': 'application/json', 'Content-Disposition': 'attachment; filename="review-v1.json"' } })
      : evalResponses(endpoint));
    if (!URL.createObjectURL) Object.defineProperty(URL, 'createObjectURL', { configurable: true, writable: true, value: () => '' });
    if (!URL.revokeObjectURL) Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, writable: true, value: () => {} });
    const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:actual-export');
    const revoke = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const downloads: string[] = [];
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) { downloads.push(this.download); });
    renderWithClient(<EvalLab />);
    expect(screen.getByRole('button', { name: /Download actual export/ }).hasAttribute('disabled')).toBe(true);
    fireEvent.change(screen.getByLabelText('Dataset version'), { target: { value: 'review-v1' } });
    fireEvent.click(screen.getByRole('button', { name: /Download actual export/ }));
    await screen.findByText('Export response downloaded.');
    const blob = createUrl.mock.calls[0][0] as Blob;
    // Native fetch returns a Node Blob; jsdom FileReader only accepts its own Blob.
    const text = typeof blob.text === 'function' ? await blob.text() : await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(reader.error);
      reader.readAsText(blob);
    });
    expect(text).toBe(bytes);
    expect(downloads).toEqual(['review-v1.json']);
    expect(revoke).toHaveBeenCalledWith('blob:actual-export');
    const request = fetch.mock.calls.find(([url]) => String(url).includes('/eval/export'));
    expect(String(request?.[0])).toContain('dataset_version=review-v1&publication=false');
    expect(new Headers(request?.[1]?.headers).get('Authorization')).toBe('Bearer test-access');
    expect(accountFixture.role).toBe('user');
  });
});
