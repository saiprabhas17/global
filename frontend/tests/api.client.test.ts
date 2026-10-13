import { describe, expect, it, vi } from 'vitest';
import { apiClient, ApiError } from '../src/api/client';
import { fetchHealth } from '../src/api/health';
import { getAccessToken, getRefreshToken, setSessionTokens } from '../src/lib/session';
import { jsonResponse, mockApi, readyFixture, signInFixture } from './helpers';

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

describe('typed API requests and readiness', () => {
  it.each(['/graph/full', '/graph/claims', '/graph/timeline?article_id=actual-id', '/corpus/stats', '/corpus/articles?q=river'])('requests live %s data without a fixture fallback', async (endpoint) => {
    const data = { source: 'actual-api', endpoint };
    const fetch = mockApi(() => jsonResponse(data));
    expect(await apiClient.get(endpoint)).toEqual(data);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][0]).toBe(`/api/v1${endpoint}`);
  });

  it('does not replace a graph failure with fabricated nodes', async () => {
    const fetch = mockApi(() => jsonResponse({ detail: 'Graph projection unavailable' }, 503));
    await expect(apiClient.get('/graph/full')).rejects.toMatchObject({ status: 503, message: 'Graph projection unavailable' });
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('applies authentication and cancellation before mutation requests', async () => {
    const fetch = mockApi(() => jsonResponse({}));
    await expect(apiClient.post('/graph/mutation/compare', {}, { authenticated: true })).rejects.toMatchObject({ status: 401 });
    signInFixture();
    const controller = new AbortController();
    controller.abort();
    await expect(apiClient.post('/graph/mutation/compare', {}, { authenticated: true, signal: controller.signal })).rejects.toMatchObject({ kind: 'aborted' });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('uses the real ready aggregate value', async () => {
    mockApi(() => jsonResponse(readyFixture));
    expect(await fetchHealth()).toEqual(readyFixture);
  });

  it('preserves per-dependency readiness JSON from HTTP 503', async () => {
    const partial = { status: 'not_ready', postgres: 'ok', neo4j: 'down: unavailable', redis: 'ok' };
    mockApi(() => jsonResponse(partial, 503));
    expect(await fetchHealth()).toEqual(partial);
    await expect(apiClient.get('/health/ready')).rejects.toMatchObject({ status: 503, body: partial, kind: 'http' });
  });

  it('does not mistake a non-readiness 503 or old ok aggregate for readiness', async () => {
    const fetch = mockApi(() => jsonResponse({ detail: 'Maintenance' }, 503));
    await expect(fetchHealth()).rejects.toMatchObject({ status: 503, body: { detail: 'Maintenance' } });
    fetch.mockImplementation(async () => jsonResponse({ ...readyFixture, status: 'ok' }));
    await expect(fetchHealth()).rejects.toMatchObject({ status: 200, message: expect.stringContaining('Invalid readiness') });
  });

  it('retains status/body and validation details in ApiError', async () => {
    const body = { detail: [{ loc: ['body', 'claim_text'], msg: 'Too short' }], trace: 'validation' };
    mockApi(() => jsonResponse(body, 422));
    const error = await apiClient.post('/verdicts/check', {}).catch((failure: unknown) => failure);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 422, body, message: 'Too short' });
  });

  it('bounds a hanging request and aborts its fetch signal', async () => {
    vi.useFakeTimers();
    const fetch = mockApi(() => new Promise<Response>(() => {}));
    const assertion = expect(apiClient.get('/slow', { timeoutMs: 50 })).rejects.toMatchObject({ status: 0, kind: 'timeout' });
    await vi.advanceTimersByTimeAsync(50);
    await assertion;
    expect(fetch.mock.calls[0][1]?.signal?.aborted).toBe(true);
  });

  it('also bounds a hanging response body', async () => {
    vi.useFakeTimers();
    mockApi(() => {
      const response = jsonResponse({});
      vi.spyOn(response, 'text').mockImplementation(() => new Promise<string>(() => {}));
      return response;
    });
    const assertion = expect(apiClient.get('/slow-body', { timeoutMs: 50 })).rejects.toMatchObject({ kind: 'timeout' });
    await vi.advanceTimersByTimeAsync(50);
    await assertion;
  });

  it('distinguishes caller cancellation from network errors and clears timers', async () => {
    const controller = new AbortController();
    mockApi(() => new Promise<Response>(() => {}));
    const request = apiClient.get('/cancel', { signal: controller.signal });
    controller.abort();
    await expect(request).rejects.toMatchObject({ status: 0, kind: 'aborted' });
    mockApi(async () => { throw new TypeError('network failed'); });
    await expect(apiClient.get('/network')).rejects.toMatchObject({ status: 0, kind: 'network' });
  });

  it('does not send an already aborted request', async () => {
    const fetch = mockApi(() => jsonResponse({}));
    const controller = new AbortController();
    controller.abort();
    await expect(apiClient.get('/cancel', { signal: controller.signal })).rejects.toMatchObject({ kind: 'aborted' });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('requires a memory bearer token and never sends cookie credentials', async () => {
    const fetch = mockApi(() => jsonResponse({ status: 'accepted' }));
    await expect(apiClient.post('/private', {}, { authenticated: true })).rejects.toMatchObject({ status: 401 });
    expect(fetch).not.toHaveBeenCalled();
    signInFixture();
    await apiClient.post('/private', { input: 'real request' }, { authenticated: true });
    const init = fetch.mock.calls[0][1];
    expect(new Headers(init?.headers).get('Authorization')).toBe('Bearer test-access');
    expect(init?.credentials).toBe('omit');
    expect(window.localStorage.length).toBe(0);
    expect(window.sessionStorage.length).toBe(0);
  });

  it('handles successful empty 204 logout responses', async () => {
    mockApi(() => new Response(null, { status: 204 }));
    expect(await apiClient.post('/auth/logout', {})).toBeUndefined();
  });

  it('shares one refresh rotation across concurrent unauthorized requests', async () => {
    signInFixture();
    const refresh = deferred<Response>();
    const fetch = mockApi((endpoint, init) => {
      if (endpoint === '/auth/refresh') return refresh.promise;
      return new Headers(init?.headers).get('Authorization') === 'Bearer rotated-access'
        ? jsonResponse({ value: endpoint }) : jsonResponse({ detail: 'Expired' }, 401);
    });
    const a = apiClient.get('/private-a', { authenticated: true });
    const b = apiClient.get('/private-b', { authenticated: true });
    await vi.waitFor(() => expect(fetch.mock.calls.filter(([url]) => String(url).endsWith('/auth/refresh'))).toHaveLength(1));
    refresh.resolve(jsonResponse({ access_token: 'rotated-access', refresh_token: 'rotated-refresh' }));
    expect(await Promise.all([a, b])).toEqual([{ value: '/private-a' }, { value: '/private-b' }]);
    expect(getAccessToken()).toBe('rotated-access');
    expect(getRefreshToken()).toBe('rotated-refresh');
    const rotation = fetch.mock.calls.find(([url]) => String(url).endsWith('/auth/refresh'));
    expect(JSON.parse(String(rotation?.[1]?.body))).toEqual({ refresh_token: 'test-refresh' });
    expect(new Headers(rotation?.[1]?.headers).has('Authorization')).toBe(false);
  });

  it('clears a rejected session rather than retrying unauthorized requests forever', async () => {
    setSessionTokens({ access_token: 'expired-access', refresh_token: 'revoked-refresh' });
    const fetch = mockApi(() => jsonResponse({ detail: 'Session revoked' }, 401));
    await expect(apiClient.get('/private', { authenticated: true })).rejects.toMatchObject({ status: 401 });
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(getAccessToken()).toBeUndefined();
    expect(getRefreshToken()).toBeUndefined();
  });
});
