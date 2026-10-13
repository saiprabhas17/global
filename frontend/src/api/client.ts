import {
  clearSession, getAccessToken, getRefreshToken, getSessionRevision, setSessionTokens,
  type SessionTokens,
} from '../lib/session';

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '/api/v1').replace(/\/$/, '');
const DEFAULT_TIMEOUT_MS = 15_000;

export type ApiErrorKind = 'http' | 'network' | 'timeout' | 'aborted';

function errorMessage(status: number, body: unknown): string {
  if (typeof body === 'string' && body) return body;
  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = body.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      const messages = detail.flatMap((item: unknown) =>
        item && typeof item === 'object' && 'msg' in item && typeof item.msg === 'string' ? [item.msg] : [],
      );
      if (messages.length) return messages.join('; ');
    }
  }
  return `HTTP ${status}`;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: unknown,
    message = errorMessage(status, body),
    public readonly kind: ApiErrorKind = 'http',
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export interface ApiRequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  authenticated?: boolean;
  anonymous?: boolean;
  refreshOnUnauthorized?: boolean;
}

export interface ApiDownload {
  blob: Blob;
  filename: string;
}

async function readBody(response: Response): Promise<unknown> {
  if (response.status === 204) return undefined;
  const text = await response.text();
  if (!text) return undefined;
  try { return JSON.parse(text) as unknown; } catch { return text; }
}

async function performRequest<T>(
  method: 'GET' | 'POST', endpoint: string, body: unknown,
  options: ApiRequestOptions, download = false,
): Promise<T> {
  const accessToken = options.anonymous ? undefined : getAccessToken();
  if (options.authenticated && !accessToken) {
    throw new ApiError(401, { detail: 'Sign in to use this feature.' });
  }
  if (options.signal?.aborted) {
    throw new ApiError(0, null, 'Request cancelled.', 'aborted');
  }

  const controller = new AbortController();
  let timedOut = false;
  const abort = () => controller.abort();
  options.signal?.addEventListener('abort', abort, { once: true });
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  let onAbort: () => void = () => {};
  const cancelled = new Promise<never>((_resolve, reject) => {
    onAbort = () => reject(new ApiError(
      0, null, timedOut ? `Request timed out after ${timeoutMs / 1000}s. Please retry.` : 'Request cancelled.',
      timedOut ? 'timeout' : 'aborted',
    ));
    controller.signal.addEventListener('abort', onAbort, { once: true });
  });

  try {
    const operation = async (): Promise<T> => {
      const headers = new Headers({ Accept: download ? '*/*' : 'application/json' });
      if (body !== undefined) headers.set('Content-Type', 'application/json');
      if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`);
      const response = await fetch(`${BASE_URL}${endpoint}`, {
        method, headers, body: body === undefined ? undefined : JSON.stringify(body),
        credentials: 'omit', signal: controller.signal,
      });
      if (!response.ok) throw new ApiError(response.status, await readBody(response));
      if (download) {
        const disposition = response.headers.get('Content-Disposition') ?? '';
        const filename = disposition.match(/filename="?([^";]+)"?/i)?.[1]?.trim().replace(/[\\/]/g, '_')
          || 'gmee-eval-export.json';
        return { blob: await response.blob(), filename } as T;
      }
      return await readBody(response) as T;
    };
    return await Promise.race([operation(), cancelled]);
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(0, null, 'Unable to reach the GMEE API. Check your connection and retry.', 'network');
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener('abort', abort);
    controller.signal.removeEventListener('abort', onAbort);
  }
}

let refreshInFlight: Promise<void> | null = null;

export async function rotateAccessToken(): Promise<void> {
  if (refreshInFlight) return refreshInFlight;
  const refreshToken = getRefreshToken();
  if (!refreshToken) throw new ApiError(401, { detail: 'This session has no refresh token. Sign in again.' });
  const revision = getSessionRevision();
  const rotation = (async () => {
    const next = await performRequest<SessionTokens>('POST', '/auth/refresh', { refresh_token: refreshToken }, { anonymous: true });
    if (revision !== getSessionRevision()) throw new ApiError(401, { detail: 'The session changed. Sign in again.' });
    if (!next?.access_token) throw new ApiError(200, next, 'The authentication response did not contain an access token.');
    setSessionTokens(next);
  })();
  refreshInFlight = rotation;
  try {
    await rotation;
  } catch (error) {
    if (error instanceof ApiError && error.status === 401 && revision === getSessionRevision()) clearSession();
    throw error;
  } finally {
    if (refreshInFlight === rotation) refreshInFlight = null;
  }
}

async function request<T>(method: 'GET' | 'POST', endpoint: string, body: unknown, options: ApiRequestOptions = {}, download = false): Promise<T> {
  const revision = getSessionRevision();
  const token = getAccessToken();
  try {
    return await performRequest<T>(method, endpoint, body, options, download);
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 401 || !options.authenticated) throw error;
    if (options.signal?.aborted) throw new ApiError(0, null, 'Request cancelled.', 'aborted');
    if (options.refreshOnUnauthorized !== false && getRefreshToken() && revision === getSessionRevision()) {
      // Concurrent protected requests share one rotation, never reuse the old refresh token.
      if (getAccessToken() === token) await rotateAccessToken();
      try { return await performRequest<T>(method, endpoint, body, options, download); } catch (retryError) {
        if (retryError instanceof ApiError && retryError.status === 401 && revision === getSessionRevision()) clearSession();
        throw retryError;
      }
    }
    if (revision === getSessionRevision()) clearSession();
    throw error;
  }
}

export const apiClient = {
  get: <T = unknown>(endpoint: string, options?: ApiRequestOptions): Promise<T> => request('GET', endpoint, undefined, options),
  post: <T = unknown>(endpoint: string, body: unknown, options?: ApiRequestOptions): Promise<T> => request('POST', endpoint, body, options),
  download: (endpoint: string, options?: ApiRequestOptions): Promise<ApiDownload> => request('GET', endpoint, undefined, options, true),
};
