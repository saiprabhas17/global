import { fireEvent, screen, waitFor } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { AccountPanel } from '../src/components/AccountPanel';
import { getAccessToken, getRefreshToken } from '../src/lib/session';
import { accountFixture, jsonResponse, mockApi, renderWithClient, signInFixture } from './helpers';

function openLogin() {
  renderWithClient(<AccountPanel />);
  fireEvent.click(screen.getByRole('button', { name: 'Sign in' }));
  fireEvent.change(screen.getByLabelText('Email'), { target: { value: accountFixture.email } });
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'strong-password' } });
}

describe('memory-only account UI', () => {
  it('portals dialogs outside transformed or blurred header ancestors', () => {
    renderWithClient(<header style={{ backdropFilter: 'blur(4px)' }}><AccountPanel /></header>);
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }));
    const dialog = screen.getByRole('dialog');
    expect(dialog.closest('header')).toBeNull();
    expect(dialog.parentElement?.parentElement).toBe(document.body);
    fireEvent.click(screen.getByRole('button', { name: 'Close account' }));
    expect(screen.queryByRole('dialog')).toBeNull();
  });
  it('logs in and verifies /me using a bearer token without persisting credentials', async () => {
    const fetch = mockApi((endpoint) => endpoint === '/auth/login'
      ? jsonResponse({ access_token: 'login-access', refresh_token: 'login-refresh', token_type: 'bearer' })
      : jsonResponse(accountFixture));
    openLogin();
    fireEvent.click(screen.getByRole('button', { name: 'Log in' }));
    expect(await screen.findByText('Corpus Reviewer')).toBeTruthy();
    expect(getAccessToken()).toBe('login-access');
    expect(getRefreshToken()).toBe('login-refresh');
    expect(JSON.parse(String(fetch.mock.calls[0][1]?.body))).toEqual({ email: accountFixture.email, password: 'strong-password' });
    expect(String(fetch.mock.calls[1][0])).toContain('/auth/me');
    expect(new Headers(fetch.mock.calls[1][1]?.headers).get('Authorization')).toBe('Bearer login-access');
    expect(fetch.mock.calls.every(([, init]) => init?.credentials === 'omit')).toBe(true);
    expect(window.localStorage.length).toBe(0);
    expect(window.sessionStorage.length).toBe(0);
    expect(screen.queryByText('login-access')).toBeNull();
  });

  it('registers with the existing optional full_name contract', async () => {
    const fetch = mockApi((endpoint) => endpoint === '/auth/register'
      ? jsonResponse({ access_token: 'registered-access', refresh_token: 'registered-refresh' }, 201)
      : jsonResponse(accountFixture));
    openLogin();
    fireEvent.click(screen.getByRole('button', { name: 'Register' }));
    fireEvent.change(screen.getByLabelText('Full name (optional)'), { target: { value: '  Corpus Reviewer  ' } });
    expect(screen.getByLabelText('Password').getAttribute('minlength')).toBe('10');
    fireEvent.click(screen.getByRole('button', { name: 'Create account' }));
    await screen.findByText('Corpus Reviewer');
    expect(String(fetch.mock.calls[0][0])).toContain('/auth/register');
    expect(JSON.parse(String(fetch.mock.calls[0][1]?.body))).toEqual({ email: accountFixture.email, password: 'strong-password', full_name: 'Corpus Reviewer' });
  });

  it('shows authentication errors without claiming a signed-in identity', async () => {
    mockApi(() => jsonResponse({ detail: 'Incorrect email or password' }, 401));
    openLogin();
    fireEvent.click(screen.getByRole('button', { name: 'Log in' }));
    expect(await screen.findByText('Incorrect email or password')).toBeTruthy();
    expect(getAccessToken()).toBeUndefined();
    expect(screen.queryByText('Corpus Reviewer')).toBeNull();
  });

  it('rotates refresh tokens in memory and logs out with the actual rotated credentials', async () => {
    signInFixture();
    const fetch = mockApi((endpoint) => {
      if (endpoint === '/auth/refresh') return jsonResponse({ access_token: 'rotated-access', refresh_token: 'rotated-refresh' });
      if (endpoint === '/auth/logout') return new Response(null, { status: 204 });
      return jsonResponse(accountFixture);
    });
    renderWithClient(<AccountPanel />);
    fireEvent.click(screen.getByRole('button', { name: 'Account' }));
    fireEvent.click(screen.getByRole('button', { name: 'Rotate session' }));
    await screen.findByText(/Session rotated and account re-verified/);
    expect(getRefreshToken()).toBe('rotated-refresh');
    fireEvent.click(screen.getByRole('button', { name: 'Log out' }));
    await screen.findByText(/Signed out. In-memory credentials/);
    const refresh = fetch.mock.calls.find(([url]) => String(url).endsWith('/auth/refresh'));
    const logout = fetch.mock.calls.find(([url]) => String(url).endsWith('/auth/logout'));
    expect(JSON.parse(String(refresh?.[1]?.body))).toEqual({ refresh_token: 'test-refresh' });
    expect(JSON.parse(String(logout?.[1]?.body))).toEqual({ refresh_token: 'rotated-refresh' });
    expect(new Headers(logout?.[1]?.headers).get('Authorization')).toBe('Bearer rotated-access');
    expect(getAccessToken()).toBeUndefined();
    expect(getRefreshToken()).toBeUndefined();
    expect(window.localStorage.length).toBe(0);
  });

  it('clears local credentials and reports uncertainty if server logout fails', async () => {
    signInFixture();
    mockApi(() => jsonResponse({ detail: 'Redis unavailable' }, 503));
    renderWithClient(<AccountPanel />);
    fireEvent.click(screen.getByRole('button', { name: 'Account' }));
    fireEvent.click(screen.getByRole('button', { name: 'Log out' }));
    expect(await screen.findByText(/Server-side logout could not be confirmed/)).toBeTruthy();
    expect(screen.getByText('Redis unavailable')).toBeTruthy();
    expect(getAccessToken()).toBeUndefined();
    expect(getRefreshToken()).toBeUndefined();
  });

  it('labels the dialog, traps keyboard focus, closes on Escape and restores the opener', async () => {
    renderWithClient(<AccountPanel />);
    const opener = screen.getByRole('button', { name: 'Sign in' });
    opener.focus();
    fireEvent.click(opener);
    const dialog = screen.getByRole('dialog', { name: 'GMEE Account' });
    expect(document.activeElement).toBe(dialog);
    fireEvent.keyDown(dialog, { key: 'Tab' });
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Close account' }));
    fireEvent.keyDown(dialog, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(document.activeElement).toBe(opener);
  });
});
