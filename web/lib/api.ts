// web/lib/api.ts
export async function fetchJson(url: string, opts: RequestInit = {}) {
  const token = localStorage.getItem('devToken');
  const headers = {
    ...(opts.headers || {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  return fetch(url, { ...opts, headers });
}