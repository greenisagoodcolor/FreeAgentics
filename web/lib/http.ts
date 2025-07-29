import { ensureDevToken } from './auth';

export async function api(path: string, opts: RequestInit = {}) {
  const token = await ensureDevToken();
  if (!token) {
    throw new Error('No authentication token available');
  }
  
  const doFetch = (tok: string) =>
    fetch(path, {
      ...opts,
      headers: { ...(opts.headers || {}), Authorization: `Bearer ${tok}` },
    });

  let res = await doFetch(token);
  if (res.status === 401) {
    const fresh = await ensureDevToken(); // renews & caches
    if (!fresh) {
      throw new Error('Failed to refresh authentication token');
    }
    res = await doFetch(fresh);
  }
  return res;
}