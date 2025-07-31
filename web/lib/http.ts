const AUTH_TOKEN_KEY = "fa.jwt";

async function ensureDevToken(): Promise<string | null> {
  // First try localStorage
  const stored = localStorage.getItem(AUTH_TOKEN_KEY);
  if (stored && stored !== "dev") {
    // Basic JWT expiration check
    try {
      const payload = JSON.parse(atob(stored.split('.')[1]));
      const now = Math.floor(Date.now() / 1000);
      if (payload.exp > now) {
        return stored;
      }
    } catch {
      // Invalid token, continue to fetch new one
    }
  }

  // Fetch dev token from backend
  try {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const response = await fetch(`${backendUrl}/api/v1/dev-config`);
    if (response.ok) {
      const config = await response.json();
      const token = config.auth?.token;
      if (token) {
        localStorage.setItem(AUTH_TOKEN_KEY, token);
        return token;
      }
    }
  } catch (error) {
    console.error('Failed to fetch dev token:', error);
  }

  // Fallback to "dev" token for development
  const devToken = "dev";
  localStorage.setItem(AUTH_TOKEN_KEY, devToken);
  return devToken;
}

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