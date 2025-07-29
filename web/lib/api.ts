// web/lib/api.ts
export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public statusText?: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function api<T>(
  input: RequestInfo,
  init: RequestInit = {}
): Promise<T> {
  const token = typeof window !== 'undefined' ? localStorage.getItem("fa.jwt") : null;
  const headers = new Headers(init.headers);
  headers.set('Content-Type', 'application/json');
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  
  try {
    const response = await fetch(input, { ...init, headers, credentials: "include" });
    
    if (response.status === 401) {
      if (typeof window !== 'undefined') {
        localStorage.removeItem("fa.jwt");
        location.reload();
      }
      throw new ApiError("Unauthenticated", 401);
    }
    
    if (!response.ok) {
      throw new ApiError(
        `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        response.statusText
      );
    }
    
    return response.json();
  } catch (error) {
    console.error('API request failed:', error);
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(error instanceof Error ? error.message : 'Unknown error');
  }
}

function getAuthHeaders(): HeadersInit {
  const token = typeof window !== 'undefined' ? localStorage.getItem("fa.jwt") : null;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function fetchJson(url: string, opts: RequestInit = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...getAuthHeaders(),
    ...(opts.headers || {}),
  };
  
  try {
    const response = await fetch(url, { ...opts, headers });
    
    if (!response.ok) {
      throw new ApiError(
        `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        response.statusText
      );
    }
    
    return response;
  } catch (error) {
    console.error('API request failed:', error);
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(error instanceof Error ? error.message : 'Unknown error');
  }
}

const getBaseUrl = () => process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function apiGet(endpoint: string, opts: RequestInit = {}) {
  const response = await fetchJson(`${getBaseUrl()}${endpoint}`, { method: 'GET', ...opts });
  return response.json();
}

export async function apiPost(endpoint: string, data?: unknown, opts: RequestInit = {}) {
  const response = await fetchJson(`${getBaseUrl()}${endpoint}`, {
    method: 'POST',
    body: data ? JSON.stringify(data) : undefined,
    ...opts
  });
  return response.json();
}

export async function apiPut(endpoint: string, data?: unknown, opts: RequestInit = {}) {
  const response = await fetchJson(`${getBaseUrl()}${endpoint}`, {
    method: 'PUT',
    body: data ? JSON.stringify(data) : undefined,
    ...opts
  });
  return response.json();
}

export async function apiDelete(endpoint: string, opts: RequestInit = {}) {
  const response = await fetchJson(`${getBaseUrl()}${endpoint}`, { method: 'DELETE', ...opts });
  return response.json();
}