interface ApiRequestOptions extends RequestInit {
  headers?: HeadersInit;
}

class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function getAuthHeaders(): HeadersInit {
  const token = typeof window !== 'undefined' ? localStorage.getItem("fa.jwt") : null;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function apiFetch(url: string, options: ApiRequestOptions = {}): Promise<Response> {
  const authHeaders = getAuthHeaders();
  
  const headers = {
    "Content-Type": "application/json",
    ...authHeaders,
    ...options.headers,
  };

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "Unknown error");
    throw new ApiError(response.status, response.statusText, errorText);
  }

  return response;
}

export async function apiGet<T = any>(url: string): Promise<T> {
  const response = await apiFetch(url, { method: "GET" });
  return response.json();
}

export async function apiPost<T = any>(url: string, data?: any): Promise<T> {
  const response = await apiFetch(url, {
    method: "POST",
    body: data ? JSON.stringify(data) : undefined,
  });
  return response.json();
}

export async function apiPut<T = any>(url: string, data?: any): Promise<T> {
  const response = await apiFetch(url, {
    method: "PUT",
    body: data ? JSON.stringify(data) : undefined,
  });
  return response.json();
}

export async function apiDelete<T = any>(url: string): Promise<T> {
  const response = await apiFetch(url, { method: "DELETE" });
  return response.json();
}

export { ApiError, apiFetch };