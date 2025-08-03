/**
 * Centralized WebSocket URL construction utility
 * Ensures consistent URL formatting across the application
 */

export interface WebSocketEndpoints {
  dev: string;
  demo: string;
  auth: string;
}

/**
 * Validates if a URL is a valid WebSocket URL
 */
export function isValidWebSocketUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'ws:' || parsed.protocol === 'wss:';
  } catch {
    return false;
  }
}

/**
 * Ensures a URL has the WebSocket protocol
 */
export function ensureWebSocketProtocol(url: string): string {
  if (url.startsWith('http://')) {
    return url.replace('http://', 'ws://');
  }
  if (url.startsWith('https://')) {
    return url.replace('https://', 'wss://');
  }
  if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
    // Assume ws:// for local development
    return `ws://${url}`;
  }
  return url;
}

/**
 * Constructs WebSocket URLs for different endpoints
 * Handles both base URLs and complete URLs from environment
 */
export function getWebSocketUrl(endpoint: 'dev' | 'demo' | 'auth' = 'dev'): string {
  const envUrl = process.env.NEXT_PUBLIC_WS_URL;
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
  
  // Start with env URL or derive from backend URL
  let baseUrl = envUrl || backendUrl;
  
  // Ensure WebSocket protocol
  baseUrl = ensureWebSocketProtocol(baseUrl);
  
  // Check if the URL already includes the full path
  if (baseUrl.includes('/api/v1/ws/')) {
    // If it's already a complete URL, validate and return
    if (!isValidWebSocketUrl(baseUrl)) {
      throw new Error(`Invalid WebSocket URL: ${baseUrl}`);
    }
    return baseUrl;
  }
  
  // Remove trailing slash
  baseUrl = baseUrl.replace(/\/$/, '');
  
  // Construct full URL with endpoint
  const fullUrl = `${baseUrl}/api/v1/ws/${endpoint}`;
  
  // Validate the constructed URL
  if (!isValidWebSocketUrl(fullUrl)) {
    throw new Error(`Invalid WebSocket URL constructed: ${fullUrl}`);
  }
  
  return fullUrl;
}

/**
 * Gets WebSocket URL with authentication token
 */
export function getAuthenticatedWebSocketUrl(
  endpoint: 'dev' | 'demo' | 'auth' = 'dev',
  token?: string
): string {
  const baseUrl = getWebSocketUrl(endpoint);
  
  if (!token) {
    return baseUrl;
  }
  
  // Add token as query parameter
  const url = new URL(baseUrl);
  url.searchParams.set('token', token);
  return url.toString();
}

/**
 * Gets all WebSocket endpoints
 */
export function getAllWebSocketEndpoints(): WebSocketEndpoints {
  return {
    dev: getWebSocketUrl('dev'),
    demo: getWebSocketUrl('demo'),
    auth: getWebSocketUrl('auth'),
  };
}