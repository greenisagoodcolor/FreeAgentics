/**
 * WebSocket URL construction utilities
 * Centralizes WebSocket URL logic to prevent inconsistencies
 */

export type WebSocketEndpoint = 'dev' | 'demo' | 'auth';

/**
 * Constructs a complete WebSocket URL for the specified endpoint
 * Handles both base URLs and complete URLs from environment variables
 */
export function getWebSocketURL(endpoint: WebSocketEndpoint = 'dev'): string {
  const baseUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
  
  // If the base URL already includes an API path, use it as-is
  if (baseUrl.includes("/api/v1/ws/") || baseUrl.includes("/ws/")) {
    return baseUrl;
  }
  
  // Otherwise, construct the URL with the appropriate endpoint
  const endpointPath = `/api/v1/ws/${endpoint}`;
  return `${baseUrl}${endpointPath}`;
}

/**
 * Validates that a WebSocket URL has the correct format
 */
export function validateWebSocketURL(url: string): boolean {
  try {
    const wsUrl = new URL(url);
    
    // Must be a WebSocket protocol
    if (!['ws:', 'wss:'].includes(wsUrl.protocol)) {
      return false;
    }
    
    // Must have a valid path pattern
    const validPathPrefixes = [
      '/api/v1/ws/',
      '/ws/', // legacy
    ];
    
    return validPathPrefixes.some(prefix => wsUrl.pathname.startsWith(prefix));
  } catch {
    return false;
  }
}

/**
 * Gets WebSocket URL with authentication token if provided
 */
export function getAuthenticatedWebSocketURL(endpoint: WebSocketEndpoint = 'dev', token?: string): string {
  const baseUrl = getWebSocketURL(endpoint);
  
  if (!token || endpoint === 'demo') {
    return baseUrl;
  }
  
  const url = new URL(baseUrl);
  url.searchParams.set('token', token);
  return url.toString();
}