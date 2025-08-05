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
  
  try {
    // If we have a complete WebSocket URL in env, use it
    if (envUrl) {
      // Check if it's already a complete WebSocket URL with path
      if (envUrl.includes('/api/v1/ws/')) {
        if (!isValidWebSocketUrl(envUrl)) {
          console.error(`Invalid WebSocket URL in NEXT_PUBLIC_WS_URL: ${envUrl}`);
          throw new Error(`Invalid WebSocket URL: ${envUrl}`);
        }
        return envUrl;
      }
      
      // If it's a base WebSocket URL, append the path
      const baseUrl = ensureWebSocketProtocol(envUrl).replace(/\/$/, '');
      const fullUrl = `${baseUrl}/api/v1/ws/${endpoint}`;
      
      if (!isValidWebSocketUrl(fullUrl)) {
        console.error(`Constructed invalid WebSocket URL: ${fullUrl} from base: ${envUrl}`);
        throw new Error(`Invalid WebSocket URL constructed: ${fullUrl}`);
      }
      
      return fullUrl;
    }
    
    // Fallback: derive from backend URL
    const baseUrl = ensureWebSocketProtocol(backendUrl).replace(/\/$/, '');
    const fullUrl = `${baseUrl}/api/v1/ws/${endpoint}`;
    
    if (!isValidWebSocketUrl(fullUrl)) {
      console.error(`Fallback WebSocket URL is invalid: ${fullUrl} from backend: ${backendUrl}`);
      throw new Error(`Invalid WebSocket URL constructed: ${fullUrl}`);
    }
    
    console.log(`Using fallback WebSocket URL: ${fullUrl} (NEXT_PUBLIC_WS_URL not set)`);
    return fullUrl;
    
  } catch (error) {
    console.error('WebSocket URL construction failed:', {
      envUrl,
      backendUrl,
      endpoint,
      error: error instanceof Error ? error.message : String(error)
    });
    throw error;
  }
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