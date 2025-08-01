/**
 * @jest-environment jsdom
 */

import { renderHook } from '@testing-library/react';
import { useWebSocket } from '../use-websocket';

// Mock useAuth hook
jest.mock('../use-auth', () => ({
  useAuth: () => ({
    isAuthenticated: false,
    isLoading: false,
    token: null,
  }),
}));

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  url: string;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    // Store the URL that was actually used for connection
    MockWebSocket.lastConnectionUrl = url;
  }

  send(data: string) {}
  close() {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) this.onclose();
  }

  static lastConnectionUrl: string = '';
}

global.WebSocket = MockWebSocket as any;

// Mock console.log to capture URL logging
const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

describe('useWebSocket URL Construction', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    consoleSpy.mockClear();
    MockWebSocket.lastConnectionUrl = '';
    
    // Clear environment variable
    delete (process.env as any).NEXT_PUBLIC_WS_URL;
  });

  afterAll(() => {
    consoleSpy.mockRestore();
  });

  test('constructs correct dev endpoint URL by default', async () => {
    const { result } = renderHook(() => useWebSocket());

    // Wait for connection attempt
    await new Promise(resolve => setTimeout(resolve, 150));

    // Should log the final URL with proper path
    expect(consoleSpy).toHaveBeenCalledWith(
      '[WebSocket] Final URL:',
      'ws://localhost:8000/api/v1/ws/dev'
    );
  });

  test('constructs URL from environment variable', async () => {
    (process.env as any).NEXT_PUBLIC_WS_URL = 'ws://production-server:8080';
    
    const { result } = renderHook(() => useWebSocket());

    // Wait for connection attempt
    await new Promise(resolve => setTimeout(resolve, 150));

    // Should construct URL from environment variable with dev path appended
    expect(consoleSpy).toHaveBeenCalledWith(
      '[WebSocket] Final URL:',
      'ws://production-server:8080/api/v1/ws/dev'
    );
  });

  test('preserves path when constructing WebSocket URL', async () => {
    // Set an environment variable with existing path
    (process.env as any).NEXT_PUBLIC_WS_URL = 'ws://server:8080/custom/path';
    
    const { result } = renderHook(() => useWebSocket());

    // Wait for connection attempt
    await new Promise(resolve => setTimeout(resolve, 150));

    // Should properly combine paths
    expect(consoleSpy).toHaveBeenCalledWith(
      '[WebSocket] Final URL:',
      'ws://server:8080/custom/path/api/v1/ws/dev'
    );
  });

  test('logs dev endpoint connection message', async () => {
    const { result } = renderHook(() => useWebSocket());

    // Wait for connection attempt
    await new Promise(resolve => setTimeout(resolve, 150));

    // Should log dev endpoint connection
    expect(consoleSpy).toHaveBeenCalledWith(
      '[WebSocket] Connecting to dev endpoint without auth...'
    );
  });
});