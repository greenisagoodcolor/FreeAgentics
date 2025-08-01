import { getWebSocketURL, validateWebSocketURL, getAuthenticatedWebSocketURL } from '../websocket-url';

// Mock environment variables
const originalEnv = process.env;

beforeEach(() => {
  jest.resetModules();
  process.env = { ...originalEnv };
});

afterAll(() => {
  process.env = originalEnv;
});

describe('getWebSocketURL', () => {
  it('should use default localhost URL when no env var is set', () => {
    delete process.env.NEXT_PUBLIC_WS_URL;
    const url = getWebSocketURL('dev');
    expect(url).toBe('ws://localhost:8000/api/v1/ws/dev');
  });

  it('should append path to base URL from env var', () => {
    process.env.NEXT_PUBLIC_WS_URL = 'ws://localhost:8001';
    const url = getWebSocketURL('dev');
    expect(url).toBe('ws://localhost:8001/api/v1/ws/dev');
  });

  it('should use complete URL from env var as-is if it includes API path', () => {
    process.env.NEXT_PUBLIC_WS_URL = 'ws://localhost:8001/api/v1/ws/custom';
    const url = getWebSocketURL('dev');
    expect(url).toBe('ws://localhost:8001/api/v1/ws/custom');
  });

  it('should handle different endpoints', () => {
    delete process.env.NEXT_PUBLIC_WS_URL;
    expect(getWebSocketURL('dev')).toBe('ws://localhost:8000/api/v1/ws/dev');
    expect(getWebSocketURL('demo')).toBe('ws://localhost:8000/api/v1/ws/demo');
    expect(getWebSocketURL('auth')).toBe('ws://localhost:8000/api/v1/ws/auth');
  });

  it('should handle legacy ws path format', () => {
    process.env.NEXT_PUBLIC_WS_URL = 'ws://localhost:8001/ws/demo';
    const url = getWebSocketURL('dev');
    expect(url).toBe('ws://localhost:8001/ws/demo');
  });
});

describe('validateWebSocketURL', () => {
  it('should validate correct WebSocket URLs', () => {
    expect(validateWebSocketURL('ws://localhost:8000/api/v1/ws/dev')).toBe(true);
    expect(validateWebSocketURL('wss://localhost:8000/api/v1/ws/demo')).toBe(true);
    expect(validateWebSocketURL('ws://example.com/ws/demo')).toBe(true);
    expect(validateWebSocketURL('ws://server:8080/api/v1/ws/custom')).toBe(true);
  });

  it('should reject invalid protocols', () => {
    expect(validateWebSocketURL('http://localhost:8000/api/v1/ws/dev')).toBe(false);
    expect(validateWebSocketURL('https://localhost:8000/api/v1/ws/dev')).toBe(false);
  });

  it('should reject invalid paths', () => {
    expect(validateWebSocketURL('ws://localhost:8000/')).toBe(false);
    expect(validateWebSocketURL('ws://localhost:8000/api/v1/wrong')).toBe(false);
    expect(validateWebSocketURL('ws://localhost:8000/random/path')).toBe(false);
  });

  it('should reject malformed URLs', () => {
    expect(validateWebSocketURL('not-a-url')).toBe(false);
    expect(validateWebSocketURL('')).toBe(false);
    expect(validateWebSocketURL('ws://')).toBe(false);
  });
});

describe('getAuthenticatedWebSocketURL', () => {
  it('should add token for non-demo endpoints', () => {
    delete process.env.NEXT_PUBLIC_WS_URL;
    const url = getAuthenticatedWebSocketURL('dev', 'test-token');
    expect(url).toBe('ws://localhost:8000/api/v1/ws/dev?token=test-token');
  });

  it('should not add token for demo endpoint', () => {
    delete process.env.NEXT_PUBLIC_WS_URL;
    const url = getAuthenticatedWebSocketURL('demo', 'test-token');
    expect(url).toBe('ws://localhost:8000/api/v1/ws/demo');
  });

  it('should not add token when none provided', () => {
    delete process.env.NEXT_PUBLIC_WS_URL;
    const url = getAuthenticatedWebSocketURL('dev');
    expect(url).toBe('ws://localhost:8000/api/v1/ws/dev');
  });

  it('should handle existing query parameters', () => {
    process.env.NEXT_PUBLIC_WS_URL = 'ws://localhost:8000';
    const url = getAuthenticatedWebSocketURL('auth', 'test-token');
    expect(url).toBe('ws://localhost:8000/api/v1/ws/auth?token=test-token');
  });
});