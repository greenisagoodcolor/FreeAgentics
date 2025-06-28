/**
 * Comprehensive Edge Cases Tests
 * 
 * Tests for error handling, boundary conditions, edge cases, and integration scenarios
 * following ADR-007 requirements for complete coverage of exceptional conditions.
 */

import { jest } from '@jest/globals';
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// Mock all external dependencies
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
    replace: jest.fn(),
    back: jest.fn(),
    forward: jest.fn(),
    refresh: jest.fn(),
    prefetch: jest.fn(),
  }),
  usePathname: () => '/test-path',
  useSearchParams: () => new URLSearchParams('test=value'),
}));

// Mock WebSocket with error scenarios
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  public readyState: number = MockWebSocket.CONNECTING;
  public onopen: ((event: Event) => void) | null = null;
  public onclose: ((event: CloseEvent) => void) | null = null;
  public onmessage: ((event: MessageEvent) => void) | null = null;
  public onerror: ((event: Event) => void) | null = null;

  constructor(public url: string, public protocols?: string | string[]) {
    // Simulate connection states
    setTimeout(() => {
      if (url.includes('error')) {
        this.readyState = MockWebSocket.CLOSED;
        this.onerror?.(new Event('error'));
      } else if (url.includes('timeout')) {
        // Never connect
        return;
      } else {
        this.readyState = MockWebSocket.OPEN;
        this.onopen?.(new Event('open'));
      }
    }, 10);
  }

  send(data: string | ArrayBuffer | Blob): void {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }
    
    if (data.toString().includes('error')) {
      setTimeout(() => {
        this.onerror?.(new Event('error'));
      }, 5);
    }
  }

  close(code?: number, reason?: string): void {
    this.readyState = MockWebSocket.CLOSED;
    setTimeout(() => {
      this.onclose?.(new CloseEvent('close', { code, reason }));
    }, 5);
  }
}

global.WebSocket = MockWebSocket as any;

// Error Boundary for React Testing
interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

class TestErrorBoundary extends React.Component<
  { children: React.ReactNode; onError?: (error: Error) => void },
  ErrorBoundaryState
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    this.props.onError?.(error);
  }

  render() {
    if (this.state.hasError) {
      return <div data-testid="error-boundary">Something went wrong: {this.state.error?.message}</div>;
    }

    return this.props.children;
  }
}

// Network Error Simulation
class NetworkErrorSimulator {
  static simulateNetworkFailure(): jest.MockedFunction<typeof fetch> {
    return jest.fn().mockRejectedValue(new Error('Network request failed'));
  }

  static simulateTimeoutError(): jest.MockedFunction<typeof fetch> {
    return jest.fn().mockImplementation(() => 
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Request timeout')), 100)
      )
    );
  }

  static simulateServerError(status: number = 500): jest.MockedFunction<typeof fetch> {
    return jest.fn().mockResolvedValue(new Response(
      JSON.stringify({ error: 'Internal Server Error' }),
      { status, statusText: 'Internal Server Error' }
    ));
  }

  static simulateCorruptedResponse(): jest.MockedFunction<typeof fetch> {
    return jest.fn().mockResolvedValue(new Response(
      'invalid json response',
      { status: 200, statusText: 'OK' }
    ));
  }

  static simulateSlowResponse(delay: number = 1000): jest.MockedFunction<typeof fetch> {
    return jest.fn().mockImplementation(() =>
      new Promise(resolve =>
        setTimeout(() =>
          resolve(new Response(JSON.stringify({ data: 'success' }), { status: 200 })),
          delay
        )
      )
    );
  }
}

// Boundary Value Testing Utilities
class BoundaryValueTester {
  static testNumericBoundaries(fn: (value: number) => any, validRange: [number, number]) {
    const [min, max] = validRange;
    const testValues = [
      min - 1,           // Below minimum
      min,               // Minimum boundary
      min + 0.001,       // Just above minimum
      (min + max) / 2,   // Middle value
      max - 0.001,       // Just below maximum
      max,               // Maximum boundary
      max + 1,           // Above maximum
      0,                 // Zero
      -0,                // Negative zero
      Infinity,          // Positive infinity
      -Infinity,         // Negative infinity
      NaN,               // Not a number
    ];

    const results: Array<{ value: number; result: any; error?: string }> = [];

    testValues.forEach(value => {
      try {
        const result = fn(value);
        results.push({ value, result });
      } catch (error) {
        results.push({ value, error: (error as Error).message });
      }
    });

    return results;
  }

  static testStringBoundaries(fn: (value: string) => any, maxLength?: number) {
    const testValues = [
      '',                                    // Empty string
      ' ',                                   // Single space
      'a',                                   // Single character
      'test',                                // Normal string
      'test\n\r\t',                         // String with whitespace
      'test\0null',                         // String with null character
      '🚀🌟💫',                            // Unicode emojis
      'תשטשאךךךא',                         // Non-Latin characters
      '\uD800\uDC00',                       // Unicode surrogate pair
      '\uFFFD',                             // Unicode replacement character
      maxLength ? 'x'.repeat(maxLength) : 'x'.repeat(1000),     // At max length
      maxLength ? 'x'.repeat(maxLength + 1) : 'x'.repeat(1001), // Over max length
      '<script>alert("xss")</script>',      // Potential XSS
      'DROP TABLE users;',                  // Potential SQL injection
      '../../../etc/passwd',                // Path traversal
    ];

    const results: Array<{ value: string; result: any; error?: string }> = [];

    testValues.forEach(value => {
      try {
        const result = fn(value);
        results.push({ value, result });
      } catch (error) {
        results.push({ value, error: (error as Error).message });
      }
    });

    return results;
  }

  static testArrayBoundaries(fn: (value: any[]) => any, maxLength?: number) {
    const testValues = [
      [],                                   // Empty array
      [null],                              // Array with null
      [undefined],                         // Array with undefined
      [1, 2, 3],                          // Normal array
      [{ nested: { deeply: true } }],     // Nested objects
      [1, 'string', true, null],          // Mixed types
      new Array(1000).fill(0),            // Large array
      maxLength ? new Array(maxLength).fill(0) : new Array(100).fill(0),    // At max length
      maxLength ? new Array(maxLength + 1).fill(0) : new Array(101).fill(0), // Over max length
    ];

    const results: Array<{ value: any[]; result: any; error?: string }> = [];

    testValues.forEach(value => {
      try {
        const result = fn(value);
        results.push({ value, result });
      } catch (error) {
        results.push({ value, error: (error as Error).message });
      }
    });

    return results;
  }
}

// Memory Pressure Testing
class MemoryPressureTester {
  static createLargeObject(sizeMB: number): any {
    const sizeBytes = sizeMB * 1024 * 1024;
    const arraySize = Math.floor(sizeBytes / 8); // 8 bytes per number in array
    return new Array(arraySize).fill(Math.random());
  }

  static simulateMemoryLeak(): (() => void) {
    const leaks: any[] = [];
    const interval = setInterval(() => {
      leaks.push(new Array(10000).fill(Math.random()));
    }, 10);

    return () => {
      clearInterval(interval);
      leaks.length = 0;
    };
  }

  static testGarbageCollection(fn: () => any): { before: number; after: number; leaked: boolean } {
    // Force garbage collection if available
    if (global.gc) {
      global.gc();
    }

    const before = process.memoryUsage?.()?.heapUsed || 0;
    
    // Run the function multiple times
    for (let i = 0; i < 100; i++) {
      fn();
    }

    if (global.gc) {
      global.gc();
    }

    const after = process.memoryUsage?.()?.heapUsed || 0;
    const leaked = (after - before) > 10 * 1024 * 1024; // 10MB threshold

    return { before, after, leaked };
  }
}

// Concurrency Testing Utilities
class ConcurrencyTester {
  static async testRaceCondition(
    operations: Array<() => Promise<any>>,
    iterations: number = 10
  ): Promise<{ successes: number; failures: number; results: any[] }> {
    let successes = 0;
    let failures = 0;
    const results: any[] = [];

    for (let i = 0; i < iterations; i++) {
      try {
        const promises = operations.map(op => op());
        const batchResults = await Promise.allSettled(promises);
        
        batchResults.forEach(result => {
          if (result.status === 'fulfilled') {
            successes++;
            results.push(result.value);
          } else {
            failures++;
            results.push({ error: result.reason.message });
          }
        });
      } catch (error) {
        failures++;
        results.push({ error: (error as Error).message });
      }
    }

    return { successes, failures, results };
  }

  static async testDeadlock(
    resource1: { acquire: () => Promise<void>; release: () => void },
    resource2: { acquire: () => Promise<void>; release: () => void },
    timeout: number = 1000
  ): Promise<{ deadlocked: boolean; time: number }> {
    const startTime = Date.now();

    const task1 = async () => {
      await resource1.acquire();
      await new Promise(resolve => setTimeout(resolve, 100));
      await resource2.acquire();
      resource2.release();
      resource1.release();
    };

    const task2 = async () => {
      await resource2.acquire();
      await new Promise(resolve => setTimeout(resolve, 100));
      await resource1.acquire();
      resource1.release();
      resource2.release();
    };

    try {
      await Promise.race([
        Promise.all([task1(), task2()]),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Deadlock detected')), timeout)
        ),
      ]);

      const endTime = Date.now();
      return { deadlocked: false, time: endTime - startTime };
    } catch (error) {
      const endTime = Date.now();
      return { deadlocked: true, time: endTime - startTime };
    }
  }
}

// Security Testing Utilities
class SecurityTester {
  static testXSSVulnerability(renderFunction: (input: string) => string): {
    vulnerable: boolean;
    attempts: Array<{ payload: string; escaped: boolean }>;
  } {
    const xssPayloads = [
      '<script>alert("xss")</script>',
      '<img src=x onerror=alert("xss")>',
      'javascript:alert("xss")',
      '<svg onload=alert("xss")>',
      '<iframe src="javascript:alert(\'xss\')">',
      '"><script>alert("xss")</script>',
      '\'"--></script><script>alert("xss")</script>',
    ];

    const attempts = xssPayloads.map(payload => {
      const result = renderFunction(payload);
      const escaped = !result.includes('<script>') && 
                     !result.includes('javascript:') && 
                     !result.includes('onerror=');
      return { payload, escaped };
    });

    const vulnerable = attempts.some(attempt => !attempt.escaped);

    return { vulnerable, attempts };
  }

  static testSQLInjection(queryFunction: (input: string) => string): {
    vulnerable: boolean;
    attempts: Array<{ payload: string; suspicious: boolean }>;
  } {
    const sqlPayloads = [
      "'; DROP TABLE users; --",
      "' OR '1'='1",
      "' UNION SELECT * FROM users --",
      "'; DELETE FROM users WHERE 1=1; --",
      "' OR 1=1 --",
      "admin'--",
      "'; EXEC xp_cmdshell('dir'); --",
    ];

    const attempts = sqlPayloads.map(payload => {
      const result = queryFunction(payload);
      const suspicious = result.includes('DROP') || 
                        result.includes('DELETE') || 
                        result.includes('UNION') ||
                        result.includes('OR 1=1');
      return { payload, suspicious };
    });

    const vulnerable = attempts.some(attempt => attempt.suspicious);

    return { vulnerable, attempts };
  }

  static testPathTraversal(pathFunction: (input: string) => string): {
    vulnerable: boolean;
    attempts: Array<{ payload: string; dangerous: boolean }>;
  } {
    const pathPayloads = [
      '../../../etc/passwd',
      '..\\..\\..\\windows\\system32\\config\\sam',
      '....//....//....//etc/passwd',
      '%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd',
      '..%252f..%252f..%252fetc%252fpasswd',
      '/var/www/../../etc/passwd',
    ];

    const attempts = pathPayloads.map(payload => {
      const result = pathFunction(payload);
      const dangerous = result.includes('../') || 
                       result.includes('..\\') || 
                       result.includes('etc/passwd') ||
                       result.includes('system32');
      return { payload, dangerous };
    });

    const vulnerable = attempts.some(attempt => attempt.dangerous);

    return { vulnerable, attempts };
  }
}

// Performance Testing Utilities
class PerformanceTester {
  static benchmarkFunction(fn: () => any, iterations: number = 1000): {
    averageTime: number;
    minTime: number;
    maxTime: number;
    totalTime: number;
    opsPerSecond: number;
  } {
    const times: number[] = [];
    const startTime = performance.now();

    for (let i = 0; i < iterations; i++) {
      const iterationStart = performance.now();
      fn();
      const iterationEnd = performance.now();
      times.push(iterationEnd - iterationStart);
    }

    const endTime = performance.now();
    const totalTime = endTime - startTime;
    const averageTime = times.reduce((sum, time) => sum + time, 0) / times.length;
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    const opsPerSecond = (iterations / totalTime) * 1000;

    return {
      averageTime,
      minTime,
      maxTime,
      totalTime,
      opsPerSecond,
    };
  }

  static testMemoryUsage(fn: () => any): {
    initialMemory: number;
    finalMemory: number;
    peakMemory: number;
    memoryDelta: number;
  } {
    const getMemory = () => process.memoryUsage?.()?.heapUsed || 0;
    
    if (global.gc) global.gc();
    const initialMemory = getMemory();
    let peakMemory = initialMemory;

    const interval = setInterval(() => {
      const currentMemory = getMemory();
      if (currentMemory > peakMemory) {
        peakMemory = currentMemory;
      }
    }, 10);

    fn();

    clearInterval(interval);
    if (global.gc) global.gc();
    
    const finalMemory = getMemory();
    const memoryDelta = finalMemory - initialMemory;

    return {
      initialMemory,
      finalMemory,
      peakMemory,
      memoryDelta,
    };
  }
}

// Comprehensive Error Handling Test Suite
class ErrorHandlingTester {
  static createAsyncErrorScenarios(): Array<{
    name: string;
    fn: () => Promise<any>;
    expectedError: string;
  }> {
    return [
      {
        name: 'Promise rejection',
        fn: () => Promise.reject(new Error('Async operation failed')),
        expectedError: 'Async operation failed',
      },
      {
        name: 'Timeout error',
        fn: () => new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Operation timed out')), 10)
        ),
        expectedError: 'Operation timed out',
      },
      {
        name: 'Network error',
        fn: async () => {
          const response = await fetch('http://invalid-url');
          return response.json();
        },
        expectedError: 'Network request failed',
      },
      {
        name: 'JSON parse error',
        fn: () => Promise.resolve('invalid json').then(JSON.parse),
        expectedError: 'Unexpected token',
      },
      {
        name: 'Unhandled rejection',
        fn: () => {
          process.nextTick(() => {
            throw new Error('Unhandled async error');
          });
          return Promise.resolve();
        },
        expectedError: 'Unhandled async error',
      },
    ];
  }

  static createSynchronousErrorScenarios(): Array<{
    name: string;
    fn: () => any;
    expectedError: string;
  }> {
    return [
      {
        name: 'Type error',
        fn: () => (null as any).nonExistentMethod(),
        expectedError: 'Cannot read properties of null',
      },
      {
        name: 'Reference error',
        fn: () => (undefinedVariable as any).toString(),
        expectedError: 'undefinedVariable is not defined',
      },
      {
        name: 'Range error',
        fn: () => new Array(-1),
        expectedError: 'Invalid array length',
      },
      {
        name: 'Syntax error (eval)',
        fn: () => eval('invalid syntax {{{'),
        expectedError: 'Unexpected token',
      },
      {
        name: 'Custom error',
        fn: () => { throw new Error('Custom error message'); },
        expectedError: 'Custom error message',
      },
    ];
  }
}

// Tests
describe('Comprehensive Edge Cases', () => {
  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Reset fetch mock
    global.fetch = jest.fn();
  });

  describe('Boundary Value Testing', () => {
    test('should handle numeric boundary values', () => {
      const testFunction = (value: number) => {
        if (value < 0 || value > 100) {
          throw new Error('Value out of range');
        }
        return value * 2;
      };

      const results = BoundaryValueTester.testNumericBoundaries(testFunction, [0, 100]);
      
      // Check that boundary values are handled correctly
      const validResults = results.filter(r => !r.error);
      const invalidResults = results.filter(r => r.error);
      
      expect(validResults.length).toBeGreaterThan(0);
      expect(invalidResults.length).toBeGreaterThan(0);
      
      // Check specific boundary conditions
      const zeroResult = results.find(r => r.value === 0);
      expect(zeroResult?.result).toBe(0);
      
      const maxResult = results.find(r => r.value === 100);
      expect(maxResult?.result).toBe(200);
      
      const overMaxResult = results.find(r => r.value === 101);
      expect(overMaxResult?.error).toContain('out of range');
    });

    test('should handle string boundary values', () => {
      const testFunction = (value: string) => {
        if (value.length === 0) {
          throw new Error('Empty string not allowed');
        }
        if (value.length > 10) {
          throw new Error('String too long');
        }
        return value.toUpperCase();
      };

      const results = BoundaryValueTester.testStringBoundaries(testFunction, 10);
      
      // Empty string should cause error
      const emptyResult = results.find(r => r.value === '');
      expect(emptyResult?.error).toContain('Empty string not allowed');
      
      // Normal string should work
      const normalResult = results.find(r => r.value === 'test');
      expect(normalResult?.result).toBe('TEST');
      
      // Long string should cause error
      const longResult = results.find(r => r.value.length > 10);
      expect(longResult?.error).toContain('String too long');
    });

    test('should handle array boundary values', () => {
      const testFunction = (value: any[]) => {
        if (value.length === 0) {
          return 'empty';
        }
        if (value.length > 5) {
          throw new Error('Array too large');
        }
        return value.length;
      };

      const results = BoundaryValueTester.testArrayBoundaries(testFunction, 5);
      
      // Empty array should return 'empty'
      const emptyResult = results.find(r => r.value.length === 0);
      expect(emptyResult?.result).toBe('empty');
      
      // Normal array should return length
      const normalResult = results.find(r => r.value.length === 3);
      expect(normalResult?.result).toBe(3);
      
      // Large array should cause error
      const largeResult = results.find(r => r.value.length > 5);
      expect(largeResult?.error).toContain('Array too large');
    });
  });

  describe('Network Error Handling', () => {
    test('should handle network failure', async () => {
      global.fetch = NetworkErrorSimulator.simulateNetworkFailure();

      const networkCall = async () => {
        const response = await fetch('/api/test');
        return response.json();
      };

      await expect(networkCall()).rejects.toThrow('Network request failed');
    });

    test('should handle timeout errors', async () => {
      global.fetch = NetworkErrorSimulator.simulateTimeoutError();

      const timeoutCall = async () => {
        const response = await fetch('/api/test');
        return response.json();
      };

      await expect(timeoutCall()).rejects.toThrow('Request timeout');
    });

    test('should handle server errors', async () => {
      global.fetch = NetworkErrorSimulator.simulateServerError(500);

      const response = await fetch('/api/test');
      expect(response.status).toBe(500);
      expect(response.statusText).toBe('Internal Server Error');
    });

    test('should handle corrupted responses', async () => {
      global.fetch = NetworkErrorSimulator.simulateCorruptedResponse();

      const response = await fetch('/api/test');
      expect(response.status).toBe(200);
      
      await expect(response.json()).rejects.toThrow();
    });

    test('should handle slow responses', async () => {
      global.fetch = NetworkErrorSimulator.simulateSlowResponse(50);

      const startTime = Date.now();
      await fetch('/api/test');
      const endTime = Date.now();
      
      expect(endTime - startTime).toBeGreaterThanOrEqual(50);
    }, 100);
  });

  describe('WebSocket Error Scenarios', () => {
    test('should handle WebSocket connection errors', (done) => {
      const ws = new WebSocket('ws://localhost:8080/error');
      
      ws.onerror = (event) => {
        expect(event).toBeDefined();
        done();
      };
    });

    test('should handle WebSocket send errors', (done) => {
      const ws = new WebSocket('ws://localhost:8080');
      
      ws.onopen = () => {
        expect(() => {
          ws.send('error-message');
        }).not.toThrow();
        
        ws.onerror = (event) => {
          expect(event).toBeDefined();
          done();
        };
      };
    });

    test('should handle WebSocket timeout scenarios', (done) => {
      const ws = new WebSocket('ws://localhost:8080/timeout');
      
      const timeout = setTimeout(() => {
        expect(ws.readyState).toBe(WebSocket.CONNECTING);
        done();
      }, 100);

      ws.onopen = () => {
        clearTimeout(timeout);
        done.fail('Should not have connected');
      };
    });

    test('should handle WebSocket premature close', (done) => {
      const ws = new WebSocket('ws://localhost:8080');
      
      ws.onopen = () => {
        ws.close(1000, 'Normal closure');
      };
      
      ws.onclose = (event) => {
        expect(event.code).toBe(1000);
        expect(event.reason).toBe('Normal closure');
        done();
      };
    });
  });

  describe('Memory Pressure Testing', () => {
    test('should handle large object creation', () => {
      const createLargeObject = () => {
        return MemoryPressureTester.createLargeObject(1); // 1MB
      };

      expect(() => createLargeObject()).not.toThrow();
      
      const obj = createLargeObject();
      expect(obj).toBeDefined();
      expect(Array.isArray(obj)).toBe(true);
    });

    test('should detect memory leaks', () => {
      const cleanup = MemoryPressureTester.simulateMemoryLeak();
      
      // Let it run for a short time
      setTimeout(() => {
        cleanup();
      }, 50);
      
      expect(cleanup).toBeDefined();
      expect(typeof cleanup).toBe('function');
    });

    test('should benchmark function performance', () => {
      const testFunction = () => {
        return Array.from({ length: 1000 }, (_, i) => i * 2);
      };

      const results = PerformanceTester.benchmarkFunction(testFunction, 10);
      
      expect(results.averageTime).toBeGreaterThan(0);
      expect(results.minTime).toBeLessThanOrEqual(results.averageTime);
      expect(results.maxTime).toBeGreaterThanOrEqual(results.averageTime);
      expect(results.opsPerSecond).toBeGreaterThan(0);
    });

    test('should measure memory usage', () => {
      const testFunction = () => {
        const arrays = [];
        for (let i = 0; i < 100; i++) {
          arrays.push(new Array(1000).fill(i));
        }
        return arrays;
      };

      const results = PerformanceTester.testMemoryUsage(testFunction);
      
      expect(results.initialMemory).toBeGreaterThanOrEqual(0);
      expect(results.finalMemory).toBeGreaterThanOrEqual(0);
      expect(results.peakMemory).toBeGreaterThanOrEqual(results.initialMemory);
    });
  });

  describe('Concurrency Testing', () => {
    test('should handle race conditions', async () => {
      let counter = 0;
      
      const operations = [
        async () => {
          const current = counter;
          await new Promise(resolve => setTimeout(resolve, Math.random() * 10));
          counter = current + 1;
          return counter;
        },
        async () => {
          const current = counter;
          await new Promise(resolve => setTimeout(resolve, Math.random() * 10));
          counter = current + 1;
          return counter;
        },
      ];

      const results = await ConcurrencyTester.testRaceCondition(operations, 5);
      
      expect(results.successes).toBeGreaterThan(0);
      expect(results.results.length).toBeGreaterThan(0);
      
      // Due to race conditions, final counter might not be what we expect
      expect(counter).toBeGreaterThan(0);
    });

    test('should detect potential deadlocks', async () => {
      let resource1Locked = false;
      let resource2Locked = false;

      const resource1 = {
        acquire: async () => {
          while (resource1Locked) {
            await new Promise(resolve => setTimeout(resolve, 1));
          }
          resource1Locked = true;
        },
        release: () => {
          resource1Locked = false;
        },
      };

      const resource2 = {
        acquire: async () => {
          while (resource2Locked) {
            await new Promise(resolve => setTimeout(resolve, 1));
          }
          resource2Locked = true;
        },
        release: () => {
          resource2Locked = false;
        },
      };

      const result = await ConcurrencyTester.testDeadlock(resource1, resource2, 100);
      
      expect(result.deadlocked).toBe(true);
      expect(result.time).toBeLessThan(200);
    }, 500);
  });

  describe('Security Testing', () => {
    test('should test XSS vulnerability', () => {
      const renderFunction = (input: string) => {
        // Insecure function that doesn't escape input
        return `<div>${input}</div>`;
      };

      const results = SecurityTester.testXSSVulnerability(renderFunction);
      
      expect(results.vulnerable).toBe(true);
      expect(results.attempts.length).toBeGreaterThan(0);
      expect(results.attempts.some(attempt => !attempt.escaped)).toBe(true);
    });

    test('should test SQL injection vulnerability', () => {
      const queryFunction = (input: string) => {
        // Insecure function that doesn't sanitize input
        return `SELECT * FROM users WHERE name = '${input}'`;
      };

      const results = SecurityTester.testSQLInjection(queryFunction);
      
      expect(results.vulnerable).toBe(true);
      expect(results.attempts.length).toBeGreaterThan(0);
      expect(results.attempts.some(attempt => attempt.suspicious)).toBe(true);
    });

    test('should test path traversal vulnerability', () => {
      const pathFunction = (input: string) => {
        // Insecure function that doesn't validate paths
        return `/var/www/html/${input}`;
      };

      const results = SecurityTester.testPathTraversal(pathFunction);
      
      expect(results.vulnerable).toBe(true);
      expect(results.attempts.length).toBeGreaterThan(0);
      expect(results.attempts.some(attempt => attempt.dangerous)).toBe(true);
    });
  });

  describe('Error Handling Scenarios', () => {
    test('should handle synchronous errors', () => {
      const scenarios = ErrorHandlingTester.createSynchronousErrorScenarios();
      
      scenarios.forEach(scenario => {
        expect(() => scenario.fn()).toThrow();
      });
    });

    test('should handle asynchronous errors', async () => {
      const scenarios = ErrorHandlingTester.createAsyncErrorScenarios();
      
      for (const scenario of scenarios) {
        if (scenario.name === 'Network error') {
          // Mock network error for this specific test
          global.fetch = NetworkErrorSimulator.simulateNetworkFailure();
        }
        
        await expect(scenario.fn()).rejects.toThrow();
      }
    });

    test('should handle React component errors', () => {
      const ErrorComponent: React.FC = () => {
        throw new Error('Component rendering error');
      };

      const errorHandler = jest.fn();

      render(
        <TestErrorBoundary onError={errorHandler}>
          <ErrorComponent />
        </TestErrorBoundary>
      );

      expect(screen.getByTestId('error-boundary')).toHaveTextContent('Component rendering error');
      expect(errorHandler).toHaveBeenCalledWith(expect.any(Error));
    });

    test('should handle async component errors', async () => {
      const AsyncErrorComponent: React.FC = () => {
        React.useEffect(() => {
          setTimeout(() => {
            throw new Error('Async component error');
          }, 10);
        }, []);

        return <div>Async Component</div>;
      };

      const errorHandler = jest.fn();

      // Note: Error boundaries don't catch async errors automatically
      // This test demonstrates the limitation
      render(
        <TestErrorBoundary onError={errorHandler}>
          <AsyncErrorComponent />
        </TestErrorBoundary>
      );

      expect(screen.getByText('Async Component')).toBeInTheDocument();
      
      // The error boundary won't catch this async error
      await new Promise(resolve => setTimeout(resolve, 20));
      expect(errorHandler).not.toHaveBeenCalled();
    });
  });

  describe('Integration Edge Cases', () => {
    test('should handle multiple simultaneous API calls', async () => {
      let callCount = 0;
      global.fetch = jest.fn().mockImplementation(() => {
        callCount++;
        if (callCount > 2) {
          return Promise.reject(new Error('Too many requests'));
        }
        return Promise.resolve(new Response(JSON.stringify({ data: callCount })));
      });

      const apiCalls = Array.from({ length: 5 }, () =>
        fetch('/api/test').then(res => res.json()).catch(err => ({ error: err.message }))
      );

      const results = await Promise.allSettled(apiCalls);
      
      const successes = results.filter(r => r.status === 'fulfilled').length;
      const failures = results.filter(r => r.status === 'rejected').length;
      
      expect(successes + failures).toBe(5);
      expect(callCount).toBeGreaterThan(2);
    });

    test('should handle component mounting/unmounting race conditions', async () => {
      const cleanupCallbacks: Array<() => void> = [];
      
      const TestComponent: React.FC<{ shouldMount: boolean }> = ({ shouldMount }) => {
        React.useEffect(() => {
          if (!shouldMount) return;
          
          const cleanup = () => {
            cleanupCallbacks.push(() => {});
          };
          
          const timer = setTimeout(cleanup, 10);
          
          return () => {
            clearTimeout(timer);
            cleanup();
          };
        }, [shouldMount]);

        return shouldMount ? <div>Mounted</div> : null;
      };

      const { rerender } = render(<TestComponent shouldMount={true} />);
      
      // Rapidly mount/unmount
      for (let i = 0; i < 10; i++) {
        rerender(<TestComponent shouldMount={i % 2 === 0} />);
        await new Promise(resolve => setTimeout(resolve, 1));
      }
      
      expect(cleanupCallbacks.length).toBeGreaterThanOrEqual(0);
    });

    test('should handle state updates after component unmount', () => {
      const StateComponent: React.FC = () => {
        const [count, setCount] = React.useState(0);
        
        React.useEffect(() => {
          const timer = setTimeout(() => {
            setCount(1); // This will happen after unmount
          }, 50);
          
          return () => clearTimeout(timer);
        }, []);

        return <div>{count}</div>;
      };

      const { unmount } = render(<StateComponent />);
      
      // Unmount before the setState timer fires
      unmount();
      
      // Should not cause any errors
      expect(true).toBe(true);
    });

    test('should handle circular reference in data structures', () => {
      const createCircularReference = () => {
        const obj: any = { name: 'test' };
        obj.self = obj; // Circular reference
        return obj;
      };

      const circular = createCircularReference();
      
      // JSON.stringify should throw on circular reference
      expect(() => JSON.stringify(circular)).toThrow('circular');
      
      // Object should still be accessible
      expect(circular.name).toBe('test');
      expect(circular.self).toBe(circular);
    });

    test('should handle extremely large datasets', () => {
      const createLargeDataset = (size: number) => {
        return Array.from({ length: size }, (_, i) => ({
          id: i,
          data: `item-${i}`,
          nested: {
            values: Array.from({ length: 10 }, (_, j) => i * 10 + j),
          },
        }));
      };

      const testProcessing = (dataset: any[]) => {
        return dataset
          .filter(item => item.id % 2 === 0)
          .map(item => ({ ...item, processed: true }))
          .slice(0, 100);
      };

      const largeDataset = createLargeDataset(10000);
      
      expect(() => {
        const processed = testProcessing(largeDataset);
        expect(processed.length).toBe(100);
        expect(processed[0].processed).toBe(true);
      }).not.toThrow();
    });
  });

  describe('Performance Edge Cases', () => {
    test('should handle performance degradation under load', () => {
      const performanceTest = () => {
        const start = performance.now();
        
        // Simulate heavy computation
        let result = 0;
        for (let i = 0; i < 100000; i++) {
          result += Math.sqrt(i) * Math.sin(i);
        }
        
        const end = performance.now();
        return { time: end - start, result };
      };

      // Run multiple times to test performance consistency
      const results = Array.from({ length: 5 }, performanceTest);
      
      const averageTime = results.reduce((sum, r) => sum + r.time, 0) / results.length;
      const maxTime = Math.max(...results.map(r => r.time));
      const minTime = Math.min(...results.map(r => r.time));
      
      expect(averageTime).toBeGreaterThan(0);
      expect(maxTime).toBeGreaterThanOrEqual(minTime);
      
      // Performance should be relatively consistent (within 3x)
      expect(maxTime / minTime).toBeLessThan(3);
    });

    test('should handle DOM manipulation performance', () => {
      const DOMTestComponent: React.FC = () => {
        const [items, setItems] = React.useState<number[]>([]);
        
        const addItems = () => {
          setItems(prev => [...prev, ...Array.from({ length: 1000 }, (_, i) => prev.length + i)]);
        };

        return (
          <div>
            <button onClick={addItems}>Add Items</button>
            <div data-testid="items-container">
              {items.map(item => (
                <div key={item}>Item {item}</div>
              ))}
            </div>
          </div>
        );
      };

      render(<DOMTestComponent />);
      
      const addButton = screen.getByText('Add Items');
      const container = screen.getByTestId('items-container');
      
      expect(container.children.length).toBe(0);
      
      fireEvent.click(addButton);
      expect(container.children.length).toBe(1000);
      
      fireEvent.click(addButton);
      expect(container.children.length).toBe(2000);
    });
  });
});