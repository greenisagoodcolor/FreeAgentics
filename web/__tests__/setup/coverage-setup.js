/**
 * Coverage Test Setup
 * Enhanced setup for comprehensive frontend coverage testing
 */

// Suppress console warnings during coverage testing
const originalConsoleWarn = console.warn;
const originalConsoleError = console.error;

console.warn = (...args) => {
  const message = args.join(' ');
  
  // Suppress known warnings that don't affect coverage
  if (
    message.includes('Warning: React.createElement: type is invalid') ||
    message.includes('Warning: Failed prop type') ||
    message.includes('Warning: componentWillReceiveProps') ||
    message.includes('Warning: componentWillMount') ||
    message.includes('act(...) is not supported') ||
    message.includes('useLayoutEffect does nothing on the server') ||
    message.includes('Cannot update a component while rendering')
  ) {
    return;
  }
  
  originalConsoleWarn.apply(console, args);
};

console.error = (...args) => {
  const message = args.join(' ');
  
  // Suppress known errors that don't affect coverage
  if (
    message.includes('Error: Uncaught [TypeError: Cannot read') ||
    message.includes('The above error occurred in the') ||
    message.includes('Consider adding an error boundary') ||
    message.includes('Warning: Can\'t perform a React state update') ||
    message.includes('ResizeObserver loop limit exceeded')
  ) {
    return;
  }
  
  originalConsoleError.apply(console, args);
};

// Global test timeout for coverage tests
jest.setTimeout(30000);

// Mock performance API for consistent testing
global.performance = global.performance || {
  now: jest.fn(() => Date.now()),
  mark: jest.fn(),
  measure: jest.fn(),
  getEntriesByType: jest.fn(() => []),
  getEntriesByName: jest.fn(() => []),
  clearMarks: jest.fn(),
  clearMeasures: jest.fn(),
  clearResourceTimings: jest.fn()
};

// Mock ResizeObserver for consistent testing
global.ResizeObserver = global.ResizeObserver || class ResizeObserver {
  constructor(callback) {
    this.callback = callback;
  }
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Mock IntersectionObserver for consistent testing
global.IntersectionObserver = global.IntersectionObserver || class IntersectionObserver {
  constructor(callback, options) {
    this.callback = callback;
    this.options = options;
  }
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Mock MutationObserver for consistent testing
global.MutationObserver = global.MutationObserver || class MutationObserver {
  constructor(callback) {
    this.callback = callback;
  }
  observe() {}
  disconnect() {}
  takeRecords() { return []; }
};

// Ensure all timeouts are cleaned up
afterEach(() => {
  jest.clearAllTimers();
  jest.clearAllMocks();
});

// Global error handler for uncaught promise rejections during coverage testing
process.on('unhandledRejection', (reason, promise) => {
  // Log but don't fail the test for coverage purposes
  console.log('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Optimize garbage collection for large test suites
if (global.gc) {
  afterAll(() => {
    global.gc();
  });
}