const config = require('./jest.config');

module.exports = {
  ...config,
  // Optimized settings for comprehensive coverage testing
  testTimeout: 30000,
  maxWorkers: 2,
  silent: true,
  verbose: false,
  
  // Enhanced coverage settings for maximum precision
  collectCoverage: true,
  coverageReporters: [
    'text',
    'text-summary',
    'html',
    'lcov',
    'clover',
    'json',
    'json-summary'
  ],
  
  // Comprehensive coverage collection
  collectCoverageFrom: [
    'app/**/*.{ts,tsx}',
    'components/**/*.{ts,tsx}',
    'lib/**/*.{ts,tsx}',
    'hooks/**/*.{ts,tsx}',
    'contexts/**/*.{ts,tsx}',
    '!**/*.d.ts',
    '!**/node_modules/**',
    '!**/__tests__/**',
    '!**/test-utils/**',
    '!**/*.test.{ts,tsx}',
    '!**/*.spec.{ts,tsx}',
    '!**/coverage/**',
    '!components/ui/**', // Exclude shadcn/ui components as per original config
  ],
  
  // Enhanced coverage thresholds for comprehensive testing
  coverageThreshold: {
    global: {
      statements: 80,
      branches: 70,
      functions: 75,
      lines: 80
    },
    './lib/': {
      statements: 85,
      branches: 75,
      functions: 80,
      lines: 85
    },
    './components/': {
      statements: 75,
      branches: 65,
      functions: 70,
      lines: 75
    },
    './hooks/': {
      statements: 80,
      branches: 70,
      functions: 75,
      lines: 80
    },
    './contexts/': {
      statements: 85,
      branches: 75,
      functions: 80,
      lines: 85
    }
  },
  
  // Test patterns specifically for comprehensive coverage
  testMatch: [
    '<rootDir>/__tests__/**/*.{test,spec}.{ts,tsx}',
    '<rootDir>/**/__tests__/**/*.{test,spec}.{ts,tsx}',
    '<rootDir>/**/(*.)+(spec|test).{ts,tsx}'
  ],
  
  // Performance optimizations for large test suites
  resetMocks: true,
  restoreMocks: true,
  clearMocks: true,
  
  // Enhanced setup for comprehensive testing
  setupFilesAfterEnv: [
    '<rootDir>/jest.setup.js',
    '<rootDir>/__tests__/setup/coverage-setup.js'
  ],
  
  // Memory management for large test suites
  workerIdleMemoryLimit: '512MB',
  
  // Enhanced error reporting
  errorOnDeprecated: false,
  bail: false,
  
  // Optimize for CI/CD environments
  detectOpenHandles: true,
  forceExit: true,
  
  // Cache settings for performance
  cache: true,
  cacheDirectory: '<rootDir>/node_modules/.cache/jest-coverage'
};