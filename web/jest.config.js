const nextJest = require("next/jest");

const createJestConfig = nextJest({
  // Provide the path to your Next.js app to load next.config.js and .env files
  dir: "./",
});

// Add any custom config to be passed to Jest
const customJestConfig = {
  // Add more setup options before each test is run
  setupFilesAfterEnv: ["<rootDir>/jest.setup.js"],

  // Test environment
  testEnvironment: "jest-environment-jsdom",

  // Module name mapping for absolute imports
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/$1",
    "^@/components/(.*)$": "<rootDir>/components/$1",
    "^@/lib/(.*)$": "<rootDir>/lib/$1",
    "^@/hooks/(.*)$": "<rootDir>/hooks/$1",
    "^@/contexts/(.*)$": "<rootDir>/contexts/$1",
    "^@/types/(.*)$": "<rootDir>/types/$1",
    // Mock lodash-es to use lodash instead
    "^lodash-es$": "lodash",
    "^lodash-es/(.*)$": "lodash/$1",
    // Mock nanoid for Jest compatibility
    "^nanoid$": "<rootDir>/__mocks__/nanoid.js",
  },

  // Coverage configuration
  collectCoverageFrom: [
    "components/**/*.{ts,tsx}",
    "lib/**/*.{ts,tsx}",
    "hooks/**/*.{ts,tsx}",
    "contexts/**/*.{ts,tsx}",
    "app/**/*.{ts,tsx}",
    "!**/*.d.ts",
    "!**/node_modules/**",
    "!**/.next/**",
    "!**/coverage/**",
    "!components/ui/**", // UI components have complex dependencies
  ],

  // Coverage thresholds
  coverageThreshold: {
    global: {
      branches: 50,
      functions: 50,
      lines: 50,
      statements: 50,
    },
  },

  // Test patterns - exclude E2E tests and setup files
  testMatch: [
    "**/__tests__/**/*.(test|spec).(ts|tsx|js)",
    "**/*.(test|spec).(ts|tsx|js)",
  ],

  // Ignore patterns - exclude E2E tests, setup files only
  testPathIgnorePatterns: [
    "<rootDir>/.next/",
    "<rootDir>/node_modules/",
    "<rootDir>/e2e/",
    "<rootDir>/test-results/",
    "<rootDir>/__tests__/setup/", // Exclude setup files
    "<rootDir>/__tests__/test-helpers/", // Exclude helper files
  ],

  // Transform patterns - handle ESM modules properly
  transformIgnorePatterns: [
    "node_modules/(?!(lodash-es|nanoid|uuid|@?[^/]+/.*\\.(m)?js$))",
  ],

  // Transform patterns
  transform: {
    "^.+\\.(ts|tsx)$": "ts-jest",
  },

  // Module file extensions
  moduleFileExtensions: ["ts", "tsx", "js", "jsx"],

  // Setup for canvas and other browser APIs
  testEnvironmentOptions: {
    customExportConditions: [""],
  },

  // Verbose output
  verbose: true,

  // Performance optimizations from comprehensive testing strategy
  testTimeout: 10000, // Increased timeout for complex component tests (10 seconds)
  maxWorkers: 1, // Force single worker to prevent resource contention and hanging

  // Cache configuration for faster subsequent runs
  cacheDirectory: "<rootDir>/.jest-cache",

  // Optimize test execution
  passWithNoTests: true,

  // Force exit and detect open handles to prevent hanging
  forceExit: true,
  detectOpenHandles: true,

  // Reduce memory usage
  logHeapUsage: true,
};

// createJestConfig is exported this way to ensure that next/jest can load the Next.js config which is async
module.exports = createJestConfig(customJestConfig);
