const nextJest = require("next/jest");

const createJestConfig = nextJest({
  // Provide the path to your Next.js app to load next.config.js and .env files
  dir: "./",
});

// Add any custom config to be passed to Jest
const customJestConfig = {
  rootDir: ".",
  setupFilesAfterEnv: ["<rootDir>/jest.setup.js"],
  testEnvironment: "jest-environment-jsdom",
  testPathIgnorePatterns: [
    "<rootDir>/node_modules/",
    "<rootDir>/__tests__/test-utils.tsx", // Ignore test utility files
    "<rootDir>/__tests__/__mocks__/fileMock.js", // Ignore mock file but allow test
  ],
  reporters: [
    "default",
    // Only add jest-junit reporter if running in CI
    ...(process.env.CI
      ? [["jest-junit", { outputDirectory: "../test-reports", outputName: "junit.xml" }]]
      : []),
  ],
  collectCoverageFrom: [
    "components/**/*.{js,jsx,ts,tsx}",
    "lib/**/*.{js,jsx,ts,tsx}",
    "hooks/**/*.{js,jsx,ts,tsx}",
    "utils/**/*.{js,jsx,ts,tsx}",
    "app/**/*.{js,jsx,ts,tsx}",
    "!**/*.d.ts",
    "!**/node_modules/**",
  ],
  coverageThreshold: {
    global: {
      branches: 30,
      functions: 29,
      lines: 35,
      statements: 34,
    },
  },
};

// Export the config with a custom moduleNameMapper that will be merged with Next.js defaults
module.exports = async () => {
  const jestConfig = await createJestConfig(customJestConfig)();

  // Ensure our custom moduleNameMapper takes precedence and is correctly configured
  jestConfig.moduleNameMapper = {
    // First, handle CSS and static files (Next.js defaults)
    "^.+\\.(css|sass|scss)$": "identity-obj-proxy",
    "^.+\\.(png|jpg|jpeg|gif|webp|avif|ico|bmp|svg)$": `__tests__/__mocks__/fileMock.js`,

    // Then our custom path mappings - be explicit about rootDir resolution
    "^@/components/(.*)$": "<rootDir>/components/$1",
    "^@/lib/(.*)$": "<rootDir>/lib/$1",
    "^@/hooks/(.*)$": "<rootDir>/hooks/$1",
    "^@/utils/(.*)$": "<rootDir>/utils/$1",
    "^@/types/(.*)$": "<rootDir>/types/$1",
    "^@/(.*)$": "<rootDir>/$1", // Catch-all for any @/ imports

    // Keep any existing Next.js moduleNameMapper entries that don't conflict
    ...Object.fromEntries(
      Object.entries(jestConfig.moduleNameMapper || {}).filter(([key]) => !key.startsWith("^@/")),
    ),
  };

  // Add additional CI-specific configuration
  if (process.env.CI) {
    jestConfig.verbose = true;
    jestConfig.forceExit = true;
    jestConfig.detectOpenHandles = true;
    jestConfig.clearMocks = true;
    // Add cache directory clearing for CI
    jestConfig.cacheDirectory = "/tmp/jest-cache";
  }

  // Debug: Log configuration in CI
  if (process.env.CI) {
    console.log("Jest moduleNameMapper:", JSON.stringify(jestConfig.moduleNameMapper, null, 2));
    console.log("Jest rootDir:", jestConfig.rootDir);
  }

  return jestConfig;
};
