#!/bin/bash
# Fixed test environment setup with proper coordination
# This script ensures ALL test outputs go to timestamped directories

set -e

# Colors for output
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

# Get or create consistent timestamp
if [ -z "$TEST_TIMESTAMP" ]; then
    export TEST_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
fi

if [ -z "$TEST_REPORT_DIR" ]; then
    export TEST_REPORT_DIR="tests/reports/$TEST_TIMESTAMP"
fi

echo -e "${CYAN}Setting up test environment with timestamp: $TEST_TIMESTAMP${NC}"
echo -e "${CYAN}Report directory: $TEST_REPORT_DIR${NC}"

# Clean up any existing artifacts first
./scripts/cleanup-test-artifacts.sh

# Create comprehensive directory structure
mkdir -p "$TEST_REPORT_DIR"/{python,frontend,quality,security,integration,performance,visual}
mkdir -p "$TEST_REPORT_DIR"/python/{coverage,pytest,mypy}
mkdir -p "$TEST_REPORT_DIR"/frontend/{jest,playwright,coverage}
mkdir -p "$TEST_REPORT_DIR"/quality/{flake8,eslint,black,prettier,mypy-html,mypy-reports}
mkdir -p "$TEST_REPORT_DIR"/security/{bandit,safety,audit}

# Force remove any existing root-level artifacts
rm -rf htmlcov coverage_core_ai mypy_reports .coverage coverage.xml coverage.json
rm -rf test-results playwright-report
rm -rf web/test-results web/playwright-report

# Set comprehensive environment variables for all tools
export COVERAGE_FILE="$PWD/$TEST_REPORT_DIR/python/coverage/.coverage"
export COVERAGE_HTML_DIR="$PWD/$TEST_REPORT_DIR/python/coverage/html"
export COVERAGE_XML_FILE="$PWD/$TEST_REPORT_DIR/python/coverage/coverage.xml"
export COVERAGE_JSON_FILE="$PWD/$TEST_REPORT_DIR/python/coverage/coverage.json"

# MyPy
export MYPY_CACHE_DIR="$PWD/$TEST_REPORT_DIR/python/mypy/.mypy_cache"

# Pytest
export PYTEST_CACHE_DIR="$PWD/$TEST_REPORT_DIR/python/pytest/.pytest_cache"

# Jest/Frontend
export JEST_COVERAGE_DIR="$PWD/$TEST_REPORT_DIR/frontend/jest/coverage"

# Playwright
export PLAYWRIGHT_HTML_REPORT="$PWD/$TEST_REPORT_DIR/frontend/playwright/report"
export PLAYWRIGHT_TEST_OUTPUT_DIR="$PWD/$TEST_REPORT_DIR/frontend/playwright/test-results"
export PLAYWRIGHT_JSON_OUTPUT="$PWD/$TEST_REPORT_DIR/frontend/playwright/results.json"
export PLAYWRIGHT_JUNIT_OUTPUT="$PWD/$TEST_REPORT_DIR/frontend/playwright/results.xml"

# Create custom configuration files that override defaults
# Custom pytest.ini for this test run
cat > "$TEST_REPORT_DIR/pytest.ini" << EOF
[tool:pytest]
cache_dir = $PWD/$TEST_REPORT_DIR/python/pytest/.pytest_cache
EOF

# Custom .coveragerc that overrides the default
cat > "$TEST_REPORT_DIR/.coveragerc" << EOF
[run]
branch = True
source = agents,inference,coalitions,world,api,infrastructure
data_file = $PWD/$TEST_REPORT_DIR/python/coverage/.coverage
omit =
    */tests/*
    */test_*.py
    */__pycache__/*
    */venv/*
    */env/*
    */.venv/*
    */site-packages/*
    */migrations/*
    */alembic/*
    */manage.py
    */setup.py
    */conftest.py

[report]
precision = 2
show_missing = True
skip_covered = False

[html]
directory = $PWD/$TEST_REPORT_DIR/python/coverage/html

[xml]
output = $PWD/$TEST_REPORT_DIR/python/coverage/coverage.xml

[json]
output = $PWD/$TEST_REPORT_DIR/python/coverage/coverage.json
EOF

# Export the custom config file path
export COVERAGE_RC_FILE="$PWD/$TEST_REPORT_DIR/.coveragerc"

# Create Jest config override for web directory
cat > "web/jest.config.override.js" << EOF
const baseConfig = require('./jest.config.js');

module.exports = {
  ...baseConfig,
  coverageDirectory: '$PWD/$TEST_REPORT_DIR/frontend/jest/coverage',
  cacheDirectory: '$PWD/$TEST_REPORT_DIR/frontend/jest/.jest_cache',
};
EOF

# Create custom Playwright config for this run
cat > "web/playwright.config.override.ts" << EOF
import { defineConfig } from '@playwright/test';
import baseConfig from './playwright.config';

export default defineConfig({
  ...baseConfig,
  outputDir: '$PWD/$TEST_REPORT_DIR/frontend/playwright/test-results',
  reporter: [
    ['html', { outputFolder: '$PWD/$TEST_REPORT_DIR/frontend/playwright/report' }],
    ['json', { outputFile: '$PWD/$TEST_REPORT_DIR/frontend/playwright/results.json' }],
    ['junit', { outputFile: '$PWD/$TEST_REPORT_DIR/frontend/playwright/results.xml' }],
  ],
});
EOF

# Create summary file
cat > "$TEST_REPORT_DIR/test-summary.md" << EOF
# Test Report - $TEST_TIMESTAMP

Generated: $(date)

## Test Execution Summary

EOF

# Create symlink for latest
rm -f tests/reports/latest
ln -sfn "$TEST_TIMESTAMP" tests/reports/latest

# Export all paths for child processes
export TEST_PYTEST_CONFIG="$PWD/$TEST_REPORT_DIR/pytest.ini"
export TEST_JEST_CONFIG="$PWD/web/jest.config.override.js"
export TEST_PLAYWRIGHT_CONFIG="$PWD/web/playwright.config.override.ts"

echo -e "${GREEN}✅ Test environment configured successfully${NC}"
echo -e "${GREEN}✅ All outputs will go to: $TEST_REPORT_DIR${NC}"

# Write environment info for debugging
cat > "$TEST_REPORT_DIR/environment.sh" << EOF
# Environment variables set by test setup
export TEST_TIMESTAMP="$TEST_TIMESTAMP"
export TEST_REPORT_DIR="$TEST_REPORT_DIR"
export COVERAGE_FILE="$COVERAGE_FILE"
export COVERAGE_RC_FILE="$COVERAGE_RC_FILE"
export MYPY_CACHE_DIR="$MYPY_CACHE_DIR"
export JEST_COVERAGE_DIR="$JEST_COVERAGE_DIR"
export PLAYWRIGHT_HTML_REPORT="$PLAYWRIGHT_HTML_REPORT"
export PLAYWRIGHT_TEST_OUTPUT_DIR="$PLAYWRIGHT_TEST_OUTPUT_DIR"
export TEST_JEST_CONFIG="$TEST_JEST_CONFIG"
export TEST_PLAYWRIGHT_CONFIG="$TEST_PLAYWRIGHT_CONFIG"
EOF