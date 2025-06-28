#!/bin/bash
# Cleanup script to remove test artifacts from inappropriate locations
# This ensures a clean slate before running tests

set -e

echo "🧹 Cleaning up test artifacts from root and web directories..."

# Remove root-level test artifacts
rm -rf htmlcov coverage_core_ai mypy_reports
rm -rf test-results playwright-report .test-reports
rm -f .coverage coverage.xml coverage.json
rm -f .pre-commit-bandit-report.json

# Remove web directory test artifacts
rm -rf web/test-results web/playwright-report web/coverage
rm -f web/dashboard-debug.png

# Remove any Jest/Playwright config overrides from previous runs
rm -f web/jest.config.override.js
rm -f web/playwright.config.override.ts

# Remove pytest cache directories that might be at root
find . -maxdepth 2 -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
find . -maxdepth 2 -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null || true

echo "✅ Cleanup complete"