#!/bin/bash
# Setup test environment with proper output directories
# This script ensures all test outputs go to timestamped directories

set -e

# Get timestamp from environment or generate new one
TIMESTAMP=${TEST_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
REPORT_DIR=${TEST_REPORT_DIR:-"tests/reports/$TIMESTAMP"}

# Export for child processes
export TEST_TIMESTAMP=$TIMESTAMP
export TEST_REPORT_DIR=$REPORT_DIR

# Create comprehensive directory structure
echo "Setting up test report directories at: $REPORT_DIR"
mkdir -p "$REPORT_DIR"/{python,frontend,quality,security,integration,performance,visual}
mkdir -p "$REPORT_DIR"/python/{coverage,pytest,mypy}
mkdir -p "$REPORT_DIR"/frontend/{jest,playwright,coverage}
mkdir -p "$REPORT_DIR"/quality/{flake8,eslint,black,prettier}
mkdir -p "$REPORT_DIR"/security/{bandit,safety,audit}

# Set environment variables for various test tools
# Python coverage
export COVERAGE_FILE="$REPORT_DIR/python/coverage/.coverage"
export COVERAGE_HTML_DIR="$REPORT_DIR/python/coverage/html"
export COVERAGE_XML_FILE="$REPORT_DIR/python/coverage/coverage.xml"
export COVERAGE_JSON_FILE="$REPORT_DIR/python/coverage/coverage.json"

# MyPy
export MYPY_CACHE_DIR="$REPORT_DIR/python/mypy/.mypy_cache"

# Pytest
export PYTEST_CACHE_DIR="$REPORT_DIR/python/pytest/.pytest_cache"

# Jest
export JEST_COVERAGE_DIR="$REPORT_DIR/frontend/jest/coverage"

# Playwright
export PLAYWRIGHT_HTML_REPORT="$REPORT_DIR/frontend/playwright/report"
export PLAYWRIGHT_TEST_OUTPUT_DIR="$REPORT_DIR/frontend/playwright/test-results"
export PLAYWRIGHT_JSON_OUTPUT="$REPORT_DIR/frontend/playwright/results.json"
export PLAYWRIGHT_JUNIT_OUTPUT="$REPORT_DIR/frontend/playwright/results.xml"

# Create symbolic links for legacy paths (backward compatibility)
# This ensures tools that expect certain paths still work
if [ ! -e "htmlcov" ]; then
    ln -sf "$REPORT_DIR/python/coverage/html" htmlcov
fi

if [ ! -e "coverage_core_ai" ]; then
    ln -sf "$REPORT_DIR/python/coverage/html" coverage_core_ai
fi

# Create .coveragerc override if needed
if [ -f "config/.coveragerc" ]; then
    cp config/.coveragerc "$REPORT_DIR/.coveragerc"
    # Update paths in the copy
    sed -i.bak "s|directory = coverage/html|directory = $REPORT_DIR/python/coverage/html|g" "$REPORT_DIR/.coveragerc"
    sed -i.bak "s|output = coverage/coverage.xml|output = $REPORT_DIR/python/coverage/coverage.xml|g" "$REPORT_DIR/.coveragerc"
    sed -i.bak "s|output = coverage/coverage.json|output = $REPORT_DIR/python/coverage/coverage.json|g" "$REPORT_DIR/.coveragerc"
    export COVERAGE_RC_FILE="$REPORT_DIR/.coveragerc"
fi

# Create summary file
cat > "$REPORT_DIR/README.md" << EOF
# Test Report - $TIMESTAMP

Generated: $(date)

## Directory Structure

- **python/** - Python test outputs
  - **coverage/** - Coverage reports (HTML, XML, JSON)
  - **pytest/** - Pytest outputs and cache
  - **mypy/** - Type checking reports
- **frontend/** - Frontend test outputs
  - **jest/** - Jest test results and coverage
  - **playwright/** - E2E test results
  - **coverage/** - Frontend coverage reports
- **quality/** - Code quality reports
  - **flake8/** - Python linting
  - **eslint/** - JavaScript linting
  - **black/** - Python formatting
  - **prettier/** - JavaScript formatting
- **security/** - Security analysis
  - **bandit/** - Python security
  - **safety/** - Dependency vulnerabilities
  - **audit/** - npm audit results

## Environment Variables Set

- COVERAGE_FILE=$COVERAGE_FILE
- MYPY_CACHE_DIR=$MYPY_CACHE_DIR
- JEST_COVERAGE_DIR=$JEST_COVERAGE_DIR
- PLAYWRIGHT_HTML_REPORT=$PLAYWRIGHT_HTML_REPORT
EOF

echo "Test environment configured for timestamp: $TIMESTAMP"
echo "Report directory: $REPORT_DIR"