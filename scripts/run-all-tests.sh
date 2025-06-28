#!/bin/bash
# Comprehensive Test Runner with Timestamped Reports
# Executes all tests and saves results to tests/reports with timestamps

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_DIR="tests/reports/${TIMESTAMP}"

# Export for child processes
export TEST_TIMESTAMP=$TIMESTAMP
export TEST_REPORT_DIR=$REPORT_DIR

# Setup test environment with proper paths
echo -e "${BLUE}Setting up test environment...${NC}"
./scripts/setup-test-environment.sh

# Summary file
SUMMARY_FILE="${REPORT_DIR}/test-summary.md"

# Initialize summary
cat > "${SUMMARY_FILE}" << EOF
# Test Execution Report
**Timestamp**: ${TIMESTAMP}
**Date**: $(date)
**Repository**: FreeAgentics

## Test Results Summary

EOF

# Function to run command and capture output
run_test() {
    local test_name=$1
    local command=$2
    local output_file=$3
    local category=$4
    
    echo -e "${CYAN}Running ${test_name}...${NC}"
    echo "### ${test_name}" >> "${SUMMARY_FILE}"
    echo "**Category**: ${category}" >> "${SUMMARY_FILE}"
    echo "**Start Time**: $(date +"%H:%M:%S")" >> "${SUMMARY_FILE}"
    
    # Run the command
    if eval "${command}" > "${output_file}" 2>&1; then
        echo -e "${GREEN}✓ ${test_name} passed${NC}"
        echo "**Status**: ✅ PASSED" >> "${SUMMARY_FILE}"
    else
        echo -e "${RED}✗ ${test_name} failed${NC}"
        echo "**Status**: ❌ FAILED" >> "${SUMMARY_FILE}"
        echo "See: ${output_file}" >> "${SUMMARY_FILE}"
    fi
    
    echo "**End Time**: $(date +"%H:%M:%S")" >> "${SUMMARY_FILE}"
    echo "" >> "${SUMMARY_FILE}"
}

# Python environment setup
if [ -d "venv" ]; then
    echo -e "${BLUE}Activating Python virtual environment...${NC}"
    source venv/bin/activate
fi

echo -e "${BLUE}Starting comprehensive test suite at ${TIMESTAMP}${NC}"
echo ""

# 1. Static Analysis & Type Checking
echo -e "${YELLOW}Phase 1: Static Analysis & Type Checking${NC}"
echo "## Phase 1: Static Analysis & Type Checking" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

run_test "Python Type Checking (MyPy)" \
    "mypy --config-file=pyproject.toml agents inference coalitions world api infrastructure --html-report=${REPORT_DIR}/quality/mypy-html" \
    "${REPORT_DIR}/quality/mypy-output.log" \
    "Type Safety"

run_test "TypeScript Type Checking" \
    "cd web && npx tsc --noEmit --pretty" \
    "${REPORT_DIR}/quality/typescript-output.log" \
    "Type Safety"

# 2. Code Quality & Linting
echo -e "${YELLOW}Phase 2: Code Quality & Linting${NC}"
echo "## Phase 2: Code Quality & Linting" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

run_test "Python Linting (Flake8)" \
    "flake8 -vv --show-source --statistics --count --benchmark --format='%(path)s:%(row)d:%(col)d: [%(code)s] %(text)s' --output-file=${REPORT_DIR}/quality/flake8-report.txt agents inference coalitions world api infrastructure" \
    "${REPORT_DIR}/quality/flake8-output.log" \
    "Code Quality"

run_test "Python Formatting Check (Black)" \
    "black --check --line-length=100 agents inference coalitions world api infrastructure" \
    "${REPORT_DIR}/quality/black-output.log" \
    "Code Quality"

run_test "JavaScript/TypeScript Linting" \
    "cd web && npm run lint" \
    "${REPORT_DIR}/quality/eslint-output.log" \
    "Code Quality"

# 3. Security Analysis
echo -e "${YELLOW}Phase 3: Security Analysis${NC}"
echo "## Phase 3: Security Analysis" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

run_test "Python Security Scan (Bandit)" \
    "bandit -r agents inference coalitions world api infrastructure -f json -o ${REPORT_DIR}/security/bandit-report.json" \
    "${REPORT_DIR}/security/bandit-output.log" \
    "Security"

run_test "Dependency Vulnerability Check" \
    "safety check --json --output ${REPORT_DIR}/security/safety-report.json" \
    "${REPORT_DIR}/security/safety-output.log" \
    "Security"

# 4. Python Unit Tests
echo -e "${YELLOW}Phase 4: Python Unit Tests${NC}"
echo "## Phase 4: Python Unit Tests" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

run_test "Python Unit Tests" \
    "COVERAGE_FILE=${COVERAGE_FILE} python -m pytest tests/unit/ -vvv --tb=long --cov=agents --cov=inference --cov=coalitions --cov=world --cov-report=html:${COVERAGE_HTML_DIR} --cov-report=json:${COVERAGE_JSON_FILE} --cov-report=xml:${COVERAGE_XML_FILE} --cov-report=term --junitxml=${REPORT_DIR}/python/unit-tests.xml" \
    "${REPORT_DIR}/python/unit-tests.log" \
    "Unit Tests"

# 5. Python Integration Tests
echo -e "${YELLOW}Phase 5: Python Integration Tests${NC}"
echo "## Phase 5: Python Integration Tests" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

run_test "Python Integration Tests" \
    "python -m pytest tests/integration/ -vvv --tb=long --junitxml=${REPORT_DIR}/python/integration-tests.xml" \
    "${REPORT_DIR}/python/integration-tests.log" \
    "Integration Tests"

# 6. Python Advanced Tests
echo -e "${YELLOW}Phase 6: Python Advanced Tests${NC}"
echo "## Phase 6: Python Advanced Tests" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

run_test "Property-Based Tests" \
    "python -m pytest tests/property/ -vvv --tb=long --junitxml=${REPORT_DIR}/python/property-tests.xml" \
    "${REPORT_DIR}/python/property-tests.log" \
    "Mathematical Invariants"

run_test "Behavior-Driven Tests" \
    "python -m pytest tests/behavior/ -vvv --tb=long --junitxml=${REPORT_DIR}/python/behavior-tests.xml" \
    "${REPORT_DIR}/python/behavior-tests.log" \
    "BDD Scenarios"

run_test "Security Tests" \
    "python -m pytest tests/security/ -vvv --tb=long --junitxml=${REPORT_DIR}/python/security-tests.xml" \
    "${REPORT_DIR}/python/security-tests.log" \
    "Security Testing"

run_test "Chaos Engineering Tests" \
    "python -m pytest tests/chaos/ -vvv --tb=long --junitxml=${REPORT_DIR}/python/chaos-tests.xml" \
    "${REPORT_DIR}/python/chaos-tests.log" \
    "Resilience Testing"

run_test "Contract Tests" \
    "python -m pytest tests/contract/ -vvv --tb=long --junitxml=${REPORT_DIR}/python/contract-tests.xml" \
    "${REPORT_DIR}/python/contract-tests.log" \
    "API Contracts"

run_test "Compliance Tests" \
    "python -m pytest tests/compliance/ -vvv --tb=long --junitxml=${REPORT_DIR}/python/compliance-tests.xml" \
    "${REPORT_DIR}/python/compliance-tests.log" \
    "Architecture Compliance"

# 7. Frontend Unit Tests
echo -e "${YELLOW}Phase 7: Frontend Unit Tests${NC}"
echo "## Phase 7: Frontend Unit Tests" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

run_test "Jest Unit Tests" \
    "cd web && npm run test:ci -- --coverage --coverageDirectory=../tests/reports/${TIMESTAMP}/frontend/coverage --json --outputFile=../tests/reports/${TIMESTAMP}/frontend/jest-results.json" \
    "${REPORT_DIR}/frontend/jest-tests.log" \
    "Frontend Unit Tests"

# 8. End-to-End Tests
echo -e "${YELLOW}Phase 8: End-to-End Tests${NC}"
echo "## Phase 8: End-to-End Tests" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Kill any existing processes on ports
echo -e "${BLUE}Cleaning up ports...${NC}"
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Start development server for E2E tests
echo -e "${BLUE}Starting development server for E2E tests...${NC}"
cd web && npm run dev > "${REPORT_DIR}/frontend/dev-server.log" 2>&1 &
DEV_SERVER_PID=$!
cd ..

# Wait for server to start
sleep 15

run_test "Playwright E2E Tests" \
    "cd web && npx playwright test --reporter=html:../tests/reports/${TIMESTAMP}/frontend/playwright-report --reporter=json:../tests/reports/${TIMESTAMP}/frontend/playwright-results.json" \
    "${REPORT_DIR}/frontend/e2e-tests.log" \
    "End-to-End Tests"

# Kill development server
kill $DEV_SERVER_PID 2>/dev/null || true

# 9. Performance Analysis
echo -e "${YELLOW}Phase 9: Performance Analysis${NC}"
echo "## Phase 9: Performance Analysis" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

run_test "Bundle Size Analysis" \
    "cd web && npm run analyze -- --json > ../tests/reports/${TIMESTAMP}/performance/bundle-analysis.json" \
    "${REPORT_DIR}/performance/bundle-size.log" \
    "Performance"

# 10. Generate Final Report
echo -e "${YELLOW}Phase 10: Generating Final Report${NC}"
echo "## Test Execution Complete" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "**Total Duration**: Started at ${TIMESTAMP}, completed at $(date +"%Y%m%d_%H%M%S")" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Generate HTML index
cat > "${REPORT_DIR}/index.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Test Report - ${TIMESTAMP}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        .pass { color: green; }
        .fail { color: red; }
        .section { margin: 20px 0; padding: 10px; border: 1px solid #ddd; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>FreeAgentics Test Report</h1>
    <p><strong>Generated:</strong> $(date)</p>
    <p><strong>Timestamp:</strong> ${TIMESTAMP}</p>
    
    <h2>Report Sections</h2>
    <ul>
        <li><a href="test-summary.md">Test Summary</a></li>
        <li><a href="python/coverage-html/index.html">Python Coverage Report</a></li>
        <li><a href="frontend/coverage/lcov-report/index.html">Frontend Coverage Report</a></li>
        <li><a href="frontend/playwright-report/index.html">E2E Test Report</a></li>
        <li><a href="quality/mypy-html/index.html">Type Checking Report</a></li>
    </ul>
    
    <h2>Quick Links</h2>
    <div class="section">
        <h3>Quality Reports</h3>
        <ul>
            <li><a href="quality/mypy-output.log">MyPy Output</a></li>
            <li><a href="quality/typescript-output.log">TypeScript Output</a></li>
            <li><a href="quality/flake8-report.txt">Flake8 Report</a></li>
            <li><a href="quality/eslint-output.log">ESLint Output</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h3>Security Reports</h3>
        <ul>
            <li><a href="security/bandit-report.json">Bandit Security Scan</a></li>
            <li><a href="security/safety-report.json">Dependency Vulnerabilities</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h3>Test Results</h3>
        <ul>
            <li><a href="python/unit-tests.xml">Python Unit Test Results (XML)</a></li>
            <li><a href="python/integration-tests.xml">Python Integration Test Results (XML)</a></li>
            <li><a href="frontend/jest-results.json">Frontend Test Results (JSON)</a></li>
            <li><a href="frontend/playwright-results.json">E2E Test Results (JSON)</a></li>
        </ul>
    </div>
</body>
</html>
EOF

echo -e "${GREEN}Test execution complete!${NC}"
echo -e "${BLUE}Reports saved to: ${REPORT_DIR}${NC}"
echo -e "${BLUE}View summary at: ${REPORT_DIR}/test-summary.md${NC}"
echo -e "${BLUE}View HTML report at: ${REPORT_DIR}/index.html${NC}"

# Create a symlink to latest report
ln -sfn "${TIMESTAMP}" "tests/reports/latest"
echo -e "${CYAN}Latest report symlinked to: tests/reports/latest${NC}"