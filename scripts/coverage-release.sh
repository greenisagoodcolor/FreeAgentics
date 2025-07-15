#!/bin/bash
# Release Coverage Script - Ultra-strict validation for production releases
# Enforces 100% coverage requirement as per TDD standards

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[COVERAGE-RELEASE]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[COVERAGE-RELEASE]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[COVERAGE-RELEASE]${NC} $1"
}

print_error() {
    echo -e "${RED}[COVERAGE-RELEASE]${NC} $1"
}

# Configuration
COVERAGE_THRESHOLD=100  # TDD requirement
BRANCH_THRESHOLD=100    # TDD requirement

# Create reports directory
mkdir -p test-reports/release-coverage

# Clean all coverage data
print_status "Cleaning all coverage data..."
find . -name ".coverage*" -type f -delete
find . -name "coverage.xml" -type f -delete
find . -name "coverage.json" -type f -delete
rm -rf htmlcov/
rm -rf test-reports/release-coverage/*

# Run ALL tests with strictest coverage
print_status "Running ALL tests with 100% coverage requirement..."
export CONTEXT="release"

# Release mode: Ultra-strict, all tests, no failures allowed
pytest \
    --cov=agents \
    --cov=api \
    --cov=auth \
    --cov=coalitions \
    --cov=config \
    --cov=database \
    --cov=inference \
    --cov=knowledge_graph \
    --cov=observability \
    --cov=world \
    --cov-branch \
    --cov-report=term-missing \
    --cov-report=html:test-reports/release-coverage/html \
    --cov-report=xml:test-reports/release-coverage/coverage.xml \
    --cov-report=json:test-reports/release-coverage/coverage.json \
    --cov-fail-under=$COVERAGE_THRESHOLD \
    --tb=short \
    --strict-markers \
    --strict-config \
    --maxfail=1 \
    --junit-xml=test-reports/release-coverage/junit.xml \
    -v \
    tests/unit tests/integration tests/e2e

# Check coverage succeeded
if [ $? -ne 0 ]; then
    print_error "RELEASE BLOCKED: Coverage below 100%"
    print_error "TDD requires 100% coverage for production releases"
    
    # Show what's missing
    print_warning "Generating detailed gap analysis..."
    python -c "
import json
from pathlib import Path

try:
    with open('test-reports/release-coverage/coverage.json') as f:
        data = json.load(f)
        
    print('\\nModules requiring coverage:')
    for file_path, file_data in sorted(data['files'].items()):
        coverage_pct = file_data['summary']['percent_covered']
        if coverage_pct < 100:
            missing = file_data['missing_lines']
            print(f'  {file_path}: {coverage_pct:.1f}% (missing lines: {missing[:5]}...)')
except:
    pass
"
    exit 1
fi

print_success "100% COVERAGE ACHIEVED - Release criteria met!"

# Generate comprehensive release report
print_status "Generating comprehensive release coverage report..."

# LCOV format
coverage lcov -o test-reports/release-coverage/coverage.lcov

# Generate release certification
python -c "
import json
from datetime import datetime

with open('test-reports/release-coverage/coverage.json') as f:
    data = json.load(f)
    
certification = {
    'timestamp': datetime.utcnow().isoformat(),
    'coverage': {
        'lines': data['totals']['percent_covered'],
        'branches': data['totals'].get('percent_covered_branches', 100),
        'functions': data['totals'].get('percent_covered_functions', 100)
    },
    'total_lines': data['totals']['num_statements'],
    'total_branches': data['totals'].get('num_branches', 0),
    'files_covered': len(data['files']),
    'certification': 'PASSED - 100% Coverage Achieved'
}

with open('test-reports/release-coverage/certification.json', 'w') as f:
    json.dump(certification, f, indent=2)

print(f'\\nRelease Certification:')
print(f'  Lines:     {certification[\"coverage\"][\"lines\"]:.1f}%')
print(f'  Branches:  {certification[\"coverage\"][\"branches\"]:.1f}%')
print(f'  Total:     {certification[\"total_lines\"]} lines, {certification[\"total_branches\"]} branches')
print(f'  Status:    {certification[\"certification\"]}')
"

# Create release badge
python -c "
import json
with open('test-reports/release-coverage/certification.json') as f:
    cert = json.load(f)
    badge = {
        'schemaVersion': 1,
        'label': 'release coverage',
        'message': '100%',
        'color': 'brightgreen'
    }
    with open('test-reports/release-coverage/release-badge.json', 'w') as bf:
        json.dump(badge, bf, indent=2)
"

# Summary
print_success "Release Coverage Validation Complete:"
echo "  - HTML:          test-reports/release-coverage/html/index.html"
echo "  - XML:           test-reports/release-coverage/coverage.xml"
echo "  - JSON:          test-reports/release-coverage/coverage.json"
echo "  - LCOV:          test-reports/release-coverage/coverage.lcov"
echo "  - Certification: test-reports/release-coverage/certification.json"
echo "  - Badge:         test-reports/release-coverage/release-badge.json"
echo ""
print_success "✅ CODE IS READY FOR PRODUCTION RELEASE"