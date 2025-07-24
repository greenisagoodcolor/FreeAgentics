#!/bin/bash
# CI/CD Coverage Script - Strict validation for continuous integration
# Enforces coverage requirements and generates reports for CI systems

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[COVERAGE-CI]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[COVERAGE-CI]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[COVERAGE-CI]${NC} $1"
}

print_error() {
    echo -e "${RED}[COVERAGE-CI]${NC} $1"
}

# Configuration
COVERAGE_THRESHOLD=${COVERAGE_THRESHOLD:-80}  # Default 80% for CI
BRANCH_THRESHOLD=${BRANCH_THRESHOLD:-75}     # Default 75% for branch coverage

# Create reports directory
mkdir -p test-reports/coverage

# Clean previous coverage data
print_status "Cleaning previous coverage data..."
rm -f .coverage
rm -f .coverage.*
rm -rf htmlcov/
rm -rf test-reports/coverage/*

# Run tests with coverage
print_status "Running tests with coverage (CI mode)..."
export CONTEXT="ci"

# CI mode: Strict validation, all tests
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
    --cov-report=html:test-reports/coverage/html \
    --cov-report=xml:test-reports/coverage/coverage.xml \
    --cov-report=json:test-reports/coverage/coverage.json \
    --cov-fail-under=$COVERAGE_THRESHOLD \
    --tb=short \
    --strict-markers \
    --strict-config \
    --junit-xml=test-reports/junit.xml \
    tests/

# Check coverage succeeded
if [ $? -eq 0 ]; then
    print_success "Coverage threshold met: >= $COVERAGE_THRESHOLD%"
else
    print_error "Coverage below threshold: < $COVERAGE_THRESHOLD%"
    exit 1
fi

# Generate additional reports
print_status "Generating additional coverage reports..."

# LCOV format for some CI tools
coverage lcov -o test-reports/coverage/coverage.lcov

# Generate coverage badge data
python -c "
import json
with open('test-reports/coverage/coverage.json') as f:
    data = json.load(f)
    total = data['totals']['percent_covered']
    print(f'Coverage: {total:.1f}%')

    # Create badge JSON
    badge = {
        'schemaVersion': 1,
        'label': 'coverage',
        'message': f'{total:.1f}%',
        'color': 'brightgreen' if total >= 90 else 'green' if total >= 80 else 'yellow' if total >= 70 else 'red'
    }
    with open('test-reports/coverage/badge.json', 'w') as bf:
        json.dump(badge, bf, indent=2)
"

# Generate uncovered modules report
print_status "Analyzing uncovered modules..."
python -c "
import json
from pathlib import Path

with open('test-reports/coverage/coverage.json') as f:
    data = json.load(f)

uncovered = []
low_coverage = []

for file_path, file_data in data['files'].items():
    coverage_pct = file_data['summary']['percent_covered']
    if coverage_pct == 0:
        uncovered.append(file_path)
    elif coverage_pct < 50:
        low_coverage.append((file_path, coverage_pct))

if uncovered:
    print(f'\\n{len(uncovered)} modules with 0% coverage:')
    for module in sorted(uncovered)[:10]:  # Show first 10
        print(f'  - {module}')
    if len(uncovered) > 10:
        print(f'  ... and {len(uncovered) - 10} more')

if low_coverage:
    print(f'\\n{len(low_coverage)} modules with <50% coverage:')
    for module, pct in sorted(low_coverage, key=lambda x: x[1])[:10]:
        print(f'  - {module}: {pct:.1f}%')
    if len(low_coverage) > 10:
        print(f'  ... and {len(low_coverage) - 10} more')

# Save to file
report = {
    'uncovered_modules': uncovered,
    'low_coverage_modules': [{'path': p, 'coverage': c} for p, c in low_coverage]
}
with open('test-reports/coverage/gaps.json', 'w') as f:
    json.dump(report, f, indent=2)
"

# Summary
print_success "CI Coverage reports generated:"
echo "  - HTML:     test-reports/coverage/html/index.html"
echo "  - XML:      test-reports/coverage/coverage.xml"
echo "  - JSON:     test-reports/coverage/coverage.json"
echo "  - LCOV:     test-reports/coverage/coverage.lcov"
echo "  - Badge:    test-reports/coverage/badge.json"
echo "  - Gaps:     test-reports/coverage/gaps.json"
echo "  - JUnit:    test-reports/junit.xml"

# Exit with appropriate code
exit 0
