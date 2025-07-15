#!/bin/bash
# Development Coverage Script - Fast feedback for TDD cycles
# Generates coverage reports with relaxed thresholds for development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[COVERAGE-DEV]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[COVERAGE-DEV]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[COVERAGE-DEV]${NC} $1"
}

print_error() {
    echo -e "${RED}[COVERAGE-DEV]${NC} $1"
}

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "No virtual environment detected. Activating .venv..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    else
        print_error "No .venv directory found. Please create one with: python -m venv .venv"
        exit 1
    fi
fi

# Clean previous coverage data
print_status "Cleaning previous coverage data..."
rm -f .coverage
rm -f .coverage.*
rm -rf htmlcov/

# Run tests with coverage
print_status "Running tests with coverage (development mode)..."
export CONTEXT="development"

# Development mode: Fast execution, no strict requirements
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
    --cov-report=term-missing:skip-covered \
    --cov-report=html \
    --cov-report=json \
    --tb=short \
    --maxfail=5 \
    -v \
    ${@:-tests/unit} || true

# Generate reports
print_status "Generating coverage reports..."

# Terminal report with missing lines
coverage report --show-missing --skip-covered

# Generate detailed HTML report
coverage html

# Generate JSON report for analysis
coverage json --pretty-print

# Generate XML for CI tools
coverage xml

# Generate LCOV for additional tooling
coverage lcov

# Summary
print_success "Coverage reports generated:"
echo "  - HTML:     htmlcov/index.html"
echo "  - JSON:     coverage.json"
echo "  - XML:      coverage.xml"
echo "  - LCOV:     coverage.lcov"
echo ""

# Show coverage summary
print_status "Coverage Summary:"
coverage report --format=markdown | head -20 || coverage report | head -20

# Open HTML report if on a desktop
if command -v xdg-open &> /dev/null; then
    print_status "Opening HTML report in browser..."
    xdg-open htmlcov/index.html &
elif command -v open &> /dev/null; then
    print_status "Opening HTML report in browser..."
    open htmlcov/index.html &
fi