#!/bin/bash
# Fast TDD Test Script - Parallel execution for rapid feedback
# Optimized for TDD Red-Green-Refactor cycles

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[TDD-FAST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[TDD-FAST]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[TDD-FAST]${NC} $1"
}

print_error() {
    echo -e "${RED}[TDD-FAST]${NC} $1"
}

# Ensure we're in the project root
if [ ! -f "pyproject.toml" ]; then
    print_error "Error: Must be run from project root directory"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "No virtual environment detected. Activating .venv..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "No .venv directory found. Please create one with: python -m venv .venv"
        exit 1
    fi
fi

print_status "Running fast parallel tests for TDD cycle..."

# Get number of CPU cores for optimal parallelization
CORES=$(nproc)
print_status "Using $CORES CPU cores for parallel execution"

# Run tests with optimized settings for TDD
pytest \
    -n $CORES \
    --tb=short \
    --no-header \
    --disable-warnings \
    --lf \
    --ff \
    --cov-fail-under=100 \
    --cov-branch \
    --cov-report=term-missing:skip-covered \
    -q \
    "$@"

if [ $? -eq 0 ]; then
    print_success "✅ All tests passed! Ready for next TDD cycle."
else
    print_error "❌ Tests failed. Fix failing tests before continuing TDD cycle."
    exit 1
fi
