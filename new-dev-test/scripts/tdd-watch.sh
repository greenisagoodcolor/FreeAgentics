#!/bin/bash
# TDD Watch Script - Continuous Testing for Test-Driven Development
# This script starts pytest-watch for automatic test execution on file changes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[TDD-WATCH]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[TDD-WATCH]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[TDD-WATCH]${NC} $1"
}

print_error() {
    echo -e "${RED}[TDD-WATCH]${NC} $1"
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

# Check if pytest-watch is installed
if ! command -v ptw &> /dev/null; then
    print_error "pytest-watch not found. Installing..."
    pip install pytest-watch
fi

print_status "Starting TDD Watch Mode..."
print_status "This will run tests automatically when Python files change"
print_status "Press Ctrl+C to stop watching"
print_status ""

# Run pytest-watch with our configuration
ptw \
    --config .pytest-watch.yml \
    --clear \
    --wait \
    --runner "pytest --tb=short -x --lf --ff --no-header" \
    tests/
