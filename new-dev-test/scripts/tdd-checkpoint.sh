#!/bin/bash
# TDD Reality Checkpoint Script
# Runs comprehensive checks as required by TDD principles in CLAUDE.MD

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[TDD-CHECKPOINT]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[TDD-CHECKPOINT]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[TDD-CHECKPOINT]${NC} $1"
}

print_error() {
    echo -e "${RED}[TDD-CHECKPOINT]${NC} $1"
}

# Track overall success
OVERALL_SUCCESS=true

# Function to run a check and track success
run_check() {
    local check_name="$1"
    local command="$2"

    print_status "Running $check_name..."

    if eval "$command"; then
        print_success "$check_name: ✅ PASSED"
        return 0
    else
        print_error "$check_name: ❌ FAILED"
        OVERALL_SUCCESS=false
        return 1
    fi
}

print_status "Starting TDD Reality Checkpoint..."
print_status "This validates all code meets TDD requirements from CLAUDE.MD"
print_status ""

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

# Run all TDD checks
print_status "================================================"
print_status "RUNNING TDD REALITY CHECKPOINT"
print_status "================================================"

# 1. Format Check
run_check "Code Formatting" "ruff format --check ."

# 2. Linting Check
run_check "Code Linting" "ruff check ."

# 3. Type Checking
run_check "Type Checking" "python -m mypy . --ignore-missing-imports --no-error-summary"

# 4. Test Execution with 100% Coverage
run_check "Test Suite with 100% Coverage" "pytest --cov-fail-under=100 --cov-branch --tb=short"

# 5. Security Check
run_check "Security Scanning" "bandit -r . -x tests/ -f json -o /dev/null"

# 6. Import Validation
run_check "Import Validation" "python -c 'import agents, api, auth, coalitions, database, inference, knowledge_graph, observability, world; print(\"All imports successful\")'"

print_status ""
print_status "================================================"

if [ "$OVERALL_SUCCESS" = true ]; then
    print_success "🎉 ALL CHECKS PASSED - TDD REALITY CHECKPOINT SUCCESSFUL"
    print_success "Code is ready for production deployment"
    exit 0
else
    print_error "❌ SOME CHECKS FAILED - MUST FIX ALL ISSUES"
    print_error "TDD requires ALL checks to pass before continuing"
    exit 1
fi
