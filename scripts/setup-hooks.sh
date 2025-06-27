#!/bin/bash
#
# FreeAgentics Pre-Commit Hooks Setup & Validation
# Expert Committee: Robert C. Martin, Kent Beck, Rich Hickey, Conor Heins
#
# This script sets up and validates all pre-commit hooks across the entire project
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BOLD}${BLUE}FreeAgentics Pre-Commit Hooks Setup${NC}"
echo "Expert Committee: Robert C. Martin, Kent Beck, Rich Hickey, Conor Heins"
echo "============================================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1${NC}"
    else
        echo -e "${RED}❌ $1${NC}"
        exit 1
    fi
}

echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"

# Check Python
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    echo -e "${GREEN}✅ Python 3 found: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.11+${NC}"
    exit 1
fi

# Check pip
if command_exists pip; then
    echo -e "${GREEN}✅ pip found${NC}"
else
    echo -e "${RED}❌ pip not found. Please install pip${NC}"
    exit 1
fi

# Check Node.js
if command_exists node; then
    NODE_VERSION=$(node --version 2>&1)
    echo -e "${GREEN}✅ Node.js found: $NODE_VERSION${NC}"
else
    echo -e "${RED}❌ Node.js not found. Please install Node.js 18+${NC}"
    exit 1
fi

# Check npm
if command_exists npm; then
    NPM_VERSION=$(npm --version 2>&1)
    echo -e "${GREEN}✅ npm found: $NPM_VERSION${NC}"
else
    echo -e "${RED}❌ npm not found. Please install npm${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}📦 Installing Python dependencies...${NC}"

# Install Python development dependencies
pip install -r requirements-dev.txt
print_status "Python dependencies installed"

echo ""
echo -e "${YELLOW}🔧 Setting up pre-commit hooks...${NC}"

# Install pre-commit hooks
pre-commit install --install-hooks
print_status "Pre-commit hooks installed"

# Install commit-msg hook
pre-commit install --hook-type commit-msg
print_status "Commit message validation installed"

echo ""
echo -e "${YELLOW}📦 Setting up frontend hooks...${NC}"

# Setup frontend dependencies and Husky
cd web
npm install
print_status "Frontend dependencies installed"

# Initialize Husky
npx husky install
print_status "Husky initialized"

# Make sure hooks are executable
chmod +x .husky/pre-commit .husky/commit-msg 2>/dev/null || true

cd "$PROJECT_ROOT"

echo ""
echo -e "${YELLOW}🧪 Running validation tests...${NC}"

# Test pre-commit configuration
echo "Validating pre-commit configuration..."
pre-commit validate-config
print_status "Pre-commit configuration valid"

# Test a simple hook run (just check basic syntax)
echo "Testing basic hook functionality..."
echo "# Test comment" > /tmp/test_file.py
pre-commit run trailing-whitespace --files /tmp/test_file.py >/dev/null 2>&1
print_status "Basic hooks functional"
rm -f /tmp/test_file.py

echo ""
echo -e "${BOLD}${GREEN}🎉 Pre-commit hooks setup completed successfully!${NC}"
echo ""
echo -e "${YELLOW}📋 Quick validation commands:${NC}"
echo "  make validate-hooks    # Run all hooks on all files"
echo "  pre-commit run --all-files  # Manual hook execution"
echo "  make hooks-update      # Update hook versions"
echo ""
echo -e "${YELLOW}🔧 Expert Committee Quality Standards Active:${NC}"
echo "  - Robert C. Martin: Code cleanliness and readability"
echo "  - Kent Beck: Incremental improvement and testing"
echo "  - Rich Hickey: Simplicity and correctness"
echo "  - Conor Heins: Mathematical rigor and type safety"
echo ""
echo -e "${BLUE}Ready for development with automated quality gates! 🚀${NC}"
