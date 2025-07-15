#!/bin/bash
# Simple integration test runner for tests that don't require external services
# For nemesis-level testing of basic integration points

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo -e "${BLUE}FreeAgentics Simple Integration Test Runner${NC}"
echo -e "${BLUE}===========================================${NC}\n"

# Source test environment if it exists
if [ -f ".env.test" ]; then
    echo -e "${BLUE}Loading test environment from .env.test...${NC}"
    set -a
    source ".env.test"
    set +a
else
    echo -e "${YELLOW}Warning: .env.test not found. Some tests may fail.${NC}"
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${RED}Error: Virtual environment not found. Run 'make install' first.${NC}"
    exit 1
fi

# Tests that don't require external services
SIMPLE_TESTS=(
    "test_coordination_interface_simple.py"
    "test_pymdp_validation.py"
    "test_action_sampling_issue.py"
    "test_nemesis_pymdp_validation.py"
    "test_pymdp_hard_failure_integration.py"
)

echo -e "\n${BLUE}Running integration tests that don't require external services...${NC}"
echo -e "${BLUE}Tests to run: ${#SIMPLE_TESTS[@]}${NC}\n"

# Track results
PASSED=0
FAILED=0
FAILED_TESTS=()

# Run each test
for test in "${SIMPLE_TESTS[@]}"; do
    test_path="tests/integration/$test"
    
    if [ -f "$test_path" ]; then
        echo -e "${BLUE}Running: $test${NC}"
        
        if pytest "$test_path" -v --tb=short --timeout=60; then
            echo -e "${GREEN}✓ $test passed${NC}\n"
            ((PASSED++))
        else
            echo -e "${RED}✗ $test failed${NC}\n"
            ((FAILED++))
            FAILED_TESTS+=("$test")
        fi
    else
        echo -e "${YELLOW}⚠ $test not found, skipping${NC}\n"
    fi
done

# Summary
echo -e "\n${BLUE}Test Summary${NC}"
echo -e "${BLUE}============${NC}"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -gt 0 ]; then
    echo -e "\n${RED}Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}- $test${NC}"
    done
    echo -e "\n${YELLOW}To debug a specific test:${NC}"
    echo -e "  pytest tests/integration/<test_name> -vvs --pdb"
    exit 1
else
    echo -e "\n${GREEN}All simple integration tests passed!${NC}"
    echo -e "\n${BLUE}Next steps:${NC}"
    echo -e "  1. Run full integration tests: ./scripts/run-integration-tests.sh"
    echo -e "  2. Run specific scenarios: pytest tests/integration/test_comprehensive_gnn_llm_coalition_integration.py"
    exit 0
fi