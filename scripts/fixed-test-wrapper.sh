#!/bin/bash
# Fixed test wrapper that uses shared timestamp and proper environment
# Usage: ./scripts/fixed-test-wrapper.sh <test-name> <command>

set -e

TEST_NAME=$1
shift
COMMAND="$@"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

# Ensure we have the environment set up
if [ -z "$TEST_TIMESTAMP" ] || [ -z "$TEST_REPORT_DIR" ]; then
    echo -e "${RED}Error: Test environment not initialized. Run setup first.${NC}"
    exit 1
fi

# Ensure the report directory exists
mkdir -p "$TEST_REPORT_DIR"

# Log file (use absolute paths)
LOG_FILE="$(pwd)/${TEST_REPORT_DIR}/${TEST_NAME}.log"
SUMMARY_FILE="$(pwd)/${TEST_REPORT_DIR}/test-summary.md"

# Ensure summary file exists
if [ ! -f "$SUMMARY_FILE" ]; then
    cat > "$SUMMARY_FILE" << EOF
# Test Execution Report
**Timestamp**: ${TEST_TIMESTAMP}
**Date**: $(date)

## Test Results

EOF
fi

# Add test to summary
echo "### ${TEST_NAME}" >> "$SUMMARY_FILE"
echo "**Start**: $(date +"%H:%M:%S")" >> "$SUMMARY_FILE"

# Run the command with proper environment
echo -e "${CYAN}Running ${TEST_NAME}...${NC}"
echo "Command: $COMMAND" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "Environment:" >> "$LOG_FILE"
echo "  TEST_TIMESTAMP=$TEST_TIMESTAMP" >> "$LOG_FILE"
echo "  TEST_REPORT_DIR=$TEST_REPORT_DIR" >> "$LOG_FILE"
echo "  COVERAGE_FILE=$COVERAGE_FILE" >> "$LOG_FILE"
echo "  COVERAGE_RC_FILE=$COVERAGE_RC_FILE" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"

# Source the environment file to ensure all variables are set
if [ -f "$TEST_REPORT_DIR/environment.sh" ]; then
    source "$TEST_REPORT_DIR/environment.sh"
fi

if eval "$COMMAND" >> "$LOG_FILE" 2>&1; then
    echo -e "${GREEN}✓ ${TEST_NAME} passed${NC}"
    echo "**Status**: ✅ PASSED" >> "$SUMMARY_FILE"
    EXIT_CODE=0
else
    EXIT_CODE=$?
    echo -e "${RED}✗ ${TEST_NAME} failed${NC}"
    echo "**Status**: ❌ FAILED (exit code: $EXIT_CODE)" >> "$SUMMARY_FILE"
fi

echo "**End**: $(date +"%H:%M:%S")" >> "$SUMMARY_FILE"
echo "**Log**: [${TEST_NAME}.log](${TEST_NAME}.log)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

exit $EXIT_CODE