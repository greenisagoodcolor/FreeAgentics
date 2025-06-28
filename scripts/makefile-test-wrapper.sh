#!/bin/bash
# Test wrapper script for timestamped output
# Usage: ./scripts/makefile-test-wrapper.sh <test-name> <command>

set -e

TEST_NAME=$1
shift
COMMAND="$@"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get timestamp from environment or generate new one
if [ -z "$TEST_TIMESTAMP" ]; then
    TEST_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
fi

if [ -z "$TEST_REPORT_DIR" ]; then
    TEST_REPORT_DIR="tests/reports/${TEST_TIMESTAMP}"
fi

# Create directories if needed
mkdir -p "${TEST_REPORT_DIR}"/{python,frontend,quality,security,integration,performance}
mkdir -p "${TEST_REPORT_DIR}"/quality/{mypy-html,mypy-reports}

# Log file
LOG_FILE="${TEST_REPORT_DIR}/${TEST_NAME}.log"
SUMMARY_FILE="${TEST_REPORT_DIR}/test-summary.md"

# Initialize summary if not exists
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

# Run the command
echo -e "${CYAN}Running ${TEST_NAME}...${NC}"
echo "Command: $COMMAND" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"

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

# Update symlink
ln -sfn "${TEST_TIMESTAMP}" "tests/reports/latest"

exit $EXIT_CODE