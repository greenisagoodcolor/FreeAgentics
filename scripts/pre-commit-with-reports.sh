#!/bin/bash
# Pre-commit wrapper that outputs to timestamped directories
# This ensures pre-commit test outputs don't clutter the root

set -e

# Get timestamp
TIMESTAMP=${PRE_COMMIT_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
REPORT_DIR="tests/reports/pre-commit-${TIMESTAMP}"

# Create report directory
mkdir -p "$REPORT_DIR"/{security,quality}

# Export paths for tools that respect environment variables
export BANDIT_REPORT_FILE="$REPORT_DIR/security/bandit-report.json"
export MYPY_CACHE_DIR="$REPORT_DIR/quality/.mypy_cache"

# Create a temporary config that redirects outputs
PRE_COMMIT_CONFIG_TEMP="$REPORT_DIR/.pre-commit-config.yaml"
cp .pre-commit-config.yaml "$PRE_COMMIT_CONFIG_TEMP"

# Run pre-commit with custom paths
echo "Running pre-commit with outputs to: $REPORT_DIR"

# Capture output
PRE_COMMIT_OUTPUT="$REPORT_DIR/pre-commit-output.log"

if pre-commit run --config "$PRE_COMMIT_CONFIG_TEMP" "$@" > "$PRE_COMMIT_OUTPUT" 2>&1; then
    echo "✅ Pre-commit checks passed"
    cat "$PRE_COMMIT_OUTPUT"
    exit 0
else
    EXIT_CODE=$?
    echo "❌ Pre-commit checks failed"
    cat "$PRE_COMMIT_OUTPUT"
    echo ""
    echo "Full output saved to: $PRE_COMMIT_OUTPUT"
    echo "Reports saved to: $REPORT_DIR"
    exit $EXIT_CODE
fi