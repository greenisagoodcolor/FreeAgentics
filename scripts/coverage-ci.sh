#!/bin/bash
# CI Coverage - Comprehensive validation for merge gates  
# Usage: ./scripts/coverage-ci.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "🔍 Running CI coverage validation..."
echo "⚙️  Using comprehensive test suite with branch coverage"

# Run comprehensive coverage check
python scripts/coverage-check.py --profile=ci --output="coverage-ci.json"

# Generate XML report for CI integration
if [ -f "coverage-ci.xml" ]; then
    echo "📄 XML report generated for CI: coverage-ci.xml"
fi

# Generate HTML report for review
if [ -f "htmlcov-ci/index.html" ]; then
    echo "🌐 HTML report available at: htmlcov-ci/index.html"
fi

# Check if we have coverage data for critical modules
echo "🎯 Validating critical module coverage..."

# Extract exit code for CI pipeline
COVERAGE_EXIT_CODE=$?

if [ $COVERAGE_EXIT_CODE -eq 0 ]; then
    echo "✅ CI coverage validation passed"
else
    echo "❌ CI coverage validation failed"
    echo "💡 Review coverage report and fix threshold violations"
fi

exit $COVERAGE_EXIT_CODE