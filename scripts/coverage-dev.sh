#!/bin/bash
# Development Coverage - Fast feedback for active development
# Usage: ./scripts/coverage-dev.sh [module_name]

set -e

MODULE=${1:-""}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "🚀 Running development coverage analysis..."

if [ -n "$MODULE" ]; then
    echo "📁 Focusing on module: $MODULE"
    python scripts/coverage-check.py --profile=dev --module="$MODULE" --output="coverage-dev-$MODULE.json"
else
    echo "📊 Analyzing all core modules"
    python scripts/coverage-check.py --profile=dev --output="coverage-dev.json"
fi

# Open HTML report if available
if [ -f "htmlcov-dev/index.html" ]; then
    echo "🌐 HTML report available at: htmlcov-dev/index.html"
fi

echo "✅ Development coverage analysis complete"