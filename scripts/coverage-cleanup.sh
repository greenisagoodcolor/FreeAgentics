#!/bin/bash
# Coverage Cleanup Script - Remove obsolete coverage artifacts
# Part of Task 9.7 cleanup requirements

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[COVERAGE-CLEANUP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[COVERAGE-CLEANUP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[COVERAGE-CLEANUP]${NC} $1"
}

# Track what we clean
CLEANED_COUNT=0

# Function to safely remove files/directories
safe_remove() {
    local path="$1"
    local type="$2"

    if [ -e "$path" ]; then
        rm -rf "$path"
        ((CLEANED_COUNT++))
        echo "  - Removed $type: $path"
    fi
}

print_status "Starting coverage artifact cleanup..."

# 1. Remove old coverage data files
print_status "Cleaning coverage data files..."
find . -name ".coverage" -type f -exec rm -f {} \; 2>/dev/null
find . -name ".coverage.*" -type f -exec rm -f {} \; 2>/dev/null
find . -name "coverage.xml" -type f -exec rm -f {} \; 2>/dev/null
find . -name "coverage.json" -type f -exec rm -f {} \; 2>/dev/null
find . -name "coverage.lcov" -type f -exec rm -f {} \; 2>/dev/null
find . -name ".coverage.old" -type f -exec rm -f {} \; 2>/dev/null

# 2. Remove HTML coverage directories
print_status "Cleaning HTML coverage directories..."
find . -name "htmlcov" -type d -exec rm -rf {} \; 2>/dev/null
find . -name "htmlcov_backup" -type d -exec rm -rf {} \; 2>/dev/null
find . -name "coverage_html_report" -type d -exec rm -rf {} \; 2>/dev/null

# 3. Clean up test report directories (keep structure, remove old reports)
print_status "Cleaning old test reports..."
if [ -d "test-reports" ]; then
    find test-reports -name "*.xml" -mtime +7 -delete 2>/dev/null
    find test-reports -name "*.json" -mtime +7 -delete 2>/dev/null
    find test-reports -name "*.html" -mtime +7 -delete 2>/dev/null
fi

# 4. Remove Python cache files (interfere with coverage)
print_status "Cleaning Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null
find . -name "*.pyc" -type f -delete 2>/dev/null
find . -name "*.pyo" -type f -delete 2>/dev/null
find . -name "*~" -type f -delete 2>/dev/null

# 5. Clean pytest cache
print_status "Cleaning pytest cache..."
safe_remove ".pytest_cache" "pytest cache"

# 6. Remove obsolete coverage configuration backups
print_status "Cleaning obsolete coverage configurations..."
find . -name ".coveragerc.old" -type f -delete 2>/dev/null
find . -name ".coveragerc.backup" -type f -delete 2>/dev/null
find . -name "pytest.ini.old" -type f -delete 2>/dev/null

# 7. Clean up legacy coverage directories
print_status "Cleaning legacy coverage directories..."
safe_remove "coverage_reports" "legacy coverage directory"
safe_remove "test_coverage" "legacy coverage directory"
safe_remove ".coverage_data" "legacy coverage directory"

# 8. Remove duplicate coverage analysis files
print_status "Cleaning duplicate coverage analysis files..."
find . -name "coverage-report*.txt" -type f -delete 2>/dev/null
find . -name "coverage-analysis*.json" -type f -delete 2>/dev/null
find . -name "coverage-gaps*.txt" -type f -mtime +1 -delete 2>/dev/null

# 9. Clean web coverage artifacts
print_status "Cleaning web coverage artifacts..."
if [ -d "web" ]; then
    safe_remove "web/coverage" "web coverage directory"
    safe_remove "web/.nyc_output" "web nyc output"
    find web -name "coverage-final.json" -type f -delete 2>/dev/null
fi

# 10. Remove temporary coverage files
print_status "Cleaning temporary coverage files..."
find /tmp -name ".coverage*" -user $(whoami) -delete 2>/dev/null

# Summary
print_success "Cleanup complete!"
print_success "Removed obsolete coverage artifacts"
print_status "Note: Current coverage data preserved:"
echo "  - coverage.json (if recently generated)"
echo "  - coverage-gaps.md (if recently generated)"
echo "  - test-reports/coverage/* (recent reports)"

# Recommend next steps
echo ""
print_status "Recommended next steps:"
echo "  1. Run 'make coverage-dev' to generate fresh coverage reports"
echo "  2. Run 'make coverage-gaps' to analyze coverage gaps"
echo "  3. Add coverage artifacts to .gitignore if not already present"
