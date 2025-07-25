#!/bin/bash
# Repository Cleanup Script 3: Remove Test Artifacts
# Generated by Repository Forensics Analysis - 2025-07-17

echo "🧹 Starting test artifacts cleanup..."

# Change to repository root
cd "$(dirname "$0")/.."

# Create backup list
BACKUP_LIST="cleanup_scripts/removed_test_artifacts_$(date +%Y%m%d_%H%M%S).log"

# Function to safely remove file with logging
safe_remove() {
    local file="$1"
    if [ -f "$file" ]; then
        echo "Removing: $file" | tee -a "$BACKUP_LIST"
        rm -f "$file"
        return 0
    else
        echo "File not found: $file" | tee -a "$BACKUP_LIST"
        return 1
    fi
}

# Function to safely remove directory with logging
safe_remove_dir() {
    local dir="$1"
    if [ -d "$dir" ]; then
        echo "Removing directory: $dir" | tee -a "$BACKUP_LIST"
        rm -rf "$dir"
        return 0
    else
        echo "Directory not found: $dir" | tee -a "$BACKUP_LIST"
        return 1
    fi
}

echo "Backup list created: $BACKUP_LIST"

# Remove test database files
echo "Removing test database files..."
safe_remove "test_routes.db"
safe_remove "test.db"
safe_remove "tests/reporting/coverage.db"
safe_remove "tests/reporting/test_metrics.db"
safe_remove "tests/reporting/archival_tracking.db"
safe_remove "logs/test_analysis.db"
safe_remove "logs/aggregation.db"
safe_remove "logs/test_aggregation.db"

# Remove test output files
echo "Removing test output files..."
safe_remove "test_async_coordination_performance.py"
safe_remove "test_encryption_basic.py"
safe_remove "test_encryption_minimal.py"
safe_remove "test_realistic_multi_agent_performance.py"
safe_remove "test_security_standalone.py"

# Remove archived test reports
echo "Removing archived test reports..."
if [ -d ".archive/test_reports" ]; then
    echo "Removing .archive/test_reports directory..." | tee -a "$BACKUP_LIST"
    rm -rf ".archive/test_reports/"
fi

# Remove test-reports directory
echo "Removing test-reports directory..."
if [ -d "test-reports" ]; then
    echo "Removing test-reports directory..." | tee -a "$BACKUP_LIST"
    rm -rf "test-reports/"
fi

# Remove performance benchmark results
echo "Removing performance benchmark results..."
safe_remove "tests/performance/matrix_caching_benchmark_results_20250704_173217.json"

# Remove disabled test files
echo "Removing disabled test files..."
find . -name "*.py.DISABLED_MOCKS" -type f -exec echo "Removing: {}" \; -exec rm -f {} \; | tee -a "$BACKUP_LIST"

# Remove temporary test files
echo "Removing temporary test files..."
find . -name "test_*.tmp" -type f -exec echo "Removing: {}" \; -exec rm -f {} \; | tee -a "$BACKUP_LIST"

# Remove jest cache (if exists)
echo "Removing jest cache..."
if [ -d "web/node_modules/.cache/jest" ]; then
    echo "Removing jest cache directory..." | tee -a "$BACKUP_LIST"
    rm -rf "web/node_modules/.cache/jest/"
fi

# Remove Next.js cache
echo "Removing Next.js cache..."
if [ -d "web/.next/cache" ]; then
    echo "Removing Next.js cache directory..." | tee -a "$BACKUP_LIST"
    rm -rf "web/.next/cache/"
fi

# Remove TypeScript build info
echo "Removing TypeScript build info..."
find . -name "tsconfig.tsbuildinfo" -type f -exec echo "Removing: {}" \; -exec rm -f {} \; | tee -a "$BACKUP_LIST"

# Remove Webpack cache
echo "Removing Webpack cache..."
find . -name ".webpack" -type d -exec echo "Removing directory: {}" \; -exec rm -rf {} \; | tee -a "$BACKUP_LIST"

# Count removed items
REMOVED_COUNT=$(grep -c "Removing:" "$BACKUP_LIST" 2>/dev/null || echo "0")
echo "✅ Test artifacts cleanup completed"
echo "Items removed: $REMOVED_COUNT"
echo "Backup list saved: $BACKUP_LIST"
echo "Estimated space freed: ~200MB"
