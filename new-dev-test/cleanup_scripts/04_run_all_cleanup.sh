#!/bin/bash
# Repository Cleanup Script 4: Master Cleanup Script
# Generated by Repository Forensics Analysis - 2025-07-17

echo "🚀 Starting comprehensive repository cleanup..."
echo "Timestamp: $(date)"

# Change to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Create master log file
MASTER_LOG="cleanup_scripts/master_cleanup_$(date +%Y%m%d_%H%M%S).log"
echo "Master cleanup log: $MASTER_LOG" | tee "$MASTER_LOG"

# Make all scripts executable
chmod +x cleanup_scripts/*.sh

# Function to run script with logging
run_script() {
    local script="$1"
    local description="$2"

    echo "" | tee -a "$MASTER_LOG"
    echo "========================================" | tee -a "$MASTER_LOG"
    echo "Running: $description" | tee -a "$MASTER_LOG"
    echo "Script: $script" | tee -a "$MASTER_LOG"
    echo "========================================" | tee -a "$MASTER_LOG"

    if [ -f "$script" ]; then
        bash "$script" 2>&1 | tee -a "$MASTER_LOG"
        local exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo "✅ $description completed successfully" | tee -a "$MASTER_LOG"
        else
            echo "❌ $description failed with exit code: $exit_code" | tee -a "$MASTER_LOG"
        fi

        return $exit_code
    else
        echo "❌ Script not found: $script" | tee -a "$MASTER_LOG"
        return 1
    fi
}

# Pre-cleanup repository size
echo "Calculating pre-cleanup repository size..." | tee -a "$MASTER_LOG"
PRE_SIZE=$(du -sh . 2>/dev/null | cut -f1)
echo "Repository size before cleanup: $PRE_SIZE" | tee -a "$MASTER_LOG"

# Run cleanup scripts in order
echo "Starting cleanup sequence..." | tee -a "$MASTER_LOG"

# Script 1: Remove Python cache files
run_script "cleanup_scripts/01_remove_cache_files.sh" "Python Cache Cleanup"

# Script 2: Remove obsolete reports
run_script "cleanup_scripts/02_remove_obsolete_reports.sh" "Obsolete Reports Cleanup"

# Script 3: Remove test artifacts
run_script "cleanup_scripts/03_remove_test_artifacts.sh" "Test Artifacts Cleanup"

# Additional cleanup actions
echo "" | tee -a "$MASTER_LOG"
echo "Running additional cleanup actions..." | tee -a "$MASTER_LOG"

# Remove empty directories
echo "Removing empty directories..." | tee -a "$MASTER_LOG"
find . -type d -empty -not -path "./.git/*" -exec echo "Removing empty directory: {}" \; -delete 2>/dev/null | tee -a "$MASTER_LOG"

# Remove temporary files
echo "Removing temporary files..." | tee -a "$MASTER_LOG"
find . -name "*.tmp" -type f -exec echo "Removing: {}" \; -delete 2>/dev/null | tee -a "$MASTER_LOG"
find . -name "*.temp" -type f -exec echo "Removing: {}" \; -delete 2>/dev/null | tee -a "$MASTER_LOG"
find . -name "*~" -type f -exec echo "Removing: {}" \; -delete 2>/dev/null | tee -a "$MASTER_LOG"

# Remove IDE files
echo "Removing IDE files..." | tee -a "$MASTER_LOG"
find . -name ".vscode/settings.json" -type f -exec echo "Removing: {}" \; -delete 2>/dev/null | tee -a "$MASTER_LOG"
find . -name ".idea" -type d -exec echo "Removing directory: {}" \; -exec rm -rf {} \; 2>/dev/null | tee -a "$MASTER_LOG"

# Post-cleanup repository size
echo "" | tee -a "$MASTER_LOG"
echo "Calculating post-cleanup repository size..." | tee -a "$MASTER_LOG"
POST_SIZE=$(du -sh . 2>/dev/null | cut -f1)
echo "Repository size after cleanup: $POST_SIZE" | tee -a "$MASTER_LOG"

# Summary report
echo "" | tee -a "$MASTER_LOG"
echo "========================================" | tee -a "$MASTER_LOG"
echo "CLEANUP SUMMARY" | tee -a "$MASTER_LOG"
echo "========================================" | tee -a "$MASTER_LOG"
echo "Pre-cleanup size: $PRE_SIZE" | tee -a "$MASTER_LOG"
echo "Post-cleanup size: $POST_SIZE" | tee -a "$MASTER_LOG"
echo "Cleanup completed at: $(date)" | tee -a "$MASTER_LOG"
echo "Master log file: $MASTER_LOG" | tee -a "$MASTER_LOG"

# List all generated log files
echo "" | tee -a "$MASTER_LOG"
echo "Generated log files:" | tee -a "$MASTER_LOG"
ls -la cleanup_scripts/*.log 2>/dev/null | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "✅ Comprehensive repository cleanup completed!" | tee -a "$MASTER_LOG"
echo "Review the log files for detailed information about removed files." | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "⚠️  IMPORTANT: Review the removed files lists before committing changes." | tee -a "$MASTER_LOG"
echo "⚠️  IMPORTANT: Test the application after cleanup to ensure nothing critical was removed." | tee -a "$MASTER_LOG"
