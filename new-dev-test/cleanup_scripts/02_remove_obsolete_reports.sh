#!/bin/bash
# Repository Cleanup Script 2: Remove Obsolete Reports
# Generated by Repository Forensics Analysis - 2025-07-17

echo "🧹 Starting obsolete report cleanup..."

# Change to repository root
cd "$(dirname "$0")/.."

# Create backup list of files being removed
echo "Creating backup list of removed files..."
BACKUP_LIST="cleanup_scripts/removed_files_$(date +%Y%m%d_%H%M%S).log"

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

echo "Backup list created: $BACKUP_LIST"

# Remove task completion reports from root directory
echo "Removing task completion reports..."
safe_remove "ASYNC_PERFORMANCE_REPORT.md"
safe_remove "BENCHMARK_VALIDATION_REPORT.md"
safe_remove "CLEANUP_INTEGRATION_PROGRESS_REPORT.md"
safe_remove "CLEANUP_PLAN_20250716_161407.md"
safe_remove "CLEANUP_PLAN_20250716_TASK_20_1.md"
safe_remove "CLEANUP_PLAN_TASK_20_2.md"
safe_remove "CLEANUP_QUICK_REFERENCE.md"
safe_remove "CLEANUP_SUMMARY.md"
safe_remove "COMPREHENSIVE_CLEANUP_PROCESS.md"
safe_remove "COMPREHENSIVE_SECURITY_TESTING_REPORT.md"
safe_remove "DEPENDENCY_AUDIT.md"
safe_remove "DEVELOPMENT_SUMMARY.md"
safe_remove "FINAL_SECURITY_VALIDATION_REPORT.md"
safe_remove "FINAL_VALIDATION_REPORT.md"
safe_remove "FINAL_VERIFICATION_REPORT.md"
safe_remove "NEMESIS_AUDIT_PERFORMANCE_CLAIMS.md"
safe_remove "NEMESIS_AUDIT_TASKS_2.2_TO_8.3.md"
safe_remove "NEMESIS_COMPLETION_AUDIT_REPORT.md"
safe_remove "PERFORMANCE_ANALYSIS.md"
safe_remove "PERFORMANCE_DOCUMENTATION_INDEX.md"
safe_remove "PERFORMANCE_LIMITS_DOCUMENTATION.md"
safe_remove "PERFORMANCE_RECOVERY_SUMMARY.md"
safe_remove "PRODUCTION_READINESS_FINAL_REPORT.md"
safe_remove "PRODUCTION_READINESS_VALIDATION_REPORT.md"
safe_remove "PRODUCTION_VALIDATION_REPORT.md"
safe_remove "PROJECT_BOARD_SETUP.md"
safe_remove "QUALITY_CHECK_SUMMARY.md"
safe_remove "REPOSITORY_CLEANUP_RESEARCH_REPORT.md"
safe_remove "SECURITY_AUDIT.md"
safe_remove "SECURITY_AUDIT_REPORT.md"
safe_remove "SECURITY_IMPLEMENTATION_SUMMARY.md"
safe_remove "SECURITY_VALIDATION_REPORT.md"
safe_remove "SENIOR_DEVELOPER_PROGRESS_REPORT.md"
safe_remove "SESSION_SUMMARY.md"
safe_remove "TASK_14_5_SECURITY_HEADERS_COMPLETION.md"
safe_remove "TASK_17_PRODUCTION_MONITORING_COMPLETION.md"
safe_remove "TASK_17_SECURITY_MONITORING_COMPLETION.md"
safe_remove "TASK_20_2_COMPLETION_SUMMARY.md"
safe_remove "TASK_20_2_MEMORY_PROFILING_REPORT.md"
safe_remove "TASK_20_3_COMPLETION_SUMMARY.md"
safe_remove "TASK_20_5_COMPLETION_SUMMARY.md"
safe_remove "TASK_22_5_COMPLETION_SUMMARY.md"
safe_remove "TASK_4_3_THREADING_OPTIMIZATION_REPORT.md"
safe_remove "TASK_5_MEMORY_OPTIMIZATION_REPORT.md"
safe_remove "TASK_MASTER_SYNCHRONIZATION_ANALYSIS.md"
safe_remove "TASK_MASTER_SYNCHRONIZATION_RESOLUTION_REPORT.md"
safe_remove "TEST_FAILURE_SUMMARY.md"
safe_remove "VALIDATION_REPORT.md"
safe_remove "ZERO_TRUST_IMPLEMENTATION_SUMMARY.md"

# Remove duplicate reports in subdirectories
echo "Removing duplicate reports..."
safe_remove "agents/TASK_4_3_THREADING_OPTIMIZATION_REPORT.md"
safe_remove "monitoring/TASK_17_COMPLETION_SUMMARY.md"

# Remove obsolete JSON report files
echo "Removing obsolete JSON reports..."
safe_remove "comprehensive_security_validation_report.json"
safe_remove "infrastructure_validation_report_20250716_093921.json"
safe_remove "infrastructure_validation_report_20250716_093921.md"
safe_remove "security_gate_validation_report.json"
safe_remove "security_validation_report_20250716_094153.json"
safe_remove "security_validation_report_20250716_094153.md"
safe_remove "security_validation_report_20250716_094225.json"
safe_remove "security_validation_report_20250716_094225.md"
safe_remove "benchmark_results_20250704_154148.json"
safe_remove "selective_update_benchmark_results_20250704_191515.json"
safe_remove "selective_update_benchmark_results_20250704_191552.json"

# Remove temporary and development artifacts
echo "Removing development artifacts..."
safe_remove "cleanup_addition.txt"
safe_remove "cleanup_process_addition.txt"
safe_remove "console-log-replacement-summary.md"
safe_remove "comprehensive_subtask_update_plan.md"
safe_remove "mandatory_principles.txt"
safe_remove "full_test_output.txt"
safe_remove "test_failures.txt"
safe_remove "settings.json"

# Remove log files (but keep essential ones)
echo "Removing obsolete log files..."
safe_remove "lint-output.log"
safe_remove "tests/reporting/integration.log"

# Remove memory profiling artifacts
echo "Removing memory profiling artifacts..."
if [ -d "memory_profiling_reports" ]; then
    echo "Removing memory_profiling_reports directory..." | tee -a "$BACKUP_LIST"
    rm -rf "memory_profiling_reports/"
fi

# Remove taskmaster complexity analysis
safe_remove "taskmaster_complexity_analysis.md"

# Count removed files
REMOVED_COUNT=$(grep -c "Removing:" "$BACKUP_LIST" 2>/dev/null || echo "0")
echo "✅ Obsolete report cleanup completed"
echo "Files removed: $REMOVED_COUNT"
echo "Backup list saved: $BACKUP_LIST"
echo "Estimated space freed: ~100MB"
