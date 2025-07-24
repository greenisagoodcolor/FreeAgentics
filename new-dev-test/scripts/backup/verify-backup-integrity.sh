#!/bin/bash
# FreeAgentics Backup Integrity Verification Script
# Performs comprehensive backup verification and testing

set -euo pipefail

# Configuration
BACKUP_ROOT="${BACKUP_ROOT:-/var/backups/freeagentics}"
VERIFICATION_DIR="$BACKUP_ROOT/verification"
LOG_FILE="$BACKUP_ROOT/logs/verification_$(date +%Y%m%d_%H%M%S).log"
TEMP_DIR="/tmp/freeagentics-verify-$$"

# Create necessary directories
mkdir -p "$VERIFICATION_DIR" "$(dirname "$LOG_FILE")" "$TEMP_DIR"

# Cleanup on exit
trap "rm -rf $TEMP_DIR" EXIT

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Display usage
usage() {
    cat << EOF
FreeAgentics Backup Integrity Verification

Usage: $0 [options]

Options:
    --backup-id <ID>           Verify specific backup
    --latest                   Verify latest backup (default)
    --all                      Verify all backups
    --deep                     Perform deep verification (test restore)
    --type <type>              Verify specific backup type (database/redis/config/kg/app_state)
    --checksum-only            Only verify checksums
    --restore-test             Perform restore test
    --report                   Generate detailed report
    --help                     Display this help

Examples:
    $0 --latest --deep
    $0 --backup-id full_20240718_020000
    $0 --all --checksum-only
    $0 --type database --restore-test

EOF
}

# Parse command line arguments
BACKUP_ID=""
VERIFY_MODE="latest"
DEEP_VERIFY=false
BACKUP_TYPE=""
CHECKSUM_ONLY=false
RESTORE_TEST=false
GENERATE_REPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --backup-id)
            BACKUP_ID="$2"
            VERIFY_MODE="specific"
            shift 2
            ;;
        --latest)
            VERIFY_MODE="latest"
            shift
            ;;
        --all)
            VERIFY_MODE="all"
            shift
            ;;
        --deep)
            DEEP_VERIFY=true
            shift
            ;;
        --type)
            BACKUP_TYPE="$2"
            shift 2
            ;;
        --checksum-only)
            CHECKSUM_ONLY=true
            shift
            ;;
        --restore-test)
            RESTORE_TEST=true
            shift
            ;;
        --report)
            GENERATE_REPORT=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            error_exit "Unknown option: $1"
            ;;
    esac
done

# Find backups to verify
find_backups() {
    local backups=()

    case "$VERIFY_MODE" in
        specific)
            if [[ -z "$BACKUP_ID" ]]; then
                error_exit "Backup ID required for specific mode"
            fi
            backups+=("$BACKUP_ID")
            ;;
        latest)
            # Find latest backup metadata
            local latest_metadata=$(ls -t "$BACKUP_ROOT/metadata"/*.json 2>/dev/null | head -1)
            if [[ -n "$latest_metadata" ]]; then
                BACKUP_ID=$(basename "$latest_metadata" .json)
                backups+=("$BACKUP_ID")
            else
                error_exit "No backup metadata found"
            fi
            ;;
        all)
            # Find all backup metadata files
            for metadata in "$BACKUP_ROOT/metadata"/*.json; do
                if [[ -f "$metadata" ]]; then
                    backups+=("$(basename "$metadata" .json)")
                fi
            done
            ;;
    esac

    echo "${backups[@]}"
}

# Verify backup checksums
verify_checksums() {
    local backup_id="$1"
    local metadata_file="$BACKUP_ROOT/metadata/${backup_id}.json"

    log "Verifying checksums for backup: $backup_id"

    if [[ ! -f "$metadata_file" ]]; then
        log "ERROR: Metadata file not found: $metadata_file"
        return 1
    fi

    local checksum_failures=0
    local total_files=0

    # Read metadata
    local backup_data=$(cat "$metadata_file")

    # Extract file list and checksums
    echo "$backup_data" | jq -r '.files[]' | while read -r file_path; do
        ((total_files++))

        if [[ -f "$file_path" ]]; then
            # Get expected checksum from metadata
            local expected_checksum=$(echo "$backup_data" | jq -r --arg file "$file_path" '.checksums[$file]')

            if [[ -n "$expected_checksum" && "$expected_checksum" != "null" ]]; then
                # Calculate actual checksum
                local actual_checksum=$(sha256sum "$file_path" | cut -d' ' -f1)

                if [[ "$actual_checksum" != "$expected_checksum" ]]; then
                    log "ERROR: Checksum mismatch for $file_path"
                    log "  Expected: $expected_checksum"
                    log "  Actual:   $actual_checksum"
                    ((checksum_failures++))
                else
                    log "OK: Checksum verified for $(basename "$file_path")"
                fi
            else
                log "WARNING: No checksum found for $file_path"
            fi
        else
            log "ERROR: File not found: $file_path"
            ((checksum_failures++))
        fi
    done

    if [[ $checksum_failures -eq 0 ]]; then
        log "SUCCESS: All checksums verified for backup $backup_id"
        return 0
    else
        log "ERROR: $checksum_failures checksum failures out of $total_files files"
        return 1
    fi
}

# Verify database backup
verify_database_backup() {
    local backup_file="$1"

    log "Verifying database backup: $backup_file"

    # Check if file exists and is readable
    if [[ ! -r "$backup_file" ]]; then
        log "ERROR: Database backup file not readable: $backup_file"
        return 1
    fi

    # Check file integrity
    if [[ "$backup_file" == *.gz ]]; then
        if ! gzip -t "$backup_file" 2>/dev/null; then
            log "ERROR: Database backup file is corrupted (gzip test failed)"
            return 1
        fi
    fi

    # If deep verify, test restore
    if [[ "$DEEP_VERIFY" == true ]] || [[ "$RESTORE_TEST" == true ]]; then
        log "Performing database restore test..."

        # Create test database
        local test_db="freeagentics_verify_$(date +%s)"

        if createdb -h localhost -U postgres "$test_db" 2>/dev/null; then
            # Attempt restore
            if [[ "$backup_file" == *.gz ]]; then
                if gunzip -c "$backup_file" | pg_restore -d "$test_db" -v 2>&1 | grep -q "ERROR"; then
                    log "ERROR: Database restore test failed"
                    dropdb -h localhost -U postgres "$test_db" 2>/dev/null
                    return 1
                fi
            else
                if ! pg_restore -d "$test_db" -v "$backup_file" 2>&1 | grep -q "ERROR"; then
                    log "ERROR: Database restore test failed"
                    dropdb -h localhost -U postgres "$test_db" 2>/dev/null
                    return 1
                fi
            fi

            # Run basic queries
            if psql -h localhost -U postgres -d "$test_db" -c "SELECT COUNT(*) FROM agents;" >/dev/null 2>&1; then
                log "SUCCESS: Database restore test passed"
                dropdb -h localhost -U postgres "$test_db" 2>/dev/null
                return 0
            else
                log "ERROR: Database query test failed"
                dropdb -h localhost -U postgres "$test_db" 2>/dev/null
                return 1
            fi
        else
            log "WARNING: Could not create test database (missing permissions?)"
        fi
    fi

    log "SUCCESS: Database backup verification completed"
    return 0
}

# Verify Redis backup
verify_redis_backup() {
    local backup_file="$1"

    log "Verifying Redis backup: $backup_file"

    # Check file integrity
    if ! tar -tzf "$backup_file" >/dev/null 2>&1; then
        log "ERROR: Redis backup file is corrupted"
        return 1
    fi

    # Extract and verify RDB file
    if [[ "$DEEP_VERIFY" == true ]]; then
        local extract_dir="$TEMP_DIR/redis_verify"
        mkdir -p "$extract_dir"

        if tar -xzf "$backup_file" -C "$extract_dir"; then
            if [[ -f "$extract_dir/dump.rdb" ]]; then
                # Use redis-check-rdb if available
                if command -v redis-check-rdb >/dev/null 2>&1; then
                    if redis-check-rdb "$extract_dir/dump.rdb" 2>&1 | grep -q "OK"; then
                        log "SUCCESS: Redis RDB file is valid"
                        return 0
                    else
                        log "ERROR: Redis RDB file validation failed"
                        return 1
                    fi
                else
                    log "WARNING: redis-check-rdb not available, skipping deep verification"
                fi
            fi
        fi
    fi

    log "SUCCESS: Redis backup verification completed"
    return 0
}

# Verify configuration backup
verify_config_backup() {
    local backup_file="$1"

    log "Verifying configuration backup: $backup_file"

    # Check file integrity
    if ! tar -tzf "$backup_file" >/dev/null 2>&1; then
        log "ERROR: Configuration backup file is corrupted"
        return 1
    fi

    # List contents
    local file_count=$(tar -tzf "$backup_file" | wc -l)
    log "Configuration backup contains $file_count files"

    # Verify critical files exist
    local critical_files=(
        ".env"
        "docker-compose.yml"
        "nginx/"
    )

    for file in "${critical_files[@]}"; do
        if tar -tzf "$backup_file" | grep -q "$file"; then
            log "OK: Critical file found: $file"
        else
            log "WARNING: Critical file missing: $file"
        fi
    done

    log "SUCCESS: Configuration backup verification completed"
    return 0
}

# Verify knowledge graph backup
verify_knowledge_graph_backup() {
    local backup_file="$1"

    log "Verifying knowledge graph backup: $backup_file"

    # Check file integrity
    if [[ "$backup_file" == *.gz ]]; then
        if ! gzip -t "$backup_file" 2>/dev/null; then
            log "ERROR: Knowledge graph backup file is corrupted"
            return 1
        fi

        # Check SQL content
        if [[ "$DEEP_VERIFY" == true ]]; then
            if gunzip -c "$backup_file" | grep -q "knowledge_graph_nodes"; then
                log "OK: Knowledge graph nodes table found"
            else
                log "ERROR: Knowledge graph nodes table not found in backup"
                return 1
            fi

            if gunzip -c "$backup_file" | grep -q "knowledge_graph_edges"; then
                log "OK: Knowledge graph edges table found"
            else
                log "ERROR: Knowledge graph edges table not found in backup"
                return 1
            fi
        fi
    fi

    log "SUCCESS: Knowledge graph backup verification completed"
    return 0
}

# Verify application state backup
verify_app_state_backup() {
    local backup_file="$1"

    log "Verifying application state backup: $backup_file"

    # Check file integrity
    if ! tar -tzf "$backup_file" >/dev/null 2>&1; then
        log "ERROR: Application state backup file is corrupted"
        return 1
    fi

    # Extract and verify JSON
    if [[ "$DEEP_VERIFY" == true ]]; then
        local extract_dir="$TEMP_DIR/app_state_verify"
        mkdir -p "$extract_dir"

        if tar -xzf "$backup_file" -C "$extract_dir"; then
            local state_file=$(find "$extract_dir" -name "state.json" | head -1)

            if [[ -f "$state_file" ]]; then
                if jq empty "$state_file" 2>/dev/null; then
                    log "OK: Application state JSON is valid"

                    # Check for required fields
                    local agents_count=$(jq '.agents | length' "$state_file")
                    local coalitions_count=$(jq '.coalitions | length' "$state_file")

                    log "Application state contains $agents_count agents and $coalitions_count coalitions"
                else
                    log "ERROR: Application state JSON is invalid"
                    return 1
                fi
            else
                log "ERROR: state.json not found in backup"
                return 1
            fi
        fi
    fi

    log "SUCCESS: Application state backup verification completed"
    return 0
}

# Verify single backup
verify_backup() {
    local backup_id="$1"
    local metadata_file="$BACKUP_ROOT/metadata/${backup_id}.json"

    log "===== Verifying backup: $backup_id ====="

    # Check metadata exists
    if [[ ! -f "$metadata_file" ]]; then
        log "ERROR: Metadata file not found for backup $backup_id"
        return 1
    fi

    # Read metadata
    local backup_type=$(jq -r '.backup_type' "$metadata_file")
    local status=$(jq -r '.status' "$metadata_file")
    local timestamp=$(jq -r '.timestamp' "$metadata_file")
    local size_bytes=$(jq -r '.size_bytes' "$metadata_file")

    log "Backup Type: $backup_type"
    log "Status: $status"
    log "Timestamp: $timestamp"
    log "Size: $(numfmt --to=iec-i --suffix=B "$size_bytes" 2>/dev/null || echo "$size_bytes bytes")"

    # Verify checksums
    if ! verify_checksums "$backup_id"; then
        return 1
    fi

    # Skip further verification if checksum-only mode
    if [[ "$CHECKSUM_ONLY" == true ]]; then
        return 0
    fi

    # Verify each backup file
    local verification_failed=false

    jq -r '.files[]' "$metadata_file" | while read -r file_path; do
        if [[ -f "$file_path" ]]; then
            # Determine file type and verify accordingly
            case "$file_path" in
                */postgres_*.gz)
                    if [[ -z "$BACKUP_TYPE" ]] || [[ "$BACKUP_TYPE" == "database" ]]; then
                        verify_database_backup "$file_path" || verification_failed=true
                    fi
                    ;;
                */redis_*.tar.gz)
                    if [[ -z "$BACKUP_TYPE" ]] || [[ "$BACKUP_TYPE" == "redis" ]]; then
                        verify_redis_backup "$file_path" || verification_failed=true
                    fi
                    ;;
                */config_*.tar.gz)
                    if [[ -z "$BACKUP_TYPE" ]] || [[ "$BACKUP_TYPE" == "config" ]]; then
                        verify_config_backup "$file_path" || verification_failed=true
                    fi
                    ;;
                */kg_*.sql.gz)
                    if [[ -z "$BACKUP_TYPE" ]] || [[ "$BACKUP_TYPE" == "kg" ]]; then
                        verify_knowledge_graph_backup "$file_path" || verification_failed=true
                    fi
                    ;;
                */state_*.tar.gz)
                    if [[ -z "$BACKUP_TYPE" ]] || [[ "$BACKUP_TYPE" == "app_state" ]]; then
                        verify_app_state_backup "$file_path" || verification_failed=true
                    fi
                    ;;
            esac
        fi
    done

    if [[ "$verification_failed" == true ]]; then
        log "ERROR: Backup verification failed for $backup_id"
        return 1
    else
        log "SUCCESS: Backup verification completed for $backup_id"
        return 0
    fi
}

# Generate verification report
generate_report() {
    local report_file="$VERIFICATION_DIR/verification_report_$(date +%Y%m%d_%H%M%S).html"

    log "Generating verification report: $report_file"

    cat > "$report_file" <<HTML
<!DOCTYPE html>
<html>
<head>
    <title>FreeAgentics Backup Verification Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .success { color: green; }
        .error { color: red; }
        .warning { color: orange; }
    </style>
</head>
<body>
    <h1>FreeAgentics Backup Verification Report</h1>
    <p>Generated: $(date)</p>

    <h2>Verification Summary</h2>
    <table>
        <tr>
            <th>Backup ID</th>
            <th>Type</th>
            <th>Status</th>
            <th>Size</th>
            <th>Timestamp</th>
            <th>Verification</th>
        </tr>
HTML

    # Add verification results
    for metadata_file in "$BACKUP_ROOT/metadata"/*.json; do
        if [[ -f "$metadata_file" ]]; then
            local backup_id=$(basename "$metadata_file" .json)
            local backup_type=$(jq -r '.backup_type' "$metadata_file")
            local status=$(jq -r '.status' "$metadata_file")
            local size_bytes=$(jq -r '.size_bytes' "$metadata_file")
            local timestamp=$(jq -r '.timestamp' "$metadata_file")

            # Check if verified
            local verification_status="Not Verified"
            local verification_class="warning"

            if grep -q "$backup_id" "$LOG_FILE" 2>/dev/null; then
                if grep -q "SUCCESS.*$backup_id" "$LOG_FILE"; then
                    verification_status="Passed"
                    verification_class="success"
                elif grep -q "ERROR.*$backup_id" "$LOG_FILE"; then
                    verification_status="Failed"
                    verification_class="error"
                fi
            fi

            cat >> "$report_file" <<HTML
        <tr>
            <td>$backup_id</td>
            <td>$backup_type</td>
            <td>$status</td>
            <td>$(numfmt --to=iec-i --suffix=B "$size_bytes" 2>/dev/null || echo "$size_bytes")</td>
            <td>$timestamp</td>
            <td class="$verification_class">$verification_status</td>
        </tr>
HTML
        fi
    done

    cat >> "$report_file" <<HTML
    </table>

    <h2>Storage Usage</h2>
    <pre>
$(df -h "$BACKUP_ROOT")
    </pre>

    <h2>Verification Log</h2>
    <pre>
$(tail -n 100 "$LOG_FILE")
    </pre>
</body>
</html>
HTML

    log "Report generated: $report_file"
}

# Main execution
main() {
    log "Starting backup verification process..."

    # Find backups to verify
    backups=($(find_backups))

    if [[ ${#backups[@]} -eq 0 ]]; then
        error_exit "No backups found to verify"
    fi

    log "Found ${#backups[@]} backup(s) to verify"

    # Verify each backup
    local total_verified=0
    local total_failed=0

    for backup_id in "${backups[@]}"; do
        if verify_backup "$backup_id"; then
            ((total_verified++))
        else
            ((total_failed++))
        fi

        echo # Blank line between backups
    done

    # Summary
    log "===== Verification Summary ====="
    log "Total backups: ${#backups[@]}"
    log "Verified successfully: $total_verified"
    log "Failed verification: $total_failed"

    # Generate report if requested
    if [[ "$GENERATE_REPORT" == true ]]; then
        generate_report
    fi

    # Exit with appropriate code
    if [[ $total_failed -gt 0 ]]; then
        exit 1
    else
        exit 0
    fi
}

# Run main function
main
