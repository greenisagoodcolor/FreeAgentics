#!/bin/bash
# Database Restore Script for FreeAgentics
# Handles full restore, point-in-time recovery, and verification

set -euo pipefail

# Configuration
BACKUP_ROOT="${BACKUP_ROOT:-/var/backups/freeagentics}"
DB_NAME="${POSTGRES_DB:-freeagentics}"
DB_USER="${POSTGRES_USER:-freeagentics}"
DB_HOST="${DB_HOST:-postgres}"
DB_PORT="${DB_PORT:-5432}"
RESTORE_LOG="/var/log/freeagentics/restore-$(date +%Y%m%d-%H%M%S).log"

# Create log directory
mkdir -p "$(dirname "$RESTORE_LOG")"

# Logging function
log() {
    local level="$1"
    shift
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$RESTORE_LOG"
}

# Error handling
error_exit() {
    log "ERROR" "$1"
    exit 1
}

# Display usage
usage() {
    cat << EOF
Usage: $0 <command> [options]

Commands:
    full <backup-file>          Restore from a full backup file
    pitr <timestamp>            Point-in-time recovery to specified timestamp
    latest                      Restore from the latest available backup
    verify                      Verify current database integrity
    list                        List available backups

Options:
    --force                     Skip confirmation prompts
    --no-stop                   Don't stop services (for testing)
    --target-db <name>          Restore to a different database name

Examples:
    $0 full /var/backups/freeagentics/daily/postgres_backup_20240115.sql.gz
    $0 pitr "2024-01-15 14:30:00 UTC"
    $0 latest --force
EOF
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."

    # Check required commands
    for cmd in pg_restore psql pg_isready gunzip; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            error_exit "Required command not found: $cmd"
        fi
    done

    # Check database connectivity
    if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" >/dev/null 2>&1; then
        error_exit "PostgreSQL server is not available"
    fi
}

# List available backups
list_backups() {
    log "INFO" "Available backups:"
    echo
    echo "Local backups:"
    find "$BACKUP_ROOT/daily" -name "postgres_*.sql.gz" -type f -printf "%T+ %p\n" | \
        sort -r | head -20 | while read -r timestamp file; do
        size=$(du -h "$file" | cut -f1)
        echo "  $(basename "$file") - $(date -d "${timestamp%.*}" '+%Y-%m-%d %H:%M:%S') - $size"
    done

    # Check for S3 backups if AWS CLI is available
    if command -v aws >/dev/null 2>&1 && [[ -n "${S3_BUCKET:-}" ]]; then
        echo
        echo "S3 backups:"
        aws s3 ls "s3://${S3_BUCKET}/daily/" --recursive | \
            grep "postgres_.*\.sql\.gz$" | sort -r | head -10
    fi
}

# Get confirmation
get_confirmation() {
    if [[ "${FORCE:-false}" == "true" ]]; then
        return 0
    fi

    echo
    echo "WARNING: This will restore the database and may result in data loss!"
    echo "Current database: $DB_NAME on $DB_HOST:$DB_PORT"
    echo
    read -p "Are you sure you want to proceed? (yes/NO): " confirmation

    if [[ "$confirmation" != "yes" ]]; then
        log "INFO" "Restore cancelled by user"
        exit 0
    fi
}

# Stop application services
stop_services() {
    if [[ "${NO_STOP:-false}" == "true" ]]; then
        log "WARNING" "Skipping service stop (--no-stop flag)"
        return 0
    fi

    log "INFO" "Stopping application services..."

    # Stop Docker services
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose stop backend-prod web worker
    fi

    # Wait for connections to close
    sleep 5
}

# Start application services
start_services() {
    if [[ "${NO_STOP:-false}" == "true" ]]; then
        return 0
    fi

    log "INFO" "Starting application services..."

    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose up -d backend-prod web worker
    fi
}

# Restore from full backup
restore_full() {
    local backup_file="$1"

    if [[ ! -f "$backup_file" ]]; then
        error_exit "Backup file not found: $backup_file"
    fi

    log "INFO" "Restoring from: $backup_file"

    # Create temporary directory for extraction
    local temp_dir="/tmp/freeagentics-restore-$$"
    mkdir -p "$temp_dir"
    trap "rm -rf $temp_dir" EXIT

    # Extract backup if compressed
    local restore_file="$backup_file"
    if [[ "$backup_file" =~ \.gz$ ]]; then
        log "INFO" "Extracting compressed backup..."
        gunzip -c "$backup_file" > "$temp_dir/restore.sql"
        restore_file="$temp_dir/restore.sql"
    fi

    # Get confirmation
    get_confirmation

    # Stop services
    stop_services

    # Create restore point
    log "INFO" "Creating restore point..."
    PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
        -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        --format=custom --file="$BACKUP_ROOT/restore-points/before-restore-$(date +%Y%m%d-%H%M%S).sql" \
        2>/dev/null || log "WARNING" "Failed to create restore point"

    # Drop and recreate database
    log "INFO" "Preparing database for restore..."
    PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres \
        -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$DB_NAME' AND pid <> pg_backend_pid();" \
        >/dev/null 2>&1

    # Perform restore
    log "INFO" "Restoring database..."
    local target_db="${TARGET_DB:-$DB_NAME}"

    if [[ "$restore_file" =~ \.sql$ ]]; then
        # Plain SQL format
        PGPASSWORD="$POSTGRES_PASSWORD" psql \
            -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
            -d postgres -f "$restore_file" >> "$RESTORE_LOG" 2>&1
    else
        # Custom format
        PGPASSWORD="$POSTGRES_PASSWORD" pg_restore \
            -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
            -d postgres --clean --if-exists --create \
            "$restore_file" >> "$RESTORE_LOG" 2>&1
    fi

    if [[ $? -eq 0 ]]; then
        log "INFO" "Database restore completed successfully"
    else
        error_exit "Database restore failed - check $RESTORE_LOG for details"
    fi

    # Start services
    start_services

    # Verify restore
    verify_database
}

# Point-in-time recovery
restore_pitr() {
    local target_time="$1"

    log "INFO" "Performing point-in-time recovery to: $target_time"

    # Validate timestamp format
    if ! date -d "$target_time" >/dev/null 2>&1; then
        error_exit "Invalid timestamp format: $target_time"
    fi

    # Find appropriate base backup
    local base_backup
    base_backup=$(find "$BACKUP_ROOT/daily" -name "postgres_*.sql.gz" \
        -newermt "$(date -d "$target_time - 1 day")" \
        ! -newermt "$target_time" \
        -type f | sort | tail -1)

    if [[ -z "$base_backup" ]]; then
        error_exit "No suitable base backup found for PITR to $target_time"
    fi

    log "INFO" "Using base backup: $base_backup"

    # Get confirmation
    get_confirmation

    # Stop services
    stop_services

    # Restore base backup
    restore_full "$base_backup"

    # Apply WAL logs up to target time
    log "INFO" "Applying WAL logs up to $target_time..."

    # Configure recovery.conf for PITR
    cat > "$PGDATA/recovery.conf" << EOF
restore_command = 'cp /var/backups/freeagentics/wal/%f %p'
recovery_target_time = '$target_time'
recovery_target_action = 'promote'
EOF

    # Restart PostgreSQL to apply WAL logs
    if command -v systemctl >/dev/null 2>&1; then
        systemctl restart postgresql
    else
        pg_ctl restart -D "$PGDATA"
    fi

    # Wait for recovery to complete
    log "INFO" "Waiting for recovery to complete..."
    while [[ -f "$PGDATA/recovery.conf" ]]; do
        sleep 2
    done

    log "INFO" "Point-in-time recovery completed"

    # Start services
    start_services

    # Verify restore
    verify_database
}

# Restore from latest backup
restore_latest() {
    log "INFO" "Finding latest backup..."

    local latest_backup
    latest_backup=$(find "$BACKUP_ROOT/daily" -name "postgres_*.sql.gz" -type f -printf "%T@ %p\n" | \
        sort -rn | head -1 | cut -d' ' -f2)

    if [[ -z "$latest_backup" ]]; then
        error_exit "No backups found in $BACKUP_ROOT/daily"
    fi

    log "INFO" "Latest backup: $latest_backup"
    restore_full "$latest_backup"
}

# Verify database integrity
verify_database() {
    log "INFO" "Verifying database integrity..."

    # Check connection
    if ! PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        -c "SELECT 1;" >/dev/null 2>&1; then
        error_exit "Cannot connect to restored database"
    fi

    # Run basic integrity checks
    local checks_passed=0
    local total_checks=0

    # Check table count
    ((total_checks++))
    local table_count
    table_count=$(PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d ' ')

    if [[ "$table_count" -gt 0 ]]; then
        ((checks_passed++))
        log "INFO" "✓ Found $table_count tables"
    else
        log "ERROR" "✗ No tables found"
    fi

    # Check for required tables
    ((total_checks++))
    local required_tables=("users" "agents" "coalitions" "knowledge_graph")
    local missing_tables=()

    for table in "${required_tables[@]}"; do
        if ! PGPASSWORD="$POSTGRES_PASSWORD" psql \
            -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
            -c "SELECT 1 FROM $table LIMIT 1;" >/dev/null 2>&1; then
            missing_tables+=("$table")
        fi
    done

    if [[ ${#missing_tables[@]} -eq 0 ]]; then
        ((checks_passed++))
        log "INFO" "✓ All required tables present"
    else
        log "ERROR" "✗ Missing tables: ${missing_tables[*]}"
    fi

    # Check constraints
    ((total_checks++))
    local constraint_count
    constraint_count=$(PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        -t -c "SELECT COUNT(*) FROM information_schema.table_constraints WHERE constraint_schema = 'public';" | tr -d ' ')

    if [[ "$constraint_count" -gt 0 ]]; then
        ((checks_passed++))
        log "INFO" "✓ Found $constraint_count constraints"
    else
        log "WARNING" "⚠ No constraints found"
    fi

    # Summary
    log "INFO" "Verification complete: $checks_passed/$total_checks checks passed"

    if [[ "$checks_passed" -eq "$total_checks" ]]; then
        log "INFO" "Database verification successful"
        return 0
    else
        log "WARNING" "Database verification completed with warnings"
        return 1
    fi
}

# Main execution
main() {
    case "${1:-}" in
        "full")
            [[ -z "${2:-}" ]] && { usage; exit 1; }
            check_prerequisites
            restore_full "$2"
            ;;
        "pitr")
            [[ -z "${2:-}" ]] && { usage; exit 1; }
            check_prerequisites
            restore_pitr "$2"
            ;;
        "latest")
            check_prerequisites
            restore_latest
            ;;
        "verify")
            check_prerequisites
            verify_database
            ;;
        "list")
            list_backups
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=true
            shift
            ;;
        --no-stop)
            NO_STOP=true
            shift
            ;;
        --target-db)
            TARGET_DB="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Run main function
main "$@"
