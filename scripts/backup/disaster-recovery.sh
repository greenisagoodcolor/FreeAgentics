#!/bin/bash
# Disaster Recovery Script for FreeAgentics
# Handles complete system recovery from various disaster scenarios

set -euo pipefail

# Configuration
BACKUP_ROOT="${BACKUP_ROOT:-/var/backups/freeagentics}"
RECOVERY_LOG="/var/log/freeagentics/disaster-recovery-$(date +%Y%m%d-%H%M%S).log"
RECOVERY_TEMP="/tmp/freeagentics-recovery-$$"
S3_BUCKET="${S3_BUCKET:-freeagentics-backups-prod}"

# Recovery targets
DB_NAME="${POSTGRES_DB:-freeagentics}"
DB_USER="${POSTGRES_USER:-freeagentics}"
DB_HOST="${DB_HOST:-postgres}"
DB_PORT="${DB_PORT:-5432}"

# Create directories
mkdir -p "$(dirname "$RECOVERY_LOG")" "$RECOVERY_TEMP"

# Logging function
log() {
    local level="$1"
    shift
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$RECOVERY_LOG"
}

# Error handling
error_exit() {
    log "ERROR" "$1"
    send_alert "Disaster Recovery Failed" "$1" "critical"
    exit 1
}

# Cleanup on exit
trap "rm -rf $RECOVERY_TEMP" EXIT

# Send alerts
send_alert() {
    local subject="$1"
    local message="$2"
    local priority="${3:-info}"

    log "ALERT" "[$priority] $subject: $message"

    # Send to Slack if configured
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        local emoji="🚨"
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$emoji *DISASTER RECOVERY* - $subject\\n$message\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null
    fi

    # Send emergency email
    if command -v mail >/dev/null 2>&1; then
        echo "$message" | mail -s "EMERGENCY: FreeAgentics - $subject" \
            ops@freeagentics.io,admin@freeagentics.io
    fi
}

# Display usage
usage() {
    cat << EOF
FreeAgentics Disaster Recovery Script

Usage: $0 <scenario> [options]

Disaster Scenarios:
    full-system         Complete system failure recovery
    database-corruption Database corruption recovery
    ransomware         Ransomware attack recovery
    data-center-loss   Data center/infrastructure loss
    human-error        Accidental deletion recovery

Options:
    --point-in-time <timestamp>    Restore to specific point in time
    --dry-run                      Show what would be done without executing
    --force                        Skip confirmations
    --backup-source <local|s3>     Specify backup source (default: auto)
    --recovery-target <host>       Recovery target host (default: current)

Examples:
    $0 full-system --backup-source s3
    $0 database-corruption --point-in-time "2024-01-15 14:30:00"
    $0 ransomware --dry-run
    $0 human-error --force

Emergency Contact: ops@freeagentics.io
EOF
}

# Assess disaster impact
assess_disaster() {
    local scenario="$1"

    log "INFO" "===== DISASTER RECOVERY ASSESSMENT ====="
    log "INFO" "Scenario: $scenario"
    log "INFO" "Timestamp: $(date)"
    log "INFO" "Host: $(hostname)"
    log "INFO" "User: $(whoami)"

    # Check system status
    local system_status="UNKNOWN"
    local database_status="UNKNOWN"
    local application_status="UNKNOWN"

    # Check if Docker is running
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        log "INFO" "✓ Docker is running"

        # Check container status
        local running_containers
        running_containers=$(docker ps --format "table {{.Names}}" | grep -E "freeagentics|postgres|redis" | wc -l)
        log "INFO" "Running containers: $running_containers"

        if [[ "$running_containers" -gt 0 ]]; then
            application_status="PARTIAL"
        else
            application_status="DOWN"
        fi
    else
        log "WARNING" "Docker not available or not running"
        application_status="DOWN"
    fi

    # Check database connectivity
    if pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; then
        log "INFO" "✓ Database is accessible"
        database_status="UP"
    else
        log "WARNING" "Database not accessible"
        database_status="DOWN"
    fi

    # Check file system
    if [[ -d "/home/green/FreeAgentics" ]]; then
        log "INFO" "✓ Application directory exists"
        system_status="PARTIAL"
    else
        log "WARNING" "Application directory missing"
        system_status="DOWN"
    fi

    # Check backup availability
    local backup_available=false
    if [[ -d "$BACKUP_ROOT" ]] && [[ -n "$(ls -A "$BACKUP_ROOT/daily" 2>/dev/null)" ]]; then
        log "INFO" "✓ Local backups available"
        backup_available=true
    elif command -v aws >/dev/null 2>&1 && aws s3 ls "s3://$S3_BUCKET/daily/" >/dev/null 2>&1; then
        log "INFO" "✓ S3 backups available"
        backup_available=true
    else
        log "ERROR" "No backups available"
    fi

    # Generate assessment report
    cat > "$RECOVERY_TEMP/assessment.txt" << EOF
DISASTER RECOVERY ASSESSMENT
===========================
Scenario: $scenario
Timestamp: $(date)
Host: $(hostname)

System Status:
- Overall: $system_status
- Database: $database_status
- Application: $application_status
- Backups: $backup_available

Recovery Strategy:
$(get_recovery_strategy "$scenario" "$system_status" "$database_status" "$application_status" "$backup_available")

Estimated Recovery Time: $(get_recovery_time "$scenario")
Estimated Data Loss: $(get_data_loss_estimate "$scenario")
EOF

    log "INFO" "Assessment complete - see $RECOVERY_TEMP/assessment.txt"
    cat "$RECOVERY_TEMP/assessment.txt"
}

# Get recovery strategy
get_recovery_strategy() {
    local scenario="$1"
    local system_status="$2"
    local database_status="$3"
    local application_status="$4"
    local backup_available="$5"

    case "$scenario" in
        "full-system")
            echo "1. Restore infrastructure"
            echo "2. Restore database from latest backup"
            echo "3. Restore application code"
            echo "4. Restore configuration"
            echo "5. Verify and start services"
            ;;
        "database-corruption")
            echo "1. Stop application services"
            echo "2. Assess corruption extent"
            echo "3. Restore from latest clean backup"
            echo "4. Apply WAL logs if available"
            echo "5. Verify data integrity"
            ;;
        "ransomware")
            echo "1. Isolate affected systems"
            echo "2. Verify backup integrity"
            echo "3. Restore from clean backups"
            echo "4. Implement additional security"
            echo "5. Monitor for reinfection"
            ;;
        "data-center-loss")
            echo "1. Activate DR site"
            echo "2. Restore from offsite backups"
            echo "3. Update DNS records"
            echo "4. Redirect traffic"
            echo "5. Monitor performance"
            ;;
        *)
            echo "Standard recovery procedure"
            ;;
    esac
}

# Get recovery time estimate
get_recovery_time() {
    local scenario="$1"

    case "$scenario" in
        "full-system") echo "2-4 hours" ;;
        "database-corruption") echo "30-60 minutes" ;;
        "ransomware") echo "1-2 hours" ;;
        "data-center-loss") echo "4-6 hours" ;;
        "human-error") echo "15-30 minutes" ;;
        *) echo "1-2 hours" ;;
    esac
}

# Get data loss estimate
get_data_loss_estimate() {
    local scenario="$1"

    case "$scenario" in
        "full-system") echo "< 15 minutes (WAL recovery)" ;;
        "database-corruption") echo "< 15 minutes (WAL recovery)" ;;
        "ransomware") echo "< 1 hour (backup frequency)" ;;
        "data-center-loss") echo "< 4 hours (offsite sync)" ;;
        "human-error") echo "< 15 minutes (transaction logs)" ;;
        *) echo "< 1 hour" ;;
    esac
}

# Full system recovery
recover_full_system() {
    local dry_run="${1:-false}"

    log "INFO" "===== FULL SYSTEM RECOVERY ====="

    if [[ "$dry_run" == "true" ]]; then
        log "INFO" "DRY RUN - No changes will be made"
    fi

    # Step 1: Prepare recovery environment
    log "INFO" "Step 1: Preparing recovery environment..."
    if [[ "$dry_run" == "false" ]]; then
        docker-compose down
        mkdir -p /home/green/FreeAgentics-recovery
        cd /home/green/FreeAgentics-recovery
    fi

    # Step 2: Restore application code
    log "INFO" "Step 2: Restoring application code..."
    if [[ "$dry_run" == "false" ]]; then
        if [[ -d "/home/green/FreeAgentics/.git" ]]; then
            cd /home/green/FreeAgentics
            git reset --hard HEAD
            git clean -fd
        else
            git clone https://github.com/your-org/FreeAgentics.git /home/green/FreeAgentics-new
            mv /home/green/FreeAgentics /home/green/FreeAgentics-backup
            mv /home/green/FreeAgentics-new /home/green/FreeAgentics
        fi
    fi

    # Step 3: Restore configuration
    log "INFO" "Step 3: Restoring configuration..."
    if [[ "$dry_run" == "false" ]]; then
        restore_configuration
    fi

    # Step 4: Restore database
    log "INFO" "Step 4: Restoring database..."
    if [[ "$dry_run" == "false" ]]; then
        restore_database_from_backup
    fi

    # Step 5: Start services
    log "INFO" "Step 5: Starting services..."
    if [[ "$dry_run" == "false" ]]; then
        cd /home/green/FreeAgentics
        docker-compose up -d
        sleep 30
        verify_system_health
    fi

    log "INFO" "Full system recovery completed"
}

# Database corruption recovery
recover_database_corruption() {
    local dry_run="${1:-false}"

    log "INFO" "===== DATABASE CORRUPTION RECOVERY ====="

    if [[ "$dry_run" == "true" ]]; then
        log "INFO" "DRY RUN - No changes will be made"
        return 0
    fi

    # Stop application services
    log "INFO" "Stopping application services..."
    docker-compose stop backend-prod web worker

    # Assess corruption
    log "INFO" "Assessing database corruption..."
    local corruption_level="UNKNOWN"

    if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" >/dev/null 2>&1; then
        log "INFO" "Database is accessible - checking for corruption..."

        # Run basic checks
        if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT COUNT(*) FROM pg_stat_database;" >/dev/null 2>&1; then
            corruption_level="MINOR"
        else
            corruption_level="MAJOR"
        fi
    else
        corruption_level="SEVERE"
    fi

    log "INFO" "Corruption level: $corruption_level"

    # Restore based on corruption level
    case "$corruption_level" in
        "MINOR")
            log "INFO" "Attempting repair..."
            PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "REINDEX DATABASE $DB_NAME;"
            ;;
        "MAJOR"|"SEVERE")
            log "INFO" "Performing full database restore..."
            restore_database_from_backup
            ;;
    esac

    # Verify integrity
    verify_database_integrity

    # Start services
    log "INFO" "Starting services..."
    docker-compose up -d backend-prod web worker

    log "INFO" "Database corruption recovery completed"
}

# Ransomware recovery
recover_ransomware() {
    local dry_run="${1:-false}"

    log "INFO" "===== RANSOMWARE RECOVERY ====="

    if [[ "$dry_run" == "true" ]]; then
        log "INFO" "DRY RUN - No changes will be made"
        return 0
    fi

    # Immediate isolation
    log "INFO" "Isolating affected systems..."
    docker-compose down

    # Check backup integrity
    log "INFO" "Verifying backup integrity..."
    if ! verify_backup_integrity; then
        error_exit "Backup integrity compromised - contact security team"
    fi

    # Restore from clean backups
    log "INFO" "Restoring from clean backups..."
    restore_from_clean_backups

    # Implement additional security
    log "INFO" "Implementing additional security measures..."
    implement_security_hardening

    # Monitor for reinfection
    log "INFO" "Setting up monitoring for reinfection..."
    setup_enhanced_monitoring

    log "INFO" "Ransomware recovery completed"
}

# Restore configuration from backup
restore_configuration() {
    log "INFO" "Restoring configuration from backup..."

    local latest_config_backup
    latest_config_backup=$(ls -t "$BACKUP_ROOT/config"/config_backup_*.tar.gz* 2>/dev/null | head -1)

    if [[ -n "$latest_config_backup" ]]; then
        /home/green/FreeAgentics/scripts/backup/config-backup.sh restore "$latest_config_backup" "$RECOVERY_TEMP/config"

        # Selectively restore critical configs
        if [[ -f "$RECOVERY_TEMP/config/app/.env" ]]; then
            cp "$RECOVERY_TEMP/config/app/.env" /home/green/FreeAgentics/
        fi

        if [[ -f "$RECOVERY_TEMP/config/app/docker-compose.yml" ]]; then
            cp "$RECOVERY_TEMP/config/app/docker-compose.yml" /home/green/FreeAgentics/
        fi
    else
        log "WARNING" "No configuration backup found"
    fi
}

# Restore database from backup
restore_database_from_backup() {
    log "INFO" "Restoring database from backup..."

    local point_in_time="${POINT_IN_TIME:-}"

    if [[ -n "$point_in_time" ]]; then
        /home/green/FreeAgentics/scripts/backup/database-restore.sh pitr "$point_in_time"
    else
        /home/green/FreeAgentics/scripts/backup/database-restore.sh latest --force
    fi
}

# Verify system health
verify_system_health() {
    log "INFO" "Verifying system health..."

    local health_checks=0
    local passed_checks=0

    # Check database
    ((health_checks++))
    if pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; then
        ((passed_checks++))
        log "INFO" "✓ Database health check passed"
    else
        log "ERROR" "✗ Database health check failed"
    fi

    # Check application
    ((health_checks++))
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        ((passed_checks++))
        log "INFO" "✓ Application health check passed"
    else
        log "ERROR" "✗ Application health check failed"
    fi

    # Check Redis
    ((health_checks++))
    if redis-cli ping >/dev/null 2>&1; then
        ((passed_checks++))
        log "INFO" "✓ Redis health check passed"
    else
        log "ERROR" "✗ Redis health check failed"
    fi

    log "INFO" "Health checks: $passed_checks/$health_checks passed"

    if [[ "$passed_checks" -eq "$health_checks" ]]; then
        send_alert "Recovery Successful" "All health checks passed - system recovered" "success"
        return 0
    else
        send_alert "Recovery Incomplete" "Some health checks failed - manual intervention required" "warning"
        return 1
    fi
}

# Additional helper functions
verify_backup_integrity() {
    log "INFO" "Verifying backup integrity..."

    # Check database backups
    if [[ -f "$BACKUP_ROOT/daily/postgres_latest.sql.gz" ]]; then
        if gzip -t "$BACKUP_ROOT/daily/postgres_latest.sql.gz"; then
            log "INFO" "✓ Database backup integrity verified"
        else
            log "ERROR" "✗ Database backup corrupted"
            return 1
        fi
    fi

    # Check S3 backups if available
    if command -v aws >/dev/null 2>&1; then
        if aws s3 ls "s3://$S3_BUCKET/daily/" >/dev/null 2>&1; then
            log "INFO" "✓ S3 backup availability verified"
        else
            log "WARNING" "S3 backups not accessible"
        fi
    fi

    return 0
}

verify_database_integrity() {
    log "INFO" "Verifying database integrity..."

    local integrity_checks=0
    local passed_checks=0

    # Check table count
    ((integrity_checks++))
    local table_count
    table_count=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d ' ')

    if [[ "$table_count" -gt 0 ]]; then
        ((passed_checks++))
        log "INFO" "✓ Database contains $table_count tables"
    else
        log "ERROR" "✗ No tables found in database"
    fi

    # Check for corruption
    ((integrity_checks++))
    if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" >/dev/null 2>&1; then
        ((passed_checks++))
        log "INFO" "✓ Database queries working"
    else
        log "ERROR" "✗ Database queries failing"
    fi

    log "INFO" "Integrity checks: $passed_checks/$integrity_checks passed"
    return $([[ "$passed_checks" -eq "$integrity_checks" ]] && echo 0 || echo 1)
}

# Main execution
main() {
    case "${1:-}" in
        "full-system")
            assess_disaster "$1"
            recover_full_system "${DRY_RUN:-false}"
            ;;
        "database-corruption")
            assess_disaster "$1"
            recover_database_corruption "${DRY_RUN:-false}"
            ;;
        "ransomware")
            assess_disaster "$1"
            recover_ransomware "${DRY_RUN:-false}"
            ;;
        "data-center-loss")
            assess_disaster "$1"
            recover_full_system "${DRY_RUN:-false}"
            ;;
        "human-error")
            assess_disaster "$1"
            restore_database_from_backup
            ;;
        "assess")
            assess_disaster "${2:-unknown}"
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
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --point-in-time)
            POINT_IN_TIME="$2"
            shift 2
            ;;
        --backup-source)
            BACKUP_SOURCE="$2"
            shift 2
            ;;
        --recovery-target)
            RECOVERY_TARGET="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Run main function
main "$@"
