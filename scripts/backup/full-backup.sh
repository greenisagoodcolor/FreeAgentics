#!/bin/bash
# Comprehensive Full Backup Script for FreeAgentics
# Handles database, Redis, configuration, and offsite sync

set -euo pipefail

# Load environment variables
source /etc/freeagentics/backup.env 2>/dev/null || true

# Configuration
BACKUP_ROOT="${BACKUP_ROOT:-/var/backups/freeagentics}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DATE=$(date +"%Y-%m-%d")
LOG_FILE="${BACKUP_ROOT}/logs/full-backup-${TIMESTAMP}.log"

# Database settings
DB_NAME="${POSTGRES_DB:-freeagentics}"
DB_USER="${POSTGRES_USER:-freeagentics}"
DB_HOST="${DB_HOST:-postgres}"
DB_PORT="${DB_PORT:-5432}"

# Redis settings
REDIS_HOST="${REDIS_HOST:-redis}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"

# Retention settings
DB_RETENTION_DAYS="${DB_RETENTION_DAYS:-30}"
REDIS_RETENTION_DAYS="${REDIS_RETENTION_DAYS:-7}"
CONFIG_RETENTION_DAYS="${CONFIG_RETENTION_DAYS:-90}"

# Offsite settings
ENABLE_OFFSITE="${ENABLE_OFFSITE:-true}"
S3_BUCKET="${S3_BUCKET:-freeagentics-backups-prod}"
S3_REGION="${S3_REGION:-us-east-1}"

# Notification settings
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
EMAIL_ALERTS="${EMAIL_ALERTS:-ops@freeagentics.io}"

# Create necessary directories
mkdir -p "$BACKUP_ROOT"/{daily,redis,config,logs,temp}

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $message" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    local message="$1"
    log "ERROR" "$message"
    send_notification "Backup Failed" "$message" "error"
    exit 1
}

# Send notifications
send_notification() {
    local subject="$1"
    local message="$2"
    local status="${3:-info}"

    # Log the notification
    log "INFO" "Notification: $subject - $message"

    # Send to Slack if configured
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        local emoji="ℹ️"
        [[ "$status" == "error" ]] && emoji="❌"
        [[ "$status" == "success" ]] && emoji="✅"
        [[ "$status" == "warning" ]] && emoji="⚠️"

        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$emoji *FreeAgentics Backup* - $subject\\n$message\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || log "WARNING" "Failed to send Slack notification"
    fi

    # Send email if configured
    if command -v mail >/dev/null 2>&1 && [[ -n "$EMAIL_ALERTS" ]]; then
        echo "$message" | mail -s "FreeAgentics Backup: $subject" "$EMAIL_ALERTS" || \
            log "WARNING" "Failed to send email notification"
    fi
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."

    # Check required commands
    local required_commands=("pg_dump" "redis-cli" "gzip" "tar")
    [[ "$ENABLE_OFFSITE" == "true" ]] && required_commands+=("aws")

    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            error_exit "Required command not found: $cmd"
        fi
    done

    # Check disk space
    local available_space
    available_space=$(df -BG "$BACKUP_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ "$available_space" -lt 10 ]]; then
        send_notification "Low Disk Space" "Only ${available_space}GB available in $BACKUP_ROOT" "warning"
    fi

    log "INFO" "Prerequisites check completed"
}

# Backup PostgreSQL database
backup_database() {
    log "INFO" "Starting PostgreSQL backup..."

    local db_backup_file="$BACKUP_ROOT/daily/postgres_${DB_NAME}_${TIMESTAMP}.sql"
    local db_compressed_file="${db_backup_file}.gz"

    # Check database connectivity
    if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; then
        error_exit "Cannot connect to PostgreSQL database"
    fi

    # Create backup with custom format for faster restore
    if PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --verbose \
        --no-password \
        --clean \
        --if-exists \
        --create \
        --format=custom \
        --file="$db_backup_file" 2>> "$LOG_FILE"; then

        log "INFO" "Database backup completed successfully"

        # Compress the backup
        gzip -9 "$db_backup_file"

        # Verify backup
        if [[ -f "$db_compressed_file" ]]; then
            local backup_size
            backup_size=$(du -h "$db_compressed_file" | cut -f1)
            log "INFO" "Database backup size: $backup_size"

            # Quick integrity check
            if ! gzip -t "$db_compressed_file"; then
                error_exit "Database backup integrity check failed"
            fi
        else
            error_exit "Database backup file not found after compression"
        fi
    else
        error_exit "PostgreSQL backup failed"
    fi

    # Create a symlink to latest backup
    ln -sf "$db_compressed_file" "$BACKUP_ROOT/daily/postgres_latest.sql.gz"
}

# Backup Redis data
backup_redis() {
    log "INFO" "Starting Redis backup..."

    local redis_backup_dir="$BACKUP_ROOT/redis/${TIMESTAMP}"
    mkdir -p "$redis_backup_dir"

    # Force a synchronous save
    local redis_cli_cmd="redis-cli -h $REDIS_HOST -p $REDIS_PORT"
    [[ -n "$REDIS_PASSWORD" ]] && redis_cli_cmd="$redis_cli_cmd -a $REDIS_PASSWORD --no-auth-warning"

    if $redis_cli_cmd BGSAVE | grep -q "Background saving started"; then
        log "INFO" "Redis background save initiated"

        # Wait for save to complete
        while [[ "$($redis_cli_cmd LASTSAVE)" == "$($redis_cli_cmd LASTSAVE)" ]]; do
            sleep 1
        done

        # Copy RDB file
        local rdb_location
        rdb_location=$($redis_cli_cmd CONFIG GET dir | tail -1)/$($redis_cli_cmd CONFIG GET dbfilename | tail -1)

        if [[ -f "$rdb_location" ]]; then
            cp "$rdb_location" "$redis_backup_dir/dump.rdb"

            # Also backup AOF if enabled
            local aof_enabled
            aof_enabled=$($redis_cli_cmd CONFIG GET appendonly | tail -1)
            if [[ "$aof_enabled" == "yes" ]]; then
                local aof_location
                aof_location=$($redis_cli_cmd CONFIG GET dir | tail -1)/$($redis_cli_cmd CONFIG GET appendfilename | tail -1)
                [[ -f "$aof_location" ]] && cp "$aof_location" "$redis_backup_dir/appendonly.aof"
            fi

            # Compress Redis backup
            tar -czf "$BACKUP_ROOT/redis/redis_backup_${TIMESTAMP}.tar.gz" -C "$redis_backup_dir" .
            rm -rf "$redis_backup_dir"

            log "INFO" "Redis backup completed successfully"
        else
            log "WARNING" "Redis RDB file not found at expected location"
        fi
    else
        log "WARNING" "Redis backup failed - continuing with other backups"
    fi
}

# Backup configuration files
backup_configuration() {
    log "INFO" "Starting configuration backup..."

    local config_backup_file="$BACKUP_ROOT/config/config_backup_${TIMESTAMP}.tar.gz"
    local temp_config_dir="$BACKUP_ROOT/temp/config_${TIMESTAMP}"

    mkdir -p "$temp_config_dir"

    # List of configuration files and directories to backup
    local config_items=(
        "/etc/freeagentics/"
        "/etc/nginx/sites-available/freeagentics"
        "/etc/letsencrypt/live/"
        "/home/green/FreeAgentics/.env*"
        "/home/green/FreeAgentics/docker-compose*.yml"
        "/home/green/FreeAgentics/nginx/"
    )

    # Copy configuration files
    for item in "${config_items[@]}"; do
        if [[ -e "$item" ]]; then
            # Create directory structure
            local dest_dir="$temp_config_dir/$(dirname "$item")"
            mkdir -p "$dest_dir"
            cp -r "$item" "$dest_dir/" 2>/dev/null || log "WARNING" "Failed to copy $item"
        fi
    done

    # Create encrypted archive
    tar -czf "$config_backup_file" -C "$temp_config_dir" . 2>> "$LOG_FILE"

    # Clean up temp directory
    rm -rf "$temp_config_dir"

    if [[ -f "$config_backup_file" ]]; then
        log "INFO" "Configuration backup completed successfully"
    else
        log "WARNING" "Configuration backup may have failed"
    fi
}

# Sync backups to offsite location
sync_offsite() {
    if [[ "$ENABLE_OFFSITE" != "true" ]]; then
        log "INFO" "Offsite sync disabled - skipping"
        return 0
    fi

    log "INFO" "Starting offsite sync to S3..."

    # Check AWS credentials
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        log "WARNING" "AWS credentials not configured - skipping offsite sync"
        return 1
    fi

    # Sync database backups
    if aws s3 sync "$BACKUP_ROOT/daily/" "s3://$S3_BUCKET/daily/" \
        --exclude "*" \
        --include "postgres_${DB_NAME}_${TIMESTAMP}*" \
        --storage-class STANDARD_IA \
        --region "$S3_REGION" >> "$LOG_FILE" 2>&1; then
        log "INFO" "Database backup synced to S3"
    else
        log "WARNING" "Failed to sync database backup to S3"
    fi

    # Sync Redis backups
    if aws s3 sync "$BACKUP_ROOT/redis/" "s3://$S3_BUCKET/redis/" \
        --exclude "*" \
        --include "redis_backup_${TIMESTAMP}*" \
        --storage-class STANDARD_IA \
        --region "$S3_REGION" >> "$LOG_FILE" 2>&1; then
        log "INFO" "Redis backup synced to S3"
    else
        log "WARNING" "Failed to sync Redis backup to S3"
    fi

    # Sync configuration backups
    if aws s3 sync "$BACKUP_ROOT/config/" "s3://$S3_BUCKET/config/" \
        --exclude "*" \
        --include "config_backup_${TIMESTAMP}*" \
        --storage-class STANDARD_IA \
        --region "$S3_REGION" >> "$LOG_FILE" 2>&1; then
        log "INFO" "Configuration backup synced to S3"
    else
        log "WARNING" "Failed to sync configuration backup to S3"
    fi

    log "INFO" "Offsite sync completed"
}

# Clean up old backups
cleanup_old_backups() {
    log "INFO" "Cleaning up old backups..."

    # Clean database backups
    find "$BACKUP_ROOT/daily" -name "postgres_*.sql.gz" -mtime +$DB_RETENTION_DAYS -delete 2>/dev/null || true

    # Clean Redis backups
    find "$BACKUP_ROOT/redis" -name "redis_backup_*.tar.gz" -mtime +$REDIS_RETENTION_DAYS -delete 2>/dev/null || true

    # Clean configuration backups
    find "$BACKUP_ROOT/config" -name "config_backup_*.tar.gz" -mtime +$CONFIG_RETENTION_DAYS -delete 2>/dev/null || true

    # Clean old logs
    find "$BACKUP_ROOT/logs" -name "*.log" -mtime +30 -delete 2>/dev/null || true

    log "INFO" "Cleanup completed"
}

# Generate backup report
generate_report() {
    log "INFO" "Generating backup report..."

    local report_file="$BACKUP_ROOT/logs/backup-report-${BACKUP_DATE}.txt"

    {
        echo "FreeAgentics Backup Report"
        echo "========================="
        echo "Date: $BACKUP_DATE"
        echo "Timestamp: $TIMESTAMP"
        echo ""
        echo "Backup Summary:"
        echo "--------------"

        # Database backup info
        if [[ -f "$BACKUP_ROOT/daily/postgres_${DB_NAME}_${TIMESTAMP}.sql.gz" ]]; then
            local db_size
            db_size=$(du -h "$BACKUP_ROOT/daily/postgres_${DB_NAME}_${TIMESTAMP}.sql.gz" | cut -f1)
            echo "✓ PostgreSQL Database: $db_size"
        else
            echo "✗ PostgreSQL Database: FAILED"
        fi

        # Redis backup info
        if [[ -f "$BACKUP_ROOT/redis/redis_backup_${TIMESTAMP}.tar.gz" ]]; then
            local redis_size
            redis_size=$(du -h "$BACKUP_ROOT/redis/redis_backup_${TIMESTAMP}.tar.gz" | cut -f1)
            echo "✓ Redis Data: $redis_size"
        else
            echo "✗ Redis Data: FAILED or SKIPPED"
        fi

        # Configuration backup info
        if [[ -f "$BACKUP_ROOT/config/config_backup_${TIMESTAMP}.tar.gz" ]]; then
            local config_size
            config_size=$(du -h "$BACKUP_ROOT/config/config_backup_${TIMESTAMP}.tar.gz" | cut -f1)
            echo "✓ Configuration: $config_size"
        else
            echo "✗ Configuration: FAILED"
        fi

        echo ""
        echo "Storage Usage:"
        echo "-------------"
        df -h "$BACKUP_ROOT" | grep -E "Filesystem|$BACKUP_ROOT"

        echo ""
        echo "Retention Status:"
        echo "----------------"
        echo "Database backups: $(find "$BACKUP_ROOT/daily" -name "postgres_*.sql.gz" -type f | wc -l) files"
        echo "Redis backups: $(find "$BACKUP_ROOT/redis" -name "redis_backup_*.tar.gz" -type f | wc -l) files"
        echo "Config backups: $(find "$BACKUP_ROOT/config" -name "config_backup_*.tar.gz" -type f | wc -l) files"

    } > "$report_file"

    log "INFO" "Report generated: $report_file"
}

# Main execution
main() {
    log "INFO" "===== Starting FreeAgentics Full Backup ====="

    # Lock file to prevent concurrent backups
    local lock_file="/var/run/freeagentics-backup.lock"
    if [[ -f "$lock_file" ]]; then
        error_exit "Another backup is already running (lock file exists)"
    fi

    # Create lock file
    echo $$ > "$lock_file"
    trap "rm -f $lock_file" EXIT

    # Execute backup steps
    check_prerequisites
    backup_database
    backup_redis
    backup_configuration
    sync_offsite
    cleanup_old_backups
    generate_report

    # Send success notification
    send_notification "Backup Completed" "All backup tasks completed successfully for $BACKUP_DATE" "success"

    log "INFO" "===== Backup completed successfully ====="
}

# Run main function
main "$@"
