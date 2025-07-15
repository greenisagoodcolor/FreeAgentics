#!/bin/bash
# Production Database Backup Script for FreeAgentics
# Performs automated backups with rotation and monitoring

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/var/backups/freeagentics}"
DB_NAME="${POSTGRES_DB:-freeagentics}"
DB_USER="${POSTGRES_USER:-freeagentics}"
DB_HOST="${DB_HOST:-postgres}"
DB_PORT="${DB_PORT:-5432}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Timestamp for backup files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/freeagentics_backup_$TIMESTAMP.sql"
COMPRESSED_BACKUP="$BACKUP_FILE.gz"

# Function to send notifications
send_notification() {
    local message="$1"
    local status="$2"
    
    echo "[$(date)] $message"
    
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🗄️ FreeAgentics DB Backup $status: $message\"}" \
            "$SLACK_WEBHOOK" || true
    fi
}

# Function to cleanup old backups
cleanup_old_backups() {
    echo "Cleaning up backups older than $RETENTION_DAYS days..."
    find "$BACKUP_DIR" -name "freeagentics_backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete || true
}

# Function to perform backup
perform_backup() {
    echo "Starting database backup for $DB_NAME..."
    
    # Check if database is accessible
    if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"; then
        send_notification "Database not accessible at $DB_HOST:$DB_PORT" "FAILED"
        exit 1
    fi
    
    # Perform the backup
    if pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        --no-password --verbose --clean --if-exists --create > "$BACKUP_FILE"; then
        
        # Compress the backup
        gzip "$BACKUP_FILE"
        
        # Check backup integrity
        if [[ -f "$COMPRESSED_BACKUP" ]]; then
            BACKUP_SIZE=$(du -h "$COMPRESSED_BACKUP" | cut -f1)
            send_notification "Backup completed successfully. Size: $BACKUP_SIZE" "SUCCESS"
        else
            send_notification "Backup file not found after compression" "FAILED"
            exit 1
        fi
    else
        send_notification "pg_dump failed" "FAILED"
        exit 1
    fi
}

# Function to verify backup
verify_backup() {
    echo "Verifying backup integrity..."
    if gunzip -t "$COMPRESSED_BACKUP"; then
        echo "Backup integrity verified"
        return 0
    else
        send_notification "Backup integrity check failed" "FAILED"
        return 1
    fi
}

# Main execution
main() {
    case "${1:-backup}" in
        "backup")
            perform_backup
            verify_backup
            cleanup_old_backups
            ;;
        "restore")
            if [[ -z "${2:-}" ]]; then
                echo "Usage: $0 restore <backup_file>"
                exit 1
            fi
            RESTORE_FILE="$2"
            if [[ -f "$RESTORE_FILE" ]]; then
                echo "Restoring from $RESTORE_FILE..."
                gunzip -c "$RESTORE_FILE" | psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"
                send_notification "Database restored from $RESTORE_FILE" "SUCCESS"
            else
                echo "Backup file not found: $RESTORE_FILE"
                exit 1
            fi
            ;;
        "cleanup")
            cleanup_old_backups
            ;;
        "verify")
            if [[ -z "${2:-}" ]]; then
                echo "Usage: $0 verify <backup_file>"
                exit 1
            fi
            if gunzip -t "$2"; then
                echo "Backup file $2 is valid"
            else
                echo "Backup file $2 is corrupted"
                exit 1
            fi
            ;;
        *)
            echo "Usage: $0 {backup|restore|cleanup|verify} [file]"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"