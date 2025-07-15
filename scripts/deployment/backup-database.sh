#!/bin/bash
# Database backup script for FreeAgentics

set -euo pipefail

# Configuration
ENVIRONMENT="${1:-production}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/freeagentics}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
COMPRESS="${COMPRESS:-true}"
ENCRYPT="${ENCRYPT:-false}"
ENCRYPTION_KEY="${ENCRYPTION_KEY:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging
log() {
    echo -e "${GREEN}[BACKUP]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $*" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $*"
}

# Create backup directory
create_backup_dir() {
    log "Creating backup directory: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    
    # Set proper permissions
    chmod 750 "$BACKUP_DIR"
    
    # Create subdirectories
    mkdir -p "$BACKUP_DIR/daily"
    mkdir -p "$BACKUP_DIR/weekly"
    mkdir -p "$BACKUP_DIR/monthly"
}

# Get database configuration
get_db_config() {
    local env_file="$PROJECT_ROOT/.env.$ENVIRONMENT"
    
    if [[ ! -f "$env_file" ]]; then
        error "Environment file not found: $env_file"
        exit 1
    fi
    
    # Extract database configuration
    DB_HOST=$(grep -E '^DATABASE_HOST=' "$env_file" | cut -d'=' -f2 | tr -d '"' || echo "postgres")
    DB_PORT=$(grep -E '^DATABASE_PORT=' "$env_file" | cut -d'=' -f2 | tr -d '"' || echo "5432")
    DB_NAME=$(grep -E '^DATABASE_NAME=' "$env_file" | cut -d'=' -f2 | tr -d '"' || echo "freeagentics")
    DB_USER=$(grep -E '^DATABASE_USER=' "$env_file" | cut -d'=' -f2 | tr -d '"' || echo "freeagentics")
    
    log "Database configuration:"
    log "  Host: $DB_HOST"
    log "  Port: $DB_PORT"
    log "  Database: $DB_NAME"
    log "  User: $DB_USER"
}

# Create database backup
create_backup() {
    local backup_type="$1"
    local timestamp="$2"
    
    log "Starting $backup_type backup..."
    
    # Generate backup filename
    local backup_filename="freeagentics_${ENVIRONMENT}_${backup_type}_${timestamp}.sql"
    local backup_path="$BACKUP_DIR/$backup_type/$backup_filename"
    
    # Create backup using pg_dump
    log "Running pg_dump..."
    
    if docker ps | grep -q "freeagentics.*postgres"; then
        # Use docker exec if postgres is running in container
        docker exec freeagentics-postgres pg_dump \
            -U "$DB_USER" \
            -d "$DB_NAME" \
            --no-owner \
            --no-privileges \
            --clean \
            --if-exists \
            --verbose \
            > "$backup_path" 2>/dev/null
    else
        # Use direct connection
        PGPASSWORD="$DB_PASSWORD" pg_dump \
            -h "$DB_HOST" \
            -p "$DB_PORT" \
            -U "$DB_USER" \
            -d "$DB_NAME" \
            --no-owner \
            --no-privileges \
            --clean \
            --if-exists \
            --verbose \
            > "$backup_path"
    fi
    
    if [[ $? -eq 0 ]] && [[ -f "$backup_path" ]] && [[ -s "$backup_path" ]]; then
        log "Backup created successfully: $backup_path"
        
        # Get backup size
        local backup_size
        backup_size=$(du -h "$backup_path" | cut -f1)
        log "Backup size: $backup_size"
        
        # Compress backup if enabled
        if [[ "$COMPRESS" == "true" ]]; then
            log "Compressing backup..."
            gzip "$backup_path"
            backup_path="${backup_path}.gz"
            
            local compressed_size
            compressed_size=$(du -h "$backup_path" | cut -f1)
            log "Compressed size: $compressed_size"
        fi
        
        # Encrypt backup if enabled
        if [[ "$ENCRYPT" == "true" ]] && [[ -n "$ENCRYPTION_KEY" ]]; then
            log "Encrypting backup..."
            openssl enc -aes-256-cbc -salt -in "$backup_path" -out "${backup_path}.enc" -pass pass:"$ENCRYPTION_KEY"
            rm "$backup_path"
            backup_path="${backup_path}.enc"
            log "Backup encrypted: $backup_path"
        fi
        
        # Set proper permissions
        chmod 600 "$backup_path"
        
        # Verify backup integrity
        verify_backup "$backup_path"
        
        echo "$backup_path"
    else
        error "Backup creation failed!"
        exit 1
    fi
}

# Verify backup integrity
verify_backup() {
    local backup_path="$1"
    
    log "Verifying backup integrity..."
    
    if [[ "$backup_path" == *.gz ]]; then
        # Verify gzip integrity
        if gzip -t "$backup_path"; then
            log "✓ Backup compression integrity verified"
        else
            error "✗ Backup compression integrity failed"
            return 1
        fi
    elif [[ "$backup_path" == *.enc ]]; then
        # For encrypted files, we can't easily verify SQL content
        # Just check if file exists and has content
        if [[ -f "$backup_path" ]] && [[ -s "$backup_path" ]]; then
            log "✓ Encrypted backup file exists and has content"
        else
            error "✗ Encrypted backup file validation failed"
            return 1
        fi
    else
        # Verify SQL syntax
        if head -20 "$backup_path" | grep -q "PostgreSQL database dump"; then
            log "✓ Backup SQL format verified"
        else
            error "✗ Backup SQL format verification failed"
            return 1
        fi
    fi
}

# Clean old backups
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    # Clean daily backups older than retention period
    find "$BACKUP_DIR/daily" -name "*.sql*" -mtime +$RETENTION_DAYS -delete
    
    # Clean weekly backups older than 4 weeks
    find "$BACKUP_DIR/weekly" -name "*.sql*" -mtime +28 -delete
    
    # Clean monthly backups older than 12 months
    find "$BACKUP_DIR/monthly" -name "*.sql*" -mtime +365 -delete
    
    log "Backup cleanup completed"
}

# Upload backup to cloud storage
upload_to_cloud() {
    local backup_path="$1"
    
    if [[ -z "${AWS_S3_BUCKET:-}" ]]; then
        warning "AWS_S3_BUCKET not set, skipping cloud upload"
        return
    fi
    
    log "Uploading backup to S3..."
    
    local s3_key="backups/freeagentics/$(basename "$backup_path")"
    
    if aws s3 cp "$backup_path" "s3://$AWS_S3_BUCKET/$s3_key"; then
        log "✓ Backup uploaded to S3: s3://$AWS_S3_BUCKET/$s3_key"
    else
        error "✗ Failed to upload backup to S3"
    fi
}

# Send notifications
send_notifications() {
    local status="$1"
    local backup_path="$2"
    local backup_size="$3"
    
    # Slack notification
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        local color
        if [[ "$status" == "success" ]]; then
            color="good"
        else
            color="danger"
        fi
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"Database Backup $status\",
                    \"fields\": [
                        {\"title\": \"Environment\", \"value\": \"$ENVIRONMENT\", \"short\": true},
                        {\"title\": \"Size\", \"value\": \"$backup_size\", \"short\": true},
                        {\"title\": \"Path\", \"value\": \"$backup_path\", \"short\": false}
                    ],
                    \"footer\": \"FreeAgentics Backup System\",
                    \"ts\": $(date +%s)
                }]
            }" \
            "$SLACK_WEBHOOK" || true
    fi
    
    # Email notification
    if [[ -n "${BACKUP_EMAIL:-}" ]]; then
        local subject="FreeAgentics Database Backup $status"
        local body="Environment: $ENVIRONMENT\nBackup Path: $backup_path\nSize: $backup_size\nTimestamp: $(date)"
        
        echo -e "$body" | mail -s "$subject" "$BACKUP_EMAIL" || true
    fi
}

# Restore from backup
restore_backup() {
    local backup_path="$1"
    local target_db="${2:-${DB_NAME}_restore}"
    
    log "Restoring from backup: $backup_path"
    
    if [[ ! -f "$backup_path" ]]; then
        error "Backup file not found: $backup_path"
        exit 1
    fi
    
    # Prepare backup file for restoration
    local temp_file="/tmp/restore_$(date +%s).sql"
    
    if [[ "$backup_path" == *.gz ]]; then
        gunzip -c "$backup_path" > "$temp_file"
    elif [[ "$backup_path" == *.enc ]]; then
        if [[ -z "$ENCRYPTION_KEY" ]]; then
            error "Encryption key required for encrypted backup"
            exit 1
        fi
        openssl enc -aes-256-cbc -d -in "$backup_path" -out "$temp_file" -pass pass:"$ENCRYPTION_KEY"
    else
        cp "$backup_path" "$temp_file"
    fi
    
    # Create target database
    log "Creating target database: $target_db"
    
    if docker ps | grep -q "freeagentics.*postgres"; then
        docker exec freeagentics-postgres createdb -U "$DB_USER" "$target_db" || true
        docker exec -i freeagentics-postgres psql -U "$DB_USER" -d "$target_db" < "$temp_file"
    else
        PGPASSWORD="$DB_PASSWORD" createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$target_db" || true
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$target_db" < "$temp_file"
    fi
    
    if [[ $? -eq 0 ]]; then
        log "✓ Backup restored successfully to database: $target_db"
    else
        error "✗ Backup restoration failed"
        exit 1
    fi
    
    # Clean up temporary file
    rm -f "$temp_file"
}

# Main execution
main() {
    local action="${2:-backup}"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    case "$action" in
        "backup")
            log "Starting database backup process for $ENVIRONMENT"
            
            create_backup_dir
            get_db_config
            
            # Determine backup type based on day
            local backup_type="daily"
            if [[ $(date +%u) -eq 7 ]]; then
                backup_type="weekly"
            elif [[ $(date +%d) -eq 01 ]]; then
                backup_type="monthly"
            fi
            
            # Create backup
            local backup_path
            backup_path=$(create_backup "$backup_type" "$timestamp")
            
            # Get backup size
            local backup_size
            backup_size=$(du -h "$backup_path" | cut -f1)
            
            # Upload to cloud if configured
            upload_to_cloud "$backup_path"
            
            # Clean old backups
            cleanup_old_backups
            
            # Send notifications
            send_notifications "success" "$backup_path" "$backup_size"
            
            log "Backup process completed successfully"
            log "Backup location: $backup_path"
            log "Backup size: $backup_size"
            ;;
            
        "restore")
            local backup_file="$3"
            local target_db="$4"
            
            if [[ -z "$backup_file" ]]; then
                error "Backup file path required for restore"
                exit 1
            fi
            
            get_db_config
            restore_backup "$backup_file" "$target_db"
            ;;
            
        "list")
            log "Available backups:"
            find "$BACKUP_DIR" -name "*.sql*" -type f -exec ls -lh {} \; | sort -k6,7
            ;;
            
        *)
            echo "Usage: $0 [environment] [action] [options]"
            echo "Actions:"
            echo "  backup          - Create database backup (default)"
            echo "  restore <file>  - Restore from backup file"
            echo "  list            - List available backups"
            exit 1
            ;;
    esac
}

# Handle command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --retention-days)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        --no-compress)
            COMPRESS="false"
            shift
            ;;
        --encrypt)
            ENCRYPT="true"
            shift
            ;;
        --encryption-key)
            ENCRYPTION_KEY="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Run main function
main "$@"