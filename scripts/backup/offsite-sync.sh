#!/bin/bash
# Offsite Backup Sync Script for FreeAgentics
# Syncs local backups to AWS S3 with lifecycle management

set -euo pipefail

# Configuration
BACKUP_ROOT="${BACKUP_ROOT:-/var/backups/freeagentics}"
S3_BUCKET="${S3_BUCKET:-freeagentics-backups-prod}"
S3_REGION="${S3_REGION:-us-east-1}"
LOG_FILE="${BACKUP_ROOT}/logs/offsite-sync-$(date +%Y%m%d-%H%M%S).log"

# Retention settings for S3
S3_STANDARD_DAYS="${S3_STANDARD_DAYS:-7}"
S3_IA_DAYS="${S3_IA_DAYS:-30}"
S3_GLACIER_DAYS="${S3_GLACIER_DAYS:-90}"
S3_DELETE_DAYS="${S3_DELETE_DAYS:-365}"

# Notification settings
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
BANDWIDTH_LIMIT="${BANDWIDTH_LIMIT:-50M}"  # Limit upload bandwidth

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    local level="$1"
    shift
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR" "$1"
    send_notification "Offsite Sync Failed" "$1" "error"
    exit 1
}

# Send notifications
send_notification() {
    local subject="$1"
    local message="$2"
    local status="${3:-info}"
    
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        local emoji="ℹ️"
        [[ "$status" == "error" ]] && emoji="❌"
        [[ "$status" == "success" ]] && emoji="✅"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$emoji *S3 Sync* - $subject\\n$message\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws >/dev/null 2>&1; then
        error_exit "AWS CLI not installed"
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        error_exit "AWS credentials not configured or invalid"
    fi
    
    # Check S3 bucket access
    if ! aws s3 ls "s3://$S3_BUCKET" >/dev/null 2>&1; then
        error_exit "Cannot access S3 bucket: $S3_BUCKET"
    fi
    
    log "INFO" "Prerequisites check passed"
}

# Setup S3 lifecycle policy
setup_lifecycle_policy() {
    log "INFO" "Setting up S3 lifecycle policy..."
    
    local policy_file="/tmp/s3-lifecycle-policy-$$.json"
    
    cat > "$policy_file" << EOF
{
    "Rules": [
        {
            "ID": "BackupLifecycle",
            "Status": "Enabled",
            "Prefix": "",
            "Transitions": [
                {
                    "Days": $S3_STANDARD_DAYS,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "Days": $S3_IA_DAYS,
                    "StorageClass": "GLACIER"
                }
            ],
            "Expiration": {
                "Days": $S3_DELETE_DAYS
            }
        }
    ]
}
EOF
    
    if aws s3api put-bucket-lifecycle-configuration \
        --bucket "$S3_BUCKET" \
        --lifecycle-configuration "file://$policy_file" \
        --region "$S3_REGION" 2>/dev/null; then
        log "INFO" "Lifecycle policy updated successfully"
    else
        log "WARNING" "Failed to update lifecycle policy"
    fi
    
    rm -f "$policy_file"
}

# Sync database backups
sync_database_backups() {
    log "INFO" "Syncing database backups to S3..."
    
    local sync_count=0
    local total_size=0
    
    # Get list of local files not in S3
    while IFS= read -r local_file; do
        local filename=$(basename "$local_file")
        local s3_key="daily/$filename"
        
        # Check if file exists in S3
        if ! aws s3 ls "s3://$S3_BUCKET/$s3_key" >/dev/null 2>&1; then
            log "INFO" "Uploading: $filename"
            
            if aws s3 cp "$local_file" "s3://$S3_BUCKET/$s3_key" \
                --storage-class STANDARD \
                --metadata "backup-date=$(date -r "$local_file" +%Y-%m-%d)" \
                --bandwidth "$BANDWIDTH_LIMIT" \
                >> "$LOG_FILE" 2>&1; then
                ((sync_count++))
                total_size=$((total_size + $(stat -c%s "$local_file")))
                log "INFO" "✓ Uploaded: $filename"
            else
                log "ERROR" "✗ Failed to upload: $filename"
            fi
        fi
    done < <(find "$BACKUP_ROOT/daily" -name "postgres_*.sql.gz" -mtime -7 -type f)
    
    log "INFO" "Database sync complete: $sync_count files, $(numfmt --to=iec $total_size)"
}

# Sync Redis backups
sync_redis_backups() {
    log "INFO" "Syncing Redis backups to S3..."
    
    aws s3 sync "$BACKUP_ROOT/redis" "s3://$S3_BUCKET/redis/" \
        --exclude "*" \
        --include "redis_backup_*.tar.gz" \
        --storage-class STANDARD_IA \
        --delete \
        --bandwidth "$BANDWIDTH_LIMIT" \
        >> "$LOG_FILE" 2>&1 || log "WARNING" "Redis sync encountered errors"
}

# Sync configuration backups
sync_config_backups() {
    log "INFO" "Syncing configuration backups to S3..."
    
    # Encrypt sensitive configuration backups
    aws s3 sync "$BACKUP_ROOT/config" "s3://$S3_BUCKET/config/" \
        --exclude "*" \
        --include "config_backup_*.tar.gz" \
        --storage-class STANDARD \
        --sse AES256 \
        --bandwidth "$BANDWIDTH_LIMIT" \
        >> "$LOG_FILE" 2>&1 || log "WARNING" "Config sync encountered errors"
}

# Download backup from S3
download_backup() {
    local s3_path="$1"
    local local_path="${2:-$BACKUP_ROOT/downloads}"
    
    log "INFO" "Downloading backup from S3: $s3_path"
    
    mkdir -p "$local_path"
    
    if aws s3 cp "$s3_path" "$local_path/" \
        --region "$S3_REGION" \
        >> "$LOG_FILE" 2>&1; then
        log "INFO" "Download complete: $(basename "$s3_path")"
        echo "$local_path/$(basename "$s3_path")"
    else
        error_exit "Failed to download: $s3_path"
    fi
}

# List S3 backups
list_s3_backups() {
    local backup_type="${1:-all}"
    
    log "INFO" "Listing S3 backups (type: $backup_type)..."
    
    case "$backup_type" in
        "database"|"db")
            aws s3 ls "s3://$S3_BUCKET/daily/" --recursive | grep "postgres_.*\.sql\.gz$" | sort -r
            ;;
        "redis")
            aws s3 ls "s3://$S3_BUCKET/redis/" --recursive | grep "redis_backup_.*\.tar\.gz$" | sort -r
            ;;
        "config")
            aws s3 ls "s3://$S3_BUCKET/config/" --recursive | grep "config_backup_.*\.tar\.gz$" | sort -r
            ;;
        "all"|*)
            echo "=== Database Backups ==="
            aws s3 ls "s3://$S3_BUCKET/daily/" --recursive | grep "postgres_.*\.sql\.gz$" | sort -r | head -10
            echo
            echo "=== Redis Backups ==="
            aws s3 ls "s3://$S3_BUCKET/redis/" --recursive | grep "redis_backup_.*\.tar\.gz$" | sort -r | head -5
            echo
            echo "=== Configuration Backups ==="
            aws s3 ls "s3://$S3_BUCKET/config/" --recursive | grep "config_backup_.*\.tar\.gz$" | sort -r | head -5
            ;;
    esac
}

# Verify S3 backup integrity
verify_s3_backup() {
    local s3_path="$1"
    
    log "INFO" "Verifying S3 backup: $s3_path"
    
    # Get object metadata
    local metadata
    metadata=$(aws s3api head-object --bucket "$S3_BUCKET" --key "${s3_path#s3://$S3_BUCKET/}" 2>/dev/null)
    
    if [[ -n "$metadata" ]]; then
        local size=$(echo "$metadata" | jq -r '.ContentLength')
        local etag=$(echo "$metadata" | jq -r '.ETag' | tr -d '"')
        local last_modified=$(echo "$metadata" | jq -r '.LastModified')
        
        log "INFO" "Backup found:"
        log "INFO" "  Size: $(numfmt --to=iec $size)"
        log "INFO" "  ETag: $etag"
        log "INFO" "  Last Modified: $last_modified"
        
        # Download and verify checksum
        local temp_file="/tmp/verify-$$"
        if aws s3 cp "$s3_path" "$temp_file" --quiet; then
            local local_md5=$(md5sum "$temp_file" | cut -d' ' -f1)
            rm -f "$temp_file"
            
            if [[ "$local_md5" == "$etag" ]]; then
                log "INFO" "✓ Checksum verification passed"
                return 0
            else
                log "ERROR" "✗ Checksum mismatch"
                return 1
            fi
        fi
    else
        log "ERROR" "Backup not found in S3"
        return 1
    fi
}

# Generate sync report
generate_report() {
    log "INFO" "Generating sync report..."
    
    local report_file="$BACKUP_ROOT/logs/s3-sync-report-$(date +%Y%m%d).txt"
    
    {
        echo "S3 Offsite Sync Report"
        echo "====================="
        echo "Date: $(date)"
        echo "Bucket: s3://$S3_BUCKET"
        echo
        
        # Storage usage
        echo "S3 Storage Usage:"
        aws s3 ls "s3://$S3_BUCKET" --recursive --summarize | tail -2
        echo
        
        # Recent uploads
        echo "Recent Uploads (last 24 hours):"
        aws s3api list-objects-v2 \
            --bucket "$S3_BUCKET" \
            --query "Contents[?LastModified>='$(date -u -d '24 hours ago' '+%Y-%m-%dT%H:%M:%S.000Z')'].{Key: Key, Size: Size, LastModified: LastModified}" \
            --output table
        echo
        
        # Storage class distribution
        echo "Storage Class Distribution:"
        aws s3api list-objects-v2 \
            --bucket "$S3_BUCKET" \
            --query "Contents[].StorageClass" \
            --output text | sort | uniq -c
            
    } > "$report_file"
    
    log "INFO" "Report saved to: $report_file"
}

# Main execution
main() {
    case "${1:-sync}" in
        "sync")
            check_prerequisites
            setup_lifecycle_policy
            sync_database_backups
            sync_redis_backups
            sync_config_backups
            generate_report
            send_notification "Sync Completed" "All backups synced to S3 successfully" "success"
            ;;
        "download")
            [[ -z "${2:-}" ]] && error_exit "Usage: $0 download <s3-path> [local-path]"
            check_prerequisites
            download_backup "$2" "${3:-}"
            ;;
        "list")
            check_prerequisites
            list_s3_backups "${2:-all}"
            ;;
        "verify")
            [[ -z "${2:-}" ]] && error_exit "Usage: $0 verify <s3-path>"
            check_prerequisites
            verify_s3_backup "$2"
            ;;
        "setup-lifecycle")
            check_prerequisites
            setup_lifecycle_policy
            ;;
        "--file")
            # Sync specific file
            [[ -z "${2:-}" ]] && error_exit "Usage: $0 --file <backup-file>"
            check_prerequisites
            if [[ -f "$2" ]]; then
                aws s3 cp "$2" "s3://$S3_BUCKET/manual/$(basename "$2")" \
                    --storage-class STANDARD
            else
                error_exit "File not found: $2"
            fi
            ;;
        *)
            echo "Usage: $0 {sync|download|list|verify|setup-lifecycle} [options]"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"