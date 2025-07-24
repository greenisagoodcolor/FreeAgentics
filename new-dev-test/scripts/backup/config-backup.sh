#!/bin/bash
# Configuration Backup Script for FreeAgentics
# Backs up all configuration files, certificates, and environment settings

set -euo pipefail

# Configuration
BACKUP_ROOT="${BACKUP_ROOT:-/var/backups/freeagentics}"
CONFIG_BACKUP_DIR="$BACKUP_ROOT/config"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEMP_DIR="/tmp/freeagentics-config-backup-$$"
ENCRYPTION_KEY_FILE="/etc/freeagentics/backup-encryption.key"

# Create directories
mkdir -p "$CONFIG_BACKUP_DIR" "$BACKUP_ROOT/logs"
LOG_FILE="$BACKUP_ROOT/logs/config-backup-$TIMESTAMP.log"

# Logging function
log() {
    local level="$1"
    shift
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR" "$1"
    rm -rf "$TEMP_DIR"
    exit 1
}

# Cleanup on exit
trap "rm -rf $TEMP_DIR" EXIT

# Create encryption key if it doesn't exist
create_encryption_key() {
    if [[ ! -f "$ENCRYPTION_KEY_FILE" ]]; then
        log "INFO" "Creating new encryption key..."
        mkdir -p "$(dirname "$ENCRYPTION_KEY_FILE")"
        openssl rand -base64 32 > "$ENCRYPTION_KEY_FILE"
        chmod 600 "$ENCRYPTION_KEY_FILE"
        log "WARNING" "New encryption key created - back this up securely!"
    fi
}

# Collect configuration files
collect_configs() {
    log "INFO" "Collecting configuration files..."

    mkdir -p "$TEMP_DIR"

    # System configuration
    local system_configs=(
        "/etc/freeagentics/"
        "/etc/nginx/sites-available/freeagentics"
        "/etc/nginx/sites-enabled/freeagentics"
        "/etc/systemd/system/freeagentics*.service"
        "/etc/logrotate.d/freeagentics"
        "/etc/cron.d/freeagentics*"
    )

    # Application configuration
    local app_configs=(
        "/home/green/FreeAgentics/.env"
        "/home/green/FreeAgentics/.env.production"
        "/home/green/FreeAgentics/docker-compose.yml"
        "/home/green/FreeAgentics/docker-compose.production.yml"
        "/home/green/FreeAgentics/nginx/"
        "/home/green/FreeAgentics/alembic.ini"
        "/home/green/FreeAgentics/pyproject.toml"
    )

    # SSL certificates
    local ssl_configs=(
        "/etc/letsencrypt/live/"
        "/etc/letsencrypt/renewal/"
        "/etc/ssl/certs/freeagentics*"
    )

    # Monitoring configuration
    local monitoring_configs=(
        "/etc/prometheus/prometheus.yml"
        "/etc/grafana/grafana.ini"
        "/etc/alertmanager/alertmanager.yml"
    )

    # Create directory structure in temp
    for category in system app ssl monitoring; do
        mkdir -p "$TEMP_DIR/$category"
    done

    # Copy system configs
    for item in "${system_configs[@]}"; do
        if [[ -e "$item" ]]; then
            cp -rL "$item" "$TEMP_DIR/system/" 2>/dev/null || \
                log "WARNING" "Failed to copy: $item"
        fi
    done

    # Copy application configs
    for item in "${app_configs[@]}"; do
        if [[ -e "$item" ]]; then
            # Strip sensitive data from .env files
            if [[ "$item" =~ \.env ]]; then
                sed 's/\(PASSWORD\|SECRET\|KEY\)=.*/\1=<REDACTED>/' "$item" > \
                    "$TEMP_DIR/app/$(basename "$item").sanitized"
                # Keep encrypted version with actual values
                cp "$item" "$TEMP_DIR/app/$(basename "$item")"
            else
                cp -rL "$item" "$TEMP_DIR/app/" 2>/dev/null || \
                    log "WARNING" "Failed to copy: $item"
            fi
        fi
    done

    # Copy SSL configs (be careful with permissions)
    for item in "${ssl_configs[@]}"; do
        if [[ -e "$item" ]]; then
            cp -rL "$item" "$TEMP_DIR/ssl/" 2>/dev/null || \
                log "WARNING" "Failed to copy SSL: $item"
        fi
    done

    # Copy monitoring configs
    for item in "${monitoring_configs[@]}"; do
        if [[ -e "$item" ]]; then
            cp -rL "$item" "$TEMP_DIR/monitoring/" 2>/dev/null || \
                log "WARNING" "Failed to copy: $item"
        fi
    done

    # Add metadata
    cat > "$TEMP_DIR/backup-metadata.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "date": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "kernel": "$(uname -r)",
    "docker_version": "$(docker --version 2>/dev/null || echo 'N/A')",
    "postgres_version": "$(psql --version 2>/dev/null | head -1 || echo 'N/A')",
    "backup_tool_version": "1.0.0"
}
EOF

    # Create inventory of backed up files
    find "$TEMP_DIR" -type f -printf "%P\n" | sort > "$TEMP_DIR/inventory.txt"

    log "INFO" "Configuration collection complete"
}

# Create encrypted archive
create_archive() {
    log "INFO" "Creating encrypted archive..."

    local archive_name="config_backup_${TIMESTAMP}.tar.gz"
    local archive_path="$CONFIG_BACKUP_DIR/$archive_name"
    local encrypted_path="${archive_path}.enc"

    # Create tar archive
    tar -czf "$archive_path" -C "$TEMP_DIR" . 2>> "$LOG_FILE"

    if [[ ! -f "$archive_path" ]]; then
        error_exit "Failed to create archive"
    fi

    # Encrypt the archive
    create_encryption_key

    if openssl enc -aes-256-cbc -salt -in "$archive_path" -out "$encrypted_path" \
        -pass file:"$ENCRYPTION_KEY_FILE" 2>> "$LOG_FILE"; then
        log "INFO" "Archive encrypted successfully"
        rm -f "$archive_path"  # Remove unencrypted version
    else
        log "WARNING" "Encryption failed - keeping unencrypted archive"
        rm -f "$encrypted_path"
    fi

    # Create checksum
    if [[ -f "$encrypted_path" ]]; then
        sha256sum "$encrypted_path" > "${encrypted_path}.sha256"
        log "INFO" "Checksum created"
    else
        sha256sum "$archive_path" > "${archive_path}.sha256"
    fi

    # Set appropriate permissions
    chmod 600 "$CONFIG_BACKUP_DIR"/config_backup_*

    log "INFO" "Archive created: ${encrypted_path:-$archive_path}"
}

# Verify backup
verify_backup() {
    local backup_file="${1:-}"

    if [[ -z "$backup_file" ]]; then
        # Verify latest backup
        backup_file=$(ls -t "$CONFIG_BACKUP_DIR"/config_backup_*.tar.gz* | head -1)
    fi

    if [[ ! -f "$backup_file" ]]; then
        error_exit "Backup file not found: $backup_file"
    fi

    log "INFO" "Verifying backup: $backup_file"

    # Check checksum if available
    if [[ -f "${backup_file}.sha256" ]]; then
        if sha256sum -c "${backup_file}.sha256" >/dev/null 2>&1; then
            log "INFO" "✓ Checksum verification passed"
        else
            log "ERROR" "✗ Checksum verification failed"
            return 1
        fi
    fi

    # Test extraction
    local test_dir="/tmp/config-verify-$$"
    mkdir -p "$test_dir"

    if [[ "$backup_file" =~ \.enc$ ]]; then
        # Decrypt and extract
        if openssl enc -aes-256-cbc -d -in "$backup_file" \
            -pass file:"$ENCRYPTION_KEY_FILE" 2>/dev/null | \
            tar -tzf - >/dev/null 2>&1; then
            log "INFO" "✓ Archive integrity verified"
        else
            log "ERROR" "✗ Archive integrity check failed"
            rm -rf "$test_dir"
            return 1
        fi
    else
        # Just test extraction
        if tar -tzf "$backup_file" >/dev/null 2>&1; then
            log "INFO" "✓ Archive integrity verified"
        else
            log "ERROR" "✗ Archive integrity check failed"
            rm -rf "$test_dir"
            return 1
        fi
    fi

    rm -rf "$test_dir"
    log "INFO" "Backup verification complete"
    return 0
}

# Restore configuration
restore_config() {
    local backup_file="$1"
    local restore_dir="${2:-/tmp/config-restore-$(date +%Y%m%d-%H%M%S)}"

    if [[ ! -f "$backup_file" ]]; then
        error_exit "Backup file not found: $backup_file"
    fi

    log "INFO" "Restoring configuration from: $backup_file"
    log "INFO" "Restore directory: $restore_dir"

    mkdir -p "$restore_dir"

    # Decrypt and extract
    if [[ "$backup_file" =~ \.enc$ ]]; then
        log "INFO" "Decrypting archive..."
        if ! openssl enc -aes-256-cbc -d -in "$backup_file" \
            -pass file:"$ENCRYPTION_KEY_FILE" | \
            tar -xzf - -C "$restore_dir"; then
            error_exit "Failed to decrypt and extract archive"
        fi
    else
        if ! tar -xzf "$backup_file" -C "$restore_dir"; then
            error_exit "Failed to extract archive"
        fi
    fi

    log "INFO" "Configuration restored to: $restore_dir"
    log "INFO" "Review the files and manually copy needed configurations"

    # Show inventory
    if [[ -f "$restore_dir/inventory.txt" ]]; then
        echo
        echo "Restored files:"
        cat "$restore_dir/inventory.txt"
    fi
}

# Clean old backups
cleanup_old_backups() {
    local retention_days="${1:-90}"

    log "INFO" "Cleaning up backups older than $retention_days days..."

    find "$CONFIG_BACKUP_DIR" -name "config_backup_*" -mtime +$retention_days -delete

    log "INFO" "Cleanup complete"
}

# Main execution
main() {
    case "${1:-backup}" in
        "backup")
            collect_configs
            create_archive
            cleanup_old_backups
            log "INFO" "Configuration backup completed successfully"
            ;;
        "verify")
            verify_backup "${2:-}"
            ;;
        "restore")
            [[ -z "${2:-}" ]] && error_exit "Usage: $0 restore <backup-file> [restore-dir]"
            restore_config "$2" "${3:-}"
            ;;
        "list")
            log "INFO" "Available configuration backups:"
            ls -lth "$CONFIG_BACKUP_DIR"/config_backup_* 2>/dev/null | head -20
            ;;
        "cleanup")
            cleanup_old_backups "${2:-90}"
            ;;
        *)
            echo "Usage: $0 {backup|verify|restore|list|cleanup} [options]"
            echo ""
            echo "Commands:"
            echo "  backup              Create new configuration backup"
            echo "  verify [file]       Verify backup integrity"
            echo "  restore <file> [dir] Restore configuration from backup"
            echo "  list                List available backups"
            echo "  cleanup [days]      Remove backups older than N days (default: 90)"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
