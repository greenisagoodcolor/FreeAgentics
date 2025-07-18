#!/bin/bash
# FreeAgentics Backup System Installation Script
# Installs and configures the automated backup system

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

# Install system dependencies
install_dependencies() {
    log_info "Installing system dependencies..."

    # Update package list
    apt-get update

    # Install required packages
    apt-get install -y \
        postgresql-client \
        redis-tools \
        python3-pip \
        python3-venv \
        awscli \
        azure-cli \
        jq \
        curl \
        gzip \
        tar \
        cron

    # Install Google Cloud SDK
    if ! command -v gcloud &> /dev/null; then
        log_info "Installing Google Cloud SDK..."
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
        curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
        apt-get update && apt-get install -y google-cloud-sdk
    fi
}

# Install Python dependencies
install_python_dependencies() {
    log_info "Installing Python dependencies..."

    # Create virtual environment if it doesn't exist
    if [[ ! -d "/opt/freeagentics-backup/venv" ]]; then
        python3 -m venv /opt/freeagentics-backup/venv
    fi

    # Activate virtual environment and install dependencies
    source /opt/freeagentics-backup/venv/bin/activate

    pip install --upgrade pip
    pip install \
        boto3 \
        azure-storage-blob \
        google-cloud-storage \
        b2sdk \
        psycopg2-binary \
        redis \
        pyyaml \
        schedule \
        requests \
        prometheus-client

    deactivate
}

# Create backup user
create_backup_user() {
    log_info "Creating backup user..."

    # Create user if it doesn't exist
    if ! id -u freeagentics &>/dev/null; then
        useradd -r -s /bin/bash -m -d /home/freeagentics freeagentics
    fi

    # Add user to necessary groups
    usermod -a -G docker freeagentics 2>/dev/null || true
}

# Setup directory structure
setup_directories() {
    log_info "Setting up backup directories..."

    # Create backup directories
    mkdir -p /var/backups/freeagentics/{daily,incremental,redis,config,knowledge_graph,app_state,logs,metadata,verification,metrics,temp}

    # Create configuration directory
    mkdir -p /etc/freeagentics

    # Set permissions
    chown -R freeagentics:freeagentics /var/backups/freeagentics
    chmod -R 750 /var/backups/freeagentics

    # Create log directory
    mkdir -p /var/log/freeagentics
    chown freeagentics:freeagentics /var/log/freeagentics
}

# Copy configuration template
setup_configuration() {
    log_info "Setting up configuration..."

    # Copy configuration template if not exists
    if [[ ! -f "/etc/freeagentics/backup.env" ]]; then
        cp "$SCRIPT_DIR/backup-config.env.template" "/etc/freeagentics/backup.env"
        chmod 600 /etc/freeagentics/backup.env
        chown freeagentics:freeagentics /etc/freeagentics/backup.env

        log_warn "Configuration template copied to /etc/freeagentics/backup.env"
        log_warn "Please edit this file and set your specific values before starting the backup service"
    fi

    # Generate encryption key if not exists
    if [[ ! -f "/etc/freeagentics/backup-encryption.key" ]]; then
        log_info "Generating encryption key..."
        openssl rand -base64 32 > /etc/freeagentics/backup-encryption.key
        chmod 600 /etc/freeagentics/backup-encryption.key
        chown freeagentics:freeagentics /etc/freeagentics/backup-encryption.key
    fi
}

# Setup PostgreSQL access
setup_postgresql_access() {
    log_info "Setting up PostgreSQL access..."

    # Create .pgpass file for passwordless access
    if [[ ! -f "/home/freeagentics/.pgpass" ]]; then
        touch /home/freeagentics/.pgpass
        chmod 600 /home/freeagentics/.pgpass
        chown freeagentics:freeagentics /home/freeagentics/.pgpass

        log_warn "Please add PostgreSQL credentials to /home/freeagentics/.pgpass"
        log_warn "Format: hostname:port:database:username:password"
    fi
}

# Install systemd service
install_systemd_service() {
    log_info "Installing systemd service..."

    # Copy service file
    cp "$SCRIPT_DIR/freeagentics-backup.service" /etc/systemd/system/

    # Copy timer file
    cat > /etc/systemd/system/freeagentics-backup.timer <<EOF
[Unit]
Description=FreeAgentics Backup Timer
Requires=freeagentics-backup.service

[Timer]
# Run immediately on boot
OnBootSec=5min

# Run every 6 hours
OnUnitActiveSec=6h

# Randomize start time by up to 5 minutes
RandomizedDelaySec=300

# Make timer persistent
Persistent=true

[Install]
WantedBy=timers.target
EOF

    # Reload systemd
    systemctl daemon-reload

    # Enable service and timer
    systemctl enable freeagentics-backup.timer

    log_info "Systemd service installed and enabled"
}

# Setup cron jobs for legacy support
setup_cron_jobs() {
    log_info "Setting up cron jobs..."

    # Create cron file
    cat > /etc/cron.d/freeagentics-backup <<EOF
# FreeAgentics Backup Schedule
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin

# Full backup daily at 2 AM
0 2 * * * freeagentics /usr/bin/python3 $PROJECT_ROOT/scripts/backup/automated-backup-system.py --run-now >> /var/log/freeagentics/backup-cron.log 2>&1

# Cleanup old backups weekly on Sunday at 4 AM
0 4 * * 0 freeagentics /usr/bin/python3 $PROJECT_ROOT/scripts/backup/automated-backup-system.py --cleanup >> /var/log/freeagentics/backup-cron.log 2>&1

# Test disaster recovery weekly on Sunday at 5 AM
0 5 * * 0 freeagentics /usr/bin/python3 $PROJECT_ROOT/scripts/backup/automated-backup-system.py --test-restore >> /var/log/freeagentics/backup-cron.log 2>&1
EOF

    chmod 644 /etc/cron.d/freeagentics-backup
}

# Setup monitoring integration
setup_monitoring() {
    log_info "Setting up monitoring integration..."

    # Create Prometheus textfile directory
    mkdir -p /var/lib/prometheus/textfile_collector
    chown freeagentics:freeagentics /var/lib/prometheus/textfile_collector

    # Create monitoring script
    cat > /usr/local/bin/freeagentics-backup-metrics <<'EOF'
#!/bin/bash
# Generate Prometheus metrics for backup status

METRICS_FILE="/var/lib/prometheus/textfile_collector/freeagentics_backup.prom"
METADATA_DIR="/var/backups/freeagentics/metadata"

# Get latest backup status
if [[ -d "$METADATA_DIR" ]]; then
    LATEST_BACKUP=$(ls -t "$METADATA_DIR"/*.json 2>/dev/null | head -1)

    if [[ -f "$LATEST_BACKUP" ]]; then
        STATUS=$(jq -r '.status' "$LATEST_BACKUP")
        SIZE=$(jq -r '.size_bytes' "$LATEST_BACKUP")
        DURATION=$(jq -r '.duration_seconds' "$LATEST_BACKUP")
        TIMESTAMP=$(jq -r '.timestamp' "$LATEST_BACKUP")

        # Convert status to numeric
        STATUS_NUM=0
        [[ "$STATUS" == "completed" || "$STATUS" == "verified" ]] && STATUS_NUM=1

        # Write metrics
        cat > "$METRICS_FILE" <<METRICS
# HELP freeagentics_backup_status Backup status (1=success, 0=failure)
# TYPE freeagentics_backup_status gauge
freeagentics_backup_status{environment="production"} $STATUS_NUM

# HELP freeagentics_backup_size_bytes Size of last backup in bytes
# TYPE freeagentics_backup_size_bytes gauge
freeagentics_backup_size_bytes{environment="production"} $SIZE

# HELP freeagentics_backup_duration_seconds Duration of last backup in seconds
# TYPE freeagentics_backup_duration_seconds gauge
freeagentics_backup_duration_seconds{environment="production"} $DURATION

# HELP freeagentics_last_successful_backup_timestamp Timestamp of last successful backup
# TYPE freeagentics_last_successful_backup_timestamp gauge
freeagentics_last_successful_backup_timestamp{environment="production"} $(date -d "$TIMESTAMP" +%s)
METRICS
    fi
fi
EOF

    chmod +x /usr/local/bin/freeagentics-backup-metrics

    # Add to cron for regular updates
    echo "*/5 * * * * freeagentics /usr/local/bin/freeagentics-backup-metrics" >> /etc/cron.d/freeagentics-backup
}

# Setup log rotation
setup_log_rotation() {
    log_info "Setting up log rotation..."

    cat > /etc/logrotate.d/freeagentics-backup <<EOF
/var/log/freeagentics/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 640 freeagentics freeagentics
    sharedscripts
    postrotate
        systemctl reload rsyslog > /dev/null 2>&1 || true
    endscript
}

/var/backups/freeagentics/logs/*.log {
    weekly
    rotate 52
    compress
    delaycompress
    notifempty
    create 640 freeagentics freeagentics
}
EOF
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    # Check directories
    [[ -d "/var/backups/freeagentics" ]] || log_error "Backup directory not found"
    [[ -d "/etc/freeagentics" ]] || log_error "Configuration directory not found"

    # Check service files
    [[ -f "/etc/systemd/system/freeagentics-backup.service" ]] || log_error "Service file not found"
    [[ -f "/etc/systemd/system/freeagentics-backup.timer" ]] || log_error "Timer file not found"

    # Check configuration
    [[ -f "/etc/freeagentics/backup.env" ]] || log_error "Configuration file not found"

    # Check Python script
    [[ -f "$PROJECT_ROOT/scripts/backup/automated-backup-system.py" ]] || log_error "Backup script not found"

    log_info "Installation verification complete"
}

# Display next steps
display_next_steps() {
    cat <<EOF

${GREEN}=== FreeAgentics Backup System Installation Complete ===${NC}

${YELLOW}Next Steps:${NC}

1. Edit the configuration file:
   ${GREEN}vim /etc/freeagentics/backup.env${NC}

2. Add PostgreSQL credentials to:
   ${GREEN}vim /home/freeagentics/.pgpass${NC}
   Format: hostname:port:database:username:password

3. Configure cloud storage credentials:
   - AWS: ${GREEN}aws configure${NC} (as freeagentics user)
   - Azure: ${GREEN}az login${NC}
   - GCP: ${GREEN}gcloud auth login${NC}

4. Test the backup system:
   ${GREEN}sudo -u freeagentics python3 $PROJECT_ROOT/scripts/backup/automated-backup-system.py --run-now${NC}

5. Start the backup service:
   ${GREEN}systemctl start freeagentics-backup.timer${NC}
   ${GREEN}systemctl status freeagentics-backup.timer${NC}

6. Monitor backup logs:
   ${GREEN}journalctl -u freeagentics-backup -f${NC}
   ${GREEN}tail -f /var/log/freeagentics/backup.log${NC}

7. Access Grafana dashboard:
   Import dashboard from: ${GREEN}$PROJECT_ROOT/monitoring/dashboards/backup_monitoring_dashboard.json${NC}

${YELLOW}Important:${NC}
- Ensure database credentials are properly configured
- Set up offsite storage credentials
- Configure notification webhooks (Slack, email, PagerDuty)
- Review and adjust retention policies
- Test disaster recovery procedures

Documentation: ${GREEN}$PROJECT_ROOT/docs/operations/DISASTER_RECOVERY_PROCEDURES.md${NC}

EOF
}

# Main installation flow
main() {
    log_info "Starting FreeAgentics Backup System installation..."

    check_root
    install_dependencies
    create_backup_user
    setup_directories
    install_python_dependencies
    setup_configuration
    setup_postgresql_access
    install_systemd_service
    setup_cron_jobs
    setup_monitoring
    setup_log_rotation
    verify_installation
    display_next_steps

    log_info "Installation completed successfully!"
}

# Run main function
main "$@"
