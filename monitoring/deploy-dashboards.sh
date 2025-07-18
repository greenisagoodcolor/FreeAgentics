#!/bin/bash

# FreeAgentics Grafana Dashboard Deployment Script
# This script deploys all FreeAgentics dashboards to Grafana

set -e

# Configuration
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
GRAFANA_USER="${GRAFANA_USER:-admin}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin}"
DASHBOARD_DIR="$(dirname "$0")/grafana/dashboards"
PROVISIONING_DIR="$(dirname "$0")/grafana/provisioning"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Check if Grafana is accessible
check_grafana() {
    log "Checking Grafana accessibility at $GRAFANA_URL..."

    if ! curl -s -f "$GRAFANA_URL/api/health" > /dev/null; then
        error "Grafana is not accessible at $GRAFANA_URL"
        error "Please ensure Grafana is running and accessible"
        exit 1
    fi

    success "Grafana is accessible"
}

# Create folder for dashboards
create_folder() {
    local folder_title="$1"
    local folder_uid="$2"

    log "Creating folder: $folder_title"

    local folder_json=$(cat <<EOF
{
  "uid": "$folder_uid",
  "title": "$folder_title"
}
EOF
)

    curl -s -X POST \
        -H "Content-Type: application/json" \
        -u "$GRAFANA_USER:$GRAFANA_PASSWORD" \
        -d "$folder_json" \
        "$GRAFANA_URL/api/folders" > /dev/null || true

    success "Folder created: $folder_title"
}

# Deploy a single dashboard
deploy_dashboard() {
    local dashboard_file="$1"
    local dashboard_name=$(basename "$dashboard_file" .json)

    log "Deploying dashboard: $dashboard_name"

    # Read dashboard JSON
    local dashboard_json=$(cat "$dashboard_file")

    # Wrap in dashboard object for import
    local import_json=$(cat <<EOF
{
  "dashboard": $dashboard_json,
  "folderId": 0,
  "overwrite": true
}
EOF
)

    # Deploy dashboard
    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -u "$GRAFANA_USER:$GRAFANA_PASSWORD" \
        -d "$import_json" \
        "$GRAFANA_URL/api/dashboards/db")

    if echo "$response" | grep -q "success"; then
        success "Dashboard deployed successfully: $dashboard_name"
    else
        error "Failed to deploy dashboard: $dashboard_name"
        echo "$response"
        return 1
    fi
}

# Deploy all dashboards
deploy_dashboards() {
    log "Deploying FreeAgentics dashboards..."

    # Create main folder
    create_folder "FreeAgentics" "freeagentics-folder"

    # Deploy each dashboard
    local dashboard_files=(
        "$DASHBOARD_DIR/freeagentics-system-overview.json"
        "$DASHBOARD_DIR/freeagentics-agent-coordination.json"
        "$DASHBOARD_DIR/freeagentics-memory-heatmap.json"
        "$DASHBOARD_DIR/freeagentics-api-performance.json"
        "$DASHBOARD_DIR/freeagentics-capacity-planning.json"
    )

    for dashboard_file in "${dashboard_files[@]}"; do
        if [[ -f "$dashboard_file" ]]; then
            deploy_dashboard "$dashboard_file"
        else
            warning "Dashboard file not found: $dashboard_file"
        fi
    done

    success "All dashboards deployed successfully"
}

# Verify deployment
verify_deployment() {
    log "Verifying dashboard deployment..."

    local dashboards=$(curl -s -u "$GRAFANA_USER:$GRAFANA_PASSWORD" \
        "$GRAFANA_URL/api/search?query=FreeAgentics")

    local dashboard_count=$(echo "$dashboards" | jq length)

    if [[ $dashboard_count -gt 0 ]]; then
        success "Found $dashboard_count FreeAgentics dashboards"

        # List deployed dashboards
        echo "$dashboards" | jq -r '.[].title' | while read -r title; do
            log "  - $title"
        done
    else
        error "No FreeAgentics dashboards found"
        return 1
    fi
}

# Create Grafana data source
create_datasource() {
    log "Creating Prometheus datasource..."

    local datasource_json=$(cat <<EOF
{
  "name": "Prometheus",
  "type": "prometheus",
  "url": "http://prometheus:9090",
  "access": "proxy",
  "isDefault": true,
  "basicAuth": false,
  "jsonData": {
    "httpMethod": "POST",
    "manageAlerts": true,
    "prometheusType": "Prometheus",
    "prometheusVersion": "2.40.0",
    "cacheLevel": "High"
  }
}
EOF
)

    curl -s -X POST \
        -H "Content-Type: application/json" \
        -u "$GRAFANA_USER:$GRAFANA_PASSWORD" \
        -d "$datasource_json" \
        "$GRAFANA_URL/api/datasources" > /dev/null || true

    success "Prometheus datasource created"
}

# Set up alerts
setup_alerts() {
    log "Setting up Grafana alerts..."

    # Create notification channel for Slack
    local notification_json=$(cat <<EOF
{
  "name": "freeagentics-alerts",
  "type": "slack",
  "settings": {
    "url": "${SLACK_WEBHOOK_URL:-https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK}",
    "channel": "#freeagentics-alerts",
    "username": "Grafana",
    "title": "FreeAgentics Alert",
    "text": "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"
  }
}
EOF
)

    curl -s -X POST \
        -H "Content-Type: application/json" \
        -u "$GRAFANA_USER:$GRAFANA_PASSWORD" \
        -d "$notification_json" \
        "$GRAFANA_URL/api/alert-notifications" > /dev/null || true

    success "Alert notification channel created"
}

# Main deployment function
main() {
    log "Starting FreeAgentics dashboard deployment..."

    # Check dependencies
    if ! command -v curl &> /dev/null; then
        error "curl is required but not installed"
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        error "jq is required but not installed"
        exit 1
    fi

    # Check if dashboard files exist
    if [[ ! -d "$DASHBOARD_DIR" ]]; then
        error "Dashboard directory not found: $DASHBOARD_DIR"
        exit 1
    fi

    # Run deployment steps
    check_grafana
    create_datasource
    deploy_dashboards
    setup_alerts
    verify_deployment

    success "FreeAgentics dashboard deployment completed successfully!"
    log "Access your dashboards at: $GRAFANA_URL"
    log "Default credentials: $GRAFANA_USER/$GRAFANA_PASSWORD"
}

# Handle command line arguments
case "${1:-}" in
    "verify")
        check_grafana
        verify_deployment
        ;;
    "datasource")
        check_grafana
        create_datasource
        ;;
    "dashboards")
        check_grafana
        deploy_dashboards
        ;;
    "alerts")
        check_grafana
        setup_alerts
        ;;
    "")
        main
        ;;
    *)
        echo "Usage: $0 [verify|datasource|dashboards|alerts]"
        echo "  verify      - Verify deployment"
        echo "  datasource  - Create datasource only"
        echo "  dashboards  - Deploy dashboards only"
        echo "  alerts      - Setup alerts only"
        echo "  (no args)   - Full deployment"
        exit 1
        ;;
esac
