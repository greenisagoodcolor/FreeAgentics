#!/bin/bash
# Deploy Grafana dashboards for FreeAgentics monitoring

set -e

# Configuration
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
GRAFANA_API_KEY="${GRAFANA_API_KEY}"
DASHBOARD_DIR="/home/green/FreeAgentics/monitoring/dashboards"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}FreeAgentics Dashboard Deployment${NC}"
echo "================================="

# Check if Grafana API key is set
if [ -z "$GRAFANA_API_KEY" ]; then
    echo -e "${RED}Error: GRAFANA_API_KEY environment variable is not set${NC}"
    echo "Please set it with: export GRAFANA_API_KEY=your-api-key"
    exit 1
fi

# Function to deploy a dashboard
deploy_dashboard() {
    local dashboard_file=$1
    local dashboard_name=$(basename "$dashboard_file" .json)

    echo -n "Deploying dashboard: $dashboard_name... "

    # Read dashboard JSON
    dashboard_json=$(cat "$dashboard_file")

    # Wrap in API request format
    api_payload=$(jq -n --argjson dashboard "$dashboard_json" '{
        dashboard: $dashboard.dashboard,
        overwrite: true,
        folderUid: "freeagentics"
    }')

    # Deploy to Grafana
    response=$(curl -s -X POST \
        -H "Authorization: Bearer $GRAFANA_API_KEY" \
        -H "Content-Type: application/json" \
        -d "$api_payload" \
        "$GRAFANA_URL/api/dashboards/db")

    # Check response
    if echo "$response" | jq -e '.status == "success"' > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        dashboard_url=$(echo "$response" | jq -r '.url')
        echo "  URL: $GRAFANA_URL$dashboard_url"
    else
        echo -e "${RED}✗${NC}"
        echo "  Error: $(echo "$response" | jq -r '.message // "Unknown error"')"
    fi
}

# Create folder for FreeAgentics dashboards
echo "Creating dashboard folder..."
folder_response=$(curl -s -X POST \
    -H "Authorization: Bearer $GRAFANA_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "uid": "freeagentics",
        "title": "FreeAgentics",
        "overwrite": true
    }' \
    "$GRAFANA_URL/api/folders")

if echo "$folder_response" | jq -e '.uid' > /dev/null 2>&1; then
    echo -e "Folder created/updated ${GREEN}✓${NC}"
else
    echo -e "${YELLOW}Warning: Could not create folder${NC}"
fi

# Deploy all dashboards
echo -e "\nDeploying dashboards..."
for dashboard in "$DASHBOARD_DIR"/*.json; do
    if [ -f "$dashboard" ]; then
        deploy_dashboard "$dashboard"
    fi
done

# Create datasource if it doesn't exist
echo -e "\nChecking Prometheus datasource..."
datasource_exists=$(curl -s -H "Authorization: Bearer $GRAFANA_API_KEY" \
    "$GRAFANA_URL/api/datasources/name/Prometheus" | jq -e '.id' > /dev/null 2>&1 && echo "yes" || echo "no")

if [ "$datasource_exists" = "no" ]; then
    echo "Creating Prometheus datasource..."
    curl -s -X POST \
        -H "Authorization: Bearer $GRAFANA_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "Prometheus",
            "type": "prometheus",
            "url": "http://prometheus:9090",
            "access": "proxy",
            "isDefault": true
        }' \
        "$GRAFANA_URL/api/datasources" > /dev/null

    if [ $? -eq 0 ]; then
        echo -e "Prometheus datasource created ${GREEN}✓${NC}"
    else
        echo -e "${RED}Failed to create Prometheus datasource${NC}"
    fi
else
    echo -e "Prometheus datasource already exists ${GREEN}✓${NC}"
fi

# Set up alerts
echo -e "\nSetting up alert notification channels..."

# Slack notification channel
slack_channel=$(cat <<EOF
{
    "name": "FreeAgentics Slack",
    "type": "slack",
    "isDefault": false,
    "sendReminder": true,
    "frequency": "5m",
    "settings": {
        "url": "${SLACK_WEBHOOK_URL:-}",
        "recipient": "#freeagentics-alerts",
        "username": "Grafana"
    }
}
EOF
)

if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
    curl -s -X POST \
        -H "Authorization: Bearer $GRAFANA_API_KEY" \
        -H "Content-Type: application/json" \
        -d "$slack_channel" \
        "$GRAFANA_URL/api/alert-notifications" > /dev/null

    if [ $? -eq 0 ]; then
        echo -e "Slack notification channel created ${GREEN}✓${NC}"
    fi
else
    echo -e "${YELLOW}Skipping Slack setup (SLACK_WEBHOOK_URL not set)${NC}"
fi

# PagerDuty notification channel
pagerduty_channel=$(cat <<EOF
{
    "name": "FreeAgentics PagerDuty",
    "type": "pagerduty",
    "isDefault": false,
    "sendReminder": false,
    "settings": {
        "integrationKey": "${PAGERDUTY_INTEGRATION_KEY:-}"
    }
}
EOF
)

if [ -n "${PAGERDUTY_INTEGRATION_KEY:-}" ]; then
    curl -s -X POST \
        -H "Authorization: Bearer $GRAFANA_API_KEY" \
        -H "Content-Type: application/json" \
        -d "$pagerduty_channel" \
        "$GRAFANA_URL/api/alert-notifications" > /dev/null

    if [ $? -eq 0 ]; then
        echo -e "PagerDuty notification channel created ${GREEN}✓${NC}"
    fi
else
    echo -e "${YELLOW}Skipping PagerDuty setup (PAGERDUTY_INTEGRATION_KEY not set)${NC}"
fi

echo -e "\n${GREEN}Dashboard deployment complete!${NC}"
echo "Access your dashboards at: $GRAFANA_URL"
echo ""
echo "Available dashboards:"
echo "  - System Overview: $GRAFANA_URL/d/freeagentics-overview"
echo "  - Agent Performance: $GRAFANA_URL/d/freeagentics-agents"
echo "  - Security Monitoring: $GRAFANA_URL/d/freeagentics-security"
echo "  - Business Metrics: $GRAFANA_URL/d/freeagentics-business"
