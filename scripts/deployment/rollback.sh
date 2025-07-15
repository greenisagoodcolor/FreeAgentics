#!/bin/bash
# Emergency rollback script for FreeAgentics

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROLLBACK_VERSION="${1:-previous}"
ROLLBACK_DB="${ROLLBACK_DB:-false}"
LOG_FILE="/var/log/freeagentics/rollback-$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Logging functions
log() {
    echo -e "${GREEN}[ROLLBACK]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

# Save current state for analysis
save_current_state() {
    log "Saving current deployment state for analysis..."
    
    STATE_DIR="/var/backups/freeagentics/failed-deployment-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$STATE_DIR"
    
    # Save container logs
    log "Collecting container logs..."
    for container in $(docker ps -a --format "{{.Names}}" | grep freeagentics); do
        docker logs "$container" > "$STATE_DIR/${container}.log" 2>&1 || true
    done
    
    # Save container states
    docker ps -a > "$STATE_DIR/container_states.txt"
    
    # Save environment info
    cp "$PROJECT_ROOT/.env.production" "$STATE_DIR/" 2>/dev/null || true
    
    # Save metrics snapshot
    curl -s "http://localhost:9090/api/v1/query?query=up" > "$STATE_DIR/prometheus_snapshot.json" 2>/dev/null || true
    
    log "Current state saved to: $STATE_DIR"
}

# Stop problematic deployment
stop_current_deployment() {
    log "Stopping current deployment..."
    
    # Try graceful shutdown first
    timeout 30 docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" down || {
        warning "Graceful shutdown failed, forcing container removal..."
        docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" kill
        docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" rm -f
    }
    
    # Clean up any orphaned containers
    docker ps -a | grep freeagentics | awk '{print $1}' | xargs -r docker rm -f
    
    log "Current deployment stopped"
}

# Restore previous version
restore_previous_version() {
    log "Restoring previous version..."
    
    if [[ "$ROLLBACK_VERSION" == "previous" ]]; then
        # Use backup tags
        for service in api web worker; do
            if docker image inspect "freeagentics-${service}:backup" &> /dev/null; then
                docker tag "freeagentics-${service}:backup" "freeagentics-${service}:current"
                log "Restored $service from backup tag"
            else
                error "No backup found for $service"
            fi
        done
    else
        # Use specific version
        log "Pulling version: $ROLLBACK_VERSION"
        docker pull "your-registry.com/freeagentics:${ROLLBACK_VERSION}"
        docker pull "your-registry.com/freeagentics-web:${ROLLBACK_VERSION}"
        
        docker tag "your-registry.com/freeagentics:${ROLLBACK_VERSION}" "freeagentics-api:current"
        docker tag "your-registry.com/freeagentics-web:${ROLLBACK_VERSION}" "freeagentics-web:current"
    fi
}

# Database rollback
rollback_database() {
    if [[ "$ROLLBACK_DB" != "true" ]]; then
        log "Skipping database rollback (ROLLBACK_DB not set to true)"
        return
    fi
    
    warning "Rolling back database migrations..."
    
    # Get current revision
    current_revision=$(docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
        run --rm api alembic current 2>/dev/null | grep -oE '[a-f0-9]{12}' | head -1)
    
    log "Current database revision: $current_revision"
    
    # Get previous revision
    previous_revision=$(docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
        run --rm api alembic history 2>/dev/null | \
        grep -A1 "$current_revision" | tail -1 | grep -oE '[a-f0-9]{12}' || echo "")
    
    if [[ -n "$previous_revision" ]]; then
        log "Rolling back to revision: $previous_revision"
        docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
            run --rm api alembic downgrade "$previous_revision"
    else
        error "Could not determine previous revision"
        return 1
    fi
    
    # Verify rollback
    new_revision=$(docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
        run --rm api alembic current 2>/dev/null | grep -oE '[a-f0-9]{12}' | head -1)
    
    if [[ "$new_revision" == "$previous_revision" ]]; then
        log "Database rolled back successfully"
    else
        error "Database rollback may have failed"
    fi
}

# Start previous version
start_previous_version() {
    log "Starting previous version..."
    
    # Use stored environment if available
    BACKUP_ENV=$(find /var/backups/freeagentics -name ".env.production" -type f | \
        sort -r | head -1)
    
    if [[ -n "$BACKUP_ENV" ]] && [[ -f "$BACKUP_ENV" ]]; then
        warning "Using backup environment from: $BACKUP_ENV"
        cp "$BACKUP_ENV" "$PROJECT_ROOT/.env.production.rollback"
        export ENV_FILE="$PROJECT_ROOT/.env.production.rollback"
    fi
    
    # Start services
    docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" up -d
    
    # Wait for services to start
    log "Waiting for services to start..."
    sleep 30
}

# Verify rollback success
verify_rollback() {
    log "Verifying rollback..."
    
    # Run health checks
    if "$PROJECT_ROOT/scripts/deployment/health-check.sh" --quick; then
        log "Health checks passed ✓"
    else
        error "Health checks failed after rollback"
        return 1
    fi
    
    # Check service versions
    api_version=$(docker exec freeagentics-api cat /app/VERSION 2>/dev/null || echo "unknown")
    log "API version after rollback: $api_version"
    
    return 0
}

# Clear corrupted data
clear_corrupted_data() {
    log "Clearing potentially corrupted data..."
    
    # Clear Redis cache
    log "Flushing Redis cache..."
    docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
        exec -T redis redis-cli FLUSHDB || warning "Failed to flush Redis"
    
    # Clear temporary files
    log "Clearing temporary files..."
    docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
        exec -T api rm -rf /tmp/* || true
}

# Update load balancer
update_load_balancer() {
    log "Updating load balancer configuration..."
    
    # Remove failed instances from load balancer
    # This is environment-specific - adjust for your setup
    
    # For HAProxy
    if command -v haproxy &> /dev/null; then
        # Reload HAProxy with previous config
        if [[ -f /etc/haproxy/haproxy.cfg.backup ]]; then
            cp /etc/haproxy/haproxy.cfg.backup /etc/haproxy/haproxy.cfg
            systemctl reload haproxy
        fi
    fi
    
    # For AWS ALB
    if command -v aws &> /dev/null; then
        # Deregister unhealthy targets
        aws elbv2 describe-target-health \
            --target-group-arn "$TARGET_GROUP_ARN" \
            --query 'TargetHealthDescriptions[?TargetHealth.State!=`healthy`].Target.Id' \
            --output text | xargs -r -n1 aws elbv2 deregister-targets \
            --target-group-arn "$TARGET_GROUP_ARN" --targets Id=
    fi
}

# Send notifications
send_notifications() {
    local status=$1
    local message=$2
    
    # Slack notification
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"text\":\":rotating_light: Rollback $status\",
                \"attachments\": [{
                    \"color\": \"$([ "$status" = "completed" ] && echo "good" || echo "danger")\",
                    \"text\": \"$message\",
                    \"fields\": [
                        {\"title\": \"Environment\", \"value\": \"Production\", \"short\": true},
                        {\"title\": \"Timestamp\", \"value\": \"$(date)\", \"short\": true}
                    ]
                }]
            }" \
            "$SLACK_WEBHOOK" || true
    fi
    
    # Email notification
    if [[ -n "${ALERT_EMAIL:-}" ]]; then
        echo "$message" | mail -s "FreeAgentics Rollback $status" "$ALERT_EMAIL" || true
    fi
    
    # PagerDuty
    if [[ -n "${PAGERDUTY_KEY:-}" ]]; then
        curl -X POST https://events.pagerduty.com/v2/enqueue \
            -H 'Content-Type: application/json' \
            -d "{
                \"routing_key\": \"$PAGERDUTY_KEY\",
                \"event_action\": \"trigger\",
                \"payload\": {
                    \"summary\": \"FreeAgentics Rollback $status\",
                    \"severity\": \"error\",
                    \"source\": \"deployment-system\",
                    \"custom_details\": {
                        \"message\": \"$message\",
                        \"environment\": \"production\"
                    }
                }
            }" || true
    fi
}

# Main rollback procedure
main() {
    echo -e "\n${RED}╔════════════════════════════════════════╗${NC}"
    echo -e "${RED}║        EMERGENCY ROLLBACK INITIATED     ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════╝${NC}\n"
    
    log "Starting emergency rollback procedure"
    log "Rollback version: $ROLLBACK_VERSION"
    log "Database rollback: $ROLLBACK_DB"
    
    # Confirm rollback
    if [[ "${FORCE_ROLLBACK:-false}" != "true" ]]; then
        echo -e "${YELLOW}This will rollback the production deployment.${NC}"
        echo -e "${YELLOW}Are you sure? (yes/no)${NC}"
        read -r response
        if [[ "$response" != "yes" ]]; then
            log "Rollback cancelled by user"
            exit 0
        fi
    fi
    
    # Send initial notification
    send_notifications "initiated" "Emergency rollback has been initiated for production environment"
    
    # Execute rollback steps
    {
        save_current_state
        stop_current_deployment
        restore_previous_version
        
        if [[ "$ROLLBACK_DB" == "true" ]]; then
            rollback_database
        fi
        
        clear_corrupted_data
        start_previous_version
        update_load_balancer
        
        if verify_rollback; then
            log "Rollback completed successfully! ✓"
            send_notifications "completed" "Rollback completed successfully. Services have been restored."
            
            echo -e "\n${GREEN}╔════════════════════════════════════════╗${NC}"
            echo -e "${GREEN}║      ROLLBACK COMPLETED SUCCESSFULLY    ║${NC}"
            echo -e "${GREEN}╚════════════════════════════════════════╝${NC}\n"
            
            # Generate rollback report
            echo -e "${GREEN}=== Rollback Summary ===${NC}"
            echo "Start Time: $(grep 'Starting emergency' "$LOG_FILE" | cut -d' ' -f4-5)"
            echo "End Time: $(date +'%Y-%m-%d %H:%M:%S')"
            echo "Rolled back to: $ROLLBACK_VERSION"
            echo "Database rolled back: $ROLLBACK_DB"
            echo "Log file: $LOG_FILE"
            
            exit 0
        else
            error "Rollback verification failed!"
            send_notifications "failed" "Rollback completed but verification failed. Manual intervention required."
            exit 1
        fi
        
    } || {
        error "Rollback procedure failed!"
        send_notifications "failed" "Rollback procedure failed. Immediate manual intervention required!"
        
        echo -e "\n${RED}╔════════════════════════════════════════╗${NC}"
        echo -e "${RED}║        ROLLBACK FAILED - MANUAL         ║${NC}"
        echo -e "${RED}║       INTERVENTION REQUIRED!            ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════╝${NC}\n"
        
        echo "Next steps:"
        echo "1. Check logs at: $LOG_FILE"
        echo "2. Review saved state at: $STATE_DIR"
        echo "3. Contact senior ops team immediately"
        echo "4. Consider manual database restore from backup"
        
        exit 1
    }
}

# Handle script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            ROLLBACK_VERSION="$2"
            shift 2
            ;;
        --rollback-db)
            ROLLBACK_DB="true"
            shift
            ;;
        --force)
            FORCE_ROLLBACK="true"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --version VERSION    Rollback to specific version (default: previous)"
            echo "  --rollback-db       Also rollback database migrations"
            echo "  --force             Skip confirmation prompt"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main rollback procedure
main