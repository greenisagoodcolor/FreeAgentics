#!/bin/bash
# FreeAgentics Production Deployment Script
# This script handles the complete production deployment process

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-your-registry.com}"
VERSION="${VERSION:-latest}"
DEPLOYMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DEPLOYMENT_DIR/../.." && pwd)"

# Logging
LOG_FILE="/var/log/freeagentics/deployment-$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_FILE"
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check if running as appropriate user
    if [[ $EUID -eq 0 ]]; then
       error "This script should not be run as root"
       exit 1
    fi
    
    # Check required tools
    for tool in docker docker-compose git curl jq; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    # Verify environment file exists
    if [[ ! -f "$PROJECT_ROOT/.env.production" ]]; then
        error "Production environment file not found: $PROJECT_ROOT/.env.production"
        exit 1
    fi
    
    # Check disk space
    available_space=$(df -BG /var/lib/docker | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $available_space -lt 10 ]]; then
        error "Insufficient disk space. At least 10GB required, only ${available_space}GB available"
        exit 1
    fi
    
    log "Pre-deployment checks passed ✓"
}

# Backup current deployment
backup_current() {
    log "Backing up current deployment..."
    
    BACKUP_DIR="/var/backups/freeagentics/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup docker-compose files
    if [[ -f "$PROJECT_ROOT/docker-compose.production.yml" ]]; then
        cp "$PROJECT_ROOT/docker-compose.production.yml" "$BACKUP_DIR/"
    fi
    
    # Backup environment files
    cp "$PROJECT_ROOT/.env.production" "$BACKUP_DIR/"
    
    # Export current container states
    docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" > "$BACKUP_DIR/container_states.txt"
    
    # Tag current images for rollback
    for service in api web worker redis postgres; do
        if docker image inspect "freeagentics-${service}:current" &> /dev/null; then
            docker tag "freeagentics-${service}:current" "freeagentics-${service}:backup"
        fi
    done
    
    log "Backup completed at: $BACKUP_DIR"
}

# Pull new images
pull_images() {
    log "Pulling new images..."
    
    # Login to registry if credentials provided
    if [[ -n "${DOCKER_USERNAME:-}" ]] && [[ -n "${DOCKER_PASSWORD:-}" ]]; then
        echo "$DOCKER_PASSWORD" | docker login "$DOCKER_REGISTRY" -u "$DOCKER_USERNAME" --password-stdin
    fi
    
    # Pull images
    docker pull "${DOCKER_REGISTRY}/freeagentics:${VERSION}"
    docker pull "${DOCKER_REGISTRY}/freeagentics-web:${VERSION}"
    
    # Tag as current
    docker tag "${DOCKER_REGISTRY}/freeagentics:${VERSION}" "freeagentics-api:current"
    docker tag "${DOCKER_REGISTRY}/freeagentics-web:${VERSION}" "freeagentics-web:current"
    
    log "Images pulled successfully"
}

# Database migration
run_migrations() {
    log "Running database migrations..."
    
    # Check current migration status
    current_revision=$(docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
        run --rm api alembic current 2>/dev/null | grep -oE '[a-f0-9]{12}' | head -1)
    
    log "Current database revision: $current_revision"
    
    # Run migrations
    docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
        run --rm api alembic upgrade head
    
    # Verify migration success
    new_revision=$(docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
        run --rm api alembic current 2>/dev/null | grep -oE '[a-f0-9]{12}' | head -1)
    
    log "Database migrated to revision: $new_revision"
    
    if [[ "$current_revision" == "$new_revision" ]]; then
        warning "No new migrations were applied"
    fi
}

# Blue-green deployment
deploy_blue_green() {
    log "Starting blue-green deployment..."
    
    # Determine current active color
    if docker ps | grep -q "freeagentics-blue"; then
        CURRENT_COLOR="blue"
        NEW_COLOR="green"
    else
        CURRENT_COLOR="green"
        NEW_COLOR="blue"
    fi
    
    log "Current active: $CURRENT_COLOR, deploying to: $NEW_COLOR"
    
    # Start new color environment
    export DEPLOYMENT_COLOR=$NEW_COLOR
    docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
        -f "$PROJECT_ROOT/docker-compose.${NEW_COLOR}.yml" \
        up -d
    
    # Wait for services to be healthy
    log "Waiting for services to be healthy..."
    sleep 10
    
    # Health check
    if ! "$DEPLOYMENT_DIR/health-check.sh" "$NEW_COLOR"; then
        error "Health check failed for $NEW_COLOR environment"
        
        # Rollback
        docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
            -f "$PROJECT_ROOT/docker-compose.${NEW_COLOR}.yml" \
            down
        exit 1
    fi
    
    # Switch traffic to new color
    log "Switching traffic to $NEW_COLOR..."
    "$DEPLOYMENT_DIR/switch-traffic.sh" "$NEW_COLOR"
    
    # Verify traffic switch
    sleep 5
    if ! "$DEPLOYMENT_DIR/verify-traffic.sh" "$NEW_COLOR"; then
        error "Traffic switch verification failed"
        
        # Rollback traffic
        "$DEPLOYMENT_DIR/switch-traffic.sh" "$CURRENT_COLOR"
        exit 1
    fi
    
    # Stop old color environment
    log "Stopping $CURRENT_COLOR environment..."
    export DEPLOYMENT_COLOR=$CURRENT_COLOR
    docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
        -f "$PROJECT_ROOT/docker-compose.${CURRENT_COLOR}.yml" \
        down
    
    log "Blue-green deployment completed successfully"
}

# Rolling update deployment
deploy_rolling_update() {
    log "Starting rolling update deployment..."
    
    # Scale up with new version
    docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
        up -d --scale api=6 --no-recreate
    
    # Get container IDs
    OLD_CONTAINERS=$(docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
        ps -q api | head -3)
    
    # Update containers one by one
    for container in $OLD_CONTAINERS; do
        log "Updating container: $container"
        
        # Remove from load balancer
        "$DEPLOYMENT_DIR/remove-from-lb.sh" "$container"
        
        # Stop old container
        docker stop "$container"
        docker rm "$container"
        
        # Wait for new container to be healthy
        sleep 30
        
        # Verify health
        if ! "$DEPLOYMENT_DIR/health-check.sh"; then
            error "Health check failed during rolling update"
            exit 1
        fi
    done
    
    # Scale back to normal
    docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
        up -d --scale api=3
    
    log "Rolling update completed successfully"
}

# Post-deployment tasks
post_deployment() {
    log "Running post-deployment tasks..."
    
    # Clear caches
    log "Clearing application caches..."
    docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
        exec -T redis redis-cli FLUSHDB
    
    # Warm up caches
    log "Warming up caches..."
    "$DEPLOYMENT_DIR/cache-warmer.sh"
    
    # Run smoke tests
    log "Running smoke tests..."
    if ! "$DEPLOYMENT_DIR/smoke-tests.sh"; then
        warning "Some smoke tests failed"
    fi
    
    # Update monitoring
    log "Updating monitoring dashboards..."
    "$DEPLOYMENT_DIR/update-monitoring.sh"
    
    # Send notifications
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Deployment completed successfully for version $VERSION\"}" \
            "$SLACK_WEBHOOK"
    fi
    
    log "Post-deployment tasks completed"
}

# Rollback procedure
rollback() {
    error "Rollback initiated..."
    
    # Stop current deployment
    docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" down
    
    # Restore backup images
    for service in api web worker; do
        if docker image inspect "freeagentics-${service}:backup" &> /dev/null; then
            docker tag "freeagentics-${service}:backup" "freeagentics-${service}:current"
        fi
    done
    
    # Start previous version
    docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" up -d
    
    # Rollback database if needed
    if [[ "${ROLLBACK_DB:-false}" == "true" ]]; then
        warning "Rolling back database migrations..."
        docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" \
            run --rm api alembic downgrade -1
    fi
    
    log "Rollback completed"
}

# Main deployment flow
main() {
    log "Starting FreeAgentics production deployment"
    log "Environment: $DEPLOYMENT_ENV"
    log "Version: $VERSION"
    
    # Set trap for rollback on error
    trap 'rollback' ERR
    
    # Run deployment steps
    pre_deployment_checks
    backup_current
    pull_images
    run_migrations
    
    # Choose deployment strategy
    case "${DEPLOYMENT_STRATEGY:-blue-green}" in
        "blue-green")
            deploy_blue_green
            ;;
        "rolling")
            deploy_rolling_update
            ;;
        *)
            error "Unknown deployment strategy: $DEPLOYMENT_STRATEGY"
            exit 1
            ;;
    esac
    
    post_deployment
    
    # Remove error trap
    trap - ERR
    
    log "Deployment completed successfully! 🎉"
    log "Version $VERSION is now live"
    
    # Deployment report
    echo -e "\n${GREEN}=== Deployment Summary ===${NC}"
    echo "Start Time: $(grep 'Starting FreeAgentics' "$LOG_FILE" | cut -d' ' -f1-2)"
    echo "End Time: $(date +'%Y-%m-%d %H:%M:%S')"
    echo "Version: $VERSION"
    echo "Strategy: ${DEPLOYMENT_STRATEGY:-blue-green}"
    echo "Log File: $LOG_FILE"
}

# Run main function
main "$@"