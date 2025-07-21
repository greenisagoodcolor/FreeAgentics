#!/bin/bash
# FreeAgentics Production Deployment Script
# Automated deployment with zero-downtime, health checks, and rollback capability

set -euo pipefail

# Configuration
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.production.yml}"
BACKUP_BEFORE_DEPLOY="${BACKUP_BEFORE_DEPLOY:-true}"
HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-10}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-30}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Function to send notifications
send_notification() {
    local message="$1"
    local status="${2:-INFO}"
    local emoji="📢"

    case $status in
        "SUCCESS") emoji="✅" ;;
        "ERROR") emoji="❌" ;;
        "WARNING") emoji="⚠️" ;;
        "START") emoji="🚀" ;;
        "ROLLBACK") emoji="🔙" ;;
    esac

    log "$message"

    if [[ -n "$SLACK_WEBHOOK" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$emoji FreeAgentics Deployment [$ENVIRONMENT]: $message\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null 
    fi
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."

    # Check if running as appropriate user
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root. Consider using a non-root user with docker group membership."
    fi

    # Check required commands
    local required_commands=("docker" "docker-compose" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error "Required command '$cmd' not found"
            exit 1
        fi
    done

    # Check if Docker is running
    if ! docker info &> /dev/null; then
        error "Docker is not running"
        exit 1
    fi

    # Check if compose file exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        error "Compose file '$COMPOSE_FILE' not found"
        exit 1
    fi

    # Check if environment file exists
    if [[ ! -f ".env.${ENVIRONMENT}" ]]; then
        error "Environment file '.env.${ENVIRONMENT}' not found"
        exit 1
    fi

    log "Prerequisites check completed successfully"
}

# Function to backup database
backup_database() {
    if [[ "$BACKUP_BEFORE_DEPLOY" != "true" ]]; then
        info "Database backup skipped (BACKUP_BEFORE_DEPLOY=false)"
        return 0
    fi

    log "Creating database backup before deployment..."

    # Ensure backup script exists
    if [[ ! -f "scripts/database-backup.sh" ]]; then
        warn "Database backup script not found, skipping backup"
        return 0
    fi

    # Run backup
    if ./scripts/database-backup.sh backup; then
        log "Database backup completed successfully"
    else
        error "Database backup failed"
        return 1
    fi
}

# Function to build images
build_images() {
    log "Building Docker images for version $VERSION..."

    # Set environment
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1

    # Build images with version tag
    if docker-compose -f "$COMPOSE_FILE" build \
        --parallel \
        --build-arg VERSION="$VERSION" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"; then
        log "Docker images built successfully"
    else
        error "Failed to build Docker images"
        return 1
    fi

    # Tag images with version
    log "Tagging images with version $VERSION..."
    docker-compose -f "$COMPOSE_FILE" images --format json | jq -r '.Repository + ":" + .Tag' | while read -r image; do
        if [[ "$image" != "null:null" ]]; then
            local base_name
            base_name=$(echo "$image" | cut -d':' -f1)
            docker tag "$image" "${base_name}:${VERSION}" 
        fi
    done
}

# Function to run pre-deployment tests
run_pre_deployment_tests() {
    log "Running pre-deployment tests..."

    # Test database migration
    log "Testing database migration..."
    if docker-compose -f "$COMPOSE_FILE" run --rm migration alembic check 2>/dev/null; then
        log "Database migration test passed"
    else
        warn "Database migration test failed or not applicable"
    fi

    # Test configuration
    log "Testing application configuration..."
    if docker-compose -f "$COMPOSE_FILE" config >/dev/null; then
        log "Docker Compose configuration is valid"
    else
        error "Docker Compose configuration is invalid"
        return 1
    fi

    # Additional tests can be added here
    log "Pre-deployment tests completed"
}

# Function to perform zero-downtime deployment
deploy_zero_downtime() {
    log "Starting zero-downtime deployment..."

    # Check if services are already running
    local running_services
    running_services=$(docker-compose -f "$COMPOSE_FILE" ps --services --filter "status=running" | wc -l)

    if [[ "$running_services" -gt 0 ]]; then
        log "Performing rolling update of existing services..."

        # Update services one by one
        local services=("backend" "frontend" "nginx")
        for service in "${services[@]}"; do
            log "Updating $service..."

            # Scale up new instance
            docker-compose -f "$COMPOSE_FILE" up -d --scale "$service"=2 "$service"

            # Wait for new instance to be healthy
            sleep 30

            # Scale down to 1 (removes old instance)
            docker-compose -f "$COMPOSE_FILE" up -d --scale "$service"=1 "$service"

            log "$service updated successfully"
        done
    else
        log "No existing services found, performing fresh deployment..."
        docker-compose -f "$COMPOSE_FILE" up -d
    fi
}

# Function to run database migrations
run_migrations() {
    log "Running database migrations..."

    # Check if database is accessible
    if ! docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U freeagentics; then
        error "Database is not accessible"
        return 1
    fi

    # Run migrations
    if docker-compose -f "$COMPOSE_FILE" run --rm migration; then
        log "Database migrations completed successfully"
    else
        error "Database migrations failed"
        return 1
    fi
}

# Function to wait for services to be healthy
wait_for_health() {
    log "Waiting for services to become healthy..."

    local retries=0
    local max_retries="$HEALTH_CHECK_RETRIES"
    local interval="$HEALTH_CHECK_INTERVAL"

    while [[ $retries -lt $max_retries ]]; do
        local healthy_services=0
        local total_services=0

        # Check each service health
        while IFS= read -r service; do
            if [[ -n "$service" ]]; then
                total_services=$((total_services + 1))

                # Check if service has health check
                local health_status
                health_status=$(docker inspect "$(docker-compose -f "$COMPOSE_FILE" ps -q "$service" | head -1)" \
                    --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")

                if [[ "$health_status" == "healthy" ]]; then
                    healthy_services=$((healthy_services + 1))
                elif [[ "$health_status" == "unknown" ]]; then
                    # For services without health check, check if running
                    if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
                        healthy_services=$((healthy_services + 1))
                    fi
                fi
            fi
        done < <(docker-compose -f "$COMPOSE_FILE" ps --services)

        info "Health check: $healthy_services/$total_services services healthy"

        if [[ $healthy_services -eq $total_services ]] && [[ $total_services -gt 0 ]]; then
            log "All services are healthy"
            return 0
        fi

        retries=$((retries + 1))
        if [[ $retries -lt $max_retries ]]; then
            info "Waiting ${interval}s before next health check (attempt $retries/$max_retries)..."
            sleep "$interval"
        fi
    done

    error "Services did not become healthy within expected time"
    return 1
}

# Function to run post-deployment tests
run_post_deployment_tests() {
    log "Running post-deployment tests..."

    # Test API endpoints
    local api_url="http://localhost:8000"

    # Test health endpoint
    if curl -f -s "$api_url/health" >/dev/null; then
        log "API health check passed"
    else
        error "API health check failed"
        return 1
    fi

    # Test authentication endpoint
    if curl -f -s "$api_url/api/v1/auth/health" >/dev/null; then
        log "Authentication endpoint test passed"
    else
        warn "Authentication endpoint test failed (may be expected)"
    fi

    # Test database connectivity
    if docker-compose -f "$COMPOSE_FILE" exec -T backend python -c "
from database.session import get_db
try:
    db = next(get_db())
    db.execute('SELECT 1')
    print('Database connectivity OK')
except Exception as e:
    print(f'Database connectivity failed: {e}')
    exit(1)
"; then
        log "Database connectivity test passed"
    else
        error "Database connectivity test failed"
        return 1
    fi

    log "Post-deployment tests completed successfully"
}

# Function to rollback deployment
rollback_deployment() {
    log "Starting deployment rollback..."

    # Get previous version from Git
    local previous_version
    previous_version=$(git describe --tags --abbrev=0 HEAD~1 2>/dev/null || echo "previous")

    warn "Rolling back to version: $previous_version"

    # Stop current deployment
    docker-compose -f "$COMPOSE_FILE" down

    # Checkout previous version (if using Git)
    if git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
        git checkout HEAD~1
    fi

    # Restore from backup if available
    local latest_backup
    latest_backup=$(ls -t /var/backups/freeagentics/freeagentics_backup_*.sql.gz 2>/dev/null | head -1 || echo "")

    if [[ -n "$latest_backup" ]]; then
        warn "Restoring database from backup: $latest_backup"
        ./scripts/database-backup.sh restore "$latest_backup" 
    fi

    # Deploy previous version
    docker-compose -f "$COMPOSE_FILE" up -d

    # Wait for rollback to be healthy
    if wait_for_health; then
        send_notification "Rollback completed successfully to version $previous_version" "SUCCESS"
    else
        send_notification "Rollback failed - manual intervention required" "ERROR"
        exit 1
    fi
}

# Function to cleanup old images
cleanup_old_images() {
    log "Cleaning up old Docker images..."

    # Remove unused images older than 7 days
    docker image prune -f --filter "until=168h" 

    # Remove dangling images
    docker image prune -f 

    log "Docker cleanup completed"
}

# Function to display deployment summary
show_deployment_summary() {
    log "=== DEPLOYMENT SUMMARY ==="
    log "Environment: $ENVIRONMENT"
    log "Version: $VERSION"
    log "Compose File: $COMPOSE_FILE"
    log "Timestamp: $(date)"

    # Show running services
    log "Running Services:"
    docker-compose -f "$COMPOSE_FILE" ps

    # Show resource usage
    log "Resource Usage:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 
}

# Main deployment function
main() {
    local start_time
    start_time=$(date +%s)

    send_notification "Starting deployment of version $VERSION" "START"

    # Trap for cleanup on exit
    trap 'echo "Deployment interrupted"; exit 1' INT TERM

    # Pre-deployment steps
    check_prerequisites
    backup_database || {
        error "Backup failed - aborting deployment"
        exit 1
    }

    # Build and test
    build_images || {
        error "Image build failed - aborting deployment"
        exit 1
    }

    run_pre_deployment_tests || {
        error "Pre-deployment tests failed - aborting deployment"
        exit 1
    }

    # Deployment
    run_migrations || {
        error "Database migrations failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback_deployment
        fi
        exit 1
    }

    deploy_zero_downtime || {
        error "Deployment failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback_deployment
        fi
        exit 1
    }

    # Post-deployment verification
    wait_for_health || {
        error "Health checks failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback_deployment
        fi
        exit 1
    }

    run_post_deployment_tests || {
        error "Post-deployment tests failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback_deployment
        fi
        exit 1
    }

    # Cleanup and summary
    cleanup_old_images
    show_deployment_summary

    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))

    send_notification "Deployment completed successfully in ${duration}s" "SUCCESS"
    log "🎉 Deployment completed successfully!"
}

# Script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -v, --version VERSION     Set deployment version (default: latest)"
    echo "  -e, --environment ENV     Set environment (default: production)"
    echo "  -f, --file FILE          Set compose file (default: docker-compose.production.yml)"
    echo "  --no-backup              Skip database backup"
    echo "  --no-rollback            Disable automatic rollback on failure"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  SLACK_WEBHOOK            Slack webhook URL for notifications"
    echo "  HEALTH_CHECK_RETRIES     Number of health check retries (default: 10)"
    echo "  HEALTH_CHECK_INTERVAL    Health check interval in seconds (default: 30)"
    echo ""
    echo "Examples:"
    echo "  $0                       # Deploy latest version to production"
    echo "  $0 -v v1.2.3            # Deploy specific version"
    echo "  $0 -e staging           # Deploy to staging environment"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            COMPOSE_FILE="docker-compose.${2}.yml"
            shift 2
            ;;
        -f|--file)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        --no-backup)
            BACKUP_BEFORE_DEPLOY=false
            shift
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute main function
main "$@"
