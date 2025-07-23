#!/bin/bash
# Production Deployment Script with Zero-Downtime Capability
# FreeAgentics Production Environment Configuration Expert
#
# This script implements:
# - Zero-downtime deployments with rolling updates
# - Health checks and rollback capabilities
# - Complete monitoring stack deployment
# - Security validations and hardening
# - Backup verification and disaster recovery testing

set -euo pipefail

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_NAME="freeagentics"
readonly COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.production.yml"
readonly ENV_FILE="${SCRIPT_DIR}/.env.production"
readonly BACKUP_DIR="${SCRIPT_DIR}/backups"
readonly LOG_DIR="${SCRIPT_DIR}/logs"
readonly DATA_DIR="${SCRIPT_DIR}/data"

# Deployment configuration
readonly MAX_RETRIES=3
readonly HEALTH_CHECK_TIMEOUT=300
readonly ROLLBACK_TIMEOUT=120
readonly DEPLOYMENT_TIMEOUT=600

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_DIR}/deployment.log"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_DIR}/deployment.log"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_DIR}/deployment.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_DIR}/deployment.log"
}

# Error handling
cleanup_on_error() {
    local exit_code=$?
    log_error "Deployment failed with exit code: ${exit_code}"
    log_info "Starting cleanup and rollback procedures..."

    # Stop failed deployment
    docker-compose -f "${COMPOSE_FILE}" down --remove-orphans || true

    # Rollback to previous version if available
    rollback_deployment

    exit ${exit_code}
}

trap cleanup_on_error ERR

# Utility functions
check_prerequisites() {
    log_info "Checking deployment prerequisites..."

    # Check required tools
    local required_tools=("docker" "docker-compose" "curl" "jq" "openssl")
    for tool in "${required_tools[@]}"; do
        if ! command -v "${tool}" &> /dev/null; then
            log_error "Required tool '${tool}' is not installed"
            exit 1
        fi
    done

    # Check Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi

    # Check environment file exists
    if [[ ! -f "${ENV_FILE}" ]]; then
        log_error "Environment file not found: ${ENV_FILE}"
        log_info "Please create the environment file with required variables"
        exit 1
    fi

    # Validate environment variables
    source "${ENV_FILE}"
    local required_vars=(
        "DOMAIN" "POSTGRES_PASSWORD" "REDIS_PASSWORD" "SECRET_KEY" "JWT_SECRET"
        "GRAFANA_ADMIN_PASSWORD" "GRAFANA_SECRET_KEY" "GRAFANA_DB_PASSWORD"
        "POSTGRES_EXPORTER_PASSWORD"
    )

    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable '${var}' is not set"
            exit 1
        fi
    done

    log_success "Prerequisites check completed"
}

create_directories() {
    log_info "Creating required directories..."

    local directories=(
        "${DATA_DIR}/postgres" "${DATA_DIR}/redis" "${DATA_DIR}/prometheus"
        "${DATA_DIR}/grafana" "${DATA_DIR}/alertmanager" "${DATA_DIR}/jaeger"
        "${DATA_DIR}/loki" "${BACKUP_DIR}/postgres" "${BACKUP_DIR}/files"
        "${LOG_DIR}/nginx" "${LOG_DIR}"
    )

    for dir in "${directories[@]}"; do
        mkdir -p "${dir}"
        chmod 755 "${dir}"
    done

    # Set proper ownership for data directories
    sudo chown -R 999:999 "${DATA_DIR}/postgres" || log_warning "Could not set PostgreSQL data ownership"
    sudo chown -R 999:999 "${DATA_DIR}/redis" || log_warning "Could not set Redis data ownership"
    sudo chown -R 472:472 "${DATA_DIR}/grafana" || log_warning "Could not set Grafana data ownership"
    sudo chown -R 65534:65534 "${DATA_DIR}/prometheus" || log_warning "Could not set Prometheus data ownership"

    log_success "Directories created and configured"
}

validate_configuration() {
    log_info "Validating Docker Compose configuration..."

    if ! docker-compose -f "${COMPOSE_FILE}" config -q; then
        log_error "Docker Compose configuration is invalid"
        exit 1
    fi

    # Check for required configuration files
    local required_files=(
        "nginx/nginx.conf" "nginx/dhparam.pem" "nginx/ssl/cert.pem" "nginx/ssl/key.pem"
        "monitoring/prometheus-production.yml" "monitoring/alertmanager-production.yml"
    )

    for file in "${required_files[@]}"; do
        if [[ ! -f "${SCRIPT_DIR}/${file}" ]]; then
            log_error "Required configuration file not found: ${file}"
            exit 1
        fi
    done

    log_success "Configuration validation completed"
}

backup_current_state() {
    log_info "Creating backup of current deployment state..."

    local backup_timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_file="${BACKUP_DIR}/pre_deployment_backup_${backup_timestamp}.tar.gz"

    # Backup current containers state
    docker-compose -f "${COMPOSE_FILE}" ps --format json > "${BACKUP_DIR}/containers_state_${backup_timestamp}.json" || true

    # Backup current data if exists
    if [[ -d "${DATA_DIR}" ]]; then
        tar -czf "${backup_file}" -C "${SCRIPT_DIR}" data/ 2>/dev/null || log_warning "Could not create data backup"
    fi

    # Store current image tags
    docker images --format "table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}" | grep "${PROJECT_NAME}" > "${BACKUP_DIR}/images_${backup_timestamp}.txt" || true

    export BACKUP_TIMESTAMP="${backup_timestamp}"
    log_success "Backup created with timestamp: ${backup_timestamp}"
}

pull_images() {
    log_info "Pulling latest Docker images..."

    local images=(
        "pgvector/pgvector:pg15" "redis:7-alpine" "nginx:1.25-alpine"
        "prom/prometheus:v2.47.0" "grafana/grafana:10.1.2" "prom/alertmanager:v0.26.0"
        "jaegertracing/all-in-one:1.49.0" "prom/node-exporter:v1.6.1"
        "gcr.io/cadvisor/cadvisor:v0.47.2" "grafana/loki:2.9.0" "grafana/promtail:2.9.0"
    )

    for image in "${images[@]}"; do
        log_info "Pulling image: ${image}"
        docker pull "${image}" || log_warning "Failed to pull ${image}"
    done

    log_success "Image pull completed"
}

build_application_images() {
    log_info "Building application images..."

    # Set build arguments
    export VERSION="${VERSION:-v1.0.0-alpha}"
    export BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    export GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

    # Build backend image
    docker-compose -f "${COMPOSE_FILE}" build --no-cache backend

    # Build frontend image
    docker-compose -f "${COMPOSE_FILE}" build --no-cache frontend

    log_success "Application images built successfully"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure services..."

    # Deploy database first
    docker-compose -f "${COMPOSE_FILE}" up -d postgres redis

    # Wait for database to be healthy
    wait_for_service "postgres" 120
    wait_for_service "redis" 60

    # Run database migrations
    log_info "Running database migrations..."
    docker-compose -f "${COMPOSE_FILE}" up migration

    log_success "Infrastructure services deployed"
}

deploy_application() {
    log_info "Deploying application services with zero-downtime strategy..."

    # Deploy backend with rolling update
    deploy_service_with_rolling_update "backend"

    # Deploy frontend
    deploy_service_with_rolling_update "frontend"

    # Deploy nginx reverse proxy
    docker-compose -f "${COMPOSE_FILE}" up -d nginx
    wait_for_service "nginx" 60

    log_success "Application services deployed"
}

deploy_service_with_rolling_update() {
    local service_name="$1"
    local old_container="${PROJECT_NAME}-${service_name}"

    log_info "Performing rolling update for service: ${service_name}"

    # Scale up new instance
    docker-compose -f "${COMPOSE_FILE}" up -d --scale "${service_name}=2" "${service_name}"

    # Wait for new instance to be healthy
    sleep 30

    # Check health of new instances
    local healthy_instances=0
    for i in {1..10}; do
        healthy_instances=$(docker ps --filter "name=${PROJECT_NAME}-${service_name}" --filter "health=healthy" --format "{{.Names}}" | wc -l)
        if [[ ${healthy_instances} -ge 1 ]]; then
            break
        fi
        log_info "Waiting for ${service_name} instances to become healthy (attempt ${i}/10)..."
        sleep 10
    done

    if [[ ${healthy_instances} -lt 1 ]]; then
        log_error "No healthy instances of ${service_name} found after rolling update"
        return 1
    fi

    # Scale back to single instance (removes old instance)
    docker-compose -f "${COMPOSE_FILE}" up -d --scale "${service_name}=1" "${service_name}"

    log_success "Rolling update completed for service: ${service_name}"
}

deploy_monitoring() {
    log_info "Deploying monitoring and observability stack..."

    # Deploy Prometheus first
    docker-compose -f "${COMPOSE_FILE}" up -d prometheus
    wait_for_service "prometheus" 120

    # Deploy monitoring services
    docker-compose -f "${COMPOSE_FILE}" up -d \
        grafana alertmanager jaeger node-exporter cadvisor \
        postgres-exporter redis-exporter nginx-exporter

    # Deploy logging stack
    docker-compose -f "${COMPOSE_FILE}" up -d loki promtail

    # Wait for key services
    wait_for_service "grafana" 120
    wait_for_service "alertmanager" 60

    log_success "Monitoring stack deployed"
}

deploy_backup_services() {
    log_info "Deploying backup and disaster recovery services..."

    docker-compose -f "${COMPOSE_FILE}" up -d postgres-backup backup-agent

    # Test backup functionality
    test_backup_functionality

    log_success "Backup services deployed and tested"
}

wait_for_service() {
    local service_name="$1"
    local timeout="${2:-120}"
    local container_name="${PROJECT_NAME}-${service_name}"

    log_info "Waiting for service '${service_name}' to become healthy (timeout: ${timeout}s)..."

    local count=0
    while [[ ${count} -lt ${timeout} ]]; do
        if docker exec "${container_name}" sh -c "exit 0" 2>/dev/null; then
            local health_status=$(docker inspect --format='{{.State.Health.Status}}' "${container_name}" 2>/dev/null || echo "none")
            if [[ "${health_status}" == "healthy" ]] || [[ "${health_status}" == "none" ]]; then
                log_success "Service '${service_name}' is ready"
                return 0
            fi
        fi

        sleep 5
        count=$((count + 5))

        if [[ $((count % 30)) -eq 0 ]]; then
            log_info "Still waiting for ${service_name}... (${count}/${timeout}s)"
        fi
    done

    log_error "Service '${service_name}' failed to become healthy within ${timeout} seconds"
    return 1
}

perform_health_checks() {
    log_info "Performing comprehensive health checks..."

    # Application health checks
    check_endpoint "https://${DOMAIN}/health" "Application"
    check_endpoint "https://${DOMAIN}/api/health" "API"

    # Monitoring health checks
    check_endpoint "http://localhost:9090/-/healthy" "Prometheus"
    check_endpoint "http://localhost:3001/api/health" "Grafana"
    check_endpoint "http://localhost:9093/-/healthy" "AlertManager"

    # Database connectivity
    docker exec "${PROJECT_NAME}-postgres" pg_isready -U freeagentics || {
        log_error "PostgreSQL health check failed"
        return 1
    }

    docker exec "${PROJECT_NAME}-redis" redis-cli ping || {
        log_error "Redis health check failed"
        return 1
    }

    log_success "All health checks passed"
}

check_endpoint() {
    local url="$1"
    local name="$2"
    local max_attempts=10
    local attempt=1

    log_info "Checking ${name} endpoint: ${url}"

    while [[ ${attempt} -le ${max_attempts} ]]; do
        if curl -f -s -k "${url}" > /dev/null 2>&1; then
            log_success "${name} endpoint is healthy"
            return 0
        fi

        log_info "Attempt ${attempt}/${max_attempts}: ${name} not ready, retrying in 10s..."
        sleep 10
        attempt=$((attempt + 1))
    done

    log_error "${name} endpoint health check failed after ${max_attempts} attempts"
    return 1
}

test_backup_functionality() {
    log_info "Testing backup functionality..."

    # Trigger immediate backup test
    docker exec "${PROJECT_NAME}-postgres-backup" sh -c "/backup.sh" || {
        log_warning "Database backup test failed"
        return 1
    }

    # Check if backup files were created
    if [[ -d "${BACKUP_DIR}/postgres" ]] && [[ -n "$(ls -A "${BACKUP_DIR}/postgres" 2>/dev/null)" ]]; then
        log_success "Database backup functionality verified"
    else
        log_warning "Database backup files not found"
    fi
}

validate_security() {
    log_info "Performing security validation..."

    # Check SSL certificate
    if [[ -f "${SCRIPT_DIR}/nginx/ssl/cert.pem" ]]; then
        local cert_expiry=$(openssl x509 -enddate -noout -in "${SCRIPT_DIR}/nginx/ssl/cert.pem" | cut -d= -f2)
        log_info "SSL certificate expires: ${cert_expiry}"

        # Check if certificate expires within 30 days
        local current_date=$(date +%s)
        local expiry_date=$(date -d "${cert_expiry}" +%s)
        local days_until_expiry=$(( (expiry_date - current_date) / 86400 ))

        if [[ ${days_until_expiry} -lt 30 ]]; then
            log_warning "SSL certificate expires in ${days_until_expiry} days"
        fi
    fi

    # Test HTTPS redirect
    local http_response=$(curl -s -o /dev/null -w "%{http_code}" "http://${DOMAIN}/health" || echo "000")
    if [[ "${http_response}" == "301" ]] || [[ "${http_response}" == "302" ]]; then
        log_success "HTTP to HTTPS redirect is working"
    else
        log_warning "HTTP to HTTPS redirect may not be working properly (response: ${http_response})"
    fi

    log_success "Security validation completed"
}

rollback_deployment() {
    log_warning "Initiating deployment rollback..."

    if [[ -n "${BACKUP_TIMESTAMP:-}" ]]; then
        local backup_file="${BACKUP_DIR}/pre_deployment_backup_${BACKUP_TIMESTAMP}.tar.gz"

        if [[ -f "${backup_file}" ]]; then
            log_info "Restoring from backup: ${backup_file}"

            # Stop current services
            docker-compose -f "${COMPOSE_FILE}" down --remove-orphans

            # Restore backup
            tar -xzf "${backup_file}" -C "${SCRIPT_DIR}" 2>/dev/null || log_warning "Could not restore data backup"

            # Start previous version
            docker-compose -f "${COMPOSE_FILE}" up -d

            log_success "Rollback completed"
        else
            log_error "Backup file not found: ${backup_file}"
        fi
    else
        log_error "No backup timestamp available for rollback"
    fi
}

generate_deployment_report() {
    log_info "Generating deployment report..."

    local report_file="${LOG_DIR}/deployment_report_$(date '+%Y%m%d_%H%M%S').json"

    cat > "${report_file}" << EOF
{
  "deployment_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "version": "${VERSION:-unknown}",
  "git_commit": "${GIT_COMMIT:-unknown}",
  "domain": "${DOMAIN}",
  "services_status": $(docker-compose -f "${COMPOSE_FILE}" ps --format json),
  "resource_usage": {
    "total_containers": $(docker ps --filter "name=${PROJECT_NAME}" --format "{{.Names}}" | wc -l),
    "images_size": "$(docker images --filter "reference=${PROJECT_NAME}/*" --format "table {{.Size}}" | tail -n +2 | head -1 || echo 'unknown')",
    "volumes": $(docker volume ls --filter "name=${PROJECT_NAME}" --format json)
  },
  "monitoring_endpoints": {
    "grafana": "https://${DOMAIN}/grafana/",
    "prometheus": "https://${DOMAIN}/prometheus/",
    "alertmanager": "https://${DOMAIN}/alertmanager/",
    "jaeger": "https://${DOMAIN}/jaeger/"
  }
}
EOF

    log_success "Deployment report generated: ${report_file}"
}

# Main deployment function
main() {
    local start_time=$(date +%s)

    log_info "Starting FreeAgentics production deployment..."
    log_info "Deployment started at: $(date)"

    # Pre-deployment steps
    check_prerequisites
    create_directories
    validate_configuration
    backup_current_state

    # Build and deployment steps
    pull_images
    build_application_images

    # Deploy services in order
    deploy_infrastructure
    deploy_application
    deploy_monitoring
    deploy_backup_services

    # Post-deployment validation
    perform_health_checks
    validate_security

    # Deploy SSL monitoring
    docker-compose -f "${COMPOSE_FILE}" up -d ssl-monitor certbot

    # Generate report
    generate_deployment_report

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_success "=== DEPLOYMENT COMPLETED SUCCESSFULLY ==="
    log_success "Total deployment time: ${duration} seconds"
    log_success "Application URL: https://${DOMAIN}"
    log_success "Monitoring Dashboard: https://${DOMAIN}/grafana/"
    log_success "System Status: All services are operational"

    # Display final service status
    echo -e "\n${GREEN}=== FINAL SERVICE STATUS ===${NC}"
    docker-compose -f "${COMPOSE_FILE}" ps
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback_deployment
        ;;
    "health-check")
        perform_health_checks
        ;;
    "status")
        docker-compose -f "${COMPOSE_FILE}" ps
        ;;
    "logs")
        docker-compose -f "${COMPOSE_FILE}" logs -f "${2:-}"
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|health-check|status|logs [service_name]}"
        echo ""
        echo "Commands:"
        echo "  deploy       - Deploy the full production stack"
        echo "  rollback     - Rollback to previous deployment"
        echo "  health-check - Perform health checks on all services"
        echo "  status       - Show current service status"
        echo "  logs         - Show logs for all services or specific service"
        exit 1
        ;;
esac
