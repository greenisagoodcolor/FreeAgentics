#!/bin/bash
# Complete Production Deployment Script for FreeAgentics
# This script orchestrates the entire production deployment process

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-auto}"
DEPLOYMENT_STRATEGY="${DEPLOYMENT_STRATEGY:-blue-green}"
ENABLE_MONITORING="${ENABLE_MONITORING:-true}"
ENABLE_ISTIO="${ENABLE_ISTIO:-true}"
ENABLE_SECURITY="${ENABLE_SECURITY:-true}"
ENABLE_BACKUP="${ENABLE_BACKUP:-true}"
ENABLE_NOTIFICATIONS="${ENABLE_NOTIFICATIONS:-true}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
SKIP_SECURITY_SCAN="${SKIP_SECURITY_SCAN:-false}"
FORCE_DEPLOY="${FORCE_DEPLOY:-false}"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$SCRIPT_DIR/scripts/deployment"
K8S_DIR="$SCRIPT_DIR/k8s"
MONITORING_DIR="$SCRIPT_DIR/monitoring"
DOCS_DIR="$SCRIPT_DIR/docs"

# Logging
LOG_DIR="/var/log/freeagentics"
LOG_FILE="$LOG_DIR/complete-deployment-$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

# Deployment state
DEPLOYMENT_ID="$(date +%s)"
STATE_FILE="/tmp/freeagentics-deployment-$DEPLOYMENT_ID.json"

# Initialize deployment state
init_deployment_state() {
    cat > "$STATE_FILE" <<EOF
{
  "deployment_id": "$DEPLOYMENT_ID",
  "version": "$VERSION",
  "environment": "$ENVIRONMENT",
  "deployment_mode": "$DEPLOYMENT_MODE",
  "strategy": "$DEPLOYMENT_STRATEGY",
  "start_time": "$(date -Iseconds)",
  "status": "initializing",
  "phases": {
    "pre_checks": "pending",
    "build": "pending",
    "security_scan": "pending",
    "tests": "pending",
    "infrastructure": "pending",
    "application": "pending",
    "monitoring": "pending",
    "verification": "pending",
    "cleanup": "pending"
  },
  "rollback_data": {},
  "notifications_sent": []
}
EOF
}

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]${NC} $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $*" | tee -a "$LOG_FILE" >&2
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] [WARNING]${NC} $*" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]${NC} $*" | tee -a "$LOG_FILE"
}

debug() {
    echo -e "${PURPLE}[$(date +'%Y-%m-%d %H:%M:%S')] [DEBUG]${NC} $*" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

# Update deployment state
update_deployment_state() {
    local key="$1"
    local value="$2"

    jq --arg key "$key" --arg value "$value" '.[$key] = $value' "$STATE_FILE" > "$STATE_FILE.tmp"
    mv "$STATE_FILE.tmp" "$STATE_FILE"
}

update_phase_state() {
    local phase="$1"
    local state="$2"

    jq --arg phase "$phase" --arg state "$state" '.phases[$phase] = $state' "$STATE_FILE" > "$STATE_FILE.tmp"
    mv "$STATE_FILE.tmp" "$STATE_FILE"
}

# Get deployment state
get_deployment_state() {
    local key="$1"
    jq -r --arg key "$key" '.[$key]' "$STATE_FILE"
}

# Display banner
display_banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                      FreeAgentics Production Deployment                      ║
║                                                                              ║
║                    Complete Infrastructure & Application                     ║
║                           Deployment Automation                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Display deployment summary
display_deployment_info() {
    echo -e "${BLUE}=== Deployment Configuration ===${NC}"
    echo "Version: $VERSION"
    echo "Environment: $ENVIRONMENT"
    echo "Deployment Mode: $DEPLOYMENT_MODE"
    echo "Strategy: $DEPLOYMENT_STRATEGY"
    echo "Monitoring: $ENABLE_MONITORING"
    echo "Istio Service Mesh: $ENABLE_ISTIO"
    echo "Security Scanning: $([ "$SKIP_SECURITY_SCAN" = "true" ] && echo "Disabled" || echo "Enabled")"
    echo "Testing: $([ "$SKIP_TESTS" = "true" ] && echo "Disabled" || echo "Enabled")"
    echo "Backup: $ENABLE_BACKUP"
    echo "Notifications: $ENABLE_NOTIFICATIONS"
    echo "Dry Run: $DRY_RUN"
    echo "Log File: $LOG_FILE"
    echo "State File: $STATE_FILE"
    echo ""
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Starting pre-deployment checks..."
    update_phase_state "pre_checks" "running"

    # Check required tools
    local required_tools=("docker" "kubectl" "jq" "curl")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &>/dev/null; then
            error "$tool is required but not installed"
            exit 1
        fi
    done

    # Check deployment mode
    if [[ "$DEPLOYMENT_MODE" == "auto" ]]; then
        if kubectl cluster-info &>/dev/null; then
            DEPLOYMENT_MODE="kubernetes"
            info "Auto-detected Kubernetes deployment mode"
        elif docker info &>/dev/null; then
            DEPLOYMENT_MODE="docker"
            info "Auto-detected Docker deployment mode"
        else
            error "Cannot detect deployment mode"
            exit 1
        fi
        update_deployment_state "deployment_mode" "$DEPLOYMENT_MODE"
    fi

    # Check version
    if [[ "$VERSION" == "latest" ]] && [[ "$FORCE_DEPLOY" != "true" ]]; then
        warning "Deploying 'latest' version to production"
        read -p "Are you sure you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi

    # Check environment
    if [[ "$ENVIRONMENT" == "production" ]] && [[ "$FORCE_DEPLOY" != "true" ]]; then
        warning "Deploying to PRODUCTION environment"
        read -p "Are you sure you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi

    # Check system resources
    check_system_resources

    # Check external dependencies
    check_external_dependencies

    update_phase_state "pre_checks" "completed"
    success "Pre-deployment checks completed successfully"
}

# Check system resources
check_system_resources() {
    log "Checking system resources..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Check Kubernetes resources
        local nodes_ready=$(kubectl get nodes --no-headers | grep -c "Ready")
        local nodes_total=$(kubectl get nodes --no-headers | wc -l)

        if [[ $nodes_ready -lt $nodes_total ]]; then
            warning "Not all Kubernetes nodes are ready ($nodes_ready/$nodes_total)"
        fi

        # Check resource availability
        kubectl describe nodes | grep -A 5 "Allocated resources" | tee -a "$LOG_FILE"

    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        # Check Docker resources
        local available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
        local available_disk=$(df -BG /var/lib/docker | awk 'NR==2 {print $4}' | sed 's/G//')

        if [[ $available_memory -lt 2048 ]]; then
            warning "Low available memory: ${available_memory}MB"
        fi

        if [[ $available_disk -lt 10 ]]; then
            warning "Low available disk space: ${available_disk}GB"
        fi
    fi

    log "System resources check completed"
}

# Check external dependencies
check_external_dependencies() {
    log "Checking external dependencies..."

    # Check Docker registry
    if ! docker pull alpine:latest &>/dev/null; then
        error "Cannot pull images from Docker registry"
        exit 1
    fi

    # Check DNS resolution
    if ! nslookup google.com &>/dev/null; then
        error "DNS resolution not working"
        exit 1
    fi

    # Check internet connectivity
    if ! curl -s --max-time 10 https://google.com &>/dev/null; then
        error "Internet connectivity not available"
        exit 1
    fi

    log "External dependencies check completed"
}

# Build and push images
build_and_push_images() {
    log "Building and pushing Docker images..."
    update_phase_state "build" "running"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN: Skipping image build and push"
        update_phase_state "build" "skipped"
        return
    fi

    # Build backend image
    log "Building backend image..."
    docker build -t "freeagentics/backend:$VERSION" -f Dockerfile.production .

    # Build frontend image
    log "Building frontend image..."
    docker build -t "freeagentics/frontend:$VERSION" -f web/Dockerfile.production web/

    # Push images if registry is configured
    if [[ -n "${DOCKER_REGISTRY:-}" ]]; then
        log "Pushing images to registry..."

        # Login to registry
        if [[ -n "${DOCKER_USERNAME:-}" && -n "${DOCKER_PASSWORD:-}" ]]; then
            echo "$DOCKER_PASSWORD" | docker login "$DOCKER_REGISTRY" -u "$DOCKER_USERNAME" --password-stdin
        fi

        # Tag and push images
        docker tag "freeagentics/backend:$VERSION" "$DOCKER_REGISTRY/freeagentics/backend:$VERSION"
        docker tag "freeagentics/frontend:$VERSION" "$DOCKER_REGISTRY/freeagentics/frontend:$VERSION"

        docker push "$DOCKER_REGISTRY/freeagentics/backend:$VERSION"
        docker push "$DOCKER_REGISTRY/freeagentics/frontend:$VERSION"
    fi

    update_phase_state "build" "completed"
    success "Images built and pushed successfully"
}

# Security scanning
security_scan() {
    if [[ "$SKIP_SECURITY_SCAN" == "true" ]]; then
        log "Skipping security scan"
        update_phase_state "security_scan" "skipped"
        return
    fi

    log "Running security scans..."
    update_phase_state "security_scan" "running"

    # Scan Docker images
    if command -v trivy &>/dev/null; then
        log "Scanning Docker images with Trivy..."
        trivy image --exit-code 0 --severity HIGH,CRITICAL "freeagentics/backend:$VERSION" | tee -a "$LOG_FILE"
        trivy image --exit-code 0 --severity HIGH,CRITICAL "freeagentics/frontend:$VERSION" | tee -a "$LOG_FILE"
    else
        warning "Trivy not installed, skipping image scanning"
    fi

    # Scan dependencies
    if [[ -f "requirements.txt" ]]; then
        log "Scanning Python dependencies..."
        if command -v safety &>/dev/null; then
            safety check -r requirements.txt | tee -a "$LOG_FILE"
        else
            warning "Safety not installed, skipping Python dependency scanning"
        fi
    fi

    if [[ -f "web/package.json" ]]; then
        log "Scanning Node.js dependencies..."
        cd web && npm audit --audit-level=high && cd ..
    fi

    # Run security tests
    if [[ -f "$SCRIPT_DIR/scripts/security/run_security_tests.py" ]]; then
        log "Running security tests..."
        python "$SCRIPT_DIR/scripts/security/run_security_tests.py" | tee -a "$LOG_FILE"
    fi

    update_phase_state "security_scan" "completed"
    success "Security scanning completed"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log "Skipping tests"
        update_phase_state "tests" "skipped"
        return
    fi

    log "Running tests..."
    update_phase_state "tests" "running"

    # Unit tests
    log "Running unit tests..."
    if [[ -f "pytest.ini" ]]; then
        python -m pytest tests/ -v --tb=short | tee -a "$LOG_FILE"
    else
        warning "No pytest configuration found, skipping unit tests"
    fi

    # Integration tests
    log "Running integration tests..."
    if [[ -f "$SCRIPT_DIR/scripts/run-integration-tests.sh" ]]; then
        "$SCRIPT_DIR/scripts/run-integration-tests.sh" | tee -a "$LOG_FILE"
    else
        warning "No integration tests found"
    fi

    # Performance tests
    log "Running performance tests..."
    if [[ -f "$SCRIPT_DIR/tests/performance/run_performance_monitoring.py" ]]; then
        python "$SCRIPT_DIR/tests/performance/run_performance_monitoring.py" | tee -a "$LOG_FILE"
    else
        warning "No performance tests found"
    fi

    update_phase_state "tests" "completed"
    success "Tests completed successfully"
}

# Deploy infrastructure
deploy_infrastructure() {
    log "Deploying infrastructure..."
    update_phase_state "infrastructure" "running"

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        log "Deploying to Kubernetes..."

        # Deploy base infrastructure
        kubectl apply -f "$K8S_DIR/namespace.yaml" || true
        kubectl apply -f "$K8S_DIR/persistent-volumes.yaml" || true
        kubectl apply -f "$K8S_DIR/secrets.yaml" || true
        kubectl apply -f "$K8S_DIR/postgres-deployment.yaml"
        kubectl apply -f "$K8S_DIR/redis-deployment.yaml"

        # Wait for database to be ready
        kubectl wait --for=condition=ready pod -l app=postgres -n freeagentics-prod --timeout=300s
        kubectl wait --for=condition=ready pod -l app=redis -n freeagentics-prod --timeout=300s

        # Deploy Istio service mesh
        if [[ "$ENABLE_ISTIO" == "true" ]]; then
            log "Deploying Istio service mesh..."
            kubectl apply -f "$K8S_DIR/istio-service-mesh.yaml" || true
        fi

        # Deploy enhanced autoscaling
        kubectl apply -f "$K8S_DIR/autoscaling-enhanced.yaml" || true

    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        log "Deploying with Docker Compose..."

        # Start infrastructure services
        docker-compose -f docker-compose.production.yml up -d postgres redis

        # Wait for services to be ready
        sleep 30

        # Check service health
        docker-compose -f docker-compose.production.yml ps
    fi

    update_phase_state "infrastructure" "completed"
    success "Infrastructure deployed successfully"
}

# Deploy application
deploy_application() {
    log "Deploying application..."
    update_phase_state "application" "running"

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Use enhanced Kubernetes deployment script
        "$K8S_DIR/deploy-k8s-enhanced.sh" \
            --version "$VERSION" \
            --strategy "$DEPLOYMENT_STRATEGY" \
            --namespace freeagentics-prod \
            $([ "$ENABLE_ISTIO" = "false" ] && echo "--no-istio") \
            $([ "$ENABLE_MONITORING" = "false" ] && echo "--no-monitoring")

    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        # Use zero-downtime deployment script
        "$DEPLOYMENT_DIR/zero-downtime-deploy.sh" \
            --version "$VERSION" \
            --strategy "$DEPLOYMENT_STRATEGY" \
            --env "$ENVIRONMENT" \
            --mode docker \
            $([ "$ENABLE_MONITORING" = "false" ] && echo "--no-monitoring") \
            $([ "$ENABLE_NOTIFICATIONS" = "false" ] && echo "--no-notifications") \
            $([ "$ENABLE_BACKUP" = "false" ] && echo "--no-backup")
    fi

    update_phase_state "application" "completed"
    success "Application deployed successfully"
}

# Deploy monitoring
deploy_monitoring() {
    if [[ "$ENABLE_MONITORING" != "true" ]]; then
        log "Skipping monitoring deployment"
        update_phase_state "monitoring" "skipped"
        return
    fi

    log "Deploying monitoring stack..."
    update_phase_state "monitoring" "running"

    # Deploy monitoring stack
    "$MONITORING_DIR/deploy-production-monitoring-enhanced.sh" \
        --env "$ENVIRONMENT" \
        --domain "${DOMAIN:-freeagentics.com}" \
        --cluster "${CLUSTER_NAME:-freeagentics-prod}"

    update_phase_state "monitoring" "completed"
    success "Monitoring stack deployed successfully"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    update_phase_state "verification" "running"

    # Health checks
    log "Running health checks..."
    run_health_checks

    # Smoke tests
    log "Running smoke tests..."
    run_smoke_tests

    # Performance validation
    log "Running performance validation..."
    run_performance_validation

    # Security validation
    log "Running security validation..."
    run_security_validation

    update_phase_state "verification" "completed"
    success "Deployment verification completed successfully"
}

# Health checks
run_health_checks() {
    local domain="${DOMAIN:-freeagentics.com}"
    local max_retries=10
    local retry_delay=10

    for endpoint in "/health" "/api/v1/health" "/api/v1/system/info"; do
        log "Checking endpoint: $endpoint"

        for ((i=1; i<=max_retries; i++)); do
            if curl -sf "https://$domain$endpoint" --max-time 30 >/dev/null; then
                log "✓ $endpoint is healthy"
                break
            elif [[ $i -eq $max_retries ]]; then
                error "✗ $endpoint failed health check after $max_retries attempts"
                return 1
            else
                log "Attempt $i/$max_retries failed for $endpoint, retrying in ${retry_delay}s..."
                sleep $retry_delay
            fi
        done
    done

    log "All health checks passed"
}

# Smoke tests
run_smoke_tests() {
    if [[ -f "$DEPLOYMENT_DIR/smoke-tests.sh" ]]; then
        "$DEPLOYMENT_DIR/smoke-tests.sh"
    else
        warning "No smoke tests found, skipping"
    fi
}

# Performance validation
run_performance_validation() {
    local domain="${DOMAIN:-freeagentics.com}"

    # Check response times
    log "Checking response times..."
    local response_time=$(curl -o /dev/null -s -w '%{time_total}' "https://$domain/api/v1/health")

    if (( $(echo "$response_time > 1.0" | bc -l) )); then
        warning "High response time: ${response_time}s"
    else
        log "Response time acceptable: ${response_time}s"
    fi

    # Check if monitoring is collecting metrics
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        log "Checking metrics collection..."
        if curl -sf "https://$domain/prometheus/api/v1/query?query=up" --max-time 10 >/dev/null; then
            log "✓ Metrics collection is working"
        else
            warning "✗ Metrics collection might not be working"
        fi
    fi
}

# Security validation
run_security_validation() {
    local domain="${DOMAIN:-freeagentics.com}"

    # Check SSL certificate
    log "Checking SSL certificate..."
    local cert_info=$(echo | openssl s_client -servername "$domain" -connect "$domain:443" 2>/dev/null | openssl x509 -noout -dates)
    log "Certificate info: $cert_info"

    # Check security headers
    log "Checking security headers..."
    local headers=$(curl -sI "https://$domain" | grep -i "strict-transport-security\|x-frame-options\|x-content-type-options\|x-xss-protection")
    if [[ -n "$headers" ]]; then
        log "✓ Security headers present"
    else
        warning "✗ Security headers missing"
    fi

    # Check for basic authentication
    log "Checking authentication..."
    local auth_response=$(curl -s -o /dev/null -w '%{http_code}' "https://$domain/api/v1/auth/me")
    if [[ "$auth_response" == "401" ]]; then
        log "✓ Authentication is working"
    else
        warning "✗ Authentication might not be working properly"
    fi
}

# Cleanup
cleanup() {
    log "Running cleanup tasks..."
    update_phase_state "cleanup" "running"

    # Clean up build artifacts
    if [[ "$DRY_RUN" != "true" ]]; then
        docker image prune -f
        docker builder prune -f
    fi

    # Remove temporary files
    rm -f /tmp/deployment-*.tmp

    # Archive logs
    if [[ -d "$LOG_DIR" ]]; then
        find "$LOG_DIR" -name "*.log" -mtime +7 -exec gzip {} \;
        find "$LOG_DIR" -name "*.log.gz" -mtime +30 -delete
    fi

    update_phase_state "cleanup" "completed"
    success "Cleanup completed"
}

# Send notifications
send_notifications() {
    if [[ "$ENABLE_NOTIFICATIONS" != "true" ]]; then
        return
    fi

    log "Sending deployment notifications..."

    local status=$(get_deployment_state "status")
    local duration=$(get_deployment_duration)

    local message="🚀 FreeAgentics deployment completed!
**Status**: $status
**Version**: $VERSION
**Environment**: $ENVIRONMENT
**Strategy**: $DEPLOYMENT_STRATEGY
**Duration**: $duration
**Log**: $LOG_FILE"

    # Send Slack notification
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$SLACK_WEBHOOK"
    fi

    # Send Teams notification
    if [[ -n "${TEAMS_WEBHOOK:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$TEAMS_WEBHOOK"
    fi

    # Send email notification
    if [[ -n "${EMAIL_TO:-}" ]]; then
        echo "$message" | mail -s "FreeAgentics Deployment Complete" "$EMAIL_TO"
    fi

    log "Notifications sent"
}

# Get deployment duration
get_deployment_duration() {
    local start_time=$(get_deployment_state "start_time")
    local end_time=$(date -Iseconds)
    local duration_seconds=$(( $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) ))

    if [[ $duration_seconds -lt 60 ]]; then
        echo "${duration_seconds}s"
    elif [[ $duration_seconds -lt 3600 ]]; then
        echo "$((duration_seconds / 60))m $((duration_seconds % 60))s"
    else
        echo "$((duration_seconds / 3600))h $(((duration_seconds % 3600) / 60))m"
    fi
}

# Generate deployment report
generate_deployment_report() {
    log "Generating deployment report..."

    local report_file="$LOG_DIR/deployment-report-$DEPLOYMENT_ID.json"

    # Add final state to deployment state file
    update_deployment_state "end_time" "$(date -Iseconds)"
    update_deployment_state "duration" "$(get_deployment_duration)"
    update_deployment_state "log_file" "$LOG_FILE"
    update_deployment_state "report_file" "$report_file"

    # Copy state file to report
    cp "$STATE_FILE" "$report_file"

    log "Deployment report generated: $report_file"
}

# Error handling
handle_error() {
    local exit_code=$?
    local line_number=$1

    error "Deployment failed at line $line_number with exit code $exit_code"
    update_deployment_state "status" "failed"
    update_deployment_state "error_line" "$line_number"
    update_deployment_state "error_code" "$exit_code"

    # Send failure notification
    if [[ "$ENABLE_NOTIFICATIONS" == "true" ]]; then
        local message="❌ FreeAgentics deployment failed!
**Version**: $VERSION
**Environment**: $ENVIRONMENT
**Error**: Line $line_number, Exit code $exit_code
**Log**: $LOG_FILE"

        if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
            curl -X POST -H 'Content-type: application/json' \
                --data "{\"text\":\"$message\"}" \
                "$SLACK_WEBHOOK"
        fi
    fi

    log "Deployment failed. Check logs for details."
    exit $exit_code
}

# Main deployment function
main() {
    # Set error trap
    trap 'handle_error $LINENO' ERR

    # Initialize
    init_deployment_state
    display_banner
    display_deployment_info

    # Deployment phases
    pre_deployment_checks
    build_and_push_images
    security_scan
    run_tests
    deploy_infrastructure
    deploy_application
    deploy_monitoring
    verify_deployment
    cleanup

    # Finalize
    update_deployment_state "status" "completed"
    generate_deployment_report
    send_notifications

    # Success message
    success "🎉 FreeAgentics deployment completed successfully!"
    success "Version $VERSION is now live in $ENVIRONMENT"
    success "Deployment took $(get_deployment_duration)"
    success "Log file: $LOG_FILE"
    success "Report file: $LOG_DIR/deployment-report-$DEPLOYMENT_ID.json"
}

# Show help
show_help() {
    cat << EOF
FreeAgentics Complete Production Deployment Script

Usage: $0 [OPTIONS]

Options:
  --version VERSION         Application version to deploy (required)
  --env ENV                 Environment: production|staging|development (default: production)
  --mode MODE               Deployment mode: auto|kubernetes|docker (default: auto)
  --strategy STRATEGY       Deployment strategy: blue-green|canary|rolling (default: blue-green)
  --domain DOMAIN           Domain name for the application
  --cluster CLUSTER         Kubernetes cluster name
  --no-monitoring           Disable monitoring deployment
  --no-istio               Disable Istio service mesh
  --no-security            Disable security features
  --no-backup              Disable backup creation
  --no-notifications       Disable notifications
  --skip-tests             Skip running tests
  --skip-security-scan     Skip security scanning
  --dry-run                Perform dry run without actual deployment
  --force                  Force deployment without confirmations
  --help                   Show this help message

Environment Variables:
  DOCKER_REGISTRY          Docker registry URL
  DOCKER_USERNAME          Docker registry username
  DOCKER_PASSWORD          Docker registry password
  SLACK_WEBHOOK           Slack webhook URL for notifications
  TEAMS_WEBHOOK           Microsoft Teams webhook URL
  EMAIL_TO                Email address for notifications
  DOMAIN                  Domain name for the application
  CLUSTER_NAME            Kubernetes cluster name

Examples:
  $0 --version v1.2.3 --env production
  $0 --version v1.2.3 --env staging --mode docker
  $0 --version v1.2.3 --strategy canary --no-istio
  $0 --version v1.2.3 --dry-run --skip-tests

For more information, see the documentation at:
  docs/production/PRODUCTION_OPERATIONS_RUNBOOK.md
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --mode)
            DEPLOYMENT_MODE="$2"
            shift 2
            ;;
        --strategy)
            DEPLOYMENT_STRATEGY="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --cluster)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        --no-monitoring)
            ENABLE_MONITORING="false"
            shift
            ;;
        --no-istio)
            ENABLE_ISTIO="false"
            shift
            ;;
        --no-security)
            ENABLE_SECURITY="false"
            shift
            ;;
        --no-backup)
            ENABLE_BACKUP="false"
            shift
            ;;
        --no-notifications)
            ENABLE_NOTIFICATIONS="false"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --skip-security-scan)
            SKIP_SECURITY_SCAN="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --force)
            FORCE_DEPLOY="true"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$VERSION" ]]; then
    error "Version is required. Use --version to specify."
    show_help
    exit 1
fi

# Run main function
main "$@"
