#!/bin/bash
# Zero-Downtime Deployment Script for FreeAgentics
# Supports multiple deployment strategies with automatic rollback

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Configuration
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
DEPLOYMENT_STRATEGY="${DEPLOYMENT_STRATEGY:-blue-green}"
VERSION="${VERSION:-latest}"
PREVIOUS_VERSION="${PREVIOUS_VERSION:-}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-your-registry.com}"
NAMESPACE="${NAMESPACE:-freeagentics-prod}"
CLUSTER_NAME="${CLUSTER_NAME:-freeagentics-prod}"
DOMAIN="${DOMAIN:-freeagentics.com}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
ROLLBACK_TIMEOUT="${ROLLBACK_TIMEOUT:-600}"
TRAFFIC_SPLIT_DURATION="${TRAFFIC_SPLIT_DURATION:-300}"
CANARY_PERCENTAGE="${CANARY_PERCENTAGE:-10}"
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-auto}"  # auto, kubernetes, docker
ENABLE_MONITORING="${ENABLE_MONITORING:-true}"
ENABLE_NOTIFICATIONS="${ENABLE_NOTIFICATIONS:-true}"
ENABLE_BACKUP="${ENABLE_BACKUP:-true}"
ENABLE_SMOKE_TESTS="${ENABLE_SMOKE_TESTS:-true}"
DRY_RUN="${DRY_RUN:-false}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Logging
LOG_FILE="/var/log/freeagentics/zero-downtime-deploy-$(date +%Y%m%d_%H%M%S).log"
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

info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_FILE"
}

debug() {
    echo -e "${PURPLE}[DEBUG]${NC} $*" | tee -a "$LOG_FILE"
}

# Deployment state tracking
DEPLOYMENT_STATE_FILE="/tmp/freeagentics-deployment-state.json"
ROLLBACK_DATA_FILE="/tmp/freeagentics-rollback-data.json"

# Initialize deployment state
init_deployment_state() {
    cat > "$DEPLOYMENT_STATE_FILE" <<EOF
{
  "deployment_id": "$(date +%s)",
  "version": "$VERSION",
  "previous_version": "$PREVIOUS_VERSION",
  "strategy": "$DEPLOYMENT_STRATEGY",
  "environment": "$DEPLOYMENT_ENV",
  "start_time": "$(date -Iseconds)",
  "status": "initializing",
  "current_phase": "init",
  "rollback_available": false,
  "health_checks_passed": false,
  "smoke_tests_passed": false,
  "traffic_switched": false
}
EOF
}

# Update deployment state
update_deployment_state() {
    local key="$1"
    local value="$2"

    jq --arg key "$key" --arg value "$value" '.[$key] = $value' "$DEPLOYMENT_STATE_FILE" > "$DEPLOYMENT_STATE_FILE.tmp"
    mv "$DEPLOYMENT_STATE_FILE.tmp" "$DEPLOYMENT_STATE_FILE"
}

# Get deployment state
get_deployment_state() {
    local key="$1"
    jq -r --arg key "$key" '.[$key]' "$DEPLOYMENT_STATE_FILE"
}

# Auto-detect deployment mode
detect_deployment_mode() {
    if [[ "$DEPLOYMENT_MODE" == "auto" ]]; then
        if kubectl cluster-info &>/dev/null; then
            DEPLOYMENT_MODE="kubernetes"
            info "Auto-detected Kubernetes deployment mode"
        elif docker info &>/dev/null; then
            DEPLOYMENT_MODE="docker"
            info "Auto-detected Docker deployment mode"
        else
            error "Cannot detect deployment mode. Please specify --mode"
            exit 1
        fi
    fi
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    update_deployment_state "current_phase" "pre-checks"

    # Check required tools
    local required_tools=("jq" "curl")

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        required_tools+=("kubectl")
    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        required_tools+=("docker" "docker-compose")
    fi

    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &>/dev/null; then
            error "$tool is required but not installed"
            exit 1
        fi
    done

    # Check connectivity
    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        if ! kubectl cluster-info &>/dev/null; then
            error "Cannot connect to Kubernetes cluster"
            exit 1
        fi

        if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
            error "Namespace $NAMESPACE does not exist"
            exit 1
        fi
    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        if ! docker info &>/dev/null; then
            error "Docker daemon is not running"
            exit 1
        fi
    fi

    # Check if new version is different from current
    check_version_diff

    # Check system resources
    check_system_resources

    # Validate deployment configuration
    validate_deployment_config

    log "Pre-deployment checks completed ✓"
}

# Check version difference
check_version_diff() {
    log "Checking version differences..."

    local current_version=""

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        current_version=$(kubectl get deployment backend -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || echo "")
    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        current_version=$(docker ps --filter="name=freeagentics-backend" --format="{{.Image}}" 2>/dev/null || echo "")
    fi

    if [[ -n "$current_version" ]]; then
        PREVIOUS_VERSION=$(echo "$current_version" | cut -d: -f2)
        update_deployment_state "previous_version" "$PREVIOUS_VERSION"

        if [[ "$VERSION" == "$PREVIOUS_VERSION" ]]; then
            warning "New version ($VERSION) is the same as current version ($PREVIOUS_VERSION)"
            if [[ "$DRY_RUN" != "true" ]]; then
                read -p "Continue anyway? (y/N): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    exit 0
                fi
            fi
        fi
    fi

    log "Version check completed: $PREVIOUS_VERSION -> $VERSION"
}

# Check system resources
check_system_resources() {
    log "Checking system resources..."

    # Check memory
    local available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [[ $available_memory -lt 1024 ]]; then
        warning "Low available memory: ${available_memory}MB"
    fi

    # Check disk space
    local available_disk=$(df -BG /var/lib/docker 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || echo "0")
    if [[ $available_disk -lt 5 ]]; then
        warning "Low disk space: ${available_disk}GB"
    fi

    log "System resources check completed ✓"
}

# Validate deployment configuration
validate_deployment_config() {
    log "Validating deployment configuration..."

    # Check required environment variables
    local required_vars=("VERSION" "DOCKER_REGISTRY")

    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done

    # Validate deployment strategy
    case "$DEPLOYMENT_STRATEGY" in
        "blue-green"|"canary"|"rolling"|"recreate")
            log "Using deployment strategy: $DEPLOYMENT_STRATEGY"
            ;;
        *)
            error "Invalid deployment strategy: $DEPLOYMENT_STRATEGY"
            exit 1
            ;;
    esac

    log "Deployment configuration validated ✓"
}

# Create deployment backup
create_deployment_backup() {
    if [[ "$ENABLE_BACKUP" != "true" ]]; then
        return
    fi

    log "Creating deployment backup..."
    update_deployment_state "current_phase" "backup"

    local backup_dir="/var/backups/freeagentics/deployment-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Backup Kubernetes resources
        kubectl get deployment backend -n "$NAMESPACE" -o yaml > "$backup_dir/backend-deployment.yaml"
        kubectl get service backend -n "$NAMESPACE" -o yaml > "$backup_dir/backend-service.yaml"
        kubectl get ingress -n "$NAMESPACE" -o yaml > "$backup_dir/ingress.yaml"
        kubectl get configmap -n "$NAMESPACE" -o yaml > "$backup_dir/configmaps.yaml"

        # Backup Istio resources if present
        if kubectl get virtualservice -n "$NAMESPACE" &>/dev/null; then
            kubectl get virtualservice -n "$NAMESPACE" -o yaml > "$backup_dir/virtualservices.yaml"
            kubectl get destinationrule -n "$NAMESPACE" -o yaml > "$backup_dir/destinationrules.yaml"
        fi
    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        # Backup Docker Compose files
        cp "$PROJECT_ROOT/docker-compose.production.yml" "$backup_dir/"
        cp "$PROJECT_ROOT/.env.production" "$backup_dir/"

        # Export current container states
        docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" > "$backup_dir/container_states.txt"
        docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}" > "$backup_dir/images.txt"
    fi

    # Create rollback data
    cat > "$ROLLBACK_DATA_FILE" <<EOF
{
  "backup_dir": "$backup_dir",
  "previous_version": "$PREVIOUS_VERSION",
  "deployment_mode": "$DEPLOYMENT_MODE",
  "namespace": "$NAMESPACE",
  "created_at": "$(date -Iseconds)"
}
EOF

    update_deployment_state "rollback_available" "true"
    log "Backup created at: $backup_dir ✓"
}

# Build and push new images
build_and_push_images() {
    log "Building and pushing new images..."
    update_deployment_state "current_phase" "build"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN: Skipping image build and push"
        return
    fi

    # Build backend image
    log "Building backend image..."
    docker build -t "$DOCKER_REGISTRY/freeagentics:$VERSION" -f Dockerfile.production "$PROJECT_ROOT"

    # Build frontend image
    log "Building frontend image..."
    docker build -t "$DOCKER_REGISTRY/freeagentics-web:$VERSION" -f web/Dockerfile.production "$PROJECT_ROOT/web"

    # Push images
    if [[ -n "${DOCKER_USERNAME:-}" && -n "${DOCKER_PASSWORD:-}" ]]; then
        echo "$DOCKER_PASSWORD" | docker login "$DOCKER_REGISTRY" -u "$DOCKER_USERNAME" --password-stdin
    fi

    log "Pushing images..."
    docker push "$DOCKER_REGISTRY/freeagentics:$VERSION"
    docker push "$DOCKER_REGISTRY/freeagentics-web:$VERSION"

    log "Images built and pushed ✓"
}

# Deploy using selected strategy
deploy_application() {
    log "Deploying application using $DEPLOYMENT_STRATEGY strategy..."
    update_deployment_state "current_phase" "deploy"

    case "$DEPLOYMENT_STRATEGY" in
        "blue-green")
            deploy_blue_green
            ;;
        "canary")
            deploy_canary
            ;;
        "rolling")
            deploy_rolling
            ;;
        "recreate")
            deploy_recreate
            ;;
    esac
}

# Blue-Green deployment
deploy_blue_green() {
    log "Executing blue-green deployment..."

    # Determine current and new environments
    local current_env="blue"
    local new_env="green"

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        if kubectl get deployment backend-green -n "$NAMESPACE" &>/dev/null; then
            current_env="green"
            new_env="blue"
        fi
    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        if docker ps | grep -q "freeagentics-backend-green"; then
            current_env="green"
            new_env="blue"
        fi
    fi

    log "Current environment: $current_env, new environment: $new_env"

    # Deploy to new environment
    deploy_to_environment "$new_env"

    # Wait for new environment to be ready
    wait_for_environment_ready "$new_env"

    # Run health checks
    if ! run_health_checks "$new_env"; then
        error "Health checks failed for $new_env environment"
        cleanup_failed_environment "$new_env"
        trigger_rollback
        return 1
    fi

    # Run smoke tests
    if [[ "$ENABLE_SMOKE_TESTS" == "true" ]]; then
        if ! run_smoke_tests "$new_env"; then
            error "Smoke tests failed for $new_env environment"
            cleanup_failed_environment "$new_env"
            trigger_rollback
            return 1
        fi
    fi

    # Switch traffic
    switch_traffic "$new_env"

    # Verify traffic switch
    if ! verify_traffic_switch "$new_env"; then
        error "Traffic switch verification failed"
        switch_traffic "$current_env"
        trigger_rollback
        return 1
    fi

    # Cleanup old environment
    cleanup_old_environment "$current_env"

    log "Blue-green deployment completed ✓"
}

# Canary deployment
deploy_canary() {
    log "Executing canary deployment..."

    # Deploy canary version
    deploy_canary_version

    # Gradually increase traffic
    local percentages=(5 10 25 50 75 100)

    for percentage in "${percentages[@]}"; do
        log "Increasing canary traffic to $percentage%..."

        update_canary_traffic "$percentage"

        # Wait for traffic to stabilize
        sleep "$TRAFFIC_SPLIT_DURATION"

        # Monitor metrics
        if ! monitor_canary_metrics "$percentage"; then
            error "Canary metrics check failed at $percentage%"
            rollback_canary
            trigger_rollback
            return 1
        fi

        # Run health checks
        if ! run_health_checks "canary"; then
            error "Health checks failed for canary at $percentage%"
            rollback_canary
            trigger_rollback
            return 1
        fi
    done

    # Promote canary to main
    promote_canary

    log "Canary deployment completed ✓"
}

# Rolling deployment
deploy_rolling() {
    log "Executing rolling deployment..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Update Kubernetes deployment
        kubectl set image deployment/backend backend="$DOCKER_REGISTRY/freeagentics:$VERSION" -n "$NAMESPACE"
        kubectl set image deployment/frontend frontend="$DOCKER_REGISTRY/freeagentics-web:$VERSION" -n "$NAMESPACE"

        # Wait for rollout to complete
        kubectl rollout status deployment/backend -n "$NAMESPACE" --timeout="${ROLLBACK_TIMEOUT}s"
        kubectl rollout status deployment/frontend -n "$NAMESPACE" --timeout="${ROLLBACK_TIMEOUT}s"

    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        # Update Docker Compose services
        export VERSION="$VERSION"
        docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" up -d --no-deps backend frontend

        # Wait for services to be ready
        sleep 30
    fi

    log "Rolling deployment completed ✓"
}

# Recreate deployment
deploy_recreate() {
    log "Executing recreate deployment..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Delete and recreate deployments
        kubectl delete deployment backend frontend -n "$NAMESPACE" --wait=true
        kubectl apply -f "$PROJECT_ROOT/k8s/backend-deployment.yaml"
        kubectl apply -f "$PROJECT_ROOT/k8s/frontend-deployment.yaml"

        # Wait for deployments to be ready
        kubectl wait --for=condition=available deployment/backend -n "$NAMESPACE" --timeout="${ROLLBACK_TIMEOUT}s"
        kubectl wait --for=condition=available deployment/frontend -n "$NAMESPACE" --timeout="${ROLLBACK_TIMEOUT}s"

    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        # Stop and recreate containers
        docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" down backend frontend
        export VERSION="$VERSION"
        docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" up -d backend frontend

        # Wait for services to be ready
        sleep 60
    fi

    log "Recreate deployment completed ✓"
}

# Deploy to specific environment (blue-green)
deploy_to_environment() {
    local env="$1"
    log "Deploying to $env environment..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN: Skipping deployment to $env environment"
        return
    fi

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Create environment-specific deployment
        sed "s/name: backend/name: backend-$env/g" "$PROJECT_ROOT/k8s/backend-deployment.yaml" | \
        sed "s/app: backend/app: backend-$env/g" | \
        sed "s|image: .*|image: $DOCKER_REGISTRY/freeagentics:$VERSION|g" | \
        kubectl apply -f -

        sed "s/name: frontend/name: frontend-$env/g" "$PROJECT_ROOT/k8s/frontend-deployment.yaml" | \
        sed "s/app: frontend/app: frontend-$env/g" | \
        sed "s|image: .*|image: $DOCKER_REGISTRY/freeagentics-web:$VERSION|g" | \
        kubectl apply -f -

    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        # Create environment-specific containers
        docker run -d \
            --name "freeagentics-backend-$env" \
            --network freeagentics-network \
            "$DOCKER_REGISTRY/freeagentics:$VERSION"

        docker run -d \
            --name "freeagentics-frontend-$env" \
            --network freeagentics-network \
            "$DOCKER_REGISTRY/freeagentics-web:$VERSION"
    fi

    log "Deployment to $env environment completed ✓"
}

# Wait for environment to be ready
wait_for_environment_ready() {
    local env="$1"
    log "Waiting for $env environment to be ready..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        kubectl wait --for=condition=available deployment/backend-$env -n "$NAMESPACE" --timeout="${HEALTH_CHECK_TIMEOUT}s"
        kubectl wait --for=condition=available deployment/frontend-$env -n "$NAMESPACE" --timeout="${HEALTH_CHECK_TIMEOUT}s"
    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        local timeout_count=0
        while [[ $timeout_count -lt $HEALTH_CHECK_TIMEOUT ]]; do
            if docker ps | grep -q "freeagentics-backend-$env" && docker ps | grep -q "freeagentics-frontend-$env"; then
                break
            fi
            sleep 5
            timeout_count=$((timeout_count + 5))
        done

        if [[ $timeout_count -ge $HEALTH_CHECK_TIMEOUT ]]; then
            error "Timeout waiting for $env environment to be ready"
            return 1
        fi
    fi

    log "$env environment is ready ✓"
}

# Run health checks
run_health_checks() {
    local env="${1:-}"
    log "Running health checks for $env environment..."

    local backend_url="http://backend:8000"
    local frontend_url="http://frontend:3000"

    if [[ -n "$env" && "$env" != "main" ]]; then
        backend_url="http://backend-$env:8000"
        frontend_url="http://frontend-$env:3000"
    fi

    # Backend health check
    local backend_healthy=false
    for i in {1..10}; do
        if curl -sf "$backend_url/health" --max-time 10 >/dev/null; then
            backend_healthy=true
            break
        fi
        sleep 5
    done

    if [[ "$backend_healthy" != "true" ]]; then
        error "Backend health check failed"
        return 1
    fi

    # Frontend health check
    local frontend_healthy=false
    for i in {1..10}; do
        if curl -sf "$frontend_url/" --max-time 10 >/dev/null; then
            frontend_healthy=true
            break
        fi
        sleep 5
    done

    if [[ "$frontend_healthy" != "true" ]]; then
        error "Frontend health check failed"
        return 1
    fi

    # Database connectivity check
    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        kubectl exec -n "$NAMESPACE" deployment/backend -- python -c "
import asyncio
from database.session import get_db_session
async def test_db():
    async with get_db_session() as session:
        result = await session.execute('SELECT 1')
        return result.scalar()
print('Database OK:', asyncio.run(test_db()))
"
    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        docker exec freeagentics-backend python -c "
import asyncio
from database.session import get_db_session
async def test_db():
    async with get_db_session() as session:
        result = await session.execute('SELECT 1')
        return result.scalar()
print('Database OK:', asyncio.run(test_db()))
"
    fi

    update_deployment_state "health_checks_passed" "true"
    log "Health checks passed ✓"
}

# Run smoke tests
run_smoke_tests() {
    local env="${1:-}"
    log "Running smoke tests for $env environment..."

    local base_url="https://$DOMAIN"
    if [[ -n "$env" && "$env" != "main" ]]; then
        base_url="http://backend-$env:8000"
    fi

    # Test critical endpoints
    local endpoints=("/health" "/api/v1/health" "/api/v1/agents" "/api/v1/auth/me")

    for endpoint in "${endpoints[@]}"; do
        log "Testing endpoint: $endpoint"
        if ! curl -sf "$base_url$endpoint" --max-time 30 >/dev/null; then
            error "Smoke test failed for endpoint: $endpoint"
            return 1
        fi
    done

    # Test database operations
    log "Testing database operations..."
    local test_response=$(curl -sf "$base_url/api/v1/health/database" --max-time 30)
    if [[ "$test_response" != *"healthy"* ]]; then
        error "Database smoke test failed"
        return 1
    fi

    # Test cache operations
    log "Testing cache operations..."
    local cache_response=$(curl -sf "$base_url/api/v1/health/cache" --max-time 30)
    if [[ "$cache_response" != *"healthy"* ]]; then
        error "Cache smoke test failed"
        return 1
    fi

    update_deployment_state "smoke_tests_passed" "true"
    log "Smoke tests passed ✓"
}

# Switch traffic to new environment
switch_traffic() {
    local env="$1"
    log "Switching traffic to $env environment..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN: Skipping traffic switch to $env environment"
        return
    fi

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Update service selectors
        kubectl patch service backend -n "$NAMESPACE" -p "{\"spec\":{\"selector\":{\"app\":\"backend-$env\"}}}"
        kubectl patch service frontend -n "$NAMESPACE" -p "{\"spec\":{\"selector\":{\"app\":\"frontend-$env\"}}}"

        # Update Istio VirtualService if present
        if kubectl get virtualservice freeagentics-vs -n "$NAMESPACE" &>/dev/null; then
            kubectl patch virtualservice freeagentics-vs -n "$NAMESPACE" --type=merge -p "
{
  \"spec\": {
    \"http\": [
      {
        \"route\": [
          {
            \"destination\": {
              \"host\": \"backend-$env\"
            }
          }
        ]
      }
    ]
  }
}
"
        fi
    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        # Update load balancer configuration
        # This would depend on your specific load balancer setup
        log "Updating load balancer configuration for $env environment..."

        # Example: Update nginx upstream
        # This is a placeholder - implement based on your load balancer
        if command -v nginx &>/dev/null; then
            # Update nginx configuration
            sed -i "s/backend-blue/backend-$env/g; s/backend-green/backend-$env/g" /etc/nginx/conf.d/freeagentics.conf
            nginx -s reload
        fi
    fi

    update_deployment_state "traffic_switched" "true"
    log "Traffic switched to $env environment ✓"
}

# Verify traffic switch
verify_traffic_switch() {
    local env="$1"
    log "Verifying traffic switch to $env environment..."

    # Wait for traffic to stabilize
    sleep 30

    # Test that traffic is actually going to the new environment
    local test_endpoint="https://$DOMAIN/api/v1/system/version"

    for i in {1..10}; do
        local response=$(curl -sf "$test_endpoint" --max-time 10)
        if [[ "$response" == *"$VERSION"* ]]; then
            log "Traffic switch verification successful ✓"
            return 0
        fi
        sleep 10
    done

    error "Traffic switch verification failed"
    return 1
}

# Monitor canary metrics
monitor_canary_metrics() {
    local percentage="$1"
    log "Monitoring canary metrics at $percentage%..."

    # Check error rate
    local error_rate=$(curl -sf "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~\"5..\"}[5m])/rate(http_requests_total[5m])*100" | jq -r '.data.result[0].value[1]' || echo "0")

    if (( $(echo "$error_rate > 5" | bc -l) )); then
        error "Canary error rate too high: $error_rate%"
        return 1
    fi

    # Check response time
    local response_time=$(curl -sf "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))*1000" | jq -r '.data.result[0].value[1]' || echo "0")

    if (( $(echo "$response_time > 500" | bc -l) )); then
        error "Canary response time too high: ${response_time}ms"
        return 1
    fi

    log "Canary metrics look good at $percentage% ✓"
}

# Update canary traffic
update_canary_traffic() {
    local percentage="$1"
    local stable_percentage=$((100 - percentage))

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Update Istio VirtualService
        kubectl patch virtualservice backend-canary -n "$NAMESPACE" --type=merge -p "
{
  \"spec\": {
    \"http\": [
      {
        \"route\": [
          {
            \"destination\": {
              \"host\": \"backend\",
              \"subset\": \"stable\"
            },
            \"weight\": $stable_percentage
          },
          {
            \"destination\": {
              \"host\": \"backend\",
              \"subset\": \"canary\"
            },
            \"weight\": $percentage
          }
        ]
      }
    ]
  }
}
"
    fi

    log "Canary traffic updated to $percentage% ✓"
}

# Promote canary to main
promote_canary() {
    log "Promoting canary to main..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Update main deployment with canary image
        kubectl set image deployment/backend backend="$DOCKER_REGISTRY/freeagentics:$VERSION" -n "$NAMESPACE"

        # Remove canary deployment
        kubectl delete deployment backend-canary -n "$NAMESPACE" --ignore-not-found

        # Reset VirtualService
        kubectl patch virtualservice backend-canary -n "$NAMESPACE" --type=merge -p '
{
  "spec": {
    "http": [
      {
        "route": [
          {
            "destination": {
              "host": "backend"
            },
            "weight": 100
          }
        ]
      }
    ]
  }
}
'
    fi

    log "Canary promoted to main ✓"
}

# Rollback canary
rollback_canary() {
    log "Rolling back canary deployment..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Reset traffic to stable version
        kubectl patch virtualservice backend-canary -n "$NAMESPACE" --type=merge -p '
{
  "spec": {
    "http": [
      {
        "route": [
          {
            "destination": {
              "host": "backend",
              "subset": "stable"
            },
            "weight": 100
          }
        ]
      }
    ]
  }
}
'

        # Remove canary deployment
        kubectl delete deployment backend-canary -n "$NAMESPACE" --ignore-not-found
    fi

    log "Canary rollback completed ✓"
}

# Cleanup failed environment
cleanup_failed_environment() {
    local env="$1"
    log "Cleaning up failed $env environment..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        kubectl delete deployment "backend-$env" -n "$NAMESPACE" --ignore-not-found
        kubectl delete deployment "frontend-$env" -n "$NAMESPACE" --ignore-not-found
        kubectl delete service "backend-$env" -n "$NAMESPACE" --ignore-not-found
        kubectl delete service "frontend-$env" -n "$NAMESPACE" --ignore-not-found
    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        docker stop "freeagentics-backend-$env" "freeagentics-frontend-$env" 2>/dev/null || true
        docker rm "freeagentics-backend-$env" "freeagentics-frontend-$env" 2>/dev/null || true
    fi

    log "Failed $env environment cleaned up ✓"
}

# Cleanup old environment
cleanup_old_environment() {
    local env="$1"
    log "Cleaning up old $env environment..."

    # Wait for connections to drain
    sleep 30

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        kubectl delete deployment "backend-$env" -n "$NAMESPACE" --ignore-not-found
        kubectl delete deployment "frontend-$env" -n "$NAMESPACE" --ignore-not-found
    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        docker stop "freeagentics-backend-$env" "freeagentics-frontend-$env" 2>/dev/null || true
        docker rm "freeagentics-backend-$env" "freeagentics-frontend-$env" 2>/dev/null || true
    fi

    log "Old $env environment cleaned up ✓"
}

# Trigger rollback
trigger_rollback() {
    error "Triggering rollback due to deployment failure..."
    update_deployment_state "status" "rolling_back"
    update_deployment_state "current_phase" "rollback"

    if [[ ! -f "$ROLLBACK_DATA_FILE" ]]; then
        error "No rollback data available"
        return 1
    fi

    local rollback_data=$(cat "$ROLLBACK_DATA_FILE")
    local previous_version=$(echo "$rollback_data" | jq -r '.previous_version')

    if [[ "$previous_version" == "null" || -z "$previous_version" ]]; then
        error "No previous version available for rollback"
        return 1
    fi

    log "Rolling back to version: $previous_version"

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Rollback Kubernetes deployments
        kubectl rollout undo deployment/backend -n "$NAMESPACE"
        kubectl rollout undo deployment/frontend -n "$NAMESPACE"

        # Wait for rollback to complete
        kubectl rollout status deployment/backend -n "$NAMESPACE" --timeout="${ROLLBACK_TIMEOUT}s"
        kubectl rollout status deployment/frontend -n "$NAMESPACE" --timeout="${ROLLBACK_TIMEOUT}s"
    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        # Rollback Docker containers
        docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" down backend frontend

        # Start previous version
        export VERSION="$previous_version"
        docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" up -d backend frontend
    fi

    # Verify rollback
    if run_health_checks; then
        log "Rollback completed successfully ✓"
        update_deployment_state "status" "rolled_back"
        send_rollback_notification
    else
        error "Rollback failed - system may be in inconsistent state"
        update_deployment_state "status" "rollback_failed"
    fi
}

# Post-deployment tasks
post_deployment_tasks() {
    log "Running post-deployment tasks..."
    update_deployment_state "current_phase" "post-deployment"

    # Update monitoring dashboards
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        update_monitoring_dashboards
    fi

    # Send success notification
    if [[ "$ENABLE_NOTIFICATIONS" == "true" ]]; then
        send_success_notification
    fi

    # Generate deployment report
    generate_deployment_report

    log "Post-deployment tasks completed ✓"
}

# Update monitoring dashboards
update_monitoring_dashboards() {
    log "Updating monitoring dashboards..."

    # Reload Prometheus configuration
    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        kubectl exec -n "$NAMESPACE" deployment/prometheus -- curl -X POST http://localhost:9090/-/reload
    elif [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        docker exec freeagentics-prometheus curl -X POST http://localhost:9090/-/reload
    fi

    # Update Grafana dashboards
    if [[ -f "$SCRIPT_DIR/../monitoring/deploy-dashboards.sh" ]]; then
        "$SCRIPT_DIR/../monitoring/deploy-dashboards.sh"
    fi

    log "Monitoring dashboards updated ✓"
}

# Send success notification
send_success_notification() {
    log "Sending success notification..."

    local message="✅ Zero-downtime deployment completed successfully!
**Version:** $VERSION
**Strategy:** $DEPLOYMENT_STRATEGY
**Environment:** $DEPLOYMENT_ENV
**Duration:** $(get_deployment_duration)
**Health Checks:** $(get_deployment_state "health_checks_passed")
**Smoke Tests:** $(get_deployment_state "smoke_tests_passed")"

    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$SLACK_WEBHOOK"
    fi

    if [[ -n "${TEAMS_WEBHOOK:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$TEAMS_WEBHOOK"
    fi

    log "Success notification sent ✓"
}

# Send rollback notification
send_rollback_notification() {
    log "Sending rollback notification..."

    local message="🔄 Deployment rollback completed!
**Version:** $VERSION -> $PREVIOUS_VERSION
**Strategy:** $DEPLOYMENT_STRATEGY
**Environment:** $DEPLOYMENT_ENV
**Reason:** Deployment failure"

    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$SLACK_WEBHOOK"
    fi

    if [[ -n "${TEAMS_WEBHOOK:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$TEAMS_WEBHOOK"
    fi

    log "Rollback notification sent ✓"
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

    local report_file="/var/log/freeagentics/deployment-report-$(date +%Y%m%d_%H%M%S).json"

    cat > "$report_file" <<EOF
{
  "deployment_id": "$(get_deployment_state "deployment_id")",
  "version": "$VERSION",
  "previous_version": "$PREVIOUS_VERSION",
  "strategy": "$DEPLOYMENT_STRATEGY",
  "environment": "$DEPLOYMENT_ENV",
  "deployment_mode": "$DEPLOYMENT_MODE",
  "start_time": "$(get_deployment_state "start_time")",
  "end_time": "$(date -Iseconds)",
  "duration": "$(get_deployment_duration)",
  "status": "$(get_deployment_state "status")",
  "health_checks_passed": $(get_deployment_state "health_checks_passed"),
  "smoke_tests_passed": $(get_deployment_state "smoke_tests_passed"),
  "traffic_switched": $(get_deployment_state "traffic_switched"),
  "rollback_available": $(get_deployment_state "rollback_available"),
  "log_file": "$LOG_FILE",
  "report_file": "$report_file"
}
EOF

    log "Deployment report generated: $report_file ✓"
}

# Main deployment function
main() {
    log "Starting zero-downtime deployment"
    log "Version: $VERSION"
    log "Strategy: $DEPLOYMENT_STRATEGY"
    log "Environment: $DEPLOYMENT_ENV"
    log "Mode: $DEPLOYMENT_MODE"

    # Initialize deployment state
    init_deployment_state

    # Set error trap
    trap 'trigger_rollback' ERR

    # Auto-detect deployment mode
    detect_deployment_mode

    # Run deployment steps
    pre_deployment_checks
    create_deployment_backup
    build_and_push_images
    deploy_application

    # Run final health checks
    if ! run_health_checks; then
        error "Final health checks failed"
        trigger_rollback
        exit 1
    fi

    # Run final smoke tests
    if [[ "$ENABLE_SMOKE_TESTS" == "true" ]]; then
        if ! run_smoke_tests; then
            error "Final smoke tests failed"
            trigger_rollback
            exit 1
        fi
    fi

    # Remove error trap
    trap - ERR

    # Post-deployment tasks
    post_deployment_tasks

    update_deployment_state "status" "completed"
    update_deployment_state "current_phase" "completed"

    log "Zero-downtime deployment completed successfully! 🚀"
    log "Version $VERSION is now live"

    # Display summary
    echo ""
    echo "=== Deployment Summary ==="
    echo "Version: $PREVIOUS_VERSION -> $VERSION"
    echo "Strategy: $DEPLOYMENT_STRATEGY"
    echo "Environment: $DEPLOYMENT_ENV"
    echo "Mode: $DEPLOYMENT_MODE"
    echo "Duration: $(get_deployment_duration)"
    echo "Status: $(get_deployment_state "status")"
    echo "Log: $LOG_FILE"
    echo "Report: $(find /var/log/freeagentics -name "deployment-report-*.json" -newest -print -quit 2>/dev/null || echo "Not generated")"
}

# Parse command line arguments
show_help() {
    cat <<EOF
Zero-Downtime Deployment Script for FreeAgentics

Usage: $0 [OPTIONS]

Options:
  --version VERSION         Application version to deploy (required)
  --strategy STRATEGY       Deployment strategy: blue-green|canary|rolling|recreate (default: blue-green)
  --env ENV                 Environment: production|staging|development (default: production)
  --mode MODE               Deployment mode: auto|kubernetes|docker (default: auto)
  --namespace NAMESPACE     Kubernetes namespace (default: freeagentics-prod)
  --domain DOMAIN           Domain name (default: freeagentics.com)
  --registry REGISTRY       Docker registry (default: your-registry.com)
  --timeout SECONDS         Health check timeout (default: 300)
  --no-monitoring           Disable monitoring updates
  --no-notifications        Disable notifications
  --no-backup               Disable backup creation
  --no-smoke-tests          Disable smoke tests
  --dry-run                 Perform dry run without actual deployment
  --help                    Show this help message

Examples:
  $0 --version v1.2.3 --strategy blue-green
  $0 --version v1.2.3 --strategy canary --env staging
  $0 --version v1.2.3 --mode docker --no-monitoring
  $0 --version v1.2.3 --dry-run

Environment Variables:
  DOCKER_USERNAME           Docker registry username
  DOCKER_PASSWORD           Docker registry password
  SLACK_WEBHOOK             Slack webhook URL for notifications
  TEAMS_WEBHOOK             Microsoft Teams webhook URL

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --strategy)
            DEPLOYMENT_STRATEGY="$2"
            shift 2
            ;;
        --env)
            DEPLOYMENT_ENV="$2"
            shift 2
            ;;
        --mode)
            DEPLOYMENT_MODE="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --timeout)
            HEALTH_CHECK_TIMEOUT="$2"
            shift 2
            ;;
        --no-monitoring)
            ENABLE_MONITORING="false"
            shift
            ;;
        --no-notifications)
            ENABLE_NOTIFICATIONS="false"
            shift
            ;;
        --no-backup)
            ENABLE_BACKUP="false"
            shift
            ;;
        --no-smoke-tests)
            ENABLE_SMOKE_TESTS="false"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
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
