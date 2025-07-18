#!/bin/bash
# Enhanced Kubernetes Deployment Script for FreeAgentics
# Supports blue-green deployment, canary releases, and zero-downtime updates

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
NAMESPACE="${NAMESPACE:-freeagentics-prod}"
DEPLOYMENT_STRATEGY="${DEPLOYMENT_STRATEGY:-rolling}"
VERSION="${VERSION:-latest}"
CLUSTER_NAME="${CLUSTER_NAME:-freeagentics-prod}"
REGION="${REGION:-us-west-2}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-~/.kube/config}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-your-registry.com}"
ISTIO_ENABLED="${ISTIO_ENABLED:-true}"
MONITORING_ENABLED="${MONITORING_ENABLED:-true}"
BACKUP_ENABLED="${BACKUP_ENABLED:-true}"

# Logging
LOG_FILE="/var/log/freeagentics/k8s-deployment-$(date +%Y%m%d_%H%M%S).log"
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

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is required but not installed"
        exit 1
    fi

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check namespace
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        warning "Namespace $NAMESPACE does not exist. Creating..."
        kubectl create namespace "$NAMESPACE"
        kubectl label namespace "$NAMESPACE" istio-injection=enabled
    fi

    # Check Istio installation
    if [[ "$ISTIO_ENABLED" == "true" ]]; then
        if ! kubectl get namespace istio-system &> /dev/null; then
            warning "Istio not installed. Installing..."
            install_istio
        fi
    fi

    # Check required secrets
    check_secrets

    log "Pre-deployment checks completed ✓"
}

# Install Istio
install_istio() {
    log "Installing Istio..."

    # Download Istio
    if ! command -v istioctl &> /dev/null; then
        curl -L https://istio.io/downloadIstio | sh -
        export PATH="$PATH:$PWD/istio-*/bin"
    fi

    # Install Istio
    istioctl install --set values.defaultRevision=default -y

    # Enable sidecar injection
    kubectl label namespace "$NAMESPACE" istio-injection=enabled --overwrite

    log "Istio installation completed ✓"
}

# Check required secrets
check_secrets() {
    log "Checking required secrets..."

    required_secrets=(
        "freeagentics-secrets"
        "grafana-secrets"
        "postgres-secrets"
        "redis-secrets"
    )

    for secret in "${required_secrets[@]}"; do
        if ! kubectl get secret "$secret" -n "$NAMESPACE" &> /dev/null; then
            warning "Secret $secret not found. Creating from template..."
            create_secret_template "$secret"
        fi
    done
}

# Create secret templates
create_secret_template() {
    local secret_name="$1"

    case "$secret_name" in
        "freeagentics-secrets")
            kubectl create secret generic "$secret_name" -n "$NAMESPACE" \
                --from-literal=database-url="postgresql://freeagentics:changeme@postgres:5432/freeagentics" \
                --from-literal=redis-url="redis://:changeme@redis:6379" \
                --from-literal=secret-key="$(openssl rand -base64 32)" \
                --from-literal=jwt-secret="$(openssl rand -base64 32)"
            ;;
        "grafana-secrets")
            kubectl create secret generic "$secret_name" -n "$NAMESPACE" \
                --from-literal=admin-password="$(openssl rand -base64 16)"
            ;;
        "postgres-secrets")
            kubectl create secret generic "$secret_name" -n "$NAMESPACE" \
                --from-literal=postgres-password="$(openssl rand -base64 16)"
            ;;
        "redis-secrets")
            kubectl create secret generic "$secret_name" -n "$NAMESPACE" \
                --from-literal=redis-password="$(openssl rand -base64 16)"
            ;;
    esac
}

# Backup current deployment
backup_deployment() {
    log "Creating deployment backup..."

    BACKUP_DIR="/var/backups/freeagentics-k8s/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"

    # Export current deployments
    kubectl get deployment -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/deployments.yaml"
    kubectl get service -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/services.yaml"
    kubectl get configmap -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/configmaps.yaml"
    kubectl get secret -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/secrets.yaml"

    # Export Istio configs
    if [[ "$ISTIO_ENABLED" == "true" ]]; then
        kubectl get virtualservice -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/virtualservices.yaml"
        kubectl get destinationrule -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/destinationrules.yaml"
        kubectl get gateway -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/gateways.yaml"
    fi

    log "Backup created at: $BACKUP_DIR"
}

# Deploy base infrastructure
deploy_infrastructure() {
    log "Deploying infrastructure components..."

    # Apply namespace
    kubectl apply -f namespace.yaml

    # Apply persistent volumes
    kubectl apply -f persistent-volumes.yaml

    # Deploy database
    log "Deploying PostgreSQL..."
    kubectl apply -f postgres-deployment.yaml

    # Deploy Redis
    log "Deploying Redis..."
    kubectl apply -f redis-deployment.yaml

    # Wait for database to be ready
    kubectl wait --for=condition=ready pod -l app=postgres -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=redis -n "$NAMESPACE" --timeout=300s

    log "Infrastructure deployment completed ✓"
}

# Deploy application
deploy_application() {
    log "Deploying application components..."

    case "$DEPLOYMENT_STRATEGY" in
        "rolling")
            deploy_rolling_update
            ;;
        "blue-green")
            deploy_blue_green
            ;;
        "canary")
            deploy_canary
            ;;
        *)
            error "Unknown deployment strategy: $DEPLOYMENT_STRATEGY"
            exit 1
            ;;
    esac
}

# Rolling update deployment
deploy_rolling_update() {
    log "Performing rolling update deployment..."

    # Update image tags
    kubectl set image deployment/backend backend="$DOCKER_REGISTRY/freeagentics:$VERSION" -n "$NAMESPACE"
    kubectl set image deployment/frontend frontend="$DOCKER_REGISTRY/freeagentics-web:$VERSION" -n "$NAMESPACE"

    # Wait for rollout
    kubectl rollout status deployment/backend -n "$NAMESPACE" --timeout=600s
    kubectl rollout status deployment/frontend -n "$NAMESPACE" --timeout=600s

    log "Rolling update completed ✓"
}

# Blue-green deployment
deploy_blue_green() {
    log "Performing blue-green deployment..."

    # Determine current environment
    if kubectl get service backend-blue -n "$NAMESPACE" &> /dev/null; then
        CURRENT_ENV="blue"
        NEW_ENV="green"
    else
        CURRENT_ENV="green"
        NEW_ENV="blue"
    fi

    log "Current environment: $CURRENT_ENV, deploying to: $NEW_ENV"

    # Deploy to new environment
    sed "s/backend/backend-$NEW_ENV/g" backend-deployment.yaml | kubectl apply -f -
    sed "s/frontend/frontend-$NEW_ENV/g" frontend-deployment.yaml | kubectl apply -f -

    # Wait for new environment to be ready
    kubectl wait --for=condition=ready pod -l app=backend-$NEW_ENV -n "$NAMESPACE" --timeout=600s
    kubectl wait --for=condition=ready pod -l app=frontend-$NEW_ENV -n "$NAMESPACE" --timeout=600s

    # Run health checks
    if ! run_health_checks "$NEW_ENV"; then
        error "Health checks failed for $NEW_ENV environment"
        cleanup_failed_environment "$NEW_ENV"
        exit 1
    fi

    # Switch traffic
    switch_traffic "$NEW_ENV"

    # Cleanup old environment
    cleanup_old_environment "$CURRENT_ENV"

    log "Blue-green deployment completed ✓"
}

# Canary deployment
deploy_canary() {
    log "Performing canary deployment..."

    # Deploy canary version
    kubectl apply -f backend-canary.yaml

    # Gradually increase traffic
    for weight in 10 25 50 75 100; do
        log "Increasing canary traffic to $weight%..."
        update_canary_weight "$weight"
        sleep 60

        if ! run_health_checks "canary"; then
            error "Health checks failed for canary at $weight%"
            rollback_canary
            exit 1
        fi
    done

    # Promote canary to main
    promote_canary

    log "Canary deployment completed ✓"
}

# Deploy monitoring stack
deploy_monitoring() {
    if [[ "$MONITORING_ENABLED" != "true" ]]; then
        return
    fi

    log "Deploying monitoring stack..."

    # Deploy Prometheus
    kubectl apply -f monitoring-stack.yaml

    # Wait for monitoring to be ready
    kubectl wait --for=condition=ready pod -l app=prometheus -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=grafana -n "$NAMESPACE" --timeout=300s

    log "Monitoring stack deployed ✓"
}

# Deploy Istio configuration
deploy_istio_config() {
    if [[ "$ISTIO_ENABLED" != "true" ]]; then
        return
    fi

    log "Deploying Istio configuration..."

    # Apply service mesh configs
    kubectl apply -f istio-service-mesh.yaml

    # Apply autoscaling configs
    kubectl apply -f autoscaling-enhanced.yaml

    log "Istio configuration deployed ✓"
}

# Run health checks
run_health_checks() {
    local env="${1:-}"
    log "Running health checks for $env environment..."

    # Define service URLs based on environment
    if [[ -n "$env" && "$env" != "main" ]]; then
        BACKEND_URL="http://backend-$env:8000"
        FRONTEND_URL="http://frontend-$env:3000"
    else
        BACKEND_URL="http://backend:8000"
        FRONTEND_URL="http://frontend:3000"
    fi

    # Backend health check
    kubectl run health-check-backend --image=curlimages/curl:latest --rm -it --restart=Never -n "$NAMESPACE" \
        --command -- curl -f "$BACKEND_URL/health" --max-time 30

    # Frontend health check
    kubectl run health-check-frontend --image=curlimages/curl:latest --rm -it --restart=Never -n "$NAMESPACE" \
        --command -- curl -f "$FRONTEND_URL/" --max-time 30

    # Database connectivity check
    kubectl run health-check-db --image=postgres:15 --rm -it --restart=Never -n "$NAMESPACE" \
        --command -- psql -h postgres -U freeagentics -d freeagentics -c "SELECT 1"

    log "Health checks completed ✓"
}

# Switch traffic (for blue-green)
switch_traffic() {
    local new_env="$1"
    log "Switching traffic to $new_env environment..."

    # Update service selectors
    kubectl patch service backend -n "$NAMESPACE" -p "{\"spec\":{\"selector\":{\"app\":\"backend-$new_env\"}}}"
    kubectl patch service frontend -n "$NAMESPACE" -p "{\"spec\":{\"selector\":{\"app\":\"frontend-$new_env\"}}}"

    # Update Istio virtual service if enabled
    if [[ "$ISTIO_ENABLED" == "true" ]]; then
        kubectl patch virtualservice freeagentics-vs -n "$NAMESPACE" --type=merge -p "{\"spec\":{\"http\":[{\"route\":[{\"destination\":{\"host\":\"backend-$new_env\"}}]}]}}"
    fi

    log "Traffic switched to $new_env ✓"
}

# Update canary weight
update_canary_weight() {
    local weight="$1"
    local stable_weight=$((100 - weight))

    kubectl patch virtualservice backend-canary -n "$NAMESPACE" --type=merge -p "{\"spec\":{\"http\":[{\"route\":[{\"destination\":{\"host\":\"backend\",\"subset\":\"v1\"},\"weight\":$stable_weight},{\"destination\":{\"host\":\"backend\",\"subset\":\"v2\"},\"weight\":$weight}]}]}}"
}

# Promote canary
promote_canary() {
    log "Promoting canary to main..."

    # Update main deployment
    kubectl patch deployment backend -n "$NAMESPACE" -p "{\"spec\":{\"template\":{\"metadata\":{\"labels\":{\"version\":\"v2\"}}}}}"

    # Remove canary deployment
    kubectl delete deployment backend-canary -n "$NAMESPACE"

    # Reset virtual service
    kubectl patch virtualservice backend-canary -n "$NAMESPACE" --type=merge -p '{"spec":{"http":[{"route":[{"destination":{"host":"backend","subset":"v1"},"weight":100}]}]}}'

    log "Canary promoted ✓"
}

# Rollback canary
rollback_canary() {
    log "Rolling back canary deployment..."

    # Reset virtual service
    kubectl patch virtualservice backend-canary -n "$NAMESPACE" --type=merge -p '{"spec":{"http":[{"route":[{"destination":{"host":"backend","subset":"v1"},"weight":100}]}]}}'

    # Remove canary deployment
    kubectl delete deployment backend-canary -n "$NAMESPACE"

    log "Canary rollback completed ✓"
}

# Cleanup functions
cleanup_failed_environment() {
    local env="$1"
    log "Cleaning up failed $env environment..."

    kubectl delete deployment "backend-$env" -n "$NAMESPACE" --ignore-not-found
    kubectl delete deployment "frontend-$env" -n "$NAMESPACE" --ignore-not-found
    kubectl delete service "backend-$env" -n "$NAMESPACE" --ignore-not-found
    kubectl delete service "frontend-$env" -n "$NAMESPACE" --ignore-not-found
}

cleanup_old_environment() {
    local env="$1"
    log "Cleaning up old $env environment..."

    # Give some time for connections to drain
    sleep 30

    kubectl delete deployment "backend-$env" -n "$NAMESPACE" --ignore-not-found
    kubectl delete deployment "frontend-$env" -n "$NAMESPACE" --ignore-not-found
    kubectl delete service "backend-$env" -n "$NAMESPACE" --ignore-not-found
    kubectl delete service "frontend-$env" -n "$NAMESPACE" --ignore-not-found
}

# Post-deployment tasks
post_deployment() {
    log "Running post-deployment tasks..."

    # Create backup
    if [[ "$BACKUP_ENABLED" == "true" ]]; then
        create_post_deployment_backup
    fi

    # Run smoke tests
    run_smoke_tests

    # Update monitoring dashboards
    update_monitoring_dashboards

    # Send notifications
    send_notifications

    log "Post-deployment tasks completed ✓"
}

# Create post-deployment backup
create_post_deployment_backup() {
    log "Creating post-deployment backup..."

    # Create database backup
    kubectl exec -it deployment/postgres -n "$NAMESPACE" -- pg_dump -U freeagentics freeagentics > "/tmp/post-deployment-backup-$(date +%Y%m%d_%H%M%S).sql"

    log "Post-deployment backup created ✓"
}

# Run smoke tests
run_smoke_tests() {
    log "Running smoke tests..."

    # Apply smoke test job
    kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: smoke-tests-$(date +%s)
  namespace: $NAMESPACE
spec:
  template:
    spec:
      containers:
      - name: smoke-tests
        image: curlimages/curl:latest
        command: ["/bin/sh", "-c"]
        args:
        - |
          echo "Running smoke tests..."
          curl -f http://backend:8000/health || exit 1
          curl -f http://frontend:3000/ || exit 1
          echo "All smoke tests passed!"
      restartPolicy: Never
  backoffLimit: 3
EOF

    # Wait for smoke tests to complete
    kubectl wait --for=condition=complete job -l app=smoke-tests -n "$NAMESPACE" --timeout=300s

    log "Smoke tests completed ✓"
}

# Update monitoring dashboards
update_monitoring_dashboards() {
    if [[ "$MONITORING_ENABLED" != "true" ]]; then
        return
    fi

    log "Updating monitoring dashboards..."

    # Reload Grafana dashboards
    kubectl exec -it deployment/grafana -n "$NAMESPACE" -- curl -X POST http://admin:admin@localhost:3000/api/admin/provisioning/dashboards/reload

    log "Monitoring dashboards updated ✓"
}

# Send notifications
send_notifications() {
    log "Sending deployment notifications..."

    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"✅ Kubernetes deployment completed successfully!\n**Version:** $VERSION\n**Strategy:** $DEPLOYMENT_STRATEGY\n**Namespace:** $NAMESPACE\n**Cluster:** $CLUSTER_NAME\"}" \
            "$SLACK_WEBHOOK"
    fi

    if [[ -n "${TEAMS_WEBHOOK:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"✅ FreeAgentics K8s deployment completed successfully! Version: $VERSION, Strategy: $DEPLOYMENT_STRATEGY\"}" \
            "$TEAMS_WEBHOOK"
    fi

    log "Notifications sent ✓"
}

# Rollback function
rollback() {
    error "Deployment failed. Initiating rollback..."

    case "$DEPLOYMENT_STRATEGY" in
        "rolling")
            kubectl rollout undo deployment/backend -n "$NAMESPACE"
            kubectl rollout undo deployment/frontend -n "$NAMESPACE"
            ;;
        "blue-green")
            # Rollback is handled in the blue-green function
            ;;
        "canary")
            rollback_canary
            ;;
    esac

    log "Rollback completed"
}

# Main deployment function
main() {
    log "Starting FreeAgentics Kubernetes deployment"
    log "Namespace: $NAMESPACE"
    log "Version: $VERSION"
    log "Strategy: $DEPLOYMENT_STRATEGY"
    log "Cluster: $CLUSTER_NAME"

    # Set trap for rollback on error
    trap 'rollback' ERR

    # Run deployment steps
    pre_deployment_checks
    backup_deployment
    deploy_infrastructure
    deploy_application

    if [[ "$MONITORING_ENABLED" == "true" ]]; then
        deploy_monitoring
    fi

    if [[ "$ISTIO_ENABLED" == "true" ]]; then
        deploy_istio_config
    fi

    post_deployment

    # Remove error trap
    trap - ERR

    log "Kubernetes deployment completed successfully! 🚀"
    log "Version $VERSION is now live on $CLUSTER_NAME"

    # Deployment summary
    echo -e "\n${GREEN}=== Deployment Summary ===${NC}"
    echo "Namespace: $NAMESPACE"
    echo "Version: $VERSION"
    echo "Strategy: $DEPLOYMENT_STRATEGY"
    echo "Cluster: $CLUSTER_NAME"
    echo "Istio: $ISTIO_ENABLED"
    echo "Monitoring: $MONITORING_ENABLED"
    echo "Log File: $LOG_FILE"
    echo ""
    echo "Access URLs:"
    echo "- Application: https://$(kubectl get ingress freeagentics-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}')"
    echo "- Grafana: https://$(kubectl get ingress freeagentics-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}')/grafana"
    echo "- Prometheus: https://$(kubectl get ingress freeagentics-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}')/prometheus"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --strategy)
            DEPLOYMENT_STRATEGY="$2"
            shift 2
            ;;
        --cluster)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        --no-istio)
            ISTIO_ENABLED="false"
            shift
            ;;
        --no-monitoring)
            MONITORING_ENABLED="false"
            shift
            ;;
        --help)
            echo "FreeAgentics Kubernetes Deployment Script"
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --namespace NAME      Kubernetes namespace (default: freeagentics-prod)"
            echo "  --version VERSION     Application version (default: latest)"
            echo "  --strategy STRATEGY   Deployment strategy: rolling|blue-green|canary (default: rolling)"
            echo "  --cluster NAME        Cluster name (default: freeagentics-prod)"
            echo "  --no-istio           Disable Istio service mesh"
            echo "  --no-monitoring      Disable monitoring stack"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"
