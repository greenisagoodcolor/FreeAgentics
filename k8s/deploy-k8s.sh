#!/bin/bash
# Kubernetes Production Deployment Script for FreeAgentics
# Comprehensive deployment with health checks, rollback capability, and monitoring

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-freeagentics-prod}"
KUBECTL_CONTEXT="${KUBECTL_CONTEXT:-}"
DRY_RUN="${DRY_RUN:-false}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-600}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"
BACKUP_BEFORE_DEPLOY="${BACKUP_BEFORE_DEPLOY:-true}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

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
            --data "{\"text\":\"$emoji FreeAgentics K8s Deployment: $message\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check required commands
    local required_commands=("kubectl" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error "Required command '$cmd' not found"
            exit 1
        fi
    done
    
    # Check kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Set kubectl context if specified
    if [[ -n "$KUBECTL_CONTEXT" ]]; then
        kubectl config use-context "$KUBECTL_CONTEXT"
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        warn "Namespace '$NAMESPACE' does not exist, creating..."
        kubectl create namespace "$NAMESPACE"
    fi
    
    log "Prerequisites check completed successfully"
}

# Function to validate manifests
validate_manifests() {
    log "Validating Kubernetes manifests..."
    
    local manifest_files=(
        "namespace.yaml"
        "secrets.yaml"
        "persistent-volumes.yaml"
        "postgres-deployment.yaml"
        "redis-deployment.yaml"
        "backend-deployment.yaml"
        "frontend-deployment.yaml"
        "ingress.yaml"
        "monitoring-stack.yaml"
        "database-migration-job.yaml"
    )
    
    for file in "${manifest_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            error "Manifest file '$file' not found"
            exit 1
        fi
        
        if ! kubectl apply --dry-run=client -f "$file" &> /dev/null; then
            error "Manifest file '$file' is invalid"
            exit 1
        fi
    done
    
    log "All manifest files validated successfully"
}

# Function to backup current state
backup_current_state() {
    if [[ "$BACKUP_BEFORE_DEPLOY" != "true" ]]; then
        info "Backup skipped (BACKUP_BEFORE_DEPLOY=false)"
        return 0
    fi
    
    log "Creating backup of current deployment state..."
    
    local backup_dir="backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup current deployments
    kubectl get deployments -n "$NAMESPACE" -o yaml > "$backup_dir/deployments.yaml" 2>/dev/null || true
    kubectl get services -n "$NAMESPACE" -o yaml > "$backup_dir/services.yaml" 2>/dev/null || true
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "$backup_dir/configmaps.yaml" 2>/dev/null || true
    kubectl get secrets -n "$NAMESPACE" -o yaml > "$backup_dir/secrets.yaml" 2>/dev/null || true
    kubectl get ingress -n "$NAMESPACE" -o yaml > "$backup_dir/ingress.yaml" 2>/dev/null || true
    
    # Backup database if possible
    if kubectl get pod -n "$NAMESPACE" -l component=postgres --field-selector=status.phase=Running &> /dev/null; then
        log "Creating database backup..."
        kubectl exec -n "$NAMESPACE" -l component=postgres -- pg_dump -U freeagentics -d freeagentics > "$backup_dir/database.sql" 2>/dev/null || warn "Database backup failed"
    fi
    
    log "Backup completed in directory: $backup_dir"
    echo "$backup_dir" > .last_backup
}

# Function to deploy infrastructure components
deploy_infrastructure() {
    log "Deploying infrastructure components..."
    
    # Deploy namespace
    kubectl apply -f namespace.yaml
    
    # Deploy secrets and configmaps
    kubectl apply -f secrets.yaml
    
    # Deploy persistent volumes
    kubectl apply -f persistent-volumes.yaml
    
    # Wait for PVCs to be bound
    log "Waiting for persistent volumes to be bound..."
    kubectl wait --for=condition=Bound pvc --all -n "$NAMESPACE" --timeout=300s
    
    log "Infrastructure components deployed successfully"
}

# Function to deploy database
deploy_database() {
    log "Deploying database..."
    
    kubectl apply -f postgres-deployment.yaml
    
    # Wait for database to be ready
    log "Waiting for database to be ready..."
    kubectl wait --for=condition=Ready pod -l component=postgres -n "$NAMESPACE" --timeout=300s
    
    # Run database migrations
    log "Running database migrations..."
    kubectl apply -f database-migration-job.yaml
    
    # Wait for migration job to complete
    kubectl wait --for=condition=Complete job/database-migration -n "$NAMESPACE" --timeout=300s
    
    log "Database deployed and migrated successfully"
}

# Function to deploy cache
deploy_cache() {
    log "Deploying cache..."
    
    kubectl apply -f redis-deployment.yaml
    
    # Wait for cache to be ready
    log "Waiting for cache to be ready..."
    kubectl wait --for=condition=Ready pod -l component=redis -n "$NAMESPACE" --timeout=300s
    
    log "Cache deployed successfully"
}

# Function to deploy application
deploy_application() {
    log "Deploying application..."
    
    # Deploy backend
    kubectl apply -f backend-deployment.yaml
    
    # Wait for backend to be ready
    log "Waiting for backend to be ready..."
    kubectl wait --for=condition=Ready pod -l component=backend -n "$NAMESPACE" --timeout=300s
    
    # Deploy frontend
    kubectl apply -f frontend-deployment.yaml
    
    # Wait for frontend to be ready
    log "Waiting for frontend to be ready..."
    kubectl wait --for=condition=Ready pod -l component=frontend -n "$NAMESPACE" --timeout=300s
    
    log "Application deployed successfully"
}

# Function to deploy ingress
deploy_ingress() {
    log "Deploying ingress..."
    
    kubectl apply -f ingress.yaml
    
    # Wait for ingress to be ready
    log "Waiting for ingress to be ready..."
    sleep 30  # Give ingress controller time to process
    
    log "Ingress deployed successfully"
}

# Function to deploy monitoring
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    kubectl apply -f monitoring-stack.yaml
    
    # Wait for monitoring components to be ready
    log "Waiting for monitoring components to be ready..."
    kubectl wait --for=condition=Ready pod -l component=prometheus -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=Ready pod -l component=grafana -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=Ready pod -l component=alertmanager -n "$NAMESPACE" --timeout=300s
    
    log "Monitoring stack deployed successfully"
}

# Function to run health checks
run_health_checks() {
    log "Running health checks..."
    
    # Check pod status
    local failed_pods
    failed_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running --no-headers | wc -l)
    
    if [[ $failed_pods -gt 0 ]]; then
        error "Found $failed_pods failed pods:"
        kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running
        return 1
    fi
    
    # Check service endpoints
    local services=("backend" "frontend" "postgres" "redis")
    for service in "${services[@]}"; do
        local endpoints
        endpoints=$(kubectl get endpoints "$service" -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -w)
        
        if [[ $endpoints -eq 0 ]]; then
            error "Service '$service' has no endpoints"
            return 1
        fi
        
        log "Service '$service' has $endpoints endpoint(s)"
    done
    
    # Test application endpoints
    local ingress_ip
    ingress_ip=$(kubectl get ingress freeagentics-ingress -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [[ -n "$ingress_ip" ]]; then
        log "Testing application endpoints..."
        
        # Test backend health
        if curl -f -s -k "https://$ingress_ip/health" &> /dev/null; then
            log "Backend health check passed"
        else
            warn "Backend health check failed"
        fi
        
        # Test frontend
        if curl -f -s -k "https://$ingress_ip/" &> /dev/null; then
            log "Frontend health check passed"
        else
            warn "Frontend health check failed"
        fi
    else
        warn "Ingress IP not yet available, skipping endpoint tests"
    fi
    
    log "Health checks completed"
}

# Function to rollback deployment
rollback_deployment() {
    log "Starting deployment rollback..."
    
    if [[ ! -f .last_backup ]]; then
        error "No backup found for rollback"
        return 1
    fi
    
    local backup_dir
    backup_dir=$(cat .last_backup)
    
    if [[ ! -d "$backup_dir" ]]; then
        error "Backup directory '$backup_dir' not found"
        return 1
    fi
    
    warn "Rolling back to backup: $backup_dir"
    
    # Rollback deployments
    if [[ -f "$backup_dir/deployments.yaml" ]]; then
        kubectl apply -f "$backup_dir/deployments.yaml" || true
    fi
    
    # Rollback services
    if [[ -f "$backup_dir/services.yaml" ]]; then
        kubectl apply -f "$backup_dir/services.yaml" || true
    fi
    
    # Rollback configmaps
    if [[ -f "$backup_dir/configmaps.yaml" ]]; then
        kubectl apply -f "$backup_dir/configmaps.yaml" || true
    fi
    
    # Rollback ingress
    if [[ -f "$backup_dir/ingress.yaml" ]]; then
        kubectl apply -f "$backup_dir/ingress.yaml" || true
    fi
    
    # Wait for rollback to complete
    log "Waiting for rollback to complete..."
    kubectl rollout status deployment/backend -n "$NAMESPACE" --timeout=300s || true
    kubectl rollout status deployment/frontend -n "$NAMESPACE" --timeout=300s || true
    
    # Restore database if needed
    if [[ -f "$backup_dir/database.sql" ]]; then
        warn "Database restore not implemented in rollback - manual intervention required"
    fi
    
    send_notification "Rollback completed to backup $backup_dir" "SUCCESS"
    log "Rollback completed"
}

# Function to show deployment status
show_deployment_status() {
    log "=== DEPLOYMENT STATUS ==="
    
    # Show pods
    log "Pods:"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    # Show services
    log "Services:"
    kubectl get services -n "$NAMESPACE"
    
    # Show ingress
    log "Ingress:"
    kubectl get ingress -n "$NAMESPACE"
    
    # Show resource usage
    log "Resource Usage:"
    kubectl top pods -n "$NAMESPACE" --no-headers | head -10 || true
    
    # Show events
    log "Recent Events:"
    kubectl get events -n "$NAMESPACE" --sort-by=.metadata.creationTimestamp | tail -10
}

# Function to cleanup old resources
cleanup_old_resources() {
    log "Cleaning up old resources..."
    
    # Delete completed jobs older than 1 day
    kubectl get jobs -n "$NAMESPACE" --field-selector=status.conditions[0].type=Complete \
        -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.completionTime}{"\n"}{end}' | \
        while read job_name completion_time; do
            if [[ -n "$completion_time" ]]; then
                completion_timestamp=$(date -d "$completion_time" +%s)
                current_timestamp=$(date +%s)
                age_seconds=$((current_timestamp - completion_timestamp))
                age_days=$((age_seconds / 86400))
                
                if [[ $age_days -gt 1 ]]; then
                    log "Deleting old completed job: $job_name (age: $age_days days)"
                    kubectl delete job "$job_name" -n "$NAMESPACE"
                fi
            fi
        done
    
    # Delete old replica sets
    kubectl get replicasets -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.spec.replicas}{" "}{.status.replicas}{"\n"}{end}' | \
        while read rs_name spec_replicas status_replicas; do
            if [[ "$spec_replicas" == "0" && "$status_replicas" == "0" ]]; then
                log "Deleting old replica set: $rs_name"
                kubectl delete replicaset "$rs_name" -n "$NAMESPACE"
            fi
        done
    
    log "Cleanup completed"
}

# Main deployment function
main() {
    local start_time
    start_time=$(date +%s)
    
    send_notification "Starting Kubernetes deployment" "START"
    
    # Trap for cleanup on exit
    trap 'echo "Deployment interrupted"; exit 1' INT TERM
    
    # Change to k8s directory
    cd "$(dirname "$0")"
    
    # Pre-deployment steps
    check_prerequisites
    validate_manifests
    backup_current_state || {
        error "Backup failed - aborting deployment"
        exit 1
    }
    
    # Deployment steps
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN MODE - No actual deployment will occur"
        kubectl apply --dry-run=client -f .
        exit 0
    fi
    
    deploy_infrastructure || {
        error "Infrastructure deployment failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback_deployment
        fi
        exit 1
    }
    
    deploy_database || {
        error "Database deployment failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback_deployment
        fi
        exit 1
    }
    
    deploy_cache || {
        error "Cache deployment failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback_deployment
        fi
        exit 1
    }
    
    deploy_application || {
        error "Application deployment failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback_deployment
        fi
        exit 1
    }
    
    deploy_ingress || {
        error "Ingress deployment failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback_deployment
        fi
        exit 1
    }
    
    deploy_monitoring || {
        error "Monitoring deployment failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback_deployment
        fi
        exit 1
    }
    
    # Post-deployment verification
    run_health_checks || {
        error "Health checks failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback_deployment
        fi
        exit 1
    }
    
    # Cleanup and status
    cleanup_old_resources
    show_deployment_status
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    send_notification "Kubernetes deployment completed successfully in ${duration}s" "SUCCESS"
    log "🎉 Deployment completed successfully!"
}

# Script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -n, --namespace NAMESPACE    Set target namespace (default: freeagentics-prod)"
    echo "  -c, --context CONTEXT       Set kubectl context"
    echo "  -d, --dry-run               Perform dry run only"
    echo "  -t, --timeout SECONDS      Set wait timeout (default: 600)"
    echo "  --no-backup                 Skip backup before deployment"
    echo "  --no-rollback               Disable automatic rollback on failure"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  SLACK_WEBHOOK              Slack webhook URL for notifications"
    echo "  KUBECTL_CONTEXT            Kubernetes context to use"
    echo ""
    echo "Examples:"
    echo "  $0                         # Deploy to production"
    echo "  $0 -d                      # Dry run"
    echo "  $0 -n staging              # Deploy to staging namespace"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -c|--context)
            KUBECTL_CONTEXT="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -t|--timeout)
            WAIT_TIMEOUT="$2"
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