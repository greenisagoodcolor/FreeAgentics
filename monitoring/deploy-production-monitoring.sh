#!/bin/bash
# Production Monitoring Stack Deployment Script
# Deploys Prometheus, Grafana, AlertManager, and dashboards for FreeAgentics

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-freeagentics-prod}"
MONITORING_NAMESPACE="${MONITORING_NAMESPACE:-freeagentics-monitoring}"
KUBECTL_CONTEXT="${KUBECTL_CONTEXT:-}"
DRY_RUN="${DRY_RUN:-false}"
GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
PAGERDUTY_ROUTING_KEY="${PAGERDUTY_ROUTING_KEY:-}"

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

# Function to check prerequisites
check_prerequisites() {
    log "Checking monitoring deployment prerequisites..."
    
    # Check required commands
    local required_commands=("kubectl" "curl" "jq" "yq")
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
    
    # Check if main namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        error "Application namespace '$NAMESPACE' does not exist"
        exit 1
    fi
    
    # Create monitoring namespace if it doesn't exist
    if ! kubectl get namespace "$MONITORING_NAMESPACE" &> /dev/null; then
        log "Creating monitoring namespace: $MONITORING_NAMESPACE"
        kubectl create namespace "$MONITORING_NAMESPACE"
    fi
    
    # Check if Grafana admin password is set
    if [[ -z "$GRAFANA_ADMIN_PASSWORD" ]]; then
        GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 32)
        warn "Grafana admin password not set, generated: $GRAFANA_ADMIN_PASSWORD"
    fi
    
    log "Prerequisites check completed successfully"
}

# Function to create monitoring secrets
create_monitoring_secrets() {
    log "Creating monitoring secrets..."
    
    # Create Grafana admin credentials
    kubectl create secret generic grafana-admin-secret \
        --from-literal=admin-password="$GRAFANA_ADMIN_PASSWORD" \
        -n "$MONITORING_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create AlertManager configuration secret
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        sed "s|YOUR/SLACK/WEBHOOK|${SLACK_WEBHOOK}|g" alertmanager-production.yml > /tmp/alertmanager-config.yml
    else
        cp alertmanager-production.yml /tmp/alertmanager-config.yml
    fi
    
    if [[ -n "$PAGERDUTY_ROUTING_KEY" ]]; then
        sed -i "s|REPLACE_WITH_PAGERDUTY_ROUTING_KEY|${PAGERDUTY_ROUTING_KEY}|g" /tmp/alertmanager-config.yml
    fi
    
    kubectl create secret generic alertmanager-config \
        --from-file=alertmanager.yml=/tmp/alertmanager-config.yml \
        -n "$MONITORING_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create Prometheus configuration secret
    kubectl create secret generic prometheus-config \
        --from-file=prometheus.yml=prometheus-production.yml \
        --from-file=alerts.yml=prometheus-alerts.yml \
        -n "$MONITORING_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Cleanup temporary files
    rm -f /tmp/alertmanager-config.yml
    
    log "Monitoring secrets created successfully"
}

# Function to deploy Prometheus
deploy_prometheus() {
    log "Deploying Prometheus..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: $MONITORING_NAMESPACE
  labels:
    app: prometheus
    component: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
        component: monitoring
    spec:
      serviceAccountName: prometheus
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        args:
        - --config.file=/etc/prometheus/prometheus.yml
        - --storage.tsdb.path=/prometheus
        - --web.console.libraries=/usr/share/prometheus/console_libraries
        - --web.console.templates=/usr/share/prometheus/consoles
        - --storage.tsdb.retention.time=15d
        - --web.enable-lifecycle
        - --web.enable-admin-api
        - --log.level=info
        ports:
        - containerPort: 9090
          name: web
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
        - name: prometheus-storage
          mountPath: /prometheus
        - name: prometheus-rules
          mountPath: /etc/prometheus/rules
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /-/healthy
            port: 9090
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /-/ready
            port: 9090
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: prometheus-config
        secret:
          secretName: prometheus-config
      - name: prometheus-storage
        persistentVolumeClaim:
          claimName: prometheus-storage
      - name: prometheus-rules
        secret:
          secretName: prometheus-config
          items:
          - key: alerts.yml
            path: alerts.yml
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: $MONITORING_NAMESPACE
  labels:
    app: prometheus
    component: monitoring
spec:
  type: ClusterIP
  ports:
  - port: 9090
    targetPort: 9090
    name: web
  selector:
    app: prometheus
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-storage
  namespace: $MONITORING_NAMESPACE
  labels:
    app: prometheus
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: prometheus
  namespace: $MONITORING_NAMESPACE
  labels:
    app: prometheus
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: prometheus
  labels:
    app: prometheus
rules:
- apiGroups: [""]
  resources:
  - nodes
  - nodes/proxy
  - services
  - endpoints
  - pods
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources:
  - deployments
  - replicasets
  - daemonsets
  - statefulsets
  verbs: ["get", "list", "watch"]
- apiGroups: ["extensions"]
  resources:
  - ingresses
  verbs: ["get", "list", "watch"]
- nonResourceURLs: ["/metrics"]
  verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: prometheus
  labels:
    app: prometheus
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: prometheus
subjects:
- kind: ServiceAccount
  name: prometheus
  namespace: $MONITORING_NAMESPACE
EOF
    
    # Wait for Prometheus to be ready
    kubectl wait --for=condition=Ready pod -l app=prometheus -n "$MONITORING_NAMESPACE" --timeout=300s
    
    log "Prometheus deployed successfully"
}

# Function to deploy Grafana
deploy_grafana() {
    log "Deploying Grafana..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: $MONITORING_NAMESPACE
  labels:
    app: grafana
    component: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
        component: monitoring
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: grafana-admin-secret
              key: admin-password
        - name: GF_SECURITY_ADMIN_USER
          value: admin
        - name: GF_INSTALL_PLUGINS
          value: grafana-piechart-panel,grafana-worldmap-panel,grafana-clock-panel
        - name: GF_PATHS_PROVISIONING
          value: /etc/grafana/provisioning
        - name: GF_ALERTING_ENABLED
          value: "true"
        - name: GF_UNIFIED_ALERTING_ENABLED
          value: "true"
        - name: GF_USERS_ALLOW_SIGN_UP
          value: "false"
        - name: GF_ANALYTICS_REPORTING_ENABLED
          value: "false"
        - name: GF_ANALYTICS_CHECK_FOR_UPDATES
          value: "false"
        ports:
        - containerPort: 3000
          name: web
        volumeMounts:
        - name: grafana-storage
          mountPath: /var/lib/grafana
        - name: grafana-provisioning
          mountPath: /etc/grafana/provisioning
        - name: grafana-dashboards
          mountPath: /var/lib/grafana/dashboards
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: grafana-storage
        persistentVolumeClaim:
          claimName: grafana-storage
      - name: grafana-provisioning
        configMap:
          name: grafana-provisioning
      - name: grafana-dashboards
        configMap:
          name: grafana-dashboards
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: $MONITORING_NAMESPACE
  labels:
    app: grafana
    component: monitoring
spec:
  type: ClusterIP
  ports:
  - port: 3000
    targetPort: 3000
    name: web
  selector:
    app: grafana
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafana-storage
  namespace: $MONITORING_NAMESPACE
  labels:
    app: grafana
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd
EOF
    
    # Wait for Grafana to be ready
    kubectl wait --for=condition=Ready pod -l app=grafana -n "$MONITORING_NAMESPACE" --timeout=300s
    
    log "Grafana deployed successfully"
}

# Function to deploy AlertManager
deploy_alertmanager() {
    log "Deploying AlertManager..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alertmanager
  namespace: $MONITORING_NAMESPACE
  labels:
    app: alertmanager
    component: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: alertmanager
  template:
    metadata:
      labels:
        app: alertmanager
        component: monitoring
    spec:
      containers:
      - name: alertmanager
        image: prom/alertmanager:latest
        args:
        - --config.file=/etc/alertmanager/alertmanager.yml
        - --storage.path=/alertmanager
        - --web.external-url=https://yourdomain.com/alertmanager
        - --cluster.listen-address=0.0.0.0:9094
        - --log.level=info
        ports:
        - containerPort: 9093
          name: web
        - containerPort: 9094
          name: cluster
        volumeMounts:
        - name: alertmanager-config
          mountPath: /etc/alertmanager
        - name: alertmanager-storage
          mountPath: /alertmanager
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /-/healthy
            port: 9093
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /-/ready
            port: 9093
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: alertmanager-config
        secret:
          secretName: alertmanager-config
      - name: alertmanager-storage
        persistentVolumeClaim:
          claimName: alertmanager-storage
---
apiVersion: v1
kind: Service
metadata:
  name: alertmanager
  namespace: $MONITORING_NAMESPACE
  labels:
    app: alertmanager
    component: monitoring
spec:
  type: ClusterIP
  ports:
  - port: 9093
    targetPort: 9093
    name: web
  - port: 9094
    targetPort: 9094
    name: cluster
  selector:
    app: alertmanager
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: alertmanager-storage
  namespace: $MONITORING_NAMESPACE
  labels:
    app: alertmanager
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: fast-ssd
EOF
    
    # Wait for AlertManager to be ready
    kubectl wait --for=condition=Ready pod -l app=alertmanager -n "$MONITORING_NAMESPACE" --timeout=300s
    
    log "AlertManager deployed successfully"
}

# Function to create Grafana provisioning
create_grafana_provisioning() {
    log "Creating Grafana provisioning configuration..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-provisioning
  namespace: $MONITORING_NAMESPACE
  labels:
    app: grafana
data:
  datasource.yml: |
    apiVersion: 1
    datasources:
    - name: Prometheus
      type: prometheus
      url: http://prometheus:9090
      access: proxy
      isDefault: true
      editable: true
    - name: AlertManager
      type: alertmanager
      url: http://alertmanager:9093
      access: proxy
      editable: true
  dashboards.yml: |
    apiVersion: 1
    providers:
    - name: 'freeagentics'
      orgId: 1
      folder: 'FreeAgentics'
      type: file
      disableDeletion: false
      editable: true
      options:
        path: /var/lib/grafana/dashboards
  alerting.yml: |
    apiVersion: 1
    contactPoints:
    - orgId: 1
      name: slack-alerts
      receivers:
      - uid: slack-receiver
        type: slack
        settings:
          url: ${SLACK_WEBHOOK}
          channel: '#alerts'
          username: 'Grafana'
          title: 'Grafana Alert'
    policies:
    - orgId: 1
      receiver: slack-alerts
      group_by: ['alertname']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
EOF
    
    log "Grafana provisioning configuration created"
}

# Function to deploy dashboards
deploy_dashboards() {
    log "Deploying Grafana dashboards..."
    
    # Create ConfigMap with dashboard files
    kubectl create configmap grafana-dashboards \
        --from-file=grafana/dashboards/ \
        -n "$MONITORING_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log "Grafana dashboards deployed successfully"
}

# Function to create monitoring ingress
create_monitoring_ingress() {
    log "Creating monitoring ingress..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: monitoring-ingress
  namespace: $MONITORING_NAMESPACE
  labels:
    app: monitoring
    component: ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: monitoring-basic-auth
    nginx.ingress.kubernetes.io/auth-realm: 'Authentication Required'
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - monitoring.yourdomain.com
    secretName: monitoring-tls
  rules:
  - host: monitoring.yourdomain.com
    http:
      paths:
      - path: /grafana
        pathType: Prefix
        backend:
          service:
            name: grafana
            port:
              number: 3000
      - path: /prometheus
        pathType: Prefix
        backend:
          service:
            name: prometheus
            port:
              number: 9090
      - path: /alertmanager
        pathType: Prefix
        backend:
          service:
            name: alertmanager
            port:
              number: 9093
EOF
    
    log "Monitoring ingress created successfully"
}

# Function to run post-deployment tests
run_post_deployment_tests() {
    log "Running post-deployment tests..."
    
    # Test Prometheus
    local prometheus_pod
    prometheus_pod=$(kubectl get pods -n "$MONITORING_NAMESPACE" -l app=prometheus -o jsonpath='{.items[0].metadata.name}')
    
    if kubectl exec -n "$MONITORING_NAMESPACE" "$prometheus_pod" -- wget -q -O - http://localhost:9090/-/healthy | grep -q "Prometheus is Healthy"; then
        log "✅ Prometheus health check passed"
    else
        error "❌ Prometheus health check failed"
        return 1
    fi
    
    # Test Grafana
    local grafana_pod
    grafana_pod=$(kubectl get pods -n "$MONITORING_NAMESPACE" -l app=grafana -o jsonpath='{.items[0].metadata.name}')
    
    if kubectl exec -n "$MONITORING_NAMESPACE" "$grafana_pod" -- curl -s http://localhost:3000/api/health | grep -q "ok"; then
        log "✅ Grafana health check passed"
    else
        error "❌ Grafana health check failed"
        return 1
    fi
    
    # Test AlertManager
    local alertmanager_pod
    alertmanager_pod=$(kubectl get pods -n "$MONITORING_NAMESPACE" -l app=alertmanager -o jsonpath='{.items[0].metadata.name}')
    
    if kubectl exec -n "$MONITORING_NAMESPACE" "$alertmanager_pod" -- wget -q -O - http://localhost:9093/-/healthy | grep -q "OK"; then
        log "✅ AlertManager health check passed"
    else
        error "❌ AlertManager health check failed"
        return 1
    fi
    
    log "All post-deployment tests passed successfully"
}

# Function to show deployment status
show_deployment_status() {
    log "=== MONITORING DEPLOYMENT STATUS ==="
    
    # Show pods
    log "Monitoring Pods:"
    kubectl get pods -n "$MONITORING_NAMESPACE" -o wide
    
    # Show services
    log "Monitoring Services:"
    kubectl get services -n "$MONITORING_NAMESPACE"
    
    # Show ingress
    log "Monitoring Ingress:"
    kubectl get ingress -n "$MONITORING_NAMESPACE"
    
    # Show storage
    log "Monitoring Storage:"
    kubectl get pvc -n "$MONITORING_NAMESPACE"
    
    # Show access information
    log "=== ACCESS INFORMATION ==="
    log "Grafana Admin Password: $GRAFANA_ADMIN_PASSWORD"
    log "Grafana URL: https://monitoring.yourdomain.com/grafana"
    log "Prometheus URL: https://monitoring.yourdomain.com/prometheus"
    log "AlertManager URL: https://monitoring.yourdomain.com/alertmanager"
}

# Main deployment function
main() {
    local start_time
    start_time=$(date +%s)
    
    log "Starting monitoring stack deployment"
    
    # Change to monitoring directory
    cd "$(dirname "$0")"
    
    # Pre-deployment steps
    check_prerequisites
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN MODE - No actual deployment will occur"
        exit 0
    fi
    
    # Deployment steps
    create_monitoring_secrets
    create_grafana_provisioning
    deploy_prometheus
    deploy_grafana
    deploy_alertmanager
    deploy_dashboards
    create_monitoring_ingress
    
    # Post-deployment verification
    run_post_deployment_tests
    show_deployment_status
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "🎉 Monitoring stack deployment completed successfully in ${duration}s!"
}

# Script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -n, --namespace NAMESPACE           Set application namespace (default: freeagentics-prod)"
    echo "  -m, --monitoring-namespace NAMESPACE Set monitoring namespace (default: freeagentics-monitoring)"
    echo "  -c, --context CONTEXT               Set kubectl context"
    echo "  -d, --dry-run                      Perform dry run only"
    echo "  -p, --grafana-password PASSWORD    Set Grafana admin password"
    echo "  -s, --slack-webhook URL            Set Slack webhook URL"
    echo "  -P, --pagerduty-key KEY            Set PagerDuty routing key"
    echo "  -h, --help                         Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  GRAFANA_ADMIN_PASSWORD            Grafana admin password"
    echo "  SLACK_WEBHOOK                     Slack webhook URL"
    echo "  PAGERDUTY_ROUTING_KEY             PagerDuty routing key"
    echo ""
    echo "Examples:"
    echo "  $0                                # Deploy monitoring stack"
    echo "  $0 -d                            # Dry run"
    echo "  $0 -p mypassword                 # Set Grafana password"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -m|--monitoring-namespace)
            MONITORING_NAMESPACE="$2"
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
        -p|--grafana-password)
            GRAFANA_ADMIN_PASSWORD="$2"
            shift 2
            ;;
        -s|--slack-webhook)
            SLACK_WEBHOOK="$2"
            shift 2
            ;;
        -P|--pagerduty-key)
            PAGERDUTY_ROUTING_KEY="$2"
            shift 2
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