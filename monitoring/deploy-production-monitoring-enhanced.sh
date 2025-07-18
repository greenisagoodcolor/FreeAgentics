#!/bin/bash
# Enhanced Production Monitoring Deployment Script
# Deploys comprehensive monitoring stack with advanced alerting and observability

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
MONITORING_NAMESPACE="${MONITORING_NAMESPACE:-freeagentics-monitoring}"
PROMETHEUS_VERSION="${PROMETHEUS_VERSION:-latest}"
GRAFANA_VERSION="${GRAFANA_VERSION:-latest}"
ALERTMANAGER_VERSION="${ALERTMANAGER_VERSION:-latest}"
JAEGER_VERSION="${JAEGER_VERSION:-latest}"
LOKI_VERSION="${LOKI_VERSION:-latest}"
TEMPO_VERSION="${TEMPO_VERSION:-latest}"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
CLUSTER_NAME="${CLUSTER_NAME:-freeagentics-prod}"
DOMAIN="${DOMAIN:-freeagentics.com}"
BACKUP_ENABLED="${BACKUP_ENABLED:-true}"
ENABLE_DISTRIBUTED_TRACING="${ENABLE_DISTRIBUTED_TRACING:-true}"
ENABLE_LOG_AGGREGATION="${ENABLE_LOG_AGGREGATION:-true}"
ENABLE_ADVANCED_METRICS="${ENABLE_ADVANCED_METRICS:-true}"

# Logging
LOG_FILE="/var/log/freeagentics/monitoring-deployment-$(date +%Y%m%d_%H%M%S).log"
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

    # Check required tools
    for tool in docker docker-compose kubectl; do
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

    # Check cluster connectivity (if running on Kubernetes)
    if kubectl cluster-info &> /dev/null; then
        DEPLOYMENT_MODE="kubernetes"
        info "Kubernetes cluster detected, using K8s deployment mode"
    else
        DEPLOYMENT_MODE="docker"
        info "No Kubernetes cluster detected, using Docker Compose deployment mode"
    fi

    # Check available resources
    check_system_resources

    log "Pre-deployment checks completed ✓"
}

# Check system resources
check_system_resources() {
    log "Checking system resources..."

    # Check memory
    available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [[ $available_memory -lt 2048 ]]; then
        warning "Low available memory: ${available_memory}MB. Monitoring stack may require more resources."
    fi

    # Check disk space
    available_disk=$(df -BG /var/lib/docker | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $available_disk -lt 10 ]]; then
        warning "Low disk space: ${available_disk}GB. Monitoring data may fill up disk quickly."
    fi

    log "System resources check completed ✓"
}

# Create monitoring namespace and directories
setup_monitoring_infrastructure() {
    log "Setting up monitoring infrastructure..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Create namespace
        if ! kubectl get namespace "$MONITORING_NAMESPACE" &> /dev/null; then
            kubectl create namespace "$MONITORING_NAMESPACE"
            kubectl label namespace "$MONITORING_NAMESPACE" monitoring=enabled
        fi

        # Create monitoring service account
        kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: monitoring-service-account
  namespace: $MONITORING_NAMESPACE
  labels:
    app: freeagentics-monitoring
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: monitoring-cluster-role
rules:
- apiGroups: [""]
  resources: ["nodes", "nodes/proxy", "services", "endpoints", "pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["extensions"]
  resources: ["ingresses"]
  verbs: ["get", "list", "watch"]
- nonResourceURLs: ["/metrics"]
  verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: monitoring-cluster-role-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: monitoring-cluster-role
subjects:
- kind: ServiceAccount
  name: monitoring-service-account
  namespace: $MONITORING_NAMESPACE
EOF
    else
        # Create Docker network for monitoring
        docker network create freeagentics-monitoring || true

        # Create data directories
        mkdir -p /var/lib/freeagentics/monitoring/{prometheus,grafana,alertmanager,loki,tempo}
        chown -R 65534:65534 /var/lib/freeagentics/monitoring/prometheus
        chown -R 472:472 /var/lib/freeagentics/monitoring/grafana
        chown -R 65534:65534 /var/lib/freeagentics/monitoring/alertmanager
        chown -R 10001:10001 /var/lib/freeagentics/monitoring/loki
        chown -R 10001:10001 /var/lib/freeagentics/monitoring/tempo
    fi

    log "Monitoring infrastructure setup completed ✓"
}

# Deploy Prometheus
deploy_prometheus() {
    log "Deploying Prometheus..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Deploy Prometheus on Kubernetes
        kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: $MONITORING_NAMESPACE
data:
  prometheus.yml: |
$(cat prometheus-production.yml | sed 's/^/    /')
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: $MONITORING_NAMESPACE
  labels:
    app: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      serviceAccountName: monitoring-service-account
      containers:
      - name: prometheus
        image: prom/prometheus:$PROMETHEUS_VERSION
        ports:
        - containerPort: 9090
        args:
        - --config.file=/etc/prometheus/prometheus.yml
        - --storage.tsdb.path=/prometheus
        - --storage.tsdb.retention.time=30d
        - --storage.tsdb.retention.size=10GB
        - --web.enable-lifecycle
        - --web.enable-admin-api
        - --log.level=info
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
        - name: prometheus-storage
          mountPath: /prometheus
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
        configMap:
          name: prometheus-config
      - name: prometheus-storage
        emptyDir:
          sizeLimit: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: $MONITORING_NAMESPACE
  labels:
    app: prometheus
spec:
  ports:
  - port: 9090
    targetPort: 9090
  selector:
    app: prometheus
EOF
    else
        # Deploy Prometheus with Docker Compose
        docker run -d \
            --name freeagentics-prometheus \
            --network freeagentics-monitoring \
            -p 9090:9090 \
            -v "$(pwd)/prometheus-production.yml":/etc/prometheus/prometheus.yml:ro \
            -v "$(pwd)/rules":/etc/prometheus/rules:ro \
            -v /var/lib/freeagentics/monitoring/prometheus:/prometheus \
            prom/prometheus:$PROMETHEUS_VERSION \
            --config.file=/etc/prometheus/prometheus.yml \
            --storage.tsdb.path=/prometheus \
            --storage.tsdb.retention.time=30d \
            --storage.tsdb.retention.size=10GB \
            --web.enable-lifecycle \
            --web.enable-admin-api \
            --log.level=info
    fi

    log "Prometheus deployment completed ✓"
}

# Deploy Grafana
deploy_grafana() {
    log "Deploying Grafana..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Deploy Grafana on Kubernetes
        kubectl apply -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: grafana-credentials
  namespace: $MONITORING_NAMESPACE
type: Opaque
stringData:
  admin-password: "$(openssl rand -base64 16)"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: $MONITORING_NAMESPACE
  labels:
    app: grafana
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:$GRAFANA_VERSION
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: grafana-credentials
              key: admin-password
        - name: GF_SECURITY_ADMIN_USER
          value: "admin"
        - name: GF_USERS_ALLOW_SIGN_UP
          value: "false"
        - name: GF_INSTALL_PLUGINS
          value: "grafana-clock-panel,grafana-simple-json-datasource,grafana-piechart-panel,grafana-worldmap-panel,grafana-polystat-panel"
        - name: GF_SECURITY_DISABLE_GRAVATAR
          value: "true"
        - name: GF_ANALYTICS_REPORTING_ENABLED
          value: "false"
        - name: GF_ANALYTICS_CHECK_FOR_UPDATES
          value: "false"
        - name: GF_SERVER_DOMAIN
          value: "$DOMAIN"
        - name: GF_SERVER_ROOT_URL
          value: "https://$DOMAIN/grafana"
        - name: GF_SERVER_SERVE_FROM_SUB_PATH
          value: "true"
        volumeMounts:
        - name: grafana-storage
          mountPath: /var/lib/grafana
        - name: grafana-provisioning
          mountPath: /etc/grafana/provisioning
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
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
        emptyDir:
          sizeLimit: 5Gi
      - name: grafana-provisioning
        configMap:
          name: grafana-provisioning
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: $MONITORING_NAMESPACE
  labels:
    app: grafana
spec:
  ports:
  - port: 3000
    targetPort: 3000
  selector:
    app: grafana
EOF
    else
        # Deploy Grafana with Docker
        docker run -d \
            --name freeagentics-grafana \
            --network freeagentics-monitoring \
            -p 3000:3000 \
            -e GF_SECURITY_ADMIN_PASSWORD="$(openssl rand -base64 16)" \
            -e GF_SECURITY_ADMIN_USER=admin \
            -e GF_USERS_ALLOW_SIGN_UP=false \
            -e GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource,grafana-piechart-panel,grafana-worldmap-panel,grafana-polystat-panel \
            -e GF_SECURITY_DISABLE_GRAVATAR=true \
            -e GF_ANALYTICS_REPORTING_ENABLED=false \
            -e GF_ANALYTICS_CHECK_FOR_UPDATES=false \
            -v /var/lib/freeagentics/monitoring/grafana:/var/lib/grafana \
            -v "$(pwd)/grafana/provisioning":/etc/grafana/provisioning:ro \
            -v "$(pwd)/grafana/dashboards":/var/lib/grafana/dashboards:ro \
            grafana/grafana:$GRAFANA_VERSION
    fi

    log "Grafana deployment completed ✓"
}

# Deploy AlertManager
deploy_alertmanager() {
    log "Deploying AlertManager..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Deploy AlertManager on Kubernetes
        kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: $MONITORING_NAMESPACE
data:
  alertmanager.yml: |
$(cat alertmanager-production.yml | sed 's/^/    /')
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alertmanager
  namespace: $MONITORING_NAMESPACE
  labels:
    app: alertmanager
spec:
  replicas: 1
  selector:
    matchLabels:
      app: alertmanager
  template:
    metadata:
      labels:
        app: alertmanager
    spec:
      containers:
      - name: alertmanager
        image: prom/alertmanager:$ALERTMANAGER_VERSION
        ports:
        - containerPort: 9093
        args:
        - --config.file=/etc/alertmanager/alertmanager.yml
        - --storage.path=/alertmanager
        - --web.external-url=https://$DOMAIN/alertmanager
        - --web.route-prefix=/alertmanager
        - --log.level=info
        volumeMounts:
        - name: alertmanager-config
          mountPath: /etc/alertmanager
        - name: alertmanager-storage
          mountPath: /alertmanager
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
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
        configMap:
          name: alertmanager-config
      - name: alertmanager-storage
        emptyDir:
          sizeLimit: 2Gi
---
apiVersion: v1
kind: Service
metadata:
  name: alertmanager
  namespace: $MONITORING_NAMESPACE
  labels:
    app: alertmanager
spec:
  ports:
  - port: 9093
    targetPort: 9093
  selector:
    app: alertmanager
EOF
    else
        # Deploy AlertManager with Docker
        docker run -d \
            --name freeagentics-alertmanager \
            --network freeagentics-monitoring \
            -p 9093:9093 \
            -v "$(pwd)/alertmanager-production.yml":/etc/alertmanager/alertmanager.yml:ro \
            -v /var/lib/freeagentics/monitoring/alertmanager:/alertmanager \
            prom/alertmanager:$ALERTMANAGER_VERSION \
            --config.file=/etc/alertmanager/alertmanager.yml \
            --storage.path=/alertmanager \
            --web.external-url=http://localhost:9093 \
            --log.level=info
    fi

    log "AlertManager deployment completed ✓"
}

# Deploy Jaeger for distributed tracing
deploy_jaeger() {
    if [[ "$ENABLE_DISTRIBUTED_TRACING" != "true" ]]; then
        return
    fi

    log "Deploying Jaeger for distributed tracing..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Deploy Jaeger on Kubernetes
        kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger
  namespace: $MONITORING_NAMESPACE
  labels:
    app: jaeger
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jaeger
  template:
    metadata:
      labels:
        app: jaeger
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/all-in-one:$JAEGER_VERSION
        ports:
        - containerPort: 16686
        - containerPort: 14268
        - containerPort: 6831
          protocol: UDP
        - containerPort: 6832
          protocol: UDP
        env:
        - name: COLLECTOR_OTLP_ENABLED
          value: "true"
        - name: COLLECTOR_ZIPKIN_HOST_PORT
          value: ":9411"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: jaeger
  namespace: $MONITORING_NAMESPACE
  labels:
    app: jaeger
spec:
  ports:
  - port: 16686
    targetPort: 16686
    name: query
  - port: 14268
    targetPort: 14268
    name: collector
  - port: 6831
    targetPort: 6831
    protocol: UDP
    name: agent-compact
  - port: 6832
    targetPort: 6832
    protocol: UDP
    name: agent-binary
  selector:
    app: jaeger
EOF
    else
        # Deploy Jaeger with Docker
        docker run -d \
            --name freeagentics-jaeger \
            --network freeagentics-monitoring \
            -p 16686:16686 \
            -p 14268:14268 \
            -p 6831:6831/udp \
            -p 6832:6832/udp \
            -e COLLECTOR_OTLP_ENABLED=true \
            -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
            jaegertracing/all-in-one:$JAEGER_VERSION
    fi

    log "Jaeger deployment completed ✓"
}

# Deploy Loki for log aggregation
deploy_loki() {
    if [[ "$ENABLE_LOG_AGGREGATION" != "true" ]]; then
        return
    fi

    log "Deploying Loki for log aggregation..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Deploy Loki on Kubernetes
        kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: loki-config
  namespace: $MONITORING_NAMESPACE
data:
  loki.yml: |
    auth_enabled: false
    server:
      http_listen_port: 3100
    ingester:
      lifecycler:
        address: 127.0.0.1
        ring:
          kvstore:
            store: inmemory
          replication_factor: 1
    schema_config:
      configs:
        - from: 2020-10-24
          store: boltdb-shipper
          object_store: filesystem
          schema: v11
          index:
            prefix: index_
            period: 24h
    storage_config:
      boltdb_shipper:
        active_index_directory: /loki/boltdb-shipper-active
        cache_location: /loki/boltdb-shipper-cache
        shared_store: filesystem
      filesystem:
        directory: /loki/chunks
    limits_config:
      enforce_metric_name: false
      reject_old_samples: true
      reject_old_samples_max_age: 168h
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loki
  namespace: $MONITORING_NAMESPACE
  labels:
    app: loki
spec:
  replicas: 1
  selector:
    matchLabels:
      app: loki
  template:
    metadata:
      labels:
        app: loki
    spec:
      containers:
      - name: loki
        image: grafana/loki:$LOKI_VERSION
        ports:
        - containerPort: 3100
        args:
        - -config.file=/etc/loki/loki.yml
        volumeMounts:
        - name: loki-config
          mountPath: /etc/loki
        - name: loki-storage
          mountPath: /loki
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
      volumes:
      - name: loki-config
        configMap:
          name: loki-config
      - name: loki-storage
        emptyDir:
          sizeLimit: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: loki
  namespace: $MONITORING_NAMESPACE
  labels:
    app: loki
spec:
  ports:
  - port: 3100
    targetPort: 3100
  selector:
    app: loki
EOF
    else
        # Deploy Loki with Docker
        docker run -d \
            --name freeagentics-loki \
            --network freeagentics-monitoring \
            -p 3100:3100 \
            -v /var/lib/freeagentics/monitoring/loki:/loki \
            grafana/loki:$LOKI_VERSION \
            -config.file=/etc/loki/local-config.yaml
    fi

    log "Loki deployment completed ✓"
}

# Deploy Node Exporter for system metrics
deploy_node_exporter() {
    log "Deploying Node Exporter..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Deploy Node Exporter as DaemonSet on Kubernetes
        kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-exporter
  namespace: $MONITORING_NAMESPACE
  labels:
    app: node-exporter
spec:
  selector:
    matchLabels:
      app: node-exporter
  template:
    metadata:
      labels:
        app: node-exporter
    spec:
      hostNetwork: true
      hostPID: true
      containers:
      - name: node-exporter
        image: prom/node-exporter:latest
        ports:
        - containerPort: 9100
        args:
        - --path.procfs=/host/proc
        - --path.sysfs=/host/sys
        - --collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)
        - --collector.systemd
        - --collector.processes
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
        - name: root
          mountPath: /rootfs
          readOnly: true
        resources:
          requests:
            memory: "64Mi"
            cpu: "50m"
          limits:
            memory: "128Mi"
            cpu: "100m"
        securityContext:
          runAsNonRoot: true
          runAsUser: 65534
      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
      - name: root
        hostPath:
          path: /
      tolerations:
      - key: node-role.kubernetes.io/master
        effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: node-exporter
  namespace: $MONITORING_NAMESPACE
  labels:
    app: node-exporter
spec:
  ports:
  - port: 9100
    targetPort: 9100
  selector:
    app: node-exporter
EOF
    else
        # Deploy Node Exporter with Docker
        docker run -d \
            --name freeagentics-node-exporter \
            --network freeagentics-monitoring \
            -p 9100:9100 \
            -v /proc:/host/proc:ro \
            -v /sys:/host/sys:ro \
            -v /:/rootfs:ro \
            --pid host \
            prom/node-exporter:latest \
            --path.procfs=/host/proc \
            --path.sysfs=/host/sys \
            --collector.filesystem.ignored-mount-points='^/(sys|proc|dev|host|etc)($$|/)'
    fi

    log "Node Exporter deployment completed ✓"
}

# Deploy cAdvisor for container metrics
deploy_cadvisor() {
    log "Deploying cAdvisor..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Deploy cAdvisor as DaemonSet on Kubernetes
        kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: cadvisor
  namespace: $MONITORING_NAMESPACE
  labels:
    app: cadvisor
spec:
  selector:
    matchLabels:
      app: cadvisor
  template:
    metadata:
      labels:
        app: cadvisor
    spec:
      hostNetwork: true
      containers:
      - name: cadvisor
        image: gcr.io/cadvisor/cadvisor:latest
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: rootfs
          mountPath: /rootfs
          readOnly: true
        - name: var-run
          mountPath: /var/run
          readOnly: true
        - name: sys
          mountPath: /sys
          readOnly: true
        - name: docker
          mountPath: /var/lib/docker
          readOnly: true
        - name: disk
          mountPath: /dev/disk
          readOnly: true
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
      volumes:
      - name: rootfs
        hostPath:
          path: /
      - name: var-run
        hostPath:
          path: /var/run
      - name: sys
        hostPath:
          path: /sys
      - name: docker
        hostPath:
          path: /var/lib/docker
      - name: disk
        hostPath:
          path: /dev/disk
      tolerations:
      - key: node-role.kubernetes.io/master
        effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: cadvisor
  namespace: $MONITORING_NAMESPACE
  labels:
    app: cadvisor
spec:
  ports:
  - port: 8080
    targetPort: 8080
  selector:
    app: cadvisor
EOF
    else
        # Deploy cAdvisor with Docker
        docker run -d \
            --name freeagentics-cadvisor \
            --network freeagentics-monitoring \
            -p 8080:8080 \
            -v /:/rootfs:ro \
            -v /var/run:/var/run:rw \
            -v /sys:/sys:ro \
            -v /var/lib/docker/:/var/lib/docker:ro \
            -v /dev/disk/:/dev/disk:ro \
            --privileged \
            gcr.io/cadvisor/cadvisor:latest
    fi

    log "cAdvisor deployment completed ✓"
}

# Setup monitoring ingress
setup_monitoring_ingress() {
    if [[ "$DEPLOYMENT_MODE" != "kubernetes" ]]; then
        return
    fi

    log "Setting up monitoring ingress..."

    kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: monitoring-ingress
  namespace: $MONITORING_NAMESPACE
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rewrite-target: /\$2
    nginx.ingress.kubernetes.io/configuration-snippet: |
      rewrite ^(/prometheus)$ \$1/ redirect;
      rewrite ^(/grafana)$ \$1/ redirect;
      rewrite ^(/alertmanager)$ \$1/ redirect;
spec:
  tls:
  - hosts:
    - $DOMAIN
    secretName: monitoring-tls
  rules:
  - host: $DOMAIN
    http:
      paths:
      - path: /prometheus(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: prometheus
            port:
              number: 9090
      - path: /grafana(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: grafana
            port:
              number: 3000
      - path: /alertmanager(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: alertmanager
            port:
              number: 9093
EOF

    if [[ "$ENABLE_DISTRIBUTED_TRACING" == "true" ]]; then
        kubectl patch ingress monitoring-ingress -n "$MONITORING_NAMESPACE" --type='json' -p='[
          {
            "op": "add",
            "path": "/spec/rules/0/http/paths/-",
            "value": {
              "path": "/jaeger(/|$)(.*)",
              "pathType": "Prefix",
              "backend": {
                "service": {
                  "name": "jaeger",
                  "port": {
                    "number": 16686
                  }
                }
              }
            }
          }
        ]'
    fi

    log "Monitoring ingress setup completed ✓"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for monitoring services to be ready..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Wait for Kubernetes deployments
        kubectl wait --for=condition=available deployment/prometheus -n "$MONITORING_NAMESPACE" --timeout=300s
        kubectl wait --for=condition=available deployment/grafana -n "$MONITORING_NAMESPACE" --timeout=300s
        kubectl wait --for=condition=available deployment/alertmanager -n "$MONITORING_NAMESPACE" --timeout=300s

        if [[ "$ENABLE_DISTRIBUTED_TRACING" == "true" ]]; then
            kubectl wait --for=condition=available deployment/jaeger -n "$MONITORING_NAMESPACE" --timeout=300s
        fi

        if [[ "$ENABLE_LOG_AGGREGATION" == "true" ]]; then
            kubectl wait --for=condition=available deployment/loki -n "$MONITORING_NAMESPACE" --timeout=300s
        fi
    else
        # Wait for Docker containers
        for service in prometheus grafana alertmanager; do
            for i in {1..30}; do
                if docker ps | grep -q "freeagentics-${service}"; then
                    log "$service is running"
                    break
                fi
                sleep 10
            done
        done
    fi

    log "All monitoring services are ready ✓"
}

# Configure Grafana datasources and dashboards
configure_grafana() {
    log "Configuring Grafana datasources and dashboards..."

    # Wait for Grafana to be fully ready
    sleep 30

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        GRAFANA_URL="http://grafana:3000"
        kubectl exec -n "$MONITORING_NAMESPACE" deployment/grafana -- grafana-cli admin reset-admin-password admin
    else
        GRAFANA_URL="http://localhost:3000"
    fi

    # Import dashboards
    "./deploy-dashboards.sh" || warning "Failed to import some dashboards"

    log "Grafana configuration completed ✓"
}

# Run health checks
run_health_checks() {
    log "Running health checks..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Kubernetes health checks
        kubectl get pods -n "$MONITORING_NAMESPACE" -o wide

        # Check if all pods are running
        NOT_READY=$(kubectl get pods -n "$MONITORING_NAMESPACE" --field-selector=status.phase!=Running --no-headers | wc -l)
        if [[ $NOT_READY -gt 0 ]]; then
            warning "$NOT_READY pods are not in Running state"
        fi
    else
        # Docker health checks
        docker ps --filter="name=freeagentics-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

        # Test endpoints
        for service in prometheus:9090 grafana:3000 alertmanager:9093; do
            name=$(echo "$service" | cut -d: -f1)
            port=$(echo "$service" | cut -d: -f2)

            if curl -sf "http://localhost:$port/api/health" &>/dev/null || curl -sf "http://localhost:$port/-/healthy" &>/dev/null; then
                log "$name health check passed ✓"
            else
                warning "$name health check failed"
            fi
        done
    fi

    log "Health checks completed ✓"
}

# Create backup job
create_backup_job() {
    if [[ "$BACKUP_ENABLED" != "true" ]]; then
        return
    fi

    log "Creating backup job..."

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        # Create backup CronJob for Kubernetes
        kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: monitoring-backup
  namespace: $MONITORING_NAMESPACE
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: alpine:latest
            command: ["/bin/sh", "-c"]
            args:
            - |
              apk add --no-cache curl
              BACKUP_DIR="/backups/\$(date +%Y%m%d_%H%M%S)"
              mkdir -p "\$BACKUP_DIR"

              # Backup Prometheus data
              curl -XPOST http://prometheus:9090/api/v1/admin/tsdb/snapshot

              # Backup Grafana dashboards
              curl -u admin:admin http://grafana:3000/api/search | jq -r '.[] | select(.type == "dash-db") | .uid' | while read uid; do
                curl -u admin:admin "http://grafana:3000/api/dashboards/uid/\$uid" > "\$BACKUP_DIR/dashboard-\$uid.json"
              done

              echo "Backup completed at \$BACKUP_DIR"
            volumeMounts:
            - name: backup-storage
              mountPath: /backups
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
EOF
    else
        # Create backup script for Docker
        cat > /etc/cron.d/monitoring-backup <<EOF
0 2 * * * root /usr/local/bin/backup-monitoring.sh
EOF

        cat > /usr/local/bin/backup-monitoring.sh <<'EOF'
#!/bin/bash
BACKUP_DIR="/var/backups/freeagentics-monitoring/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup Prometheus data
docker exec freeagentics-prometheus curl -XPOST http://localhost:9090/api/v1/admin/tsdb/snapshot

# Backup Grafana data
docker exec freeagentics-grafana tar -czf - /var/lib/grafana > "$BACKUP_DIR/grafana-data.tar.gz"

# Backup AlertManager data
docker exec freeagentics-alertmanager tar -czf - /alertmanager > "$BACKUP_DIR/alertmanager-data.tar.gz"

echo "Backup completed at $BACKUP_DIR"
EOF
        chmod +x /usr/local/bin/backup-monitoring.sh
    fi

    log "Backup job created ✓"
}

# Post-deployment tasks
post_deployment() {
    log "Running post-deployment tasks..."

    # Create backup job
    create_backup_job

    # Display access information
    display_access_info

    # Send notifications
    send_notifications

    log "Post-deployment tasks completed ✓"
}

# Display access information
display_access_info() {
    log "Displaying access information..."

    echo ""
    echo "=== Monitoring Stack Access Information ==="
    echo ""

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        echo "Kubernetes Deployment:"
        echo "- Namespace: $MONITORING_NAMESPACE"
        echo "- Cluster: $CLUSTER_NAME"
        echo ""
        echo "Access URLs (via ingress):"
        echo "- Prometheus: https://$DOMAIN/prometheus"
        echo "- Grafana: https://$DOMAIN/grafana"
        echo "- AlertManager: https://$DOMAIN/alertmanager"

        if [[ "$ENABLE_DISTRIBUTED_TRACING" == "true" ]]; then
            echo "- Jaeger: https://$DOMAIN/jaeger"
        fi

        echo ""
        echo "Port Forward Commands:"
        echo "- Prometheus: kubectl port-forward -n $MONITORING_NAMESPACE svc/prometheus 9090:9090"
        echo "- Grafana: kubectl port-forward -n $MONITORING_NAMESPACE svc/grafana 3000:3000"
        echo "- AlertManager: kubectl port-forward -n $MONITORING_NAMESPACE svc/alertmanager 9093:9093"

        echo ""
        echo "Grafana Credentials:"
        echo "- Username: admin"
        echo "- Password: $(kubectl get secret grafana-credentials -n "$MONITORING_NAMESPACE" -o jsonpath='{.data.admin-password}' | base64 -d)"
    else
        echo "Docker Deployment:"
        echo ""
        echo "Access URLs:"
        echo "- Prometheus: http://localhost:9090"
        echo "- Grafana: http://localhost:3000"
        echo "- AlertManager: http://localhost:9093"
        echo "- Node Exporter: http://localhost:9100"
        echo "- cAdvisor: http://localhost:8080"

        if [[ "$ENABLE_DISTRIBUTED_TRACING" == "true" ]]; then
            echo "- Jaeger: http://localhost:16686"
        fi

        if [[ "$ENABLE_LOG_AGGREGATION" == "true" ]]; then
            echo "- Loki: http://localhost:3100"
        fi

        echo ""
        echo "Grafana Credentials:"
        echo "- Username: admin"
        echo "- Password: $(docker exec freeagentics-grafana printenv GF_SECURITY_ADMIN_PASSWORD)"
    fi

    echo ""
    echo "=== End of Access Information ==="
}

# Send notifications
send_notifications() {
    log "Sending deployment notifications..."

    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"✅ Monitoring stack deployed successfully!\n**Environment:** $DEPLOYMENT_ENV\n**Mode:** $DEPLOYMENT_MODE\n**Cluster:** $CLUSTER_NAME\n**Domain:** $DOMAIN\"}" \
            "$SLACK_WEBHOOK"
    fi

    if [[ -n "${TEAMS_WEBHOOK:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"✅ FreeAgentics monitoring stack deployed successfully! Environment: $DEPLOYMENT_ENV, Mode: $DEPLOYMENT_MODE\"}" \
            "$TEAMS_WEBHOOK"
    fi

    log "Notifications sent ✓"
}

# Main deployment function
main() {
    log "Starting FreeAgentics monitoring stack deployment"
    log "Environment: $DEPLOYMENT_ENV"
    log "Mode: $DEPLOYMENT_MODE"
    log "Cluster: $CLUSTER_NAME"
    log "Domain: $DOMAIN"
    log "Namespace: $MONITORING_NAMESPACE"

    # Run deployment steps
    pre_deployment_checks
    setup_monitoring_infrastructure
    deploy_prometheus
    deploy_grafana
    deploy_alertmanager

    if [[ "$ENABLE_DISTRIBUTED_TRACING" == "true" ]]; then
        deploy_jaeger
    fi

    if [[ "$ENABLE_LOG_AGGREGATION" == "true" ]]; then
        deploy_loki
    fi

    deploy_node_exporter
    deploy_cadvisor

    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        setup_monitoring_ingress
    fi

    wait_for_services
    configure_grafana
    run_health_checks
    post_deployment

    log "Monitoring stack deployment completed successfully! 📊"
    log "Access the monitoring dashboards using the URLs provided above"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace)
            MONITORING_NAMESPACE="$2"
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
        --env)
            DEPLOYMENT_ENV="$2"
            shift 2
            ;;
        --no-tracing)
            ENABLE_DISTRIBUTED_TRACING="false"
            shift
            ;;
        --no-logging)
            ENABLE_LOG_AGGREGATION="false"
            shift
            ;;
        --no-backup)
            BACKUP_ENABLED="false"
            shift
            ;;
        --help)
            echo "FreeAgentics Monitoring Stack Deployment Script"
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --namespace NAME      Monitoring namespace (default: freeagentics-monitoring)"
            echo "  --domain DOMAIN       Domain name (default: freeagentics.com)"
            echo "  --cluster NAME        Cluster name (default: freeagentics-prod)"
            echo "  --env ENV             Environment (default: production)"
            echo "  --no-tracing          Disable distributed tracing"
            echo "  --no-logging          Disable log aggregation"
            echo "  --no-backup           Disable backup job"
            echo "  --help                Show this help message"
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
