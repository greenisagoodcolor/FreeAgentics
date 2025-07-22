#!/usr/bin/env python3
"""
Deploy security metrics and monitoring configuration.
"""

import os
import sys
from pathlib import Path
from typing import Dict

import yaml


class SecurityMetricsDeployer:
    """Deploy security monitoring and metrics."""

    def __init__(self):
        self.metrics_config = {
            "authentication": [
                "auth_login_attempts_total",
                "auth_login_failures_total",
                "auth_login_success_total",
                "auth_token_issued_total",
                "auth_token_revoked_total",
                "auth_session_created_total",
                "auth_session_destroyed_total",
            ],
            "authorization": [
                "rbac_access_granted_total",
                "rbac_access_denied_total",
                "rbac_permission_checks_total",
                "rbac_role_assignments_total",
            ],
            "api_security": [
                "api_requests_total",
                "api_request_errors_total",
                "api_rate_limit_exceeded_total",
                "api_authentication_failures_total",
                "api_suspicious_requests_total",
            ],
            "data_security": [
                "sensitive_data_access_total",
                "data_encryption_operations_total",
                "data_decryption_operations_total",
                "data_export_bytes_total",
                "unencrypted_transmission_total",
            ],
            "threat_detection": [
                "security_incidents_total",
                "security_events_total",
                "suspicious_activity_total",
                "injection_attempts_total",
                "xss_attempts_total",
                "csrf_attempts_total",
                "sql_injection_attempts_total",
            ],
            "compliance": [
                "compliance_score",
                "audit_log_entries_total",
                "audit_log_tampering_attempts_total",
                "security_policy_violations_total",
            ],
            "infrastructure": [
                "ssl_certificate_expiry_seconds",
                "container_escape_attempts_total",
                "kubernetes_api_unauthorized_total",
                "file_integrity_violations_total",
            ],
        }

    def create_prometheus_config(self) -> Dict:
        """Create Prometheus configuration for security metrics."""
        config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s",
                "external_labels": {"monitor": "freeagentics-security"},
            },
            "rule_files": ["/etc/prometheus/rules/security-alerts.yml"],
            "scrape_configs": [
                {
                    "job_name": "freeagentics-security",
                    "static_configs": [
                        {
                            "targets": ["localhost:8000"],
                            "labels": {
                                "service": "freeagentics",
                                "environment": os.getenv("ENVIRONMENT", "production"),
                            },
                        }
                    ],
                    "metrics_path": "/metrics",
                    "scheme": "https",
                    "tls_config": {"insecure_skip_verify": False},
                }
            ],
            "alerting": {
                "alertmanagers": [{"static_configs": [{"targets": ["alertmanager:9093"]}]}]
            },
        }

        return config

    def create_grafana_datasource(self) -> Dict:
        """Create Grafana datasource configuration."""
        return {
            "apiVersion": 1,
            "datasources": [
                {
                    "name": "Prometheus-Security",
                    "type": "prometheus",
                    "access": "proxy",
                    "url": "http://prometheus:9090",
                    "basicAuth": False,
                    "isDefault": False,
                    "jsonData": {
                        "timeInterval": "15s",
                        "queryTimeout": "60s",
                        "httpMethod": "POST",
                    },
                }
            ],
        }

    def create_alertmanager_config(self) -> Dict:
        """Create Alertmanager configuration."""
        return {
            "global": {"resolve_timeout": "5m"},
            "route": {
                "group_by": ["alertname", "cluster", "service"],
                "group_wait": "10s",
                "group_interval": "10s",
                "repeat_interval": "12h",
                "receiver": "security-team",
                "routes": [
                    {
                        "match": {
                            "severity": "critical",
                            "category": "security",
                        },
                        "receiver": "security-critical",
                        "continue": True,
                    },
                    {
                        "match": {"severity": "high", "category": "security"},
                        "receiver": "security-high",
                        "continue": True,
                    },
                ],
            },
            "receivers": [
                {
                    "name": "security-team",
                    "email_configs": [
                        {
                            "to": "security@freeagentics.com",
                            "from": "alerts@freeagentics.com",
                            "smarthost": "smtp.gmail.com:587",
                            "auth_username": "alerts@freeagentics.com",
                            "auth_password": os.getenv("SMTP_PASSWORD", ""),
                            "headers": {"Subject": "Security Alert: {{ .GroupLabels.alertname }}"},
                        }
                    ],
                },
                {
                    "name": "security-critical",
                    "pagerduty_configs": [
                        {
                            "service_key": os.getenv("PAGERDUTY_KEY", ""),
                            "description": "{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}",
                        }
                    ],
                    "slack_configs": [
                        {
                            "api_url": os.getenv("SLACK_WEBHOOK", ""),
                            "channel": "#security-alerts",
                            "title": "Critical Security Alert",
                            "text": "{{ .CommonAnnotations.description }}",
                        }
                    ],
                },
                {
                    "name": "security-high",
                    "slack_configs": [
                        {
                            "api_url": os.getenv("SLACK_WEBHOOK", ""),
                            "channel": "#security-alerts",
                            "title": "High Security Alert",
                            "text": "{{ .CommonAnnotations.description }}",
                        }
                    ],
                },
            ],
            "inhibit_rules": [
                {
                    "source_match": {"severity": "critical"},
                    "target_match": {"severity": "warning"},
                    "equal": ["alertname", "dev", "instance"],
                }
            ],
        }

    def create_metric_collection_code(self) -> str:
        """Generate Python code for metric collection."""
        return '''"""
Security metrics collection for Prometheus.
"""

from prometheus_client import Counter, Gauge, Histogram, Info
from functools import wraps
import time
from typing import Callable

# Authentication metrics
auth_login_attempts = Counter(
    'auth_login_attempts_total',
    'Total number of login attempts',
    ['status', 'method']
)

auth_login_failures = Counter(
    'auth_login_failures_total',
    'Total number of failed login attempts',
    ['reason', 'ip']
)

auth_login_success = Counter(
    'auth_login_success_total',
    'Total number of successful logins',
    ['method']
)

auth_token_issued = Counter(
    'auth_token_issued_total',
    'Total number of tokens issued',
    ['type']
)

auth_token_revoked = Counter(
    'auth_token_revoked_total',
    'Total number of tokens revoked',
    ['reason']
)

# Authorization metrics
rbac_access_granted = Counter(
    'rbac_access_granted_total',
    'Total number of granted access requests',
    ['resource', 'action']
)

rbac_access_denied = Counter(
    'rbac_access_denied_total',
    'Total number of denied access requests',
    ['resource', 'action', 'reason']
)

# API Security metrics
api_requests = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

api_rate_limit_exceeded = Counter(
    'api_rate_limit_exceeded_total',
    'Total number of rate limit violations',
    ['endpoint', 'ip']
)

api_suspicious_requests = Counter(
    'api_suspicious_requests_total',
    'Total number of suspicious API requests',
    ['type', 'endpoint']
)

# Data security metrics
sensitive_data_access = Counter(
    'sensitive_data_access_total',
    'Total sensitive data access events',
    ['resource', 'operation', 'user']
)

data_encryption_operations = Counter(
    'data_encryption_operations_total',
    'Total encryption operations',
    ['algorithm', 'key_size']
)

data_export_bytes = Counter(
    'data_export_bytes_total',
    'Total bytes exported',
    ['format', 'user']
)

# Threat detection metrics
security_incidents = Counter(
    'security_incidents_total',
    'Total security incidents',
    ['type', 'severity']
)

injection_attempts = Counter(
    'injection_attempts_total',
    'Total injection attempts detected',
    ['type', 'endpoint']
)

# Compliance metrics
compliance_score = Gauge(
    'compliance_score',
    'Current compliance score',
    ['framework']
)

audit_log_entries = Counter(
    'audit_log_entries_total',
    'Total audit log entries',
    ['action', 'user', 'resource']
)

# Infrastructure metrics
ssl_certificate_expiry = Gauge(
    'ssl_certificate_expiry_seconds',
    'SSL certificate expiry time in seconds',
    ['domain']
)

# Decorators for automatic metric collection
def track_authentication(func: Callable) -> Callable:
    """Track authentication metrics."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            auth_login_success.labels(method=func.__name__).inc()
            return result
        except Exception as e:
            auth_login_failures.labels(
                reason=type(e).__name__,
                ip=kwargs.get('ip', 'unknown')
            ).inc()
            raise
    return wrapper

def track_api_request(func: Callable) -> Callable:
    """Track API request metrics."""
    @wraps(func)
    async def wrapper(request, *args, **kwargs):
        start_time = time.time()
        status = 500

        try:
            response = await func(request, *args, **kwargs)
            status = response.status_code
            return response
        finally:
            api_requests.labels(
                method=request.method,
                endpoint=request.url.path,
                status=status
            ).inc()

            # Check for suspicious patterns
            if any(pattern in str(request.url) for pattern in [
                '../', '%2e%2e', 'UNION', 'SELECT', '<script'
            ]):
                api_suspicious_requests.labels(
                    type='injection',
                    endpoint=request.url.path
                ).inc()

    return wrapper

def track_data_access(resource: str) -> Callable:
    """Track sensitive data access."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get('user', 'system')
            sensitive_data_access.labels(
                resource=resource,
                operation=func.__name__,
                user=user
            ).inc()
            return await func(*args, **kwargs)
        return wrapper
    return decorator
'''

    def deploy_configs(self):
        """Deploy all monitoring configurations."""
        print("Deploying security monitoring configuration...")

        # Create directories
        dirs = [
            "monitoring/prometheus",
            "monitoring/grafana/provisioning/datasources",
            "monitoring/grafana/provisioning/dashboards",
            "monitoring/alertmanager",
            "observability/metrics",
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Deploy Prometheus config
        prometheus_config = self.create_prometheus_config()
        with open("monitoring/prometheus/prometheus-security.yml", "w") as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        print("✓ Prometheus configuration deployed")

        # Deploy Grafana datasource
        datasource_config = self.create_grafana_datasource()
        with open(
            "monitoring/grafana/provisioning/datasources/prometheus-security.yml",
            "w",
        ) as f:
            yaml.dump(datasource_config, f, default_flow_style=False)
        print("✓ Grafana datasource configuration deployed")

        # Deploy Alertmanager config
        alertmanager_config = self.create_alertmanager_config()
        with open("monitoring/alertmanager/alertmanager-security.yml", "w") as f:
            yaml.dump(alertmanager_config, f, default_flow_style=False)
        print("✓ Alertmanager configuration deployed")

        # Deploy metric collection code
        metrics_code = self.create_metric_collection_code()
        with open("observability/metrics/security_metrics.py", "w") as f:
            f.write(metrics_code)
        print("✓ Security metrics collection code deployed")

        # Create Docker Compose for monitoring stack
        docker_compose = {
            "version": "3.8",
            "services": {
                "prometheus": {
                    "image": "prom/prometheus:latest",
                    "container_name": "prometheus-security",
                    "volumes": [
                        "./monitoring/prometheus:/etc/prometheus",
                        "prometheus_data:/prometheus",
                    ],
                    "command": [
                        "--config.file=/etc/prometheus/prometheus-security.yml",
                        "--storage.tsdb.path=/prometheus",
                        "--web.console.libraries=/etc/prometheus/console_libraries",
                        "--web.console.templates=/etc/prometheus/consoles",
                        "--storage.tsdb.retention.time=30d",
                        "--web.enable-lifecycle",
                    ],
                    "ports": ["9090:9090"],
                    "networks": ["monitoring"],
                },
                "grafana": {
                    "image": "grafana/grafana:latest",
                    "container_name": "grafana-security",
                    "volumes": [
                        "./monitoring/grafana/provisioning:/etc/grafana/provisioning",
                        "./monitoring/grafana/dashboards:/var/lib/grafana/dashboards",
                        "grafana_data:/var/lib/grafana",
                    ],
                    "environment": {
                        "GF_SECURITY_ADMIN_USER": "${GRAFANA_USER:-admin}",
                        "GF_SECURITY_ADMIN_PASSWORD": "${GRAFANA_PASSWORD:-admin}",
                        "GF_USERS_ALLOW_SIGN_UP": "false",
                    },
                    "ports": ["3000:3000"],
                    "networks": ["monitoring"],
                },
                "alertmanager": {
                    "image": "prom/alertmanager:latest",
                    "container_name": "alertmanager-security",
                    "volumes": ["./monitoring/alertmanager:/etc/alertmanager"],
                    "command": [
                        "--config.file=/etc/alertmanager/alertmanager-security.yml",
                        "--storage.path=/alertmanager",
                    ],
                    "ports": ["9093:9093"],
                    "networks": ["monitoring"],
                },
            },
            "networks": {"monitoring": {"driver": "bridge"}},
            "volumes": {"prometheus_data": {}, "grafana_data": {}},
        }

        with open("docker-compose.monitoring.yml", "w") as f:
            yaml.dump(docker_compose, f, default_flow_style=False)
        print("✓ Docker Compose monitoring stack deployed")

        print("\nSecurity monitoring deployment complete!")
        print("\nTo start the monitoring stack:")
        print("  docker-compose -f docker-compose.monitoring.yml up -d")

        return True


if __name__ == "__main__":
    deployer = SecurityMetricsDeployer()
    if not deployer.deploy_configs():
        sys.exit(1)
