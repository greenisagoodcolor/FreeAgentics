# FreeAgentics Production Operations Guide

## Overview

This comprehensive guide provides production operations procedures for the FreeAgentics system. It serves as the single source of truth for deployment, monitoring, incident response, and maintenance procedures.

## Table of Contents

1. [Production Overview](#production-overview)
2. [Deployment Procedures](#deployment-procedures)
3. [Monitoring and Alerting](#monitoring-and-alerting)
4. [Incident Response](#incident-response)
5. [Troubleshooting](#troubleshooting)
6. [Backup and Recovery](#backup-and-recovery)
7. [Scaling and Performance](#scaling-and-performance)
8. [Security Operations](#security-operations)
9. [Maintenance Procedures](#maintenance-procedures)
10. [Emergency Procedures](#emergency-procedures)

## Production Overview

### System Architecture

FreeAgentics is a distributed system consisting of:

- **API Service**: FastAPI-based backend
- **Web Frontend**: Next.js application
- **Database**: PostgreSQL with Redis caching
- **Message Queue**: Redis for async processing
- **Monitoring**: Prometheus, Grafana, and custom metrics
- **Load Balancer**: NGINX with SSL termination

### Production Environments

| Environment | Purpose | URL | Database |
|-------------|---------|-----|----------|
| Production | Live system | https://api.freeagentics.io | prod-db |
| Staging | Pre-production testing | https://staging.freeagentics.io | staging-db |
| Development | Active development | https://dev.freeagentics.io | dev-db |

### Infrastructure Components

#### Core Services
- **API Servers**: 3 replicas behind load balancer
- **Database**: PostgreSQL master with read replicas
- **Cache**: Redis cluster for session and data caching
- **File Storage**: S3-compatible storage for artifacts

#### Monitoring Stack
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **ELK Stack**: Log aggregation and analysis
- **Jaeger**: Distributed tracing

#### Security Components
- **WAF**: Web Application Firewall
- **SSL/TLS**: Let's Encrypt certificates
- **VPN**: Secure access to production environment
- **Secrets Management**: HashiCorp Vault

## Deployment Procedures

### Pre-Deployment Checklist

Before any production deployment:

- [ ] **Code Review**: All changes peer-reviewed
- [ ] **Testing**: All tests passing in CI/CD
- [ ] **Security Scan**: No critical vulnerabilities
- [ ] **Performance Test**: Load testing completed
- [ ] **Documentation**: Updated relevant documentation
- [ ] **Rollback Plan**: Prepared and tested
- [ ] **Team Notification**: Stakeholders informed
- [ ] **Monitoring**: Health checks ready

### Deployment Process

#### 1. Prepare Deployment

```bash
# Set deployment variables
export RELEASE_VERSION=$(git describe --tags --always)
export DEPLOY_ENV=production
export ROLLBACK_VERSION=$(curl -s https://api.freeagentics.io/health | jq -r '.version')

# Pre-deployment checks
./scripts/deployment/pre-deployment-checks.sh

# Create deployment branch
git checkout -b deploy/${RELEASE_VERSION}
```

#### 2. Database Migrations

```bash
# Check for pending migrations
./scripts/deployment/check-migrations.sh

# Run migrations in staging first
./scripts/deployment/migrate-database.sh --env staging --dry-run

# Apply migrations to production
./scripts/deployment/migrate-database.sh --env production
```

#### 3. Deploy Application

```bash
# Deploy API service
./scripts/deployment/deploy-api.sh --version ${RELEASE_VERSION}

# Deploy frontend
./scripts/deployment/deploy-frontend.sh --version ${RELEASE_VERSION}

# Verify deployment
./scripts/deployment/verify-deployment.sh
```

#### 4. Post-Deployment Verification

```bash
# Health check
curl -f https://api.freeagentics.io/health

# Smoke tests
./scripts/deployment/smoke-tests.sh

# Monitor for 15 minutes
./scripts/deployment/post-deployment-monitor.sh --duration 15m
```

### Blue-Green Deployment

For zero-downtime deployments:

```bash
# 1. Deploy to green environment
./scripts/deployment/deploy-blue-green.sh --target green --version ${RELEASE_VERSION}

# 2. Verify green environment
./scripts/deployment/verify-environment.sh --env green

# 3. Switch traffic
./scripts/deployment/switch-traffic.sh --from blue --to green

# 4. Monitor and rollback if needed
./scripts/deployment/monitor-traffic-switch.sh
```

### Rollback Procedures

#### Immediate Rollback

```bash
# Quick rollback to previous version
./scripts/deployment/rollback.sh --to ${ROLLBACK_VERSION}

# Verify rollback
./scripts/deployment/verify-rollback.sh
```

#### Database Rollback

```bash
# Rollback database migrations (if needed)
./scripts/deployment/rollback-migrations.sh --to-version ${ROLLBACK_VERSION}

# Verify data integrity
./scripts/deployment/verify-data-integrity.sh
```

## Monitoring and Alerting

### Key Metrics

#### System Metrics
- **CPU Usage**: Target < 70%
- **Memory Usage**: Target < 80%
- **Disk Usage**: Target < 85%
- **Network I/O**: Monitor for anomalies

#### Application Metrics
- **API Response Time**: Target < 500ms (95th percentile)
- **Error Rate**: Target < 0.1%
- **Throughput**: Requests per second
- **Database Connections**: Monitor pool utilization

#### Business Metrics
- **Agent Coordination Success Rate**: Target > 99%
- **Active Users**: Track user engagement
- **Feature Usage**: Monitor adoption rates

### Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|---------|
| CPU Usage | 70% | 85% | Scale up |
| Memory Usage | 80% | 90% | Investigate/Scale |
| Error Rate | 0.1% | 1% | Immediate response |
| Response Time | 500ms | 1000ms | Performance review |
| Disk Usage | 85% | 95% | Add storage |

### Grafana Dashboards

#### System Overview Dashboard
- **URL**: http://monitoring.freeagentics.io/d/system-overview
- **Metrics**: CPU, memory, disk, network
- **Alerts**: Integrated with PagerDuty

#### Application Performance Dashboard
- **URL**: http://monitoring.freeagentics.io/d/app-performance
- **Metrics**: API metrics, error rates, response times
- **Alerts**: Slack integration for team notifications

#### Agent Coordination Dashboard
- **URL**: http://monitoring.freeagentics.io/d/agent-coordination
- **Metrics**: Agent success rates, coordination metrics
- **Alerts**: Critical for system functionality

### Alert Management

#### PagerDuty Integration

```yaml
# Alert routing configuration
alerts:
  - name: critical-system-alert
    conditions:
      - metric: cpu_usage
        threshold: 85
        duration: 5m
    actions:
      - type: pagerduty
        service: production-critical

  - name: application-error-alert
    conditions:
      - metric: error_rate
        threshold: 1
        duration: 2m
    actions:
      - type: slack
        channel: "#alerts"
      - type: pagerduty
        service: production-major
```

#### Slack Integration

```bash
# Configure Slack webhook
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

# Test alert
./scripts/monitoring/test-alert.sh --type slack --severity warning
```

## Incident Response

### Incident Classification

#### Severity Levels
- **P0 (Critical)**: System down, data loss, security breach
- **P1 (High)**: Major functionality impaired
- **P2 (Medium)**: Minor functionality affected
- **P3 (Low)**: Cosmetic issues, documentation

#### Response Times
- **P0**: 15 minutes
- **P1**: 30 minutes
- **P2**: 2 hours
- **P3**: Next business day

### Incident Response Process

#### 1. Detection and Alerting
```bash
# Automated detection via monitoring
# Manual reporting via:
./scripts/incident/report-incident.sh --severity P1 --description "API response time > 5s"
```

#### 2. Initial Response
```bash
# Acknowledge incident
./scripts/incident/acknowledge.sh --incident-id INC-2024-001

# Assess severity
./scripts/incident/assess-severity.sh --incident-id INC-2024-001

# Create incident channel
./scripts/incident/create-channel.sh --incident-id INC-2024-001
```

#### 3. Investigation
```bash
# Gather system information
./scripts/incident/gather-info.sh --incident-id INC-2024-001

# Check recent deployments
./scripts/incident/check-deployments.sh --hours 24

# Analyze logs
./scripts/incident/analyze-logs.sh --timeframe 1h
```

#### 4. Mitigation
```bash
# Apply immediate fixes
./scripts/incident/apply-hotfix.sh --fix-id FIX-001

# Scale resources if needed
./scripts/incident/emergency-scale.sh --service api --replicas 5

# Implement circuit breaker
./scripts/incident/enable-circuit-breaker.sh --service external-api
```

### Communication Templates

#### Internal Alert
```
🚨 INCIDENT ALERT - P1
Issue: API response time > 5s
Impact: 30% of users experiencing slow responses
Status: Investigating
IC: @oncall-engineer
Channel: #incident-2024-001
Next Update: 30 minutes
```

#### Customer Communication
```
We are currently experiencing performance issues with our API service.
Impact: Some users may experience slower response times
Status: Our team is actively working on resolution
Workaround: Please try again in a few minutes
Next Update: Within 1 hour
```

## Troubleshooting

### Common Issues and Solutions

#### High Memory Usage
```bash
# Check memory usage
./scripts/troubleshooting/check-memory.sh

# Identify memory leaks
./scripts/troubleshooting/memory-analysis.sh

# Restart high-memory services
./scripts/troubleshooting/restart-service.sh --service api
```

#### Database Connection Issues
```bash
# Check connection pool
./scripts/troubleshooting/check-db-connections.sh

# Increase connection limit
./scripts/troubleshooting/increase-db-connections.sh

# Restart database if needed
./scripts/troubleshooting/restart-database.sh
```

#### Agent Coordination Failures
```bash
# Check agent status
./scripts/troubleshooting/check-agents.sh

# Restart coordination service
./scripts/troubleshooting/restart-coordination.sh

# Review coordination logs
./scripts/troubleshooting/analyze-coordination-logs.sh
```

### Diagnostic Commands

#### System Health
```bash
# Overall system health
make health-check

# Service status
docker-compose ps

# Resource usage
docker stats --no-stream

# Network connectivity
./scripts/troubleshooting/network-check.sh
```

#### Application Diagnostics
```bash
# API health
curl -f https://api.freeagentics.io/health

# Database connectivity
./scripts/troubleshooting/test-db-connection.sh

# Cache status
./scripts/troubleshooting/check-redis.sh

# Queue processing
./scripts/troubleshooting/check-queue.sh
```

### Log Analysis

#### Centralized Logging
```bash
# Search across all services
./scripts/troubleshooting/search-logs.sh --query "error" --timeframe 1h

# Filter by service
./scripts/troubleshooting/search-logs.sh --service api --level error

# Export logs for analysis
./scripts/troubleshooting/export-logs.sh --incident-id INC-2024-001
```

#### Performance Analysis
```bash
# Identify slow queries
./scripts/troubleshooting/slow-queries.sh

# Analyze response times
./scripts/troubleshooting/response-time-analysis.sh

# Check for memory leaks
./scripts/troubleshooting/memory-leak-detection.sh
```

## Backup and Recovery

### Backup Strategy

#### Database Backups
```bash
# Daily full backup
./scripts/backup/full-backup.sh --schedule daily

# Hourly incremental backup
./scripts/backup/incremental-backup.sh --schedule hourly

# Verify backup integrity
./scripts/backup/verify-backup.sh --backup-id latest
```

#### Application Backups
```bash
# Configuration backup
./scripts/backup/backup-config.sh

# Code repository backup
./scripts/backup/backup-repository.sh

# Certificate backup
./scripts/backup/backup-certificates.sh
```

### Recovery Procedures

#### Database Recovery
```bash
# Point-in-time recovery
./scripts/recovery/restore-database.sh --timestamp "2024-01-15 14:30:00"

# Full database restore
./scripts/recovery/restore-full-backup.sh --backup-id BACKUP-2024-01-15

# Verify data integrity
./scripts/recovery/verify-data-integrity.sh
```

#### Application Recovery
```bash
# Restore from backup
./scripts/recovery/restore-application.sh --backup-id latest

# Reconfigure services
./scripts/recovery/reconfigure-services.sh

# Verify functionality
./scripts/recovery/verify-recovery.sh
```

### Disaster Recovery

#### Recovery Time Objectives (RTO)
- **Database**: 30 minutes
- **Application**: 15 minutes
- **Full System**: 1 hour

#### Recovery Point Objectives (RPO)
- **Database**: 15 minutes
- **Configuration**: 1 hour
- **Code**: Real-time (Git)

#### DR Procedures
```bash
# Activate DR site
./scripts/disaster-recovery/activate-dr.sh

# Switch DNS to DR
./scripts/disaster-recovery/switch-dns.sh

# Verify DR functionality
./scripts/disaster-recovery/verify-dr.sh
```

## Scaling and Performance

### Auto-Scaling Configuration

#### Horizontal Scaling
```yaml
# Kubernetes HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Vertical Scaling
```bash
# Increase resource limits
./scripts/scaling/vertical-scale.sh --service api --cpu 2 --memory 4Gi

# Monitor resource usage
./scripts/scaling/monitor-resources.sh --service api
```

### Performance Optimization

#### Database Performance
```bash
# Analyze slow queries
./scripts/performance/analyze-slow-queries.sh

# Optimize indexes
./scripts/performance/optimize-indexes.sh

# Update table statistics
./scripts/performance/update-statistics.sh
```

#### Application Performance
```bash
# Profile application
./scripts/performance/profile-application.sh

# Optimize memory usage
./scripts/performance/optimize-memory.sh

# Analyze bottlenecks
./scripts/performance/analyze-bottlenecks.sh
```

### Load Testing

#### Performance Testing
```bash
# Load test API
./scripts/performance/load-test-api.sh --concurrent 100 --duration 10m

# Stress test database
./scripts/performance/stress-test-db.sh --connections 500

# Test agent coordination
./scripts/performance/test-agent-coordination.sh --agents 1000
```

#### Capacity Planning
```bash
# Analyze usage trends
./scripts/capacity/analyze-trends.sh --period 30d

# Predict capacity needs
./scripts/capacity/predict-capacity.sh --horizon 3m

# Generate capacity report
./scripts/capacity/capacity-report.sh
```

## Security Operations

### Security Monitoring

#### Security Metrics
- **Failed Authentication Attempts**: Monitor for brute force
- **Privilege Escalation**: Track permission changes
- **Data Access Patterns**: Detect anomalies
- **Network Traffic**: Monitor for suspicious activity

#### Security Alerts
```bash
# Configure security alerts
./scripts/security/configure-alerts.sh

# Test security monitoring
./scripts/security/test-monitoring.sh

# Generate security report
./scripts/security/security-report.sh
```

### Incident Response (Security)

#### Security Incident Types
- **Data Breach**: Unauthorized access to sensitive data
- **Account Compromise**: User account takeover
- **Malware Detection**: Malicious software identified
- **DDoS Attack**: Distributed denial of service

#### Response Procedures
```bash
# Isolate affected systems
./scripts/security/isolate-system.sh --host compromised-host

# Collect evidence
./scripts/security/collect-evidence.sh --incident-id SEC-2024-001

# Notify authorities (if required)
./scripts/security/notify-authorities.sh --incident-id SEC-2024-001
```

### Security Hardening

#### Regular Security Tasks
```bash
# Update security patches
./scripts/security/update-patches.sh

# Rotate credentials
./scripts/security/rotate-credentials.sh

# Review access logs
./scripts/security/review-access-logs.sh

# Scan for vulnerabilities
./scripts/security/vulnerability-scan.sh
```

## Emergency Procedures

### Emergency Contacts

| Role | Primary | Secondary | Escalation |
|------|---------|-----------|------------|
| On-Call Engineer | [Phone] | [Phone] | Team Lead |
| Database Admin | [Phone] | [Phone] | Senior DBA |
| Security Team | [Phone] | [Phone] | CISO |
| Management | [Phone] | [Phone] | CEO |

### Emergency Commands

#### System Shutdown
```bash
# Graceful shutdown
./scripts/emergency/graceful-shutdown.sh

# Emergency stop
./scripts/emergency/emergency-stop.sh

# Isolate system
./scripts/emergency/isolate-system.sh
```

#### Data Protection
```bash
# Emergency backup
./scripts/emergency/emergency-backup.sh

# Lock down access
./scripts/emergency/lockdown-access.sh

# Enable read-only mode
./scripts/emergency/enable-readonly.sh
```

### Crisis Communication

#### Communication Tree
1. **Incident Commander** → Team Lead → VP Engineering
2. **Technical Team** → Subject Matter Experts
3. **Customer Success** → Key customers
4. **Legal Team** → Compliance officers (if data involved)

#### Communication Templates
```
EMERGENCY ALERT - IMMEDIATE ACTION REQUIRED
Issue: [Brief description]
Impact: [Scope and severity]
Action Required: [What needs to be done]
Timeline: [Urgency level]
Contact: [Emergency contact]
```

## Standard Operating Procedures

### Daily Operations

#### Morning Checklist
- [ ] Check system health dashboards
- [ ] Review overnight alerts
- [ ] Verify backup completion
- [ ] Check security logs
- [ ] Review performance metrics

#### Evening Checklist
- [ ] Review daily metrics
- [ ] Check error logs
- [ ] Verify scheduled tasks
- [ ] Update incident reports
- [ ] Prepare for next day

### Weekly Operations

#### Weekly Tasks
- [ ] Review capacity utilization
- [ ] Analyze performance trends
- [ ] Update security patches
- [ ] Review incident reports
- [ ] Conduct team sync

### Monthly Operations

#### Monthly Tasks
- [ ] Full system health review
- [ ] Capacity planning update
- [ ] Security audit
- [ ] Disaster recovery test
- [ ] Documentation updates

## Documentation and Training

### Documentation Standards

#### Operations Documentation
- **Runbooks**: Step-by-step procedures
- **Troubleshooting Guides**: Common issues and solutions
- **Architecture Diagrams**: System design documentation
- **Process Documentation**: Procedures and workflows

#### Training Requirements

#### New Team Member Onboarding
- [ ] Review system architecture
- [ ] Complete runbook walkthrough
- [ ] Shadow incident response
- [ ] Practice emergency procedures

#### Ongoing Training
- [ ] Monthly incident response drill
- [ ] Quarterly disaster recovery test
- [ ] Annual security training
- [ ] Tool-specific training as needed

## Appendices

### A. Emergency Contact Information
[Contact details for key personnel]

### B. System Dependencies
[List of external dependencies and their contacts]

### C. Compliance Requirements
[Regulatory and compliance obligations]

### D. Vendor Contacts
[Third-party service providers]

### E. Change Management Process
[Procedures for managing system changes]

---

**Document Information:**
- **Version**: 1.0
- **Last Updated**: January 2024
- **Next Review**: April 2024
- **Owner**: Operations Team
- **Reviewers**: Engineering Team, Security Team, Management

**Change Log:**
- v1.0 - Initial version covering all production operations procedures
