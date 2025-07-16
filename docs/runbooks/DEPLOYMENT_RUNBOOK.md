# Deployment Runbook

## Overview

This runbook provides step-by-step procedures for deploying FreeAgentics to production environments, including pre-deployment checks, deployment execution, and post-deployment verification.

## Table of Contents

1. [Pre-Deployment Procedures](#pre-deployment-procedures)
2. [Deployment Execution](#deployment-execution)
3. [Post-Deployment Verification](#post-deployment-verification)
4. [Rollback Procedures](#rollback-procedures)
5. [Emergency Deployment](#emergency-deployment)
6. [Troubleshooting](#troubleshooting)

## Pre-Deployment Procedures

### 1. Release Preparation

#### Code Review and Testing
```bash
# Verify all tests pass
make test-all

# Check test coverage
make coverage-check

# Run security scan
make security-scan

# Verify linting
make lint-check
```

#### Environment Setup
```bash
# Set deployment variables
export RELEASE_VERSION=$(git describe --tags --always)
export DEPLOY_ENV=production
export DEPLOY_DATE=$(date +%Y%m%d_%H%M%S)
export ROLLBACK_VERSION=$(curl -s https://api.freeagentics.io/health | jq -r '.version')

# Create deployment directory
mkdir -p /tmp/deployment_${DEPLOY_DATE}
cd /tmp/deployment_${DEPLOY_DATE}

# Clone repository
git clone https://github.com/company/freeagentics.git
cd freeagentics
git checkout ${RELEASE_VERSION}
```

### 2. Pre-Deployment Checks

#### Infrastructure Health Check
```bash
# Check all services are running
docker-compose ps

# Verify database connectivity
./scripts/deployment/check-database.sh

# Check Redis connectivity
./scripts/deployment/check-redis.sh

# Verify external dependencies
./scripts/deployment/check-external-deps.sh
```

#### Resource Availability
```bash
# Check disk space
df -h

# Check memory usage
free -h

# Check CPU usage
uptime

# Check network connectivity
./scripts/deployment/network-check.sh
```

#### Security Verification
```bash
# Check SSL certificates
./scripts/deployment/check-ssl.sh

# Verify security headers
./scripts/deployment/check-security-headers.sh

# Check firewall rules
./scripts/deployment/check-firewall.sh
```

### 3. Backup Procedures

#### Database Backup
```bash
# Create pre-deployment backup
./scripts/backup/create-backup.sh --type pre-deployment --version ${RELEASE_VERSION}

# Verify backup integrity
./scripts/backup/verify-backup.sh --backup-id latest

# Store backup metadata
echo "BACKUP_ID=$(./scripts/backup/get-latest-backup-id.sh)" >> deployment.env
```

#### Configuration Backup
```bash
# Backup current configuration
./scripts/backup/backup-config.sh --tag pre-deployment-${DEPLOY_DATE}

# Backup environment variables
./scripts/backup/backup-env.sh --tag pre-deployment-${DEPLOY_DATE}

# Backup certificates
./scripts/backup/backup-certificates.sh --tag pre-deployment-${DEPLOY_DATE}
```

### 4. Notification

#### Team Notification
```bash
# Notify deployment start
./scripts/notification/deploy-start.sh \
  --version ${RELEASE_VERSION} \
  --env ${DEPLOY_ENV} \
  --deployer $(whoami)

# Create deployment channel
./scripts/notification/create-deploy-channel.sh \
  --version ${RELEASE_VERSION}
```

#### Maintenance Mode (if required)
```bash
# Enable maintenance mode for major deployments
./scripts/maintenance/enable-maintenance.sh \
  --message "Deployment in progress" \
  --duration 30m
```

## Deployment Execution

### 1. Database Migrations

#### Check for Migrations
```bash
# Check if migrations are needed
./scripts/deployment/check-migrations.sh

# Show migration plan
./scripts/deployment/show-migration-plan.sh
```

#### Run Migrations
```bash
# Run migrations in dry-run mode first
./scripts/deployment/migrate.sh --dry-run

# Apply migrations
./scripts/deployment/migrate.sh --apply

# Verify migrations
./scripts/deployment/verify-migrations.sh
```

### 2. Application Deployment

#### Stop Services (if required)
```bash
# For maintenance deployments, stop services
./scripts/deployment/stop-services.sh --graceful

# Wait for connections to drain
./scripts/deployment/wait-for-drain.sh --timeout 30s
```

#### Deploy Backend Services
```bash
# Build new images
./scripts/deployment/build-images.sh --version ${RELEASE_VERSION}

# Deploy API service
./scripts/deployment/deploy-api.sh \
  --version ${RELEASE_VERSION} \
  --replicas 3 \
  --health-check-timeout 60s

# Deploy worker services
./scripts/deployment/deploy-workers.sh \
  --version ${RELEASE_VERSION} \
  --replicas 2
```

#### Deploy Frontend
```bash
# Build frontend assets
./scripts/deployment/build-frontend.sh --version ${RELEASE_VERSION}

# Deploy frontend
./scripts/deployment/deploy-frontend.sh \
  --version ${RELEASE_VERSION} \
  --health-check-timeout 30s
```

### 3. Configuration Updates

#### Update Configuration
```bash
# Deploy new configuration
./scripts/deployment/deploy-config.sh --version ${RELEASE_VERSION}

# Update environment variables
./scripts/deployment/update-env.sh --version ${RELEASE_VERSION}

# Reload configuration
./scripts/deployment/reload-config.sh
```

#### Service Discovery Updates
```bash
# Update service registry
./scripts/deployment/update-service-registry.sh

# Update load balancer configuration
./scripts/deployment/update-load-balancer.sh
```

### 4. Rolling Deployment

#### For Zero-Downtime Deployments
```bash
# Start rolling deployment
./scripts/deployment/rolling-deploy.sh \
  --version ${RELEASE_VERSION} \
  --batch-size 1 \
  --health-check-interval 10s

# Monitor deployment progress
./scripts/deployment/monitor-rolling-deploy.sh
```

#### Health Check Integration
```bash
# Configure health checks
./scripts/deployment/configure-health-checks.sh

# Wait for healthy status
./scripts/deployment/wait-for-healthy.sh --timeout 300s
```

## Post-Deployment Verification

### 1. Health Checks

#### System Health
```bash
# Check all services are running
docker-compose ps

# Verify service health
./scripts/deployment/verify-health.sh

# Check resource usage
./scripts/deployment/check-resources.sh
```

#### Application Health
```bash
# Test API endpoints
./scripts/deployment/test-api.sh

# Verify database connectivity
./scripts/deployment/test-database.sh

# Check Redis connectivity
./scripts/deployment/test-redis.sh
```

### 2. Functional Testing

#### Smoke Tests
```bash
# Run smoke tests
./scripts/testing/smoke-tests.sh

# Test critical user journeys
./scripts/testing/critical-path-tests.sh

# Test agent coordination
./scripts/testing/test-agent-coordination.sh
```

#### Integration Tests
```bash
# Run integration tests
./scripts/testing/integration-tests.sh

# Test external API integrations
./scripts/testing/test-external-apis.sh
```

### 3. Performance Verification

#### Performance Metrics
```bash
# Check response times
./scripts/monitoring/check-response-times.sh

# Monitor error rates
./scripts/monitoring/check-error-rates.sh

# Verify throughput
./scripts/monitoring/check-throughput.sh
```

#### Load Testing
```bash
# Run basic load test
./scripts/testing/basic-load-test.sh

# Monitor system under load
./scripts/monitoring/monitor-load.sh --duration 10m
```

### 4. Security Verification

#### Security Checks
```bash
# Verify SSL/TLS configuration
./scripts/security/verify-ssl.sh

# Check security headers
./scripts/security/check-headers.sh

# Test authentication
./scripts/security/test-auth.sh
```

#### Vulnerability Scan
```bash
# Run vulnerability scan
./scripts/security/vulnerability-scan.sh

# Check for exposed endpoints
./scripts/security/endpoint-scan.sh
```

### 5. Monitoring Setup

#### Update Monitoring
```bash
# Update monitoring configuration
./scripts/monitoring/update-monitoring.sh --version ${RELEASE_VERSION}

# Restart monitoring services
./scripts/monitoring/restart-monitoring.sh

# Verify monitoring is working
./scripts/monitoring/verify-monitoring.sh
```

#### Alert Configuration
```bash
# Update alert rules
./scripts/monitoring/update-alerts.sh

# Test alert system
./scripts/monitoring/test-alerts.sh
```

## Rollback Procedures

### 1. Immediate Rollback

#### Quick Rollback
```bash
# Rollback to previous version
./scripts/deployment/rollback.sh --to-version ${ROLLBACK_VERSION}

# Verify rollback
./scripts/deployment/verify-rollback.sh
```

#### Database Rollback
```bash
# Rollback database if needed
./scripts/deployment/rollback-database.sh --to-backup ${BACKUP_ID}

# Verify data integrity
./scripts/deployment/verify-data-integrity.sh
```

### 2. Partial Rollback

#### Service-Specific Rollback
```bash
# Rollback specific service
./scripts/deployment/rollback-service.sh --service api --to-version ${ROLLBACK_VERSION}

# Rollback configuration only
./scripts/deployment/rollback-config.sh --to-version ${ROLLBACK_VERSION}
```

### 3. Rollback Verification

#### Post-Rollback Checks
```bash
# Verify system health after rollback
./scripts/deployment/verify-health.sh

# Run smoke tests
./scripts/testing/smoke-tests.sh

# Check monitoring
./scripts/monitoring/verify-monitoring.sh
```

## Emergency Deployment

### 1. Emergency Procedures

#### Hotfix Deployment
```bash
# Emergency deployment process
./scripts/deployment/emergency-deploy.sh \
  --version ${HOTFIX_VERSION} \
  --skip-migrations \
  --fast-deploy

# Minimal verification
./scripts/deployment/emergency-verify.sh
```

#### Security Patch Deployment
```bash
# Security patch deployment
./scripts/deployment/security-patch-deploy.sh \
  --patch-id ${SECURITY_PATCH_ID} \
  --priority critical
```

### 2. Post-Emergency Procedures

#### Full Verification
```bash
# Complete post-emergency verification
./scripts/deployment/post-emergency-verify.sh

# Schedule full testing
./scripts/testing/schedule-full-test.sh
```

## Troubleshooting

### Common Issues

#### Deployment Failures

**Issue**: Service fails to start
```bash
# Check logs
docker-compose logs service-name

# Check configuration
./scripts/deployment/validate-config.sh

# Check dependencies
./scripts/deployment/check-dependencies.sh
```

**Issue**: Database migration fails
```bash
# Check migration logs
./scripts/deployment/check-migration-logs.sh

# Rollback migration
./scripts/deployment/rollback-migration.sh

# Fix and retry
./scripts/deployment/retry-migration.sh
```

#### Health Check Failures

**Issue**: Health check timeouts
```bash
# Increase timeout
./scripts/deployment/extend-timeout.sh --service api --timeout 120s

# Check service logs
./scripts/deployment/check-service-logs.sh --service api

# Manual health check
curl -f http://localhost:8000/health
```

#### Performance Issues

**Issue**: High response times after deployment
```bash
# Check resource usage
./scripts/monitoring/check-resources.sh

# Analyze bottlenecks
./scripts/monitoring/analyze-bottlenecks.sh

# Consider rollback
./scripts/deployment/consider-rollback.sh
```

### Debugging Commands

#### Service Debugging
```bash
# Check service status
docker-compose ps

# View service logs
docker-compose logs --tail=100 service-name

# Execute shell in container
docker-compose exec service-name /bin/bash

# Check service configuration
./scripts/deployment/check-service-config.sh --service service-name
```

#### Database Debugging
```bash
# Check database connections
./scripts/deployment/check-db-connections.sh

# View database logs
docker-compose logs postgres

# Check migration status
./scripts/deployment/check-migration-status.sh
```

#### Network Debugging
```bash
# Check network connectivity
./scripts/deployment/network-debug.sh

# Test service-to-service communication
./scripts/deployment/test-service-communication.sh

# Check load balancer
./scripts/deployment/check-load-balancer.sh
```

## Best Practices

### 1. Deployment Planning

#### Pre-Deployment Planning
- Always plan deployments during low-traffic periods
- Have rollback plan ready before deployment
- Coordinate with team members
- Prepare communication templates

#### Testing Strategy
- Test in staging environment first
- Run comprehensive test suite
- Perform load testing
- Verify security measures

### 2. Risk Mitigation

#### Backup Strategy
- Always create backups before deployment
- Verify backup integrity
- Test restore procedures
- Document backup locations

#### Monitoring Strategy
- Monitor key metrics during deployment
- Set up alerts for deployment
- Have team member monitoring during deployment
- Prepare for quick response

### 3. Communication

#### Team Communication
- Notify team before deployment
- Provide deployment timeline
- Share progress updates
- Document any issues

#### Stakeholder Communication
- Inform stakeholders of planned deployments
- Communicate any expected downtime
- Provide status updates
- Report completion and any issues

## Deployment Checklist

### Pre-Deployment
- [ ] Code review completed
- [ ] All tests passing
- [ ] Security scan clean
- [ ] Backup created and verified
- [ ] Team notified
- [ ] Rollback plan ready
- [ ] Deployment environment prepared

### During Deployment
- [ ] Migrations completed successfully
- [ ] Services deployed and healthy
- [ ] Configuration updated
- [ ] Health checks passing
- [ ] Monitoring operational
- [ ] Team updated on progress

### Post-Deployment
- [ ] Smoke tests passing
- [ ] Performance metrics normal
- [ ] Security checks passed
- [ ] Monitoring configured
- [ ] Team notified of completion
- [ ] Documentation updated
- [ ] Rollback plan updated

### Emergency Procedures
- [ ] Immediate rollback procedure tested
- [ ] Emergency contacts available
- [ ] Escalation procedures clear
- [ ] Communication templates ready

---

**Document Information:**
- **Version**: 1.0
- **Last Updated**: January 2024
- **Next Review**: April 2024
- **Owner**: DevOps Team
- **Approved By**: Engineering Lead