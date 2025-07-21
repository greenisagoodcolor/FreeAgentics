# FreeAgentics Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying FreeAgentics in a production environment with enterprise-grade reliability, security, and monitoring capabilities.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Pre-Deployment Preparation](#pre-deployment-preparation)
4. [Production Deployment](#production-deployment)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
7. [Security Configuration](#security-configuration)
8. [Performance Optimization](#performance-optimization)
9. [Maintenance and Operations](#maintenance-and-operations)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Hardware:**
- CPU: 4 cores (8 recommended)
- RAM: 8GB (16GB recommended)
- Storage: 100GB SSD (500GB recommended)
- Network: 1Gbps connection

**Software Requirements:**
- Docker Engine 20.10+
- Docker Compose 2.0+
- Linux (Ubuntu 20.04 LTS or CentOS 8+)
- SSL certificates
- Domain name with DNS configured

### Required Tools

Install the following tools on your production server:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y curl wget openssl jq git

# CentOS/RHEL
sudo yum install -y curl wget openssl jq git
```

### Docker Installation

```bash
# Install Docker (Ubuntu)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## Architecture Overview

### Production Stack Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Nginx Proxy   │    │   SSL Termination│
│   (External)    │───▶│   Rate Limiting │───▶│   Security       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │                       │
        ┌─────────────────┐    ┌─────────────────┐
        │   Frontend      │    │   Backend       │
        │   (Next.js)     │    │   (FastAPI)     │
        │   - Static      │    │   - API         │
        │   - React UI    │    │   - WebSocket   │
        └─────────────────┘    └─────────────────┘
                    │                       │
                    │           ┌───────────┼───────────┐
                    │           │                       │
        ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
        │   PostgreSQL    │    │   Redis Cache   │    │   Vector Store  │
        │   - Primary DB  │    │   - Sessions    │    │   - Embeddings  │
        │   - Vector Ext  │    │   - Rate Limit  │    │   - Search      │
        └─────────────────┘    └─────────────────┘    └─────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          MONITORING STACK                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ Prometheus  │  │   Grafana   │  │ AlertManager│  │   Jaeger    │  │
│  │ - Metrics   │  │ - Dashboards│  │ - Alerts    │  │ - Tracing   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │    Loki     │  │  Promtail   │  │ Node Export │  │  cAdvisor   │  │
│  │ - Log Store │  │ - Log Ship  │  │ - Host Metr │  │ - Container │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         BACKUP & RECOVERY                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ DB Backup   │  │ File Backup │  │ Config Back │  │ S3 Storage  │  │
│  │ - Scheduled │  │ - Automated │  │ - Version   │  │ - Offsite   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Service Overview

**Core Application Services:**
- **Backend**: FastAPI application with multi-agent coordination
- **Frontend**: Next.js React application
- **Nginx**: Reverse proxy with SSL termination
- **PostgreSQL**: Primary database with vector extensions
- **Redis**: Caching and session storage

**Monitoring Services:**
- **Prometheus**: Metrics collection and storage
- **Grafana**: Dashboards and visualization
- **AlertManager**: Alert routing and management
- **Jaeger**: Distributed tracing
- **Loki**: Log aggregation
- **Promtail**: Log collection agent

**Infrastructure Services:**
- **Node Exporter**: System metrics
- **cAdvisor**: Container metrics
- **Database Exporters**: Specialized metrics
- **SSL Monitor**: Certificate monitoring

**Backup Services:**
- **PostgreSQL Backup**: Automated database backups
- **File Backup Agent**: Configuration and data backups
- **S3 Sync**: Offsite backup storage

## Pre-Deployment Preparation

### 1. Server Setup

```bash
# Create application user
sudo useradd -m -s /bin/bash freeagentics
sudo usermod -aG docker freeagentics
sudo su - freeagentics

# Create directory structure
mkdir -p /opt/freeagentics/{data,backups,logs,config}
cd /opt/freeagentics

# Clone repository
git clone <your-repository> .
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.production.template .env.production

# Generate secure secrets
./generate-production-secrets.sh

# Edit environment file
nano .env.production
```

**Critical Environment Variables:**
```bash
# Domain and SSL
DOMAIN=your-domain.com
HTTPS_ONLY=true

# Database (Use strong 32+ character passwords)
POSTGRES_PASSWORD=your-secure-database-password
DATABASE_URL=postgresql://freeagentics:your-secure-database-password@postgres:5432/freeagentics

# Application Security (Use 64+ character keys)
SECRET_KEY=your-64-character-secret-key
JWT_SECRET=your-64-character-jwt-secret

# Monitoring
GRAFANA_ADMIN_PASSWORD=your-grafana-password
GRAFANA_SECRET_KEY=your-grafana-secret-key
```

### 3. SSL Certificate Setup

**Option A: Let's Encrypt (Recommended)**
```bash
# Install certbot
sudo apt install certbot

# Obtain certificate
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/key.pem
sudo chown freeagentics:freeagentics nginx/ssl/*
```

**Option B: Custom Certificate**
```bash
# Place your certificates
cp your-certificate.pem nginx/ssl/cert.pem
cp your-private-key.pem nginx/ssl/key.pem
chmod 644 nginx/ssl/cert.pem
chmod 600 nginx/ssl/key.pem
```

### 4. Generate DH Parameters

```bash
# Generate strong DH parameters (this takes time)
openssl dhparam -out nginx/dhparam.pem 4096
```

### 5. Configure DNS

Ensure your domain points to your server:
```bash
# Check DNS resolution
nslookup your-domain.com
dig your-domain.com A
```

## Production Deployment

### 1. Pre-Deployment Validation

```bash
# Run comprehensive validation
./validate-production-deployment.sh

# Review validation report
cat validation_results/validation_report_*.md
```

### 2. Deploy Production Stack

```bash
# Run production deployment
./deploy-production-enhanced.sh

# Monitor deployment progress
./deploy-production-enhanced.sh logs
```

### 3. Verify Deployment

```bash
# Check service status
./deploy-production-enhanced.sh status

# Run health checks
./deploy-production-enhanced.sh health-check

# Test application endpoints
curl -k https://your-domain.com/health
curl -k https://your-domain.com/api/health
```

### 4. Post-Deployment Configuration

**Configure Grafana:**
1. Access Grafana: `https://your-domain.com/grafana/`
2. Login with admin credentials
3. Import pre-configured dashboards
4. Set up alert notifications

**Configure AlertManager:**
1. Update notification channels
2. Test alert routing
3. Configure escalation policies

## Monitoring and Observability

### Dashboard Access

- **Application**: `https://your-domain.com`
- **Grafana**: `https://your-domain.com/grafana/`
- **Prometheus**: `https://your-domain.com/prometheus/`
- **AlertManager**: `https://your-domain.com/alertmanager/`
- **Jaeger**: `https://your-domain.com/jaeger/`

### Key Metrics to Monitor

**Application Metrics:**
- Request rate and latency
- Error rates by endpoint
- Agent coordination performance
- Memory and CPU usage
- Database connection pool

**Infrastructure Metrics:**
- System resource utilization
- Container health and restart counts
- Network throughput
- Disk I/O and storage usage
- SSL certificate expiry

**Business Metrics:**
- Active user sessions
- API usage patterns
- Agent conversation quality
- System availability (SLA)

### Alert Configuration

**Critical Alerts:**
- Service downtime
- High error rates (>5%)
- Database connectivity issues
- Memory/CPU exhaustion (>90%)
- SSL certificate expiry (<30 days)

**Warning Alerts:**
- High response times (>2s)
- Elevated error rates (>1%)
- Resource usage (>80%)
- Failed backup jobs
- Unusual traffic patterns

## Backup and Disaster Recovery

### Automated Backups

**Database Backups:**
- Daily full backups
- Continuous WAL archiving
- 30-day retention policy
- Offsite backup to S3

**Configuration Backups:**
- Daily configuration snapshots
- Version-controlled deployments
- Infrastructure as Code backups

**Monitoring Data:**
- Grafana dashboard exports
- Prometheus configuration backup
- Alert rule backup

### Disaster Recovery Procedures

**RTO (Recovery Time Objective):** < 4 hours
**RPO (Recovery Point Objective):** < 1 hour

**Recovery Steps:**
1. Assess failure scope and impact
2. Activate disaster recovery team
3. Restore from latest backup
4. Verify data integrity
5. Update DNS if necessary
6. Validate all services
7. Monitor post-recovery

### Backup Verification

```bash
# Test database backup restoration
./scripts/backup/test-restore.sh

# Verify backup integrity
./scripts/backup/verify-backups.sh

# Test disaster recovery scenario
./scripts/backup/disaster-recovery-test.sh
```

## Security Configuration

### Network Security

**Firewall Configuration:**
```bash
# UFW configuration
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP (redirect)
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

**Network Isolation:**
- Services communicate via Docker network
- Database not exposed externally
- Monitoring services on localhost only
- Rate limiting on public endpoints

### Application Security

**Authentication & Authorization:**
- JWT-based authentication
- Role-based access control (RBAC)
- Multi-factor authentication (MFA) support
- Session management with Redis

**Data Protection:**
- Encryption at rest (database)
- Encryption in transit (TLS 1.3)
- Secure cookie settings
- CSRF protection
- Input validation and sanitization

**Security Headers:**
- HSTS with preload
- Content Security Policy (CSP)
- X-Frame-Options
- X-Content-Type-Options
- X-XSS-Protection

### Container Security

**Security Hardening:**
- Non-root containers
- Read-only containers where possible
- Minimal attack surface
- Security scanning (Snyk/Trivy)
- Resource limits and quotas

**Secret Management:**
- Environment variable encryption
- Secret rotation procedures
- Principle of least privilege
- Audit logging for secrets access

## Performance Optimization

### Application Performance

**Backend Optimization:**
- Connection pooling
- Async request handling
- Caching strategies
- Database query optimization
- Memory profiling and optimization

**Frontend Optimization:**
- Static asset optimization
- CDN integration
- Bundle size optimization
- Lazy loading
- Performance monitoring

### Infrastructure Performance

**Database Performance:**
- Index optimization
- Query performance monitoring
- Connection pool tuning
- Automated VACUUM and ANALYZE
- Read replicas for scaling

**Caching Strategy:**
- Redis for session storage
- Application-level caching
- Database query caching
- Static asset caching (CDN)
- Cache invalidation strategies

### Scaling Strategies

**Horizontal Scaling:**
- Load balancer configuration
- Service replica scaling
- Database read replicas
- Microservice decomposition
- Container orchestration (K8s ready)

**Vertical Scaling:**
- Resource limit optimization
- Performance profiling
- Bottleneck identification
- Capacity planning
- Auto-scaling triggers

## Maintenance and Operations

### Regular Maintenance Tasks

**Daily:**
- Monitor system health dashboards
- Check backup job status
- Review error logs and alerts
- Verify SSL certificate status
- Monitor resource utilization

**Weekly:**
- Update system packages
- Review security alerts
- Analyze performance trends
- Test backup restoration
- Update documentation

**Monthly:**
- Security patch updates
- Performance optimization review
- Capacity planning assessment
- Disaster recovery testing
- Security audit review

### Update Procedures

**Application Updates:**
1. Test in staging environment
2. Create deployment backup
3. Deploy using rolling update strategy
4. Run post-deployment validation
5. Monitor for issues
6. Rollback if necessary

**System Updates:**
1. Schedule maintenance window
2. Notify stakeholders
3. Apply security patches
4. Restart services as needed
5. Validate system operation
6. Document changes

### Health Monitoring

**Service Health Checks:**
- Application endpoint monitoring
- Database connectivity checks
- Cache service verification
- SSL certificate validation
- External dependency checks

**Performance Monitoring:**
- Response time tracking
- Throughput measurement
- Error rate monitoring
- Resource utilization alerts
- Capacity threshold warnings

## Troubleshooting

### Common Issues

**Service Won't Start:**
```bash
# Check service logs
docker-compose -f docker-compose.production.yml logs service-name

# Check resource constraints
docker stats

# Verify configuration
docker-compose config
```

**Database Connection Issues:**
```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Test database connectivity
docker exec freeagentics-postgres pg_isready -U freeagentics

# Check connection limits
docker exec freeagentics-postgres psql -U freeagentics -c "SELECT * FROM pg_stat_activity;"
```

**High Memory Usage:**
```bash
# Identify memory-intensive containers
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Check application memory leaks
docker exec freeagentics-backend python -c "import gc; print(len(gc.get_objects()))"

# Monitor memory trends in Grafana
```

**SSL Certificate Issues:**
```bash
# Check certificate validity
openssl x509 -in nginx/ssl/cert.pem -text -noout

# Test SSL configuration
openssl s_client -connect your-domain.com:443

# Check certificate expiry
./nginx/monitor-ssl.sh health-check
```

### Log Analysis

**Centralized Logging:**
- Use Loki for log aggregation
- Search logs in Grafana
- Set up log-based alerts
- Implement structured logging

**Log Locations:**
```bash
# Application logs
docker-compose logs freeagentics-backend
docker-compose logs freeagentics-frontend

# System logs
journalctl -u docker
tail -f /var/log/nginx/access.log

# Monitoring logs
docker-compose logs prometheus
docker-compose logs grafana
```

### Performance Troubleshooting

**Slow Response Times:**
1. Check application metrics in Grafana
2. Analyze database query performance
3. Review caching effectiveness
4. Examine network latency
5. Profile application code

**High CPU Usage:**
1. Identify CPU-intensive processes
2. Review application algorithms
3. Check for infinite loops
4. Optimize database queries
5. Consider horizontal scaling

**Memory Leaks:**
1. Monitor memory usage trends
2. Profile application memory
3. Check for unclosed connections
4. Review caching strategies
5. Implement memory limits

### Emergency Procedures

**Service Outage:**
1. Assess impact and scope
2. Check monitoring dashboards
3. Review recent changes
4. Implement quick fixes
5. Communicate status
6. Perform root cause analysis

**Security Incident:**
1. Isolate affected systems
2. Preserve evidence
3. Assess breach scope
4. Implement containment
5. Notify stakeholders
6. Recovery and remediation

**Data Loss:**
1. Stop all write operations
2. Assess data integrity
3. Identify last known good backup
4. Execute recovery procedures
5. Validate restored data
6. Resume operations gradually

## Support and Documentation

### Additional Resources

- **API Documentation**: `https://your-domain.com/docs`
- **Architecture Guide**: `/docs/ARCHITECTURE_OVERVIEW.md`
- **Security Guide**: `/docs/security/SECURITY_IMPLEMENTATION_GUIDE.md`
- **Monitoring Guide**: `/docs/monitoring/MONITORING_GUIDE.md`
- **Runbooks**: `/docs/runbooks/`

### Getting Help

**Internal Support:**
- Check documentation and runbooks
- Review monitoring dashboards
- Search issue tracker
- Consult team members

**External Support:**
- Community forums
- GitHub issues
- Professional support
- Security advisories

### Change Management

**Documentation Updates:**
- Keep deployment guide current
- Update runbooks after incidents
- Document configuration changes
- Maintain architecture diagrams

**Version Control:**
- Tag production releases
- Maintain deployment history
- Track configuration changes
- Review security updates

---

**Document Version:** 1.0.0
**Last Updated:** $(date)
**Maintained By:** FreeAgentics DevOps Team

For questions or support, please contact: devops@freeagentics.com