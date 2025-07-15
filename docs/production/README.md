# FreeAgentics Production Documentation

This directory contains comprehensive documentation for deploying and managing FreeAgentics in production environments.

## Documentation Overview

### Core Documentation
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Complete deployment procedures and workflows
- **[INFRASTRUCTURE_REQUIREMENTS.md](INFRASTRUCTURE_REQUIREMENTS.md)** - Hardware, network, and infrastructure specifications

### Quick Start
For a rapid production deployment:
1. Review infrastructure requirements
2. Follow pre-deployment checklist
3. Run deployment script: `./scripts/deployment/deploy-production.sh`
4. Verify deployment: `./scripts/deployment/verify-deployment.sh`

## Deployment Scripts

All deployment scripts are located in `/scripts/deployment/`:

### Primary Scripts
- **`deploy-production.sh`** - Main deployment orchestration script
- **`health-check.sh`** - Comprehensive health monitoring
- **`rollback.sh`** - Emergency rollback procedures
- **`verify-deployment.sh`** - Post-deployment verification

### Supporting Scripts
- **`migrate-database.sh`** - Database migration management
- **`backup-database.sh`** - Database backup and restore
- **`smoke-tests.sh`** - Basic functionality testing

## Deployment Strategies

### Blue-Green Deployment (Recommended)
```bash
export DEPLOYMENT_STRATEGY=blue-green
./scripts/deployment/deploy-production.sh
```

### Rolling Update
```bash
export DEPLOYMENT_STRATEGY=rolling
./scripts/deployment/deploy-production.sh
```

### Canary Deployment
```bash
export DEPLOYMENT_STRATEGY=canary
./scripts/deployment/deploy-production.sh
```

## Quick Reference

### Emergency Procedures
```bash
# Emergency rollback
./scripts/deployment/rollback.sh --force

# Emergency rollback with database
./scripts/deployment/rollback.sh --rollback-db --force

# Health check
./scripts/deployment/health-check.sh --quick
```

### Database Operations
```bash
# Create backup
./scripts/deployment/backup-database.sh production backup

# Run migrations
./scripts/deployment/migrate-database.sh production migrate

# Rollback migration
./scripts/deployment/migrate-database.sh production rollback 1
```

### Monitoring and Verification
```bash
# Full deployment verification
./scripts/deployment/verify-deployment.sh

# Run smoke tests
./scripts/deployment/smoke-tests.sh

# Check system health
curl -s https://api.freeagentics.com/health | jq '.'
```

## Environment Configuration

### Required Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@host:5432/db
DATABASE_POOL_SIZE=20

# Redis
REDIS_URL=redis://host:6379/0

# Security
JWT_SECRET_KEY=<secure-key>
API_KEY_SALT=<secure-salt>

# SSL/TLS
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem
FORCE_HTTPS=true
```

### Optional Configuration
```bash
# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
SENTRY_DSN=<sentry-dsn>

# Notifications
SLACK_WEBHOOK=<webhook-url>
ALERT_EMAIL=alerts@company.com

# Performance
RATE_LIMIT_ENABLED=true
RATE_LIMIT_DEFAULT=100/minute
```

## Infrastructure Requirements Summary

### Minimum Production Setup
- **Application Servers**: 2x (4 vCPU, 16GB RAM)
- **Database Server**: 1x (8 vCPU, 32GB RAM, 500GB SSD)
- **Redis Server**: 1x (4 vCPU, 16GB RAM)
- **Load Balancer**: 2x (2 vCPU, 4GB RAM)

### Recommended Production Setup
- **Application Cluster**: 3-10 instances (auto-scaling)
- **Database Cluster**: Primary + 2 read replicas
- **Redis Cluster**: 3 shards with replication
- **Multi-AZ deployment** for high availability

## Security Checklist

### Pre-Deployment Security
- [ ] SSL certificates installed and valid
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Database encrypted at rest
- [ ] Secrets properly managed
- [ ] Network security groups configured
- [ ] Firewall rules applied

### Post-Deployment Security
- [ ] Security scan completed
- [ ] Penetration testing performed
- [ ] Monitoring and alerting active
- [ ] Backup encryption verified
- [ ] Access logs reviewed

## Monitoring and Alerting

### Key Metrics to Monitor
- **API Response Time**: < 500ms (p95)
- **Error Rate**: < 1%
- **Database Performance**: Query time < 100ms
- **Memory Usage**: < 80%
- **CPU Usage**: < 70%
- **Disk Usage**: < 80%

### Critical Alerts
- API service down
- Database connection failure
- High error rate (>5%)
- Memory usage >90%
- Disk space >90%
- SSL certificate expiring

## Backup and Recovery

### Backup Schedule
- **Daily**: Full database backup (retained 7 days)
- **Weekly**: Full system backup (retained 4 weeks)
- **Monthly**: Archive backup (retained 12 months)

### Recovery Procedures
- **RTO (Recovery Time Objective)**: 1 hour
- **RPO (Recovery Point Objective)**: 15 minutes
- **Automated failover**: Database read replicas
- **Manual failover**: Secondary region

## Troubleshooting

### Common Issues
1. **Container won't start**: Check logs, verify environment variables
2. **Database connection failed**: Verify credentials, network connectivity
3. **High response times**: Check resource usage, database performance
4. **SSL certificate errors**: Verify certificate validity, renewal process

### Debug Commands
```bash
# Check container logs
docker logs freeagentics-api

# Check database connectivity
docker exec freeagentics-api python -c "from database.session import engine; engine.execute('SELECT 1')"

# Check Redis connectivity
docker exec freeagentics-redis redis-cli ping

# Check system resources
docker stats

# Check network connectivity
curl -v https://api.freeagentics.com/health
```

## Maintenance Schedule

### Regular Maintenance
- **Daily**: Health checks, backup verification
- **Weekly**: Security updates, performance review
- **Monthly**: Dependency updates, capacity planning
- **Quarterly**: Security audit, disaster recovery testing

### Maintenance Windows
- **Scheduled**: First Sunday of each month, 02:00-06:00 UTC
- **Emergency**: As needed with stakeholder notification
- **Major Updates**: Coordinated with business requirements

## Contact Information

### On-Call Rotation
- **Primary**: DevOps Team
- **Secondary**: Development Team
- **Escalation**: Engineering Manager

### Communication Channels
- **Slack**: #production-alerts
- **Email**: production-alerts@company.com
- **PagerDuty**: Production support rotation

## Additional Resources

### External Documentation
- [Docker Documentation](https://docs.docker.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Redis Documentation](https://redis.io/documentation)
- [Nginx Documentation](https://nginx.org/en/docs/)

### Internal Resources
- [Development Guide](../development/README.md)
- [API Documentation](../api/README.md)
- [Security Guidelines](../security/README.md)
- [Monitoring Runbooks](../runbooks/README.md)

---

**Note**: This documentation is maintained by the DevOps team. For updates or questions, please contact the production support team or create an issue in the project repository.