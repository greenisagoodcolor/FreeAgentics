# FreeAgentics Docker Deployment Guide

## 🚀 Production Deployment

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum, 8GB recommended
- 20GB disk space for data persistence

### Quick Deployment

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd freeagentics
   cp .env.example .env
   ```

2. **Configure Environment**
   Edit `.env` and set secure passwords:
   ```bash
   POSTGRES_PASSWORD=your_secure_postgres_password
   REDIS_PASSWORD=your_secure_redis_password
   ```

3. **Deploy**
   ```bash
   make docker
   ```

4. **Access Application**
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - API Health: http://localhost:8000/api/health

### Advanced Configuration

#### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_PASSWORD` | PostgreSQL password | changeme |
| `REDIS_PASSWORD` | Redis password | (empty) |
| `LOG_LEVEL` | API log level | INFO |
| `NEXT_PUBLIC_API_URL` | Frontend API URL | http://localhost:8000 |

#### Resource Limits

Default resource allocation per service:

- **PostgreSQL**: 512MB RAM, 0.5 CPU
- **Redis**: 256MB RAM, 0.25 CPU  
- **API Backend**: 2GB RAM, 1.0 CPU
- **Web Frontend**: 1GB RAM, 0.5 CPU

#### Data Persistence

Data is persisted in:
- `./infrastructure/docker/data/postgres/` - PostgreSQL data
- `./infrastructure/docker/data/redis/` - Redis data

### Security Features

#### Container Security
- ✅ Non-root users in all containers
- ✅ Read-only root filesystems where possible
- ✅ No new privileges flag
- ✅ Resource limits enforced
- ✅ Health checks for all services

#### Network Security
- ✅ Isolated Docker network
- ✅ Service-to-service communication only
- ✅ Minimal exposed ports

#### Database Security
- ✅ SCRAM-SHA-256 authentication
- ✅ Password-protected Redis
- ✅ Persistent data encryption

### Monitoring and Health Checks

All services include comprehensive health checks:

- **PostgreSQL**: Connection test every 30s
- **Redis**: Ping test every 30s
- **API**: HTTP health endpoint every 30s
- **Web**: HTTP availability every 30s

### Troubleshooting

#### Common Issues

1. **Port conflicts**
   ```bash
   make kill-ports  # Kill existing services
   ```

2. **Permission issues with data directories**
   ```bash
   sudo chown -R 1001:1001 infrastructure/docker/data/
   ```

3. **View logs**
   ```bash
   docker-compose -f infrastructure/docker/docker-compose.yml logs -f [service]
   ```

4. **Reset everything**
   ```bash
   make docker-down
   docker system prune -f
   make docker
   ```

#### Service Status

Check service status:
```bash
docker-compose -f infrastructure/docker/docker-compose.yml ps
```

#### Performance Tuning

For production environments:

1. **Increase PostgreSQL memory**:
   ```yaml
   # In docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 1G  # Increase from 512M
   ```

2. **Enable API clustering**:
   ```yaml
   # In docker-compose.yml
   command: ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
   ```

3. **Add external load balancer** for high availability

### Production Checklist

- [ ] Configure secure passwords in `.env`
- [ ] Set up external SSL termination (nginx/traefik)
- [ ] Configure backup strategy for PostgreSQL data
- [ ] Set up log aggregation (ELK/Loki)
- [ ] Configure monitoring (Prometheus/Grafana)
- [ ] Review resource limits for your workload
- [ ] Test disaster recovery procedures

### Scaling

#### Horizontal Scaling

1. **API Backend**: Increase `--workers` in uvicorn command
2. **Database**: Consider PostgreSQL read replicas
3. **Frontend**: Deploy multiple web instances behind load balancer

#### Vertical Scaling

Adjust resource limits in `docker-compose.yml` based on monitoring data.

### Backup and Recovery

#### Database Backup
```bash
docker exec freeagentics-postgres pg_dump -U freeagentics freeagentics > backup.sql
```

#### Full System Backup
```bash
# Stop services
make docker-down

# Backup data directories
tar -czf freeagentics-backup-$(date +%Y%m%d).tar.gz infrastructure/docker/data/

# Restart services
make docker
```

### Development vs Production

This configuration is optimized for production. For development:

1. Use `make dev` instead of `make docker`
2. Enable hot-reload and development tools
3. Use local databases instead of containers
4. Skip resource limits and security hardening

---

## 🆘 Support

For issues and support:
1. Check logs: `docker-compose logs -f`
2. Run validation: `infrastructure/docker/validate-docker.sh`
3. Review health checks: `docker-compose ps`
4. Consult troubleshooting section above