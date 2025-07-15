# Docker Setup Guide

## Consolidated Docker Configuration

FreeAgentics now uses a **single consolidated docker-compose.yml** file with profiles for different environments, reducing complexity and improving maintainability.

### Available Profiles

| Profile | Use Case | Command | Description |
|---------|----------|---------|-------------|
| `dev` | Development | `docker-compose --profile dev up` | Hot reload, debugging enabled |
| `demo` | Demonstrations | `docker-compose --profile demo up` | Includes seed data and test users |
| `prod` | Production-like | `docker-compose --profile prod up` | Production settings (use docker-compose.production.yml for full security) |
| `monitoring` | Observability | `--profile monitoring` | Adds Prometheus & Grafana |
| `tools` | Dev Tools | `--profile tools` | Adds pgAdmin and other utilities |

### Core Files

| File | Purpose | Notes |
|------|---------|-------|
| `docker-compose.yml` | **Main configuration** | Single file with all profiles |
| `docker-compose.production.yml` | **Production deployment** | Full security features, separate for safety |
| `Dockerfile.production` | **Backend multi-stage** | Supports dev/test/production targets |
| `web/Dockerfile.production` | **Frontend production** | Optimized for production |
| `web/Dockerfile.dev` | **Frontend development** | Hot reload for development |

### Quick Start Scripts

| Script | Profile Used | Purpose |
|--------|--------------|---------|
| `./start-simple.sh` | `dev` | Quick development start |
| `./start-demo.sh` | `demo` | Full demo with seed data |

## Usage Examples

### Development Environment
```bash
# Basic development
docker-compose --profile dev up

# Development with monitoring
docker-compose --profile dev --profile monitoring up

# Development with tools
docker-compose --profile dev --profile tools up
```

### Demo Environment
```bash
# Full demo with seed data
docker-compose --profile demo up

# Or use the convenience script
./start-demo.sh
```

### Production-like Testing
```bash
# Production profile (still uses some dev settings)
docker-compose --profile prod up

# For actual production, use:
docker-compose -f docker-compose.production.yml up -d
```

### Just the Databases
```bash
# Start only PostgreSQL and Redis
docker-compose up postgres redis
```

### Combining Profiles
```bash
# Development with all bells and whistles
docker-compose --profile dev --profile monitoring --profile tools up
```

## Port Assignments

| Service | Port | Purpose |
|---------|------|---------|
| Frontend | 3000 | Web interface |
| Backend API | 8000 | REST API |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache/sessions |
| Prometheus | 9090 | Metrics (monitoring profile) |
| Grafana | 3001 | Dashboards (monitoring profile) |
| pgAdmin | 5050 | Database UI (tools profile) |

## Environment Variables

### Default Values
The consolidated docker-compose.yml includes sensible development defaults. For production, always override these:

```bash
# Copy production template
cp .env.production.template .env

# Key variables to set:
POSTGRES_PASSWORD=<secure_password>
REDIS_PASSWORD=<secure_password>
SECRET_KEY=<64_char_random_string>
JWT_SECRET=<64_char_random_string>
```

## Migration from Old Structure

### Removed Files
The following files have been removed during consolidation:
- `docker-compose.simple.yml` → Use `--profile dev`
- `docker-compose.demo.yml` → Use `--profile demo`
- `docker-compose.monitoring.yml` → Use `--profile monitoring`
- `docker-compose.scale.yml` → Removed (outdated)
- `docker-compose-release.yml` → Removed (security issues)
- `docker-compose.observability.yml` → Removed (duplicate)

### Updated Commands
If you were using:
- `docker-compose -f docker-compose.simple.yml up` → `docker-compose --profile dev up`
- `docker-compose -f docker-compose.demo.yml up` → `docker-compose --profile demo up`
- `docker-compose -f docker-compose.monitoring.yml up` → `docker-compose --profile monitoring up`

## Security Notes

### Development Profiles (dev, demo)
- Default passwords for convenience
- CORS allows localhost origins
- Debug mode enabled
- Volume mounts for hot reload

### Production Configuration
- Use `docker-compose.production.yml` for real production
- All secrets via environment variables
- Security headers enabled
- Rate limiting and DDoS protection
- Read-only root filesystems
- Non-root user execution

## Troubleshooting

### Profile not working?
```bash
# Check Docker Compose version
docker-compose version
# Need v2.0+ for profiles support

# Verify configuration
docker-compose --profile dev config
```

### Port conflicts?
```bash
# Check what's using ports
lsof -i :3000
lsof -i :8000
```

### Clean slate needed?
```bash
# Stop everything and clean volumes
docker-compose down -v
# Then start fresh
docker-compose --profile dev up
```