#!/bin/bash

# Production SSL Deployment Script for FreeAgentics
# This script deploys the FreeAgentics application with SSL/TLS configuration

set -euo pipefail

# Configuration
DOMAIN=${DOMAIN:-"yourdomain.com"}
EMAIL=${EMAIL:-"admin@yourdomain.com"}
STAGING=${STAGING:-"false"}
BACKUP_EXISTING=${BACKUP_EXISTING:-"true"}
SKIP_CERT_SETUP=${SKIP_CERT_SETUP:-"false"}
ENV_FILE=${ENV_FILE:-".env.production"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

info() {
    echo -e "${BLUE}[DEPLOY]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if running as root or with sudo
check_permissions() {
    if [[ $EUID -ne 0 ]] && ! groups | grep -q docker; then
        error "This script requires root privileges or membership in the docker group"
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi

    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi

    # Check if domain is provided
    if [ "$DOMAIN" = "yourdomain.com" ]; then
        error "Please set DOMAIN environment variable to your actual domain"
    fi

    # Check if email is provided
    if [ "$EMAIL" = "admin@yourdomain.com" ]; then
        error "Please set EMAIL environment variable to your actual email"
    fi

    # Check if environment file exists
    if [ ! -f "$ENV_FILE" ]; then
        warn "Environment file $ENV_FILE not found. Creating from template..."
        if [ -f ".env.production.ssl.template" ]; then
            cp ".env.production.ssl.template" "$ENV_FILE"
            error "Please edit $ENV_FILE with your configuration and run again"
        else
            error "Environment template not found. Please create $ENV_FILE manually"
        fi
    fi

    success "Prerequisites check passed"
}

# Backup existing deployment
backup_existing() {
    if [ "$BACKUP_EXISTING" = "true" ]; then
        log "Creating backup of existing deployment..."

        local backup_dir="backup-$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$backup_dir"

        # Backup configuration files
        if [ -f "docker-compose.yml" ]; then
            cp "docker-compose.yml" "$backup_dir/"
        fi

        if [ -f "$ENV_FILE" ]; then
            cp "$ENV_FILE" "$backup_dir/"
        fi

        # Backup SSL certificates
        if [ -d "nginx/ssl" ]; then
            cp -r "nginx/ssl" "$backup_dir/"
        fi

        # Backup database (if running)
        if docker-compose ps postgres | grep -q "Up"; then
            log "Backing up database..."
            docker-compose exec postgres pg_dump -U freeagentics freeagentics > "$backup_dir/database.sql"
        fi

        success "Backup created in $backup_dir"
    fi
}

# Generate DH parameters
generate_dhparam() {
    log "Checking DH parameters..."

    if [ ! -f "nginx/dhparam.pem" ]; then
        log "Generating DH parameters..."
        ./nginx/generate-dhparam.sh
        success "DH parameters generated"
    else
        log "DH parameters already exist"
    fi
}

# Set up SSL certificates
setup_ssl_certificates() {
    if [ "$SKIP_CERT_SETUP" = "true" ]; then
        log "Skipping SSL certificate setup"
        return 0
    fi

    log "Setting up SSL certificates..."

    # Check if certificates already exist
    if [ -f "nginx/ssl/cert.pem" ] && [ -f "nginx/ssl/key.pem" ]; then
        log "SSL certificates already exist"

        # Check if certificates are valid
        if openssl x509 -in "nginx/ssl/cert.pem" -noout -checkend 86400; then
            log "Existing certificates are valid for at least 24 hours"
            return 0
        else
            warn "Existing certificates are expiring soon or invalid"
        fi
    fi

    # Set up Let's Encrypt certificates
    log "Setting up Let's Encrypt certificates..."

    export DOMAIN="$DOMAIN"
    export EMAIL="$EMAIL"
    export STAGING="$STAGING"

    ./nginx/certbot-setup.sh

    success "SSL certificates configured"
}

# Stop existing services
stop_services() {
    log "Stopping existing services..."

    # Stop with multiple compose files
    docker-compose -f docker-compose.yml down --remove-orphans
    docker-compose -f docker-compose.production.yml down --remove-orphans

    # Clean up unused containers and networks
    docker system prune -f

    success "Services stopped"
}

# Build and start services
start_services() {
    log "Building and starting services..."

    # Build images
    info "Building Docker images..."
    docker-compose -f docker-compose.production.yml build --no-cache

    # Start database first
    info "Starting database..."
    docker-compose -f docker-compose.production.yml up -d postgres redis

    # Wait for database to be ready
    log "Waiting for database to be ready..."
    sleep 10

    # Run migrations
    info "Running database migrations..."
    docker-compose -f docker-compose.production.yml run --rm migration

    # Start application services
    info "Starting application services..."
    docker-compose -f docker-compose.production.yml up -d backend frontend

    # Wait for application to be ready
    log "Waiting for application to be ready..."
    sleep 15

    # Start nginx
    info "Starting nginx..."
    docker-compose -f docker-compose.production.yml up -d nginx

    # Start monitoring services
    info "Starting monitoring services..."
    docker-compose -f docker-compose.production.yml up -d ssl-monitor

    success "Services started"
}

# Test deployment
test_deployment() {
    log "Testing deployment..."

    # Test HTTP redirect
    log "Testing HTTP to HTTPS redirect..."
    if curl -I -s -L "http://$DOMAIN" | grep -q "301\|302"; then
        success "HTTP to HTTPS redirect working"
    else
        warn "HTTP to HTTPS redirect may not be working"
    fi

    # Test HTTPS connection
    log "Testing HTTPS connection..."
    if curl -I -s --connect-timeout 10 "https://$DOMAIN" | grep -q "200"; then
        success "HTTPS connection working"
    else
        warn "HTTPS connection may not be working"
    fi

    # Test API endpoint
    log "Testing API endpoint..."
    if curl -I -s --connect-timeout 10 "https://$DOMAIN/api/health" | grep -q "200"; then
        success "API endpoint working"
    else
        warn "API endpoint may not be working"
    fi

    # Test SSL configuration
    log "Testing SSL configuration..."
    DOMAIN="$DOMAIN" ./nginx/test-ssl.sh > /tmp/ssl-test.log 2>&1

    if grep -q "PASS" /tmp/ssl-test.log; then
        success "SSL configuration tests passed"
    else
        warn "Some SSL configuration tests failed. Check /tmp/ssl-test.log for details"
    fi

    success "Deployment testing completed"
}

# Set up monitoring
setup_monitoring() {
    log "Setting up monitoring..."

    # Create monitoring configuration
    cat > monitoring-config.yml << EOF
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - freeagentics-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    networks:
      - freeagentics-network

networks:
  freeagentics-network:
    external: true
EOF

    # Start monitoring services
    docker-compose -f monitoring-config.yml up -d

    success "Monitoring services started"
}

# Create deployment summary
create_summary() {
    log "Creating deployment summary..."

    local summary_file="deployment-summary-$(date +%Y%m%d-%H%M%S).md"

    cat > "$summary_file" << EOF
# FreeAgentics Production Deployment Summary

**Deployment Date:** $(date)
**Domain:** $DOMAIN
**SSL Configuration:** Enabled with Let's Encrypt

## Services Status

\`\`\`
$(docker-compose -f docker-compose.production.yml ps)
\`\`\`

## SSL Certificate Information

\`\`\`
$(openssl x509 -in nginx/ssl/cert.pem -noout -text | grep -A2 -B2 "Subject:\|Issuer:\|Not Before:\|Not After:")
\`\`\`

## Security Test Results

\`\`\`
$(cat /tmp/ssl-test.log | head -20)
\`\`\`

## Access URLs

- **Frontend:** https://$DOMAIN
- **API:** https://$DOMAIN/api
- **Health Check:** https://$DOMAIN/health
- **SSL Health:** https://$DOMAIN/ssl-health

## Monitoring

- **Prometheus:** http://$(hostname -I | awk '{print $1}'):9090
- **Grafana:** http://$(hostname -I | awk '{print $1}'):3001

## Next Steps

1. Configure DNS to point $DOMAIN to this server
2. Set up SSL certificate monitoring alerts
3. Configure backup procedures
4. Review and update security policies
5. Test disaster recovery procedures

## Troubleshooting

- **Logs:** \`docker-compose -f docker-compose.production.yml logs -f\`
- **SSL Test:** \`DOMAIN=$DOMAIN ./nginx/test-ssl.sh\`
- **Certificate Renewal:** \`./scripts/renew-cert.sh\`

## Files Created

- Environment: $ENV_FILE
- SSL Certificates: nginx/ssl/
- DH Parameters: nginx/dhparam.pem
- Backup: backup-*/
- Summary: $summary_file
EOF

    success "Deployment summary created: $summary_file"
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    rm -f /tmp/ssl-test.log
    rm -f monitoring-config.yml
}

# Main deployment function
main() {
    log "Starting FreeAgentics production deployment with SSL..."

    # Trap cleanup on exit
    trap cleanup EXIT

    # Check permissions
    check_permissions

    # Check prerequisites
    check_prerequisites

    # Backup existing deployment
    backup_existing

    # Generate DH parameters
    generate_dhparam

    # Set up SSL certificates
    setup_ssl_certificates

    # Stop existing services
    stop_services

    # Start services
    start_services

    # Test deployment
    test_deployment

    # Set up monitoring
    setup_monitoring

    # Create deployment summary
    create_summary

    success "Production deployment completed successfully!"
    echo ""
    log "Access your application at: https://$DOMAIN"
    log "API documentation: https://$DOMAIN/api/docs"
    log "Health check: https://$DOMAIN/health"
    echo ""
    log "To monitor SSL certificates:"
    log "  DOMAIN=$DOMAIN ./nginx/monitor-ssl.sh health-check"
    echo ""
    log "To view logs:"
    log "  docker-compose -f docker-compose.production.yml logs -f"
    echo ""
    log "To renew SSL certificates:"
    log "  ./scripts/renew-cert.sh"
}

# Show usage if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Environment variables:"
    echo "  DOMAIN              - Domain name (required)"
    echo "  EMAIL               - Email for Let's Encrypt (required)"
    echo "  STAGING             - Use Let's Encrypt staging (default: false)"
    echo "  BACKUP_EXISTING     - Backup existing deployment (default: true)"
    echo "  SKIP_CERT_SETUP     - Skip SSL certificate setup (default: false)"
    echo "  ENV_FILE            - Environment file (default: .env.production)"
    echo ""
    echo "Examples:"
    echo "  DOMAIN=example.com EMAIL=admin@example.com $0"
    echo "  DOMAIN=example.com EMAIL=admin@example.com STAGING=true $0"
    echo ""
    exit 1
fi

# Run main function
main "$@"
