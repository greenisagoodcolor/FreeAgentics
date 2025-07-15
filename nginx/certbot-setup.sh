#!/bin/bash

# Certbot Setup Script for Docker Environment
# This script sets up Let's Encrypt certificates in a Docker-compose environment

set -euo pipefail

# Configuration
DOMAIN=${DOMAIN:-"yourdomain.com"}
EMAIL=${EMAIL:-"admin@yourdomain.com"}
STAGING=${STAGING:-"false"}
COMPOSE_FILE=${COMPOSE_FILE:-"docker-compose.yml"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if domain is provided
if [ "$DOMAIN" = "yourdomain.com" ]; then
    error "Please set DOMAIN environment variable to your actual domain"
fi

# Check if email is provided
if [ "$EMAIL" = "admin@yourdomain.com" ]; then
    error "Please set EMAIL environment variable to your actual email"
fi

# Create necessary directories
log "Creating necessary directories..."
mkdir -p ./nginx/ssl
mkdir -p ./certbot/www
mkdir -p ./certbot/conf

# Generate DH parameters if not present
if [ ! -f "./nginx/dhparam.pem" ]; then
    log "Generating DH parameters (this may take a while)..."
    openssl dhparam -out "./nginx/dhparam.pem" 2048
fi

# Create temporary nginx configuration for certificate request
log "Creating temporary nginx configuration..."
cat > ./nginx/nginx-temp.conf << EOF
events {
    worker_connections 1024;
}

http {
    server {
        listen 80;
        server_name $DOMAIN;
        
        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }
        
        location / {
            return 301 https://\$host\$request_uri;
        }
    }
}
EOF

# Create temporary docker-compose file for certificate request
log "Creating temporary docker-compose configuration..."
cat > docker-compose-certbot.yml << EOF
version: '3.8'

services:
  nginx-temp:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx/nginx-temp.conf:/etc/nginx/nginx.conf:ro
      - ./certbot/www:/var/www/certbot:ro
    depends_on:
      - certbot
    networks:
      - cert-network

  certbot:
    image: certbot/certbot
    volumes:
      - ./certbot/conf:/etc/letsencrypt
      - ./certbot/www:/var/www/certbot
    networks:
      - cert-network

networks:
  cert-network:
    driver: bridge
EOF

# Start temporary nginx
log "Starting temporary nginx for certificate request..."
docker-compose -f docker-compose-certbot.yml up -d nginx-temp

# Wait for nginx to start
sleep 5

# Request certificate
log "Requesting Let's Encrypt certificate for $DOMAIN..."
CERTBOT_ARGS="--webroot --webroot-path=/var/www/certbot --email $EMAIL --agree-tos --no-eff-email"

if [ "$STAGING" = "true" ]; then
    CERTBOT_ARGS="$CERTBOT_ARGS --staging"
    warn "Using Let's Encrypt staging environment"
fi

docker-compose -f docker-compose-certbot.yml run --rm certbot certonly $CERTBOT_ARGS -d "$DOMAIN"

# Stop temporary services
log "Stopping temporary services..."
docker-compose -f docker-compose-certbot.yml down

# Copy certificates to nginx ssl directory
log "Copying certificates to nginx ssl directory..."
if [ -d "./certbot/conf/live/$DOMAIN" ]; then
    cp "./certbot/conf/live/$DOMAIN/fullchain.pem" "./nginx/ssl/cert.pem"
    cp "./certbot/conf/live/$DOMAIN/privkey.pem" "./nginx/ssl/key.pem"
    
    # Set proper permissions
    chmod 600 "./nginx/ssl/key.pem"
    chmod 644 "./nginx/ssl/cert.pem"
    
    log "Certificates copied successfully"
else
    error "Certificate directory not found. Certificate request may have failed."
fi

# Create certificate renewal script
log "Creating certificate renewal script..."
cat > ./scripts/renew-cert.sh << 'EOF'
#!/bin/bash

# Certificate renewal script for Docker environment
set -euo pipefail

DOMAIN=${DOMAIN:-"yourdomain.com"}

# Renew certificate
docker-compose -f docker-compose-certbot.yml run --rm certbot renew --quiet --no-self-upgrade

# Copy renewed certificates
if [ -d "./certbot/conf/live/$DOMAIN" ]; then
    cp "./certbot/conf/live/$DOMAIN/fullchain.pem" "./nginx/ssl/cert.pem"
    cp "./certbot/conf/live/$DOMAIN/privkey.pem" "./nginx/ssl/key.pem"
    
    # Set proper permissions
    chmod 600 "./nginx/ssl/key.pem"
    chmod 644 "./nginx/ssl/cert.pem"
    
    # Reload nginx
    docker-compose exec nginx nginx -s reload
    
    echo "Certificate renewed successfully"
else
    echo "Certificate renewal failed"
    exit 1
fi
EOF

chmod +x ./scripts/renew-cert.sh

# Create cron job for automatic renewal
log "Creating cron job for automatic certificate renewal..."
cat > ./scripts/cert-renewal.cron << EOF
# Renew Let's Encrypt certificates twice daily
0 0,12 * * * cd $(pwd) && ./scripts/renew-cert.sh >> /var/log/cert-renewal.log 2>&1
EOF

# Clean up temporary files
rm -f ./nginx/nginx-temp.conf
rm -f docker-compose-certbot.yml

log "Certbot setup completed successfully!"
log "Certificate location: ./nginx/ssl/"
log "Renewal script: ./scripts/renew-cert.sh"
log "Cron job configuration: ./scripts/cert-renewal.cron"
log ""
log "To install the cron job, run:"
log "  sudo crontab -u root ./scripts/cert-renewal.cron"
log ""
log "Next steps:"
log "1. Update your nginx configuration to use the certificates"
log "2. Start your production environment: docker-compose up -d"
log "3. Test SSL: curl -I https://$DOMAIN"