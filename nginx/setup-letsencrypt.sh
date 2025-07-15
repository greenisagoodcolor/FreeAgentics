#!/bin/bash

# Let's Encrypt Setup Script for FreeAgentics
# This script sets up Let's Encrypt certificates with automatic renewal

set -euo pipefail

# Configuration
DOMAIN=${DOMAIN:-"yourdomain.com"}
EMAIL=${EMAIL:-"admin@yourdomain.com"}
WEBROOT="/var/www/certbot"
CERTBOT_DATA="/etc/letsencrypt"
NGINX_CONF="/etc/nginx"
STAGING=${STAGING:-"false"}

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
mkdir -p "$WEBROOT"
mkdir -p "$CERTBOT_DATA"

# Install certbot if not present
if ! command -v certbot &> /dev/null; then
    log "Installing certbot..."
    if [ -f /etc/debian_version ]; then
        apt-get update
        apt-get install -y certbot python3-certbot-nginx
    elif [ -f /etc/redhat-release ]; then
        yum install -y certbot python3-certbot-nginx
    else
        error "Unsupported OS. Please install certbot manually."
    fi
fi

# Generate DH parameters if not present
if [ ! -f "$NGINX_CONF/dhparam.pem" ]; then
    log "Generating DH parameters (this may take a while)..."
    openssl dhparam -out "$NGINX_CONF/dhparam.pem" 2048
fi

# Create temporary nginx config for initial certificate request
log "Creating temporary nginx configuration..."
cat > /tmp/nginx-temp.conf << EOF
events {
    worker_connections 1024;
}

http {
    server {
        listen 80;
        server_name $DOMAIN;
        
        location /.well-known/acme-challenge/ {
            root $WEBROOT;
        }
        
        location / {
            return 301 https://\$host\$request_uri;
        }
    }
}
EOF

# Test nginx configuration
log "Testing nginx configuration..."
nginx -t -c /tmp/nginx-temp.conf

# Stop nginx if running
if pgrep -x "nginx" > /dev/null; then
    log "Stopping nginx..."
    systemctl stop nginx
fi

# Start nginx with temporary configuration
log "Starting nginx with temporary configuration..."
nginx -c /tmp/nginx-temp.conf

# Wait for nginx to start
sleep 2

# Request Let's Encrypt certificate
log "Requesting Let's Encrypt certificate for $DOMAIN..."
CERTBOT_ARGS="--webroot --webroot-path=$WEBROOT --email $EMAIL --agree-tos --no-eff-email"

if [ "$STAGING" = "true" ]; then
    CERTBOT_ARGS="$CERTBOT_ARGS --staging"
    warn "Using Let's Encrypt staging environment"
fi

certbot certonly $CERTBOT_ARGS -d "$DOMAIN"

# Copy certificates to nginx ssl directory
log "Copying certificates to nginx ssl directory..."
mkdir -p /etc/nginx/ssl
cp "$CERTBOT_DATA/live/$DOMAIN/fullchain.pem" /etc/nginx/ssl/cert.pem
cp "$CERTBOT_DATA/live/$DOMAIN/privkey.pem" /etc/nginx/ssl/key.pem

# Set proper permissions
chmod 600 /etc/nginx/ssl/key.pem
chmod 644 /etc/nginx/ssl/cert.pem

# Stop temporary nginx
log "Stopping temporary nginx..."
nginx -s quit

# Clean up temporary configuration
rm -f /tmp/nginx-temp.conf

# Create certificate renewal script
log "Creating certificate renewal script..."
cat > /usr/local/bin/renew-cert.sh << 'EOF'
#!/bin/bash

# Certificate renewal script
set -euo pipefail

DOMAIN=${DOMAIN:-"yourdomain.com"}
CERTBOT_DATA="/etc/letsencrypt"

# Renew certificate
certbot renew --quiet --no-self-upgrade

# Copy renewed certificates
if [ -f "$CERTBOT_DATA/live/$DOMAIN/fullchain.pem" ]; then
    cp "$CERTBOT_DATA/live/$DOMAIN/fullchain.pem" /etc/nginx/ssl/cert.pem
    cp "$CERTBOT_DATA/live/$DOMAIN/privkey.pem" /etc/nginx/ssl/key.pem
    
    # Set proper permissions
    chmod 600 /etc/nginx/ssl/key.pem
    chmod 644 /etc/nginx/ssl/cert.pem
    
    # Reload nginx
    nginx -s reload
    
    echo "Certificate renewed successfully"
else
    echo "Certificate renewal failed"
    exit 1
fi
EOF

chmod +x /usr/local/bin/renew-cert.sh

# Create systemd service for certificate renewal
log "Creating systemd service for certificate renewal..."
cat > /etc/systemd/system/cert-renewal.service << EOF
[Unit]
Description=Renew Let's Encrypt certificates
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/renew-cert.sh
User=root
Environment=DOMAIN=$DOMAIN
EOF

# Create systemd timer for automatic renewal
cat > /etc/systemd/system/cert-renewal.timer << EOF
[Unit]
Description=Run cert-renewal twice daily
Requires=cert-renewal.service

[Timer]
OnCalendar=*-*-* 00,12:00:00
RandomizedDelaySec=3600
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Enable and start the timer
systemctl daemon-reload
systemctl enable cert-renewal.timer
systemctl start cert-renewal.timer

log "Let's Encrypt setup completed successfully!"
log "Certificate location: /etc/nginx/ssl/"
log "Automatic renewal enabled via systemd timer"
log "Next steps:"
log "1. Update your nginx configuration to use the certificates"
log "2. Test your SSL configuration with: nginx -t"
log "3. Reload nginx: systemctl reload nginx"
log "4. Test SSL: curl -I https://$DOMAIN"