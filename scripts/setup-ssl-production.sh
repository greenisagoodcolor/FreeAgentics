#!/bin/bash
# SSL/TLS Production Setup Script for FreeAgentics
# Handles SSL certificate generation, renewal, and configuration

set -euo pipefail

# Configuration
DOMAIN="${DOMAIN:-freeagentics.local}"
EMAIL="${EMAIL:-admin@freeagentics.local}"
NGINX_CONF_DIR="${NGINX_CONF_DIR:-./nginx}"
CERTBOT_DIR="${CERTBOT_DIR:-./certbot}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root. Consider using a non-root user with sudo."
    fi
}

# Function to install dependencies
install_dependencies() {
    log "Installing SSL/TLS dependencies..."
    
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y openssl certbot nginx-common
    elif command -v yum &> /dev/null; then
        sudo yum install -y openssl certbot
    else
        error "Unsupported package manager. Please install openssl and certbot manually."
        exit 1
    fi
}

# Function to create directories
create_directories() {
    log "Creating SSL directories..."
    
    mkdir -p "$NGINX_CONF_DIR"/{ssl,conf.d,snippets}
    mkdir -p "$CERTBOT_DIR"/{conf,www}
    mkdir -p ./logs/ssl
    
    log "SSL directories created successfully"
}

# Function to generate DH parameters
generate_dhparams() {
    local dhparam_file="$NGINX_CONF_DIR/dhparam.pem"
    
    if [[ ! -f "$dhparam_file" ]]; then
        log "Generating Diffie-Hellman parameters (this may take a while)..."
        openssl dhparam -out "$dhparam_file" 2048
        chmod 644 "$dhparam_file"
        log "DH parameters generated successfully"
    else
        log "DH parameters already exist"
    fi
}

# Function to create SSL configuration snippet
create_ssl_config() {
    log "Creating SSL configuration snippet..."
    
    cat > "$NGINX_CONF_DIR/snippets/ssl-params.conf" << 'EOF'
# SSL/TLS Configuration for FreeAgentics Production
# Modern configuration that provides excellent security

# SSL Protocols and Ciphers
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;

# SSL Session Settings
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 1d;
ssl_session_tickets off;

# DH Parameters
ssl_dhparam /etc/nginx/dhparam.pem;

# OCSP Stapling
ssl_stapling on;
ssl_stapling_verify on;
resolver 8.8.8.8 8.8.4.4 valid=300s;
resolver_timeout 5s;

# Security Headers
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
add_header X-Frame-Options SAMEORIGIN always;
add_header X-Content-Type-Options nosniff always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self' wss: https:; frame-ancestors 'self';" always;

# Perfect Forward Secrecy
ssl_ecdh_curve secp384r1;
EOF

    log "SSL configuration snippet created"
}

# Function to generate self-signed certificate for development
generate_self_signed() {
    log "Generating self-signed certificate for development..."
    
    local ssl_dir="$NGINX_CONF_DIR/ssl"
    mkdir -p "$ssl_dir"
    
    # Generate private key
    openssl genrsa -out "$ssl_dir/$DOMAIN.key" 2048
    
    # Generate certificate signing request
    openssl req -new -key "$ssl_dir/$DOMAIN.key" -out "$ssl_dir/$DOMAIN.csr" -subj "/C=US/ST=CA/L=San Francisco/O=FreeAgentics/CN=$DOMAIN"
    
    # Generate self-signed certificate
    openssl x509 -req -in "$ssl_dir/$DOMAIN.csr" -signkey "$ssl_dir/$DOMAIN.key" -out "$ssl_dir/$DOMAIN.crt" -days 365 -extensions v3_req -extfile <(
cat << EOF
[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = $DOMAIN
DNS.2 = *.$DOMAIN
DNS.3 = localhost
IP.1 = 127.0.0.1
EOF
    )
    
    # Set permissions
    chmod 600 "$ssl_dir/$DOMAIN.key"
    chmod 644 "$ssl_dir/$DOMAIN.crt"
    
    # Clean up CSR
    rm "$ssl_dir/$DOMAIN.csr"
    
    log "Self-signed certificate generated for $DOMAIN"
}

# Function to obtain Let's Encrypt certificate
obtain_letsencrypt_cert() {
    log "Obtaining Let's Encrypt certificate for $DOMAIN..."
    
    if [[ "$ENVIRONMENT" != "production" ]]; then
        warn "Let's Encrypt should only be used in production. Using staging environment."
        STAGING_FLAG="--staging"
    else
        STAGING_FLAG=""
    fi
    
    # Ensure certbot directories exist
    mkdir -p "$CERTBOT_DIR"/{conf,www}
    
    # Obtain certificate
    if certbot certonly \
        --webroot \
        --webroot-path="$CERTBOT_DIR/www" \
        --email "$EMAIL" \
        --agree-tos \
        --no-eff-email \
        --force-renewal \
        $STAGING_FLAG \
        -d "$DOMAIN"; then
        
        log "Let's Encrypt certificate obtained successfully"
        
        # Copy certificates to nginx directory
        local ssl_dir="$NGINX_CONF_DIR/ssl"
        mkdir -p "$ssl_dir"
        
        if [[ "$ENVIRONMENT" == "production" ]]; then
            CERT_DIR="/etc/letsencrypt/live/$DOMAIN"
        else
            CERT_DIR="$CERTBOT_DIR/conf/live/$DOMAIN"
        fi
        
        if [[ -d "$CERT_DIR" ]]; then
            cp "$CERT_DIR/fullchain.pem" "$ssl_dir/$DOMAIN.crt"
            cp "$CERT_DIR/privkey.pem" "$ssl_dir/$DOMAIN.key"
            chmod 644 "$ssl_dir/$DOMAIN.crt"
            chmod 600 "$ssl_dir/$DOMAIN.key"
            log "Certificates copied to nginx directory"
        fi
    else
        error "Failed to obtain Let's Encrypt certificate"
        exit 1
    fi
}

# Function to create nginx configuration for SSL
create_nginx_ssl_config() {
    log "Creating nginx SSL configuration..."
    
    cat > "$NGINX_CONF_DIR/conf.d/ssl-$DOMAIN.conf" << EOF
# SSL Configuration for $DOMAIN
server {
    listen 80;
    server_name $DOMAIN www.$DOMAIN;
    
    # Let's Encrypt challenge
    location /.well-known/acme-challenge/ {
        root $CERTBOT_DIR/www;
    }
    
    # Redirect HTTP to HTTPS
    location / {
        return 301 https://\$server_name\$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name $DOMAIN www.$DOMAIN;
    
    # SSL Certificate Configuration
    ssl_certificate /etc/nginx/ssl/$DOMAIN.crt;
    ssl_certificate_key /etc/nginx/ssl/$DOMAIN.key;
    
    # Include SSL parameters
    include /etc/nginx/snippets/ssl-params.conf;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Proxy settings for backend
    location /api/ {
        proxy_pass http://backend:8000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header X-Forwarded-Host \$host;
        proxy_set_header X-Forwarded-Port \$server_port;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Frontend application
    location / {
        proxy_pass http://frontend:3000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

    log "Nginx SSL configuration created"
}

# Function to test SSL configuration
test_ssl_config() {
    log "Testing SSL configuration..."
    
    # Test nginx configuration
    if nginx -t -c "$NGINX_CONF_DIR/nginx.conf" 2>/dev/null; then
        log "Nginx configuration is valid"
    else
        warn "Nginx configuration test failed. Please check the configuration manually."
    fi
    
    # Test SSL certificate
    local ssl_dir="$NGINX_CONF_DIR/ssl"
    if [[ -f "$ssl_dir/$DOMAIN.crt" && -f "$ssl_dir/$DOMAIN.key" ]]; then
        # Check certificate validity
        if openssl x509 -in "$ssl_dir/$DOMAIN.crt" -text -noout > /dev/null 2>&1; then
            log "SSL certificate is valid"
            
            # Display certificate info
            log "Certificate information:"
            openssl x509 -in "$ssl_dir/$DOMAIN.crt" -text -noout | grep -E "(Subject:|Issuer:|Not After)"
        else
            error "SSL certificate is invalid"
            exit 1
        fi
        
        # Check private key
        if openssl rsa -in "$ssl_dir/$DOMAIN.key" -check -noout > /dev/null 2>&1; then
            log "SSL private key is valid"
        else
            error "SSL private key is invalid"
            exit 1
        fi
        
        # Verify certificate and key match
        cert_hash=$(openssl x509 -noout -modulus -in "$ssl_dir/$DOMAIN.crt" | openssl md5)
        key_hash=$(openssl rsa -noout -modulus -in "$ssl_dir/$DOMAIN.key" | openssl md5)
        
        if [[ "$cert_hash" == "$key_hash" ]]; then
            log "Certificate and private key match"
        else
            error "Certificate and private key do not match"
            exit 1
        fi
    else
        error "SSL certificate or private key not found"
        exit 1
    fi
}

# Function to setup certificate renewal
setup_renewal() {
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "Setting up automatic certificate renewal..."
        
        # Create renewal script
        cat > "./scripts/renew-ssl.sh" << 'EOF'
#!/bin/bash
# SSL Certificate Renewal Script

set -euo pipefail

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting SSL certificate renewal..."

# Renew certificates
if certbot renew --quiet; then
    log "Certificate renewal successful"
    
    # Reload nginx
    if docker-compose exec nginx nginx -s reload; then
        log "Nginx reloaded successfully"
    else
        log "Failed to reload nginx"
        exit 1
    fi
else
    log "Certificate renewal failed"
    exit 1
fi

log "SSL certificate renewal completed"
EOF
        
        chmod +x "./scripts/renew-ssl.sh"
        
        log "Renewal script created at ./scripts/renew-ssl.sh"
        log "Add this to crontab for automatic renewal:"
        log "0 2 * * 1 /path/to/freeagentics/scripts/renew-ssl.sh >> /var/log/ssl-renewal.log 2>&1"
    fi
}

# Main function
main() {
    log "Starting SSL/TLS setup for FreeAgentics ($ENVIRONMENT environment)"
    log "Domain: $DOMAIN"
    log "Email: $EMAIL"
    
    check_root
    create_directories
    
    case "${1:-auto}" in
        "self-signed")
            log "Generating self-signed certificate..."
            generate_dhparams
            create_ssl_config
            generate_self_signed
            create_nginx_ssl_config
            test_ssl_config
            ;;
        "letsencrypt")
            log "Obtaining Let's Encrypt certificate..."
            install_dependencies
            generate_dhparams
            create_ssl_config
            obtain_letsencrypt_cert
            create_nginx_ssl_config
            test_ssl_config
            setup_renewal
            ;;
        "auto")
            if [[ "$ENVIRONMENT" == "production" ]]; then
                log "Production environment detected, using Let's Encrypt..."
                install_dependencies
                generate_dhparams
                create_ssl_config
                obtain_letsencrypt_cert
                create_nginx_ssl_config
                test_ssl_config
                setup_renewal
            else
                log "Development environment detected, using self-signed certificate..."
                generate_dhparams
                create_ssl_config
                generate_self_signed
                create_nginx_ssl_config
                test_ssl_config
            fi
            ;;
        "test")
            test_ssl_config
            ;;
        *)
            echo "Usage: $0 {self-signed|letsencrypt|auto|test}"
            echo "  self-signed  - Generate self-signed certificate"
            echo "  letsencrypt  - Obtain Let's Encrypt certificate"
            echo "  auto         - Automatically choose based on environment"
            echo "  test         - Test existing SSL configuration"
            exit 1
            ;;
    esac
    
    log "SSL/TLS setup completed successfully!"
    log "Next steps:"
    log "1. Update your DNS to point $DOMAIN to this server"
    log "2. Start the application with: docker-compose -f docker-compose.production.yml up -d"
    log "3. Test SSL: curl -I https://$DOMAIN/health"
}

# Execute main function
main "$@"