#!/bin/bash

# Generate Certificate Pin Script
# This script generates HPKP (HTTP Public Key Pinning) hashes for certificate pinning

set -euo pipefail

# Configuration
DOMAIN=${DOMAIN:-"yourdomain.com"}
CERT_FILE=${CERT_FILE:-"./nginx/ssl/cert.pem"}
KEY_FILE=${KEY_FILE:-"./nginx/ssl/key.pem"}

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
    echo -e "${BLUE}[PIN]${NC} $1"
}

# Check if OpenSSL is available
if ! command -v openssl &> /dev/null; then
    error "OpenSSL is not installed. Please install OpenSSL to continue."
fi

# Check if domain is provided
if [ "$DOMAIN" = "yourdomain.com" ]; then
    error "Please set DOMAIN environment variable to your actual domain"
fi

# Function to generate pin from certificate
generate_pin_from_cert() {
    local cert_file=$1
    local pin
    
    if [ -f "$cert_file" ]; then
        pin=$(openssl x509 -in "$cert_file" -pubkey -noout | openssl rsa -pubin -outform der 2>/dev/null | openssl dgst -sha256 -binary | openssl enc -base64)
        echo "$pin"
    else
        error "Certificate file not found: $cert_file"
    fi
}

# Function to generate pin from private key
generate_pin_from_key() {
    local key_file=$1
    local pin
    
    if [ -f "$key_file" ]; then
        pin=$(openssl rsa -in "$key_file" -pubout -outform der 2>/dev/null | openssl dgst -sha256 -binary | openssl enc -base64)
        echo "$pin"
    else
        error "Private key file not found: $key_file"
    fi
}

# Function to generate pin from remote certificate
generate_pin_from_remote() {
    local domain=$1
    local port=${2:-443}
    local pin
    
    pin=$(openssl s_client -servername "$domain" -connect "$domain:$port" </dev/null 2>/dev/null | openssl x509 -pubkey -noout | openssl rsa -pubin -outform der 2>/dev/null | openssl dgst -sha256 -binary | openssl enc -base64)
    echo "$pin"
}

# Function to generate backup pin (for certificate authority)
generate_backup_pin() {
    local domain=$1
    local port=${2:-443}
    local ca_pin
    
    # Get the CA certificate from the chain
    ca_pin=$(openssl s_client -servername "$domain" -connect "$domain:$port" </dev/null 2>/dev/null | openssl x509 -pubkey -noout | openssl rsa -pubin -outform der 2>/dev/null | openssl dgst -sha256 -binary | openssl enc -base64)
    echo "$ca_pin"
}

# Main function
main() {
    log "Generating certificate pins for $DOMAIN..."
    echo ""
    
    # Generate pin from local certificate
    if [ -f "$CERT_FILE" ]; then
        local cert_pin
        cert_pin=$(generate_pin_from_cert "$CERT_FILE")
        info "Local certificate pin: $cert_pin"
    else
        warn "Local certificate file not found: $CERT_FILE"
    fi
    
    # Generate pin from local private key
    if [ -f "$KEY_FILE" ]; then
        local key_pin
        key_pin=$(generate_pin_from_key "$KEY_FILE")
        info "Local private key pin: $key_pin"
    else
        warn "Local private key file not found: $KEY_FILE"
    fi
    
    # Generate pin from remote certificate
    log "Fetching remote certificate..."
    if command -v nc &> /dev/null && nc -z "$DOMAIN" 443 2>/dev/null; then
        local remote_pin
        remote_pin=$(generate_pin_from_remote "$DOMAIN")
        info "Remote certificate pin: $remote_pin"
    else
        warn "Cannot connect to $DOMAIN:443. Remote pin generation skipped."
    fi
    
    echo ""
    log "Certificate pinning configuration example:"
    echo ""
    
    # Generate nginx configuration example
    echo "# Add to nginx server block:"
    echo "add_header Public-Key-Pins 'pin-sha256=\"$cert_pin\"; max-age=2592000; includeSubDomains' always;"
    echo ""
    
    # Generate Apache configuration example
    echo "# Add to Apache virtual host:"
    echo "Header always set Public-Key-Pins 'pin-sha256=\"$cert_pin\"; max-age=2592000; includeSubDomains'"
    echo ""
    
    # Generate security considerations
    warn "IMPORTANT SECURITY CONSIDERATIONS:"
    echo "1. Certificate pinning can lock users out if implemented incorrectly"
    echo "2. Always include a backup pin for your CA or next certificate"
    echo "3. Test thoroughly in staging environment before production deployment"
    echo "4. Monitor certificate expiration dates closely"
    echo "5. Have a rollback plan in case of issues"
    echo ""
    
    # Generate backup pin recommendations
    log "Recommended backup pin strategies:"
    echo "1. Pin your CA's public key as a backup"
    echo "2. Pre-generate pins for your next certificate"
    echo "3. Use a shorter max-age during testing (e.g., 300 seconds)"
    echo "4. Consider using Report-Only mode first"
    echo ""
    
    # Generate monitoring recommendations
    log "Monitoring recommendations:"
    echo "1. Monitor for HPKP violation reports"
    echo "2. Set up alerts for certificate expiration"
    echo "3. Test pin validation regularly"
    echo "4. Keep track of all pinned certificates"
    echo ""
    
    log "Certificate pinning setup completed!"
    log "Remember to test thoroughly before enabling in production!"
}

# Show usage if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Environment variables:"
    echo "  DOMAIN      - Domain name (required)"
    echo "  CERT_FILE   - Path to certificate file (default: ./nginx/ssl/cert.pem)"
    echo "  KEY_FILE    - Path to private key file (default: ./nginx/ssl/key.pem)"
    echo ""
    echo "Examples:"
    echo "  DOMAIN=example.com $0"
    echo "  DOMAIN=example.com CERT_FILE=/path/to/cert.pem $0"
    echo ""
    exit 1
fi

# Run main function
main "$@"