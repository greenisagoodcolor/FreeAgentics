#!/bin/bash

# Generate DH Parameters Script
# This script generates Diffie-Hellman parameters for Perfect Forward Secrecy

set -euo pipefail

# Configuration
DH_SIZE=${DH_SIZE:-"2048"}
OUTPUT_FILE=${OUTPUT_FILE:-"./nginx/dhparam.pem"}

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

# Check if OpenSSL is available
if ! command -v openssl &> /dev/null; then
    error "OpenSSL is not installed. Please install OpenSSL to continue."
fi

# Create directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Check if DH parameters already exist
if [ -f "$OUTPUT_FILE" ]; then
    warn "DH parameters file already exists: $OUTPUT_FILE"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Aborted. Using existing DH parameters."
        exit 0
    fi
fi

# Generate DH parameters
log "Generating $DH_SIZE-bit DH parameters..."
log "This may take several minutes depending on your system..."

# Show progress
if [ "$DH_SIZE" -eq 2048 ]; then
    log "Estimated time: 1-5 minutes"
elif [ "$DH_SIZE" -eq 4096 ]; then
    log "Estimated time: 10-30 minutes"
    warn "4096-bit DH parameters provide better security but take much longer to generate"
fi

# Generate the parameters
if openssl dhparam -out "$OUTPUT_FILE" "$DH_SIZE"; then
    log "DH parameters generated successfully!"
    
    # Show file information
    log "File location: $OUTPUT_FILE"
    log "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    log "File permissions: $(ls -l "$OUTPUT_FILE" | cut -d' ' -f1)"
    
    # Set proper permissions
    chmod 644 "$OUTPUT_FILE"
    log "Permissions set to 644"
    
    # Verify the generated parameters
    if openssl dhparam -in "$OUTPUT_FILE" -text -noout > /dev/null 2>&1; then
        log "DH parameters verification: PASSED"
    else
        error "DH parameters verification: FAILED"
    fi
    
    log "DH parameters are ready for use in nginx configuration"
else
    error "Failed to generate DH parameters"
fi