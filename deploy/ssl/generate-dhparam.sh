#!/bin/bash
# Generate DH parameters for strong SSL/TLS configuration
# Required for A+ SSL Labs rating

set -e

DH_SIZE=${1:-4096}
OUTPUT_FILE=${2:-/etc/ssl/dhparam.pem}

echo "Generating ${DH_SIZE}-bit DH parameters..."
echo "This may take several minutes on slower systems."

# Check if openssl is installed
if ! command -v openssl &> /dev/null; then
    echo "Error: OpenSSL is not installed"
    exit 1
fi

# Generate DH parameters
openssl dhparam -out "${OUTPUT_FILE}" "${DH_SIZE}"

# Set appropriate permissions
chmod 644 "${OUTPUT_FILE}"

echo "DH parameters generated successfully at: ${OUTPUT_FILE}"
echo ""
echo "To use in Nginx, add this line to your SSL configuration:"
echo "ssl_dhparam ${OUTPUT_FILE};"