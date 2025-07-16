#!/bin/bash
# Test SSL/TLS configuration for security best practices
# Helps achieve A+ rating on SSL Labs

set -e

DOMAIN=${1:-freeagentics.com}
PORT=${2:-443}

echo "Testing SSL/TLS configuration for ${DOMAIN}:${PORT}"
echo "=================================================="

# Function to print test results
print_result() {
    local test_name=$1
    local result=$2
    local details=$3
    
    if [ "$result" = "PASS" ]; then
        echo "✓ ${test_name}: PASS"
    else
        echo "✗ ${test_name}: FAIL - ${details}"
    fi
}

# Test 1: Check TLS versions
echo ""
echo "1. Testing TLS versions..."
echo "--------------------------"

# Test TLS 1.3
if openssl s_client -connect "${DOMAIN}:${PORT}" -tls1_3 < /dev/null 2>/dev/null | grep -q "TLSv1.3"; then
    print_result "TLS 1.3" "PASS"
else
    print_result "TLS 1.3" "FAIL" "TLS 1.3 not supported"
fi

# Test TLS 1.2
if openssl s_client -connect "${DOMAIN}:${PORT}" -tls1_2 < /dev/null 2>/dev/null | grep -q "TLSv1.2"; then
    print_result "TLS 1.2" "PASS"
else
    print_result "TLS 1.2" "FAIL" "TLS 1.2 not supported"
fi

# Test TLS 1.1 (should fail)
if openssl s_client -connect "${DOMAIN}:${PORT}" -tls1_1 < /dev/null 2>/dev/null | grep -q "TLSv1.1"; then
    print_result "TLS 1.1 disabled" "FAIL" "TLS 1.1 is enabled (insecure)"
else
    print_result "TLS 1.1 disabled" "PASS"
fi

# Test TLS 1.0 (should fail)
if openssl s_client -connect "${DOMAIN}:${PORT}" -tls1 < /dev/null 2>/dev/null | grep -q "TLSv1"; then
    print_result "TLS 1.0 disabled" "FAIL" "TLS 1.0 is enabled (insecure)"
else
    print_result "TLS 1.0 disabled" "PASS"
fi

# Test 2: Check cipher suites
echo ""
echo "2. Testing cipher suites..."
echo "---------------------------"

# Get supported ciphers
CIPHERS=$(openssl s_client -connect "${DOMAIN}:${PORT}" -cipher 'ALL' < /dev/null 2>/dev/null | grep "Cipher" | cut -d: -f2 | tr -d ' ')

# Check for strong ciphers
STRONG_CIPHERS=("ECDHE-ECDSA-AES256-GCM-SHA384" "ECDHE-RSA-AES256-GCM-SHA384" "ECDHE-ECDSA-CHACHA20-POLY1305" "ECDHE-RSA-CHACHA20-POLY1305")
for cipher in "${STRONG_CIPHERS[@]}"; do
    if openssl s_client -connect "${DOMAIN}:${PORT}" -cipher "${cipher}" < /dev/null 2>/dev/null | grep -q "Cipher.*${cipher}"; then
        print_result "Strong cipher ${cipher}" "PASS"
    fi
done

# Test 3: Check OCSP stapling
echo ""
echo "3. Testing OCSP stapling..."
echo "---------------------------"

if openssl s_client -connect "${DOMAIN}:${PORT}" -status < /dev/null 2>/dev/null | grep -q "OCSP Response Status: successful"; then
    print_result "OCSP stapling" "PASS"
else
    print_result "OCSP stapling" "FAIL" "OCSP stapling not enabled"
fi

# Test 4: Check security headers
echo ""
echo "4. Testing security headers..."
echo "------------------------------"

# Get headers
HEADERS=$(curl -sI "https://${DOMAIN}:${PORT}")

# Check HSTS
if echo "$HEADERS" | grep -qi "Strict-Transport-Security.*max-age=.*includeSubDomains.*preload"; then
    print_result "HSTS with preload" "PASS"
else
    print_result "HSTS with preload" "FAIL" "HSTS header missing or incomplete"
fi

# Check other security headers
SECURITY_HEADERS=(
    "X-Frame-Options: DENY"
    "X-Content-Type-Options: nosniff"
    "X-XSS-Protection: 1; mode=block"
    "Referrer-Policy: strict-origin-when-cross-origin"
    "Content-Security-Policy"
    "Permissions-Policy"
)

for header in "${SECURITY_HEADERS[@]}"; do
    header_name=$(echo "$header" | cut -d: -f1)
    if echo "$HEADERS" | grep -qi "$header_name"; then
        print_result "${header_name}" "PASS"
    else
        print_result "${header_name}" "FAIL" "Header not found"
    fi
done

# Test 5: Certificate validation
echo ""
echo "5. Testing certificate..."
echo "-------------------------"

# Check certificate validity
CERT_INFO=$(openssl s_client -connect "${DOMAIN}:${PORT}" < /dev/null 2>/dev/null | openssl x509 -noout -dates 2>/dev/null)

if [ -n "$CERT_INFO" ]; then
    print_result "Certificate valid" "PASS"
    echo "  ${CERT_INFO}"
else
    print_result "Certificate valid" "FAIL" "Could not retrieve certificate"
fi

# Test 6: Check for vulnerabilities
echo ""
echo "6. Testing for known vulnerabilities..."
echo "---------------------------------------"

# Test for BEAST
if openssl s_client -connect "${DOMAIN}:${PORT}" -cipher 'DES-CBC3-SHA' < /dev/null 2>/dev/null | grep -q "Cipher"; then
    print_result "BEAST mitigation" "FAIL" "Vulnerable ciphers enabled"
else
    print_result "BEAST mitigation" "PASS"
fi

# Test for POODLE
if openssl s_client -connect "${DOMAIN}:${PORT}" -ssl3 < /dev/null 2>/dev/null | grep -q "SSLv3"; then
    print_result "POODLE mitigation" "FAIL" "SSLv3 enabled"
else
    print_result "POODLE mitigation" "PASS"
fi

echo ""
echo "=================================================="
echo "SSL/TLS configuration test complete!"
echo ""
echo "For a comprehensive test, run SSL Labs test at:"
echo "https://www.ssllabs.com/ssltest/analyze.html?d=${DOMAIN}"