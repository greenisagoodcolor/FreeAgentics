#!/bin/bash

# SSL/TLS Testing Script for FreeAgentics
# This script tests the SSL configuration for security and performance

set -euo pipefail

# Configuration
DOMAIN=${DOMAIN:-"yourdomain.com"}
PORT=${PORT:-"443"}
TIMEOUT=${TIMEOUT:-"10"}

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
}

info() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Check if domain is provided
if [ "$DOMAIN" = "yourdomain.com" ]; then
    error "Please set DOMAIN environment variable to your actual domain"
fi

# Check if required tools are installed
check_tools() {
    local tools=("openssl" "curl" "nmap")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            warn "$tool is not installed. Some tests may be skipped."
        fi
    done
}

# Test SSL certificate
test_certificate() {
    info "Testing SSL certificate for $DOMAIN..."
    
    if command -v openssl &> /dev/null; then
        # Get certificate information
        local cert_info
        cert_info=$(openssl s_client -connect "$DOMAIN:$PORT" -servername "$DOMAIN" </dev/null 2>/dev/null | openssl x509 -noout -text 2>/dev/null)
        
        if [ $? -eq 0 ]; then
            success "SSL certificate is valid"
            
            # Check certificate expiration
            local expiry_date
            expiry_date=$(echo "$cert_info" | grep "Not After" | cut -d':' -f2- | xargs)
            info "Certificate expires: $expiry_date"
            
            # Check if certificate is expiring soon (within 30 days)
            local expiry_seconds
            expiry_seconds=$(date -d "$expiry_date" +%s 2>/dev/null)
            local current_seconds
            current_seconds=$(date +%s)
            local days_until_expiry
            days_until_expiry=$(((expiry_seconds - current_seconds) / 86400))
            
            if [ "$days_until_expiry" -lt 30 ]; then
                warn "Certificate expires in $days_until_expiry days"
            else
                success "Certificate is valid for $days_until_expiry days"
            fi
            
            # Check subject alternative names
            local san
            san=$(echo "$cert_info" | grep -A1 "Subject Alternative Name" | tail -n1 | sed 's/DNS://g' | sed 's/,//g')
            if [ -n "$san" ]; then
                info "Subject Alternative Names: $san"
            fi
        else
            fail "SSL certificate is invalid or not accessible"
        fi
    else
        warn "OpenSSL not available, skipping certificate tests"
    fi
}

# Test SSL protocols
test_protocols() {
    info "Testing SSL/TLS protocols..."
    
    if command -v openssl &> /dev/null; then
        # Test TLS 1.2
        if openssl s_client -connect "$DOMAIN:$PORT" -tls1_2 </dev/null 2>/dev/null | grep -q "Verify return code: 0"; then
            success "TLS 1.2 is supported"
        else
            warn "TLS 1.2 is not supported or has issues"
        fi
        
        # Test TLS 1.3
        if openssl s_client -connect "$DOMAIN:$PORT" -tls1_3 </dev/null 2>/dev/null | grep -q "Verify return code: 0"; then
            success "TLS 1.3 is supported"
        else
            warn "TLS 1.3 is not supported"
        fi
        
        # Test deprecated protocols (should fail)
        if openssl s_client -connect "$DOMAIN:$PORT" -ssl3 </dev/null 2>/dev/null | grep -q "Verify return code: 0"; then
            fail "SSL 3.0 is supported (security risk)"
        else
            success "SSL 3.0 is disabled"
        fi
        
        if openssl s_client -connect "$DOMAIN:$PORT" -tls1 </dev/null 2>/dev/null | grep -q "Verify return code: 0"; then
            fail "TLS 1.0 is supported (security risk)"
        else
            success "TLS 1.0 is disabled"
        fi
        
        if openssl s_client -connect "$DOMAIN:$PORT" -tls1_1 </dev/null 2>/dev/null | grep -q "Verify return code: 0"; then
            fail "TLS 1.1 is supported (security risk)"
        else
            success "TLS 1.1 is disabled"
        fi
    else
        warn "OpenSSL not available, skipping protocol tests"
    fi
}

# Test cipher suites
test_ciphers() {
    info "Testing cipher suites..."
    
    if command -v openssl &> /dev/null; then
        # Get supported ciphers
        local ciphers
        ciphers=$(openssl s_client -connect "$DOMAIN:$PORT" -cipher 'ALL:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA' </dev/null 2>/dev/null | grep "Cipher" | cut -d':' -f2 | xargs)
        
        if [ -n "$ciphers" ]; then
            info "Supported cipher: $ciphers"
            
            # Check for weak ciphers
            if echo "$ciphers" | grep -q "RC4"; then
                fail "RC4 cipher is supported (security risk)"
            else
                success "RC4 cipher is disabled"
            fi
            
            if echo "$ciphers" | grep -q "DES"; then
                fail "DES cipher is supported (security risk)"
            else
                success "DES cipher is disabled"
            fi
            
            if echo "$ciphers" | grep -q "MD5"; then
                fail "MD5 cipher is supported (security risk)"
            else
                success "MD5 cipher is disabled"
            fi
        else
            warn "Could not determine supported ciphers"
        fi
    else
        warn "OpenSSL not available, skipping cipher tests"
    fi
}

# Test security headers
test_headers() {
    info "Testing security headers..."
    
    if command -v curl &> /dev/null; then
        local headers
        headers=$(curl -I -s --connect-timeout "$TIMEOUT" "https://$DOMAIN" 2>/dev/null)
        
        if [ $? -eq 0 ]; then
            # Check HSTS
            if echo "$headers" | grep -qi "strict-transport-security"; then
                success "HSTS header is present"
            else
                fail "HSTS header is missing"
            fi
            
            # Check X-Frame-Options
            if echo "$headers" | grep -qi "x-frame-options"; then
                success "X-Frame-Options header is present"
            else
                fail "X-Frame-Options header is missing"
            fi
            
            # Check X-Content-Type-Options
            if echo "$headers" | grep -qi "x-content-type-options"; then
                success "X-Content-Type-Options header is present"
            else
                fail "X-Content-Type-Options header is missing"
            fi
            
            # Check X-XSS-Protection
            if echo "$headers" | grep -qi "x-xss-protection"; then
                success "X-XSS-Protection header is present"
            else
                fail "X-XSS-Protection header is missing"
            fi
            
            # Check CSP
            if echo "$headers" | grep -qi "content-security-policy"; then
                success "Content-Security-Policy header is present"
            else
                fail "Content-Security-Policy header is missing"
            fi
        else
            warn "Could not retrieve headers from $DOMAIN"
        fi
    else
        warn "curl not available, skipping header tests"
    fi
}

# Test HTTP to HTTPS redirect
test_redirect() {
    info "Testing HTTP to HTTPS redirect..."
    
    if command -v curl &> /dev/null; then
        local redirect_response
        redirect_response=$(curl -I -s --connect-timeout "$TIMEOUT" "http://$DOMAIN" 2>/dev/null | head -n1)
        
        if echo "$redirect_response" | grep -q "301\|302"; then
            success "HTTP to HTTPS redirect is working"
        else
            fail "HTTP to HTTPS redirect is not working"
        fi
    else
        warn "curl not available, skipping redirect test"
    fi
}

# Test OCSP stapling
test_ocsp() {
    info "Testing OCSP stapling..."
    
    if command -v openssl &> /dev/null; then
        local ocsp_response
        ocsp_response=$(openssl s_client -connect "$DOMAIN:$PORT" -servername "$DOMAIN" -status </dev/null 2>/dev/null | grep "OCSP Response Status")
        
        if echo "$ocsp_response" | grep -q "successful"; then
            success "OCSP stapling is working"
        else
            warn "OCSP stapling is not working or not configured"
        fi
    else
        warn "OpenSSL not available, skipping OCSP test"
    fi
}

# Test SSL Labs rating (requires ssllabs-scan tool)
test_ssllabs() {
    info "Testing SSL Labs rating..."
    
    if command -v ssllabs-scan &> /dev/null; then
        local rating
        rating=$(ssllabs-scan --quiet --usecache "$DOMAIN" | grep "Grade" | cut -d':' -f2 | xargs)
        
        if [ -n "$rating" ]; then
            if [ "$rating" = "A+" ]; then
                success "SSL Labs rating: $rating"
            elif [ "$rating" = "A" ]; then
                success "SSL Labs rating: $rating"
            else
                warn "SSL Labs rating: $rating (consider improvements)"
            fi
        else
            warn "Could not determine SSL Labs rating"
        fi
    else
        warn "ssllabs-scan not available, skipping SSL Labs test"
        info "You can manually test at: https://www.ssllabs.com/ssltest/analyze.html?d=$DOMAIN"
    fi
}

# Test port scan for common vulnerabilities
test_ports() {
    info "Testing port security..."
    
    if command -v nmap &> /dev/null; then
        local open_ports
        open_ports=$(nmap -p 80,443,8080,8443 "$DOMAIN" 2>/dev/null | grep "open" | cut -d'/' -f1)
        
        if echo "$open_ports" | grep -q "80"; then
            success "Port 80 is open (expected for HTTP redirect)"
        fi
        
        if echo "$open_ports" | grep -q "443"; then
            success "Port 443 is open (expected for HTTPS)"
        fi
        
        if echo "$open_ports" | grep -q "8080\|8443"; then
            warn "Additional ports are open: $open_ports"
        fi
    else
        warn "nmap not available, skipping port scan"
    fi
}

# Main test execution
main() {
    log "Starting SSL/TLS security tests for $DOMAIN..."
    echo ""
    
    check_tools
    
    test_certificate
    echo ""
    
    test_protocols
    echo ""
    
    test_ciphers
    echo ""
    
    test_headers
    echo ""
    
    test_redirect
    echo ""
    
    test_ocsp
    echo ""
    
    test_ssllabs
    echo ""
    
    test_ports
    echo ""
    
    log "SSL/TLS testing completed!"
    log "For a comprehensive analysis, visit: https://www.ssllabs.com/ssltest/analyze.html?d=$DOMAIN"
}

# Run main function
main "$@"