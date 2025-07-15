#!/bin/bash

# SSL Certificate Monitoring Script
# This script monitors SSL certificates for expiration and security issues

set -euo pipefail

# Configuration
DOMAIN=${DOMAIN:-"yourdomain.com"}
PORT=${PORT:-"443"}
WARNING_DAYS=${WARNING_DAYS:-"30"}
CRITICAL_DAYS=${CRITICAL_DAYS:-"7"}
SLACK_WEBHOOK=${SLACK_WEBHOOK:-""}
EMAIL_TO=${EMAIL_TO:-""}
LOG_FILE=${LOG_FILE:-"/var/log/ssl-monitor.log"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${GREEN}[INFO]${NC} $message"
    echo "[$timestamp] [INFO] $message" >> "$LOG_FILE"
}

warn() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${YELLOW}[WARN]${NC} $message"
    echo "[$timestamp] [WARN] $message" >> "$LOG_FILE"
}

error() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${RED}[ERROR]${NC} $message"
    echo "[$timestamp] [ERROR] $message" >> "$LOG_FILE"
}

critical() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${RED}[CRITICAL]${NC} $message"
    echo "[$timestamp] [CRITICAL] $message" >> "$LOG_FILE"
}

success() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${GREEN}[SUCCESS]${NC} $message"
    echo "[$timestamp] [SUCCESS] $message" >> "$LOG_FILE"
}

# Check if domain is provided
if [ "$DOMAIN" = "yourdomain.com" ]; then
    error "Please set DOMAIN environment variable to your actual domain"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Function to send Slack notification
send_slack_notification() {
    local message="$1"
    local color="$2"
    
    if [ -n "$SLACK_WEBHOOK" ]; then
        local payload="{\"text\":\"SSL Monitor Alert\",\"attachments\":[{\"color\":\"$color\",\"fields\":[{\"title\":\"Domain\",\"value\":\"$DOMAIN\",\"short\":true},{\"title\":\"Message\",\"value\":\"$message\",\"short\":false}]}]}"
        
        if command -v curl &> /dev/null; then
            curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK" > /dev/null
        fi
    fi
}

# Function to send email notification
send_email_notification() {
    local subject="$1"
    local message="$2"
    
    if [ -n "$EMAIL_TO" ] && command -v mail &> /dev/null; then
        echo "$message" | mail -s "$subject" "$EMAIL_TO"
    fi
}

# Function to check certificate expiration
check_certificate_expiration() {
    log "Checking certificate expiration for $DOMAIN..."
    
    if ! command -v openssl &> /dev/null; then
        error "OpenSSL is not installed"
        return 1
    fi
    
    # Get certificate expiration date
    local cert_end_date
    cert_end_date=$(echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:$PORT" 2>/dev/null | openssl x509 -noout -enddate 2>/dev/null | cut -d= -f2)
    
    if [ -z "$cert_end_date" ]; then
        error "Could not retrieve certificate expiration date"
        send_slack_notification "Could not retrieve certificate expiration date" "danger"
        send_email_notification "SSL Certificate Error - $DOMAIN" "Could not retrieve certificate expiration date for $DOMAIN"
        return 1
    fi
    
    # Calculate days until expiration
    local end_epoch
    end_epoch=$(date -d "$cert_end_date" +%s)
    local current_epoch
    current_epoch=$(date +%s)
    local days_until_expiry
    days_until_expiry=$(( (end_epoch - current_epoch) / 86400 ))
    
    log "Certificate expires in $days_until_expiry days ($cert_end_date)"
    
    # Check expiration status
    if [ "$days_until_expiry" -le "$CRITICAL_DAYS" ]; then
        critical "Certificate expires in $days_until_expiry days - CRITICAL"
        send_slack_notification "Certificate expires in $days_until_expiry days - CRITICAL" "danger"
        send_email_notification "SSL Certificate CRITICAL - $DOMAIN" "Certificate for $DOMAIN expires in $days_until_expiry days ($cert_end_date)"
        return 2
    elif [ "$days_until_expiry" -le "$WARNING_DAYS" ]; then
        warn "Certificate expires in $days_until_expiry days - WARNING"
        send_slack_notification "Certificate expires in $days_until_expiry days - WARNING" "warning"
        send_email_notification "SSL Certificate WARNING - $DOMAIN" "Certificate for $DOMAIN expires in $days_until_expiry days ($cert_end_date)"
        return 1
    else
        success "Certificate is valid for $days_until_expiry days"
        return 0
    fi
}

# Function to check certificate chain
check_certificate_chain() {
    log "Checking certificate chain for $DOMAIN..."
    
    if ! command -v openssl &> /dev/null; then
        error "OpenSSL is not installed"
        return 1
    fi
    
    # Check certificate chain
    local chain_result
    chain_result=$(echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:$PORT" -verify 5 2>&1)
    
    if echo "$chain_result" | grep -q "Verify return code: 0"; then
        success "Certificate chain is valid"
        return 0
    else
        error "Certificate chain validation failed"
        local error_details
        error_details=$(echo "$chain_result" | grep "Verify return code:" | head -n1)
        error "Details: $error_details"
        send_slack_notification "Certificate chain validation failed: $error_details" "danger"
        send_email_notification "SSL Certificate Chain Error - $DOMAIN" "Certificate chain validation failed for $DOMAIN: $error_details"
        return 1
    fi
}

# Function to check SSL configuration
check_ssl_config() {
    log "Checking SSL configuration for $DOMAIN..."
    
    if ! command -v openssl &> /dev/null; then
        error "OpenSSL is not installed"
        return 1
    fi
    
    local issues=0
    
    # Check TLS version support
    if echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:$PORT" -tls1_2 2>/dev/null | grep -q "Protocol.*TLSv1.2"; then
        success "TLS 1.2 is supported"
    else
        warn "TLS 1.2 is not supported"
        ((issues++))
    fi
    
    if echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:$PORT" -tls1_3 2>/dev/null | grep -q "Protocol.*TLSv1.3"; then
        success "TLS 1.3 is supported"
    else
        warn "TLS 1.3 is not supported"
    fi
    
    # Check for weak protocols (should fail)
    if echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:$PORT" -ssl3 2>/dev/null | grep -q "Protocol.*SSLv3"; then
        error "SSL 3.0 is supported (security risk)"
        ((issues++))
    fi
    
    if echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:$PORT" -tls1 2>/dev/null | grep -q "Protocol.*TLSv1"; then
        error "TLS 1.0 is supported (security risk)"
        ((issues++))
    fi
    
    if echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:$PORT" -tls1_1 2>/dev/null | grep -q "Protocol.*TLSv1.1"; then
        error "TLS 1.1 is supported (security risk)"
        ((issues++))
    fi
    
    if [ $issues -gt 0 ]; then
        warn "Found $issues SSL configuration issues"
        send_slack_notification "Found $issues SSL configuration issues" "warning"
        return 1
    else
        success "SSL configuration looks good"
        return 0
    fi
}

# Function to check OCSP stapling
check_ocsp_stapling() {
    log "Checking OCSP stapling for $DOMAIN..."
    
    if ! command -v openssl &> /dev/null; then
        error "OpenSSL is not installed"
        return 1
    fi
    
    local ocsp_response
    ocsp_response=$(echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:$PORT" -status 2>/dev/null | grep "OCSP Response Status")
    
    if echo "$ocsp_response" | grep -q "successful"; then
        success "OCSP stapling is working"
        return 0
    else
        warn "OCSP stapling is not working or not configured"
        return 1
    fi
}

# Function to check security headers
check_security_headers() {
    log "Checking security headers for $DOMAIN..."
    
    if ! command -v curl &> /dev/null; then
        error "curl is not installed"
        return 1
    fi
    
    local headers
    headers=$(curl -I -s --connect-timeout 10 "https://$DOMAIN" 2>/dev/null)
    local issues=0
    
    if echo "$headers" | grep -qi "strict-transport-security"; then
        success "HSTS header is present"
    else
        warn "HSTS header is missing"
        ((issues++))
    fi
    
    if echo "$headers" | grep -qi "x-frame-options"; then
        success "X-Frame-Options header is present"
    else
        warn "X-Frame-Options header is missing"
        ((issues++))
    fi
    
    if echo "$headers" | grep -qi "x-content-type-options"; then
        success "X-Content-Type-Options header is present"
    else
        warn "X-Content-Type-Options header is missing"
        ((issues++))
    fi
    
    if echo "$headers" | grep -qi "content-security-policy"; then
        success "Content-Security-Policy header is present"
    else
        warn "Content-Security-Policy header is missing"
        ((issues++))
    fi
    
    if [ $issues -gt 0 ]; then
        warn "Found $issues missing security headers"
        return 1
    else
        success "All security headers are present"
        return 0
    fi
}

# Function to run comprehensive SSL health check
run_ssl_health_check() {
    log "Starting SSL health check for $DOMAIN..."
    
    local overall_status=0
    
    # Check certificate expiration
    if ! check_certificate_expiration; then
        ((overall_status++))
    fi
    
    # Check certificate chain
    if ! check_certificate_chain; then
        ((overall_status++))
    fi
    
    # Check SSL configuration
    if ! check_ssl_config; then
        ((overall_status++))
    fi
    
    # Check OCSP stapling
    if ! check_ocsp_stapling; then
        ((overall_status++))
    fi
    
    # Check security headers
    if ! check_security_headers; then
        ((overall_status++))
    fi
    
    # Generate summary
    echo ""
    if [ $overall_status -eq 0 ]; then
        success "SSL health check passed - all tests successful"
        send_slack_notification "SSL health check passed - all tests successful" "good"
    else
        warn "SSL health check completed with $overall_status issues"
        send_slack_notification "SSL health check completed with $overall_status issues" "warning"
    fi
    
    return $overall_status
}

# Function to generate SSL report
generate_ssl_report() {
    log "Generating SSL report for $DOMAIN..."
    
    local report_file="/tmp/ssl-report-$(date +%Y%m%d-%H%M%S).txt"
    
    {
        echo "SSL Certificate Report for $DOMAIN"
        echo "Generated: $(date)"
        echo "============================================="
        echo ""
        
        # Certificate information
        echo "Certificate Information:"
        echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:$PORT" 2>/dev/null | openssl x509 -noout -text 2>/dev/null | grep -A2 -B2 "Subject:\|Issuer:\|Not Before:\|Not After:\|DNS:"
        echo ""
        
        # SSL configuration
        echo "SSL Configuration:"
        echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:$PORT" 2>/dev/null | grep -E "Protocol|Cipher|Verify return code"
        echo ""
        
        # Security headers
        echo "Security Headers:"
        curl -I -s --connect-timeout 10 "https://$DOMAIN" 2>/dev/null | grep -i "strict-transport-security\|x-frame-options\|x-content-type-options\|content-security-policy"
        echo ""
        
    } > "$report_file"
    
    log "SSL report generated: $report_file"
    
    # Send report via email if configured
    if [ -n "$EMAIL_TO" ] && command -v mail &> /dev/null; then
        mail -s "SSL Report - $DOMAIN" "$EMAIL_TO" < "$report_file"
    fi
}

# Main function
main() {
    local command=${1:-"health-check"}
    
    case "$command" in
        "health-check")
            run_ssl_health_check
            ;;
        "expiration")
            check_certificate_expiration
            ;;
        "chain")
            check_certificate_chain
            ;;
        "config")
            check_ssl_config
            ;;
        "ocsp")
            check_ocsp_stapling
            ;;
        "headers")
            check_security_headers
            ;;
        "report")
            generate_ssl_report
            ;;
        "help")
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  health-check  - Run comprehensive SSL health check (default)"
            echo "  expiration    - Check certificate expiration only"
            echo "  chain         - Check certificate chain only"
            echo "  config        - Check SSL configuration only"
            echo "  ocsp          - Check OCSP stapling only"
            echo "  headers       - Check security headers only"
            echo "  report        - Generate detailed SSL report"
            echo "  help          - Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  DOMAIN          - Domain to check (required)"
            echo "  PORT            - Port to check (default: 443)"
            echo "  WARNING_DAYS    - Days before expiration to warn (default: 30)"
            echo "  CRITICAL_DAYS   - Days before expiration to alert (default: 7)"
            echo "  SLACK_WEBHOOK   - Slack webhook URL for notifications"
            echo "  EMAIL_TO        - Email address for notifications"
            echo "  LOG_FILE        - Log file location (default: /var/log/ssl-monitor.log)"
            ;;
        *)
            error "Unknown command: $command"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"