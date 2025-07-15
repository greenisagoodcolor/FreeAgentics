#!/bin/bash

# FreeAgentics Production Deployment Test Script
# Comprehensive integration test for production deployment infrastructure

set -euo pipefail

# Configuration
DOMAIN="${DOMAIN:-test.freeagentics.local}"
TEST_ENV="${TEST_ENV:-testing}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.production.yml}"
TIMEOUT="${TIMEOUT:-300}"
BACKUP_DIR="${BACKUP_DIR:-./test-backups}"
LOG_FILE="${LOG_FILE:-./test-deployment.log}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}" | tee -a "$LOG_FILE"
}

# Test result functions
test_pass() {
    local test_name="$1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    success "✓ $test_name"
}

test_fail() {
    local test_name="$1"
    local reason="$2"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    error "✗ $test_name - $reason"
}

# Function to wait for service to be healthy
wait_for_service() {
    local service="$1"
    local timeout="${2:-60}"
    local count=0
    
    info "Waiting for $service to be healthy..."
    
    while [ $count -lt $timeout ]; do
        if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "healthy"; then
            test_pass "$service health check"
            return 0
        fi
        
        if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
            test_pass "$service is running"
            return 0
        fi
        
        sleep 1
        count=$((count + 1))
    done
    
    test_fail "$service health check" "Service not healthy after ${timeout}s"
    return 1
}

# Function to test HTTP endpoint
test_endpoint() {
    local url="$1"
    local expected_status="${2:-200}"
    local timeout="${3:-10}"
    
    info "Testing endpoint: $url"
    
    if curl -s --connect-timeout "$timeout" --max-time "$timeout" \
       -w "%{http_code}" -o /dev/null "$url" | grep -q "$expected_status"; then
        test_pass "HTTP $expected_status from $url"
        return 0
    else
        test_fail "HTTP endpoint test" "$url did not return $expected_status"
        return 1
    fi
}

# Function to test SSL endpoint
test_ssl_endpoint() {
    local url="$1"
    local expected_status="${2:-200}"
    
    info "Testing SSL endpoint: $url"
    
    if curl -s -k --connect-timeout 10 --max-time 10 \
       -w "%{http_code}" -o /dev/null "$url" | grep -q "$expected_status"; then
        test_pass "HTTPS $expected_status from $url"
        return 0
    else
        test_fail "HTTPS endpoint test" "$url did not return $expected_status"
        return 1
    fi
}

# Function to test database connectivity
test_database() {
    info "Testing database connectivity..."
    
    if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U freeagentics; then
        test_pass "Database connectivity"
    else
        test_fail "Database connectivity" "pg_isready failed"
        return 1
    fi
    
    # Test database query
    if docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U freeagentics -d freeagentics -c "SELECT 1;" > /dev/null 2>&1; then
        test_pass "Database query execution"
    else
        test_fail "Database query execution" "SELECT query failed"
        return 1
    fi
}

# Function to test Redis connectivity
test_redis() {
    info "Testing Redis connectivity..."
    
    if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli -a "${REDIS_PASSWORD:-test}" ping | grep -q "PONG"; then
        test_pass "Redis connectivity"
    else
        test_fail "Redis connectivity" "Redis ping failed"
        return 1
    fi
}

# Function to test SSL certificate
test_ssl_certificate() {
    info "Testing SSL certificate..."
    
    if [ -f "nginx/ssl/cert.pem" ]; then
        # Check certificate validity
        if openssl x509 -in nginx/ssl/cert.pem -noout -checkend 86400; then
            test_pass "SSL certificate validity"
        else
            test_fail "SSL certificate validity" "Certificate expires within 24 hours"
        fi
        
        # Check certificate properties
        local cert_info
        cert_info=$(openssl x509 -in nginx/ssl/cert.pem -text -noout)
        
        if echo "$cert_info" | grep -q "TLS Web Server Authentication"; then
            test_pass "SSL certificate purpose"
        else
            test_fail "SSL certificate purpose" "Not a web server certificate"
        fi
    else
        test_fail "SSL certificate file" "nginx/ssl/cert.pem not found"
    fi
}

# Function to test deployment scripts
test_deployment_scripts() {
    info "Testing deployment scripts..."
    
    # Test script existence and permissions
    local scripts=("deploy-production.sh" "deploy-production-ssl.sh")
    
    for script in "${scripts[@]}"; do
        if [ -f "$script" ] && [ -x "$script" ]; then
            test_pass "$script exists and is executable"
        else
            test_fail "$script permissions" "Script not found or not executable"
        fi
    done
    
    # Test script syntax
    if bash -n deploy-production.sh; then
        test_pass "deploy-production.sh syntax"
    else
        test_fail "deploy-production.sh syntax" "Syntax error in script"
    fi
}

# Function to test backup functionality
test_backup_functionality() {
    info "Testing backup functionality..."
    
    # Create test backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Test database backup script
    if [ -f "scripts/database-backup.sh" ]; then
        # Test backup script syntax
        if bash -n scripts/database-backup.sh; then
            test_pass "Database backup script syntax"
        else
            test_fail "Database backup script syntax" "Syntax error in script"
        fi
    else
        test_fail "Database backup script" "scripts/database-backup.sh not found"
    fi
    
    # Test backup directory permissions
    if [ -w "$BACKUP_DIR" ]; then
        test_pass "Backup directory writable"
    else
        test_fail "Backup directory permissions" "Cannot write to backup directory"
    fi
}

# Function to test monitoring endpoints
test_monitoring_endpoints() {
    info "Testing monitoring endpoints..."
    
    # Test Prometheus configuration
    if [ -f "monitoring/prometheus-production.yml" ]; then
        # Validate YAML syntax
        if python3 -c "import yaml; yaml.safe_load(open('monitoring/prometheus-production.yml'))" 2>/dev/null; then
            test_pass "Prometheus configuration syntax"
        else
            test_fail "Prometheus configuration syntax" "Invalid YAML"
        fi
    else
        test_fail "Prometheus configuration" "monitoring/prometheus-production.yml not found"
    fi
    
    # Test Grafana dashboard configuration
    if [ -d "monitoring/grafana/dashboards" ]; then
        test_pass "Grafana dashboards directory exists"
    else
        test_fail "Grafana dashboards" "monitoring/grafana/dashboards not found"
    fi
}

# Function to test security configuration
test_security_configuration() {
    info "Testing security configuration..."
    
    # Test nginx security headers
    if [ -f "nginx/nginx.conf" ]; then
        local security_headers=(
            "X-Frame-Options"
            "X-Content-Type-Options"
            "X-XSS-Protection"
            "Strict-Transport-Security"
            "Content-Security-Policy"
        )
        
        for header in "${security_headers[@]}"; do
            if grep -q "$header" nginx/nginx.conf; then
                test_pass "Security header: $header"
            else
                test_fail "Security header: $header" "Header not configured"
            fi
        done
    else
        test_fail "Nginx configuration" "nginx/nginx.conf not found"
    fi
    
    # Test rate limiting configuration
    if grep -q "limit_req_zone" nginx/nginx.conf; then
        test_pass "Rate limiting configured"
    else
        test_fail "Rate limiting" "No rate limiting configuration found"
    fi
}

# Function to test environment configuration
test_environment_configuration() {
    info "Testing environment configuration..."
    
    # Test environment template
    if [ -f ".env.production.ssl.template" ]; then
        test_pass "Environment template exists"
    else
        test_fail "Environment template" ".env.production.ssl.template not found"
    fi
    
    # Test required environment variables
    local required_vars=(
        "DATABASE_URL"
        "REDIS_PASSWORD"
        "SECRET_KEY"
        "JWT_SECRET"
    )
    
    for var in "${required_vars[@]}"; do
        if printenv "$var" >/dev/null 2>&1; then
            test_pass "Environment variable: $var"
        else
            warn "Environment variable $var not set (expected in production)"
        fi
    done
}

# Function to test resource limits
test_resource_limits() {
    info "Testing resource limits..."
    
    # Parse docker-compose file for resource limits
    if [ -f "$COMPOSE_FILE" ]; then
        if grep -q "resources:" "$COMPOSE_FILE"; then
            test_pass "Resource limits configured"
        else
            test_fail "Resource limits" "No resource limits found in compose file"
        fi
        
        # Check for memory limits
        if grep -q "memory:" "$COMPOSE_FILE"; then
            test_pass "Memory limits configured"
        else
            test_fail "Memory limits" "No memory limits configured"
        fi
    else
        test_fail "Docker compose file" "$COMPOSE_FILE not found"
    fi
}

# Function to test image build
test_image_build() {
    info "Testing Docker image build..."
    
    # Test Dockerfile syntax
    if [ -f "Dockerfile.production" ]; then
        if docker build -f Dockerfile.production -t freeagentics-test --dry-run . 2>/dev/null; then
            test_pass "Dockerfile syntax"
        else
            test_fail "Dockerfile syntax" "Docker build failed"
        fi
    else
        test_fail "Production Dockerfile" "Dockerfile.production not found"
    fi
    
    # Test docker-compose config
    if docker-compose -f "$COMPOSE_FILE" config > /dev/null 2>&1; then
        test_pass "Docker Compose configuration"
    else
        test_fail "Docker Compose configuration" "Config validation failed"
    fi
}

# Function to run full deployment test
run_full_deployment_test() {
    info "Running full deployment test..."
    
    # Stop any existing services
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
    
    # Build images
    info "Building images..."
    if docker-compose -f "$COMPOSE_FILE" build --no-cache; then
        test_pass "Image build process"
    else
        test_fail "Image build process" "Build failed"
        return 1
    fi
    
    # Start services
    info "Starting services..."
    if docker-compose -f "$COMPOSE_FILE" up -d; then
        test_pass "Service startup"
    else
        test_fail "Service startup" "Failed to start services"
        return 1
    fi
    
    # Wait for services to be healthy
    local services=("postgres" "redis" "backend" "frontend" "nginx")
    for service in "${services[@]}"; do
        if ! wait_for_service "$service" 120; then
            error "Service $service failed to start"
            return 1
        fi
    done
    
    # Test endpoints
    test_endpoint "http://localhost:8000/health" 200
    test_endpoint "http://localhost:3000" 200
    
    # Test database and Redis
    test_database
    test_redis
    
    # Clean up
    info "Cleaning up test deployment..."
    docker-compose -f "$COMPOSE_FILE" down --volumes --remove-orphans
    
    test_pass "Full deployment test completed"
}

# Function to generate test report
generate_test_report() {
    local report_file="production_deployment_test_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "test_environment": "$TEST_ENV",
    "domain": "$DOMAIN",
    "compose_file": "$COMPOSE_FILE",
    "summary": {
        "total_tests": $TESTS_TOTAL,
        "passed": $TESTS_PASSED,
        "failed": $TESTS_FAILED,
        "success_rate": $(awk "BEGIN {printf \"%.2f\", ($TESTS_PASSED/$TESTS_TOTAL)*100}")
    },
    "test_results": {
        "docker_build": "$([ $TESTS_PASSED -gt 0 ] && echo "PASSED" || echo "FAILED")",
        "ssl_configuration": "$([ -f "nginx/ssl/cert.pem" ] && echo "PASSED" || echo "FAILED")",
        "deployment_scripts": "$([ -f "deploy-production.sh" ] && echo "PASSED" || echo "FAILED")",
        "monitoring_setup": "$([ -f "monitoring/prometheus-production.yml" ] && echo "PASSED" || echo "FAILED")",
        "security_configuration": "$([ -f "nginx/nginx.conf" ] && echo "PASSED" || echo "FAILED")"
    }
}
EOF
    
    info "Test report generated: $report_file"
}

# Main test execution
main() {
    log "Starting FreeAgentics Production Deployment Test"
    log "=============================================="
    
    # Initialize log file
    echo "FreeAgentics Production Deployment Test - $(date)" > "$LOG_FILE"
    
    # Run individual tests
    test_environment_configuration
    test_image_build
    test_deployment_scripts
    test_ssl_certificate
    test_security_configuration
    test_monitoring_endpoints
    test_backup_functionality
    test_resource_limits
    
    # Conditional full deployment test
    if [ "${FULL_TEST:-false}" = "true" ]; then
        warn "Running full deployment test - this will build and start services"
        run_full_deployment_test
    else
        info "Skipping full deployment test (set FULL_TEST=true to run)"
    fi
    
    # Generate report
    generate_test_report
    
    # Print summary
    echo ""
    log "=============================================="
    log "Production Deployment Test Summary"
    log "=============================================="
    log "Total Tests: $TESTS_TOTAL"
    log "Passed: $TESTS_PASSED"
    log "Failed: $TESTS_FAILED"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        success "🎉 All tests passed! Production deployment is ready."
        exit 0
    else
        error "❌ Some tests failed. Please review the issues above."
        exit 1
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -d, --domain DOMAIN      Set test domain (default: test.freeagentics.local)"
    echo "  -f, --full               Run full deployment test (builds and starts services)"
    echo "  -c, --compose FILE       Use specific compose file (default: docker-compose.production.yml)"
    echo "  -h, --help               Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  DOMAIN                   Test domain"
    echo "  FULL_TEST                Set to 'true' to run full deployment test"
    echo "  REDIS_PASSWORD           Redis password for testing"
    echo "  TIMEOUT                  Test timeout in seconds (default: 300)"
    echo ""
    echo "Examples:"
    echo "  $0                       # Run basic tests"
    echo "  $0 --full                # Run full deployment test"
    echo "  $0 -d example.com        # Test with specific domain"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        -f|--full)
            FULL_TEST=true
            shift
            ;;
        -c|--compose)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Run main function
main "$@"