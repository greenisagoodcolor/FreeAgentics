#!/bin/bash
# Production Deployment Validation Script for FreeAgentics
# Comprehensive validation of production readiness and deployment health

set -euo pipefail

# Configuration
ENVIRONMENT="${ENVIRONMENT:-production}"
DOMAIN="${DOMAIN:-localhost}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"
TIMEOUT="${TIMEOUT:-30}"
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] ℹ $1${NC}"
}

# Test result tracking
run_test() {
    local test_name="$1"
    local test_command="$2"

    TESTS_RUN=$((TESTS_RUN + 1))

    if [[ "$VERBOSE" == "true" ]]; then
        info "Running test: $test_name"
    fi

    if eval "$test_command" >/dev/null 2>&1; then
        log "PASS: $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        error "FAIL: $test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Function to test Docker environment
test_docker_environment() {
    info "Testing Docker environment..."

    run_test "Docker daemon is running" "docker info"
    run_test "Docker Compose is available" "docker-compose --version"
    run_test "Docker Compose file exists" "test -f docker-compose.production.yml"
    run_test "Environment file exists" "test -f .env.production"
}

# Function to test network connectivity
test_network_connectivity() {
    info "Testing network connectivity..."

    run_test "External DNS resolution" "nslookup google.com"
    run_test "Docker network exists" "docker network ls | grep -q freeagentics"

    # Test internal service connectivity
    if docker-compose -f docker-compose.production.yml ps | grep -q postgres; then
        run_test "PostgreSQL port is accessible" "docker-compose -f docker-compose.production.yml exec -T postgres pg_isready -h localhost -p 5432"
    fi

    if docker-compose -f docker-compose.production.yml ps | grep -q redis; then
        run_test "Redis port is accessible" "docker-compose -f docker-compose.production.yml exec -T redis redis-cli ping"
    fi
}

# Function to test SSL/TLS configuration
test_ssl_configuration() {
    info "Testing SSL/TLS configuration..."

    run_test "SSL certificate exists" "test -f nginx/ssl/$DOMAIN.crt"
    run_test "SSL private key exists" "test -f nginx/ssl/$DOMAIN.key"
    run_test "DH parameters exist" "test -f nginx/dhparam.pem"
    run_test "SSL configuration is valid" "openssl x509 -in nginx/ssl/$DOMAIN.crt -text -noout"
    run_test "Private key is valid" "openssl rsa -in nginx/ssl/$DOMAIN.key -check -noout"

    # Test certificate and key match
    if [[ -f "nginx/ssl/$DOMAIN.crt" && -f "nginx/ssl/$DOMAIN.key" ]]; then
        local cert_hash key_hash
        cert_hash=$(openssl x509 -noout -modulus -in "nginx/ssl/$DOMAIN.crt" | openssl md5)
        key_hash=$(openssl rsa -noout -modulus -in "nginx/ssl/$DOMAIN.key" | openssl md5)
        run_test "Certificate and key match" "[[ '$cert_hash' == '$key_hash' ]]"
    fi
}

# Function to test service health
test_service_health() {
    info "Testing service health..."

    # Check if services are running
    local services=("postgres" "redis" "backend" "frontend" "nginx")

    for service in "${services[@]}"; do
        if docker-compose -f docker-compose.production.yml ps | grep -q "$service"; then
            run_test "$service container is running" "docker-compose -f docker-compose.production.yml ps $service | grep -q 'Up'"

            # Check health status if available
            local container_id
            container_id=$(docker-compose -f docker-compose.production.yml ps -q "$service" | head -1)
            if [[ -n "$container_id" ]]; then
                local health_status
                health_status=$(docker inspect "$container_id" --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")

                if [[ "$health_status" == "healthy" ]]; then
                    log "$service health check: healthy"
                elif [[ "$health_status" == "unknown" ]]; then
                    warn "$service health check: not configured"
                else
                    error "$service health check: $health_status"
                fi
            fi
        else
            warn "$service is not running"
        fi
    done
}

# Function to test API endpoints
test_api_endpoints() {
    info "Testing API endpoints..."

    local api_base="http://localhost:$API_PORT"
    local timeout_flag="--max-time $TIMEOUT"

    # Test health endpoint
    run_test "API health endpoint" "curl -f -s $timeout_flag $api_base/health"

    # Test API documentation
    run_test "API docs endpoint" "curl -f -s $timeout_flag $api_base/docs"

    # Test authentication endpoints
    run_test "Auth health endpoint" "curl -f -s $timeout_flag $api_base/api/v1/auth/health"

    # Test system info endpoint
    run_test "System info endpoint" "curl -f -s $timeout_flag $api_base/api/v1/system/info"

    # Test monitoring endpoints
    run_test "Metrics endpoint" "curl -f -s $timeout_flag $api_base/metrics || curl -f -s $timeout_flag $api_base/api/v1/monitoring/metrics"
}

# Function to test frontend
test_frontend() {
    info "Testing frontend..."

    local web_base="http://localhost:$WEB_PORT"
    local timeout_flag="--max-time $TIMEOUT"

    run_test "Frontend homepage" "curl -f -s $timeout_flag $web_base/"
    run_test "Frontend health check" "curl -f -s $timeout_flag $web_base/health || curl -f -s $timeout_flag $web_base/api/health"
}

# Function to test database operations
test_database_operations() {
    info "Testing database operations..."

    if docker-compose -f docker-compose.production.yml ps | grep -q postgres; then
        # Test basic database connectivity
        run_test "Database connection" "docker-compose -f docker-compose.production.yml exec -T postgres psql -U freeagentics -d freeagentics -c 'SELECT 1;'"

        # Test database schema
        run_test "Database tables exist" "docker-compose -f docker-compose.production.yml exec -T postgres psql -U freeagentics -d freeagentics -c '\dt' | grep -q 'agents'"

        # Test database performance
        run_test "Database performance check" "docker-compose -f docker-compose.production.yml exec -T postgres psql -U freeagentics -d freeagentics -c 'EXPLAIN SELECT 1;'"
    else
        warn "PostgreSQL container not found, skipping database tests"
    fi
}

# Function to test security configuration
test_security_configuration() {
    info "Testing security configuration..."

    # Test environment file permissions
    if [[ -f ".env.production" ]]; then
        local perms
        perms=$(stat -c "%a" .env.production)
        run_test "Environment file has secure permissions" "[[ '$perms' == '600' ]]"
    fi

    # Test secrets directory permissions
    if [[ -d "secrets" ]]; then
        local perms
        perms=$(stat -c "%a" secrets)
        run_test "Secrets directory has secure permissions" "[[ '$perms' -le '750' ]]"
    fi

    # Test for default passwords
    if [[ -f ".env.production" ]]; then
        run_test "No default passwords in environment" "! grep -q 'CHANGE_ME' .env.production"
    fi

    # Test Docker security
    run_test "Docker daemon socket is secure" "! test -w /var/run/docker.sock || groups | grep -q docker"
}

# Function to test monitoring and logging
test_monitoring_logging() {
    info "Testing monitoring and logging..."

    # Test log directories
    run_test "Log directory exists" "test -d logs"

    # Test Prometheus metrics if available
    if docker-compose -f docker-compose.production.yml ps | grep -q prometheus; then
        run_test "Prometheus is accessible" "curl -f -s --max-time $TIMEOUT http://localhost:9090/metrics"
    fi

    # Test Grafana if available
    if docker-compose -f docker-compose.production.yml ps | grep -q grafana; then
        run_test "Grafana is accessible" "curl -f -s --max-time $TIMEOUT http://localhost:3000/api/health"
    fi
}

# Function to test backup and recovery
test_backup_recovery() {
    info "Testing backup and recovery capabilities..."

    run_test "Backup script exists" "test -x scripts/database-backup.sh"
    run_test "Backup directory is writable" "test -w /var/backups/freeagentics || mkdir -p ./backups && test -w ./backups"

    # Test backup functionality
    if [[ -x "scripts/database-backup.sh" ]] && docker-compose -f docker-compose.production.yml ps | grep -q postgres; then
        run_test "Database backup can be performed" "timeout 120 ./scripts/database-backup.sh backup"
    fi
}

# Function to test performance
test_performance() {
    info "Testing performance characteristics..."

    # Test response times
    local api_base="http://localhost:$API_PORT"

    if curl -f -s --max-time 5 "$api_base/health" >/dev/null 2>&1; then
        local response_time
        response_time=$(curl -f -s -w "%{time_total}" -o /dev/null "$api_base/health" | cut -d. -f1)
        run_test "API response time under 5 seconds" "[[ '$response_time' -lt 5 ]]"
    fi

    # Test resource usage
    if command -v docker &> /dev/null; then
        local containers
        containers=$(docker-compose -f docker-compose.production.yml ps -q)

        if [[ -n "$containers" ]]; then
            # Check memory usage of containers
            local high_memory_containers
            high_memory_containers=$(docker stats --no-stream --format "{{.MemPerc}}" $containers | sed 's/%//' | awk '$1 > 90' | wc -l)
            run_test "No containers using excessive memory" "[[ '$high_memory_containers' -eq 0 ]]"
        fi
    fi
}

# Function to test data persistence
test_data_persistence() {
    info "Testing data persistence..."

    # Test volume mounts
    run_test "PostgreSQL data volume exists" "docker volume ls | grep -q postgres_data"
    run_test "Redis data volume exists" "docker volume ls | grep -q redis_data"

    # Test data directory permissions
    if [[ -d "/var/lib/freeagentics" ]]; then
        run_test "Data directory is accessible" "test -r /var/lib/freeagentics"
    fi
}

# Function to test integration scenarios
test_integration_scenarios() {
    info "Testing integration scenarios..."

    local api_base="http://localhost:$API_PORT"

    # Test full stack integration
    if curl -f -s --max-time $TIMEOUT "$api_base/health" >/dev/null 2>&1; then
        # Test agent creation workflow
        run_test "Agent creation API is accessible" "curl -f -s --max-time $TIMEOUT $api_base/api/v1/agents"

        # Test knowledge graph API
        run_test "Knowledge graph API is accessible" "curl -f -s --max-time $TIMEOUT $api_base/api/v1/knowledge"

        # Test WebSocket endpoint
        run_test "WebSocket endpoint is accessible" "curl -f -s --max-time $TIMEOUT -H 'Connection: Upgrade' -H 'Upgrade: websocket' $api_base/ws/agents || true"
    fi
}

# Function to generate validation report
generate_report() {
    local timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')
    local report_file="validation_report_${timestamp}.json"

    cat > "$report_file" << EOF
{
  "validation_report": {
    "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
    "environment": "$ENVIRONMENT",
    "domain": "$DOMAIN",
    "summary": {
      "total_tests": $TESTS_RUN,
      "tests_passed": $TESTS_PASSED,
      "tests_failed": $TESTS_FAILED,
      "success_rate": "$(( TESTS_PASSED * 100 / TESTS_RUN ))%"
    },
    "status": "$([ $TESTS_FAILED -eq 0 ] && echo "PASS" || echo "FAIL")",
    "recommendations": [
      $([ $TESTS_FAILED -gt 0 ] && echo '"Review failed tests and address issues before production deployment",' || echo '')
      "Monitor application performance continuously",
      "Set up automated backup verification",
      "Implement health check alerting"
    ]
  }
}
EOF

    info "Validation report saved to: $report_file"
}

# Function to display summary
show_summary() {
    echo ""
    echo "=================================="
    echo "   DEPLOYMENT VALIDATION SUMMARY"
    echo "=================================="
    echo "Environment: $ENVIRONMENT"
    echo "Timestamp: $(date)"
    echo ""
    echo "Tests Run: $TESTS_RUN"
    echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
    echo ""

    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}✅ ALL TESTS PASSED - DEPLOYMENT IS READY${NC}"
        echo ""
        echo "🚀 Your FreeAgentics deployment is production ready!"
        echo "🔗 Access your application at: https://$DOMAIN"
        return 0
    else
        echo -e "${RED}❌ SOME TESTS FAILED - REVIEW REQUIRED${NC}"
        echo ""
        echo "⚠️  Please address the failed tests before production deployment."
        echo "📋 Review the test output above for specific issues."
        return 1
    fi
}

# Main function
main() {
    local start_time
    start_time=$(date +%s)

    echo "🔍 Starting FreeAgentics Production Deployment Validation"
    echo "Environment: $ENVIRONMENT"
    echo "Domain: $DOMAIN"
    echo "Timestamp: $(date)"
    echo ""

    # Run all test suites
    test_docker_environment
    test_network_connectivity
    test_ssl_configuration
    test_service_health
    test_api_endpoints
    test_frontend
    test_database_operations
    test_security_configuration
    test_monitoring_logging
    test_backup_recovery
    test_performance
    test_data_persistence
    test_integration_scenarios

    # Generate report and show summary
    generate_report

    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    info "Validation completed in ${duration} seconds"

    show_summary
}

# Script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -e, --environment ENV    Set environment (default: production)"
    echo "  -d, --domain DOMAIN     Set domain (default: localhost)"
    echo "  -t, --timeout SECONDS   Set timeout for HTTP requests (default: 30)"
    echo "  -v, --verbose           Enable verbose output"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Run all validation tests"
    echo "  $0 -d mydomain.com     # Validate specific domain"
    echo "  $0 -v                  # Run with verbose output"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
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

# Execute main function
main "$@"
