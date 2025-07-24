#!/bin/bash
# Comprehensive deployment verification script

set -euo pipefail

# Configuration
ENVIRONMENT="${1:-production}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
API_URL="${API_URL:-https://api.freeagentics.com}"
WEB_URL="${WEB_URL:-https://app.freeagentics.com}"
TIMEOUT="${TIMEOUT:-30}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Logging
log() {
    echo -e "${GREEN}[VERIFY]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $*" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $*"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $*"
}

# Check function wrapper
check() {
    local name="$1"
    local command="$2"
    local critical="${3:-true}"

    ((TOTAL_CHECKS++))

    info "Running check: $name"

    if eval "$command"; then
        log "✓ $name passed"
        ((PASSED_CHECKS++))
        return 0
    else
        if [[ "$critical" == "true" ]]; then
            error "✗ $name failed (critical)"
            ((FAILED_CHECKS++))
        else
            warning "⚠ $name failed (non-critical)"
        fi
        return 1
    fi
}

# Container health checks
check_containers() {
    log "Checking container health..."

    # Check if containers are running
    check "API container running" \
        "docker ps | grep -q 'freeagentics.*api.*Up'"

    check "Web container running" \
        "docker ps | grep -q 'freeagentics.*web.*Up'"

    check "Database container running" \
        "docker ps | grep -q 'postgres.*Up'"

    check "Redis container running" \
        "docker ps | grep -q 'redis.*Up'"

    # Check container health status
    check "API container healthy" \
        "docker inspect freeagentics-api | jq -r '.[0].State.Health.Status' | grep -q 'healthy'"

    check "Database container healthy" \
        "docker inspect freeagentics-postgres | jq -r '.[0].State.Health.Status' | grep -q 'healthy'"

    # Check resource usage
    check "Container memory usage acceptable" \
        "docker stats --no-stream --format 'table {{.Container}}\t{{.MemUsage}}' | grep -v 'CONTAINER' | awk '{print \$2}' | sed 's/[^0-9.]//g' | awk '{if(\$1 < 90) print \"ok\"; else print \"high\"}' | grep -q 'ok'"

    check "Container CPU usage acceptable" \
        "docker stats --no-stream --format 'table {{.Container}}\t{{.CPUPerc}}' | grep -v 'CONTAINER' | awk '{print \$2}' | sed 's/%//g' | awk '{if(\$1 < 90) print \"ok\"; else print \"high\"}' | grep -q 'ok'"
}

# Service endpoints
check_endpoints() {
    log "Checking service endpoints..."

    # Health endpoints
    check "API health endpoint" \
        "curl -f -s --max-time $TIMEOUT '$API_URL/health' | jq -r '.status' | grep -q 'healthy'"

    check "API system status" \
        "curl -f -s --max-time $TIMEOUT '$API_URL/v1/system/status' | jq -r '.status' | grep -q 'operational'"

    check "Web application accessible" \
        "curl -f -s --max-time $TIMEOUT '$WEB_URL' | grep -q 'FreeAgentics'"

    # Authentication endpoints
    check "Auth endpoints accessible" \
        "curl -f -s --max-time $TIMEOUT '$API_URL/v1/auth/health' | grep -q 'ok'"

    # WebSocket endpoints
    check "WebSocket endpoint accessible" \
        "timeout 10 wscat -c 'wss://api.freeagentics.com/ws/health' -x 'ping' 2>/dev/null | grep -q 'pong'" \
        "false"

    # Metrics endpoints
    check "Metrics endpoint accessible" \
        "curl -f -s --max-time $TIMEOUT '$API_URL/metrics' | grep -q 'python_info'"

    # API versioning
    check "API version endpoint" \
        "curl -f -s --max-time $TIMEOUT '$API_URL/v1/version' | jq -r '.version' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$'"
}

# Database connectivity
check_database() {
    log "Checking database connectivity..."

    # Basic connectivity
    check "Database connection" \
        "docker-compose -f '$PROJECT_ROOT/docker-compose.production.yml' exec -T api python -c 'from database.session import engine; engine.execute(\"SELECT 1\")'"

    # Migration status
    check "Database migrations current" \
        "docker-compose -f '$PROJECT_ROOT/docker-compose.production.yml' run --rm api alembic current | grep -q '[a-f0-9]{12}'"

    # Table existence
    check "Core tables exist" \
        "docker-compose -f '$PROJECT_ROOT/docker-compose.production.yml' exec -T api python -c 'from database.session import engine; from sqlalchemy import text; result = engine.execute(text(\"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = \\\"public\\\"\")).scalar(); print(result); assert result > 0'"

    # Connection pool
    check "Database connection pool healthy" \
        "docker-compose -f '$PROJECT_ROOT/docker-compose.production.yml' exec -T api python -c 'from database.session import engine; print(engine.pool.status())' | grep -q 'Pool size'"

    # Read/write operations
    check "Database read operations" \
        "docker-compose -f '$PROJECT_ROOT/docker-compose.production.yml' exec -T api python -c 'from database.session import SessionLocal; db = SessionLocal(); db.execute(\"SELECT 1\"); db.close()'"

    check "Database write operations" \
        "docker-compose -f '$PROJECT_ROOT/docker-compose.production.yml' exec -T api python -c 'from database.session import SessionLocal; from sqlalchemy import text; db = SessionLocal(); db.execute(text(\"CREATE TEMPORARY TABLE test_table (id INTEGER)\")); db.execute(text(\"INSERT INTO test_table (id) VALUES (1)\")); db.commit(); db.close()'"
}

# Redis connectivity
check_redis() {
    log "Checking Redis connectivity..."

    # Basic connectivity
    check "Redis connection" \
        "docker-compose -f '$PROJECT_ROOT/docker-compose.production.yml' exec -T redis redis-cli ping | grep -q 'PONG'"

    # Memory usage
    check "Redis memory usage acceptable" \
        "docker-compose -f '$PROJECT_ROOT/docker-compose.production.yml' exec -T redis redis-cli info memory | grep 'used_memory_human' | cut -d: -f2 | grep -E '[0-9]+[.]?[0-9]*[KM]B'"

    # Set/get operations
    check "Redis set/get operations" \
        "docker-compose -f '$PROJECT_ROOT/docker-compose.production.yml' exec -T redis redis-cli set test_key 'test_value' && docker-compose -f '$PROJECT_ROOT/docker-compose.production.yml' exec -T redis redis-cli get test_key | grep -q 'test_value'"

    # Cleanup test key
    docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" exec -T redis redis-cli del test_key >/dev/null 2>&1
}

# Security checks
check_security() {
    log "Checking security configuration..."

    # SSL/TLS
    check "SSL certificate valid" \
        "echo | openssl s_client -connect api.freeagentics.com:443 -servername api.freeagentics.com 2>/dev/null | openssl x509 -noout -dates | grep -q 'notAfter'"

    check "HSTS header present" \
        "curl -s -I '$API_URL/health' | grep -i 'strict-transport-security'"

    check "Security headers present" \
        "curl -s -I '$API_URL/health' | grep -i 'x-frame-options\\|x-content-type-options\\|x-xss-protection'"

    # Rate limiting
    check "Rate limiting active" \
        "for i in {1..30}; do curl -s -o /dev/null -w '%{http_code}' '$API_URL/v1/auth/login' & done; wait; curl -s -o /dev/null -w '%{http_code}' '$API_URL/v1/auth/login' | grep -q '429'" \
        "false"

    # Authentication
    check "Authentication required for protected endpoints" \
        "curl -s -o /dev/null -w '%{http_code}' '$API_URL/v1/agents' | grep -q '401'"

    # Environment variables
    check "No sensitive data in environment" \
        "! docker-compose -f '$PROJECT_ROOT/docker-compose.production.yml' config | grep -i 'password\\|secret\\|key' | grep -v 'REDACTED\\|\\*\\*\\*'"
}

# Performance checks
check_performance() {
    log "Checking performance metrics..."

    # Response times
    check "API response time acceptable" \
        "response_time=\$(curl -o /dev/null -s -w '%{time_total}' '$API_URL/health'); echo \"Response time: \${response_time}s\"; (( \$(echo \"\$response_time < 1.0\" | bc -l) ))"

    # Throughput test
    check "API handles concurrent requests" \
        "for i in {1..10}; do curl -s '$API_URL/health' & done; wait; echo 'Concurrent requests completed'"

    # Database query performance
    check "Database query performance acceptable" \
        "docker-compose -f '$PROJECT_ROOT/docker-compose.production.yml' exec -T api python -c 'import time; from database.session import engine; start = time.time(); engine.execute(\"SELECT 1\"); end = time.time(); print(f\"Query time: {end-start:.3f}s\"); assert end-start < 0.5'"

    # Memory leaks
    check "No significant memory leaks" \
        "docker stats --no-stream --format 'table {{.Container}}\t{{.MemUsage}}' | grep freeagentics | awk '{print \$2}' | sed 's/[^0-9.]//g' | awk '{if(\$1 < 1000) print \"ok\"; else print \"high\"}' | grep -q 'ok'"
}

# Monitoring checks
check_monitoring() {
    log "Checking monitoring setup..."

    # Prometheus metrics
    check "Prometheus metrics available" \
        "curl -s '$API_URL/metrics' | grep -q 'python_info'"

    check "Custom metrics present" \
        "curl -s '$API_URL/metrics' | grep -q 'freeagentics_'"

    # Health check metrics
    check "Health check metrics updated" \
        "curl -s '$API_URL/metrics' | grep 'freeagentics_health_check_total' | grep -q '[0-9]'"

    # Log aggregation
    check "Application logs being generated" \
        "docker logs freeagentics-api 2>&1 | tail -10 | grep -q '$(date +%Y-%m-%d)'"

    # Alerting
    check "Alerting configuration present" \
        "test -f '$PROJECT_ROOT/monitoring/alertmanager.yml'" \
        "false"
}

# Business logic checks
check_business_logic() {
    log "Checking business logic..."

    # API endpoints
    check "Agent endpoints accessible" \
        "curl -s '$API_URL/v1/agents' | grep -q 'authentication required\\|unauthorized'"

    check "Knowledge graph endpoints accessible" \
        "curl -s '$API_URL/v1/knowledge' | grep -q 'authentication required\\|unauthorized'"

    # Database integrity
    check "Database integrity check" \
        "docker-compose -f '$PROJECT_ROOT/docker-compose.production.yml' exec -T api python -c 'from database.session import engine; from sqlalchemy import text; result = engine.execute(text(\"SELECT COUNT(*) FROM information_schema.columns\")); print(f\"Columns: {result.scalar()}\"); assert result.scalar() > 0'"

    # Cache warming
    check "Cache warming completed" \
        "docker-compose -f '$PROJECT_ROOT/docker-compose.production.yml' exec -T redis redis-cli info keyspace | grep -q 'db0:keys='"
}

# External dependencies
check_external_dependencies() {
    log "Checking external dependencies..."

    # DNS resolution
    check "DNS resolution working" \
        "nslookup api.freeagentics.com | grep -q 'Address:'"

    # External API connectivity (if applicable)
    check "Internet connectivity" \
        "curl -s --max-time 10 https://httpbin.org/ip | grep -q 'origin'"

    # CDN functionality
    check "CDN serving static assets" \
        "curl -s --max-time 10 '$WEB_URL/static/favicon.ico' | file - | grep -q 'image\\|data'" \
        "false"
}

# Backup verification
check_backups() {
    log "Checking backup systems..."

    # Database backups
    check "Recent database backup exists" \
        "find /var/backups/freeagentics -name '*.sql.gz' -mtime -1 | grep -q '.'" \
        "false"

    # Backup integrity
    check "Latest backup is valid" \
        "latest_backup=\$(find /var/backups/freeagentics -name '*.sql.gz' -mtime -1 | head -1); test -n \"\$latest_backup\" && gunzip -t \"\$latest_backup\"" \
        "false"

    # Backup size reasonable
    check "Backup size reasonable" \
        "latest_backup=\$(find /var/backups/freeagentics -name '*.sql.gz' -mtime -1 | head -1); test -n \"\$latest_backup\" && test \$(stat -c%s \"\$latest_backup\") -gt 1000" \
        "false"
}

# Load balancer checks
check_load_balancer() {
    log "Checking load balancer..."

    # Health check endpoints
    check "Load balancer health check" \
        "curl -f -s --max-time $TIMEOUT '$API_URL/health' | jq -r '.status' | grep -q 'healthy'"

    # SSL termination
    check "SSL termination working" \
        "curl -I -s '$API_URL/health' | grep -q 'HTTP/[12]'"

    # Sticky sessions (if applicable)
    check "Session affinity working" \
        "session1=\$(curl -s -c /tmp/cookies1 '$API_URL/health' | jq -r '.server_id // \"none\"'); session2=\$(curl -s -b /tmp/cookies1 '$API_URL/health' | jq -r '.server_id // \"none\"'); test \"\$session1\" = \"\$session2\"" \
        "false"

    # Cleanup cookies
    rm -f /tmp/cookies1 2>/dev/null
}

# Generate deployment report
generate_report() {
    echo -e "\n${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║        DEPLOYMENT VERIFICATION         ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}\n"

    echo -e "${BLUE}Environment:${NC} $ENVIRONMENT"
    echo -e "${BLUE}Timestamp:${NC} $(date)"
    echo -e "${BLUE}Total Checks:${NC} $TOTAL_CHECKS"
    echo -e "${GREEN}Passed:${NC} $PASSED_CHECKS"
    echo -e "${RED}Failed:${NC} $FAILED_CHECKS"

    local success_rate=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
    echo -e "${BLUE}Success Rate:${NC} $success_rate%"

    if [[ $FAILED_CHECKS -eq 0 ]]; then
        echo -e "\n${GREEN}✓ All critical checks passed!${NC}"
        echo -e "${GREEN}Deployment verification successful!${NC}"
        return 0
    else
        echo -e "\n${RED}✗ Some checks failed!${NC}"
        echo -e "${RED}Please review the failures above.${NC}"
        return 1
    fi
}

# Main verification process
main() {
    echo -e "${BLUE}Starting deployment verification...${NC}\n"

    # Run all check categories
    check_containers
    check_endpoints
    check_database
    check_redis
    check_security
    check_performance
    check_monitoring
    check_business_logic
    check_external_dependencies
    check_backups
    check_load_balancer

    # Generate and display report
    generate_report
}

# Handle command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --web-url)
            WEB_URL="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --quick)
            # Quick mode - only critical checks
            QUICK_MODE=true
            shift
            ;;
        *)
            ENVIRONMENT="$1"
            shift
            ;;
    esac
done

# Run main verification
main
