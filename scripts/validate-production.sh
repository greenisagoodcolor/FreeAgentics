#!/bin/bash

# FreeAgentics Production Environment Configuration Validator
# Task 21: Comprehensive production validation with nemesis-level rigor
# Validates all production configurations, connections, and readiness

set -euo pipefail

# Configuration
ENVIRONMENT="${ENVIRONMENT:-production}"
DOMAIN="${DOMAIN:-localhost}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
REDIS_PORT="${REDIS_PORT:-6379}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
TIMEOUT="${TIMEOUT:-30}"
VERBOSE="${VERBOSE:-false}"
VALIDATION_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_WARNING=0
CRITICAL_FAILURES=()

# Output files
REPORT_FILE="production_validation_report_${VALIDATION_TIMESTAMP}.json"
MARKDOWN_REPORT="production_validation_report_${VALIDATION_TIMESTAMP}.md"
LOG_FILE="production_validation_${VALIDATION_TIMESTAMP}.log"

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}" | tee -a "$LOG_FILE"
    TESTS_WARNING=$((TESTS_WARNING + 1))
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] ℹ $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}${BOLD}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_FILE"
}

critical() {
    echo -e "${RED}${BOLD}[$(date '+%Y-%m-%d %H:%M:%S')] 🚨 CRITICAL: $1${NC}" | tee -a "$LOG_FILE"
    CRITICAL_FAILURES+=("$1")
}

# Test result tracking
run_test() {
    local test_name="$1"
    local test_command="$2"
    local is_critical="${3:-false}"

    TESTS_RUN=$((TESTS_RUN + 1))

    if [[ "$VERBOSE" == "true" ]]; then
        info "Running test: $test_name"
    fi

    if eval "$test_command" >/dev/null 2>&1; then
        log "PASS: $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        if [[ "$is_critical" == "true" ]]; then
            critical "FAIL: $test_name"
        else
            error "FAIL: $test_name"
        fi
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Validate environment variables
validate_environment_variables() {
    echo -e "\n${BOLD}=== VALIDATING ENVIRONMENT VARIABLES ===${NC}"

    # Critical environment variables
    local critical_vars=(
        "DATABASE_URL"
        "REDIS_URL"
        "SECRET_KEY"
        "JWT_SECRET"
        "ALLOWED_HOSTS"
    )

    # Optional but recommended variables
    local recommended_vars=(
        "SMTP_HOST"
        "SMTP_PORT"
        "SMTP_USER"
        "MONITORING_ENABLED"
        "LOG_LEVEL"
        "BACKUP_RETENTION_DAYS"
    )

    # Check for .env.production
    if [[ -f ".env.production" ]]; then
        log "Production environment file found"

        # Load environment variables for testing
        set -a
        source .env.production
        set +a

        # Check for development values
        if grep -q "dev_secret\|localhost:5432\|password123\|your_" .env.production 2>/dev/null; then
            critical "Development values found in production environment file!"
        fi

        # Check file permissions
        local perms=$(stat -c "%a" .env.production 2>/dev/null || stat -f "%OLp" .env.production)
        if [[ "$perms" != "600" && "$perms" != "400" ]]; then
            warn "Environment file has insecure permissions: $perms (should be 600 or 400)"
        fi
    else
        critical ".env.production file not found!"
    fi

    # Validate critical variables
    for var in "${critical_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            critical "Critical environment variable not set: $var"
        else
            log "Critical variable set: $var"
        fi
    done

    # Check recommended variables
    for var in "${recommended_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            warn "Recommended variable not set: $var"
        else
            log "Recommended variable set: $var"
        fi
    done

    # Validate specific variable formats
    if [[ -n "${DATABASE_URL:-}" ]]; then
        if [[ ! "$DATABASE_URL" =~ ^postgresql:// ]]; then
            critical "DATABASE_URL must start with postgresql://"
        fi
    fi

    if [[ -n "${JWT_SECRET:-}" ]]; then
        if [[ ${#JWT_SECRET} -lt 32 ]]; then
            critical "JWT_SECRET should be at least 32 characters long"
        fi
    fi
}

# Validate SSL/TLS configuration
validate_ssl_tls() {
    echo -e "\n${BOLD}=== VALIDATING SSL/TLS CONFIGURATION ===${NC}"

    # Check certificate files
    run_test "SSL certificate exists" "test -f nginx/ssl/cert.pem || test -f nginx/ssl/$DOMAIN.crt" true
    run_test "SSL private key exists" "test -f nginx/ssl/key.pem || test -f nginx/ssl/$DOMAIN.key" true
    run_test "DH parameters exist" "test -f nginx/dhparam.pem" false

    # Validate certificate
    if [[ -f "nginx/ssl/cert.pem" ]]; then
        # Check certificate validity
        if openssl x509 -in nginx/ssl/cert.pem -noout -checkend 86400 >/dev/null 2>&1; then
            log "SSL certificate is valid for at least 24 hours"
        else
            critical "SSL certificate expires within 24 hours!"
        fi

        # Check certificate and key match
        if [[ -f "nginx/ssl/key.pem" ]]; then
            local cert_modulus=$(openssl x509 -noout -modulus -in nginx/ssl/cert.pem 2>/dev/null | openssl md5)
            local key_modulus=$(openssl rsa -noout -modulus -in nginx/ssl/key.pem 2>/dev/null | openssl md5)
            if [[ "$cert_modulus" == "$key_modulus" ]]; then
                log "SSL certificate and key match"
            else
                critical "SSL certificate and key do not match!"
            fi
        fi

        # Check certificate chain
        run_test "Certificate chain is valid" "openssl verify -CAfile nginx/ssl/ca.pem nginx/ssl/cert.pem 2>/dev/null " false
    fi

    # Check nginx SSL configuration
    if [[ -f "nginx/nginx.conf" ]]; then
        # Check for modern TLS protocols
        if grep -q "ssl_protocols.*TLSv1.2.*TLSv1.3" nginx/nginx.conf; then
            log "Modern TLS protocols configured"
        else
            warn "TLS 1.2 and 1.3 should be configured"
        fi

        # Check for strong ciphers
        if grep -q "ssl_ciphers.*ECDHE" nginx/nginx.conf; then
            log "Strong cipher suites configured"
        else
            warn "Strong cipher suites should be configured"
        fi

        # Check for HSTS
        if grep -q "Strict-Transport-Security" nginx/nginx.conf; then
            log "HSTS header configured"
        else
            warn "HSTS header should be configured for security"
        fi
    fi
}

# Validate database configuration and connections
validate_database() {
    echo -e "\n${BOLD}=== VALIDATING DATABASE CONFIGURATION ===${NC}"

    # Check PostgreSQL configuration
    if [[ -f "docker-compose.production.yml" ]]; then
        # Check if PostgreSQL service is defined
        if grep -q "postgres:" docker-compose.production.yml; then
            log "PostgreSQL service configured"

            # Check for persistent volumes
            if grep -q "postgres_data:" docker-compose.production.yml; then
                log "Database persistence configured"
            else
                critical "Database persistence not configured!"
            fi

            # Check for health checks
            if grep -q "healthcheck:" docker-compose.production.yml; then
                log "Database health checks configured"
            else
                warn "Database health checks should be configured"
            fi
        else
            critical "PostgreSQL service not found in production configuration!"
        fi
    fi

    # Test database connection if running
    if command -v psql &> /dev/null && [[ -n "${DATABASE_URL:-}" ]]; then
        if psql "$DATABASE_URL" -c "SELECT 1;" >/dev/null 2>&1; then
            log "Database connection successful"

            # Check database size
            local db_size=$(psql "$DATABASE_URL" -t -c "SELECT pg_size_pretty(pg_database_size(current_database()));" 2>/dev/null | xargs)
            info "Database size: $db_size"

            # Check table existence
            if psql "$DATABASE_URL" -t -c "\dt" 2>/dev/null | grep -q "agents\|coalitions\|knowledge"; then
                log "Core tables exist"
            else
                warn "Some core tables may be missing"
            fi

            # Check for indexes
            local index_count=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public';" 2>/dev/null | xargs)
            info "Database indexes: $index_count"
        else
            error "Database connection failed"
        fi
    fi

    # Check Alembic migrations
    if [[ -f "alembic.ini" ]]; then
        log "Alembic migrations configured"

        if [[ -d "alembic/versions" ]]; then
            local migration_count=$(ls -1 alembic/versions/*.py 2>/dev/null | wc -l)
            info "Migration files found: $migration_count"
        else
            warn "No migration files found"
        fi
    else
        critical "Alembic configuration not found!"
    fi
}

# Validate API endpoints
validate_api_endpoints() {
    echo -e "\n${BOLD}=== VALIDATING API ENDPOINTS ===${NC}"

    local api_base="http://localhost:$API_PORT"
    local timeout_flag="--max-time $TIMEOUT"

    # Core endpoints
    run_test "API health endpoint" "curl -f -s $timeout_flag $api_base/health" true
    run_test "API docs endpoint" "curl -f -s $timeout_flag $api_base/docs" false
    run_test "OpenAPI schema" "curl -f -s $timeout_flag $api_base/openapi.json" false

    # Authentication endpoints
    run_test "Auth health" "curl -f -s $timeout_flag $api_base/api/v1/auth/health" true
    run_test "Auth login endpoint" "curl -f -s $timeout_flag -X POST $api_base/api/v1/auth/login" false
    run_test "Auth refresh endpoint" "curl -f -s $timeout_flag -X POST $api_base/api/v1/auth/refresh" false

    # Core API endpoints
    run_test "Agents API" "curl -f -s $timeout_flag $api_base/api/v1/agents" false
    run_test "Coalitions API" "curl -f -s $timeout_flag $api_base/api/v1/coalitions" false
    run_test "Knowledge API" "curl -f -s $timeout_flag $api_base/api/v1/knowledge" false
    run_test "Monitoring API" "curl -f -s $timeout_flag $api_base/api/v1/monitoring/metrics" false

    # System endpoints
    run_test "System info" "curl -f -s $timeout_flag $api_base/api/v1/system/info" false
    run_test "System health" "curl -f -s $timeout_flag $api_base/api/v1/system/health" false

    # WebSocket endpoint
    run_test "WebSocket endpoint" "curl -f -s $timeout_flag -H 'Upgrade: websocket' $api_base/ws/agents" false

    # Security headers check
    if curl -f -s -I "$api_base/health" 2>/dev/null | grep -q "X-Content-Type-Options"; then
        log "Security headers present"
    else
        warn "Security headers should be configured"
    fi
}

# Validate security configurations
validate_security() {
    echo -e "\n${BOLD}=== VALIDATING SECURITY CONFIGURATIONS ===${NC}"

    # Check RBAC configuration
    run_test "RBAC module exists" "test -f auth/rbac_enhancements.py" true
    run_test "Security headers module" "test -f auth/security_headers.py" true
    run_test "Security implementation" "test -f auth/security_implementation.py" true

    # Check JWT keys
    run_test "JWT private key exists" "test -f auth/keys/jwt_private.pem" true
    run_test "JWT public key exists" "test -f auth/keys/jwt_public.pem" true

    # Check key permissions
    if [[ -f "auth/keys/jwt_private.pem" ]]; then
        local key_perms=$(stat -c "%a" auth/keys/jwt_private.pem 2>/dev/null || stat -f "%OLp" auth/keys/jwt_private.pem)
        if [[ "$key_perms" == "600" || "$key_perms" == "400" ]]; then
            log "JWT private key has secure permissions"
        else
            critical "JWT private key has insecure permissions: $key_perms"
        fi
    fi

    # Check rate limiting
    if [[ -f "nginx/nginx.conf" ]]; then
        if grep -q "limit_req_zone" nginx/nginx.conf; then
            log "Rate limiting configured in nginx"
        else
            warn "Rate limiting should be configured"
        fi
    fi

    # Check for secure cookies configuration
    if [[ -f "api/main.py" ]]; then
        if grep -q "secure=True" api/main.py || grep -q "SESSION_COOKIE_SECURE" api/main.py; then
            log "Secure cookies configured"
        else
            warn "Secure cookies should be enabled in production"
        fi
    fi

    # Check for CORS configuration
    if grep -r "CORSMiddleware\|cors" api/ 2>/dev/null | grep -q "allow_origins"; then
        log "CORS configuration found"
    else
        warn "CORS should be properly configured"
    fi
}

# Validate monitoring and alerting
validate_monitoring() {
    echo -e "\n${BOLD}=== VALIDATING MONITORING AND ALERTING ===${NC}"

    # Check Prometheus configuration
    run_test "Prometheus config exists" "test -f monitoring/prometheus.yml || test -f monitoring/prometheus-production.yml" false
    run_test "Alert rules exist" "test -d monitoring/rules && ls monitoring/rules/*.yml >/dev/null 2>&1" false
    run_test "Alertmanager config" "test -f monitoring/alertmanager.yml" false

    # Check Grafana dashboards
    if [[ -d "monitoring/grafana/dashboards" ]]; then
        local dashboard_count=$(ls -1 monitoring/grafana/dashboards/*.json 2>/dev/null | wc -l)
        if [[ $dashboard_count -gt 0 ]]; then
            log "Grafana dashboards found: $dashboard_count"
        else
            warn "No Grafana dashboards found"
        fi
    fi

    # Test Prometheus endpoint if running
    if curl -f -s --max-time 5 "http://localhost:$PROMETHEUS_PORT/api/v1/query" >/dev/null 2>&1; then
        log "Prometheus is accessible"

        # Check for key metrics
        local metrics=(
            "up"
            "http_requests_total"
            "http_request_duration_seconds"
            "process_cpu_seconds_total"
            "process_resident_memory_bytes"
        )

        for metric in "${metrics[@]}"; do
            if curl -f -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=$metric" 2>/dev/null | grep -q "success"; then
                log "Metric available: $metric"
            else
                warn "Metric not available: $metric"
            fi
        done
    else
        warn "Prometheus not accessible"
    fi

    # Check application metrics endpoint
    run_test "Application metrics endpoint" "curl -f -s --max-time 5 http://localhost:$API_PORT/metrics" false
}

# Validate backup and recovery
validate_backup_recovery() {
    echo -e "\n${BOLD}=== VALIDATING BACKUP AND RECOVERY ===${NC}"

    # Check backup scripts
    run_test "Database backup script" "test -x scripts/database-backup.sh" true
    run_test "Backup directory writable" "test -w backups || mkdir -p backups && test -w backups" false

    # Check backup configuration
    if [[ -f "scripts/database-backup.sh" ]]; then
        # Check for encryption configuration
        if grep -q "encrypt\|gpg\|openssl" scripts/database-backup.sh; then
            log "Backup encryption configured"
        else
            warn "Backup encryption should be configured"
        fi

        # Check for retention policy
        if grep -q "retention\|days\|cleanup" scripts/database-backup.sh; then
            log "Backup retention policy configured"
        else
            warn "Backup retention policy should be configured"
        fi
    fi

    # Check for backup monitoring
    if [[ -f "monitoring/rules/alerts.yml" ]]; then
        if grep -q "backup" monitoring/rules/alerts.yml; then
            log "Backup monitoring alerts configured"
        else
            warn "Backup monitoring alerts should be configured"
        fi
    fi

    # Test backup functionality (dry run)
    if [[ -x "scripts/database-backup.sh" ]] && [[ "${SKIP_BACKUP_TEST:-false}" != "true" ]]; then
        info "Testing backup functionality (dry run)..."
        if timeout 30 bash scripts/database-backup.sh --dry-run >/dev/null 2>&1; then
            log "Backup dry run successful"
        else
            warn "Backup dry run failed"
        fi
    fi
}

# Validate disaster recovery
validate_disaster_recovery() {
    echo -e "\n${BOLD}=== VALIDATING DISASTER RECOVERY ===${NC}"

    # Check DR documentation
    run_test "DR procedures documented" "test -f docs/runbooks/EMERGENCY_PROCEDURES.md" false
    run_test "Incident response plan" "test -f docs/runbooks/INCIDENT_RESPONSE.md" false

    # Check recovery scripts
    local recovery_scripts=(
        "scripts/restore-database.sh"
        "scripts/rollback-deployment.sh"
        "scripts/emergency-shutdown.sh"
    )

    for script in "${recovery_scripts[@]}"; do
        if [[ -f "$script" ]]; then
            log "Recovery script found: $script"
        else
            warn "Recovery script missing: $script"
        fi
    done

    # Check for health check automation
    if grep -r "health.*check\|healthcheck" deploy-production*.sh 2>/dev/null | grep -q "wait\|retry"; then
        log "Automated health checks in deployment"
    else
        warn "Deployment should include automated health checks"
    fi

    # Validate rollback capability
    if [[ -f "deploy-production.sh" ]]; then
        if grep -q "rollback\|previous\|revert" deploy-production.sh; then
            log "Rollback capability present in deployment"
        else
            critical "No rollback capability in deployment script!"
        fi
    fi
}

# Performance validation
validate_performance() {
    echo -e "\n${BOLD}=== VALIDATING PERFORMANCE CONFIGURATION ===${NC}"

    # Check resource limits in docker-compose
    if [[ -f "docker-compose.production.yml" ]]; then
        if grep -q "deploy:\|resources:\|limits:" docker-compose.production.yml; then
            log "Resource limits configured"
        else
            warn "Resource limits should be configured for production"
        fi

        # Check for replicas configuration
        if grep -q "replicas:" docker-compose.production.yml; then
            log "Service replication configured"
        else
            info "Consider configuring service replicas for high availability"
        fi
    fi

    # Check caching configuration
    if grep -q "redis:" docker-compose.production.yml; then
        log "Redis caching configured"

        # Test Redis connection if available
        if command -v redis-cli &> /dev/null && [[ -n "${REDIS_URL:-}" ]]; then
            if redis-cli -u "$REDIS_URL" ping >/dev/null 2>&1; then
                log "Redis connection successful"

                # Check Redis memory
                local redis_memory=$(redis-cli -u "$REDIS_URL" info memory 2>/dev/null | grep "used_memory_human" | cut -d: -f2 | tr -d '\r')
                info "Redis memory usage: $redis_memory"
            else
                warn "Redis connection failed"
            fi
        fi
    else
        warn "Redis caching not configured"
    fi

    # Check for database optimization
    if [[ -f "database/query_optimization.py" ]]; then
        log "Database query optimization module present"
    else
        warn "Consider implementing query optimization"
    fi

    # Test API response times
    if curl -f -s "http://localhost:$API_PORT/health" >/dev/null 2>&1; then
        local response_time=$(curl -o /dev/null -s -w '%{time_total}' "http://localhost:$API_PORT/health")
        local response_ms=$(echo "$response_time * 1000" | bc | cut -d. -f1)

        if [[ $response_ms -lt 200 ]]; then
            log "API response time acceptable: ${response_ms}ms"
        else
            warn "API response time high: ${response_ms}ms (should be < 200ms)"
        fi
    fi
}

# Integration testing
validate_integration() {
    echo -e "\n${BOLD}=== VALIDATING SYSTEM INTEGRATION ===${NC}"

    # Test full stack if services are running
    if curl -f -s "http://localhost:$API_PORT/health" >/dev/null 2>&1; then
        info "Testing full stack integration..."

        # Test authentication flow
        local auth_test=$(curl -s -X POST "http://localhost:$API_PORT/api/v1/auth/login" \
            -H "Content-Type: application/json" \
            -d '{"username":"test","password":"test"}' 2>/dev/null || echo "{}")

        if echo "$auth_test" | grep -q "error\|detail"; then
            log "Authentication endpoint responding correctly"
        else
            warn "Authentication endpoint may not be configured correctly"
        fi

        # Test agent creation (dry run)
        local agent_test=$(curl -s -X POST "http://localhost:$API_PORT/api/v1/agents" \
            -H "Content-Type: application/json" \
            -d '{"name":"test","type":"test"}' 2>/dev/null || echo "{}")

        if echo "$agent_test" | grep -q "error\|detail\|id"; then
            log "Agent API responding"
        else
            warn "Agent API may not be configured correctly"
        fi
    else
        info "Skipping integration tests - services not running"
    fi
}

# Generate validation report
generate_report() {
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))

    # Calculate percentages
    local pass_rate=0
    if [[ $TESTS_RUN -gt 0 ]]; then
        pass_rate=$(( (TESTS_PASSED * 100) / TESTS_RUN ))
    fi

    # Generate JSON report
    cat > "$REPORT_FILE" << EOF
{
  "validation_report": {
    "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
    "environment": "$ENVIRONMENT",
    "domain": "$DOMAIN",
    "duration_seconds": $duration,
    "summary": {
      "total_tests": $TESTS_RUN,
      "tests_passed": $TESTS_PASSED,
      "tests_failed": $TESTS_FAILED,
      "tests_warning": $TESTS_WARNING,
      "critical_failures": ${#CRITICAL_FAILURES[@]},
      "pass_rate": "$pass_rate%"
    },
    "critical_issues": [
$(printf '      "%s"' "${CRITICAL_FAILURES[@]}" | sed 's/"$/",/')
    ],
    "status": "$([ ${#CRITICAL_FAILURES[@]} -eq 0 ] && echo "READY" || echo "NOT_READY")",
    "production_ready": $([ ${#CRITICAL_FAILURES[@]} -eq 0 ] && echo "true" || echo "false")
  }
}
EOF

    # Generate Markdown report
    cat > "$MARKDOWN_REPORT" << EOF
# FreeAgentics Production Validation Report

**Generated:** $(date)
**Environment:** $ENVIRONMENT
**Domain:** $DOMAIN
**Duration:** ${duration}s

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | $TESTS_RUN |
| Passed | $TESTS_PASSED |
| Failed | $TESTS_FAILED |
| Warnings | $TESTS_WARNING |
| Critical Failures | ${#CRITICAL_FAILURES[@]} |
| Pass Rate | $pass_rate% |

## Status: $([ ${#CRITICAL_FAILURES[@]} -eq 0 ] && echo "✅ PRODUCTION READY" || echo "❌ NOT PRODUCTION READY")

$(if [ ${#CRITICAL_FAILURES[@]} -gt 0 ]; then
    echo "## Critical Issues"
    echo ""
    for issue in "${CRITICAL_FAILURES[@]}"; do
        echo "- 🚨 $issue"
    done
fi)

## Validation Details

See \`$LOG_FILE\` for complete test results.

## Next Steps

$(if [ ${#CRITICAL_FAILURES[@]} -eq 0 ]; then
    echo "1. Review warnings and optimize where possible"
    echo "2. Run load testing to verify performance"
    echo "3. Schedule disaster recovery drill"
    echo "4. Deploy to production with confidence! 🚀"
else
    echo "1. **Address all critical issues immediately**"
    echo "2. Review and fix failed tests"
    echo "3. Re-run validation after fixes"
    echo "4. Do not deploy until all critical issues are resolved"
fi)
EOF

    info "Reports generated:"
    info "  - JSON: $REPORT_FILE"
    info "  - Markdown: $MARKDOWN_REPORT"
    info "  - Full log: $LOG_FILE"
}

# Display summary
show_summary() {
    echo ""
    echo -e "${BOLD}===================================="
    echo "   PRODUCTION VALIDATION SUMMARY"
    echo "====================================${NC}"
    echo "Timestamp: $(date)"
    echo "Environment: $ENVIRONMENT"
    echo ""
    echo "Tests Run: $TESTS_RUN"
    echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
    echo -e "Warnings: ${YELLOW}$TESTS_WARNING${NC}"
    echo -e "Critical Failures: ${RED}${BOLD}${#CRITICAL_FAILURES[@]}${NC}"
    echo ""

    if [[ ${#CRITICAL_FAILURES[@]} -eq 0 ]]; then
        if [[ $TESTS_FAILED -eq 0 ]]; then
            success "✅ ALL TESTS PASSED - PRODUCTION READY!"
            echo ""
            echo "🚀 Your FreeAgentics deployment is ready for production!"
            echo "📊 Review the generated reports for optimization opportunities."
            return 0
        else
            warn "⚠️  SOME TESTS FAILED - REVIEW REQUIRED"
            echo ""
            echo "Please review failed tests and warnings before production deployment."
            return 1
        fi
    else
        critical "❌ CRITICAL FAILURES DETECTED - NOT PRODUCTION READY"
        echo ""
        echo "🛑 Critical issues must be resolved before production deployment:"
        for issue in "${CRITICAL_FAILURES[@]}"; do
            echo "   - $issue"
        done
        echo ""
        echo "Fix these issues and run validation again."
        return 2
    fi
}

# Main function
main() {
    START_TIME=$(date +%s)

    echo -e "${BOLD}${CYAN}🔍 FreeAgentics Production Environment Configuration Validator${NC}"
    echo -e "${CYAN}Task 21: Comprehensive Production Validation${NC}"
    echo "Environment: $ENVIRONMENT"
    echo "Timestamp: $(date)"
    echo ""

    # Create reports directory
    mkdir -p reports

    # Run all validations
    validate_environment_variables
    validate_ssl_tls
    validate_database
    validate_api_endpoints
    validate_security
    validate_monitoring
    validate_backup_recovery
    validate_disaster_recovery
    validate_performance
    validate_integration

    # Generate reports
    generate_report

    # Show summary and exit with appropriate code
    show_summary
}

# Help function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --environment ENV    Set environment (default: production)"
    echo "  -d, --domain DOMAIN     Set domain (default: localhost)"
    echo "  -s, --skip-backup-test  Skip backup test"
    echo "  -v, --verbose           Enable verbose output"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run all validations"
    echo "  $0 -d example.com           # Validate for specific domain"
    echo "  $0 -v                       # Run with verbose output"
    echo "  $0 -s                       # Skip backup test"
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
        -s|--skip-backup-test)
            SKIP_BACKUP_TEST=true
            shift
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
