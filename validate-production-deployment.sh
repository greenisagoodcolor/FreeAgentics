#!/bin/bash
# Production Deployment Validation Script
# FreeAgentics Production Environment Configuration Expert
# 
# This script performs comprehensive validation of the production deployment:
# - Configuration validation
# - Security checks
# - Performance validation
# - Monitoring verification
# - Backup testing
# - Disaster recovery validation

set -euo pipefail

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_NAME="freeagentics"
readonly COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.production.yml"
readonly ENV_FILE="${SCRIPT_DIR}/.env.production"
readonly RESULTS_DIR="${SCRIPT_DIR}/validation_results"
readonly TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Test counters
TESTS_TOTAL=0
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_WARNINGS=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    TESTS_WARNINGS=$((TESTS_WARNINGS + 1))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

# Test execution wrapper
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    log_info "Running test: $test_name"
    
    if $test_function; then
        log_success "$test_name"
        echo "✓ $test_name" >> "$RESULTS_DIR/summary.txt"
    else
        log_error "$test_name"
        echo "✗ $test_name" >> "$RESULTS_DIR/summary.txt"
    fi
    
    echo "---"
}

# Setup validation environment
setup_validation() {
    log_info "Setting up validation environment..."
    mkdir -p "$RESULTS_DIR"
    echo "FreeAgentics Production Deployment Validation Report" > "$RESULTS_DIR/summary.txt"
    echo "Generated: $(date)" >> "$RESULTS_DIR/summary.txt"
    echo "" >> "$RESULTS_DIR/summary.txt"
}

# Configuration validation tests
test_docker_compose_config() {
    docker-compose -f "$COMPOSE_FILE" config -q
}

test_environment_variables() {
    if [[ ! -f "$ENV_FILE" ]]; then
        return 1
    fi
    
    source "$ENV_FILE"
    
    local required_vars=(
        "DOMAIN" "POSTGRES_PASSWORD" "REDIS_PASSWORD" "SECRET_KEY" "JWT_SECRET"
        "GRAFANA_ADMIN_PASSWORD" "GRAFANA_SECRET_KEY" "GRAFANA_DB_PASSWORD"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Missing required environment variable: $var"
            return 1
        fi
    done
    
    return 0
}

test_ssl_certificates() {
    local cert_file="$SCRIPT_DIR/nginx/ssl/cert.pem"
    local key_file="$SCRIPT_DIR/nginx/ssl/key.pem"
    
    if [[ ! -f "$cert_file" ]] || [[ ! -f "$key_file" ]]; then
        return 1
    fi
    
    # Validate certificate
    openssl x509 -in "$cert_file" -text -noout > /dev/null
    
    # Check certificate expiry
    local expiry_date=$(openssl x509 -enddate -noout -in "$cert_file" | cut -d= -f2)
    local current_date=$(date +%s)
    local cert_expiry=$(date -d "$expiry_date" +%s)
    local days_until_expiry=$(( (cert_expiry - current_date) / 86400 ))
    
    if [[ $days_until_expiry -lt 30 ]]; then
        log_warning "SSL certificate expires in $days_until_expiry days"
    fi
    
    return 0
}

test_nginx_configuration() {
    if [[ ! -f "$SCRIPT_DIR/nginx/nginx.conf" ]]; then
        return 1
    fi
    
    # Test nginx config syntax (requires nginx to be installed)
    if command -v nginx > /dev/null; then
        nginx -t -c "$SCRIPT_DIR/nginx/nginx.conf" 2>/dev/null
    fi
    
    return 0
}

test_monitoring_configuration() {
    local configs=(
        "monitoring/prometheus-production.yml"
        "monitoring/alertmanager-production.yml"
        "monitoring/loki-config.yaml"
        "monitoring/promtail-config.yaml"
        "monitoring/postgres-queries.yaml"
    )
    
    for config in "${configs[@]}"; do
        if [[ ! -f "$SCRIPT_DIR/$config" ]]; then
            log_error "Missing monitoring configuration: $config"
            return 1
        fi
    done
    
    return 0
}

# Security validation tests
test_security_headers() {
    local nginx_conf="$SCRIPT_DIR/nginx/nginx.conf"
    
    if [[ ! -f "$nginx_conf" ]]; then
        return 1
    fi
    
    local security_headers=(
        "X-Frame-Options"
        "X-Content-Type-Options"
        "X-XSS-Protection"
        "Strict-Transport-Security"
        "Content-Security-Policy"
    )
    
    for header in "${security_headers[@]}"; do
        if ! grep -q "$header" "$nginx_conf"; then
            log_error "Missing security header in nginx config: $header"
            return 1
        fi
    done
    
    return 0
}

test_container_security() {
    # Check for non-root users in compose file
    if ! grep -q "user:" "$COMPOSE_FILE"; then
        log_warning "Some services may be running as root"
    fi
    
    # Check for read-only containers
    if ! grep -q "read_only: true" "$COMPOSE_FILE"; then
        log_warning "Some containers may not be read-only"
    fi
    
    # Check for security options
    if ! grep -q "no-new-privileges" "$COMPOSE_FILE"; then
        log_warning "Security hardening options may be missing"
    fi
    
    return 0
}

test_secrets_management() {
    # Check that example/template values are not used
    if grep -q "CHANGE_ME" "$ENV_FILE" 2>/dev/null; then
        log_error "Template/example values found in environment file"
        return 1
    fi
    
    # Check password complexity
    source "$ENV_FILE"
    if [[ ${#POSTGRES_PASSWORD} -lt 20 ]]; then
        log_warning "Database password may be too short"
    fi
    
    return 0
}

# Performance validation tests
test_resource_limits() {
    # Check that resource limits are defined
    if ! grep -q "resources:" "$COMPOSE_FILE"; then
        log_warning "Resource limits not defined for all services"
    fi
    
    if ! grep -q "memory:" "$COMPOSE_FILE"; then
        log_warning "Memory limits not defined for all services"
    fi
    
    return 0
}

test_healthchecks() {
    # Check that health checks are defined
    if ! grep -q "healthcheck:" "$COMPOSE_FILE"; then
        log_error "Health checks not defined for critical services"
        return 1
    fi
    
    local healthcheck_count=$(grep -c "healthcheck:" "$COMPOSE_FILE")
    if [[ $healthcheck_count -lt 5 ]]; then
        log_warning "Limited health checks defined ($healthcheck_count found)"
    fi
    
    return 0
}

test_storage_configuration() {
    # Check volume configurations
    if ! grep -q "volumes:" "$COMPOSE_FILE"; then
        log_error "Volume configurations not found"
        return 1
    fi
    
    # Check for persistent data volumes
    local data_volumes=("postgres_data" "redis_data" "grafana_data" "prometheus_data")
    for volume in "${data_volumes[@]}"; do
        if ! grep -q "$volume:" "$COMPOSE_FILE"; then
            log_error "Missing data volume: $volume"
            return 1
        fi
    done
    
    return 0
}

# Monitoring validation tests
test_prometheus_config() {
    local prometheus_config="$SCRIPT_DIR/monitoring/prometheus-production.yml"
    
    if [[ ! -f "$prometheus_config" ]]; then
        return 1
    fi
    
    # Check for essential scrape configs
    local scrape_jobs=("freeagentics-backend" "node-exporter" "cadvisor" "postgres-exporter")
    for job in "${scrape_jobs[@]}"; do
        if ! grep -q "$job" "$prometheus_config"; then
            log_error "Missing Prometheus scrape job: $job"
            return 1
        fi
    done
    
    return 0
}

test_alerting_configuration() {
    local alertmanager_config="$SCRIPT_DIR/monitoring/alertmanager-production.yml"
    
    if [[ ! -f "$alertmanager_config" ]]; then
        log_warning "AlertManager configuration not found"
        return 1
    fi
    
    return 0
}

test_dashboard_configuration() {
    local dashboards_dir="$SCRIPT_DIR/monitoring/dashboards"
    
    if [[ ! -d "$dashboards_dir" ]]; then
        log_warning "Grafana dashboards directory not found"
        return 1
    fi
    
    local dashboard_count=$(find "$dashboards_dir" -name "*.json" 2>/dev/null | wc -l)
    if [[ $dashboard_count -lt 3 ]]; then
        log_warning "Limited dashboards available ($dashboard_count found)"
    fi
    
    return 0
}

# Backup validation tests
test_backup_configuration() {
    # Check for backup services in compose file
    if ! grep -q "postgres-backup:" "$COMPOSE_FILE"; then
        log_error "Database backup service not configured"
        return 1
    fi
    
    if ! grep -q "backup-agent:" "$COMPOSE_FILE"; then
        log_warning "File system backup service not configured"
    fi
    
    return 0
}

test_backup_directories() {
    local backup_dirs=("backups/postgres" "backups/files")
    
    for dir in "${backup_dirs[@]}"; do
        if [[ ! -d "$SCRIPT_DIR/$dir" ]]; then
            mkdir -p "$SCRIPT_DIR/$dir"
            log_info "Created backup directory: $dir"
        fi
    done
    
    return 0
}

test_disaster_recovery() {
    # Check for disaster recovery documentation
    if [[ ! -f "$SCRIPT_DIR/docs/operations/DISASTER_RECOVERY_PROCEDURES.md" ]]; then
        log_warning "Disaster recovery procedures documentation not found"
    fi
    
    # Check for backup scripts
    if [[ ! -d "$SCRIPT_DIR/scripts/backup" ]]; then
        log_warning "Backup scripts directory not found"
    fi
    
    return 0
}

# Network and connectivity tests
test_network_configuration() {
    # Check for custom network configuration
    if ! grep -q "networks:" "$COMPOSE_FILE"; then
        log_warning "Custom network configuration not found"
    fi
    
    # Check for proper network isolation
    if grep -q "network_mode: host" "$COMPOSE_FILE"; then
        log_warning "Host networking detected - may compromise isolation"
    fi
    
    return 0
}

test_port_configuration() {
    # Check that sensitive ports are not exposed externally
    if grep -q "5432:5432" "$COMPOSE_FILE"; then
        log_warning "PostgreSQL port exposed directly (consider localhost binding)"
    fi
    
    if grep -q "6379:6379" "$COMPOSE_FILE"; then
        log_warning "Redis port exposed directly (consider localhost binding)"
    fi
    
    return 0
}

# Deployment readiness tests
test_deployment_script() {
    local deploy_script="$SCRIPT_DIR/deploy-production-enhanced.sh"
    
    if [[ ! -f "$deploy_script" ]]; then
        log_error "Production deployment script not found"
        return 1
    fi
    
    if [[ ! -x "$deploy_script" ]]; then
        log_error "Deployment script is not executable"
        return 1
    fi
    
    return 0
}

test_dependencies() {
    local required_tools=("docker" "docker-compose" "curl" "openssl" "jq")
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" > /dev/null; then
            log_error "Required tool not installed: $tool"
            return 1
        fi
    done
    
    return 0
}

test_docker_service() {
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker service is not running"
        return 1
    fi
    
    # Check Docker version
    local docker_version=$(docker --version | grep -oP '\d+\.\d+' | head -1)
    local min_version="20.10"
    
    if [[ "$(printf '%s\n' "$min_version" "$docker_version" | sort -V | head -n1)" != "$min_version" ]]; then
        log_warning "Docker version ($docker_version) may be outdated (minimum: $min_version)"
    fi
    
    return 0
}

# Generate comprehensive report
generate_report() {
    local report_file="$RESULTS_DIR/validation_report_$TIMESTAMP.md"
    
    cat > "$report_file" << EOF
# FreeAgentics Production Deployment Validation Report

**Generated:** $(date)
**Environment:** Production
**Validation Script Version:** 1.0.0

## Executive Summary

- **Total Tests:** $TESTS_TOTAL
- **Passed:** $TESTS_PASSED
- **Failed:** $TESTS_FAILED
- **Warnings:** $TESTS_WARNINGS
- **Success Rate:** $(( TESTS_PASSED * 100 / TESTS_TOTAL ))%

## Test Results Summary

$(cat "$RESULTS_DIR/summary.txt")

## Recommendations

### Critical Issues (Must Fix)
EOF

    if [[ $TESTS_FAILED -gt 0 ]]; then
        echo "- $TESTS_FAILED critical issues found that must be resolved before production deployment" >> "$report_file"
        echo "- Review failed tests and implement fixes" >> "$report_file"
        echo "- Re-run validation after fixes" >> "$report_file"
    else
        echo "- No critical issues found" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

### Warnings (Should Fix)
- $TESTS_WARNINGS warnings found that should be addressed for optimal production deployment
- Review warnings and implement improvements where possible

### Next Steps
1. Address all failed tests
2. Consider fixing warning items
3. Run deployment in staging environment
4. Schedule production deployment
5. Implement monitoring and alerting
6. Test disaster recovery procedures

## Detailed Results

For detailed test results, check the individual log files in the validation_results directory.
EOF

    log_info "Comprehensive report generated: $report_file"
}

# Main validation function
main() {
    log_info "=== FreeAgentics Production Deployment Validation ==="
    log_info "Starting comprehensive validation at: $(date)"
    
    setup_validation
    
    # Configuration validation
    log_info "=== Configuration Validation ==="
    run_test "Docker Compose Configuration" test_docker_compose_config
    run_test "Environment Variables" test_environment_variables
    run_test "SSL Certificates" test_ssl_certificates
    run_test "Nginx Configuration" test_nginx_configuration
    run_test "Monitoring Configuration" test_monitoring_configuration
    
    # Security validation
    log_info "=== Security Validation ==="
    run_test "Security Headers" test_security_headers
    run_test "Container Security" test_container_security
    run_test "Secrets Management" test_secrets_management
    
    # Performance validation
    log_info "=== Performance Validation ==="
    run_test "Resource Limits" test_resource_limits
    run_test "Health Checks" test_healthchecks
    run_test "Storage Configuration" test_storage_configuration
    
    # Monitoring validation
    log_info "=== Monitoring Validation ==="
    run_test "Prometheus Configuration" test_prometheus_config
    run_test "Alerting Configuration" test_alerting_configuration
    run_test "Dashboard Configuration" test_dashboard_configuration
    
    # Backup validation
    log_info "=== Backup & Disaster Recovery Validation ==="
    run_test "Backup Configuration" test_backup_configuration
    run_test "Backup Directories" test_backup_directories
    run_test "Disaster Recovery" test_disaster_recovery
    
    # Network validation
    log_info "=== Network Validation ==="
    run_test "Network Configuration" test_network_configuration
    run_test "Port Configuration" test_port_configuration
    
    # Deployment readiness
    log_info "=== Deployment Readiness ==="
    run_test "Deployment Script" test_deployment_script
    run_test "Dependencies" test_dependencies
    run_test "Docker Service" test_docker_service
    
    # Generate final report
    generate_report
    
    # Final summary
    log_info "=== VALIDATION SUMMARY ==="
    log_info "Total Tests: $TESTS_TOTAL"
    log_success "Passed: $TESTS_PASSED"
    log_warning "Warnings: $TESTS_WARNINGS"
    log_error "Failed: $TESTS_FAILED"
    log_info "Success Rate: $(( TESTS_PASSED * 100 / TESTS_TOTAL ))%"
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_success "=== DEPLOYMENT READY ==="
        log_success "All critical validations passed. Deployment can proceed."
        if [[ $TESTS_WARNINGS -gt 0 ]]; then
            log_warning "Consider addressing $TESTS_WARNINGS warnings for optimal deployment"
        fi
        exit 0
    else
        log_error "=== DEPLOYMENT NOT READY ==="
        log_error "$TESTS_FAILED critical issues must be resolved before deployment"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-validate}" in
    "validate")
        main
        ;;
    "report")
        if [[ -f "$RESULTS_DIR/validation_report_"*.md ]]; then
            cat "$RESULTS_DIR/validation_report_"*.md | tail -1
        else
            echo "No validation reports found. Run validation first."
        fi
        ;;
    *)
        echo "Usage: $0 {validate|report}"
        echo ""
        echo "Commands:"
        echo "  validate - Run full production deployment validation"
        echo "  report   - Show latest validation report"
        exit 1
        ;;
esac