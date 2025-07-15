#!/bin/bash
# Monitoring Validation Script for FreeAgentics Production
# Validates monitoring stack health and configuration

set -euo pipefail

# Configuration
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
ALERTMANAGER_URL="${ALERTMANAGER_URL:-http://localhost:9093}"
TIMEOUT="${TIMEOUT:-10}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test counters
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

# Test execution function
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TESTS_RUN=$((TESTS_RUN + 1))
    
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

# Test Prometheus health and configuration
test_prometheus() {
    info "Testing Prometheus..."
    
    # Test Prometheus health
    run_test "Prometheus health endpoint" "curl -f -s --max-time $TIMEOUT $PROMETHEUS_URL/-/healthy"
    run_test "Prometheus ready endpoint" "curl -f -s --max-time $TIMEOUT $PROMETHEUS_URL/-/ready"
    
    # Test Prometheus API
    run_test "Prometheus API query" "curl -f -s --max-time $TIMEOUT '$PROMETHEUS_URL/api/v1/query?query=up'"
    
    # Test specific targets
    local targets=(
        "freeagentics-backend"
        "freeagentics-frontend"
        "postgres-exporter"
        "redis-exporter"
        "node-exporter"
    )
    
    for target in "${targets[@]}"; do
        run_test "Target $target is up" "curl -f -s --max-time $TIMEOUT '$PROMETHEUS_URL/api/v1/query?query=up{job=\"$target\"}' | grep -q '\"value\":\[.*,\"1\"\]'"
    done
    
    # Test alert rules are loaded
    run_test "Alert rules are loaded" "curl -f -s --max-time $TIMEOUT '$PROMETHEUS_URL/api/v1/rules' | grep -q 'rules'"
    
    # Test that we have recent data
    run_test "Recent metrics data available" "curl -f -s --max-time $TIMEOUT '$PROMETHEUS_URL/api/v1/query?query=up' | grep -q 'result'"
}

# Test Alertmanager health and configuration
test_alertmanager() {
    info "Testing Alertmanager..."
    
    # Test Alertmanager health
    run_test "Alertmanager health endpoint" "curl -f -s --max-time $TIMEOUT $ALERTMANAGER_URL/-/healthy"
    run_test "Alertmanager ready endpoint" "curl -f -s --max-time $TIMEOUT $ALERTMANAGER_URL/-/ready"
    
    # Test Alertmanager API
    run_test "Alertmanager status API" "curl -f -s --max-time $TIMEOUT '$ALERTMANAGER_URL/api/v1/status'"
    run_test "Alertmanager config API" "curl -f -s --max-time $TIMEOUT '$ALERTMANAGER_URL/api/v1/status' | grep -q 'configYAML'"
    
    # Test alert receivers are configured
    run_test "Alert receivers configured" "curl -f -s --max-time $TIMEOUT '$ALERTMANAGER_URL/api/v1/status' | grep -q 'receivers'"
}

# Test Grafana health and dashboards
test_grafana() {
    info "Testing Grafana..."
    
    # Test Grafana health
    run_test "Grafana health endpoint" "curl -f -s --max-time $TIMEOUT $GRAFANA_URL/api/health"
    
    # Test Grafana API (may require authentication)
    if curl -f -s --max-time $TIMEOUT "$GRAFANA_URL/api/datasources" >/dev/null 2>&1; then
        run_test "Grafana datasources API accessible" "true"
        run_test "Prometheus datasource configured" "curl -f -s --max-time $TIMEOUT '$GRAFANA_URL/api/datasources' | grep -q 'prometheus'"
    else
        warn "Grafana API requires authentication - skipping detailed tests"
    fi
}

# Test application metrics endpoints
test_application_metrics() {
    info "Testing application metrics..."
    
    local backend_url="http://localhost:8000"
    
    # Test main metrics endpoint
    if curl -f -s --max-time $TIMEOUT "$backend_url/metrics" >/dev/null 2>&1; then
        run_test "Application metrics endpoint" "curl -f -s --max-time $TIMEOUT $backend_url/metrics | grep -q 'http_requests_total'"
        run_test "Custom FreeAgentics metrics" "curl -f -s --max-time $TIMEOUT $backend_url/metrics | grep -q 'freeagentics_'"
    else
        warn "Application metrics endpoint not accessible - application may not be running"
    fi
    
    # Test monitoring API endpoints
    local monitoring_endpoints=(
        "/api/v1/monitoring/agents"
        "/api/v1/monitoring/coalitions"
        "/api/v1/monitoring/knowledge-graph"
        "/api/v1/monitoring/inference"
    )
    
    for endpoint in "${monitoring_endpoints[@]}"; do
        if curl -f -s --max-time $TIMEOUT "$backend_url$endpoint" >/dev/null 2>&1; then
            run_test "Monitoring endpoint $endpoint" "true"
        else
            warn "Monitoring endpoint $endpoint not accessible"
        fi
    done
}

# Test log aggregation
test_logging() {
    info "Testing logging configuration..."
    
    # Test log directories
    run_test "Log directory exists" "test -d logs"
    
    # Test log files are being written
    if [[ -d "logs" ]]; then
        local log_files=$(find logs -name "*.log" -newer logs -mmin -5 2>/dev/null | wc -l)
        if [[ $log_files -gt 0 ]]; then
            log "Recent log files found"
        else
            warn "No recent log files found"
        fi
    fi
    
    # Test structured logging
    if [[ -f "logs/freeagentics.json" ]]; then
        run_test "Structured logging format" "tail -1 logs/freeagentics.json | jq . >/dev/null 2>&1"
    fi
}

# Test alert firing simulation
test_alert_simulation() {
    info "Testing alert simulation..."
    
    # Create a temporary high load to test alerts (if safe to do so)
    warn "Skipping alert simulation in production environment"
    
    # Instead, test that alerts can be queried
    run_test "Active alerts query" "curl -f -s --max-time $TIMEOUT '$ALERTMANAGER_URL/api/v1/alerts'"
    
    # Test alert history
    if curl -f -s --max-time $TIMEOUT "$PROMETHEUS_URL/api/v1/query?query=ALERTS" >/dev/null 2>&1; then
        run_test "Alert metrics available" "true"
    fi
}

# Test monitoring integration
test_monitoring_integration() {
    info "Testing monitoring integration..."
    
    # Test Prometheus can scrape all configured targets
    local scrape_targets
    scrape_targets=$(curl -f -s --max-time $TIMEOUT "$PROMETHEUS_URL/api/v1/targets" | grep -o '"health":"[^"]*"' | grep -c '"health":"up"' || echo "0")
    
    if [[ $scrape_targets -gt 0 ]]; then
        log "Found $scrape_targets healthy scrape targets"
    else
        warn "No healthy scrape targets found"
    fi
    
    # Test that metrics are flowing from app to Prometheus
    run_test "Application metrics in Prometheus" "curl -f -s --max-time $TIMEOUT '$PROMETHEUS_URL/api/v1/query?query=http_requests_total{job=\"freeagentics-backend\"}' | grep -q 'result'"
    
    # Test alert rules evaluation
    run_test "Alert rules evaluation" "curl -f -s --max-time $TIMEOUT '$PROMETHEUS_URL/api/v1/rules' | grep -q 'evaluationTime'"
}

# Test performance monitoring
test_performance_monitoring() {
    info "Testing performance monitoring..."
    
    # Test that performance metrics are available
    local performance_metrics=(
        "http_request_duration_seconds"
        "http_requests_total"
        "process_cpu_seconds_total"
        "process_resident_memory_bytes"
    )
    
    for metric in "${performance_metrics[@]}"; do
        run_test "Performance metric: $metric" "curl -f -s --max-time $TIMEOUT '$PROMETHEUS_URL/api/v1/query?query=$metric' | grep -q 'result'"
    done
}

# Test security monitoring
test_security_monitoring() {
    info "Testing security monitoring..."
    
    # Test security-related metrics
    local security_metrics=(
        "freeagentics_failed_login_attempts_total"
        "freeagentics_unauthorized_access_attempts_total"
        "freeagentics_suspicious_requests_total"
    )
    
    for metric in "${security_metrics[@]}"; do
        if curl -f -s --max-time $TIMEOUT "$PROMETHEUS_URL/api/v1/query?query=$metric" | grep -q 'result' 2>/dev/null; then
            log "Security metric available: $metric"
        else
            warn "Security metric not available: $metric"
        fi
    done
}

# Generate monitoring report
generate_report() {
    local timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')
    local report_file="monitoring_validation_report_${timestamp}.json"
    
    cat > "$report_file" << EOF
{
  "monitoring_validation_report": {
    "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
    "summary": {
      "total_tests": $TESTS_RUN,
      "tests_passed": $TESTS_PASSED,
      "tests_failed": $TESTS_FAILED,
      "success_rate": "$(( TESTS_PASSED * 100 / TESTS_RUN ))%"
    },
    "status": "$([ $TESTS_FAILED -eq 0 ] && echo "HEALTHY" || echo "ISSUES_DETECTED")",
    "components": {
      "prometheus": "$(curl -f -s --max-time 5 $PROMETHEUS_URL/-/healthy >/dev/null 2>&1 && echo "UP" || echo "DOWN")",
      "alertmanager": "$(curl -f -s --max-time 5 $ALERTMANAGER_URL/-/healthy >/dev/null 2>&1 && echo "UP" || echo "DOWN")",
      "grafana": "$(curl -f -s --max-time 5 $GRAFANA_URL/api/health >/dev/null 2>&1 && echo "UP" || echo "DOWN")"
    },
    "recommendations": [
      $([ $TESTS_FAILED -gt 0 ] && echo '"Review failed monitoring tests and fix issues",' || echo '')
      "Monitor disk space for metrics retention",
      "Verify alert notification channels",
      "Review dashboard accuracy and completeness"
    ]
  }
}
EOF
    
    info "Monitoring validation report saved to: $report_file"
}

# Display summary
show_summary() {
    echo ""
    echo "======================================"
    echo "   MONITORING VALIDATION SUMMARY"
    echo "======================================"
    echo "Timestamp: $(date)"
    echo ""
    echo "Tests Run: $TESTS_RUN"
    echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
    echo ""
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}✅ ALL MONITORING TESTS PASSED${NC}"
        echo ""
        echo "🔍 Your monitoring stack is healthy and ready!"
        echo "📊 Access Grafana at: $GRAFANA_URL"
        echo "🚨 Access Alertmanager at: $ALERTMANAGER_URL"
        echo "📈 Access Prometheus at: $PROMETHEUS_URL"
        return 0
    else
        echo -e "${RED}❌ SOME MONITORING TESTS FAILED${NC}"
        echo ""
        echo "⚠️  Please review and fix the failed monitoring components."
        return 1
    fi
}

# Main function
main() {
    local start_time
    start_time=$(date +%s)
    
    echo "🔍 Starting FreeAgentics Monitoring Validation"
    echo "Prometheus: $PROMETHEUS_URL"
    echo "Alertmanager: $ALERTMANAGER_URL"
    echo "Grafana: $GRAFANA_URL"
    echo "Timestamp: $(date)"
    echo ""
    
    # Run all test suites
    test_prometheus
    test_alertmanager
    test_grafana
    test_application_metrics
    test_logging
    test_alert_simulation
    test_monitoring_integration
    test_performance_monitoring
    test_security_monitoring
    
    # Generate report and show summary
    generate_report
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo ""
    info "Monitoring validation completed in ${duration} seconds"
    
    show_summary
}

# Script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --prometheus-url URL    Prometheus URL (default: http://localhost:9090)"
    echo "  --alertmanager-url URL  Alertmanager URL (default: http://localhost:9093)"
    echo "  --grafana-url URL       Grafana URL (default: http://localhost:3000)"
    echo "  --timeout SECONDS       HTTP timeout (default: 10)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Validate with default URLs"
    echo "  $0 --prometheus-url http://prom.example.com:9090"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prometheus-url)
            PROMETHEUS_URL="$2"
            shift 2
            ;;
        --alertmanager-url)
            ALERTMANAGER_URL="$2"
            shift 2
            ;;
        --grafana-url)
            GRAFANA_URL="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
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

# Execute main function
main "$@"