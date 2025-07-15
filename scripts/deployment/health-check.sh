#!/bin/bash
# Health check script for FreeAgentics deployment

set -euo pipefail

# Configuration
ENVIRONMENT="${1:-production}"
API_URL="${API_URL:-https://api.freeagentics.com}"
TIMEOUT="${TIMEOUT:-10}"
MAX_RETRIES="${MAX_RETRIES:-5}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[HEALTH CHECK]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

# Check function with retries
check_endpoint() {
    local endpoint=$1
    local expected_status=${2:-200}
    local retry_count=0
    
    while [[ $retry_count -lt $MAX_RETRIES ]]; do
        log "Checking $endpoint (attempt $((retry_count + 1))/$MAX_RETRIES)..."
        
        response=$(curl -s -o /dev/null -w "%{http_code}" \
            --connect-timeout "$TIMEOUT" \
            --max-time "$TIMEOUT" \
            "$API_URL$endpoint" || echo "000")
        
        if [[ "$response" == "$expected_status" ]]; then
            log "✓ $endpoint responded with $response"
            return 0
        else
            warning "$endpoint returned $response (expected $expected_status)"
            retry_count=$((retry_count + 1))
            if [[ $retry_count -lt $MAX_RETRIES ]]; then
                sleep 5
            fi
        fi
    done
    
    error "✗ $endpoint failed after $MAX_RETRIES attempts"
    return 1
}

# Check JSON response
check_json_endpoint() {
    local endpoint=$1
    local json_path=$2
    local expected_value=$3
    
    log "Checking JSON response from $endpoint..."
    
    response=$(curl -s --connect-timeout "$TIMEOUT" --max-time "$TIMEOUT" \
        "$API_URL$endpoint" | jq -r "$json_path" 2>/dev/null || echo "error")
    
    if [[ "$response" == "$expected_value" ]]; then
        log "✓ $endpoint returned expected value: $response"
        return 0
    else
        error "✗ $endpoint returned unexpected value: $response (expected: $expected_value)"
        return 1
    fi
}

# Main health checks
run_health_checks() {
    local failed=0
    
    echo -e "\n${GREEN}=== Running Health Checks ===${NC}\n"
    
    # 1. Basic connectivity
    log "1. Testing basic connectivity..."
    check_endpoint "/health" 200 || ((failed++))
    
    # 2. API status endpoint
    log "2. Testing API status..."
    check_endpoint "/v1/system/status" 200 || ((failed++))
    
    # 3. Database connectivity
    log "3. Testing database connectivity..."
    check_json_endpoint "/v1/system/status" ".database.connected" "true" || ((failed++))
    
    # 4. Redis connectivity
    log "4. Testing Redis connectivity..."
    check_json_endpoint "/v1/system/status" ".redis.connected" "true" || ((failed++))
    
    # 5. Authentication service
    log "5. Testing authentication service..."
    check_endpoint "/v1/auth/health" 200 || ((failed++))
    
    # 6. WebSocket service
    log "6. Testing WebSocket service..."
    ws_response=$(echo "ping" | timeout 5 wscat -c "wss://api.freeagentics.com/ws/health" 2>&1 || echo "failed")
    if echo "$ws_response" | grep -q "pong"; then
        log "✓ WebSocket service responded"
    else
        error "✗ WebSocket service failed to respond"
        ((failed++))
    fi
    
    # 7. Metrics endpoint
    log "7. Testing metrics endpoint..."
    check_endpoint "/metrics" 200 || ((failed++))
    
    # 8. Static assets (if applicable)
    log "8. Testing static asset serving..."
    check_endpoint "/static/health.txt" 200 || warning "Static assets not configured"
    
    # 9. Rate limiting
    log "9. Testing rate limiting..."
    for i in {1..5}; do
        curl -s -o /dev/null "$API_URL/v1/auth/login" &
    done
    wait
    
    rate_limit_response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/v1/auth/login")
    if [[ "$rate_limit_response" == "429" ]]; then
        log "✓ Rate limiting is active"
    else
        warning "Rate limiting might not be properly configured"
    fi
    
    # 10. SSL/TLS configuration
    log "10. Testing SSL/TLS configuration..."
    ssl_check=$(echo | openssl s_client -connect api.freeagentics.com:443 -servername api.freeagentics.com 2>/dev/null | openssl x509 -noout -dates 2>/dev/null)
    if [[ -n "$ssl_check" ]]; then
        log "✓ SSL certificate is valid"
        echo "$ssl_check" | grep "notAfter"
    else
        error "✗ SSL certificate check failed"
        ((failed++))
    fi
    
    # Summary
    echo -e "\n${GREEN}=== Health Check Summary ===${NC}"
    if [[ $failed -eq 0 ]]; then
        log "All health checks passed! ✓"
        return 0
    else
        error "$failed health checks failed! ✗"
        return 1
    fi
}

# Performance checks
run_performance_checks() {
    echo -e "\n${GREEN}=== Running Performance Checks ===${NC}\n"
    
    # Response time check
    log "Checking API response times..."
    total_time=0
    for i in {1..10}; do
        response_time=$(curl -s -o /dev/null -w "%{time_total}" "$API_URL/health")
        total_time=$(echo "$total_time + $response_time" | bc)
    done
    avg_time=$(echo "scale=3; $total_time / 10" | bc)
    
    log "Average response time: ${avg_time}s"
    if (( $(echo "$avg_time < 0.5" | bc -l) )); then
        log "✓ Response time is acceptable"
    else
        warning "Response time is higher than expected"
    fi
}

# Database checks
check_database_health() {
    echo -e "\n${GREEN}=== Database Health Checks ===${NC}\n"
    
    # Get database metrics
    db_metrics=$(curl -s "$API_URL/v1/monitoring/database" || echo "{}")
    
    # Connection pool
    active_connections=$(echo "$db_metrics" | jq -r '.connections.active' 2>/dev/null || echo "0")
    max_connections=$(echo "$db_metrics" | jq -r '.connections.max' 2>/dev/null || echo "100")
    
    log "Database connections: $active_connections/$max_connections"
    
    # Replication lag (if applicable)
    replication_lag=$(echo "$db_metrics" | jq -r '.replication.lag_seconds' 2>/dev/null || echo "0")
    if [[ "$replication_lag" != "null" ]] && [[ "$replication_lag" -lt 5 ]]; then
        log "✓ Replication lag: ${replication_lag}s"
    elif [[ "$replication_lag" != "null" ]]; then
        warning "High replication lag: ${replication_lag}s"
    fi
}

# Service dependency checks
check_service_dependencies() {
    echo -e "\n${GREEN}=== Service Dependency Checks ===${NC}\n"
    
    # Check Redis
    log "Checking Redis operations..."
    test_key="health_check_$(date +%s)"
    set_result=$(curl -s -X POST "$API_URL/v1/system/redis-test" \
        -H "Content-Type: application/json" \
        -d "{\"action\":\"set\",\"key\":\"$test_key\",\"value\":\"test\"}" | \
        jq -r '.success' 2>/dev/null || echo "false")
    
    if [[ "$set_result" == "true" ]]; then
        log "✓ Redis write operation successful"
    else
        error "✗ Redis write operation failed"
    fi
    
    # Check external APIs (if any)
    log "Checking external service connectivity..."
    # Add checks for any external services your app depends on
}

# Main execution
main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --api-url)
                API_URL="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --quick)
                QUICK_CHECK=true
                shift
                ;;
            *)
                ENVIRONMENT="$1"
                shift
                ;;
        esac
    done
    
    log "Starting health checks for environment: $ENVIRONMENT"
    log "API URL: $API_URL"
    
    # Run checks
    if [[ "${QUICK_CHECK:-false}" == "true" ]]; then
        # Quick check - just basic endpoints
        check_endpoint "/health" 200
    else
        # Full health check suite
        run_health_checks
        run_performance_checks
        check_database_health
        check_service_dependencies
    fi
}

# Execute main
main "$@"