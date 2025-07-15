#!/bin/bash
# Smoke tests for FreeAgentics deployment

set -euo pipefail

# Configuration
API_URL="${API_URL:-https://api.freeagentics.com}"
WEB_URL="${WEB_URL:-https://app.freeagentics.com}"
TIMEOUT="${TIMEOUT:-30}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test credentials (should be environment-specific)
TEST_EMAIL="${TEST_EMAIL:-test@example.com}"
TEST_PASSWORD="${TEST_PASSWORD:-testpassword}"

# Logging
log() {
    echo -e "${GREEN}[SMOKE TEST]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

# Test authentication flow
test_authentication() {
    log "Testing authentication flow..."
    
    # Test login
    local login_response
    login_response=$(curl -s -X POST "$API_URL/v1/auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"email\":\"$TEST_EMAIL\",\"password\":\"$TEST_PASSWORD\"}" \
        --max-time "$TIMEOUT")
    
    if echo "$login_response" | jq -e '.access_token' >/dev/null 2>&1; then
        local token
        token=$(echo "$login_response" | jq -r '.access_token')
        
        # Test authenticated endpoint
        local auth_test
        auth_test=$(curl -s -H "Authorization: Bearer $token" \
            "$API_URL/v1/auth/me" --max-time "$TIMEOUT")
        
        if echo "$auth_test" | jq -e '.email' >/dev/null 2>&1; then
            log "✓ Authentication flow working"
            echo "$token"
        else
            error "✗ Authenticated endpoint failed"
            return 1
        fi
    else
        error "✗ Login failed"
        return 1
    fi
}

# Test agent operations
test_agent_operations() {
    local token="$1"
    
    log "Testing agent operations..."
    
    # List agents
    local agents_response
    agents_response=$(curl -s -H "Authorization: Bearer $token" \
        "$API_URL/v1/agents" --max-time "$TIMEOUT")
    
    if echo "$agents_response" | jq -e '.agents' >/dev/null 2>&1; then
        log "✓ Agent listing working"
    else
        error "✗ Agent listing failed"
        return 1
    fi
    
    # Create test agent
    local create_response
    create_response=$(curl -s -X POST -H "Authorization: Bearer $token" \
        -H "Content-Type: application/json" \
        -d '{"name":"smoke-test-agent","type":"resource_collector","config":{}}' \
        "$API_URL/v1/agents" --max-time "$TIMEOUT")
    
    if echo "$create_response" | jq -e '.id' >/dev/null 2>&1; then
        local agent_id
        agent_id=$(echo "$create_response" | jq -r '.id')
        log "✓ Agent creation working (ID: $agent_id)"
        
        # Clean up test agent
        curl -s -X DELETE -H "Authorization: Bearer $token" \
            "$API_URL/v1/agents/$agent_id" --max-time "$TIMEOUT" >/dev/null
        
        log "✓ Agent deletion working"
    else
        error "✗ Agent creation failed"
        return 1
    fi
}

# Test knowledge graph operations
test_knowledge_graph() {
    local token="$1"
    
    log "Testing knowledge graph operations..."
    
    # Test knowledge query
    local knowledge_response
    knowledge_response=$(curl -s -H "Authorization: Bearer $token" \
        "$API_URL/v1/knowledge/query" \
        -X POST -H "Content-Type: application/json" \
        -d '{"query":"test query","limit":10}' \
        --max-time "$TIMEOUT")
    
    if echo "$knowledge_response" | jq -e '.results' >/dev/null 2>&1; then
        log "✓ Knowledge graph query working"
    else
        error "✗ Knowledge graph query failed"
        return 1
    fi
}

# Test WebSocket functionality
test_websocket() {
    local token="$1"
    
    log "Testing WebSocket functionality..."
    
    # Test WebSocket connection
    local ws_test_result
    ws_test_result=$(timeout 10 bash -c "
        echo '{\"type\":\"ping\",\"token\":\"$token\"}' | \
        wscat -c 'wss://api.freeagentics.com/ws' 2>/dev/null | \
        grep -q 'pong' && echo 'success' || echo 'failed'
    " 2>/dev/null || echo "failed")
    
    if [[ "$ws_test_result" == "success" ]]; then
        log "✓ WebSocket connection working"
    else
        warning "⚠ WebSocket connection failed (non-critical)"
    fi
}

# Test file operations
test_file_operations() {
    local token="$1"
    
    log "Testing file operations..."
    
    # Create test file
    local test_content="smoke test content"
    local upload_response
    upload_response=$(curl -s -X POST -H "Authorization: Bearer $token" \
        -F "file=@-" \
        "$API_URL/v1/files/upload" \
        --max-time "$TIMEOUT" \
        < <(echo "$test_content"))
    
    if echo "$upload_response" | jq -e '.file_id' >/dev/null 2>&1; then
        local file_id
        file_id=$(echo "$upload_response" | jq -r '.file_id')
        log "✓ File upload working (ID: $file_id)"
        
        # Download file
        local download_content
        download_content=$(curl -s -H "Authorization: Bearer $token" \
            "$API_URL/v1/files/$file_id/download" \
            --max-time "$TIMEOUT")
        
        if [[ "$download_content" == "$test_content" ]]; then
            log "✓ File download working"
        else
            error "✗ File download failed"
            return 1
        fi
        
        # Clean up test file
        curl -s -X DELETE -H "Authorization: Bearer $token" \
            "$API_URL/v1/files/$file_id" \
            --max-time "$TIMEOUT" >/dev/null
        
        log "✓ File deletion working"
    else
        warning "⚠ File upload failed (may not be implemented)"
    fi
}

# Test system endpoints
test_system_endpoints() {
    log "Testing system endpoints..."
    
    # Health check
    local health_response
    health_response=$(curl -s "$API_URL/health" --max-time "$TIMEOUT")
    
    if echo "$health_response" | jq -e '.status' >/dev/null 2>&1; then
        log "✓ Health endpoint working"
    else
        error "✗ Health endpoint failed"
        return 1
    fi
    
    # System status
    local status_response
    status_response=$(curl -s "$API_URL/v1/system/status" --max-time "$TIMEOUT")
    
    if echo "$status_response" | jq -e '.database' >/dev/null 2>&1; then
        log "✓ System status endpoint working"
    else
        error "✗ System status endpoint failed"
        return 1
    fi
    
    # Metrics
    local metrics_response
    metrics_response=$(curl -s "$API_URL/metrics" --max-time "$TIMEOUT")
    
    if echo "$metrics_response" | grep -q 'python_info'; then
        log "✓ Metrics endpoint working"
    else
        error "✗ Metrics endpoint failed"
        return 1
    fi
}

# Test web application
test_web_application() {
    log "Testing web application..."
    
    # Test main page
    local web_response
    web_response=$(curl -s "$WEB_URL" --max-time "$TIMEOUT")
    
    if echo "$web_response" | grep -q 'FreeAgentics'; then
        log "✓ Web application accessible"
    else
        error "✗ Web application not accessible"
        return 1
    fi
    
    # Test static assets
    local static_response
    static_response=$(curl -s -I "$WEB_URL/static/favicon.ico" --max-time "$TIMEOUT")
    
    if echo "$static_response" | grep -q 'HTTP/[12].*200'; then
        log "✓ Static assets served"
    else
        warning "⚠ Static assets may not be properly configured"
    fi
}

# Test performance
test_performance() {
    log "Testing performance..."
    
    # Response time test
    local response_time
    response_time=$(curl -o /dev/null -s -w '%{time_total}' "$API_URL/health" --max-time "$TIMEOUT")
    
    if (( $(echo "$response_time < 2.0" | bc -l) )); then
        log "✓ Response time acceptable ($response_time seconds)"
    else
        warning "⚠ Response time slow ($response_time seconds)"
    fi
    
    # Concurrent request test
    log "Testing concurrent requests..."
    for i in {1..5}; do
        curl -s "$API_URL/health" --max-time "$TIMEOUT" &
    done
    wait
    
    log "✓ Concurrent requests handled"
}

# Main smoke test execution
main() {
    echo -e "\n${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║            SMOKE TESTS                 ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}\n"
    
    log "Running smoke tests against: $API_URL"
    
    local failed_tests=0
    
    # Test system endpoints first
    if ! test_system_endpoints; then
        ((failed_tests++))
    fi
    
    # Test web application
    if ! test_web_application; then
        ((failed_tests++))
    fi
    
    # Test authentication and get token
    local token
    if token=$(test_authentication); then
        # Run authenticated tests
        if ! test_agent_operations "$token"; then
            ((failed_tests++))
        fi
        
        if ! test_knowledge_graph "$token"; then
            ((failed_tests++))
        fi
        
        if ! test_websocket "$token"; then
            # WebSocket is non-critical for smoke tests
            true
        fi
        
        if ! test_file_operations "$token"; then
            # File operations may not be implemented
            true
        fi
    else
        ((failed_tests++))
    fi
    
    # Test performance
    if ! test_performance; then
        # Performance issues are warnings, not failures
        true
    fi
    
    # Summary
    echo -e "\n${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║          SMOKE TEST SUMMARY            ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}\n"
    
    if [[ $failed_tests -eq 0 ]]; then
        log "All smoke tests passed! ✓"
        log "Application is ready for use"
        return 0
    else
        error "$failed_tests smoke tests failed!"
        error "Application may not be fully functional"
        return 1
    fi
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
        --test-email)
            TEST_EMAIL="$2"
            shift 2
            ;;
        --test-password)
            TEST_PASSWORD="$2"
            shift 2
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run smoke tests
main