#!/bin/bash
# Smoke tests for deployment validation

set -euo pipefail

ENVIRONMENT=${1:-staging}
BASE_URL=""

# Set base URL based on environment
case $ENVIRONMENT in
    staging)
        BASE_URL="https://staging.freeagentics.com"
        ;;
    production)
        BASE_URL="https://freeagentics.com"
        ;;
    *)
        echo "Unknown environment: $ENVIRONMENT"
        exit 1
        ;;
esac

echo "Running smoke tests for $ENVIRONMENT environment..."
echo "Base URL: $BASE_URL"

# Test 1: Health check endpoint
echo -n "Testing health endpoint... "
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
if [ "$HEALTH_RESPONSE" -eq 200 ]; then
    echo "✓ PASSED"
else
    echo "✗ FAILED (HTTP $HEALTH_RESPONSE)"
    exit 1
fi

# Test 2: API endpoint
echo -n "Testing API health endpoint... "
API_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/v1/health")
if [ "$API_RESPONSE" -eq 200 ]; then
    echo "✓ PASSED"
else
    echo "✗ FAILED (HTTP $API_RESPONSE)"
    exit 1
fi

# Test 3: Database connectivity
echo -n "Testing database connectivity... "
DB_CHECK=$(curl -s "$BASE_URL/api/v1/health/db" | jq -r '.status')
if [ "$DB_CHECK" = "healthy" ]; then
    echo "✓ PASSED"
else
    echo "✗ FAILED"
    exit 1
fi

# Test 4: Redis connectivity
echo -n "Testing Redis connectivity... "
REDIS_CHECK=$(curl -s "$BASE_URL/api/v1/health/redis" | jq -r '.status')
if [ "$REDIS_CHECK" = "healthy" ]; then
    echo "✓ PASSED"
else
    echo "✗ FAILED"
    exit 1
fi

# Test 5: Authentication endpoint
echo -n "Testing authentication endpoint... "
AUTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/v1/auth/login" -X POST -H "Content-Type: application/json" -d '{}')
if [ "$AUTH_RESPONSE" -eq 400 ] || [ "$AUTH_RESPONSE" -eq 422 ]; then
    echo "✓ PASSED (Expected validation error)"
else
    echo "✗ FAILED (Unexpected response: HTTP $AUTH_RESPONSE)"
    exit 1
fi

# Test 6: Frontend availability
echo -n "Testing frontend availability... "
FRONTEND_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL")
if [ "$FRONTEND_RESPONSE" -eq 200 ]; then
    echo "✓ PASSED"
else
    echo "✗ FAILED (HTTP $FRONTEND_RESPONSE)"
    exit 1
fi

# Test 7: WebSocket endpoint
echo -n "Testing WebSocket upgrade... "
WS_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Connection: Upgrade" \
    -H "Upgrade: websocket" \
    -H "Sec-WebSocket-Version: 13" \
    -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
    "$BASE_URL/api/v1/ws")
if [ "$WS_RESPONSE" -eq 101 ] || [ "$WS_RESPONSE" -eq 426 ]; then
    echo "✓ PASSED"
else
    echo "✗ FAILED (HTTP $WS_RESPONSE)"
    exit 1
fi

# Test 8: Security headers
echo -n "Testing security headers... "
HEADERS=$(curl -s -I "$BASE_URL")
if echo "$HEADERS" | grep -q "X-Content-Type-Options: nosniff" && \
   echo "$HEADERS" | grep -q "X-Frame-Options: DENY" && \
   echo "$HEADERS" | grep -q "Strict-Transport-Security:"; then
    echo "✓ PASSED"
else
    echo "✗ FAILED (Missing security headers)"
    exit 1
fi

# Test 9: Rate limiting
echo -n "Testing rate limiting... "
for i in {1..11}; do
    RATE_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/v1/health")
    if [ "$i" -eq 11 ] && [ "$RATE_RESPONSE" -eq 429 ]; then
        echo "✓ PASSED (Rate limit enforced)"
        break
    elif [ "$i" -eq 11 ]; then
        echo "⚠ WARNING (Rate limiting may not be configured)"
    fi
done

# Test 10: SSL/TLS configuration
echo -n "Testing SSL/TLS configuration... "
SSL_CHECK=$(echo | openssl s_client -connect "${BASE_URL#https://}:443" 2>/dev/null | openssl x509 -noout -dates 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✓ PASSED"
else
    echo "✗ FAILED"
    exit 1
fi

echo ""
echo "All smoke tests passed for $ENVIRONMENT environment!"
exit 0
