# WebSocket Testing Guide

This guide provides comprehensive instructions for testing WebSocket connections in FreeAgentics across different environments and scenarios.

## Quick Start Testing

### Demo Mode (No Authentication)
```bash
# Test WebSocket connection without authentication
wscat -c ws://localhost:8000/api/v1/ws/demo

# Send a test message
> {"type": "test", "message": "Hello WebSocket"}
```

### Development Mode (With Authentication)
```bash
# First, get your auth token (replace with your login endpoint)
TOKEN=$(curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password"}' \
  | jq -r '.access_token')

# Connect with authentication
wscat -c "ws://localhost:8000/api/v1/ws/dev?token=$TOKEN"
```

## Testing Tools

### 1. wscat (Command Line)
```bash
# Install
npm install -g wscat

# Basic connection test
wscat -c ws://localhost:8000/api/v1/ws/demo

# With headers
wscat -c ws://localhost:8000/api/v1/ws/demo \
  -H "X-Client-Version: 1.0.0" \
  -H "X-Client-Fingerprint: test-client"

# Interactive commands in wscat:
# > {"type": "subscribe", "events": ["agent.update", "system.metrics"]}
# > {"type": "ping", "timestamp": "2024-01-01T00:00:00Z"}
# > {"type": "unsubscribe", "events": ["system.metrics"]}
```

### 2. Browser Console Testing
```javascript
// Create test WebSocket connection
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/demo');

// Setup event handlers
ws.onopen = () => {
    console.log('✅ WebSocket connected');
    // Subscribe to events
    ws.send(JSON.stringify({
        type: 'subscribe',
        events: ['agent.update', 'belief.state', 'system.metrics']
    }));
};

ws.onmessage = (event) => {
    console.log('📨 Message received:', JSON.parse(event.data));
};

ws.onerror = (error) => {
    console.error('❌ WebSocket error:', error);
};

ws.onclose = (event) => {
    console.log('🔌 WebSocket closed:', event.code, event.reason);
};

// Test functions
function testPing() {
    ws.send(JSON.stringify({
        type: 'ping',
        timestamp: new Date().toISOString()
    }));
}

function testAgentQuery() {
    ws.send(JSON.stringify({
        type: 'agent.list',
        filters: { status: 'active' }
    }));
}

// Execute tests
testPing();
setTimeout(testAgentQuery, 1000);
```

### 3. Automated Testing with Jest
```javascript
// __tests__/websocket.test.js
const WebSocket = require('ws');

describe('WebSocket API Tests', () => {
    let ws;
    
    beforeEach((done) => {
        ws = new WebSocket('ws://localhost:8000/api/v1/ws/demo');
        ws.on('open', done);
    });
    
    afterEach(() => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.close();
        }
    });
    
    test('should connect successfully', (done) => {
        expect(ws.readyState).toBe(WebSocket.OPEN);
        done();
    });
    
    test('should receive pong for ping', (done) => {
        ws.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.type === 'pong') {
                expect(message).toHaveProperty('timestamp');
                done();
            }
        });
        
        ws.send(JSON.stringify({
            type: 'ping',
            timestamp: new Date().toISOString()
        }));
    });
    
    test('should handle subscription', (done) => {
        ws.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.type === 'subscription_confirmed') {
                expect(message.events).toContain('agent.update');
                done();
            }
        });
        
        ws.send(JSON.stringify({
            type: 'subscribe',
            events: ['agent.update']
        }));
    });
});
```

## Load Testing

### Using Artillery
```yaml
# websocket-load-test.yml
config:
  target: "ws://localhost:8000"
  phases:
    - duration: 60
      arrivalRate: 10
      name: "Warm up"
    - duration: 120
      arrivalRate: 50
      name: "Sustained load"
  processor: "./websocket-processor.js"

scenarios:
  - name: "WebSocket Test"
    engine: "ws"
    flow:
      - send: '{"type": "ping", "timestamp": "{{ $timestamp }}"}'
      - think: 1
      - send: '{"type": "subscribe", "events": ["agent.update"]}'
      - think: 5
      - loop:
        - send: '{"type": "agent.query", "id": "{{ $randomNumber(1, 100) }}"}'
        - think: 2
        count: 10
```

```javascript
// websocket-processor.js
module.exports = {
    beforeRequest: (requestParams, context, ee, next) => {
        requestParams.json.timestamp = new Date().toISOString();
        return next();
    }
};
```

Run load test:
```bash
npm install -g artillery
artillery run websocket-load-test.yml
```

## Testing Scenarios

### 1. Connection Lifecycle
```javascript
// Test connection, disconnection, and reconnection
function testConnectionLifecycle() {
    let reconnectCount = 0;
    
    function connect() {
        const ws = new WebSocket('ws://localhost:8000/api/v1/ws/demo');
        
        ws.onopen = () => {
            console.log(`Connected (attempt ${reconnectCount + 1})`);
            
            // Test graceful disconnect after 5 seconds
            if (reconnectCount === 0) {
                setTimeout(() => {
                    ws.close(1000, 'Normal closure');
                }, 5000);
            }
        };
        
        ws.onclose = (event) => {
            console.log(`Disconnected: ${event.code} - ${event.reason}`);
            
            if (reconnectCount < 3) {
                reconnectCount++;
                setTimeout(connect, 2000); // Reconnect after 2 seconds
            }
        };
    }
    
    connect();
}
```

### 2. Authentication Testing
```javascript
// Test authentication flow
async function testAuthentication() {
    // Test 1: No token (should fail in dev mode)
    const ws1 = new WebSocket('ws://localhost:8000/api/v1/ws/dev');
    ws1.onerror = () => console.log('✅ Correctly rejected connection without token');
    
    // Test 2: Invalid token
    const ws2 = new WebSocket('ws://localhost:8000/api/v1/ws/dev?token=invalid_token');
    ws2.onerror = () => console.log('✅ Correctly rejected invalid token');
    
    // Test 3: Valid token
    const token = await getValidToken(); // Implement based on your auth
    const ws3 = new WebSocket(`ws://localhost:8000/api/v1/ws/dev?token=${token}`);
    ws3.onopen = () => console.log('✅ Successfully connected with valid token');
}
```

### 3. Rate Limiting Testing
```javascript
// Test rate limiting behavior
function testRateLimiting() {
    const ws = new WebSocket('ws://localhost:8000/api/v1/ws/demo');
    let messageCount = 0;
    let rateLimitHit = false;
    
    ws.onopen = () => {
        // Send messages rapidly
        const interval = setInterval(() => {
            if (messageCount >= 150) { // Exceed rate limit
                clearInterval(interval);
                console.log(`Sent ${messageCount} messages`);
                return;
            }
            
            ws.send(JSON.stringify({
                type: 'test',
                id: messageCount++
            }));
        }, 10); // 100 messages per second
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'error' && data.code === 'RATE_LIMIT_EXCEEDED') {
            rateLimitHit = true;
            console.log('✅ Rate limit correctly enforced');
        }
    };
}
```

### 4. Error Handling Testing
```javascript
// Test various error scenarios
function testErrorHandling() {
    const ws = new WebSocket('ws://localhost:8000/api/v1/ws/demo');
    
    ws.onopen = () => {
        // Test 1: Invalid JSON
        ws.send('invalid json {]');
        
        // Test 2: Unknown message type
        ws.send(JSON.stringify({
            type: 'unknown_type',
            data: 'test'
        }));
        
        // Test 3: Missing required fields
        ws.send(JSON.stringify({
            type: 'subscribe'
            // Missing 'events' field
        }));
        
        // Test 4: Oversized message
        const largeData = 'x'.repeat(65536); // 64KB+
        ws.send(JSON.stringify({
            type: 'test',
            data: largeData
        }));
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'error') {
            console.log(`✅ Error handled: ${data.code} - ${data.message}`);
        }
    };
}
```

## Performance Testing

### Message Latency Test
```javascript
function testLatency(duration = 60000) {
    const ws = new WebSocket('ws://localhost:8000/api/v1/ws/demo');
    const latencies = [];
    let testStart;
    
    ws.onopen = () => {
        testStart = Date.now();
        console.log('Starting latency test...');
        
        // Send ping every second
        const interval = setInterval(() => {
            if (Date.now() - testStart > duration) {
                clearInterval(interval);
                analyzeResults();
                ws.close();
                return;
            }
            
            ws.send(JSON.stringify({
                type: 'ping',
                timestamp: Date.now(),
                id: latencies.length
            }));
        }, 1000);
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'pong') {
            const latency = Date.now() - data.timestamp;
            latencies.push(latency);
        }
    };
    
    function analyzeResults() {
        const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
        const sorted = [...latencies].sort((a, b) => a - b);
        const p50 = sorted[Math.floor(sorted.length * 0.5)];
        const p95 = sorted[Math.floor(sorted.length * 0.95)];
        const p99 = sorted[Math.floor(sorted.length * 0.99)];
        
        console.log('Latency Test Results:');
        console.log(`  Samples: ${latencies.length}`);
        console.log(`  Average: ${avg.toFixed(2)}ms`);
        console.log(`  P50: ${p50}ms`);
        console.log(`  P95: ${p95}ms`);
        console.log(`  P99: ${p99}ms`);
        console.log(`  Min: ${Math.min(...latencies)}ms`);
        console.log(`  Max: ${Math.max(...latencies)}ms`);
    }
}
```

## Debugging Tips

### 1. Enable Debug Logging
```bash
# Backend
export ENABLE_WEBSOCKET_LOGGING=true
export LOG_LEVEL=DEBUG

# Frontend (in browser console)
localStorage.setItem('DEBUG', 'websocket:*');
```

### 2. Monitor Network Traffic
```bash
# Using tcpdump
sudo tcpdump -i lo -A 'tcp port 8000'

# Using Wireshark
# Filter: tcp.port == 8000 && websocket
```

### 3. Check WebSocket Headers
```bash
# Verify upgrade headers
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==" \
  http://localhost:8000/api/v1/ws/demo
```

## Common Issues and Solutions

1. **Connection Immediately Closes**
   - Check authentication token validity
   - Verify WebSocket endpoint URL
   - Check server logs for errors

2. **Messages Not Received**
   - Ensure proper event subscription
   - Check message format/schema
   - Verify no rate limiting

3. **High Latency**
   - Check network conditions
   - Monitor server load
   - Review message size

4. **Connection Drops**
   - Implement heartbeat/ping mechanism
   - Check proxy/load balancer timeout settings
   - Monitor for memory leaks