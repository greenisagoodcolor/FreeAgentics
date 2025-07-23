# WebSocket API Documentation

## Overview

The FreeAgentics WebSocket API provides real-time bidirectional communication for agent monitoring, system metrics, and live updates. This enables building responsive dashboards, real-time monitoring systems, and interactive agent management interfaces.

## Connection Details

### WebSocket URL
```
wss://api.freeagentics.com/api/v1/ws
```

### Authentication
Include your JWT access token as a query parameter:
```
wss://api.freeagentics.com/api/v1/ws?token=<access_token>
```

### Connection Headers
Optional headers for enhanced security:
```
X-Client-Fingerprint: <unique_client_identifier>
X-Client-Version: <client_version>
```

## Connection Flow

### 1. Initial Connection
```javascript
const ws = new WebSocket('wss://api.freeagentics.com/api/v1/ws?token=your_access_token');

ws.onopen = function(event) {
    console.log('Connected to FreeAgentics WebSocket');
};
```

### 2. Connection Acknowledgment
Upon successful connection, the server sends:
```json
{
  "type": "connection_established",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "client_id": "client_789",
    "pooled": true,
    "session_id": "session_123"
  }
}
```

### 3. Subscribe to Events
```javascript
ws.send(JSON.stringify({
  "type": "subscribe",
  "data": {
    "events": ["agent_status", "metrics", "system_alerts"],
    "agents": ["agent_123", "agent_456"]
  }
}));
```

## Message Format

All WebSocket messages follow this structure:

```json
{
  "type": "message_type",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    // Message-specific data
  }
}
```

## Client → Server Messages

### Subscribe to Events
Subscribe to specific event types and agents:

```json
{
  "type": "subscribe",
  "data": {
    "events": ["agent_status", "metrics", "inference_completed", "system_alerts"],
    "agents": ["agent_123", "agent_456"],
    "filters": {
      "severity": ["warning", "error"],
      "components": ["inference_engine"]
    }
  }
}
```

**Parameters:**
- `events`: Array of event types to subscribe to
- `agents`: Array of specific agent IDs (optional, omit for all agents)
- `filters`: Additional filtering criteria (optional)

### Unsubscribe from Events
```json
{
  "type": "unsubscribe",
  "data": {
    "events": ["metrics"],
    "agents": ["agent_123"]
  }
}
```

### Send Commands
Execute commands on agents:
```json
{
  "type": "command",
  "data": {
    "agent_id": "agent_123",
    "command": "stop",
    "parameters": {
      "graceful": true
    }
  }
}
```

**Available Commands:**
- `start`: Start agent processing
- `stop`: Stop agent processing
- `restart`: Restart agent
- `update_config`: Update agent configuration
- `get_status`: Request immediate status update

### Ping/Pong
Keep connection alive:
```json
{
  "type": "ping"
}
```

### Configuration Updates
Update subscription configuration:
```json
{
  "type": "configure",
  "data": {
    "metrics_interval": 5000,
    "buffer_size": 100,
    "compression": true
  }
}
```

## Server → Client Messages

### Connection Events

#### Connection Established
```json
{
  "type": "connection_established",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "client_id": "client_789",
    "pooled": true,
    "session_id": "session_123",
    "server_version": "0.1.0"
  }
}
```

#### Subscription Confirmed
```json
{
  "type": "subscription_confirmed",
  "timestamp": "2024-01-15T10:30:15Z",
  "data": {
    "events": ["agent_status", "metrics"],
    "agents": ["agent_123", "agent_456"],
    "status": "active"
  }
}
```

### Agent Events

#### Agent Status Update
```json
{
  "type": "agent_status",
  "timestamp": "2024-01-15T10:31:00Z",
  "data": {
    "agent_id": "agent_123",
    "old_status": "pending",
    "new_status": "active",
    "reason": "started_by_user",
    "last_inference": "2024-01-15T10:30:45Z",
    "inference_count": 1543
  }
}
```

#### Agent Created
```json
{
  "type": "agent_created",
  "timestamp": "2024-01-15T10:32:00Z",
  "data": {
    "agent_id": "agent_789",
    "name": "New Research Agent",
    "template": "research_v2",
    "created_by": "user_456",
    "status": "pending"
  }
}
```

#### Agent Deleted
```json
{
  "type": "agent_deleted",
  "timestamp": "2024-01-15T10:33:00Z",
  "data": {
    "agent_id": "agent_456",
    "name": "Old Agent",
    "deleted_by": "user_123",
    "reason": "user_request"
  }
}
```

### Inference Events

#### Inference Started
```json
{
  "type": "inference_started",
  "timestamp": "2024-01-15T10:34:00Z",
  "data": {
    "inference_id": "inf_789",
    "agent_id": "agent_123",
    "query": "Analyze market trends",
    "estimated_duration": 30,
    "priority": "normal"
  }
}
```

#### Inference Progress
```json
{
  "type": "inference_progress",
  "timestamp": "2024-01-15T10:34:15Z",
  "data": {
    "inference_id": "inf_789",
    "agent_id": "agent_123",
    "progress": 0.65,
    "stage": "analysis",
    "estimated_remaining": 10
  }
}
```

#### Inference Completed
```json
{
  "type": "inference_completed",
  "timestamp": "2024-01-15T10:34:30Z",
  "data": {
    "inference_id": "inf_789",
    "agent_id": "agent_123",
    "status": "completed",
    "processing_time": 30.5,
    "result_summary": {
      "confidence": 0.89,
      "key_findings": ["trend_1", "trend_2"]
    }
  }
}
```

#### Inference Failed
```json
{
  "type": "inference_failed",
  "timestamp": "2024-01-15T10:34:30Z",
  "data": {
    "inference_id": "inf_789",
    "agent_id": "agent_123",
    "error_code": "TIMEOUT",
    "error_message": "Inference timed out after 300 seconds",
    "processing_time": 300.0
  }
}
```

### Metrics Events

#### Real-time Metrics
```json
{
  "type": "metrics_update",
  "timestamp": "2024-01-15T10:35:00Z",
  "data": {
    "system": {
      "cpu_usage": 45.2,
      "memory_usage": 62.8,
      "active_agents": 12,
      "queue_length": 23
    },
    "agents": {
      "agent_123": {
        "status": "active",
        "inference_rate": 2.5,
        "avg_response_time": 0.234,
        "success_rate": 0.98,
        "current_load": 0.65
      },
      "agent_456": {
        "status": "idle",
        "inference_rate": 0.0,
        "avg_response_time": 0.0,
        "success_rate": 1.0,
        "current_load": 0.0
      }
    }
  }
}
```

#### Performance Alert
```json
{
  "type": "performance_alert",
  "timestamp": "2024-01-15T10:36:00Z",
  "data": {
    "alert_id": "alert_123",
    "severity": "warning",
    "type": "high_memory_usage",
    "message": "Memory usage above 80%",
    "affected_components": ["inference_engine"],
    "current_value": 85.2,
    "threshold": 80.0,
    "recommended_action": "Scale up resources"
  }
}
```

### System Events

#### System Alert
```json
{
  "type": "system_alert",
  "timestamp": "2024-01-15T10:37:00Z",
  "data": {
    "alert_id": "alert_456",
    "severity": "error",
    "type": "service_unavailable",
    "message": "Database connection lost",
    "affected_components": ["database", "knowledge_graph"],
    "status": "investigating",
    "estimated_resolution": "2024-01-15T10:45:00Z"
  }
}
```

#### System Status Change
```json
{
  "type": "system_status",
  "timestamp": "2024-01-15T10:38:00Z",
  "data": {
    "old_status": "healthy",
    "new_status": "degraded",
    "reason": "high_load",
    "components": {
      "database": "healthy",
      "redis": "healthy",
      "inference_engine": "degraded"
    }
  }
}
```

### Knowledge Graph Events

#### Entity Created
```json
{
  "type": "entity_created",
  "timestamp": "2024-01-15T10:39:00Z",
  "data": {
    "entity_id": "entity_789",
    "type": "concept",
    "label": "Quantum Computing",
    "created_by": "user_123",
    "properties": {
      "category": "technology",
      "importance": "high"
    }
  }
}
```

#### Relationship Created
```json
{
  "type": "relationship_created",
  "timestamp": "2024-01-15T10:40:00Z",
  "data": {
    "relationship_id": "rel_456",
    "source_id": "entity_123",
    "target_id": "entity_789",
    "relationship_type": "related_to",
    "strength": 0.85,
    "created_by": "user_123"
  }
}
```

### Error Events

#### Command Error
```json
{
  "type": "command_error",
  "timestamp": "2024-01-15T10:41:00Z",
  "data": {
    "command": "start",
    "agent_id": "agent_123",
    "error_code": "AGENT_NOT_FOUND",
    "error_message": "Agent with ID 'agent_123' not found",
    "request_id": "req_789"
  }
}
```

#### Subscription Error
```json
{
  "type": "subscription_error",
  "timestamp": "2024-01-15T10:42:00Z",
  "data": {
    "error_code": "INVALID_EVENT_TYPE",
    "error_message": "Event type 'invalid_event' is not supported",
    "supported_events": ["agent_status", "metrics", "inference_completed"]
  }
}
```

### Heartbeat
```json
{
  "type": "pong",
  "timestamp": "2024-01-15T10:43:00Z",
  "data": {
    "server_time": "2024-01-15T10:43:00Z",
    "connection_duration": 780
  }
}
```

## Event Types Reference

### Agent Events
- `agent_status`: Agent status changes
- `agent_created`: New agent created
- `agent_deleted`: Agent deleted
- `agent_updated`: Agent configuration updated

### Inference Events
- `inference_started`: Inference request started
- `inference_progress`: Inference progress updates
- `inference_completed`: Inference completed successfully
- `inference_failed`: Inference failed

### Metrics Events
- `metrics_update`: Real-time system and agent metrics
- `performance_alert`: Performance-related alerts

### System Events
- `system_alert`: System-wide alerts
- `system_status`: System status changes
- `maintenance_mode`: Maintenance mode notifications

### Knowledge Graph Events
- `entity_created`: New entity created
- `entity_updated`: Entity modified
- `entity_deleted`: Entity deleted
- `relationship_created`: New relationship created
- `relationship_updated`: Relationship modified
- `relationship_deleted`: Relationship deleted

### Connection Events
- `connection_established`: Connection confirmed
- `subscription_confirmed`: Subscription confirmed
- `subscription_error`: Subscription failed
- `command_error`: Command execution failed
- `rate_limit_warning`: Rate limit approaching
- `disconnect_warning`: Disconnection imminent

## Rate Limiting

WebSocket connections are subject to rate limiting:

### Connection Limits
- Maximum concurrent connections per user: 5
- Maximum connections per IP: 20
- Connection rate: 10 per minute

### Message Limits
- Messages per minute: 100
- Message size limit: 64KB
- Subscription limit: 20 event types per connection

### Rate Limit Headers
Rate limit information is included in error responses:
```json
{
  "type": "rate_limit_warning",
  "timestamp": "2024-01-15T10:44:00Z",
  "data": {
    "limit_type": "messages_per_minute",
    "current_usage": 95,
    "limit": 100,
    "reset_time": "2024-01-15T10:45:00Z",
    "recommended_action": "Reduce message frequency"
  }
}
```

## Error Handling

### Connection Errors
```json
{
  "type": "connection_error",
  "timestamp": "2024-01-15T10:45:00Z",
  "data": {
    "error_code": "AUTHENTICATION_FAILED",
    "error_message": "Invalid or expired token",
    "close_code": 1008,
    "retry_after": 30
  }
}
```

### Common Error Codes
- `AUTHENTICATION_FAILED`: Invalid or expired token
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INVALID_MESSAGE_FORMAT`: Malformed message
- `UNSUPPORTED_EVENT_TYPE`: Invalid event type
- `AGENT_NOT_FOUND`: Referenced agent doesn't exist
- `PERMISSION_DENIED`: Insufficient permissions
- `CONNECTION_LIMIT_EXCEEDED`: Too many connections

## Connection Management

### Connection Pooling
The WebSocket API uses connection pooling for efficiency:

```json
{
  "type": "connection_pool_info",
  "timestamp": "2024-01-15T10:46:00Z",
  "data": {
    "pool_id": "pool_123",
    "active_connections": 3,
    "max_connections": 5,
    "pool_utilization": 0.6
  }
}
```

### Automatic Reconnection
Implement reconnection logic in your client:

```javascript
class WebSocketClient {
    constructor(url, token) {
        this.url = url;
        this.token = token;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }

    connect() {
        this.ws = new WebSocket(`${this.url}?token=${this.token}`);

        this.ws.onopen = () => {
            console.log('Connected');
            this.reconnectAttempts = 0;
            this.startHeartbeat();
        };

        this.ws.onclose = (event) => {
            console.log('Disconnected:', event.code, event.reason);
            this.stopHeartbeat();

            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.scheduleReconnect();
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onmessage = (event) => {
            this.handleMessage(JSON.parse(event.data));
        };
    }

    scheduleReconnect() {
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
        this.reconnectAttempts++;

        setTimeout(() => {
            console.log(`Reconnecting (attempt ${this.reconnectAttempts})...`);
            this.connect();
        }, delay);
    }

    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000); // 30 seconds
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
        }
    }

    handleMessage(message) {
        switch (message.type) {
            case 'pong':
                console.log('Heartbeat received');
                break;
            case 'agent_status':
                this.onAgentStatus(message.data);
                break;
            case 'metrics_update':
                this.onMetricsUpdate(message.data);
                break;
            // Handle other message types
        }
    }
}
```

### Graceful Disconnection
```javascript
// Clean disconnect
ws.send(JSON.stringify({
    type: 'disconnect',
    data: {
        reason: 'user_initiated'
    }
}));

ws.close(1000, 'Normal closure');
```

## Security Considerations

### Token Refresh
Monitor token expiration and refresh as needed:

```javascript
// Monitor for authentication errors
ws.onmessage = (event) => {
    const message = JSON.parse(event.data);

    if (message.type === 'connection_error' &&
        message.data.error_code === 'AUTHENTICATION_FAILED') {
        // Refresh token and reconnect
        refreshToken().then(newToken => {
            this.token = newToken;
            this.connect();
        });
    }
};
```

### Message Validation
Always validate incoming messages:

```javascript
function validateMessage(message) {
    if (!message.type || !message.timestamp) {
        throw new Error('Invalid message format');
    }

    // Validate message type
    const validTypes = ['agent_status', 'metrics_update', 'system_alert'];
    if (!validTypes.includes(message.type)) {
        throw new Error('Unknown message type');
    }

    return true;
}
```

## Usage Examples

### Real-time Dashboard
```javascript
class AgentDashboard {
    constructor() {
        this.ws = new WebSocketClient('wss://api.freeagentics.com/api/v1/ws', token);
        this.agents = new Map();
        this.metrics = {};
    }

    async start() {
        await this.ws.connect();

        // Subscribe to all relevant events
        this.ws.send({
            type: 'subscribe',
            data: {
                events: ['agent_status', 'metrics_update', 'inference_completed'],
                agents: [] // All agents
            }
        });
    }

    onAgentStatus(data) {
        this.agents.set(data.agent_id, {
            ...this.agents.get(data.agent_id),
            status: data.new_status,
            last_update: data.timestamp
        });

        this.updateUI();
    }

    onMetricsUpdate(data) {
        this.metrics = data;
        this.updateMetricsDisplay();
    }

    updateUI() {
        // Update dashboard interface
        document.getElementById('agent-count').textContent = this.agents.size;
        document.getElementById('active-agents').textContent =
            Array.from(this.agents.values()).filter(a => a.status === 'active').length;
    }
}
```

### Agent Monitor
```javascript
class AgentMonitor {
    constructor(agentId) {
        this.agentId = agentId;
        this.ws = new WebSocketClient('wss://api.freeagentics.com/api/v1/ws', token);
        this.inferences = [];
    }

    async start() {
        await this.ws.connect();

        // Subscribe to specific agent events
        this.ws.send({
            type: 'subscribe',
            data: {
                events: ['inference_started', 'inference_progress', 'inference_completed'],
                agents: [this.agentId]
            }
        });
    }

    onInferenceStarted(data) {
        this.inferences.push({
            id: data.inference_id,
            status: 'running',
            started: data.timestamp,
            progress: 0
        });

        this.updateInferenceList();
    }

    onInferenceProgress(data) {
        const inference = this.inferences.find(i => i.id === data.inference_id);
        if (inference) {
            inference.progress = data.progress;
            this.updateProgressBar(data.inference_id, data.progress);
        }
    }

    onInferenceCompleted(data) {
        const inference = this.inferences.find(i => i.id === data.inference_id);
        if (inference) {
            inference.status = 'completed';
            inference.completed = data.timestamp;
            inference.processing_time = data.processing_time;
        }

        this.updateInferenceList();
    }
}
```

## Best Practices

### 1. Connection Management
- Implement exponential backoff for reconnection
- Monitor connection health with heartbeats
- Handle network failures gracefully
- Limit concurrent connections

### 2. Message Handling
- Always validate incoming messages
- Handle unknown message types gracefully
- Implement message deduplication
- Use structured error handling

### 3. Performance Optimization
- Subscribe only to needed events
- Implement client-side filtering
- Use connection pooling
- Batch UI updates

### 4. Security
- Validate all incoming data
- Use secure WebSocket connections (wss://)
- Implement token refresh logic
- Monitor for security events

### 5. Error Recovery
- Implement retry logic
- Handle partial failures
- Provide user feedback
- Log errors for debugging

## Troubleshooting

### Common Issues

#### Connection Refused
```javascript
// Check token validity
if (error.code === 1008) {
    console.log('Authentication failed - refresh token');
    // Implement token refresh
}
```

#### Message Delivery Issues
```javascript
// Check connection state before sending
if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(message));
} else {
    console.log('Connection not ready, queuing message');
    messageQueue.push(message);
}
```

#### High Latency
```javascript
// Monitor round-trip time
const pingTime = Date.now();
ws.send(JSON.stringify({ type: 'ping' }));

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'pong') {
        const latency = Date.now() - pingTime;
        console.log(`Latency: ${latency}ms`);
    }
};
```

### Debugging Tips

1. **Enable Debug Logging**
   - Log all incoming/outgoing messages
   - Monitor connection state changes
   - Track subscription status

2. **Monitor Network**
   - Check for network interruptions
   - Monitor bandwidth usage
   - Track message delivery rates

3. **Validate Messages**
   - Ensure proper JSON format
   - Check required fields
   - Validate data types

4. **Test Scenarios**
   - Connection loss/recovery
   - Token expiration
   - Rate limiting
   - Server restarts

## Rate Limiting and Quotas

### Production Limits
- **Connections**: 5 per user, 20 per IP
- **Messages**: 100 per minute per connection
- **Subscriptions**: 20 event types per connection
- **Message Size**: 64KB maximum

### Development Limits
- **Connections**: 10 per user, 50 per IP
- **Messages**: 200 per minute per connection
- **Subscriptions**: 50 event types per connection
- **Message Size**: 128KB maximum

Monitor your usage and implement appropriate throttling to avoid hitting limits.

## Support

- **Documentation**: [Full API documentation](API_REFERENCE.md)
- **WebSocket Status**: https://status.freeagentics.com/websocket
- **Support**: support@freeagentics.com
- **Community**: https://community.freeagentics.com/websocket
