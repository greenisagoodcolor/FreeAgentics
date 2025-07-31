# WebSocket Architecture and Flow

## Architecture Overview

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│                 │         │                 │         │                 │
│  Frontend (UI)  │◄────────┤  WebSocket API  │◄────────┤  Backend Core   │
│                 │         │                 │         │                 │
└────────┬────────┘         └────────┬────────┘         └────────┬────────┘
         │                           │                           │
         │  1. Connect               │                           │
         ├──────────────────────────►│                           │
         │                           │  2. Validate Auth          │
         │                           ├──────────────────────────►│
         │                           │                           │
         │  3. Connection ACK        │◄──────────────────────────┤
         │◄──────────────────────────┤                           │
         │                           │                           │
         │  4. Subscribe Events      │                           │
         ├──────────────────────────►│                           │
         │                           │  5. Register Handlers      │
         │                           ├──────────────────────────►│
         │                           │                           │
         │  6. Real-time Updates     │◄──────────────────────────┤
         │◄──────────────────────────┤                           │
```

## Connection Flow

### 1. Initial Connection
```
Client                          Server
  │                               │
  ├─── ws://host/path ──────────►│
  │    + auth token (optional)    │
  │                               │
  │◄── 101 Switching Protocols ───┤
  │                               │
  │◄── connection_established ────┤
  │    {type, client_id, ...}     │
```

### 2. Authentication Flow

#### Demo Mode (No Auth Required)
```
Client                          Server
  │                               │
  ├─── Connect to /ws/demo ─────►│
  │                               │
  │◄── Immediate Accept ──────────┤
  │    No token validation        │
```

#### Development/Production Mode
```
Client                          Server
  │                               │
  ├─── Connect with ?token=xxx ──►│
  │                               │
  │                           ┌───┴───┐
  │                           │Validate│
  │                           │ Token  │
  │                           └───┬───┘
  │                               │
  │◄── Accept or Reject ──────────┤
  │    Based on validation        │
```

## Message Flow Patterns

### 1. Request-Response Pattern
```
Client                          Server
  │                               │
  ├─── Request ──────────────────►│
  │    {type, id, params}         │
  │                               │
  │                           ┌───┴───┐
  │                           │Process│
  │                           └───┬───┘
  │                               │
  │◄── Response ──────────────────┤
  │    {type, id, result}         │
```

### 2. Subscription Pattern
```
Client                          Server
  │                               │
  ├─── Subscribe ────────────────►│
  │    {type: "subscribe",        │
  │     events: [...]}            │
  │                               │
  │◄── Confirmation ───────────────┤
  │                               │
  │◄── Event 1 ────────────────────┤
  │◄── Event 2 ────────────────────┤
  │◄── Event N ────────────────────┤ (continuous)
  │                               │
  ├─── Unsubscribe ───────────────►│
  │                               │
  │◄── Confirmation ───────────────┤
```

### 3. Broadcast Pattern
```
                Server
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
 Client A     Client B     Client C
    │             │             │
    │◄── Broadcast Event ───────┤
    │◄── Broadcast Event ───────┤
    │◄── Broadcast Event ───────┤
```

## Component Architecture

### Frontend Components
```
┌─────────────────────────────────────────────┐
│              WebSocket Manager              │
├─────────────────────────────────────────────┤
│ • Connection Management                      │
│ • Reconnection Logic                        │
│ • Message Queue                            │
│ • Event Handlers                           │
└──────────────┬──────────────────────────────┘
               │
    ┌──────────┴──────────┬──────────────────┐
    ▼                     ▼                  ▼
┌─────────┐         ┌─────────┐        ┌─────────┐
│ Agent   │         │ System  │        │ Belief  │
│ Monitor │         │ Metrics │        │ State   │
└─────────┘         └─────────┘        └─────────┘
```

### Backend Components
```
┌─────────────────────────────────────────────┐
│           WebSocket Handler                 │
├─────────────────────────────────────────────┤
│ • Authentication Middleware                 │
│ • Connection Pool Manager                   │
│ • Message Router                           │
│ • Rate Limiter                            │
└──────────────┬──────────────────────────────┘
               │
    ┌──────────┴──────────┬──────────────────┐
    ▼                     ▼                  ▼
┌─────────┐         ┌─────────┐        ┌─────────┐
│ Agent   │         │ Event   │        │ System  │
│ Service │         │ Bus     │        │ Monitor │
└─────────┘         └─────────┘        └─────────┘
```

## Environment Configuration

### Demo Mode
```env
NEXT_PUBLIC_WS_URL=              # Empty = auto-detect demo endpoint
WS_AUTH_REQUIRED=false          # No authentication needed
WS_ENDPOINT=/api/v1/ws/demo     # Demo endpoint path
```

### Development Mode
```env
NEXT_PUBLIC_WS_URL=ws://localhost:8000/api/v1/ws/dev
WS_AUTH_REQUIRED=true
WS_ENDPOINT=/api/v1/ws/dev
```

### Production Mode
```env
NEXT_PUBLIC_WS_URL=wss://api.example.com/api/v1/ws
WS_AUTH_REQUIRED=true
WS_ENDPOINT=/api/v1/ws
WS_MAX_CONNECTIONS=1000
```

## Message Types

### Client → Server
```typescript
interface ClientMessage {
  type: 'ping' | 'subscribe' | 'unsubscribe' | 'query' | 'command';
  id?: string;            // Request ID for tracking
  timestamp: string;      // ISO 8601 timestamp
  [key: string]: any;     // Type-specific payload
}
```

### Server → Client
```typescript
interface ServerMessage {
  type: 'pong' | 'event' | 'response' | 'error' | 'broadcast';
  id?: string;            // Matches request ID if applicable
  timestamp: string;      // ISO 8601 timestamp
  [key: string]: any;     // Type-specific payload
}
```

## Connection States

```
┌─────────┐      connect()      ┌─────────────┐
│         ├────────────────────►│             │
│ CLOSED  │                     │ CONNECTING  │
│         │◄────────────────────┤             │
└─────────┘     error/timeout   └──────┬──────┘
     ▲                                  │
     │                                  │ connected
     │                                  ▼
     │                          ┌─────────────┐
     │       close()            │             │
     └──────────────────────────┤    OPEN     │
                                │             │
                                └─────────────┘
```

## Error Handling

### Connection Errors
- **1006**: Abnormal closure (network issue)
- **1008**: Policy violation (auth failure)
- **1009**: Message too large
- **1011**: Server error

### Application Errors
```json
{
  "type": "error",
  "code": "RATE_LIMIT_EXCEEDED",
  "message": "Too many requests",
  "details": {
    "limit": 100,
    "window": "1m",
    "retry_after": 45
  }
}
```

## Security Considerations

1. **Authentication**: JWT tokens in query params or headers
2. **Rate Limiting**: Per-connection and per-IP limits
3. **Message Validation**: Schema validation for all messages
4. **Connection Limits**: Max connections per user/IP
5. **Heartbeat**: Ping/pong to detect stale connections

## Performance Optimization

1. **Connection Pooling**: Reuse connections across components
2. **Message Batching**: Group multiple updates
3. **Compression**: Enable per-message deflate
4. **Binary Protocol**: Use MessagePack for large payloads
5. **Selective Subscriptions**: Only subscribe to needed events