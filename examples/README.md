# FreeAgentics API Usage Examples

This directory contains comprehensive examples for using the FreeAgentics API across different use cases and programming languages.

## Available Examples

### 1. Python API Examples (`api_usage_examples.py`)

Comprehensive Python examples using `aiohttp` demonstrating:

- **Basic Agent Lifecycle**: Create, start, interact, and manage agents
- **GMN Agent Creation**: Create agents from Generative Model Network specifications
- **Multi-Agent Coalitions**: Form and coordinate agent coalitions
- **Real-time Monitoring**: WebSocket-based system monitoring
- **Batch Operations**: Efficient bulk agent management
- **Performance Testing**: High-throughput agent interaction testing

**Prerequisites:**

```bash
pip install aiohttp websockets
```

**Usage:**

```bash
python examples/api_usage_examples.py
```

### 2. curl Examples (`curl_examples.sh`)

Shell script with curl commands covering all API endpoints:

- Agent CRUD operations
- GMN agent creation
- Coalition management
- System metrics access
- Real-time monitoring setup
- Performance monitoring

**Usage:**

```bash
chmod +x examples/curl_examples.sh
./examples/curl_examples.sh
```

## Example Scenarios

### Scenario 1: Research Simulation

Create multiple Active Inference agents exploring a grid world:

```python
# Create explorer agents with different parameters
agents = []
for i in range(5):
    agent_data = {
        "name": f"Explorer-{i}",
        "agent_type": "explorer",
        "config": {
            "grid_size": 20,
            "exploration_rate": 0.2 + i * 0.1,
            "use_pymdp": True
        }
    }
    agent = await client.create_agent(agent_data)
    agents.append(agent)
```

### Scenario 2: Production Monitoring

Set up comprehensive monitoring for a production deployment:

```python
# Monitor system metrics via WebSocket
monitor_config = {
    "type": "start_monitoring",
    "config": {
        "metrics": ["cpu_usage", "memory_usage", "free_energy", "belief_entropy"],
        "sample_rate": 1.0,
        "agents": ["agent-1", "agent-2", "agent-3"]
    }
}
```

### Scenario 3: Coalition Coordination

Create and manage agent coalitions for complex tasks:

```python
# Create resource collection coalition
coalition_data = {
    "name": "Resource Collection Team",
    "objectives": {
        "primary": "maximize_resource_collection",
        "secondary": "minimize_energy_consumption"
    },
    "strategy": "distributed_search"
}

coalition = await client.create_coalition(coalition_data)
```

## API Endpoints Reference

### Agent Management

- `POST /api/v1/agents` - Create agent
- `GET /api/v1/agents` - List agents
- `GET /api/v1/agents/{id}` - Get agent details
- `PUT /api/v1/agents/{id}` - Update agent
- `DELETE /api/v1/agents/{id}` - Delete agent
- `POST /api/v1/agents/{id}/start` - Start agent
- `POST /api/v1/agents/{id}/stop` - Stop agent
- `POST /api/v1/agents/{id}/step` - Send observation

### GMN Integration

- `POST /api/v1/agents/from-gmn` - Create agent from GMN spec
- `GET /api/v1/gmn/examples` - Get GMN examples
- `POST /api/v1/agents/{id}/gmn` - Update agent GMN spec

### Coalition Management

- `POST /api/v1/coalitions` - Create coalition
- `GET /api/v1/coalitions` - List coalitions
- `POST /api/v1/coalitions/{id}/agents/{agent_id}` - Add agent to coalition

### Monitoring & Metrics

- `GET /api/v1/metrics/{type}` - Get specific metrics
- `GET /api/v1/metrics/types` - Available metric types
- `GET /api/v1/metrics/counters` - Performance counters
- `WS /api/v1/ws/monitor/{client_id}` - Real-time monitoring

### System Health

- `GET /health` - System health check

## Error Handling Examples

### Python Error Handling

```python
try:
    agent = await client.create_agent(agent_data)
except aiohttp.ClientError as e:
    print(f"Network error: {e}")
except Exception as e:
    print(f"API error: {e}")
```

### curl Error Handling

```bash
# Check HTTP status code
STATUS=$(curl -s -o /dev/null -w '%{http_code}' "$API_BASE/agents")
if [ "$STATUS" -ne 200 ]; then
    echo "API returned status: $STATUS"
fi
```

## Performance Considerations

### Batch Operations

```python
# Create multiple agents concurrently
tasks = [
    client.create_agent(config)
    for config in agent_configs
]
agents = await asyncio.gather(*tasks)
```

### Rate Limiting

```python
# Add delays for high-frequency requests
for observation in observations:
    result = await client.agent_step(agent_id, observation)
    await asyncio.sleep(0.01)  # 100 req/sec max
```

## Monitoring Integration

### Grafana Dashboard

Connect to the monitoring WebSocket to feed real-time data to Grafana:

```javascript
const ws = new WebSocket(
  "ws://localhost:8000/api/v1/ws/monitor/grafana-client",
);
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Send to Grafana or InfluxDB
};
```

### Custom Metrics Collection

```python
# Record custom metrics
await client.session.post(
    f"{base_url}/api/v1/metrics/custom",
    json={
        "metric_type": "custom_inference_time",
        "value": inference_duration,
        "agent_id": agent_id
    }
)
```

## Troubleshooting

### Common Issues

1. **Connection Refused**

   ```bash
   # Check if server is running
   curl -s http://localhost:8000/health
   ```

2. **Agent Creation Fails**

   ```python
   # Validate agent configuration
   if not agent_data.get("name"):
       raise ValueError("Agent name is required")
   ```

3. **WebSocket Connection Issues**

   ```python
   # Handle connection failures gracefully
   try:
       async with websockets.connect(uri) as websocket:
           # ... monitoring code
   except websockets.exceptions.ConnectionClosed:
       print("WebSocket connection lost, attempting reconnect...")
   ```

### Debug Mode

Enable verbose logging for debugging:

```python
import logging
logging.getLogger('aiohttp').setLevel(logging.DEBUG)
```

For curl debugging:

```bash
curl -v http://localhost:8000/api/v1/agents
```

## Next Steps

1. **Extend Examples**: Modify examples for your specific use case
2. **Production Integration**: Integrate monitoring into your deployment pipeline
3. **Custom Agents**: Create specialized agent types using the GMN framework
4. **Scale Testing**: Use batch operations to test system limits

For more detailed information, see:

- [API Documentation](../docs/api.md)
- [Observability Guide](../OBSERVABILITY_GUIDE.md)
- [Deployment Guide](../docs/deployment.md)
