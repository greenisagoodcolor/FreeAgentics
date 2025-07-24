# WebSocket Load Testing Framework

A comprehensive framework for load testing WebSocket connections in FreeAgentics, providing tools for simulating various connection patterns, message types, and load scenarios.

## Features

### 1. **WebSocket Client Manager**

- Manages multiple concurrent WebSocket connections
- Connection pooling and reuse
- Automatic reconnection with backoff
- Per-client metrics tracking
- Batch operations with concurrency control

### 2. **Message Generators**

- **EventMessageGenerator**: Subscribe/unsubscribe events
- **CommandMessageGenerator**: Agent commands with parameters
- **QueryMessageGenerator**: System state queries
- **MonitoringMessageGenerator**: Monitoring configuration
- **MixedMessageGenerator**: Weighted mix of message types
- **RealisticScenarioGenerator**: Realistic usage patterns

### 3. **Metrics Collection**

- Connection success/failure rates
- Message throughput (sent/received)
- Round-trip latency tracking (min/max/avg/percentiles)
- Error rates and types
- Real-time statistics
- Time-series data collection
- Prometheus integration (optional)
- Export to JSON/CSV formats

### 4. **Connection Lifecycle Management**

- **Persistent**: Maintain connection with auto-reconnect
- **Intermittent**: Connect/disconnect cycles
- **Bursty**: High activity bursts with idle periods
- **Failover**: Automatic failover between servers
- Connection state tracking
- Custom lifecycle patterns

### 5. **Load Testing Scenarios**

- **Steady Load**: Constant connection and message rate
- **Burst Load**: Periodic high-activity bursts
- **Ramp Up**: Gradually increase load
- **Stress Test**: Find system breaking point
- **Realistic Usage**: Simulate real user patterns

## Installation

The framework is included in the FreeAgentics test suite. Ensure you have the required dependencies:

```bash
pip install websockets faker numpy prometheus-client
```

## Quick Start

### Basic Usage

```python
import asyncio
from tests.websocket_load import WebSocketClientManager, MetricsCollector

async def simple_test():
    # Create client manager
    manager = WebSocketClientManager(base_url="ws://localhost:8000")

    # Create metrics collector
    metrics = MetricsCollector()

    # Create and connect clients
    clients = await manager.create_clients(10)
    await manager.connect_clients(clients)

    # Send messages
    for i in range(100):
        await manager.broadcast_message({
            "type": "ping",
            "data": {"iteration": i}
        })
        await asyncio.sleep(0.1)

    # Disconnect and get metrics
    await manager.disconnect_all()
    print(metrics.generate_summary_report())

asyncio.run(simple_test())
```

### Running Load Scenarios

```bash
# Steady load test
python -m tests.websocket_load.run_load_test steady \
    --clients 100 \
    --duration 300 \
    --message-interval 1.0

# Burst load test
python -m tests.websocket_load.run_load_test burst \
    --clients 200 \
    --burst-size 100 \
    --burst-duration 30 \
    --idle-duration 60

# Ramp up test
python -m tests.websocket_load.run_load_test ramp \
    --clients 500 \
    --initial-clients 10 \
    --ramp-steps 10 \
    --step-duration 30

# Stress test
python -m tests.websocket_load.run_load_test stress \
    --max-clients 1000 \
    --target-latency 100 \
    --error-threshold 0.05

# Realistic usage test
python -m tests.websocket_load.run_load_test realistic \
    --clients 200 \
    --user-profiles active:0.2 regular:0.5 passive:0.3
```

## Advanced Usage

### Custom Message Generator

```python
from tests.websocket_load import MessageGenerator

class CustomMessageGenerator(MessageGenerator):
    def generate(self):
        message = {
            "type": "custom_event",
            "data": {
                "value": random.randint(1, 100),
                "timestamp": time.time()
            }
        }
        return self._add_metadata(message)
```

### Custom Load Scenario

```python
from tests.websocket_load import LoadScenario, ScenarioConfig

class CustomScenario(LoadScenario):
    async def run(self):
        # Create clients in waves
        for wave in range(3):
            clients = await self.client_manager.create_clients(50)
            await self.client_manager.connect_clients(clients)

            # Custom activity pattern
            for _ in range(100):
                # Send different message types based on wave
                if wave == 0:
                    message_type = "subscribe"
                elif wave == 1:
                    message_type = "command"
                else:
                    message_type = "query"

                await self.client_manager.broadcast_message({
                    "type": message_type,
                    "wave": wave
                })

                await asyncio.sleep(0.5)

            # Disconnect this wave
            for client in clients:
                await client.disconnect()
```

### Connection Pool Usage

```python
from tests.websocket_load import ConnectionPool

# Create connection pool
pool = ConnectionPool(
    client_manager,
    min_size=10,
    max_size=100,
    acquire_timeout=5.0
)

await pool.initialize()

# Use connections
async def worker(task_id):
    client = await pool.acquire()
    if client:
        try:
            # Use the connection
            await client.send_message({"task": task_id})
        finally:
            await pool.release(client)

# Run multiple workers
tasks = [worker(i) for i in range(50)]
await asyncio.gather(*tasks)

await pool.shutdown()
```

### Metrics Analysis

```python
# Real-time monitoring
async def monitor_test():
    metrics = MetricsCollector()
    await metrics.start_real_time_stats()

    # Run your test...

    # Get real-time stats during test
    while test_running:
        stats = metrics.get_real_time_stats()
        print(f"Messages/sec: {stats['messages_per_second']:.1f}")
        print(f"Current latency: {stats['current_latency_ms']:.1f}ms")
        print(f"Error rate: {stats['error_rate']:.1%}")
        await asyncio.sleep(5)

    # Get time-series data
    latency_data = metrics.get_time_series_data("latency_ms", duration_seconds=60)

    # Export results
    metrics.save_metrics(Path("test_results.json"))
```

## Metrics Output

The framework generates comprehensive metrics including:

```json
{
  "connection_metrics": {
    "total_connections": 100,
    "successful_connections": 98,
    "failed_connections": 2,
    "success_rate": 0.98
  },
  "message_metrics": {
    "messages_sent": 10000,
    "messages_received": 9950,
    "bytes_sent": 1048576,
    "bytes_received": 2097152
  },
  "latency_metrics": {
    "min_ms": 5.2,
    "max_ms": 125.6,
    "avg_ms": 23.4,
    "p50_ms": 20.1,
    "p95_ms": 45.3,
    "p99_ms": 98.7
  },
  "throughput_metrics": {
    "messages_per_second_sent": 33.3,
    "messages_per_second_received": 33.2,
    "bytes_per_second_sent": 3495.2,
    "bytes_per_second_received": 6990.5
  }
}
```

## Best Practices

1. **Start Small**: Begin with a small number of clients and gradually increase
2. **Monitor Server**: Watch server metrics during tests
3. **Use Appropriate Patterns**: Choose connection patterns that match your use case
4. **Set Realistic Intervals**: Message intervals should reflect actual usage
5. **Analyze Results**: Look for patterns in latency spikes and error rates
6. **Test Different Scenarios**: Run multiple scenario types to understand system behavior
7. **Consider Network**: Test from different network locations if applicable

## Troubleshooting

### Common Issues

1. **Connection Failures**

   - Check server is running and accessible
   - Verify WebSocket endpoint URL
   - Check for rate limiting or connection limits

2. **High Latency**

   - Reduce message rate
   - Check server performance
   - Monitor network conditions

3. **Memory Issues**
   - Limit metrics buffer size
   - Use connection pooling
   - Reduce concurrent connections

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or via command line:

```bash
python -m tests.websocket_load.run_load_test steady --debug
```

## Examples

See `example_usage.py` for comprehensive examples including:

- Basic client operations
- Message generator usage
- Connection lifecycle patterns
- Connection pooling
- Metrics collection
- Load scenarios
- Custom implementations

## Contributing

When adding new features:

1. Follow existing patterns for consistency
2. Add appropriate metrics collection
3. Include error handling
4. Document new functionality
5. Add examples if applicable
