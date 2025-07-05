# Concurrent User Simulation Framework

A comprehensive framework for simulating realistic concurrent user behaviors in FreeAgentics, integrating database operations with WebSocket interactions to stress test the system under various load patterns.

## Features

### 1. **User Personas**

- **Researcher**: Focuses on queries, analysis, and experiments
- **Coordinator**: Manages coalitions and agent coordination
- **Observer**: Passive monitoring and event subscriptions
- **Admin**: System management and monitoring
- **Developer**: Testing and debugging patterns
- **Analyst**: Data analysis and reporting

Each persona has distinct:

- Activity levels (hyperactive, active, moderate, passive, sporadic)
- Interaction patterns (continuous, bursty, scheduled, random, reactive)
- Message preferences and weights
- Error rates and response times
- Connection behaviors

### 2. **Simulation Scenarios**

#### Predefined Scenarios

- **research_conference**: Virtual conference with many researchers and observers
- **coalition_operations**: Intensive coalition formation and management
- **system_monitoring**: Heavy monitoring and admin operations
- **mixed_workload**: Realistic mixed workload with all personas
- **stress_test**: High-stress scenario to find system limits
- **burst_activity**: Burst patterns with idle periods
- **development_testing**: Developer testing patterns
- **long_running**: Long-running stability test
- **database_intensive**: Database-heavy operations
- **failover_test**: Connection recovery and failover

#### Custom Scenarios

Create custom scenarios with specific user distributions and parameters.

### 3. **Database Integration**

- Agent creation and management
- Coalition formation and tracking
- Performance metrics collection
- Transaction handling
- Connection pool management

### 4. **WebSocket Operations**

- Event subscriptions
- Command execution
- Query processing
- Monitoring configuration
- Connection lifecycle management

### 5. **Metrics Collection**

- User activity metrics
- Message throughput
- Database operation latency
- WebSocket connection statistics
- System resource usage
- Per-persona performance data

## Installation

Ensure you have the required dependencies:

```bash
pip install -r requirements-dev.txt
```

## Quick Start

### Command Line Interface

1. **List available scenarios:**

```bash
python tests/simulation/run_simulation.py list
```

2. **Run a single scenario:**

```bash
# Run mixed workload for 1 hour
python tests/simulation/run_simulation.py run mixed_workload

# Run with custom duration
python tests/simulation/run_simulation.py run stress_test --duration 1800

# Run with custom parameters
python tests/simulation/run_simulation.py run mixed_workload \
    --duration 3600 \
    --spawn-rate 5.0 \
    --warmup 300 \
    --cooldown 300 \
    --output results/mixed_test
```

3. **Run a custom scenario:**

```bash
python tests/simulation/run_simulation.py run custom \
    --custom-name "my_test" \
    --custom-description "Custom test scenario" \
    --custom-users "researcher:20,coordinator:10,observer:30" \
    --duration 1800
```

4. **Run scheduled scenarios:**

```bash
# Run daily scenario sequence
python tests/simulation/run_simulation.py schedule daily

# Run stress test sequence
python tests/simulation/run_simulation.py schedule stress
```

### Programmatic Usage

```python
import asyncio
from tests.simulation.concurrent_simulator import ConcurrentSimulator
from tests.simulation.scenarios import SimulationScenarios

async def run_simulation():
    # Get a predefined scenario
    config = SimulationScenarios.mixed_workload()

    # Customize parameters
    config.duration_seconds = 1800  # 30 minutes
    config.user_spawn_rate = 3.0

    # Create and run simulator
    simulator = ConcurrentSimulator(config)
    await simulator.run()

    # Get results
    summary = simulator.get_summary()
    print(f"Total messages: {summary['metrics']['messages']['sent']}")

asyncio.run(run_simulation())
```

## User Persona Details

### Researcher

- **Focus**: Queries and analysis
- **Activities**:
  - Complex agent history queries
  - Belief evolution analysis
  - Performance metrics collection
  - Experiment control
- **Message weights**: 40% queries, 30% commands, 20% events

### Coordinator

- **Focus**: Coalition and agent management
- **Activities**:
  - Coalition formation
  - Agent role assignment
  - Task delegation
  - Conflict resolution
- **Message weights**: 50% commands, 25% events, 15% queries

### Observer

- **Focus**: Passive monitoring
- **Activities**:
  - Event stream subscriptions
  - System overview queries
  - Activity monitoring
- **Message weights**: 60% events, 25% queries, 10% monitoring

### Admin

- **Focus**: System management
- **Activities**:
  - System health monitoring
  - Resource scaling
  - Configuration updates
  - Maintenance tasks
- **Message weights**: 40% monitoring, 30% queries, 20% commands

## Configuration Options

### SimulationConfig Parameters

```python
SimulationConfig(
    name="scenario_name",
    description="Scenario description",
    duration_seconds=3600,

    # User distribution
    user_distribution={
        PersonaType.RESEARCHER: 50,
        PersonaType.COORDINATOR: 20,
        PersonaType.OBSERVER: 30,
    },

    # Timing
    user_spawn_rate=2.0,        # Users per second
    warmup_period=300,          # Warmup seconds
    cooldown_period=300,        # Cooldown seconds

    # Database
    db_url="postgresql://localhost/freeagentics_test",
    db_pool_size=20,
    db_max_overflow=10,

    # WebSocket
    ws_base_url="ws://localhost:8000",
    ws_path="/ws",
    ws_reconnect_attempts=3,
    ws_reconnect_delay=1.0,

    # Simulation
    enable_errors=True,
    error_injection_rate=0.01,
    network_latency_range=(0.01, 0.1),

    # Monitoring
    enable_monitoring=True,
    metrics_interval=5.0,
    export_results=True,
    results_path=Path("simulation_results"),
)
```

## Metrics and Results

### Real-time Metrics

- Active users count
- Messages per second
- Current latency
- Error rates
- System resource usage

### Summary Metrics

```json
{
  "duration_seconds": 3600,
  "users": {
    "created": 200,
    "connected": 195,
    "active": 180,
    "errored": 5
  },
  "messages": {
    "sent": 150000,
    "received": 149500,
    "failed": 500,
    "success_rate": 0.997
  },
  "database": {
    "operations": 50000,
    "errors": 10,
    "avg_latency_ms": 12.5,
    "p95_latency_ms": 45.2
  },
  "websocket": {
    "connections": 195,
    "disconnections": 15,
    "errors": 5,
    "avg_latency_ms": 23.4,
    "p95_latency_ms": 67.8
  }
}
```

## Advanced Usage

### Custom User Behavior

```python
from tests.simulation.user_personas import UserBehavior, PersonaProfile

class CustomBehavior(UserBehavior):
    async def decide_next_action(self):
        # Implement custom logic
        if self.should_act():
            return {
                "type": "query",
                "query_type": "custom_query",
                "params": {...}
            }
        return None
```

### Scheduled Scenarios

Create a JSON schedule file:

```json
{
  "name": "custom_schedule",
  "scenarios": [
    {
      "name": "mixed_workload",
      "delay_minutes": 0
    },
    {
      "name": "stress_test",
      "delay_minutes": 65
    },
    {
      "name": "custom",
      "delay_minutes": 5,
      "config": {
        "name": "recovery_test",
        "duration_seconds": 1800,
        "user_counts": {
          "researcher": 20,
          "coordinator": 10
        }
      }
    }
  ]
}
```

Run the schedule:

```bash
python tests/simulation/run_simulation.py schedule custom_schedule.json
```

### Performance Monitoring

The framework integrates with the performance monitoring system:

```python
# Results include:
- CPU usage over time
- Memory usage over time
- Database query latencies
- WebSocket message latencies
- Per-persona performance metrics
```

## Best Practices

1. **Start Small**: Begin with shorter durations and fewer users
2. **Monitor Resources**: Watch system resources during simulations
3. **Gradual Ramp-up**: Use warmup periods to avoid thundering herd
4. **Analyze Results**: Review metrics to identify bottlenecks
5. **Iterate**: Adjust scenarios based on findings

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Verify WebSocket server is running
   - Check database connectivity
   - Ensure sufficient system resources

2. **High Error Rates**
   - Reduce spawn rate
   - Increase connection pool size
   - Check for rate limiting

3. **Memory Issues**
   - Limit concurrent users
   - Reduce simulation duration
   - Monitor for memory leaks

### Debug Mode

Enable verbose logging:

```bash
python tests/simulation/run_simulation.py run mixed_workload --verbose
```

## Examples

See `example_usage.py` for comprehensive examples including:

- Basic simulations
- Custom scenarios
- Stress testing
- Scheduled runs
- Results analysis
- Custom behaviors

## Integration with CI/CD

Run simulations in CI pipelines:

```yaml
# GitHub Actions example
- name: Run Load Test
  run: |
    python tests/simulation/run_simulation.py run mixed_workload \
      --duration 600 \
      --no-export
```

## Contributing

When adding new features:

1. Add appropriate user personas if needed
2. Create realistic message patterns
3. Include error handling
4. Add metrics collection
5. Document new scenarios

---

This simulation framework provides a powerful way to test FreeAgentics under realistic concurrent load, helping identify performance bottlenecks, validate system stability, and ensure scalability.
