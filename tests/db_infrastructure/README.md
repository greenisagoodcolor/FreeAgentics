# PostgreSQL Test Infrastructure

This directory contains a comprehensive PostgreSQL test infrastructure for load testing the FreeAgentics database.

## Components

### 1. **schema.sql**

- Complete database schema matching production
- Includes all tables, indexes, constraints, and triggers
- Test-specific tables for performance monitoring

### 2. **pool_config.py**

- Thread-safe connection pooling with configurable parameters
- Performance monitoring capabilities
- Connection pool statistics and management

### 3. **data_generator.py**

- Realistic test data generation for all database entities
- Configurable data volumes and relationships
- Reproducible data generation with seed support

### 4. **db_reset.py**

- Database reset utilities for clean test runs
- Schema verification and table truncation
- Snapshot and restore capabilities

### 5. **performance_monitor.py**

- Real-time performance monitoring during tests
- System resource tracking (CPU, memory, I/O)
- Detailed performance reports and statistics

### 6. **load_test.py**

- Comprehensive load testing scenarios
- Configurable concurrency and operation mix
- Stress testing to find breaking points

## Usage

### Quick Test

```bash
# Run a quick 10-second test with 5 threads
python -m tests.db_infrastructure.load_test --test quick
```

### Full Load Test

```bash
# Run a 60-second load test with 20 threads
python -m tests.db_infrastructure.load_test --test load --duration 60 --threads 20 --ops-per-second 200
```

### Stress Test

```bash
# Ramp up to 100 concurrent connections
python -m tests.db_infrastructure.load_test --test stress --threads 100
```

### Database Management

```bash
# Reset database
python -m tests.db_infrastructure.db_reset reset --database freeagentics_test

# Create snapshot
python -m tests.db_infrastructure.db_reset snapshot --database freeagentics_test --snapshot baseline

# Restore from snapshot
python -m tests.db_infrastructure.db_reset restore --database freeagentics_test --snapshot baseline

# Verify schema
python -m tests.db_infrastructure.db_reset verify --database freeagentics_test

# Get table row counts
python -m tests.db_infrastructure.db_reset counts --database freeagentics_test
```

## Configuration

### Environment Variables

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=freeagentics_test
export DB_USER=freeagentics
export DB_PASSWORD=freeagentics123
```

### Connection Pool Settings

- Min connections: 5 (default)
- Max connections: 50 (default, configurable up to 200 for stress tests)
- Thread-safe implementation for concurrent access

## Test Scenarios

### 1. Agent Operations (40% of load)

- Query active agents
- Update agent metrics
- Join with coalitions
- Track agent activity

### 2. Coalition Operations (20% of load)

- Find forming coalitions
- Update performance scores
- Manage memberships
- Calculate coalition metrics

### 3. Knowledge Graph Queries (30% of load)

- Recursive graph traversal
- Node insertion and updates
- Edge relationship queries
- Confidence score calculations

### 4. Complex Analytics (10% of load)

- Agent performance ranking
- Coalition effectiveness analysis
- Cross-table aggregations
- JSON field operations

## Performance Metrics

The infrastructure tracks:

- **Operation Metrics**: Duration, success rate, throughput
- **Database Metrics**: Query performance, cache hit ratio, connection pool stats
- **System Metrics**: CPU usage, memory consumption, disk/network I/O
- **Application Metrics**: Error rates, operation distribution, response times

## Reports

Performance reports include:

- Test configuration and duration
- Operation timing statistics
- System resource usage
- Database performance indicators
- Connection pool utilization
- Identified bottlenecks and recommendations

## Best Practices

1. **Always reset database** before load tests for consistent results
2. **Monitor system resources** to identify bottlenecks
3. **Start with lower concurrency** and gradually increase
4. **Use snapshots** for quick resets between test runs
5. **Analyze slow queries** from performance reports
6. **Check connection pool stats** to optimize pool size

## Troubleshooting

### Connection Pool Exhaustion

- Increase max_connections in pool configuration
- Check for connection leaks in application code
- Monitor pool statistics during tests

### High Error Rates

- Check PostgreSQL logs for deadlocks
- Verify indexes are properly created
- Ensure sufficient system resources

### Slow Performance

- Run VACUUM ANALYZE after large data imports
- Check cache hit ratios (should be >90%)
- Verify connection pooling is working
- Look for missing indexes in slow query logs

## Integration with CI/CD

```yaml
# Example GitHub Actions workflow
test-database-performance:
  runs-on: ubuntu-latest
  services:
    postgres:
      image: postgres:15
      env:
        POSTGRES_PASSWORD: freeagentics123
        POSTGRES_USER: freeagentics
        POSTGRES_DB: freeagentics_test
      options: >-
        --health-cmd pg_isready
        --health-interval 10s
        --health-timeout 5s
        --health-retries 5
  steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: "3.11"
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run load test
      run: |
        python -m tests.db_infrastructure.load_test --test quick
```
