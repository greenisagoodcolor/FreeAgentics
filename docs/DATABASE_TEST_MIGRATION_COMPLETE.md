# Database Test Migration - Complete Implementation

## Overview

This document summarizes the complete migration from mock-based database testing to real PostgreSQL database testing in the FreeAgentics project.

## Database Test Infrastructure Created

### 1. Core Test Infrastructure (`tests/db_infrastructure/`)

- **test_config.py**: Base configuration for database tests
  - `DatabaseTestCase` base class for all database tests
  - Test database URL configuration
  - Automatic transaction rollback after each test

- **fixtures.py**: Reusable pytest fixtures
  - `db_session`: Provides a database session for tests
  - `test_db`: Alternative fixture for database access
  - Automatic cleanup and rollback

- **factories.py**: Test data factories using Factory Boy
  - `AgentFactory`: Creates test agents with realistic data
  - `CoalitionFactory`: Creates test coalitions
  - `KnowledgeNodeFactory`: Creates knowledge graph nodes
  - `KnowledgeEdgeFactory`: Creates knowledge graph edges

- **data_generator.py**: Bulk test data generation
  - `AgentDataGenerator`: Generates realistic agent data
  - `CoalitionDataGenerator`: Generates coalition scenarios
  - `KnowledgeGraphGenerator`: Generates complex knowledge graphs

- **performance_monitor.py**: Database performance tracking
  - Query execution time monitoring
  - Connection pool statistics
  - Slow query detection

- **pool_config.py**: Database connection pool configuration
  - Optimized pool settings for tests
  - Connection recycling
  - Overflow handling

## Migrated Test Files

### 1. Knowledge Graph Tests

**File**: `tests/unit/test_knowledge_graph.py`

- Updated `test_database_storage` to use PostgreSQL instead of SQLite
- Added proper cleanup after tests
- Uses real database URL from test configuration

### 2. New Integration Tests Created

#### Coalition Database Tests

**File**: `tests/integration/test_coalition_database.py`

- Comprehensive coalition lifecycle testing
- Agent role management within coalitions
- Performance tracking over time
- Multi-coalition membership
- Trust and contribution scoring
- Complex coalition queries

Key features demonstrated:

- Real PostgreSQL transactions
- Association table management
- Complex JOIN queries
- Aggregation and statistics
- Concurrent coalition operations

#### Knowledge Graph Database Tests

**File**: `tests/integration/test_knowledge_graph_database.py`

- Graph persistence and loading
- Agent-knowledge graph integration
- Complex graph queries using SQL
- Performance testing with large datasets
- Concurrent graph access
- Batch operations

Key features demonstrated:

- JSON column queries in PostgreSQL
- Recursive CTE queries for hierarchy
- Graph merging operations
- Performance benchmarking
- Thread-safe graph operations

#### WebSocket Database Tests

**File**: `tests/integration/test_websocket_database.py`

- WebSocket connection persistence
- Event subscription tracking
- Event history and analytics
- Connection recovery after restart
- Cleanup of old connections

Key features demonstrated:

- Real-time event tracking
- Connection state management
- Analytics queries
- Data retention policies
- System restart recovery

## Database Models Used

### 1. Core Models (from `database/models.py`)

- `Agent`: Active Inference agents with PyMDP config
- `Coalition`: Multi-agent coalitions
- `KnowledgeNode`: Knowledge graph nodes
- `KnowledgeEdge`: Knowledge graph relationships
- Association tables for many-to-many relationships

### 2. Test-Specific Models

- `WebSocketConnection`: Tracks WebSocket connections
- `WebSocketSubscription`: Event subscriptions
- `WebSocketEvent`: Event history

## Testing Best Practices Implemented

### 1. Transaction Isolation

- Each test runs in its own transaction
- Automatic rollback prevents test interference
- No test data persists between tests

### 2. Factory Pattern

- Consistent test data generation
- Realistic data relationships
- Easy customization for specific test cases

### 3. Performance Considerations

- Connection pooling for efficiency
- Batch operations where appropriate
- Query optimization examples

### 4. Real-World Scenarios

- Multi-agent coordination
- Coalition formation and dissolution
- Knowledge graph evolution
- WebSocket event streaming

## Benefits of Real Database Testing

### 1. Accuracy

- Tests actual PostgreSQL features (JSON columns, arrays, UUIDs)
- Real transaction behavior
- Actual constraint enforcement

### 2. Performance Testing

- Real query performance metrics
- Connection pool behavior
- Concurrent access patterns

### 3. Integration Testing

- Cross-table relationships
- Complex queries with JOINs
- Cascade operations

### 4. Data Integrity

- Foreign key constraints
- Unique constraints
- Check constraints

## Migration Guidelines for Remaining Tests

When migrating additional tests from mocks to real database:

1. **Inherit from DatabaseTestCase**

   ```python
   from tests.db_infrastructure.test_config import DatabaseTestCase

   class TestYourFeature(DatabaseTestCase):
       def test_something(self, db_session):
           # Your test code
   ```

2. **Use Factories for Test Data**

   ```python
   from tests.db_infrastructure.factories import AgentFactory

   agent = AgentFactory(name="Test Agent")
   db_session.add(agent)
   db_session.commit()
   ```

3. **Always Clean Up**
   - The infrastructure handles rollback automatically
   - For explicit cleanup, use the session's delete() method

4. **Test Real Scenarios**
   - Don't just test CRUD operations
   - Test complex queries and relationships
   - Test concurrent access patterns

## Running the Tests

### Prerequisites

1. PostgreSQL running with test database
2. Database migrations applied
3. Test dependencies installed

### Commands

```bash
# Run all database tests
pytest tests/integration/test_*_database.py -v

# Run specific test file
pytest tests/integration/test_coalition_database.py -v

# Run with coverage
pytest tests/integration/ --cov=database --cov=knowledge_graph

# Run with performance profiling
pytest tests/integration/test_knowledge_graph_database.py::test_performance -v -s
```

## Performance Benchmarks

Based on the implemented tests:

- **Knowledge Graph Operations**:
  - Save 100 nodes + 200 edges: < 5 seconds
  - Load large graph: < 3 seconds
  - Batch updates: < 5 seconds

- **Coalition Operations**:
  - Create coalition with 10 agents: < 1 second
  - Query coalition statistics: < 0.5 seconds
  - Update trust scores: < 1 second

- **WebSocket Operations**:
  - Track 1000 events: < 2 seconds
  - Analytics queries: < 1 second
  - Connection recovery: < 0.5 seconds

## Conclusion

The migration from mock-based to real database testing provides:

1. **Confidence**: Tests reflect actual production behavior
2. **Performance**: Real performance metrics and optimization opportunities
3. **Completeness**: Tests cover complex scenarios not possible with mocks
4. **Maintainability**: Less mock code to maintain, tests are clearer

All new database-related tests should follow these patterns and use the provided infrastructure for consistency and reliability.
