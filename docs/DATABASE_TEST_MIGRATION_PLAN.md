# Database Test Migration Plan

## Overview

This document outlines the plan to replace mocked database tests with real PostgreSQL operations throughout the FreeAgentics codebase.

## Current State Analysis

### Tests Already Using Real Database Operations

1. **test_database_integration.py** - Uses real PostgreSQL for API endpoint testing
2. **test_api_agents.py** - Uses SQLite for testing with real database operations
3. **test_knowledge_graph.py** - Uses SQLite for DatabaseStorageBackend tests
4. **test_database_load.py** - Real PostgreSQL load testing implementation

### Tests with Mocks (Disabled)

1. **test_database_load_mock.py.DISABLED_MOCKS** - Mock implementation of database load testing
2. **test_websocket_stress.py.DISABLED_MOCKS** - Likely contains mocked database operations
3. **test_llm_local_manager.py.DISABLED_MOCKS** - May contain mocked storage operations

## Migration Strategy

### Phase 1: Test Infrastructure Enhancement

1. Create a centralized test database configuration
2. Implement proper test fixtures and factories
3. Set up transaction rollback for test isolation
4. Create utilities for test data generation

### Phase 2: Mock Replacement

1. Re-enable disabled mock tests
2. Replace mock objects with real database operations
3. Update assertions to work with real data
4. Ensure proper cleanup after each test

### Phase 3: Performance Test Migration

1. Convert mock load tests to use real PostgreSQL
2. Add proper connection pooling for concurrent tests
3. Implement realistic timing and performance metrics
4. Add database performance monitoring

## Implementation Details

### 1. Test Database Configuration

```python
# tests/db_infrastructure/test_config.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.base import Base

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://freeagentics:freeagentics123@localhost:5432/freeagentics_test"
)

def create_test_engine():
    """Create test database engine with proper configuration."""
    return create_engine(
        TEST_DATABASE_URL,
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True,
        echo=False
    )

def setup_test_database():
    """Setup test database schema."""
    engine = create_test_engine()
    Base.metadata.create_all(bind=engine)
    return engine

def teardown_test_database():
    """Clean up test database."""
    engine = create_test_engine()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
```

### 2. Test Fixtures and Factories

```python
# tests/db_infrastructure/factories.py
from datetime import datetime
from typing import Dict, Any
from database.models import Agent, Coalition, KnowledgeNode, KnowledgeEdge

class AgentFactory:
    """Factory for creating test agents."""

    @staticmethod
    def create(session, **kwargs) -> Agent:
        """Create a test agent with sensible defaults."""
        defaults = {
            "agent_id": f"test_agent_{datetime.now().timestamp()}",
            "name": "Test Agent",
            "agent_type": "resource_collector",
            "status": "active",
            "belief_state": {"test": True},
            "position": {"lat": 37.7749, "lon": -122.4194},
            "capabilities": ["resource_collection"],
            "created_at": datetime.utcnow()
        }
        defaults.update(kwargs)

        agent = Agent(**defaults)
        session.add(agent)
        session.commit()
        return agent

class KnowledgeGraphFactory:
    """Factory for creating test knowledge graphs."""

    @staticmethod
    def create_with_nodes(session, num_nodes: int = 10) -> Dict[str, Any]:
        """Create a knowledge graph with connected nodes."""
        nodes = []
        for i in range(num_nodes):
            node = KnowledgeNode(
                node_id=f"test_node_{i}",
                node_type="concept",
                content=f"Test concept {i}",
                metadata={"index": i},
                created_at=datetime.utcnow()
            )
            nodes.append(node)

        session.add_all(nodes)
        session.commit()

        # Create edges
        edges = []
        for i in range(num_nodes - 1):
            edge = KnowledgeEdge(
                edge_id=f"test_edge_{i}",
                source_node_id=nodes[i].node_id,
                target_node_id=nodes[i + 1].node_id,
                edge_type="relates_to",
                weight=0.5,
                created_at=datetime.utcnow()
            )
            edges.append(edge)

        session.add_all(edges)
        session.commit()

        return {"nodes": nodes, "edges": edges}
```

### 3. Transaction Rollback for Test Isolation

```python
# tests/db_infrastructure/fixtures.py
import pytest
from sqlalchemy.orm import Session
from contextlib import contextmanager

@pytest.fixture
def db_session():
    """Provide a transactional database session for tests."""
    engine = create_test_engine()
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()

@contextmanager
def isolated_db_test():
    """Context manager for isolated database tests."""
    engine = create_test_engine()
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
```

### 4. Mock Test Conversion Example

Convert the mocked database load test:

```python
# tests/performance/test_database_load_real.py
import pytest
from tests.db_infrastructure.factories import AgentFactory, KnowledgeGraphFactory
from tests.db_infrastructure.fixtures import db_session

class TestDatabaseLoadReal:
    """Real database load tests using PostgreSQL."""

    @pytest.mark.parametrize("num_agents", [10, 100, 500])
    def test_agent_creation_performance(self, db_session, num_agents):
        """Test agent creation performance with real database."""
        import time

        start_time = time.time()

        agents = []
        for i in range(num_agents):
            agent = AgentFactory.create(
                db_session,
                agent_id=f"perf_test_agent_{i}",
                name=f"Performance Test Agent {i}"
            )
            agents.append(agent)

        duration = time.time() - start_time

        # Performance assertions
        assert len(agents) == num_agents
        assert duration < num_agents * 0.1  # Max 100ms per agent

        # Verify data integrity
        agent_count = db_session.query(Agent).count()
        assert agent_count == num_agents

    def test_concurrent_operations(self, db_session):
        """Test concurrent database operations."""
        from concurrent.futures import ThreadPoolExecutor
        import threading

        # Create test data
        agents = [AgentFactory.create(db_session, agent_id=f"concurrent_{i}")
                  for i in range(20)]

        results = {"reads": 0, "updates": 0, "errors": 0}
        lock = threading.Lock()

        def read_agents():
            try:
                agents = db_session.query(Agent).all()
                with lock:
                    results["reads"] += len(agents)
            except Exception:
                with lock:
                    results["errors"] += 1

        def update_agent(agent_id):
            try:
                agent = db_session.query(Agent).filter_by(agent_id=agent_id).first()
                if agent:
                    agent.belief_state = {"updated": True}
                    db_session.commit()
                    with lock:
                        results["updates"] += 1
            except Exception:
                db_session.rollback()
                with lock:
                    results["errors"] += 1

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit read tasks
            read_futures = [executor.submit(read_agents) for _ in range(50)]

            # Submit update tasks
            update_futures = [
                executor.submit(update_agent, agent.agent_id)
                for agent in agents
            ]

            # Wait for completion
            for future in read_futures + update_futures:
                future.result()

        # Assertions
        assert results["errors"] == 0
        assert results["reads"] > 0
        assert results["updates"] == len(agents)
```

## Testing Strategy

### Unit Tests

- Use SQLite in-memory database for fast unit tests
- Transaction rollback for test isolation
- Minimal fixtures for focused testing

### Integration Tests

- Use PostgreSQL test database
- Test real database constraints and features
- Include connection pooling and concurrency

### Performance Tests

- Use PostgreSQL with realistic data volumes
- Measure actual query performance
- Test concurrent access patterns
- Monitor resource usage

## Migration Checklist

- [ ] Set up test database infrastructure
- [ ] Create data factories for all models
- [ ] Implement transaction-based test isolation
- [ ] Convert mock load tests to real database
- [ ] Update websocket stress tests
- [ ] Convert LLM storage tests
- [ ] Add database performance benchmarks
- [ ] Document test database setup
- [ ] Update CI/CD configuration
- [ ] Add test coverage reporting

## Benefits

1. **Realistic Testing**: Tests reflect actual database behavior
2. **Performance Validation**: Real performance metrics
3. **Constraint Testing**: Database constraints are enforced
4. **Concurrency Testing**: Real transaction isolation
5. **Migration Safety**: Tests validate schema changes

## Next Steps

1. Create the test infrastructure modules
2. Start with converting disabled mock tests
3. Add performance benchmarks
4. Update documentation
5. Ensure CI/CD compatibility
