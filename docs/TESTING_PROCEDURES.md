# FreeAgentics Testing Procedures and Quality Assurance

## Overview

This document outlines comprehensive testing procedures and quality assurance practices for the FreeAgentics multi-agent AI platform. Our testing strategy follows a multi-layered approach with emphasis on security, performance, and reliability.

## Testing Philosophy

### Core Principles

1. **Test-Driven Development (TDD)**: All production code must be driven by failing tests
2. **Behavior-Driven Testing**: Focus on testing user-visible behavior, not implementation details
3. **Comprehensive Coverage**: Multiple testing layers from unit to end-to-end
4. **Security-First**: Security testing integrated into every layer
5. **Performance Validation**: Continuous performance testing and benchmarking
6. **Automated Quality Gates**: Automated checks prevent quality regression

### Testing Pyramid

```
        /\        E2E Tests (10%)
       /  \       
      /    \      Integration Tests (20%)
     /      \     
    /        \    Unit Tests (70%)
   /          \   
  /__________\
```

## Test Categories

### Unit Tests (70% of test suite)

#### Purpose
- Test individual functions and classes in isolation
- Verify business logic and edge cases
- Enable safe refactoring with fast feedback
- Achieve high code coverage with meaningful tests

#### Structure
```
tests/unit/
├── test_agents/
│   ├── test_agent_manager.py
│   ├── test_base_agent.py
│   └── test_coalition_coordinator.py
├── test_auth/
│   ├── test_jwt_handler.py
│   ├── test_security_implementation.py
│   └── test_rbac_core.py
├── test_api/
│   ├── test_auth_endpoints.py
│   ├── test_agent_endpoints.py
│   └── test_inference_endpoints.py
└── test_utils/
    ├── test_data_structures.py
    └── test_validation.py
```

#### Best Practices
```python
# Good: Tests behavior, not implementation
def test_agent_processes_observation_correctly():
    agent = Agent(template="research_v2")
    observation = {"type": "text", "content": "What is AI?"}
    
    result = agent.process_observation(observation)
    
    assert result["status"] == "processed"
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0

# Bad: Tests implementation details
def test_agent_calls_internal_method():
    agent = Agent(template="research_v2")
    observation = {"type": "text", "content": "What is AI?"}
    
    with patch.object(agent, '_internal_process') as mock_process:
        agent.process_observation(observation)
        mock_process.assert_called_once()
```

#### Running Unit Tests
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_agents/test_agent_manager.py -v

# Run tests matching pattern
pytest tests/unit/ -k "test_agent" -v
```

### Integration Tests (20% of test suite)

#### Purpose
- Test interaction between components
- Verify API endpoints with real database
- Test multi-agent coordination
- Validate security integrations

#### Structure
```
tests/integration/
├── test_api_integration.py
├── test_agent_coordination.py
├── test_auth_integration.py
├── test_database_integration.py
├── test_security_monitoring.py
└── test_websocket_integration.py
```

#### Key Integration Scenarios
```python
# Test complete authentication flow
def test_complete_authentication_flow():
    # Register user
    response = client.post("/api/v1/auth/register", json={
        "username": "testuser",
        "password": "securepass123",
        "email": "test@example.com"
    })
    assert response.status_code == 201
    
    # Login
    response = client.post("/api/v1/auth/login", json={
        "username": "testuser",
        "password": "securepass123"
    })
    assert response.status_code == 200
    tokens = response.json()
    
    # Use access token
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    response = client.get("/api/v1/agents", headers=headers)
    assert response.status_code == 200
    
    # Refresh token
    response = client.post("/api/v1/auth/refresh", json={
        "refresh_token": tokens["refresh_token"]
    })
    assert response.status_code == 200

# Test multi-agent coordination
def test_multi_agent_coordination():
    # Create multiple agents
    agent1 = create_test_agent("researcher")
    agent2 = create_test_agent("analyzer")
    
    # Create coalition
    coalition = CoalitionCoordinator()
    coalition.add_agent(agent1)
    coalition.add_agent(agent2)
    
    # Test coordination
    task = {"type": "research", "query": "AI trends 2025"}
    result = coalition.coordinate_task(task)
    
    assert result["status"] == "completed"
    assert len(result["contributions"]) == 2
    assert result["synthesis"] is not None
```

#### Running Integration Tests
```bash
# Run all integration tests
pytest tests/integration/ -v

# Run with test database
TEST_DATABASE_URL=postgresql://test:test@localhost/test_db pytest tests/integration/

# Run integration tests with coverage
pytest tests/integration/ --cov=. --cov-report=html
```

### Security Tests (Integrated across all layers)

#### Purpose
- Validate security implementations
- Test against OWASP Top 10 vulnerabilities
- Verify authentication and authorization
- Test rate limiting and DDoS protection

#### Structure
```
tests/security/
├── test_authentication_attacks.py
├── test_authorization_attacks.py
├── test_rate_limiting_integration.py
├── test_websocket_security.py
├── test_rbac_comprehensive.py
├── test_jwt_security_hardening.py
└── comprehensive_security_test_suite.py
```

#### Key Security Test Scenarios
```python
# Test JWT manipulation resistance
def test_jwt_manipulation_resistance():
    # Get valid token
    token = get_valid_jwt_token()
    
    # Try to manipulate token
    manipulated_token = manipulate_jwt_claims(token, {"role": "admin"})
    
    # Verify manipulation is detected
    response = client.get("/api/v1/admin/users", 
                         headers={"Authorization": f"Bearer {manipulated_token}"})
    assert response.status_code == 401
    assert "Invalid token" in response.json()["error"]["message"]

# Test rate limiting
def test_rate_limiting_enforcement():
    # Make requests up to limit
    for i in range(60):  # Rate limit is 60/minute
        response = client.get("/api/v1/health")
        assert response.status_code == 200
    
    # Next request should be rate limited
    response = client.get("/api/v1/health")
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["error"]["message"]

# Test SQL injection prevention
def test_sql_injection_prevention():
    malicious_input = "'; DROP TABLE users; --"
    
    response = client.post("/api/v1/agents", json={
        "name": malicious_input,
        "template": "research_v2"
    })
    
    # Should not cause SQL injection
    assert response.status_code in [400, 422]  # Validation error
    
    # Verify tables still exist
    with get_db_connection() as conn:
        result = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in result.fetchall()]
        assert "users" in tables
```

#### Running Security Tests
```bash
# Run all security tests
pytest tests/security/ -v

# Run OWASP Top 10 tests
pytest tests/security/test_owasp_validation.py -v

# Run comprehensive security suite
python tests/security/comprehensive_security_test_suite.py

# Generate security report
python scripts/security/generate_security_report.py
```

### Performance Tests

#### Purpose
- Validate system performance under load
- Test scalability characteristics
- Benchmark critical operations
- Monitor for performance regressions

#### Structure
```
tests/performance/
├── test_agent_performance.py
├── test_api_performance.py
├── test_database_performance.py
├── test_threading_performance.py
├── test_memory_optimization.py
└── benchmarks/
    ├── agent_benchmarks.py
    └── coordination_benchmarks.py
```

#### Key Performance Scenarios
```python
# Test single agent performance
def test_single_agent_performance():
    agent = Agent(template="research_v2")
    observation = {"type": "text", "content": "Test query"}
    
    # Measure response time
    start_time = time.time()
    result = agent.process_observation(observation)
    end_time = time.time()
    
    response_time = end_time - start_time
    
    # Performance target: < 2 seconds
    assert response_time < 2.0
    assert result["status"] == "processed"

# Test concurrent agent performance
def test_concurrent_agent_performance():
    num_agents = 10
    agents = [Agent(template="research_v2") for _ in range(num_agents)]
    
    def process_observation(agent):
        observation = {"type": "text", "content": "Test query"}
        return agent.process_observation(observation)
    
    # Measure concurrent processing
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_agents) as executor:
        futures = [executor.submit(process_observation, agent) for agent in agents]
        results = [future.result() for future in futures]
    end_time = time.time()
    
    total_time = end_time - start_time
    
    # Should complete within 5 seconds for 10 agents
    assert total_time < 5.0
    assert all(result["status"] == "processed" for result in results)

# Test memory usage
def test_memory_usage():
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Create many agents
    agents = [Agent(template="research_v2") for _ in range(100)]
    
    # Process observations
    for agent in agents:
        agent.process_observation({"type": "text", "content": "Test"})
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (< 500MB)
    assert memory_increase < 500 * 1024 * 1024
```

#### Running Performance Tests
```bash
# Run performance tests
pytest tests/performance/ -v

# Run benchmarks
python tests/performance/benchmarks/agent_benchmarks.py

# Generate performance report
python scripts/performance/generate_performance_report.py
```

### End-to-End Tests (10% of test suite)

#### Purpose
- Test complete user workflows
- Validate system behavior from user perspective
- Test real-world scenarios
- Ensure all components work together

#### Structure
```
tests/e2e/
├── test_user_workflows.py
├── test_agent_lifecycle.py
├── test_api_client_integration.py
└── test_websocket_workflows.py
```

#### Key E2E Scenarios
```python
# Test complete user workflow
def test_complete_user_workflow():
    # User registration and login
    user_client = TestClient()
    user_client.register("testuser", "password123")
    user_client.login("testuser", "password123")
    
    # Create agent
    agent_response = user_client.post("/api/v1/agents", json={
        "name": "My Research Agent",
        "template": "research_v2",
        "parameters": {"temperature": 0.7}
    })
    assert agent_response.status_code == 201
    agent_id = agent_response.json()["id"]
    
    # Run inference
    inference_response = user_client.post("/api/v1/inference", json={
        "agent_id": agent_id,
        "query": "What are the latest AI trends?"
    })
    assert inference_response.status_code == 200
    
    # Check results
    result = inference_response.json()
    assert result["status"] == "completed"
    assert "response" in result
    assert len(result["response"]) > 0
    
    # Clean up
    user_client.delete(f"/api/v1/agents/{agent_id}")

# Test WebSocket workflow
def test_websocket_workflow():
    with TestWebSocketClient() as ws_client:
        # Connect and authenticate
        ws_client.connect()
        ws_client.authenticate("testuser", "password123")
        
        # Subscribe to agent events
        ws_client.send({
            "type": "subscribe",
            "channels": ["agent_events"]
        })
        
        # Create agent (should trigger event)
        agent_id = create_test_agent("research_v2")
        
        # Wait for event
        event = ws_client.wait_for_message(timeout=5)
        assert event["type"] == "agent_created"
        assert event["data"]["agent_id"] == agent_id
        
        # Run inference (should trigger progress events)
        run_inference(agent_id, "Test query")
        
        # Wait for completion event
        completion_event = ws_client.wait_for_message(timeout=10)
        assert completion_event["type"] == "inference_completed"
        assert completion_event["data"]["agent_id"] == agent_id
```

#### Running E2E Tests
```bash
# Run E2E tests
pytest tests/e2e/ -v

# Run E2E tests with real services
E2E_ENVIRONMENT=staging pytest tests/e2e/ -v

# Run specific E2E scenario
pytest tests/e2e/test_user_workflows.py::test_complete_user_workflow -v
```

## Test Data Management

### Test Fixtures

Create reusable test fixtures for common test scenarios:

```python
# tests/fixtures/factories.py
from factory import Factory, Faker, SubFactory
from factory.alchemy import SQLAlchemyModelFactory

class UserFactory(SQLAlchemyModelFactory):
    class Meta:
        model = User
        sqlalchemy_session_persistence = "commit"
    
    username = Faker("user_name")
    email = Faker("email")
    password_hash = "$2b$12$test_hash"
    role = "user"
    created_at = Faker("date_time_this_year")

class AgentFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Agent
        sqlalchemy_session_persistence = "commit"
    
    name = Faker("name")
    template = "research_v2"
    status = "active"
    user = SubFactory(UserFactory)
    created_at = Faker("date_time_this_year")

class CoalitionFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Coalition
        sqlalchemy_session_persistence = "commit"
    
    name = Faker("company")
    strategy = "collaborative"
    status = "active"
    created_at = Faker("date_time_this_year")
```

### Test Database Management

```python
# tests/conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base

@pytest.fixture(scope="session")
def test_db():
    # Create test database
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    TestingSessionLocal = sessionmaker(bind=engine)
    
    yield TestingSessionLocal
    
    # Cleanup
    Base.metadata.drop_all(engine)

@pytest.fixture
def db_session(test_db):
    session = test_db()
    try:
        yield session
    finally:
        session.rollback()
        session.close()

@pytest.fixture
def test_user(db_session):
    user = UserFactory()
    db_session.add(user)
    db_session.commit()
    return user

@pytest.fixture
def test_agent(db_session, test_user):
    agent = AgentFactory(user=test_user)
    db_session.add(agent)
    db_session.commit()
    return agent
```

## Continuous Integration

### CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:6
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run linting
      run: |
        flake8 .
        black --check .
        mypy .
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=. --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Run security tests
      run: |
        pytest tests/security/ -v
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Quality Gates

```bash
# scripts/quality_gates.sh
#!/bin/bash

set -e

echo "Running quality gates..."

# Code formatting
echo "Checking code formatting..."
black --check .
echo "✓ Code formatting passed"

# Linting
echo "Running linting..."
flake8 .
echo "✓ Linting passed"

# Type checking
echo "Running type checking..."
mypy .
echo "✓ Type checking passed"

# Unit tests with coverage
echo "Running unit tests..."
pytest tests/unit/ --cov=. --cov-report=html --cov-fail-under=80
echo "✓ Unit tests passed with >80% coverage"

# Integration tests
echo "Running integration tests..."
pytest tests/integration/ -v
echo "✓ Integration tests passed"

# Security tests
echo "Running security tests..."
pytest tests/security/ -v
python scripts/security/validate_security_posture.py
echo "✓ Security tests passed"

# Performance tests
echo "Running performance tests..."
pytest tests/performance/ -v
echo "✓ Performance tests passed"

echo "All quality gates passed! 🎉"
```

## Test Reporting

### Coverage Reporting

```bash
# Generate coverage report
pytest --cov=. --cov-report=html --cov-report=xml --cov-report=term

# View coverage report
open htmlcov/index.html
```

### Test Metrics

```python
# scripts/test_metrics.py
import json
import subprocess
from datetime import datetime

def collect_test_metrics():
    # Run tests with JSON output
    result = subprocess.run([
        "pytest", "--json-report", "--json-report-file=test_report.json"
    ], capture_output=True, text=True)
    
    with open("test_report.json", "r") as f:
        report = json.load(f)
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": report["summary"]["total"],
        "passed": report["summary"]["passed"],
        "failed": report["summary"]["failed"],
        "skipped": report["summary"]["skipped"],
        "duration": report["duration"],
        "coverage": get_coverage_percentage()
    }
    
    return metrics

def get_coverage_percentage():
    result = subprocess.run([
        "coverage", "report", "--format=json"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        coverage_data = json.loads(result.stdout)
        return coverage_data["totals"]["percent_covered"]
    return 0

if __name__ == "__main__":
    metrics = collect_test_metrics()
    print(json.dumps(metrics, indent=2))
```

## Best Practices

### Test Organization

1. **Clear Test Structure**: Use descriptive test names and organize tests logically
2. **Test Independence**: Each test should be independent and not rely on others
3. **Fast Feedback**: Unit tests should run quickly, integration tests moderately fast
4. **Deterministic Tests**: Tests should produce consistent results
5. **Clean Test Code**: Test code should be as clean as production code

### Test Data

1. **Use Factories**: Create test data using factory patterns
2. **Isolate Test Data**: Each test should create its own test data
3. **Clean Up**: Ensure test data is cleaned up after tests
4. **Realistic Data**: Use realistic test data that represents actual usage

### Mocking Guidelines

1. **Mock External Dependencies**: Mock external services and databases
2. **Don't Mock What You Don't Own**: Avoid mocking internal business logic
3. **Verify Interactions**: Test that mocked interactions happen correctly
4. **Integration Tests**: Use real dependencies in integration tests

### Performance Testing

1. **Baseline Measurements**: Establish performance baselines
2. **Regression Detection**: Monitor for performance regressions
3. **Load Testing**: Test under realistic load conditions
4. **Resource Monitoring**: Monitor CPU, memory, and network usage

## Troubleshooting

### Common Test Issues

#### Flaky Tests
```bash
# Run tests multiple times to identify flaky tests
pytest --count=10 tests/unit/test_flaky.py

# Use pytest-xdist for parallel execution
pytest -n auto tests/unit/
```

#### Slow Tests
```bash
# Identify slow tests
pytest --durations=10 tests/

# Profile test execution
pytest --profile tests/unit/test_slow.py
```

#### Memory Leaks in Tests
```python
# Use memory profiler
import tracemalloc

def test_memory_usage():
    tracemalloc.start()
    
    # Run test logic
    run_test_logic()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    assert current < 100 * 1024 * 1024  # Less than 100MB
```

### Test Environment Issues

#### Database Connection Issues
```bash
# Check database connection
psql -h localhost -U test_user -d test_db -c "SELECT 1;"

# Reset test database
python scripts/reset_test_database.py
```

#### Redis Connection Issues
```bash
# Check Redis connection
redis-cli ping

# Clear Redis test data
redis-cli FLUSHDB
```

## Documentation and Maintenance

### Test Documentation

1. **Test Plans**: Document test scenarios and expected outcomes
2. **Test Cases**: Maintain test case documentation
3. **Coverage Reports**: Regular coverage analysis and improvement
4. **Performance Baselines**: Document performance expectations

### Maintenance Tasks

#### Weekly
```bash
# Update test dependencies
pip install -r requirements-dev.txt --upgrade

# Run full test suite
pytest --cov=. --cov-report=html

# Review test coverage
open htmlcov/index.html
```

#### Monthly
```bash
# Analyze test metrics
python scripts/analyze_test_metrics.py

# Review and update test documentation
vim docs/TESTING_PROCEDURES.md

# Clean up obsolete tests
python scripts/cleanup_obsolete_tests.py
```

## Conclusion

This testing framework ensures high-quality, secure, and performant software through comprehensive testing practices. By following these procedures, we maintain confidence in our codebase and enable safe, rapid development.

For questions or improvements to this testing framework, please contact the Quality Assurance team or create an issue in the project repository.

---

**Document Version**: 1.0
**Last Updated**: January 16, 2025
**Next Review**: February 16, 2025
**Maintained By**: QA Team

---

*This document is part of the FreeAgentics documentation suite. Regular updates ensure it remains current with our testing practices.*