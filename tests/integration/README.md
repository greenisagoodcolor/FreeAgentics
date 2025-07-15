# FreeAgentics Integration Test Guide

This guide provides comprehensive instructions for running integration tests with nemesis-level rigor.

## Quick Start

```bash
# 1. Start test containers
./scripts/run-integration-tests.sh

# 2. Run specific test scenarios
./scripts/run-integration-tests.sh --specific test_coordination_interface_simple
```

## Prerequisites

- Docker and Docker Compose installed
- Python 3.12 virtual environment activated
- All project dependencies installed (`make install`)

## Test Infrastructure

### External Services

The integration tests require the following services:

1. **PostgreSQL** (port 5433) - Main database
2. **Redis** (port 6380) - Caching and rate limiting
3. **RabbitMQ** (port 5673) - Message queuing
4. **Elasticsearch** (port 9201) - Knowledge graph indexing
5. **MinIO** (port 9001) - S3-compatible object storage

All services are configured in `docker-compose.test.yml` with health checks.

### Environment Configuration

Test environment variables are stored in `.env.test`. To use them:

```bash
# Load test environment
source .env.test

# Or use with pytest directly
pytest tests/integration/ --env-file=.env.test
```

## Running Tests

### 1. Using the Integration Test Runner (Recommended)

```bash
# Run all integration tests with containers
./scripts/run-integration-tests.sh

# Run tests in parallel for faster execution
./scripts/run-integration-tests.sh --parallel

# Keep containers running after tests (for debugging)
./scripts/run-integration-tests.sh --no-cleanup

# Run specific test pattern
./scripts/run-integration-tests.sh --specific test_gnn_llm

# Verbose output with coverage
./scripts/run-integration-tests.sh --verbose --coverage
```

### 2. Manual Container Management

```bash
# Start test containers
docker-compose -f docker-compose.test.yml up -d

# Check health status
docker-compose -f docker-compose.test.yml ps

# Run tests
source .env.test
pytest tests/integration/ -v

# Stop containers
docker-compose -f docker-compose.test.yml down -v
```

### 3. Running Specific Test Categories

```bash
# GNN-LLM-Coalition integration tests
pytest tests/integration/test_comprehensive_gnn_llm_coalition_integration.py -v

# Critical interface tests
pytest tests/integration/test_*_interface_integration.py -v

# Performance tests
pytest tests/integration/test_*_performance*.py -v

# Coordination tests (no external deps)
pytest tests/integration/test_coordination_interface_simple.py -v
```

## Test Categories

### 1. Critical Integration Points

These tests validate data transformation at system boundaries:

- **GNN→LLM Interface** (`test_gnn_llm_interface_integration.py`)
  - Validates embedding to text transformation
  - Tests semantic preservation
  - Ensures numerical stability

- **LLM→Coalition Interface** (`test_llm_coalition_interface_integration.py`)
  - Tests strategy parsing from natural language
  - Validates coordination parameter extraction
  - Ensures formation strategies are executable

- **Coalition→Agents Interface** (`test_coalition_agents_interface_integration.py`)
  - Tests coordination message transformation
  - Validates agent action generation
  - Ensures behavioral compliance

### 2. End-to-End Scenarios

Comprehensive pipeline tests in `test_comprehensive_gnn_llm_coalition_integration.py`:

- **Resource Discovery Scenario**
  - Full pipeline from graph analysis to agent coordination
  - Tests real-world resource allocation

- **Partial Failure Resilience**
  - Simulates component failures
  - Validates graceful degradation
  - Tests recovery mechanisms

- **Performance Under Load**
  - Light (10 agents), Medium (50 agents), Heavy (100 agents)
  - Validates system scalability
  - Identifies bottlenecks

### 3. PyMDP Integration Tests

Mathematical validation of active inference components:

- `test_pymdp_validation.py` - Core PyMDP functionality
- `test_nemesis_pymdp_validation.py` - Comprehensive mathematical tests
- `test_matrix_pooling_pymdp.py` - Memory optimization validation

### 4. Security & Authentication Tests

- `test_authentication_flow.py` - JWT authentication flow
- `test_rate_limiting.py` - API rate limiting
- `test_security_headers.py` - Security header validation

## Troubleshooting

### Common Issues

1. **"DATABASE_URL environment variable is required"**
   ```bash
   source .env.test  # Load test environment
   ```

2. **"docker-compose: command not found"**
   ```bash
   # Install Docker Compose
   pip install docker-compose
   ```

3. **Container health check failures**
   ```bash
   # Check container logs
   docker-compose -f docker-compose.test.yml logs test-postgres
   ```

4. **Port conflicts**
   ```bash
   # Check for processes using test ports
   lsof -i :5433  # PostgreSQL
   lsof -i :6380  # Redis
   ```

### Debugging Failed Tests

1. **Enable verbose output**
   ```bash
   pytest tests/integration/failing_test.py -vvs
   ```

2. **Check container logs**
   ```bash
   docker-compose -f docker-compose.test.yml logs -f
   ```

3. **Interactive debugging**
   ```bash
   pytest tests/integration/failing_test.py --pdb
   ```

4. **Keep containers running**
   ```bash
   ./scripts/run-integration-tests.sh --no-cleanup
   docker exec -it freeagentics-test-postgres psql -U test_user
   ```

## Test Quality Standards

All integration tests must meet these nemesis-level standards:

1. **No Mock-Only Tests** - Tests must validate real behavior, not mock configurations
2. **Semantic Validation** - Data transformations must preserve meaning
3. **Error Injection** - All tests must handle partial failures gracefully
4. **Performance Bounds** - Response times must be <100ms for unit operations
5. **Resource Cleanup** - All tests must clean up resources properly

## Coverage Requirements

Per `Makefile.tdd`, integration tests must maintain:
- 100% line coverage for critical paths
- 100% branch coverage for error handling
- All edge cases explicitly tested

## Writing New Integration Tests

### Template for New Tests

```python
import pytest
import asyncio
from typing import Dict, Any

class TestNewIntegration:
    """Test new integration point with nemesis-level rigor."""
    
    @pytest.fixture
    def test_context(self):
        """Provide test context with all dependencies."""
        return {
            "config": self.load_test_config(),
            "validators": self.create_validators(),
            "timeout": 30
        }
    
    async def test_semantic_preservation(self, test_context):
        """Validate data transformation preserves semantic meaning."""
        # Arrange
        input_data = self.create_test_input()
        expected_semantics = self.extract_semantics(input_data)
        
        # Act
        result = await self.transform_data(input_data, test_context)
        
        # Assert
        actual_semantics = self.extract_semantics(result)
        assert self.semantics_equal(expected_semantics, actual_semantics)
    
    async def test_partial_failure_handling(self, test_context):
        """Test graceful degradation under component failures."""
        # Inject failure
        test_context["failure_mode"] = "component_timeout"
        
        # Should not raise, but degrade gracefully
        result = await self.transform_data_with_failures(test_context)
        
        # Validate degraded but functional
        assert result["status"] == "degraded"
        assert result["core_functionality"] == "preserved"
```

## Continuous Integration

For CI/CD pipelines:

```yaml
# .github/workflows/integration-tests.yml
- name: Run Integration Tests
  run: |
    ./scripts/run-integration-tests.sh --parallel --coverage
  timeout-minutes: 30
```

## Maintenance

### Weekly Tasks
1. Update test container versions
2. Review and update test data fixtures
3. Analyze test execution times

### Monthly Tasks
1. Audit test coverage gaps
2. Update integration test scenarios
3. Performance baseline updates

## Contact

For issues or questions about integration tests:
1. Check this README first
2. Review test output logs
3. Create issue with reproduction steps