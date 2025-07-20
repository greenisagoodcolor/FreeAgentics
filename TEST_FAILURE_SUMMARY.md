# Test Suite Failure Summary

## Overview

- **Total Tests**: 389 tests collected
- **Failed Tests**: 96 tests failed
- **Passed Tests**: 292 tests passed
- **Collection Errors**: 3 errors (2 test files couldn't be collected)

## Test Command

```bash
source venv/bin/activate && PYTHONPATH="$(pwd)" pytest tests/unit/ -v --tb=short
```

## Failure Breakdown by File

| Test File | Failed Tests | Primary Issue |
| ----------------------------------- | ------------ | ------------------------------------------- |
| test_llm_local_manager.py | 32 | Import issues with LocalLLMProvider classes |
| test_gnn_validator.py | 17 | Module structure/import issues |
| test_gnn_feature_extractor.py | 15 | Import and functionality issues |
| test_database_integration.py | 6 | Database model/session issues |
| test_websocket.py | 5 | WebSocket connection/subscription issues |
| test_error_handling.py | 4 | Error handling functionality |
| test_active_inference_real_fixed.py | 4 | Active inference implementation |
| test_gmn_parser.py | 3 | GMN parser functionality |
| test_base_agent.py | 3 | Base agent functionality |
| test_active_inference_real.py | 3 | Active inference implementation |
| test_llm_provider_interface.py | 2 | LLM provider interface issues |
| test_knowledge_graph.py | 2 | Knowledge graph functionality |

## Collection Errors

1. **test_api_agents.py** - Pydantic schema generation error for UserRole enum
1. **test_api_system.py** - Same Pydantic schema generation error

The main issue is that `UserRole` is defined as a plain class inheriting from `str` instead of using `Enum`, which causes Pydantic to fail when trying to generate schemas for models that use it.

## Root Causes

### 1. Pydantic Schema Generation (Collection Errors)

The `UserRole` class in `auth/security_implementation.py` needs to be an Enum for Pydantic to properly serialize it:

```python
class UserRole(str):  # This should be class UserRole(str, Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    # etc.
```

### 2. Import Structure Issues

Many tests are failing due to import issues, particularly:

- LocalLLMProvider and related classes in the LLM manager tests
- GNN validator and feature extractor module imports
- Database model import conflicts

### 3. Mock/Test Setup Issues

Several tests appear to have outdated mocks or are testing against changed interfaces.

## Recommended Fix Order

1. Fix the Pydantic UserRole enum issue (affects API tests)
1. Fix LLM manager imports and module structure (32 tests)
1. Fix GNN module imports (32 tests across validator and feature extractor)
1. Fix database session and model issues (6 tests)
1. Fix remaining individual test issues

## Next Steps

To systematically fix these issues:

1. Start with the UserRole enum fix to resolve collection errors
1. Focus on the highest-impact test files (llm_local_manager, gnn_validator)
1. Ensure all imports are properly structured
1. Update mocks and test expectations to match current implementations
