# TASK-001: Fix numpy import errors blocking test suite

**Status**: TODO
**Priority**: HIGH
**Category**: Critical Blockers
**Complexity**: Medium
**Estimated Hours**: 16
**Assignee**: test-engineer
**Milestone**: test-infrastructure-recovery

## Description

The test suite is completely blocked due to numpy import errors and missing dependencies. This is preventing all test execution and blocking development progress.

## Dependencies

- None (this is blocking other tasks)

## Risk Factors

- Dependency version conflicts
- Environment-specific issues
- Cascading failures once imports are fixed

## Subtasks

### 1.1 Install missing core dependencies (3h)

```bash
pip install httpx websockets PyJWT numpy scipy pandas
```

- Verify installation in virtual environment
- Update requirements-dev.txt with exact versions
- Test basic imports work

### 1.2 Fix module import path issues (4h)

- Resolve LocalLLMProvider import errors (32 test failures)
- Fix torch-geometric and h3 library imports (17 + 15 failures)
- Resolve SQLAlchemy session configuration issues (6 failures)
- Update import paths to use relative imports where appropriate

### 1.3 Mock WebSocket connections for tests (3h)

- Create mock WebSocket server fixture
- Update test_websocket.py to use mocks instead of real connections
- Ensure tests can run without network access

### 1.4 Resolve database schema initialization (4h)

- Fix "no such table: agents" errors
- Resolve table naming conflicts
- Ensure test database is properly initialized with migrations
- Create test fixtures for database setup/teardown

### 1.5 Document dependency resolution (2h)

- Update CLAUDE.md with new dependency information
- Create troubleshooting guide for common import errors
- Document virtual environment setup process

## Acceptance Criteria

- [ ] All dependencies installed and importable
- [ ] Test suite can be executed without import errors
- [ ] At least 50% of previously failing tests now run (even if failing)
- [ ] Documentation updated with setup instructions
- [ ] requirements-dev.txt contains all necessary dependencies

## Technical Notes

From NEMESIS audit: The test suite claims 81% pass rate but tests don't actually run due to numpy import errors. This task must be completed before any meaningful development can proceed.

## Expected Outcomes

- Test infrastructure becomes functional
- Baseline for test coverage can be established
- Development can proceed with confidence
