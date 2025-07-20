# TDD Compliance - Immediate Action Checklist

## 🚨 CRITICAL: Security Module Test Coverage (0%)

### Priority 1: Authentication & Authorization (Complete within 72 hours)

#### `/auth/jwt_handler.py` - 216 uncovered statements
- [ ] Test JWT token generation with RS256
- [ ] Test token validation with expired tokens
- [ ] Test token blacklist functionality
- [ ] Test refresh token rotation
- [ ] Test token fingerprinting
- [ ] Test key rotation warnings

```python
# tests/unit/auth/test_jwt_handler_tdd.py
def test_jwt_generation_returns_valid_token():
    """Red: Write this test first"""
    handler = JWTHandler()
    token = handler.create_access_token(user_id="123")
    assert token is not None
    # This should fail - implement after
```

#### `/auth/mfa_service.py` - 259 uncovered statements
- [ ] Test TOTP generation
- [ ] Test TOTP validation with time windows
- [ ] Test backup codes generation
- [ ] Test backup codes usage and invalidation
- [ ] Test MFA setup flow
- [ ] Test MFA enforcement logic

#### `/auth/rbac_enhancements.py` - 360 uncovered statements
- [ ] Test permission checking
- [ ] Test role hierarchy
- [ ] Test resource-based permissions
- [ ] Test permission inheritance
- [ ] Test role assignment
- [ ] Test permission caching

### Priority 2: Core Domain Logic (Complete within 1 week)

#### `/agents/base_agent.py` - 479 uncovered statements
- [ ] Test agent initialization
- [ ] Test belief updates
- [ ] Test action selection
- [ ] Test observation processing
- [ ] Test preference handling
- [ ] Test state transitions

**TDD Approach**:
```python
# Step 1: Red - Write failing test
def test_agent_processes_observation():
    agent = Agent("test-id", "test-agent")
    observation = Observation(data=[1, 0])
    
    agent.process_observation(observation)
    
    assert agent.belief_state.entropy < initial_entropy

# Step 2: Green - Minimal implementation
def process_observation(self, observation):
    self.belief_state.update(observation)

# Step 3: Refactor - Improve design
def process_observation(self, observation):
    validated_obs = self._validate_observation(observation)
    self.belief_state = self._belief_updater.update(
        self.belief_state, 
        validated_obs
    )
```

### Priority 3: Database Layer (Complete within 10 days)

#### Repository Pattern Implementation
- [ ] Create `IAgentRepository` interface
- [ ] Implement `PostgresAgentRepository`
- [ ] Test save operations
- [ ] Test retrieval operations
- [ ] Test update operations
- [ ] Test transaction handling

```python
# tests/unit/infrastructure/test_agent_repository.py
def test_repository_saves_agent():
    # Arrange
    repository = InMemoryAgentRepository()
    agent = Agent("123", "TestAgent")
    
    # Act
    repository.save(agent)
    
    # Assert
    retrieved = repository.get("123")
    assert retrieved.name == "TestAgent"
```

## 🔧 Process Changes - Immediate Implementation

### 1. Git Pre-commit Hook (Install TODAY)
```bash
#!/bin/bash
# .git/hooks/pre-commit

# Run tests for changed files
changed_files=$(git diff --cached --name-only | grep "\.py$")
for file in $changed_files; do
    test_file="tests/unit/test_${file##*/}"
    if [ ! -f "$test_file" ]; then
        echo "ERROR: No test file for $file"
        echo "Create $test_file first (TDD)"
        exit 1
    fi
done

# Check coverage
coverage run -m pytest tests/unit/
coverage report --fail-under=80
```

### 2. CI/CD Pipeline Changes (Implement within 24 hours)
```yaml
# .github/workflows/tdd-enforcement.yml
name: TDD Enforcement

on: [push, pull_request]

jobs:
  tdd-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Check test files exist
        run: |
          for src_file in $(find . -name "*.py" -path "*/agents/*" -o -path "*/auth/*"); do
            test_file=$(echo $src_file | sed 's|/src/|/tests/unit/|' | sed 's|\.py$|_test.py|')
            if [ ! -f "$test_file" ]; then
              echo "Missing test for $src_file"
              exit 1
            fi
          done
      
      - name: Run coverage check
        run: |
          coverage run -m pytest
          coverage report --fail-under=80
```

### 3. Daily TDD Practices (Start Tomorrow)

#### Morning Standup Addition
- Each developer reports on:
  - [ ] Tests written yesterday
  - [ ] Coverage percentage for their module
  - [ ] Red-Green-Refactor cycles completed

#### Pair Programming Sessions
- [ ] Monday: Auth module TDD session
- [ ] Tuesday: Agent domain TDD session  
- [ ] Wednesday: Database repository TDD session
- [ ] Thursday: API endpoint TDD session
- [ ] Friday: Integration test TDD session

### 4. TDD Metrics Dashboard

Create dashboard showing:
- [ ] Tests written today
- [ ] Coverage trend by module
- [ ] Red-Green-Refactor cycle time
- [ ] Test execution time trends

## 📋 Definition of Done - Updated

No code is considered "done" without:
1. [ ] Unit tests written FIRST (red phase)
2. [ ] Tests passing (green phase)
3. [ ] Code refactored (refactor phase)
4. [ ] Coverage ≥ 80% for the module
5. [ ] No architectural violations
6. [ ] Tests run in < 100ms (unit tests)

## 🎯 Success Metrics (Track Daily)

| Metric | Current | Day 1 Target | Week 1 Target | Month 1 Target |
|--------|---------|--------------|---------------|----------------|
| Overall Coverage | 0.58% | 5% | 25% | 80% |
| Auth Coverage | 0% | 20% | 80% | 95% |
| Agent Coverage | 0% | 10% | 50% | 85% |
| New Code Coverage | N/A | 100% | 100% | 100% |
| TDD Adoption | 0% | 50% | 90% | 100% |

## 🚦 Stop/Start/Continue

### STOP
- ❌ Writing production code without tests
- ❌ Merging PRs with <80% coverage
- ❌ Importing infrastructure in domain
- ❌ Using database models as DTOs
- ❌ Mocking in unit tests (use real domain objects)

### START  
- ✅ Writing tests first (always)
- ✅ Using repository pattern
- ✅ Dependency injection
- ✅ Architecture fitness functions
- ✅ TDD katas every morning

### CONTINUE
- ✅ Code reviews (add TDD checks)
- ✅ Architecture documentation
- ✅ Performance monitoring (after tests)

## 🏃 Quick Wins (Do Today)

1. **Create test structure**:
```bash
mkdir -p tests/unit/{agents,auth,database,api,inference}
touch tests/unit/agents/test_base_agent.py
touch tests/unit/auth/test_jwt_handler.py
```

2. **Write first failing test**:
```python
# tests/unit/auth/test_jwt_handler.py
def test_jwt_handler_exists():
    from auth.jwt_handler import JWTHandler
    assert JWTHandler is not None
```

3. **Set up coverage badge**:
```bash
coverage run -m pytest
coverage-badge -o coverage.svg
# Add to README.md
```

Remember: **No production code without a failing test first!**