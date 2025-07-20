# Release Blocker Issues - v1.0.0-alpha+

## Critical Issues Preventing Zero-Error State

### 1. Type Annotation Errors (17 total)

**knowledge_graph/nlp_entity_extractor.py**:
- Line 75: `self._entity_cache = {}` - needs type annotation
- Line 390: `grouped = {}` - needs type annotation  
- Line 413: `relationships = []` - needs type annotation

**knowledge_graph/fallback_classes.py**:
- Lines 16, 20, 61, 65: Methods returning None have `return True` statements

**inference/gnn/validator.py**:
- Line 377: Returning Any from function declared to return ValidationResult

**inference/gnn/h3_spatial_integration.py**:
- Line 37: `self.h3_cache = {}` - needs type annotation
- Line 57: Returning Any from function declared to return `str | None`
- Line 65: Returning Any from function declared to return `tuple[float, float] | None`
- Line 89: Returning Any from function declared to return `int | None`
- Line 214: `relationships = {` - needs type annotation
- Line 234: "Collection[Any]" has no attribute "append"
- Line 278: `cluster = set()` - needs type annotation

**inference/gnn/feature_extractor.py**:
- Line 327: Type mismatch - assigning list to set variable
- Line 419: `adj_list = {i: set() for i in range(num_nodes)}` - needs type annotation

### 2. Test Failures

**Memory Tracking Test**:
```
tests/benchmarks/test_performance_regression.py:82
assert peak > end  # Failed: 763.7109375 > 763.7109375
```

**CI Integration Test**:
```
tests/benchmarks/test_performance_regression.py:538
assert "Overall Status: WARNING" in comment  # String not found
```

**Benchmark Consistency Test**:
```
benchmarks/performance_suite.py:147
TypeError: BasicExplorerAgent.__init__() got an unexpected keyword argument 'num_states'
```

### 3. Missing Test Dependencies

- `pytest-benchmark` package not installed
- Causes 2 test errors for benchmark fixtures

### 4. Security Issues (39 Low Severity)

**Pseudo-random generator usage (B311)**:
- 6 instances of `random` module usage that should use `secrets`

**Exception handling patterns**:
- 4 instances of try-except-continue (B112)
- 2 instances of try-except-pass (B110)

**Shell/subprocess usage**:
- 27 instances of potential command injection vulnerabilities

## Fix Priority Order

1. **Install missing dependencies**: `pip install pytest-benchmark`
2. **Fix type annotations**: Add proper type hints to all flagged locations
3. **Fix test failures**: Update test assertions and agent initialization
4. **Address security warnings**: Replace random with secrets, improve exception handling

## Validation Commands

After fixes, run:
```bash
pre-commit run --all-files
python -m pytest --cov=. -q
make docker-build
```

All must pass with zero errors before release.