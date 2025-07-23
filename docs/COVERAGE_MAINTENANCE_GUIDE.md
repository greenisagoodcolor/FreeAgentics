# Coverage Maintenance Guide

## Overview

This guide provides comprehensive instructions for maintaining and improving test coverage in the FreeAgentics project. The project follows Test-Driven Development (TDD) principles with a goal of achieving 100% test coverage for production releases.

## Coverage Infrastructure

### Configuration Files

- **`pyproject.toml`**: Main coverage configuration under `[tool.coverage.*]` sections
- **`Makefile`**: Provides convenient coverage commands
- **`Makefile.tdd`**: TDD-specific coverage requirements (100% threshold)

### Coverage Scripts

All coverage scripts are located in the `scripts/` directory:

1. **`coverage-dev.sh`**: Development coverage with relaxed thresholds
1. **`coverage-ci.sh`**: CI/CD coverage validation (80% default threshold)
1. **`coverage-release.sh`**: Release validation (100% required)
1. **`coverage-analyze-gaps.py`**: Gap analysis and prioritization tool
1. **`coverage-cleanup.sh`**: Cleanup obsolete coverage artifacts

## Quick Start

### 1. Development Workflow

```bash
# Run development coverage (fast feedback)
make coverage-dev

# Analyze coverage gaps
make coverage-gaps

# View HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### 2. Pre-Commit Validation

```bash
# Run CI-level coverage check
make coverage-ci

# For TDD checkpoint (100% required)
make tdd-checkpoint
```

### 3. Release Validation

```bash
# Validate 100% coverage for release
make coverage-release
```

## Coverage Commands

### Makefile Targets

- `make coverage` or `make coverage-dev`: Generate development coverage reports
- `make coverage-ci`: Run CI validation with 80% threshold
- `make coverage-release`: Validate 100% coverage for release
- `make coverage-gaps`: Analyze and prioritize coverage gaps

### Direct Script Usage

```bash
# Development coverage with specific test paths
./scripts/coverage-dev.sh tests/unit

# CI coverage with custom threshold
COVERAGE_THRESHOLD=85 ./scripts/coverage-ci.sh

# Gap analysis with different output formats
python scripts/coverage-analyze-gaps.py --format json --output gaps.json
python scripts/coverage-analyze-gaps.py --format markdown --output gaps.md
```

## Coverage Reports

### Report Types

1. **Terminal Report**: Immediate feedback with missing lines
1. **HTML Report**: Interactive browser-based exploration
1. **XML Report**: For CI/CD tool integration
1. **JSON Report**: For programmatic analysis
1. **LCOV Report**: For additional tooling support

### Report Locations

- **Development**:

  - HTML: `htmlcov/index.html`
  - JSON: `coverage.json`
  - XML: `coverage.xml`

- **CI/CD**:

  - All reports in `test-reports/coverage/`
  - JUnit XML: `test-reports/junit.xml`

- **Release**:

  - All reports in `test-reports/release-coverage/`
  - Certification: `test-reports/release-coverage/certification.json`

## Gap Analysis

### Understanding the Gap Report

The gap analysis tool categorizes coverage issues:

1. **Critical Modules**: Core functionality requiring immediate attention
1. **Zero Coverage**: Modules with 0% coverage
1. **Low Coverage**: Modules with \<50% coverage
1. **Category Analysis**: Coverage by architectural layer
1. **GNN/Graph Modules**: Special focus per project requirements

### Priority Classification

- **HIGH Priority**:

  - Critical modules with \<80% coverage
  - Any module with 0% coverage
  - GNN/Graph modules with \<50% coverage

- **MEDIUM Priority**:

  - Non-critical modules with \<50% coverage
  - Category-wide coverage improvements

### Sample Gap Analysis Output

```
RECOMMENDATIONS:
1. [HIGH] Critical Modules: Focus on 8 critical modules with <80% coverage
   - agents/base_agent.py
   - agents/coalition_coordinator.py
   - agents/pymdp_adapter.py

2. [HIGH] GNN Modules: Improve coverage for 16 GNN/Graph modules
   - inference/gnn/model.py
   - inference/gnn/parser.py
   - inference/gnn/validator.py
```

## Improving Coverage

### Step-by-Step Process

1. **Identify Gaps**

   ```bash
   make coverage-gaps
   cat coverage-gaps.md
   ```

1. **Focus on Critical Modules**

   - Start with HIGH priority items
   - Target critical business logic first

1. **Write Tests Following TDD**

   ```bash
   # Start TDD watch mode
   make tdd-watch

   # Write failing test (RED)
   # Implement code (GREEN)
   # Refactor (REFACTOR)
   ```

1. **Verify Coverage Improvement**

   ```bash
   make coverage-dev
   # Check specific module coverage in HTML report
   ```

1. **Commit with Coverage Check**

   ```bash
   make test-commit  # Includes coverage validation
   ```

### Best Practices

1. **Branch Coverage**: Always include branch coverage (`--cov-branch`)
1. **Missing Lines**: Use `--cov-report=term-missing` to see exact gaps
1. **Context**: Use coverage contexts to track test types
1. **Incremental Goals**: Set achievable milestones (70% → 80% → 90% → 100%)

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Run Coverage
  run: |
    ./scripts/coverage-ci.sh

- name: Upload Coverage Reports
  uses: actions/upload-artifact@v3
  with:
    name: coverage-reports
    path: test-reports/coverage/

- name: Comment PR with Coverage
  uses: py-cov-action/python-coverage-comment-action@v3
  with:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### Coverage Badges

Badge data is automatically generated:

- Development: `coverage.json` → extract `totals.percent_covered`
- CI: `test-reports/coverage/badge.json`
- Release: `test-reports/release-coverage/release-badge.json`

## Maintenance Tasks

### Regular Cleanup

```bash
# Remove obsolete coverage artifacts
./scripts/coverage-cleanup.sh

# Or use make clean
make clean
```

### Coverage Trend Tracking

1. Save periodic snapshots:

   ```bash
   cp coverage.json "coverage-$(date +%Y%m%d).json"
   ```

1. Compare coverage over time:

   ```bash
   diff coverage-20250101.json coverage-20250201.json
   ```

### Updating Coverage Thresholds

1. **Development** (pyproject.toml):

   ```toml
   [tool.coverage.report]
   fail_under = 70  # Increase gradually
   ```

1. **CI** (environment variable):

   ```bash
   COVERAGE_THRESHOLD=85 ./scripts/coverage-ci.sh
   ```

1. **Release** (always 100%):

   - No configuration needed
   - Enforced by `coverage-release.sh`

## Troubleshooting

### Common Issues

1. **"No data to report"**

   - Ensure test files are discovered: `pytest --collect-only`
   - Check coverage source paths in `pyproject.toml`
   - Verify no syntax errors: `python -m py_compile module.py`

1. **"Module was never imported"**

   - Add `__init__.py` files to packages
   - Check import paths match coverage configuration
   - Use `--cov=package` not `--cov=package/`

1. **Missing Branch Coverage**

   - Ensure `branch = true` in `pyproject.toml`
   - Use `# pragma: no branch` sparingly
   - Write tests for all conditional paths

1. **Slow Coverage Collection**

   - Use `parallel = true` in configuration
   - Run specific test subsets during development
   - Use `coverage combine` for parallel runs

### Debug Commands

```bash
# Verbose coverage run
coverage run --debug=trace -m pytest tests/unit -v

# Check configuration
coverage debug config

# List measured files
coverage debug data
```

## Advanced Topics

### Coverage Contexts

Track coverage by test type:

```bash
# Run with context
coverage run --context=unit -m pytest tests/unit
coverage run --context=integration -m pytest tests/integration

# Generate contextual report
coverage html --show-contexts
```

### Combining Coverage Data

For parallel or distributed testing:

```bash
# Run tests in parallel
coverage run -p -m pytest tests/unit
coverage run -p -m pytest tests/integration

# Combine results
coverage combine
coverage report
```

### Custom Coverage Plugins

Create custom coverage plugins for special cases:

- Dynamic code generation
- Template engines
- Custom import mechanisms

## Resources

- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [TDD Best Practices](https://martinfowler.com/bliki/TestDrivenDevelopment.html)

## Appendix: Critical Modules

The following modules are considered critical and require high coverage:

```python
CRITICAL_MODULES = {
    'agents/base_agent.py',           # Core agent functionality
    'agents/pymdp_adapter.py',        # Active inference integration
    'agents/coalition_coordinator.py', # Multi-agent coordination
    'inference/active/gmn_parser.py',  # GMN parsing logic
    'inference/active/pymdp_integration.py',  # PyMDP integration
    'inference/gnn/model.py',         # Graph neural network core
    'api/v1/auth.py',                # Authentication endpoints
    'auth/security_implementation.py', # Security implementation
    'database/models.py',             # Database models
    'coalitions/coalition_manager.py', # Coalition management
}
```

Focus testing efforts on these modules first to ensure system stability.
