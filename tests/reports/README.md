# Test Reports Directory

This directory contains timestamped test execution reports for FreeAgentics.

## Structure

Each test run creates a timestamped directory (format: `YYYYMMDD_HHMMSS`) containing:

```
YYYYMMDD_HHMMSS/
├── test-summary.md          # High-level summary of all tests
├── index.html              # HTML navigation page for reports
├── python/                 # Python test outputs
│   ├── unit-tests.log
│   ├── integration-tests.log
│   ├── coverage-html/      # HTML coverage report
│   └── *.xml              # JUnit XML results
├── frontend/               # Frontend test outputs
│   ├── jest-tests.log
│   ├── e2e-tests.log
│   ├── coverage/          # Jest coverage report
│   └── playwright-report/  # Playwright HTML report
├── quality/               # Code quality reports
│   ├── mypy-output.log
│   ├── typescript-output.log
│   ├── flake8-report.txt
│   └── eslint-output.log
├── security/              # Security analysis
│   ├── bandit-report.json
│   └── safety-report.json
├── performance/           # Performance metrics
│   └── bundle-analysis.json
└── integration/           # Integration test results

latest -> YYYYMMDD_HHMMSS  # Symlink to most recent report
```

## Running Tests

To generate a new timestamped test report:

```bash
# Using Makefile
make test-timestamped

# Or directly
./scripts/run-all-tests.sh
```

## Viewing Reports

1. **Latest Report**: Open `tests/reports/latest/index.html` in a browser
2. **Specific Report**: Navigate to the timestamped directory
3. **Summary**: View `test-summary.md` for a quick overview

## Report Types

- **Unit Tests**: Component-level testing with coverage
- **Integration Tests**: System integration validation
- **E2E Tests**: Full user scenario testing
- **Property Tests**: Mathematical invariant validation
- **Security Tests**: Vulnerability and security checks
- **Performance Tests**: Load and optimization metrics
- **Compliance Tests**: Architecture and ADR validation

## Cleanup

Old reports can be safely deleted. The `latest` symlink always points to the most recent run.