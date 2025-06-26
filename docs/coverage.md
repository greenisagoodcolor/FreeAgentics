# Code Coverage Guide

This guide explains how to generate, view, and manage code coverage reports for the FreeAgentics project.

## Overview

FreeAgentics uses comprehensive code coverage tracking for both frontend (TypeScript/React) and backend (Python) code. Coverage reports help ensure code quality and identify untested areas.

## Quick Start

### Generate Coverage Reports

```bash
# Frontend coverage only
npm run test:coverage

# Backend coverage only
coverage run -m pytest tests/
coverage html

# Both frontend and backend
make coverage

# Full coverage workflow (generate, merge, view)
make coverage-full
```

### View Coverage Reports

```bash
# View frontend coverage in browser
npm run coverage:view

# View all coverage reports
./scripts/view-coverage.sh

# Interactive coverage manager
npm run coverage:report
```

## Coverage Tools

### Frontend (Jest)

The frontend uses Jest with the following coverage configuration:

- **Coverage Directory**: `coverage/`
- **Report Formats**: lcov, html, json, text
- **Coverage Thresholds**: 80% for statements, branches, functions, and lines
- **Excluded Files**: test files, config files, mocks, node_modules

### Backend (Coverage.py)

The backend uses Coverage.py with pytest:

- **Coverage Directory**: `coverage/html/`
- **Report Formats**: html, xml, json, terminal
- **Configuration**: `.coveragerc`
- **Excluded Files**: tests, migrations, virtual environments

### Codecov Integration

The project is configured to upload coverage reports to Codecov:

1. Set `CODECOV_TOKEN` in your environment or CI secrets
2. Coverage is automatically uploaded in CI/CD pipelines
3. Manual upload: `npm run coverage:upload`

## Coverage Scripts

### Package.json Scripts

- `test:coverage` - Run frontend tests with coverage
- `test:coverage:watch` - Run frontend tests with coverage in watch mode
- `coverage:view` - Open frontend HTML report
- `coverage:upload` - Upload to Codecov
- `coverage:report` - Interactive coverage manager
- `coverage:full` - Full coverage workflow
- `coverage:badge` - Generate coverage badge

### Makefile Commands

- `make coverage` - Generate all coverage reports
- `make coverage-view` - View reports in browser
- `make coverage-upload` - Upload to Codecov
- `make coverage-interactive` - Launch interactive manager
- `make coverage-full` - Complete coverage workflow

### Interactive Coverage Manager

Run the interactive coverage manager:

```bash
npm run coverage:report
# or
node scripts/coverage-report.js
```

Options:

1. Generate Frontend Coverage
2. Generate Backend Coverage
3. Generate All Coverage
4. View Coverage Reports
5. Upload to Codecov
6. Generate Coverage Badge
7. Full Coverage Workflow

Command-line usage:

```bash
# Specific operations
node scripts/coverage-report.js frontend
node scripts/coverage-report.js backend
node scripts/coverage-report.js all
node scripts/coverage-report.js view
node scripts/coverage-report.js upload
node scripts/coverage-report.js badge
node scripts/coverage-report.js full
```

## CI/CD Integration

### GitHub Actions

Coverage is automatically generated and uploaded in CI:

- `.github/workflows/ci.yml` - Runs tests with coverage
- `.github/workflows/coverage.yml` - Dedicated coverage workflow
- `.github/workflows/code-quality.yml` - Includes coverage checks

### Coverage Requirements

Pull requests must maintain or improve coverage:

- Frontend target: 80%
- Backend target: 75%
- Patch coverage: 80%

## Configuration Files

### codecov.yml

Configures Codecov behavior:

- Coverage targets and thresholds
- Comment settings
- Flag definitions (frontend/backend)
- Ignored files

### jest.config.js

Jest coverage configuration:

- Coverage reporters
- Collection patterns
- Coverage thresholds
- Output directory

### .coveragerc

Python coverage configuration:

- Source directories
- Omit patterns
- Report settings
- Output formats

## Best Practices

1. **Run Coverage Locally**: Check coverage before pushing

   ```bash
   make coverage-full
   ```

2. **Review Uncovered Code**: Focus on critical paths

   ```bash
   # View detailed reports
   npm run coverage:view
   ```

3. **Write Tests for New Code**: Maintain coverage levels

   - Add tests alongside new features
   - Update tests when modifying code

4. **Monitor Trends**: Use Codecov to track coverage over time

   - Set up notifications for coverage drops
   - Review coverage reports in PRs

5. **Exclude Appropriately**: Don't inflate coverage
   - Exclude generated code
   - Exclude third-party code
   - Mark unreachable code with `/* istanbul ignore next */` or `# pragma: no cover`

## Troubleshooting

### No Coverage Data

If coverage reports are empty:

1. Ensure tests are actually running
2. Check file patterns in configuration
3. Verify source maps are generated (frontend)

### Low Coverage

To improve coverage:

1. Run coverage report to identify gaps
2. Focus on untested functions/branches
3. Add edge case tests
4. Test error handling paths

### Coverage Upload Fails

If Codecov upload fails:

1. Check `CODECOV_TOKEN` is set
2. Verify network connectivity
3. Check Codecov service status
4. Review upload logs for errors

## Examples

### Adding Tests for Uncovered Code

1. Generate coverage report:

   ```bash
   npm run test:coverage
   ```

2. Open HTML report:

   ```bash
   npm run coverage:view
   ```

3. Identify uncovered lines (shown in red)

4. Write tests targeting those lines:

   ```typescript
   // Example test for uncovered function
   test("handles error case", () => {
     expect(() => riskyFunction(null)).toThrow();
   });
   ```

5. Re-run coverage to verify improvement

### Setting Up Coverage Badge

1. Generate coverage data:

   ```bash
   make coverage
   ```

2. Generate badge:

   ```bash
   npm run coverage:badge
   ```

3. Add to README:

   ```markdown
   ![Coverage](https://codecov.io/gh/username/freeagentics/branch/main/graph/badge.svg)
   ```

## Resources

- [Jest Coverage Documentation](https://jestjs.io/docs/configuration#collectcoverage-boolean)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Codecov Documentation](https://docs.codecov.com/)
- [GitHub Actions Coverage](https://github.com/marketplace/actions/codecov)
