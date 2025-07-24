# Essential CI Pipeline

This streamlined CI pipeline focuses on the core essentials for "ready to ship" code quality.

## What It Checks

### Quality Job

- **Python Linting**: Uses `ruff` for fast code formatting and style checks
- **Security Scanning**: Basic `bandit` security scan (warnings don't fail the build)
- **Frontend Linting**: ESLint and TypeScript checks
- **Frontend Build**: Ensures the React/Next.js app builds successfully

### Tests Job

- **Core Python Tests**: Runs pytest with database integration
- **Fast Feedback**: Uses `-x` flag to stop on first failure
- **PostgreSQL Integration**: Tests against real database

### Ready Job

- **Success Indicator**: Clear visual confirmation when all checks pass
- **Dependency Gate**: Only runs when both quality and tests succeed

## Design Philosophy

- **Under 50 lines**: Compact and maintainable
- **Essential checks only**: Focus on what matters for shipping
- **Fast feedback**: Optimized for quick developer iteration
- **Practical over perfect**: Warnings don't block, failures do

## Runtime

- Quality checks: ~3-5 minutes
- Tests: ~2-4 minutes
- Total: ~5-9 minutes (parallel execution)

This replaces the previous 184-line pipeline with the same core functionality in 41 lines.
