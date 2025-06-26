# Code Quality Guide

This guide documents all code quality processes, tools, and best practices for the FreeAgentics project.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Code Quality Tools](#code-quality-tools)
4. [Development Workflow](#development-workflow)
5. [CI/CD Quality Gates](#cicd-quality-gates)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Overview

FreeAgentics maintains high code quality through automated tooling, consistent standards, and continuous monitoring. Our quality toolchain includes:

- **Linting**: ESLint (JS/TS), Flake8 (Python)
- **Formatting**: Prettier (JS/TS), Black (Python)
- **Type Checking**: TypeScript, mypy
- **Testing**: Jest (frontend), pytest (backend)
- **Coverage**: Codecov integration
- **Git Hooks**: Husky + lint-staged
- **Bundle Analysis**: size-limit

## Quick Start

### Initial Setup

```bash
# Install all dependencies
npm install
pip install -r requirements-dev.txt

# Set up git hooks
npm run prepare

# Run all quality checks
npm run quality:full
```

### Daily Workflow

```bash
# Before coding
npm run quality        # Run quick quality checks

# While coding
npm run dev           # Development server with hot reload
npm run test:watch    # Run tests in watch mode

# Before committing
npm run quality:fix   # Auto-fix issues
git add .
git commit -m "feat: add new feature"  # Husky runs checks
```

## Code Quality Tools

### ESLint (JavaScript/TypeScript)

**Configuration**: `.eslintrc.js`

ESLint enforces code style and catches common errors:

```bash
# Run ESLint
npm run lint

# Auto-fix issues
npm run lint:fix

# Strict mode (fails on warnings)
npm run lint:strict
```

Key Rules:

- React Hooks rules enforced
- TypeScript strict mode
- Accessibility checks enabled
- Import order standardized

### Prettier (Code Formatting)

**Configuration**: `.prettierrc.js`

Prettier ensures consistent code formatting:

```bash
# Check formatting
npm run format:check

# Format all files
npm run format

# Format specific file types
npm run format:js
npm run format:css
npm run format:json
```

Settings:

- Single quotes
- No semicolons
- 2-space indentation
- Trailing commas
- 80-character line limit

### TypeScript

**Configuration**: `tsconfig.json`

TypeScript provides static type checking:

```bash
# Type check
npm run type-check

# Watch mode
npm run type-check:watch
```

Strict settings enabled:

- `strict: true`
- `noImplicitAny: true`
- `strictNullChecks: true`
- `noUnusedLocals: true`

### Jest (Testing)

**Configuration**: `jest.config.js`

Jest runs unit and integration tests:

```bash
# Run tests
npm test

# Watch mode
npm run test:watch

# Coverage
npm run test:coverage

# CI mode
npm run test:ci
```

Coverage thresholds:

- Statements: 80%
- Branches: 80%
- Functions: 80%
- Lines: 80%

### Python Quality Tools

**Flake8** (Linting):

```bash
# Run Flake8
npm run python:lint

# Or directly
flake8
```

**Black** (Formatting):

```bash
# Format Python code
black

# Check only
black --check
```

**mypy** (Type Checking):

```bash
# Type check Python
mypy
```

**pytest** (Testing):

```bash
# Run tests
npm run python:test

# With coverage
pytest --cov=src
```

### Husky (Git Hooks)

**Configuration**: `.husky/`

Automated checks on git operations:

- **pre-commit**: Runs lint-staged
- **pre-push**: Full quality checks
- **commit-msg**: Validates commit format

### lint-staged

**Configuration**: `.lintstagedrc.js`

Runs checks only on staged files:

```javascript
{
  '*.{js,jsx,ts,tsx}': ['eslint --fix', 'prettier --write'],
  '*.{json,md,yml,yaml}': ['prettier --write'],
  '*.py': ['black', 'flake8'],
  'package.json': ['npm run lint:package']
}
```

### Bundle Size Monitoring

**Configuration**: `.size-limit.json`

Monitors and limits bundle sizes:

```bash
# Check bundle size
npm run size

# Analyze bundle
npm run analyze
```

Current limits:

- Main bundle: 150 KB
- Total JS: 300 KB

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/new-feature

# Start development
npm run dev

# Run tests in watch mode
npm run test:watch
```

### 2. Code Quality Checks

```bash
# Quick checks
npm run quality

# Fix auto-fixable issues
npm run quality:fix

# Full quality check
npm run quality:full
```

### 3. Pre-commit Workflow

```bash
# Stage changes
git add .

# Commit (triggers pre-commit hooks)
git commit -m "feat: add new feature"

# If hooks fail, fix issues and retry
npm run quality:fix
git add .
git commit -m "feat: add new feature"
```

### 4. Pre-push Checks

```bash
# Push (triggers pre-push hooks)
git push origin feature/new-feature

# If push fails, run full checks
npm run quality:full
npm run test:ci
```

## CI/CD Quality Gates

### GitHub Actions Workflows

1. **CI Pipeline** (`.github/workflows/ci.yml`)

   - Runs on all PRs and pushes
   - Executes all quality checks
   - Generates coverage reports
   - Posts status comments

2. **Code Quality** (`.github/workflows/code-quality.yml`)

   - Deep code analysis
   - Bundle size checks
   - Security scanning
   - Performance metrics

3. **Coverage** (`.github/workflows/coverage.yml`)
   - Frontend and backend coverage
   - Codecov integration
   - Coverage trend analysis

### Quality Requirements for PRs

All pull requests must:

1. ✅ Pass all linting checks
2. ✅ Pass all tests
3. ✅ Maintain code coverage (≥80%)
4. ✅ Pass type checking
5. ✅ Follow commit conventions
6. ✅ Not exceed bundle size limits

## Best Practices

### Code Style

1. **Consistency**: Use automated formatters

   ```bash
   npm run format
   ```

2. **Type Safety**: Always use TypeScript

   ```typescript
   // ✅ Good
   function greet(name: string): string {
     return `Hello, ${name}!`;
   }

   // ❌ Bad
   function greet(name) {
     return `Hello, ${name}!`;
   }
   ```

3. **Meaningful Names**: Be descriptive

   ```typescript
   // ✅ Good
   const isUserAuthenticated = checkAuth();

   // ❌ Bad
   const flag = check();
   ```

### Testing

1. **Test Coverage**: Aim for >80%

   ```bash
   npm run test:coverage
   ```

2. **Test Structure**: Use AAA pattern

   ```typescript
   test("should calculate total correctly", () => {
     // Arrange
     const items = [{ price: 10 }, { price: 20 }];

     // Act
     const total = calculateTotal(items);

     // Assert
     expect(total).toBe(30);
   });
   ```

3. **Edge Cases**: Test boundaries

   ```typescript
   test.each([
     [[], 0],
     [null, 0],
     [undefined, 0],
     [[{ price: -10 }], 0],
   ])("handles edge case %p", (input, expected) => {
     expect(calculateTotal(input)).toBe(expected);
   });
   ```

### Commits

Follow conventional commits:

```bash
# Features
git commit -m "feat: add user authentication"

# Fixes
git commit -m "fix: resolve memory leak in agent manager"

# Documentation
git commit -m "docs: update API documentation"

# Performance
git commit -m "perf: optimize knowledge graph queries"

# Refactoring
git commit -m "refactor: simplify agent communication logic"
```

### Code Reviews

1. **Run checks locally** before pushing
2. **Keep PRs small** and focused
3. **Write descriptive** PR descriptions
4. **Respond promptly** to feedback
5. **Test manually** when needed

## Troubleshooting

### Common Issues

#### ESLint Errors

```bash
# Fix auto-fixable issues
npm run lint:fix

# Check specific file
npx eslint file.ts

# Disable rule for line (use sparingly)
// eslint-disable-next-line rule-name
```

#### TypeScript Errors

```bash
# Check specific file
npx tsc --noEmit file.ts

# Generate declaration files
npm run build:types
```

#### Test Failures

```bash
# Run specific test
npm test -- Button.test

# Update snapshots
npm test -- -u

# Debug test
node --inspect-brk node_modules/.bin/jest --runInBand
```

#### Git Hook Issues

```bash
# Skip hooks (emergency only)
git commit --no-verify -m "emergency fix"

# Reinstall hooks
npm run prepare

# Debug hook
sh .husky/pre-commit
```

#### Bundle Size Issues

```bash
# Analyze bundle
npm run analyze

# Find large dependencies
npm list --depth=0 | grep -E '\d+\.\d+\s*(MB|KB)'

# Use dynamic imports
const HeavyComponent = lazy(() => import('./HeavyComponent'));
```

### Getting Help

1. **Check documentation**: Review this guide
2. **Run diagnostics**: `npm run doctor` (if available)
3. **Check CI logs**: Review GitHub Actions output
4. **Ask team**: Post in development channel
5. **File issue**: Create GitHub issue with details

## Scripts Reference

### Quality Scripts

```bash
# Individual tools
npm run lint          # ESLint
npm run format        # Prettier
npm run type-check    # TypeScript
npm run test          # Jest

# Combined checks
npm run quality       # Quick checks
npm run quality:fix   # Auto-fix
npm run quality:full  # Everything

# CI/CD
npm run validate      # Pre-push validation
npm run ci           # Full CI checks
```

### Utility Scripts

```bash
# Analysis
npm run analyze       # Bundle analyzer
npm run size         # Size check
npm run find-deadcode # Unused exports
npm run check-deps   # Dependency check

# Development
npm run dev          # Dev server
npm run build        # Production build
npm run preview      # Preview build
```

### Python Scripts

```bash
# Quality
npm run python:lint   # Flake8
npm run python:test   # pytest
npm run backend:quality # All Python checks

# Combined
npm run all:quality   # JS + Python
npm run all:format    # Format all
```

## Interactive Tools

### FreeAgentics CLI

```bash
npm run cli
```

Interactive menu for common tasks:

- Run quality checks
- Format code
- Run tests
- Generate components
- Check bundle size

### Coverage Manager

```bash
npm run coverage:report
```

Interactive coverage reporting:

- Generate reports
- View in browser
- Upload to Codecov
- Compare changes

## Resources

- [ESLint Documentation](https://eslint.org/docs/latest/)
- [Prettier Documentation](https://prettier.io/docs/en/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [Husky Documentation](https://typicode.github.io/husky/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

Remember: **Quality is everyone's responsibility**. Use these tools to maintain high standards and catch issues early.
