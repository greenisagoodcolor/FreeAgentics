# Code Quality Troubleshooting Guide

This guide helps resolve common code quality issues in the FreeAgentics project.

## Table of Contents

1. [ESLint Issues](#eslint-issues)
2. [Prettier/Formatting Issues](#prettierformatting-issues)
3. [TypeScript Errors](#typescript-errors)
4. [Jest/Testing Issues](#jesttesting-issues)
5. [Git Hook Problems](#git-hook-problems)
6. [Coverage Issues](#coverage-issues)
7. [Bundle Size Problems](#bundle-size-problems)
8. [Python Quality Issues](#python-quality-issues)
9. [CI/CD Failures](#cicd-failures)
10. [General Solutions](#general-solutions)

## ESLint Issues

### Problem: "Parsing error: Cannot find module"

**Solution:**

```bash
# Clear cache and reinstall
rm -rf node_modules/.cache
npm install
npm run lint
```

### Problem: "React Hook useEffect has missing dependencies"

**Solution:**

```typescript
// Add dependencies
useEffect(() => {
  // effect
}, [dependency1, dependency2]);

// Or if intentional, disable for that line
useEffect(() => {
  // effect
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, []);
```

### Problem: "'X' is defined but never used"

**Solutions:**

```typescript
// Remove unused import
// Before: import { unused } from './module';

// Or prefix with underscore if intentional
const _intentionallyUnused = value;

// Or disable if necessary
// eslint-disable-next-line @typescript-eslint/no-unused-vars
```

### Problem: Import order violations

**Solution:**

```bash
# Auto-fix import order
npm run lint:fix

# Manual order (enforced by ESLint):
# 1. React imports
# 2. External packages
# 3. Internal aliases (@/)
# 4. Relative imports
# 5. CSS imports
```

## Prettier/Formatting Issues

### Problem: Prettier and ESLint conflict

**Solution:**

```bash
# Ensure prettier-eslint integration
npm install --save-dev eslint-config-prettier
npm run format
npm run lint:fix
```

### Problem: Files not being formatted

**Check:**

1. File is not in `.prettierignore`
2. File extension is supported
3. No syntax errors in file

**Solution:**

```bash
# Force format specific file
npx prettier --write path/to/file.ts

# Format all files
npm run format
```

### Problem: Line endings issues (CRLF/LF)

**Solution:**

```bash
# Configure git
git config --global core.autocrlf false

# Fix existing files
npm run format
```

## TypeScript Errors

### Problem: "Cannot find module or its corresponding type declarations"

**Solutions:**

```bash
# Install types
npm install --save-dev @types/package-name

# Or declare module
echo "declare module 'package-name';" > types/package-name.d.ts

# Clear TypeScript cache
rm -rf node_modules/.cache/typescript
```

### Problem: "Type 'X' is not assignable to type 'Y'"

**Solutions:**

```typescript
// Use type assertion (careful!)
const value = someValue as ExpectedType;

// Better: Fix the actual type
interface Props {
  value: string | number; // Allow both types
}

// Or use generics
function getValue<T>(input: T): T {
  return input;
}
```

### Problem: "Object is possibly 'null' or 'undefined'"

**Solutions:**

```typescript
// Optional chaining
const value = object?.property?.nested;

// Null coalescing
const value = input ?? defaultValue;

// Type guard
if (value !== null && value !== undefined) {
  // value is defined here
}

// Non-null assertion (use carefully)
const value = object!.property;
```

## Jest/Testing Issues

### Problem: "Cannot find module" in tests

**Solution:**

```javascript
// Check jest.config.js moduleNameMapper
module.exports = {
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/$1",
    "\\.(css|less|scss|sass)$": "<rootDir>/__mocks__/styleMock.js",
  },
};
```

### Problem: Tests timing out

**Solutions:**

```typescript
// Increase timeout for specific test
test("long running test", async () => {
  // test code
}, 10000); // 10 second timeout

// Or globally in jest.config.js
module.exports = {
  testTimeout: 10000,
};
```

### Problem: Snapshot tests failing

**Solutions:**

```bash
# Update snapshots if changes are intentional
npm test -- -u

# Update specific test snapshots
npm test ComponentName -- -u
```

### Problem: Coverage thresholds not met

**Solutions:**

```bash
# Check coverage report
npm run test:coverage
npm run coverage:view

# Focus on uncovered files
npm test -- --coverage --collectCoverageFrom='specific-file.ts'
```

## Git Hook Problems

### Problem: Pre-commit hook fails

**Debug:**

```bash
# Run hook manually
sh .husky/pre-commit

# Check what lint-staged will run
npx lint-staged --debug
```

**Solutions:**

```bash
# Fix issues
npm run quality:fix

# Emergency skip (not recommended)
git commit --no-verify -m "message"
```

### Problem: Hooks not running

**Solution:**

```bash
# Reinstall husky
rm -rf .husky
npm run prepare

# Verify hooks are executable
chmod +x .husky/*
```

### Problem: "Command not found" in hooks

**Solution:**

```bash
# Add PATH to hook
echo 'export PATH="$PATH:/usr/local/bin"' >> .husky/pre-commit
```

## Coverage Issues

### Problem: No coverage generated

**Check:**

1. Tests are actually running
2. Coverage configuration is correct
3. Source files are included

**Solution:**

```javascript
// jest.config.js
collectCoverageFrom: [
  '**/*.{ts,tsx}',
  '!**/*.d.ts',
  '!**/*.test.{ts,tsx}',
],
```

### Problem: Coverage decreased

**Solutions:**

```bash
# Find uncovered lines
npm run coverage:view

# Write targeted tests
npm test -- --coverage --watchAll
```

### Problem: Python coverage not working

**Solution:**

```bash
# Install coverage
pip install coverage pytest-cov

# Run with coverage
coverage run -m pytest
coverage report
coverage html
```

## Bundle Size Problems

### Problem: Bundle size exceeds limit

**Debug:**

```bash
# Analyze bundle
npm run analyze

# Check sizes
npm run size
```

**Solutions:**

```typescript
// Dynamic imports
const HeavyComponent = lazy(() => import('./HeavyComponent'));

// Tree shaking
import { specific } from 'large-library';
// Not: import * as all from 'large-library';

// Remove unused dependencies
npm uninstall unused-package
```

### Problem: Large dependencies

**Find them:**

```bash
# Check package sizes
npm list --depth=0 | grep -E '\d+\.\d+\s*MB'

# Use bundlephobia.com
```

**Solutions:**

1. Find lighter alternatives
2. Load on demand
3. Use CDN for large libraries

## Python Quality Issues

### Problem: Flake8 errors

**Common fixes:**

```python
# Line too long
# Split into multiple lines
long_string = (
    "First part "
    "Second part"
)

# Unused imports
# Remove them or use # noqa: F401

# Missing docstrings
def function():
    """Add docstring."""
    pass
```

### Problem: Black and Flake8 conflict

**Solution:**

```ini
# .flake8
[flake8]
max-line-length = 88
extend-ignore = E203, W503
```

### Problem: mypy type errors

**Solutions:**

```python
# Add type hints
def function(param: str) -> int:
    return len(param)

# Ignore specific errors
value = something()  # type: ignore[assignment]

# Configure mypy.ini for third-party libraries
```

## CI/CD Failures

### Problem: Works locally but fails in CI

**Common causes:**

1. Different Node/Python versions
2. Missing environment variables
3. Case sensitivity (macOS vs Linux)
4. Timezone differences

**Debug:**

```bash
# Match CI environment
node --version
python --version

# Run CI commands locally
npm run ci
```

### Problem: Timeout in GitHub Actions

**Solution:**

```yaml
# Increase timeout in workflow
jobs:
  test:
    timeout-minutes: 30
```

### Problem: Out of memory in CI

**Solution:**

```yaml
# Limit parallelism
- run: npm test -- --maxWorkers=2
```

## General Solutions

### Nuclear Option: Clean Everything

```bash
# Clean all caches and reinstall
rm -rf node_modules
rm -rf .next
rm -rf coverage
rm -rf dist
rm package-lock.json
npm cache clean --force
npm install
```

### Check Environment

```bash
# Verify versions
node --version  # Should be v18+
npm --version   # Should be v8+
python --version  # Should be 3.8+

# Check global packages
npm list -g --depth=0
```

### Update Dependencies

```bash
# Check outdated
npm outdated

# Update safely
npm update

# Update all (careful!)
npx npm-check-updates -u
npm install
```

### VS Code Settings

Ensure `.vscode/settings.json` includes:

```json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "typescript.tsdk": "node_modules/typescript/lib"
}
```

## Getting Help

If none of these solutions work:

1. **Check logs carefully** - The error message usually has the answer
2. **Search GitHub issues** - Someone may have had the same problem
3. **Minimal reproduction** - Create a minimal example that shows the issue
4. **Ask for help** with:
   - Exact error message
   - Steps to reproduce
   - What you've already tried
   - Environment details (OS, Node version, etc.)

## Prevention Tips

1. **Run quality checks before pushing**

   ```bash
   npm run quality:full
   ```

2. **Keep dependencies updated**

   ```bash
   npm outdated
   ```

3. **Don't ignore warnings** - They often become errors later

4. **Write tests as you code** - Easier than adding them later

5. **Use the tools** - They're there to help!

---

Remember: Most issues have been encountered before. Take a breath, read the error carefully, and work through it systematically.
