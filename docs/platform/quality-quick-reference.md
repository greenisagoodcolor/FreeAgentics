# Code Quality Quick Reference

## 🚀 Essential Commands

### Before Starting Work

```bash
git pull origin main
npm install
npm run quality
```

### While Developing

```bash
npm run dev          # Start dev server
npm run test:watch   # Run tests in watch mode
npm run lint:fix     # Fix linting issues
```

### Before Committing

```bash
npm run quality:fix  # Auto-fix all issues
npm run test         # Run all tests
git add .
git commit -m "type: description"
```

### Quality Check Commands

```bash
npm run quality      # Quick quality check
npm run quality:fix  # Fix auto-fixable issues
npm run quality:full # Complete quality check
```

## 📋 Commit Message Format

```
type(scope): subject

body (optional)

footer (optional)
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance

### Examples

```bash
git commit -m "feat: add user authentication"
git commit -m "fix: resolve memory leak in agent manager"
git commit -m "docs: update API documentation"
```

## 🧪 Testing Commands

```bash
npm test                # Run all tests
npm run test:watch      # Watch mode
npm run test:coverage   # With coverage
npm test Button         # Test specific file
npm test -- -u          # Update snapshots
```

## 🎨 Formatting & Linting

```bash
npm run lint            # Check linting
npm run lint:fix        # Fix linting
npm run format          # Format all files
npm run format:check    # Check formatting
```

## 📊 Code Coverage

```bash
npm run test:coverage   # Generate coverage
npm run coverage:view   # View in browser
npm run coverage:report # Interactive tool
make coverage-full      # Full workflow
```

## 🔍 Analysis Tools

```bash
npm run analyze         # Bundle analyzer
npm run size            # Check bundle size
npm run find-deadcode   # Find unused exports
npm run check-deps      # Check dependencies
```

## 🐍 Python Commands

```bash
npm run python:lint     # Lint Python code
npm run python:test     # Run Python tests
npm run backend:quality # All Python checks
black src/              # Format Python code
```

## 🏃‍♂️ Quick Workflows

### Fix Everything

```bash
npm run quality:fix
npm run format
git add .
```

### Full Quality Check

```bash
npm run quality:full
npm run test:ci
```

### Pre-Push Validation

```bash
npm run validate
```

## 🛠️ Troubleshooting

### Skip Git Hooks (Emergency Only!)

```bash
git commit --no-verify -m "emergency fix"
```

### Reset Git Hooks

```bash
rm -rf .husky
npm run prepare
```

### Clean Install

```bash
rm -rf node_modules package-lock.json
npm install
```

## 📱 Interactive Tools

```bash
npm run cli             # FreeAgentics CLI menu
make help               # Show all make commands
./scripts/view-coverage.sh  # View coverage reports
```

## ⚡ VS Code Shortcuts

- `Cmd+Shift+P` → "Format Document"
- `Cmd+Shift+P` → "ESLint: Fix all"
- `F5` → Start debugging
- `Cmd+Shift+B` → Run build task

## 🎯 Quality Checklist

Before pushing code:

- [ ] All tests pass (`npm test`)
- [ ] No linting errors (`npm run lint`)
- [ ] Code is formatted (`npm run format:check`)
- [ ] Type checks pass (`npm run type-check`)
- [ ] Coverage maintained (≥80%)
- [ ] Commit messages follow convention
- [ ] Bundle size within limits
- [ ] No console.logs in production code

## 🔗 Useful Links

- [Code Quality Guide](./code-quality.md)
- [Coverage Guide](./coverage.md)
- [Development Guide](../DEVELOPMENT.md)
- [Contributing Guide](../CONTRIBUTING.md)

---

**Remember**: When in doubt, run `npm run quality:full`!
