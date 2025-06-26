# Code Quality Documentation Index

Welcome to the FreeAgentics code quality documentation. This index helps you find the right guide for your needs.

## 📚 Documentation Overview

### [Code Quality Guide](./code-quality.md)

**Comprehensive guide to all quality tools and processes**

- Overview of quality toolchain
- Detailed tool documentation
- Development workflows
- CI/CD integration
- Best practices

### [Quick Reference Card](./quality-quick-reference.md)

**Essential commands and workflows at a glance**

- Common commands cheat sheet
- Commit message formats
- Quick workflows
- VS Code shortcuts
- Quality checklist

### [Troubleshooting Guide](./quality-troubleshooting.md)

**Solutions for common quality issues**

- ESLint and Prettier problems
- TypeScript errors
- Testing issues
- Git hook failures
- CI/CD debugging

### [Coverage Guide](./coverage.md)

**Code coverage setup and usage**

- Coverage tools overview
- Generating reports
- Viewing results
- CI integration
- Best practices

## 🎯 Quick Links by Task

### Setting Up Development Environment

1. [Initial Setup](./code-quality.md#quick-start)
2. [Tool Configuration](./code-quality.md#code-quality-tools)
3. [Git Hooks Setup](./code-quality.md#husky-git-hooks)

### Daily Development

1. [Essential Commands](./quality-quick-reference.md#essential-commands)
2. [Development Workflow](./code-quality.md#development-workflow)
3. [Testing Commands](./quality-quick-reference.md#testing-commands)

### Before Committing

1. [Pre-commit Checklist](./quality-quick-reference.md#quality-checklist)
2. [Quality Fix Commands](./quality-quick-reference.md#quick-workflows)
3. [Commit Message Format](./quality-quick-reference.md#commit-message-format)

### Fixing Issues

1. [Common Problems](./quality-troubleshooting.md)
2. [Emergency Solutions](./quality-troubleshooting.md#general-solutions)
3. [Getting Help](./quality-troubleshooting.md#getting-help)

### Code Coverage

1. [Generate Coverage](./coverage.md#quick-start)
2. [View Reports](./coverage.md#view-coverage-reports)
3. [Improve Coverage](./coverage.md#best-practices)

## 🛠️ Tool-Specific Documentation

### Frontend Tools

- **ESLint**: [Configuration](./code-quality.md#eslint-javascripttypescript) | [Troubleshooting](./quality-troubleshooting.md#eslint-issues)
- **Prettier**: [Setup](./code-quality.md#prettier-code-formatting) | [Issues](./quality-troubleshooting.md#prettierformatting-issues)
- **TypeScript**: [Config](./code-quality.md#typescript) | [Errors](./quality-troubleshooting.md#typescript-errors)
- **Jest**: [Testing](./code-quality.md#jest-testing) | [Problems](./quality-troubleshooting.md#jesttesting-issues)

### Backend Tools

- **Flake8**: [Python Linting](./code-quality.md#python-quality-tools)
- **Black**: [Python Formatting](./code-quality.md#python-quality-tools)
- **pytest**: [Python Testing](./code-quality.md#python-quality-tools)
- **Coverage.py**: [Backend Coverage](./coverage.md#backend-coveragepy)

### Infrastructure

- **Husky**: [Git Hooks](./code-quality.md#husky-git-hooks) | [Hook Issues](./quality-troubleshooting.md#git-hook-problems)
- **GitHub Actions**: [CI/CD](./code-quality.md#cicd-quality-gates) | [CI Failures](./quality-troubleshooting.md#cicd-failures)
- **Codecov**: [Integration](./coverage.md#codecov-integration)

## 📊 Interactive Tools

### FreeAgentics CLI

```bash
npm run cli
```

Interactive menu for common development tasks

### Coverage Manager

```bash
npm run coverage:report
```

Interactive coverage reporting and management

### Bundle Analyzer

```bash
npm run analyze
```

Visualize and optimize bundle sizes

## 🚀 Getting Started

New to the project? Follow this path:

1. **Read**: [Quick Start](./code-quality.md#quick-start)
2. **Reference**: [Quick Reference Card](./quality-quick-reference.md)
3. **Practice**: Run `npm run quality` to see the tools in action
4. **Learn**: Review [Best Practices](./code-quality.md#best-practices)
5. **Debug**: Keep [Troubleshooting Guide](./quality-troubleshooting.md) handy

## 📝 Related Documentation

- [Development Guide](../DEVELOPMENT.md) - Overall development setup
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [README](../README.md) - Project overview

## 💡 Tips

- **Bookmark** the [Quick Reference](./quality-quick-reference.md) for daily use
- **Run** `npm run quality:full` before pushing code
- **Use** interactive tools for complex tasks
- **Check** troubleshooting guide when stuck
- **Ask** for help if documentation doesn't cover your case

---

> "Code quality is not an act, it's a habit." - Aristotle (probably)

Remember: These tools are here to help you write better code faster, not to slow you down. Embrace them and they'll become second nature!
