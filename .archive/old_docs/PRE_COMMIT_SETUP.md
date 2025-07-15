# Pre-commit Hooks Configuration

## Overview

This project uses pre-commit hooks to maintain code quality and consistency. The hooks automatically run before each commit to catch issues early and enforce coding standards.

## Setup

### Quick Setup
```bash
# Run the setup script
./scripts/setup-pre-commit.sh
```

### Manual Setup
```bash
# Install pre-commit if not available
pip install pre-commit

# Install the hooks
pre-commit install

# Install commit-msg hook
pre-commit install --hook-type commit-msg

# Update to latest versions
pre-commit autoupdate
```

## Configured Hooks

### Python Hooks
- **black**: Code formatting with 79 character line length
- **isort**: Import sorting compatible with black
- **flake8**: Linting with docstring and bugbear checks
- **mypy**: Type checking (excludes tests/examples)
- **bandit**: Security scanning

### TypeScript/Frontend Hooks
- **typescript-check**: TypeScript compilation validation
- **eslint**: JavaScript/TypeScript linting
- **prettier**: Code formatting

### General Hooks
- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml**: YAML syntax validation
- **check-json**: JSON syntax validation
- **check-merge-conflict**: Detect merge conflict markers
- **debug-statements**: Find debug/print statements

### Security & Documentation
- **detect-secrets**: Scan for accidentally committed secrets
- **hadolint**: Docker file linting
- **mdformat**: Markdown formatting

## Usage

### Automatic (Recommended)
Hooks run automatically on `git commit`. If any hook fails, the commit is rejected.

### Manual Execution
```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black

# Run hooks on specific files
pre-commit run --files agents/base_agent.py
```

### Bypassing Hooks (Use Sparingly)
```bash
# Skip all hooks for a commit
git commit --no-verify -m "Emergency fix"

# Skip specific hook
SKIP=mypy git commit -m "Commit without mypy"
```

## Configuration Files

- **`.pre-commit-config.yaml`**: Main configuration
- **`.secrets.baseline`**: Secrets detection baseline
- **`scripts/setup-pre-commit.sh`**: Setup script

## Troubleshooting

### Common Issues

1. **Hook failures on first run**: Normal - run again after fixing issues
2. **Missing dependencies**: Install with `pip install pre-commit`
3. **TypeScript errors**: Ensure `npm install` is run in `web/` directory
4. **Slow performance**: Hooks cache results, subsequent runs are faster

### Updating Hooks
```bash
# Update to latest versions
pre-commit autoupdate

# Clean cache if needed
pre-commit clean
```

### Disabling Specific Hooks
Edit `.pre-commit-config.yaml` and add to the `exclude` pattern or comment out unwanted hooks.

## Integration with CI/CD

The configuration includes CI settings for automated pull request checks. Frontend hooks are skipped in CI to reduce complexity.

## Best Practices

1. **Run hooks before pushing**: `pre-commit run --all-files`
2. **Keep hooks updated**: Regular `pre-commit autoupdate`
3. **Fix issues promptly**: Don't accumulate hook failures
4. **Use bypass sparingly**: Only for genuine emergencies

## File Exclusions

The following are automatically excluded:
- `.archive/` directory
- `web/node_modules/`
- `__pycache__/` and other Python cache directories
- Generated files (`.next/`, `out/`, etc.)

## Hook Workflow

```
git commit
    ↓
Pre-commit hooks run
    ↓
All pass? → Commit succeeds
    ↓
Any fail? → Commit rejected, fix issues and retry
```

This ensures consistent code quality across all contributors and prevents common issues from reaching the repository.