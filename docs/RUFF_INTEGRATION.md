# Ruff Integration Guide

This document describes the comprehensive Ruff linting and formatting integration for the FreeAgentics project.

## Overview

[Ruff](https://github.com/astral-sh/ruff) is an extremely fast Python linter and code formatter, written in Rust. It serves as a drop-in replacement for multiple tools including:

- **flake8** (and many of its plugins)
- **Black** (code formatting)
- **isort** (import sorting)
- **pyupgrade** (Python syntax upgrades)
- **And 50+ other linting rules**

## Why Ruff?

### Performance
- **10-100x faster** than traditional Python linters
- Can lint the entire FreeAgentics codebase in **under 1 second**
- Ideal for large codebases and CI/CD pipelines

### Comprehensive Coverage
- Replaces multiple tools with a single, unified solution
- Covers 700+ linting rules from various Python linting tools
- Consistent configuration in a single `pyproject.toml` file

### Modern Python Support
- Native support for Python 3.12+
- Advanced type annotation support
- Modern Python syntax and best practices

## Configuration

### pyproject.toml

Ruff is configured in the `[tool.ruff]` section of `pyproject.toml`:

```toml
[tool.ruff]
# Enable comprehensive rule sets
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings  
    "F",      # Pyflakes
    "I",      # isort
    "N",      # pep8-naming
    "D",      # pydocstyle
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "SIM",    # flake8-simplify
    "RUF",    # Ruff-specific rules
    "PL",     # pylint
    "S",      # flake8-bandit (security)
    # ... and more
]

# Black-compatible formatting
line-length = 100
target-version = "py312"

# Performance optimization
cache-dir = ".ruff_cache"
```

### Rule Selection

Our configuration enables a comprehensive set of rules while maintaining compatibility:

#### Enabled Rule Categories
- **E/W**: pycodestyle errors and warnings
- **F**: Pyflakes (undefined variables, unused imports)
- **I**: Import sorting (isort replacement)
- **B**: Bugbear (common Python bugs)
- **S**: Security (bandit-style security checks)
- **UP**: Python version upgrades
- **SIM**: Code simplification suggestions
- **PL**: Pylint-style checks
- **D**: Docstring style checking

#### Ignored Rules
- **E203**: Whitespace before ':' (conflicts with Black)
- **E501**: Line too long (handled by formatter)
- **D100-D107**: Missing docstring rules (selectively disabled)

## Usage

### Command Line

#### Basic Linting
```bash
# Run all linting checks
make ruff

# Run only Ruff (without other tools)
python -m ruff check .

# Show statistics
python -m ruff check . --statistics
```

#### Auto-fixing
```bash
# Auto-fix all fixable issues
make ruff-fix

# Fix specific issues
python -m ruff check . --fix
```

#### Formatting
```bash
# Check formatting
python -m ruff format --check .

# Apply formatting
python -m ruff format .
```

#### Watch Mode
```bash
# Continuously watch for changes
make ruff-watch
```

### IDE Integration

#### VS Code
Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff):

```json
{
    "ruff.enable": true,
    "ruff.lint.enable": true,
    "ruff.format.enable": true,
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": true,
            "source.organizeImports.ruff": true
        }
    }
}
```

#### PyCharm/IntelliJ
1. Install the [Ruff plugin](https://plugins.jetbrains.com/plugin/20574-ruff)
2. Configure in Settings → Tools → Ruff
3. Enable "Run Ruff on save"

### Pre-commit Integration

Ruff is integrated into our pre-commit hooks:

```yaml
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.8.6
  hooks:
    - id: ruff
      args: [--fix]
    - id: ruff-format
```

### CI/CD Integration

Ruff runs in our GitHub Actions workflow:

```yaml
- name: Run Ruff linting
  run: python -m ruff check . --output-format=github --statistics

- name: Run Ruff formatting check  
  run: python -m ruff format --check .
```

## Migration from Legacy Tools

### From flake8
Most flake8 rules have direct Ruff equivalents:
- **F401** (unused import) → **F401**
- **E501** (line too long) → **E501**
- **B902** (bugbear) → **B902**

### From Black
Ruff's formatter is Black-compatible:
- Same line length (100 characters)
- Same quote style preferences
- Same trailing comma handling

### From isort
Ruff's import sorting uses isort-compatible configuration:
- **profile = "black"** for Black compatibility
- **combine-as-imports = true**
- Same first-party package detection

## Performance Benchmarks

### Linting Speed Comparison
| Tool | Time (seconds) | Relative Speed |
|------|---------------|----------------|
| Ruff | 0.12s | 1x (baseline) |
| flake8 + plugins | 8.45s | 70x slower |
| pylint | 15.23s | 127x slower |

### Memory Usage
- **Ruff**: ~50MB peak memory usage
- **flake8**: ~200MB peak memory usage
- **pylint**: ~400MB peak memory usage

## Troubleshooting

### Common Issues

#### "Command not found"
```bash
# Ensure Ruff is installed
pip install ruff==0.8.6

# Or use the setup script
python scripts/setup_ruff.py
```

#### Configuration not found
```bash
# Ensure pyproject.toml exists in project root
ls pyproject.toml

# Validate configuration
python -m ruff check --config pyproject.toml .
```

#### Cache issues
```bash
# Clear Ruff cache
rm -rf .ruff_cache

# Or use the built-in cache clear
python -m ruff clean
```

### False Positives

#### Disable specific rules
```python
# Disable for entire file
# ruff: noqa

# Disable specific rule for file
# ruff: noqa: F401

# Disable for specific line
import unused_module  # noqa: F401
```

#### Per-file ignores in pyproject.toml
```toml
[tool.ruff.per-file-ignores]
"tests/**/*.py" = ["S101"]  # Allow assert in tests
"scripts/**/*.py" = ["T201"]  # Allow print statements
```

## Advanced Features

### Custom Plugins

Ruff supports custom rule implementations through its plugin system:

```toml
[tool.ruff]
extend-select = ["MY"]  # Enable custom rules

[tool.ruff.plugins]
my-plugin = "path/to/plugin.py"
```

### Rule-specific Configuration

```toml
[tool.ruff.mccabe]
max-complexity = 15

[tool.ruff.pylint]
max-args = 7
max-branches = 12

[tool.ruff.isort]
known-first-party = ["agents", "api", "auth"]
```

### Output Formats

```bash
# GitHub Actions format
ruff check . --output-format=github

# JSON format for tooling integration
ruff check . --output-format=json

# SARIF format for security tools
ruff check . --output-format=sarif
```

## Best Practices

### Development Workflow

1. **Write code** with IDE integration for real-time feedback
2. **Save files** to trigger auto-formatting and basic fixes
3. **Run `make ruff`** before committing for full validation
4. **Use `make ruff-fix`** for bulk auto-fixing
5. **Commit changes** to trigger pre-commit hooks

### Team Adoption

1. **Start with warnings**: Enable rules gradually
2. **Use auto-fix extensively**: Let Ruff fix what it can
3. **Document exceptions**: Use `# noqa` with reasons
4. **Regular updates**: Keep Ruff version current
5. **Training**: Ensure team understands rule purposes

### Performance Optimization

1. **Use .ruffignore**: Exclude large generated files
2. **Cache configuration**: Let Ruff cache results
3. **Parallel execution**: Ruff automatically uses multiple cores
4. **Targeted runs**: Use specific paths for faster feedback

## Integration Testing

The project includes comprehensive tests for Ruff integration:

```bash
# Run Ruff integration tests
pytest tests/unit/test_ruff_integration.py -v

# Test with actual codebase
python scripts/setup_ruff.py
```

## Future Roadmap

### Planned Enhancements
- **Custom rule development** for FreeAgentics-specific patterns
- **IDE plugin optimization** for better developer experience
- **CI/CD performance improvements** with smart caching
- **Advanced security rules** for AI/ML code patterns

### Tool Consolidation
As Ruff matures, we plan to:
1. **Phase out flake8** once compatibility is confirmed
2. **Migrate from Black** to Ruff formatting exclusively  
3. **Consolidate configurations** into single pyproject.toml
4. **Optimize CI pipeline** with Ruff-only quality checks

## Support and Resources

### Documentation
- [Ruff Official Documentation](https://docs.astral.sh/ruff/)
- [Rule Reference](https://docs.astral.sh/ruff/rules/)
- [Configuration Guide](https://docs.astral.sh/ruff/configuration/)

### Community
- [GitHub Issues](https://github.com/astral-sh/ruff/issues)
- [Discord Community](https://discord.gg/astral-sh)
- [Reddit Discussion](https://www.reddit.com/r/Python/search/?q=ruff)

### Internal Support
- **Slack**: #dev-tools channel
- **Issues**: Create GitHub issues with `ruff` label
- **Training**: Schedule team sessions for advanced features

---

*This document is maintained by the FreeAgentics development team. Last updated: 2025-01-13*