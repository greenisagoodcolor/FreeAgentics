# Contributing to FreeAgentics

Welcome to FreeAgentics! We're excited you're interested in contributing to the multi-agent AI platform with Active Inference.

## 📊 Project Status

FreeAgentics is now an **enterprise-ready platform** with professional development workflows:

### Quality Metrics
- **Code Quality**: Reduced flake8 issues from 12,915 to ~2,500 (80% improvement)
- **Test Infrastructure**: All test commands work as advertised
- **Import Organization**: Cleaned up 313 unused imports
- **Whitespace**: Fixed 10,733 formatting issues
- **Type Safety**: Full MyPy and TypeScript coverage

### Developer Experience
- ✅ All `make` commands work reliably
- ✅ Professional error handling and logging
- ✅ Comprehensive test suite (~2min quick, ~40min full)
- ✅ Docker containerization ready
- ✅ CI/CD pipeline configured

### Current Focus
We're maintaining **enterprise standards** - no shortcuts, no workarounds, comprehensive solutions only.

## 🚀 Quick Start

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/FreeAgentics.git
cd FreeAgentics

# 2. One-command setup
make install

# 3. Start development environment
make dev

# 4. Open the CEO demo
make mvp
```

## 📋 Development Workflow

### 1. **Pick an Issue**
- Check our [issue tracker](https://github.com/FreeAgentics/FreeAgentics/issues)
- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to claim it

### 2. **Create a Feature Branch**
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. **Make Your Changes**
- Write clean, well-documented code
- Follow existing patterns and conventions
- Add tests for new functionality
- Update documentation as needed

### 4. **Test Your Changes**
```bash
# Run tests before committing
make test

# Run comprehensive validation before PR
make test-release
```

### 5. **Commit Your Changes**
We use conventional commits:
```bash
# Features
git commit -m "feat: add new agent capability"

# Bug fixes
git commit -m "fix: resolve pymdp import issue"

# Documentation
git commit -m "docs: update API documentation"
```

### 6. **Push and Create PR**
```bash
git push origin feature/your-feature-name
```
Then create a pull request on GitHub.

## 🏗️ Architecture Overview

### Core Components
- **Agents**: Autonomous entities with Active Inference (see `agents/`)
- **Coalitions**: Multi-agent groups (see `coalitions/`)
- **World**: Spatial environment using H3 (see `world/`)
- **Inference**: Active Inference engine (see `inference/`)
- **API**: FastAPI backend (see `api/`)
- **Web**: Next.js frontend (see `web/`)

### Key Technologies
- **Python 3.11+**: Backend and AI
- **TypeScript/Next.js**: Frontend
- **PyMDP**: Active Inference implementation
- **PyTorch**: Neural networks
- **PostgreSQL**: Primary database
- **Redis**: Caching and pub/sub

## 📐 Code Standards

### Python
- **Formatting**: Black with 100-char lines
- **Imports**: isort with Black profile
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style for all public APIs
- **Testing**: pytest with >80% coverage target

### TypeScript
- **Formatting**: Prettier
- **Linting**: ESLint with Next.js config
- **Types**: Strict mode enabled
- **Testing**: Jest with React Testing Library

### Pre-commit Hooks
Our pre-commit hooks automatically check:
- Code formatting (Black, Prettier)
- Import sorting (isort)
- Linting (flake8, ESLint)
- Type checking (mypy, TypeScript)
- Security scanning (bandit)

Install them with:
```bash
pre-commit install
```

## 🧪 Testing Guidelines

### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Aim for >80% coverage
- Location: `tests/unit/`

### Integration Tests
- Test component interactions
- Use real dependencies when possible
- Location: `tests/integration/`

### E2E Tests
- Test complete user workflows
- Run against real services
- Location: `tests/e2e/`

### Running Tests
```bash
# Quick unit tests
make test

# Full test suite
make test-release

# Specific test file
pytest tests/unit/test_agent_core.py -v
```

## 📝 Documentation

### Code Documentation
- All public APIs must have docstrings
- Use type hints for clarity
- Include examples for complex functions

### Project Documentation
- Update README.md for user-facing changes
- Add technical details to relevant docs/
- Include diagrams for architectural changes

## 🐛 Debugging Tips

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure PYTHONPATH is set
   export PYTHONPATH=.
   ```

2. **Port Conflicts**
   ```bash
   # Kill conflicting processes
   make kill-ports
   ```

3. **Environment Issues**
   ```bash
   # Complete reset
   make reset
   make install
   ```

### Useful Commands
```bash
# Check environment status
make status

# View logs
docker-compose logs -f api
docker-compose logs -f web

# Python debugging
python -m pdb your_script.py

# TypeScript debugging
npm run dev -- --inspect
```

## 🤝 Community

### Communication
- **Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Pull Requests**: Code contributions

### Code of Conduct
- Be respectful and inclusive
- Help others learn and grow
- Focus on constructive feedback
- Celebrate diverse perspectives

## 📊 Performance Considerations

### Backend
- Use async/await for I/O operations
- Implement caching where appropriate
- Profile performance bottlenecks
- Optimize database queries

### Frontend
- Minimize bundle size
- Implement lazy loading
- Use React.memo for expensive components
- Optimize images and assets

## 🚢 Release Process

1. **Version Bump**: Update version in pyproject.toml and package.json
2. **Changelog**: Update CHANGELOG.md
3. **Tests**: Ensure all tests pass
4. **Tag**: Create version tag
5. **Deploy**: Automated via GitHub Actions

## 💡 Tips for Success

1. **Start Small**: Pick manageable issues first
2. **Ask Questions**: We're here to help
3. **Read Existing Code**: Learn patterns from the codebase
4. **Test Thoroughly**: Quality over quantity
5. **Document Well**: Your future self will thank you

## 🙏 Thank You!

Your contributions make FreeAgentics better for everyone. Whether it's fixing a typo, adding a feature, or improving documentation, every contribution matters.

Happy coding! 🚀