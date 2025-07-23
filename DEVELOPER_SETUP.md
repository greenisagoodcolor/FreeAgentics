# FreeAgentics Developer Setup Guide

## 🚀 One-Command Local Setup

### Prerequisites
- Python 3.12+
- Node.js 18+
- Git
- Docker (optional)

### Quick Start
```bash
# Clone and setup in one command
git clone https://github.com/greenisagoodcolor/FreeAgentics.git && cd FreeAgentics && make dev-setup && make dev
```

### Manual Setup
```bash
# 1. Clone repository
git clone https://github.com/greenisagoodcolor/FreeAgentics.git
cd FreeAgentics

# 2. Setup Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Setup web environment
cd web
npm install
cd ..

# 4. Setup environment variables
cp .env.example .env
# Edit .env with your configuration

# 5. Start development servers
make dev
```

## 🐳 Docker Development Environment

### Using Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Development Docker Setup
```bash
# Build development image
docker build -f Dockerfile.dev -t freeagentics:dev .

# Run development container
docker run -p 8000:8000 -p 3000:3000 -v $(pwd):/app freeagentics:dev
```

## 🔧 VS Code Configuration

### Recommended Extensions
- Python
- TypeScript and JavaScript Language Features
- Pylance
- ESLint
- Prettier
- GitLens
- REST Client
- Docker

### VS Code Settings
Create `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "typescript.preferences.importModuleSpecifier": "relative"
}
```

### Launch Configuration
Create `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "FastAPI Debug",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/main.py",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}",
        "DEBUG": "true"
      }
    },
    {
      "name": "Next.js Debug",
      "type": "node",
      "request": "launch",
      "program": "${workspaceFolder}/web/node_modules/.bin/next",
      "args": ["dev"],
      "cwd": "${workspaceFolder}/web"
    }
  ]
}
```

## 🐛 Debugging Guides

### Backend Debugging
```python
# Add breakpoints in code
import pdb; pdb.set_trace()

# Or use VS Code debugger
# Set breakpoint and run "FastAPI Debug" configuration
```

### Frontend Debugging
```javascript
// Browser developer tools
console.log('Debug info:', data);
debugger; // Breaks in browser debugger

// React DevTools
// Install React DevTools browser extension
```

### API Debugging
```bash
# Test API endpoints
curl -X GET http://localhost:8000/api/agents
curl -X POST http://localhost:8000/api/agents \
  -H "Content-Type: application/json" \
  -d '{"description": "Test agent"}'
```

## 🧪 API Testing Tools

### Using REST Client (VS Code)
Create `api-tests.http`:
```http
### Create Agent
POST http://localhost:8000/api/agents
Content-Type: application/json

{
  "description": "An explorer agent that searches for resources"
}

### Get All Agents
GET http://localhost:8000/api/agents

### Get Specific Agent
GET http://localhost:8000/api/agents/{{agent_id}}

### Update Agent Status
PATCH http://localhost:8000/api/agents/{{agent_id}}/status
Content-Type: application/json

{
  "status": "idle"
}

### Delete Agent
DELETE http://localhost:8000/api/agents/{{agent_id}}
```

### Using Postman
Import the collection: `postman_collection.json`

### Using curl
```bash
# Create agent
curl -X POST http://localhost:8000/api/agents \
  -H "Content-Type: application/json" \
  -d '{"description": "Test agent"}'

# Get agents
curl http://localhost:8000/api/agents

# Health check
curl http://localhost:8000/health
```

## 📊 Performance Testing

### Run Performance Tests
```bash
# Complete performance suite
python month1_performance_test.py

# Individual tests
python tests/performance/load_testing_framework.py
python tests/performance/stress_testing_framework.py
```

### Memory Profiling
```bash
# Memory usage analysis
python -m memory_profiler your_script.py

# Detailed memory tracing
python -m tracemalloc your_script.py
```

## 🧪 Testing

### Run All Tests
```bash
make test
```

### Run Specific Tests
```bash
# Python tests
pytest tests/unit/test_base_agent.py -v
pytest tests/integration/ -v

# JavaScript tests
cd web && npm test
```

### Test Coverage
```bash
# Python coverage
make coverage

# JavaScript coverage
cd web && npm run test:coverage
```

## 📝 Code Quality

### Pre-commit Hooks
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Linting
```bash
# Python linting
flake8 .
mypy .

# JavaScript linting
cd web && npm run lint
```

### Formatting
```bash
# Python formatting
black .
isort .

# JavaScript formatting
cd web && npm run format
```

## 🔐 Environment Configuration

### Required Environment Variables
```bash
# .env file
DATABASE_URL=postgresql://user:password@localhost:5432/freeagentics  # pragma: allowlist secret
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
REDIS_URL=redis://localhost:6379/0
```

### Optional Configuration
```bash
# Development settings
DEBUG=true
LOG_LEVEL=debug

# Feature flags
ENABLE_WEBSOCKETS=true
ENABLE_METRICS=true
```

## 📚 Documentation

### Generate API Documentation
```bash
# FastAPI docs (automatic)
# Visit http://localhost:8000/docs

# Generate OpenAPI spec
curl http://localhost:8000/openapi.json > openapi.json
```

### Build Documentation
```bash
# Build docs
make docs

# Serve docs locally
make docs-serve
```

## 🚀 Deployment

### Development Deployment
```bash
# Build and run
make build
make run

# Or with Docker
docker-compose -f docker-compose.prod.yml up
```

### Production Deployment
See `DEPLOYMENT_GUIDE_v1.0.0-alpha.md` for detailed instructions.

## 🛠️ Common Issues and Solutions

### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### Database Connection Issues
```bash
# Check database status
pg_isready -h localhost -p 5432

# Reset database
make db-reset
```

### Module Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use the virtual environment
source venv/bin/activate
```

### TypeScript Compilation Errors
```bash
# Clear cache and reinstall
cd web
rm -rf node_modules package-lock.json
npm install
```

## 🎯 Development Workflow

### Feature Development
1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes following TDD principles
3. Run tests: `make test`
4. Run quality checks: `make quality`
5. Commit changes: `git commit -m "feat: your feature"`
6. Push and create PR: `git push origin feature/your-feature`

### Bug Fixes
1. Create bug branch: `git checkout -b fix/bug-description`
2. Write failing test that reproduces the bug
3. Fix the bug to make test pass
4. Ensure all tests pass: `make test`
5. Commit and push: `git commit -m "fix: bug description"`

### Performance Improvements
1. Establish baseline: `python month1_performance_test.py`
2. Make improvements
3. Measure impact: `python month1_performance_test.py`
4. Document improvements in commit message

## 🤝 Contributing

### Code Style
- Follow PEP 8 for Python
- Use TypeScript for JavaScript
- Add type annotations
- Write docstrings for public functions
- Keep functions small and focused

### Testing
- Write tests for all new features
- Follow TDD principles
- Aim for >90% test coverage
- Include both unit and integration tests

### Documentation
- Update README for user-facing changes
- Add docstrings for public APIs
- Update this guide for development changes
- Include examples in documentation

## 📞 Support

### Getting Help
- Check this guide first
- Search existing issues on GitHub
- Ask questions in GitHub Discussions
- Review the codebase documentation

### Reporting Issues
- Use GitHub Issues
- Include reproduction steps
- Provide environment details
- Add relevant logs or error messages

---

**Happy coding! 🚀**

The FreeAgentics development environment is designed to be productive and enjoyable.
If you encounter any issues or have suggestions for improvement, please let us know!
