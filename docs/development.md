# FreeAgentics Development Guide

This guide provides comprehensive instructions for setting up and working with the FreeAgentics development environment.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Working with Docker](#working-with-docker)
5. [Database Development](#database-development)
6. [Frontend Development](#frontend-development)
7. [Backend Development](#backend-development)
8. [Testing](#testing)
9. [Code Quality](#code-quality)
10. [Debugging](#debugging)
11. [Troubleshooting](#troubleshooting)

## Development Environment Setup

### Prerequisites

Before starting development, ensure you have the following installed:

- **Node.js** (v18.0.0 or higher)
- **Python** (v3.9 or higher)
- **Docker** and **Docker Compose** (v20.10 or higher)
- **PostgreSQL** (v13 or higher) - Optional if using Docker
- **Redis** (v6 or higher) - Optional if using Docker
- **Git** (v2.30 or higher)

### Initial Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/freeagentics.git
   cd freeagentics
   ```

2. **Set Up Environment Variables**

   ```bash
   # Use the setup script for interactive configuration
   cd environments
   ./setup-env.sh

   # Or manually copy and edit
   cp env.development .env.development
   # Edit .env.development with your configuration
   ```

3. **Install Dependencies**

   ```bash
   # Frontend dependencies
   npm install

   # Backend dependencies (includes PyMDP for Active Inference)
   pip install -r requirements.txt

   # Note: PyMDP is automatically installed as a core dependency
   # for validated Active Inference algorithms and calculations
   ```

4. **Initialize the Database**

   ```bash
   # Using the setup script
   ./scripts/setup-database.sh

   # Or manually
   cd src/database
   python manage.py init
   ```

5. **Verify Installation**

   ```bash
   # Run basic tests
   npm test
   pytest tests/unit
   ```

## Project Structure

```
freeagentics/
├── app/                    # Next.js frontend application
│   ├── (dashboard)/       # Dashboard routes
│   ├── api/              # API routes
│   ├── components/       # React components
│   └── styles/          # CSS/styling files
├── src/                   # Python backend source
│   ├── agents/          # Agent implementations
│   ├── database/        # Database models and migrations
│   ├── knowledge/       # Knowledge graph system
│   ├── simulation/      # Simulation engine
│   └── world/          # World/environment system
├── environments/          # Environment configurations
│   ├── development/     # Docker compose for dev
│   ├── demo/           # Demo environment setup
│   └── *.env files     # Environment templates
├── scripts/              # Utility scripts
├── tests/               # Test suites
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── e2e/           # End-to-end tests
├── doc/                 # Documentation
└── models/              # GNN model definitions
```

## Development Workflow

### Git Workflow

We follow a modified GitFlow workflow:

1. **Main Branches**
   - `main` - Production-ready code
   - `develop` - Integration branch for features
   - `feature/*` - Feature development
   - `hotfix/*` - Emergency fixes
   - `release/*` - Release preparation

2. **Feature Development**

   ```bash
   # Create feature branch
   git checkout -b feature/your-feature-name develop

   # Make changes and commit
   git add .
   git commit -m "feat: implement new feature"

   # Push to remote
   git push origin feature/your-feature-name

   # Create pull request
   ```

3. **Commit Message Convention**
   We use conventional commits:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `style:` - Code style changes
   - `refactor:` - Code refactoring
   - `test:` - Test additions/changes
   - `chore:` - Build process/auxiliary tool changes

### Daily Development Cycle

1. **Start Your Day**

   ```bash
   # Pull latest changes
   git pull origin develop

   # Start development environment
   docker-compose up -d
   # Or for local development
   npm run dev
   ```

2. **Check System Status**

   ```bash
   # Verify all services are running
   docker-compose ps

   # Check database connection
   cd src/database && python manage.py check

   # View logs if needed
   docker-compose logs -f
   ```

3. **Make Changes**
   - Write code following our style guides
   - Add/update tests as needed
   - Update documentation
   - Run linters before committing

4. **Before Committing**

   ```bash
   # Run tests
   npm test
   pytest

   # Run linters
   npm run lint
   python -m flake8 src/

   # Format code
   npm run format
   python -m black src/
   ```

## Working with Docker

### Docker Compose Commands

```bash
# Start all services
docker-compose -f environments/development/docker-compose.yml up -d

# Stop all services
docker-compose -f environments/development/docker-compose.yml down

# View logs
docker-compose -f environments/development/docker-compose.yml logs -f [service]

# Execute commands in containers
docker-compose -f environments/development/docker-compose.yml exec backend bash
docker-compose -f environments/development/docker-compose.yml exec frontend sh

# Rebuild containers
docker-compose -f environments/development/docker-compose.yml build --no-cache
```

### Docker Development Tips

1. **Volume Management**
   - Code is mounted as volumes for hot reload
   - Database data persists in Docker volumes
   - Use `docker volume ls` to list volumes
   - Use `docker volume prune` to clean unused volumes

2. **Network Debugging**

   ```bash
   # List networks
   docker network ls

   # Inspect network
   docker network inspect freeagentics_default
   ```

3. **Container Resources**

   ```bash
   # Monitor resource usage
   docker stats
   ```

## Database Development

### Working with Migrations

1. **Create a New Migration**

   ```bash
   cd src/database
   alembic revision -m "Add new field to agents"
   ```

2. **Auto-generate Migration**

   ```bash
   # After modifying models.py
   alembic revision --autogenerate -m "Auto migration for model changes"
   ```

3. **Apply Migrations**

   ```bash
   # Upgrade to latest
   alembic upgrade head

   # Downgrade
   alembic downgrade -1
   ```

### Database Operations

```bash
# Access database shell
docker-compose exec postgres psql -U freeagentics -d freeagentics_dev

# Backup database
docker-compose exec postgres pg_dump -U freeagentics freeagentics_dev > backup.sql

# Restore database
docker-compose exec -T postgres psql -U freeagentics freeagentics_dev < backup.sql

# Reset database
cd src/database && python manage.py drop-db && python manage.py init
```

## Frontend Development

### Development Server

```bash
# Start Next.js development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

### Component Development

1. **Creating Components**

   ```typescript
   // app/components/MyComponent.tsx
   'use client';

   import { useState } from 'react';

   interface MyComponentProps {
     title: string;
     onAction?: () => void;
   }

   export default function MyComponent({ title, onAction }: MyComponentProps) {
     const [state, setState] = useState(false);

     return (
       <div className="component-wrapper">
         <h2>{title}</h2>
         <button onClick={onAction}>Action</button>
       </div>
     );
   }
   ```

2. **Using Shadcn/UI Components**

   ```bash
   # Add a new component
   npx shadcn-ui@latest add button
   npx shadcn-ui@latest add dialog
   ```

## Backend Development

### Running the Backend

```bash
# Start FastAPI development server
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or use the Python script
python src/main.py
```

### API Development

1. **Creating New Endpoints**

   ```python
   from fastapi import APIRouter, Depends
   from sqlalchemy.orm import Session
   from src.database.connection import get_db

   router = APIRouter(prefix="/api/v1/agents", tags=["agents"])

   @router.get("/")
   async def get_agents(db: Session = Depends(get_db)):
       agents = db.query(Agent).all()
       return {"agents": agents}
   ```

## Testing

### Running Tests

```bash
# Run all tests
npm test && pytest

# Run specific test suites
npm test -- --testPathPattern=components
pytest tests/unit/test_agents.py

# Run with coverage
npm test -- --coverage
pytest --cov=src tests/
```

## Code Quality

### Linting and Formatting

1. **Python (Black, Flake8, MyPy)**

   ```bash
   # Format code
   black src/ tests/

   # Check linting
   flake8 src/ tests/

   # Type checking
   mypy src/
   ```

2. **TypeScript/JavaScript (ESLint, Prettier)**

   ```bash
   # Lint code
   npm run lint

   # Fix linting issues
   npm run lint:fix

   # Format code
   npm run format
   ```

## Debugging

### Frontend Debugging

1. **Browser DevTools**
   - Use React Developer Tools extension
   - Enable source maps in Next.js config
   - Use `debugger` statements or breakpoints

2. **VS Code Debugging**

   ```json
   // .vscode/launch.json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Next.js: debug client-side",
         "type": "chrome",
         "request": "launch",
         "url": "http://localhost:3000"
       }
     ]
   }
   ```

### Backend Debugging

1. **Python Debugging**

   ```python
   # Using debugger
   import pdb
   pdb.set_trace()
   ```

2. **Logging**

   ```python
   import logging

   logger = logging.getLogger(__name__)
   logger.info("Agent %s created", agent.id)
   ```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**

   ```bash
   # Check PostgreSQL is running
   docker-compose ps postgres

   # Check connection
   cd src/database && python manage.py check
   ```

2. **Module Import Errors**

   ```bash
   # Ensure PYTHONPATH is set
   export PYTHONPATH="${PYTHONPATH}:${PWD}"
   ```

3. **Docker Issues**

   ```bash
   # Clean rebuild
   docker-compose down -v
   docker-compose build --no-cache
   docker-compose up -d
   ```

4. **Frontend Build Issues**

   ```bash
   # Clear cache
   rm -rf .next node_modules
   npm install
   npm run build
   ```

### Getting Help

1. **Documentation**
   - Check `/doc` folder for detailed guides
   - Database: `src/database/README.md`
   - Environment: `environments/README.md`

2. **Logs**
   - Docker logs: `docker-compose logs [service]`
   - Database logs: `docker-compose logs postgres`

3. **Community**
   - GitHub Issues for bug reports
   - GitHub Discussions for questions

## Pre-Commit Hook Standards

### ✅ **Operational Status**

All pre-commit hooks have been **restored to full operational status** as of Task #31 completion.

### 🛠️ **Hook Configuration**

#### **Working Hooks (All Passing)**

- **black**: Code formatting with 100-character line length
- **isort**: Import sorting with black compatibility
- **pyupgrade**: Python syntax modernization (Python 3.9+)
- **check-added-large-files**: Large file detection (1MB threshold)
- **trim-trailing-whitespace**: Whitespace cleanup
- **fix-end-of-files**: File ending standardization
- **check-merge-conflicts**: Merge conflict detection
- **debug-statements**: Debug statement detection
- **check-docstring-first**: Docstring positioning validation

#### **Active Quality Hooks (Finding Real Issues)**

- **flake8**: Code style enforcement (100 char line length, complexity ≤10)
- **mypy**: Type checking with comprehensive configuration
- **pydocstyle**: Documentation style (Google convention)
- **bandit**: Security vulnerability scanning
- **quality-gates**: Overall quality threshold enforcement

### 🔧 **Critical Fixes Implemented**

#### **1. Syntax Error Resolution**

- **File**: `inference/gnn/alerting.py`
- **Issue**: Missing opening triple quotes in module docstring
- **Fix**: Added proper docstring delimiter
- **Impact**: Restored pyupgrade hook functionality

#### **2. Large File Exclusions**

- **Updated**: `.gitignore` with comprehensive exclusions
- **Added**: `web/.next/`, `web/out/`, `web/build/`, `web/dist/`
- **Added**: `mypy_reports/`, `bandit-report.json`, `.DS_Store`
- **Added**: Python build artifacts (`__pycache__/`, `build/`, `dist/`, `*.egg-info/`)
- **Impact**: Restored check-added-large-files hook functionality

#### **3. MyPy Configuration**

- **Updated**: `.pre-commit-config.yaml` mypy configuration
- **Changed**: `--config-file=mypy.ini` → `--config-file=pyproject.toml`
- **Added**: `numpy` to additional_dependencies
- **Impact**: Restored mypy hook functionality with proper type checking

#### **4. Module Documentation**

- **Fixed**: D100 (Missing docstring in public module) errors
- **Files**: `agents/base/data_model.py`, `agents/base/interfaces.py`
- **Method**: Moved module docstrings to file beginning (before imports)
- **Impact**: Improved pydocstyle compliance

### 🚀 **Enforcement Mechanisms**

#### **1. Pre-Commit Installation**

```bash
# Install hooks in repository
pre-commit install --install-hooks

# Validate configuration
pre-commit validate-config

# Run all hooks
pre-commit run --all-files
```

#### **2. MyPy Execution Standard**

**ALWAYS use this command format for mypy testing:**

```bash
mypy --verbose --show-traceback --show-error-context --show-column-numbers --show-error-codes --pretty --show-absolute-path your_file.py
```

#### **3. Quality Gate Enforcement**

- **Threshold**: 2000 flake8 violations (currently 2011)
- **Status**: Monitored via quality-gates hook
- **Action**: Reduce violations systematically

#### **4. File Standards**

- **Line Length**: 100 characters (flake8, black)
- **Import Sorting**: isort with black profile
- **Type Hints**: Required for new code (mypy enforced)
- **Documentation**: Module docstrings mandatory (pydocstyle)

### 📝 **Developer Workflow Updates**

#### **Before Committing (Enhanced)**

1. Run formatting: `pre-commit run black isort --all-files`
2. Check types: `mypy --verbose --show-traceback --show-error-context --show-column-numbers --show-error-codes --pretty --show-absolute-path your_file.py`
3. Test hooks: `pre-commit run --files changed_file.py`
4. Review quality gates: Check violations don't exceed threshold

#### **Handling Hook Failures**

1. **Never use `--no-verify`** (bypasses quality checks)
2. **Fix issues systematically** rather than suppressing
3. **Infrastructure issues**: Contact development team
4. **Code quality issues**: Address before committing

### 🔍 **Monitoring and Maintenance**

#### **Regular Checks**

- **Weekly**: `pre-commit validate-config`
- **Monthly**: `pre-commit autoupdate` (update hook versions)
- **Quarterly**: Review quality gate thresholds

#### **Issue Resolution Priority**

- **Syntax Errors**: Critical priority (blocks other hooks)
- **Configuration Issues**: High priority (affects infrastructure)
- **Code Quality**: Medium priority (systematic improvement)

---

For more detailed information, see:

- [Database Documentation](src/database/README.md)
- [Environment Configuration](environments/README.md)
- [API Documentation](doc/api/rest_api.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- **[ADR-002: Canonical Directory Structure](architecture/decisions/002-canonical-directory-structure.md)**
- **[ADR-003: Dependency Rules](architecture/decisions/003-dependency-rules.md)**
- **[ADR-004: Naming Conventions](architecture/decisions/004-naming-conventions.md)**

_Pre-Commit Standards Last Updated: Task #31 Completion_
