# FreeAgentics

> **Multi-agent AI platform implementing Active Inference for autonomous, mathematically-principled intelligent systems**

🛑 **v0.0.1-prototype - NOT READY FOR INVESTMENT** 🛑

Building on work from John Clippinger, Andrea Pashea, and Daniel Friedman as well as the Active Inference Institute and many others.

This is an early prototype for developers interested in Active Inference and multi-agent systems. **Only ~20% of advertised functionality is actually working.** Not ready for production use or investment evaluation.

📋 **Honest Status (2025-07-21)**: Basic PyMDP adapter working. Core pipeline (Prompt→LLM→GMN→PyMDP→KG→D3) NOT INTEGRATED. See [HONEST_PROJECT_STATUS.md](HONEST_PROJECT_STATUS.md) for full details.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-30%25-red.svg)](docs/)
[![Status](https://img.shields.io/badge/status-prototype-red.svg)](#implementation-status)
[![Investment Ready](https://img.shields.io/badge/investment-NOT_READY-red.svg)](NEMESIS_RELEASE_BLOCKERS.md)

## 🎯 What is FreeAgentics?

FreeAgentics creates **AI agents** using **Active Inference** - a mathematical framework from cognitive science. Unlike chatbots or scripted AI, our agents make decisions by minimizing free energy, leading to emergent, intelligent behavior.

## ⏰ Realistic Timeline to v1.0.0-alpha+

Based on honest assessment of current state:

- **Current**: v0.0.1-prototype (July 2025) - ~20% complete
- **Phase 1**: Fix foundation issues (2-3 weeks) - Tests, linting, observability
- **Phase 2**: Core integration (4-6 weeks) - GMN parser, LLM, Knowledge Graph, pipeline
- **Phase 3**: Production hardening (2-3 weeks) - Performance, security, deployment
- **Realistic v1.0.0-alpha+**: October-November 2025 (3-4 months)
- **Investment Ready**: December 2025 (5 months)

**Critical Blockers**: See [NEMESIS_RELEASE_BLOCKERS.md](NEMESIS_RELEASE_BLOCKERS.md) for the 50+ issues that must be resolved.

### ⚡ Features & Status (HONEST ASSESSMENT)

**Actually Working** ✅:

- 🧠 **PyMDP Adapter**: Basic Active Inference wrapper using `inferactively-pymdp` - **70% complete**
- 🗄️ **Database Integration**: PostgreSQL/SQLite backends functional - **80% complete**
- 🤖 **BasicExplorerAgent**: Simple grid world agent with belief updates - **60% complete**
- 🌐 **API Server**: FastAPI with /health endpoint (missing /metrics) - **50% complete**
- 🏗️ **Docker Builds**: Multi-arch development builds work - **70% complete**

**Partially Implemented** 🚧:

- 🎮 **Visualization**: D3 force layout exists but not integrated - **40% complete**
- 🧪 **Test Suite**: 32 tests exist but 5 files have errors, coverage broken - **40% complete**
- 🔌 **LLM Interface**: Provider interface defined, no implementation - **20% complete**
- 👥 **Multi-Agent Framework**: Basic structure, no coordination - **15% complete**

**Not Implemented** ❌:

- 📝 **GMN Parser**: No implementation found - **0% complete**
- 🔗 **End-to-End Pipeline**: Prompt→LLM→GMN→PyMDP→KG→D3 NOT CONNECTED - **0% complete**
- 📊 **Knowledge Graph Backend**: Frontend components exist, no backend - **10% complete**
- 📈 **Observability**: No /metrics, no Prometheus, no traces - **0% complete**
- 🚀 **Production Ready**: Missing benchmarks, security, deployment - **5% complete**

**Overall Project Completion: ~20%** - See [NEMESIS_RELEASE_BLOCKERS.md](NEMESIS_RELEASE_BLOCKERS.md)

## 🚀 Quick Start & Developer Setup

### **2-Minute Start** (Development Setup)

```bash
git clone https://github.com/your-org/freeagentics.git
cd freeagentics

# Option 1: Use SQLite fallback (no PostgreSQL required)
cp .env.development .env
make install
make dev

# Option 2: Use PostgreSQL (recommended for production-like testing)
# Set DATABASE_URL in .env first:
# DATABASE_URL=postgresql://postgres:postgres@localhost:5432/freeagentics_dev
make install
make dev

# Basic UI available at http://localhost:3000
# API docs at http://localhost:8000/docs
```

### **Development Environment Setup**

The project includes automatic database configuration with SQLite fallback for easier development:

#### **Database Configuration Options**

1. **SQLite Fallback (Default for Development)**:

   - Automatically activates when `DEVELOPMENT_MODE=true` and no `DATABASE_URL` is set
   - Creates `freeagentics_dev.db` in the project root
   - Perfect for quick prototyping and local development
   - **Limitations**: Not suitable for production or multi-agent testing

2. **PostgreSQL Setup (Recommended for Production-like Testing)**:
   - Required for full functionality and multi-agent coordination
   - Better performance and concurrent access support
   - Set `DATABASE_URL` in your `.env` file:
     ```bash
     DATABASE_URL=postgresql://postgres:postgres@localhost:5432/freeagentics_dev
     ```

#### **Quick Environment Setup**

```bash
# Copy the development environment template
cp .env.development .env

# The .env.development file includes:
# - DEVELOPMENT_MODE=true (enables SQLite fallback)
# - DATABASE_URL= (left empty to trigger SQLite fallback)
# - All other required development settings
```

#### **Database Setup Verification**

```bash
# Start the system
make dev

# Check which database is being used
# Look for log message: "Using SQLite database: freeagentics_dev.db"
# or "Connected to PostgreSQL database"
```

### **Try the Active Inference Demo**

```bash
# See real Active Inference in action!
make demo             # Interactive demo with PyMDP agents
# Or run directly: make demo-ai
```

### **Essential Commands**

```bash
# Development
make install           # One-time setup (Python + Node.js)
make dev              # Start both frontend (3000) + backend (8000)
make test             # Run comprehensive test suite

# Testing
make test             # Quick validation (~2 minutes)
make test-release     # Production validation (~40 minutes)
make coverage         # Generate coverage reports

# Docker & Production
make docker           # Production deployment
make docker-validate  # Validate Docker configuration

# Troubleshooting
make kill-ports       # Fix "port in use" errors
make reset            # Nuclear option: clean restart
make status           # See what's running where
```

### **Dependency Installation Details**

The `make install` command handles all dependencies automatically:

1. **Python Dependencies**:

   - Creates a virtual environment in `venv/`
   - Installs from `requirements.txt` (including `inferactively-pymdp==0.0.7.1`)
   - All Active Inference and PyMDP dependencies are pinned for stability

1. **Node.js Dependencies**:

   - Installs frontend dependencies in `web/node_modules/`
   - Uses `npm install` with package-lock.json for reproducible builds

1. **Common Installation Issues**:

   - **"Module not found"**: Run `make clean && make install`
   - **Python version mismatch**: Ensure Python 3.9+ is installed
   - **Node version issues**: Requires Node.js 18+
   - **Permission errors**: Don't use `sudo` with `make install`

## 📊 Implementation Status

### ✅ Recently Implemented & Working

- **Active Inference Engine**: Full PyMDP integration with `inferactively-pymdp` library
- **Agent System**: `BasicExplorerAgent` with real variational inference and belief updates
- **Database Integration**: PostgreSQL backend with proper data persistence (no in-memory fallbacks)
- **API Endpoints**: CRUD operations for agents with real database storage
- **Testing Infrastructure**: Comprehensive test suite with 18+ tests covering core functionality
- **Demo System**: Interactive Active Inference demonstration with real PyMDP agents
- **World Simulation**: Grid world environment for agents with observation processing

### 🚧 Partially Implemented

- **Frontend Dashboard**: Next.js setup with basic UI (TypeScript compilation working)
- **GNN Feature Extraction**: Implementation exists but needs integration testing
- **LLM Integration**: Local LLM manager structure in place but not fully integrated
- **Coalition Formation**: Framework exists but coordination logic needs implementation
- **GMN Parser**: Generalized Notation Notation infrastructure partially complete

### ❌ Not Started

- Knowledge Graph Evolution and Learning
- Hardware Deployment Pipeline
- Production Monitoring & Observability
- Authentication & Authorization
- Advanced Multi-Agent Coordination Algorithms

## 🏗️ Codebase Architecture

### **Frontend (`web/`) - Bloomberg Terminal**

```
web/
├── app/dashboard/         # Dashboard
├── components/dashboard/  # Professional widgets & TilingWindowManager
├── styles/design-tokens.css # Professional design system
└── hooks/                 # Custom React hooks
```

### **Backend (Python) - Multi-Agent AI**

```
agents/                    # Explorer, Guardian, Merchant, Scholar
inference/                 # Active Inference engine + GNN + LLM
coalitions/               # Multi-agent coordination
api/                      # FastAPI + WebSocket real-time updates
```

## 🎯 Development Workflow

```bash
# Daily cycle
make status && make dev    # Start development
# Edit code (auto-reload)
make test                  # Quick validation
make quality              # Lint + type check
git add . && git commit && git push
```

## 🧪 Testing Strategy

FreeAgentics uses a **streamlined testing approach** focused on clarity and effectiveness:

### **Core Testing Commands**

| Command           | What It Includes                    | When To Use           | Time   |
| ----------------- | ----------------------------------- | --------------------- | ------ |
| `test`            | **Unit tests (backend + frontend)** | During development    | ~2min  |
| `test-release`    | **Complete production validation**  | Before releases       | ~40min |
| `coverage`        | **Coverage reports generation**     | Quality assessment    | ~5min  |
| `docker`          | **Production Docker deployment**    | Production releases   | ~10min |
| `docker-validate` | **Docker configuration validation** | Pre-deployment checks | ~3min  |

### **Testing Philosophy**

Following Arch Linux principles of simplicity and clarity:

- **`make test`**: Fast feedback during development
- **`make test-release`**: Comprehensive validation for production
- **`make coverage`**: Detailed coverage analysis when needed

#### 🚀 `make test-release` - Production Validation (40 minutes)

**6-Phase comprehensive validation:**

1. **Code Quality Analysis** - Linting and type checking
1. **Unit Testing** - Backend and frontend with coverage
1. **Integration Testing** - API and component integration
1. **Security Scanning** - Vulnerability and dependency audits
1. **Production Build** - Full build verification
1. **Report Generation** - Comprehensive validation summary

**Generates**: Complete validation report in `test-reports/`
**Purpose**: Production readiness confirmation

## 🔧 Customization by Developer Type

**React Developers**: Explore `web/components/dashboard/TilingWindowManager.tsx` and `design-tokens.css`\
**Python/AI Developers**: Check `agents/` and `inference/engine/` for Active Inference math\
**Full-Stack Developers**: API layer in `api/main.py` and `websocket/` real-time updates

## 🔧 Environment Configuration

For detailed environment setup instructions, see **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)**.

### **Environment Variables**

The project uses environment variables for configuration. A `.env.development` file is provided as a template:

#### **Essential Variables**

```bash
# Database Configuration
DEVELOPMENT_MODE=true              # Enables SQLite fallback and dev features
DATABASE_URL=                      # Leave empty for SQLite, or set PostgreSQL URL
                                  # Example: postgresql://user:pass@localhost:5432/dbname

# Security (MUST change in production)
SECRET_KEY=dev-secret-key         # Used for session encryption
JWT_SECRET=dev-jwt-secret         # Used for JWT token signing

# API Configuration
API_HOST=0.0.0.0                  # API bind address
API_PORT=8000                     # API port
NEXT_PUBLIC_API_URL=http://localhost:8000  # Frontend API endpoint
```

#### **Optional Services**

```bash
# Redis (caching and pub/sub)
REDIS_URL=redis://localhost:6379/0  # Optional, uses in-memory fallback if not set

# Debug and Logging
DEBUG=true                         # Enable debug mode
DEBUG_SQL=false                    # Log SQL queries (verbose)
LOG_LEVEL=DEBUG                    # Logging level: DEBUG, INFO, WARNING, ERROR

# Testing
TESTING=false                      # Set by test runner automatically

# Docker
COMPOSE_PROJECT_NAME=freeagentics-dev  # Docker Compose project name
```

### **Environment Setup Guide**

1. **Quick Start (SQLite)**:

   ```bash
   cp .env.development .env
   # Ready to go! SQLite will be used automatically
   ```

2. **PostgreSQL Development**:

   ```bash
   cp .env.development .env
   # Edit .env and add your PostgreSQL connection:
   # DATABASE_URL=postgresql://postgres:password@localhost:5432/freeagentics_dev
   ```

3. **Production Setup**:
   - Generate secure keys: `openssl rand -hex 32`
   - Set `DEVELOPMENT_MODE=false`
   - Configure production PostgreSQL
   - Enable Redis for caching
   - Set appropriate log levels

## 🆘 Common Issues & Fixes

### **General Issues**

- **"Port already in use"** → `make kill-ports`
- **"Module not found"** → `make clean && make install`
- **"Tests failing"** → `make test-full --tb=long --vvv` (max verbosity)
- **"White screen"** → `make dev-frontend` (check build errors)
- **"Agents not responding"** → `make dev-backend` (check Python logs)

### **Database Issues**

- **"Database connection failed"** → Check `DATABASE_URL` in `.env` or use SQLite fallback
- **"Permission denied for schema public"** → PostgreSQL permissions issue, switch to SQLite for development
- **"SQLite database locked"** → Close other connections or restart with `make reset`
- **"Migration failed"** → For SQLite: delete `freeagentics_dev.db` and restart

## 📊 Development Status & Quality Metrics

### **Working Prototype Platform**

FreeAgentics is a functional prototype implementing real Active Inference with PyMDP. Core functionality has been implemented and tested, making it suitable for research and development purposes.

## 📚 Documentation & Project Status

Comprehensive documentation is available in the `/docs` directory:

- **[Environment Setup](ENVIRONMENT_SETUP.md)** - Detailed environment configuration guide
- **[Database Setup](DATABASE_SETUP.md)** - Database configuration quick reference
- **[Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md)** - System design and components
- **[API Reference](docs/api/API_REFERENCE.md)** - Complete API documentation
- **[Security Guide](docs/security/README.md)** - Security implementation details
- **[Performance Guide](docs/PERFORMANCE_OPTIMIZATION_GUIDE.md)** - Optimization strategies
- **[Testing Procedures](docs/TESTING_PROCEDURES.md)** - Testing guidelines

### **Current Development Status**

**Core Functionality**: Working prototype with real Active Inference using PyMDP\
**Database**: PostgreSQL integration complete and tested\
**Testing**: Comprehensive test suite covering Active Inference, database operations, and API endpoints\
**Architecture**: Clean separation of concerns with proper abstractions

### **What's Actually Working**

✅ **Active Inference Engine**: Real PyMDP implementation with variational inference\
✅ **Multi-Agent System**: BasicExplorerAgent with belief updates and free energy minimization\
✅ **Database Integration**: PostgreSQL backend with proper persistence (no in-memory fallbacks)\
✅ **API Layer**: FastAPI with comprehensive agent CRUD operations\
✅ **Testing Infrastructure**: 18+ tests covering core functionality\
✅ **Demo System**: Interactive Active Inference demonstration

### **Architecture Components**

**Backend** (`/agents/`, `/inference/`, `/api/`): Python-based Active Inference system\
**Frontend** (`/web/`): Next.js dashboard for agent visualization\
**Database** (`/database/`): PostgreSQL with SQLAlchemy ORM\
**Testing** (`/tests/`): Comprehensive unit and integration tests\
**Examples** (`/examples/`): Demonstrations of Active Inference agents

### **Perfect For**

✅ Researchers exploring Active Inference and multi-agent systems\
✅ Developers learning mathematical foundations of AI\
✅ Open-source contributors interested in cognitive science applications\
✅ Prototype development and experimentation

### **Getting Started Quickly**

1. **Run Demo**: `make demo` - See Active Inference in action
2. **Check Tests**: `make test` - Validate core functionality
3. **Explore Code**: Start with `/agents/base_agent.py` for Active Inference implementation
4. **Review Docs**: Browse `/docs` directory for detailed documentation

## 📚 Resources

- **Active Inference Theory**: [Active Inference Institute](https://www.activeinference.org/)
- **PyMDP Documentation**: [inferactively-pymdp](https://github.com/infer-actively/pymdp)
- **Live Demo**: [localhost:3000](http://localhost:3000) after `make dev`
- **API Documentation**: [localhost:8000/docs](http://localhost:8000/docs) after `make dev`
- **Research**: Designed for cognitive science, AI research, and education

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**v0.0.1-prototype** | **Research Ready** | **Development Stage** | **Open Source**

_Making Active Inference accessible and implementable for researchers and developers._
