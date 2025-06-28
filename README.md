# FreeAgentics

> **Multi-agent AI platform implementing Active Inference for autonomous, mathematically-principled intelligent systems**

🎉 **v1.0 Release - Ready for Developer Testing & Demos!** 🎉

Building on work from John Clippinger, Andrea Pashea, and Daniel Friedman as well as the active inference intstitute and many others..

This is for developers who want to test, explore, and demo a  multi-agent AI platform. It is designed to share with friends and get feedback!

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/)

## 🎯 What is FreeAgentics?

FreeAgentics creates ** AI agents** using **Active Inference** - a mathematical framework from cognitive science. Unlike chatbots or scripted AI, our agents make decisions by minimizing free energy, leading to emergent, intelligent behavior.

### ⚡ Key Features

- 🧠 **Mathematical Rigor**: Based on peer-reviewed Active Inference theory (pymdp)
- 🤖 **True Autonomy**: No hardcoded behaviors - all actions emerge from principles
- 👥 **Multi-Agent Coordination**: Agents form coalitions and collaborate dynamically
- 🎮 **Real-time Visualization**: Live dashboard showing belief states and decisions
- 🚀 **Production Ready**: Enterprise-grade performance and edge deployment
- 📝 **Natural Language**: Create agents using human-readable specifications

## 🚀 Quick Start & Developer Setup

### **30-Second Start**
```bash
git clone https://github.com/your-org/freeagentics.git
cd freeagentics
make install && make dev && make mvp
# Bloomberg Terminal dashboard opens automatically!
```

### **Essential Commands**
```bash
# Development
make install           # One-time setup (Python + Node.js)
make dev              # Start both frontend (3000) + backend (8000)
make mvp              # Open CEO-ready dashboard

# Testing (30s → 30min options)
make test             # Quick validation (30s)
make test-full        # Comprehensive (2-5min)
make test-all         # Everything (5-10min)

# Troubleshooting
make kill-ports       # Fix "port in use" errors
make reset            # Nuclear option: clean restart
make status           # See what's running where
```

## 🎮 What You Can Demo

- **Professional Interface**: Amber (#FB8B1E) color scheme, tiling windows
- **4 Autonomous AI Agents**: Explorer, Guardian, Merchant, Scholar using Active Inference
- **Real-time Visualization**: D3.js knowledge graphs, live agent activity feeds
- **Mobile Responsive**: Touch gestures, adaptive layouts, accessibility (WCAG 2.1 AA)
- **Analytics Dashboard**: Recharts-powered performance metrics

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

FreeAgentics uses a **layered testing approach** designed for different development phases:

### **Core Testing Commands**

| Command              | What It Includes                    | When To Use                | Time     | Output Location    |
| -------------------- | ----------------------------------- | -------------------------- | -------- | ------------------ |
| `test`               | **Basic unit tests**                | While coding/debugging     | 30s      | Terminal           |
| `test-full`          | **Unit tests + coverage reports**  | Before committing changes  | 2-5min   | Terminal + HTML    |
| `test-all`           | **Unit + Integration + E2E tests** | Before pushing to main     | 5-10min  | Terminal + Reports |
| `test-comprehensive` | **Complete validation suite**       | Weekly/Before releases     | 15-30min | `.test-reports/`   |

### **Detailed Breakdown**

#### 🏃‍♂️ `make test` - Quick Validation (30 seconds)
- **Frontend**: Jest unit tests only
- **Backend**: Basic pytest suite (`tests/`)
- **Purpose**: Fast feedback during development
- **Perfect for**: TDD cycles, quick sanity checks

#### 🔬 `make test-full` - Thorough Testing (2-5 minutes)  
- **Frontend**: Jest with coverage reports + verbose output
- **Backend**: pytest with full traceback (`-vvv --tb=long`) + coverage reports
- **Coverage**: HTML reports generated in `coverage/` and `htmlcov/`
- **Purpose**: Ensure code quality before committing
- **Perfect for**: Pre-commit validation, code review prep

#### 🚀 `make test-all` - Pre-Push Validation (5-10 minutes)
- **Phase 1**: All unit tests with coverage (same as `test-full`)
- **Phase 2**: Integration tests (`tests/integration/`)
- **Phase 3**: End-to-end tests (Playwright browser automation)
- **Purpose**: Comprehensive validation before pushing to main branch
- **Perfect for**: CI/CD pipeline, team collaboration

#### 🎯 `make test-comprehensive` - Expert Committee Suite (15-30 minutes)
The **complete V1 release validation** with 11 phases:
1. **Static Analysis** - TypeScript + Python type checking
2. **Security Scanning** - Bandit + Safety vulnerability checks  
3. **Code Quality** - Linting, formatting, complexity analysis
4. **Dependency Analysis** - Unused deps, bundle size validation
5. **Advanced Quality** - Black, isort, flake8, mypy deep analysis
6. **Unit Testing** - Full test suite with timeouts
7. **Integration Testing** - API + database integration
8. **Advanced Testing** - Property-based, behavior, security, chaos engineering
9. **Visual/Performance** - Screenshot validation, Lighthouse audits, load testing
10. **End-to-End** - Complete user scenarios with browser automation
11. **Coverage Analysis** - Unified reporting across frontend + backend

**Generates**: Complete quality report in `.test-reports/comprehensive-report.md`
**Purpose**: Release readiness validation, expert committee approval

### **Specialized Testing** (When Issues Arise)
```bash
make test-security          # 🔒 OWASP security validation + vulnerability scanning
make test-chaos            # 🌪️ Failure injection + system resilience testing  
make test-compliance       # 📐 Architectural compliance (ADR validation)
make test-property         # 🔬 Mathematical invariants (Active Inference)
make test-visual           # 👁️ Screenshot validation + UI regression testing
make test-performance      # ⚡ Lighthouse audits + bundle analysis + load testing
```

### **When to Use Which Test**

**During Development** → `make test` (fast feedback)  
**Before Each Commit** → `make test-full` (quality assurance)  
**Before Pushing** → `make test-all` (integration confidence)  
**Weekly/Pre-Release** → `make test-comprehensive` (production readiness)

## 🔧 Customization by Developer Type

**React Developers**: Explore `web/components/dashboard/TilingWindowManager.tsx` and `design-tokens.css`  
**Python/AI Developers**: Check `agents/` and `inference/engine/` for Active Inference math  
**Full-Stack Developers**: API layer in `api/main.py` and `websocket/` real-time updates

## 🆘 Common Issues & Fixes

**"Port already in use"** → `make kill-ports`  
**"Module not found"** → `make clean && make install`  
**"Tests failing"** → `make test-full --tb=long --vvv` (max verbosity)  
**"White screen"** → `make dev-frontend` (check build errors)  
**"Agents not responding"** → `make dev-backend` (check Python logs)

## 📊 v1 Release Status

### **What Makes This Special**
- 🏆 **Demo Ready**: Interface to test integration between agent generation, conversation, knowledge graphs, and coalitions.
- 🎨 **Enterprise UI/UX**: Enterprise-style UI/UX
- 🤖 **4 Autonomous Agents**: Active Inference, not scripted behavior
- 📊 **Real-time Visualization**: Live knowledge graphs and analytics
- 📱 **Mobile-First**: Touch gestures, accessibility, responsive design
- 🔧 **Developer-Friendly**: Comprehensive tooling and documentation

### **Perfect For**
✅ Developers exploring multi-agent AI  
✅ Demo to friends and colleagues  
✅ Learning Active Inference in practice  
✅ Contributing to open-source AI research  

### **Not Ready For**
❌ Serious investor pitches (still v1)  
❌ Production deployments (use at own risk)  
❌ Mission-critical applications  

### **Quick Stats** (TODO CHECK COVERAGE after update)
- **Lines of Code**: 50,000+ (Frontend + Backend + Tests)
- **Test Coverage**: 86.5% (83/96 passing tests)
- **Components**: 100+ React components, 15+ Python modules
- **Tech Stack**: Next.js 13+, FastAPI, D3.js, Recharts, Active Inference

## 📚 Resources

- **Documentation**: [docs/](docs/) - Complete guides for all user types
- **Live Demo**: [localhost:3000](http://localhost:3000) after `make dev`
- **Contributing**: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
- **Issues**: [GitHub Issues](https://github.com/your-org/freeagentics/issues)
- **Research**: Designed for cognitive science, AI research, and education

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**v1.0 Release** | **Developer Ready** | **Demo Ready** | **Open Source**

_Making Active Inference accessible, visual, and deployable for everyone._
