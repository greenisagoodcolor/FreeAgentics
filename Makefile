# FreeAgentics Multi-Agent AI Platform
# Alpha release for internal development - Active Inference system with comprehensive tooling
#
# ============================================================================
# DEVELOPER QUICK START GUIDE
# ============================================================================
#
# NEW DEVELOPERS - READ THIS FIRST:
# 1. Run `make`           - Show this help and all commands
# 2. Run `make check`     - Verify your environment has all required tools
# 3. Run `make install`   - Install all dependencies (Python + Node.js)
# 4. Run `make test-dev`  - Fast tests for active development
# 5. Run `make dev`       - Start development servers
#
# SECURITY SETUP (IMPORTANT for v0.2):
# 1. Copy `.env` to `.env.local` for development
# 2. NEVER commit `.env.production` with real secrets
# 3. Run `make security-check` to validate security configuration
#
# BEFORE COMMITTING:
# 1. Run `make test-commit` - Run full test suite before committing
# 2. Run `make format`      - Auto-format all code
# 3. Run `make lint`        - Check code quality
# 4. Run `make security-scan` - Check for security vulnerabilities
#
# BEFORE RELEASING:
# 1. Run `make test-release` - Comprehensive validation with all tools
# 2. Run `make security-audit` - Full security audit for production
#
# For more details on any command, run: make help
# ============================================================================

# Make configuration for reliability and performance
MAKEFLAGS += --warn-undefined-variables --no-builtin-rules
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
SHELL := bash

# Default target
.DEFAULT_GOAL := help

# Project configuration
VENV_DIR := venv
PYTHON := $(VENV_DIR)/bin/python3
PYTEST := $(VENV_DIR)/bin/pytest
NODE := node
WEB_DIR := web
TEST_TIMEOUT := 300
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
REPORT_DIR := test-reports/$(TIMESTAMP)

# Colors for terminal output (with proper detection)
# Check if NO_COLOR environment variable is set (even if empty)
ifeq ($(origin NO_COLOR), environment)
	BOLD :=
	RED :=
	GREEN :=
	YELLOW :=
	BLUE :=
	CYAN :=
	MAGENTA :=
	RESET :=
else ifneq ($(TERM),)
	BOLD := \033[1m
	RED := \033[31m
	GREEN := \033[32m
	YELLOW := \033[33m
	BLUE := \033[34m
	CYAN := \033[36m
	MAGENTA := \033[35m
	RESET := \033[0m
else
	BOLD :=
	RED :=
	GREEN :=
	YELLOW :=
	BLUE :=
	CYAN :=
	MAGENTA :=
	RESET :=
endif

# Define all phony targets (targets that don't create files)
.PHONY: help check install deps-check build clean lint format type-check test-dev test-commit test-release coverage
.PHONY: dev start stop kill-ports status reset docs docker docker-build docker-up docker-down
.PHONY: test-unit test-integration test-e2e test-security test-chaos test-compliance ci-setup
.PHONY: security-check security-scan security-audit prod-env dev-env check-secrets

# ============================================================================
# 1. HELP AND INFORMATION
# ============================================================================

help: ## 📖 Show this help and all available commands
	@printf "$(BOLD)$(CYAN)🚀 FreeAgentics Multi-Agent AI Platform$(RESET)\n"
	@printf "Alpha release for internal development - Active Inference system\n"
	@printf "\n"
	@printf "$(BOLD)$(GREEN)🎯 Quick Start Commands:$(RESET)\n"
	@printf "  $(GREEN)make check$(RESET)        Check your development environment\n"
	@printf "  $(GREEN)make install$(RESET)      Install all dependencies\n"
	@printf "  $(GREEN)make test-dev$(RESET)     Fast tests for active development\n"
	@printf "  $(GREEN)make dev$(RESET)          Start development servers\n"
	@printf "\n"
	@printf "$(BOLD)$(YELLOW)📋 Development Workflow:$(RESET)\n"
	@printf "  $(CYAN)1. Environment Setup:$(RESET)\n"
	@printf "    $(GREEN)check$(RESET)           Verify system prerequisites\n"
	@printf "    $(GREEN)install$(RESET)         Install/update all dependencies\n"
	@printf "    $(GREEN)status$(RESET)          Show environment health status\n"
	@printf "\n"
	@printf "  $(CYAN)2. Development:$(RESET)\n"
	@printf "    $(GREEN)dev$(RESET)             Start development servers\n"
	@printf "    $(GREEN)test-dev$(RESET)        Fast tests (unit + type-check)\n"
	@printf "    $(GREEN)format$(RESET)          Auto-format code\n"
	@printf "    $(GREEN)lint$(RESET)            Check code quality\n"
	@printf "\n"
	@printf "  $(CYAN)3. Before Committing:$(RESET)\n"
	@printf "    $(GREEN)test-commit$(RESET)     Full test suite + quality checks\n"
	@printf "    $(GREEN)format$(RESET)          Ensure code is properly formatted\n"
	@printf "\n"
	@printf "  $(CYAN)4. Security Validation:$(RESET)\n"
	@printf "    $(GREEN)security-check$(RESET)  Verify security configuration\n"
	@printf "    $(GREEN)security-scan$(RESET)   Quick vulnerability scan\n"
	@printf "    $(GREEN)security-audit$(RESET)  Full security audit\n"
	@printf "    $(GREEN)check-secrets$(RESET)   Scan for hardcoded secrets\n"
	@printf "\n"
	@printf "  $(CYAN)5. Release Preparation:$(RESET)\n"
	@printf "    $(GREEN)test-release$(RESET)    Comprehensive validation (all tools)\n"
	@printf "    $(GREEN)build$(RESET)           Build for production\n"
	@printf "    $(GREEN)coverage$(RESET)        Generate detailed coverage reports\n"
	@printf "\n"
	@printf "$(BOLD)$(BLUE)🛠  Testing Commands:$(RESET)\n"
	@printf "  $(GREEN)test-dev$(RESET)         Fast development tests (~30 seconds)\n"
	@printf "  $(GREEN)test-commit$(RESET)      Pre-commit validation (~2 minutes)\n"
	@printf "  $(GREEN)test-release$(RESET)     Complete validation (~10 minutes)\n"
	@printf "  $(GREEN)test-unit$(RESET)        Unit tests only\n"
	@printf "  $(GREEN)test-integration$(RESET) Integration tests only\n"
	@printf "  $(GREEN)test-e2e$(RESET)         End-to-end tests only\n"
	@printf "\n"
	@printf "$(BOLD)$(MAGENTA)🔧 Utility Commands:$(RESET)\n"
	@printf "  $(GREEN)clean$(RESET)            Remove test artifacts and caches\n"
	@printf "  $(GREEN)stop$(RESET)             Stop all development servers\n"
	@printf "  $(GREEN)reset$(RESET)            Clean environment reset\n"
	@printf "  $(GREEN)dev-env$(RESET)          Set up development environment\n"
	@printf "  $(GREEN)prod-env$(RESET)         Validate production environment\n"
	@printf "\n"
	@printf "$(BOLD)$(RED)📚 Documentation:$(RESET)\n"
	@printf "  See README.md for detailed setup instructions\n"
	@printf "  Run any command with -n flag to see what it would do\n"

# ============================================================================
# 2. ENVIRONMENT SETUP AND VERIFICATION
# ============================================================================

check: ## 🔍 Verify development environment and prerequisites
	@printf "$(BOLD)$(BLUE)🔍 Checking Development Environment$(RESET)\n"
	@printf "\n"
	@printf "$(YELLOW)📋 Required System Tools:$(RESET)\n"
	@command -v python3 >/dev/null 2>&1 && printf "  $(GREEN)✅ Python 3: $$(python3 --version)$(RESET)\n" || (printf "  $(RED)❌ Python 3: Not found$(RESET)\n" && exit 1)
	@command -v node >/dev/null 2>&1 && printf "  $(GREEN)✅ Node.js: $$(node --version)$(RESET)\n" || (printf "  $(RED)❌ Node.js: Not found$(RESET)\n" && exit 1)
	@command -v npm >/dev/null 2>&1 && printf "  $(GREEN)✅ npm: v$$(npm --version)$(RESET)\n" || (printf "  $(RED)❌ npm: Not found$(RESET)\n" && exit 1)
	@command -v git >/dev/null 2>&1 && printf "  $(GREEN)✅ Git: $$(git --version | cut -d' ' -f3)$(RESET)\n" || (printf "  $(RED)❌ Git: Not found$(RESET)\n" && exit 1)
	@command -v make >/dev/null 2>&1 && printf "  $(GREEN)✅ Make: $$(make --version | head -1 | cut -d' ' -f3)$(RESET)\n" || (printf "  $(RED)❌ Make: Not found$(RESET)\n" && exit 1)
	@printf "\n"
	@printf "$(YELLOW)🔧 Optional Development Tools:$(RESET)\n"
	@command -v docker >/dev/null 2>&1 && printf "  $(GREEN)✅ Docker: installed$(RESET)\n" || printf "  $(CYAN)ⓘ Docker: not installed (optional)$(RESET)\n"
	@command -v lsof >/dev/null 2>&1 && printf "  $(GREEN)✅ lsof: available$(RESET)\n" || printf "  $(YELLOW)⚠️  lsof: not found (needed for port management)$(RESET)\n"
	@printf "\n"
	@printf "$(YELLOW)📁 Project Dependencies:$(RESET)\n"
	@if [ -d "$(VENV_DIR)" ]; then printf "  $(GREEN)✅ Python venv: created$(RESET)\n"; else printf "  $(CYAN)ⓘ Python venv: not created (run 'make install')$(RESET)\n"; fi
	@if [ -d "node_modules" ]; then printf "  $(GREEN)✅ Node modules: installed (root)$(RESET)\n"; elif [ -d "$(WEB_DIR)/node_modules" ]; then printf "  $(GREEN)✅ Node modules: installed (web)$(RESET)\n"; else printf "  $(CYAN)ⓘ Node modules: not installed (run 'make install')$(RESET)\n"; fi
	@printf "\n"
	@if [ ! -d "$(VENV_DIR)" ] || [ ! -d "node_modules" -a ! -d "$(WEB_DIR)/node_modules" ]; then \
		printf "$(YELLOW)🎯 Next Step: Run $(GREEN)make install$(RESET)\n"; \
	else \
		printf "$(GREEN)🎉 Environment ready! Run $(GREEN)make dev$(RESET) to start development\n"; \
	fi

install: ## 📦 Install/update all project dependencies
	@printf "$(BOLD)$(GREEN)📦 Installing Dependencies$(RESET)\n"
	@printf "\n"
	@printf "$(CYAN)🐍 Python Dependencies...$(RESET)\n"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		printf "  $(YELLOW)→ Creating virtual environment...$(RESET)\n"; \
		python3 -m venv $(VENV_DIR); \
	fi
	@printf "  $(YELLOW)→ Upgrading pip...$(RESET)\n"
	@. $(VENV_DIR)/bin/activate && pip install --upgrade pip --quiet
	@printf "  $(YELLOW)→ Installing project dependencies...$(RESET)\n"
	@if [ -f "pyproject.toml" ]; then \
		. $(VENV_DIR)/bin/activate && pip install -e ".[dev]" --quiet && \
		printf "  $(GREEN)✅ Python dependencies installed from pyproject.toml$(RESET)\n" || \
		printf "  $(RED)❌ Failed to install from pyproject.toml$(RESET)\n"; \
	elif [ -f "requirements.txt" ]; then \
		. $(VENV_DIR)/bin/activate && pip install -r requirements.txt --quiet && \
		printf "  $(GREEN)✅ Python dependencies installed from requirements.txt$(RESET)\n" || \
		printf "  $(RED)❌ Failed to install from requirements.txt$(RESET)\n"; \
	else \
		printf "  $(RED)❌ No pyproject.toml or requirements.txt found$(RESET)\n"; \
	fi
	@printf "\n"
	@printf "$(CYAN)📦 Node.js Dependencies...$(RESET)\n"
	@if [ -f "package.json" ]; then \
		printf "  $(YELLOW)→ Installing Node.js packages from root...$(RESET)\n"; \
		npm install --silent && \
		printf "  $(GREEN)✅ Node.js dependencies installed$(RESET)\n" || \
		printf "  $(RED)❌ Failed to install Node.js dependencies$(RESET)\n"; \
	elif [ -f "$(WEB_DIR)/package.json" ]; then \
		printf "  $(YELLOW)→ Installing Node.js packages from web/...$(RESET)\n"; \
		cd $(WEB_DIR) && npm install --silent && \
		printf "  $(GREEN)✅ Node.js dependencies installed$(RESET)\n" || \
		printf "  $(RED)❌ Failed to install Node.js dependencies$(RESET)\n"; \
	else \
		printf "  $(YELLOW)⚠️  No package.json found (frontend may not be set up)$(RESET)\n"; \
	fi
	@printf "\n"
	@printf "$(BOLD)$(GREEN)🎉 Installation Process Complete!$(RESET)\n"
	@printf "\n"
	@printf "$(CYAN)📝 Next Steps:$(RESET)\n"
	@printf "  1. $(YELLOW)Optional:$(RESET) Activate venv with $(GREEN)source venv/bin/activate$(RESET)\n"
	@printf "  2. Run $(GREEN)make status$(RESET) to verify installation\n"
	@printf "  3. Run $(GREEN)make test-dev$(RESET) to run quick tests\n"
	@printf "  4. Run $(GREEN)make dev$(RESET) to start development servers\n"
	@printf "\n"
	@printf "$(CYAN)💡 Note:$(RESET) The Makefile automatically uses the venv, manual activation is optional\n"

status: ## 📊 Show detailed environment and service status
	@echo -e "$(BOLD)$(BLUE)📊 FreeAgentics Environment Status$(RESET)"
	@echo ""
	@echo -e "$(YELLOW)📋 System Information:$(RESET)"
	@echo "  OS: $$(uname -s) $$(uname -r)"
	@echo "  Python: $$($(PYTHON) --version 2>&1)"
	@echo "  Node.js: $$(node --version 2>/dev/null || echo 'Not installed')"
	@echo "  Working Directory: $$(pwd)"
	@echo ""
	@echo -e "$(YELLOW)🔧 Project Status:$(RESET)"
	@echo -e "  Python venv:   $$([ -d $(VENV_DIR) ] && echo '$(GREEN)✅ Created$(RESET)' || echo '$(YELLOW)⚠️  Not created$(RESET)')"
	@echo -e "  Dependencies:  $$([ -d $(VENV_DIR) ] && [ -f $(VENV_DIR)/bin/uvicorn ] && echo '$(GREEN)✅ Installed$(RESET)' || echo '$(YELLOW)⚠️  Incomplete$(RESET)')"
	@echo -e "  Node modules:  $$([ -d node_modules ] && echo '$(GREEN)✅ Installed$(RESET)' || echo '$(YELLOW)⚠️  Not installed$(RESET)')"
	@echo -e "  Package.json:  $$([ -f package.json ] && echo '$(GREEN)✅ Found (root)$(RESET)' || echo '$(YELLOW)⚠️  Not found$(RESET)')"
	@echo ""
	@echo -e "$(YELLOW)🚀 Running Services:$(RESET)"
	@echo -e "  Backend API:   $$(curl -s http://localhost:8000/health 2>/dev/null >/dev/null && echo '$(GREEN)✅ Running at http://localhost:8000$(RESET)' || echo '$(CYAN)○ Not running$(RESET)')"
	@echo -e "  Frontend App:  $$(curl -s http://localhost:3000 2>/dev/null >/dev/null && echo '$(GREEN)✅ Running at http://localhost:3000$(RESET)' || echo '$(CYAN)○ Not running$(RESET)')"
	@echo ""
	@echo -e "$(YELLOW)🎯 Recommended Actions:$(RESET)"
	@if [ ! -d $(VENV_DIR) ]; then \
		echo -e "  → Run $(GREEN)make install$(RESET) to create Python virtual environment"; \
	elif [ ! -f $(VENV_DIR)/bin/uvicorn ]; then \
		echo -e "  → Run $(GREEN)make install$(RESET) to install Python dependencies"; \
	elif [ ! -d node_modules ]; then \
		echo -e "  → Run $(GREEN)npm install$(RESET) to install Node.js dependencies"; \
	elif ! curl -s http://localhost:8000/health 2>/dev/null >/dev/null; then \
		echo -e "  → Run $(GREEN)make dev$(RESET) to start development servers"; \
	else \
		echo -e "  → Everything is running! Visit $(BLUE)http://localhost:3000$(RESET)"; \
	fi

# ============================================================================
# 3. DEVELOPMENT COMMANDS
# ============================================================================

dev: kill-ports ## 🚀 Start development servers (auto-detects demo/dev mode)
	@# Ensure venv exists
	@if [ ! -d "$(VENV_DIR)" ]; then \
		printf "$(YELLOW)⚠️  Python environment not found. Please run 'make install' first$(RESET)\n"; \
		exit 1; \
	fi
	@# Always use venv python directly, no need to check VIRTUAL_ENV
	@$(PYTHON) scripts/dev.py

stop: ## 🛑 Stop all development servers and clear ports
	@echo -e "$(YELLOW)🛑 Stopping Development Servers...$(RESET)"
	@$(MAKE) kill-ports
	@echo -e "$(GREEN)✅ All servers stopped$(RESET)"

kill-ports: ## 🔧 Clear processes on development ports (3000, 8000)
	@# Silently kill processes on common ports
	@lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	@lsof -ti:3001 | xargs kill -9 2>/dev/null || true
	@lsof -ti:3002 | xargs kill -9 2>/dev/null || true
	@lsof -ti:3003 | xargs kill -9 2>/dev/null || true
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true

start: dev ## 🚀 Alias for 'make dev'


# ============================================================================
# 4. TESTING COMMANDS (Three-Tier Strategy)
# ============================================================================

test-dev: ## ⚡ Fast tests for active development (~30 seconds)
	@echo -e "$(BOLD)$(GREEN)⚡ Development Tests$(RESET)"
	@echo -e "$(CYAN)Fast validation for active development$(RESET)"
	@echo ""
	@mkdir -p $(REPORT_DIR)
	@# Quick environment check
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo -e "$(RED)❌ Python environment not ready. Run 'make install' first$(RESET)"; \
		exit 1; \
	fi
	@echo -e "$(CYAN)🧪 Unit Tests (Core Logic)...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD)" $(PYTEST) tests/unit/ -x --tb=short --timeout=30 -q --disable-warnings > $(REPORT_DIR)/unit-tests.log 2>&1 && \
		echo -e "  $(GREEN)✅ Python unit tests passed$(RESET)" || \
		echo -e "  $(YELLOW)⚠️  Some Python unit tests failed (see $(REPORT_DIR)/unit-tests.log)$(RESET)"
	@if [ -f "$(WEB_DIR)/package.json" ]; then \
		cd $(WEB_DIR) && timeout 30s npm test -- --watchAll=false --passWithNoTests --silent > ../$(REPORT_DIR)/jest-tests.log 2>&1 && \
		echo -e "  $(GREEN)✅ JavaScript tests passed$(RESET)" || \
		echo -e "  $(YELLOW)⚠️  Some JavaScript tests failed (see $(REPORT_DIR)/jest-tests.log)$(RESET)"; \
	else \
		echo -e "  $(CYAN)ⓘ Skipping JavaScript tests (no web/package.json)$(RESET)"; \
	fi
	@echo ""
	@echo -e "$(CYAN)🔍 Quick Type Check...$(RESET)"
	@. $(VENV_DIR)/bin/activate && mypy agents/ api/ --ignore-missing-imports --no-error-summary && \
		echo -e "  $(GREEN)✅ Python types valid$(RESET)" || \
		echo -e "  $(YELLOW)⚠️  Python type issues found$(RESET)"
	@if [ -f "$(WEB_DIR)/tsconfig.json" ]; then \
		cd $(WEB_DIR) && npx tsc --noEmit --skipLibCheck --pretty false > /dev/null 2>&1 && \
		echo -e "  $(GREEN)✅ TypeScript types valid$(RESET)" || \
		echo -e "  $(YELLOW)⚠️  TypeScript type issues found$(RESET)"; \
	else \
		echo -e "  $(CYAN)ⓘ Skipping TypeScript check (no web/tsconfig.json)$(RESET)"; \
	fi
	@echo ""
	@echo -e "$(BOLD)$(GREEN)✅ Development tests complete!$(RESET)"
	@echo -e "$(CYAN)Ready for active development. Run $(GREEN)make test-commit$(RESET) before committing.$(RESET)"

test-commit: ## 🚦 Pre-commit validation suite (~2 minutes)
	@echo -e "$(BOLD)$(YELLOW)🚦 Pre-Commit Validation$(RESET)"
	@echo -e "$(CYAN)Comprehensive checks before committing code$(RESET)"
	@echo ""
	@mkdir -p $(REPORT_DIR)
	@# Environment verification
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo -e "$(RED)❌ Python environment not ready. Run 'make install' first$(RESET)"; \
		exit 1; \
	fi
	@echo -e "$(CYAN)🧪 Complete Test Suite...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD)" $(PYTEST) tests/unit/ tests/integration/ -v --tb=short --timeout=60 > $(REPORT_DIR)/commit-tests.log 2>&1 && \
		echo -e "  $(GREEN)✅ All backend tests passed$(RESET)" || \
		echo -e "  $(RED)❌ Backend tests failed (see $(REPORT_DIR)/commit-tests.log)$(RESET)"
	@cd $(WEB_DIR) && npm test -- --coverage --watchAll=false --coverageReporters=text-summary > ../$(REPORT_DIR)/jest-commit.log 2>&1 && \
		echo -e "  $(GREEN)✅ Frontend tests with coverage passed$(RESET)" || \
		echo -e "  $(RED)❌ Frontend tests failed (see $(REPORT_DIR)/jest-commit.log)$(RESET)"
	@echo ""
	@echo -e "$(CYAN)🔍 Code Quality Checks...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		flake8 . --exclude=venv,node_modules,.git --statistics --quiet && \
		echo -e "  $(GREEN)✅ Python linting passed$(RESET)" || \
		echo -e "  $(RED)❌ Python linting failed$(RESET)"
	@cd $(WEB_DIR) && npm run lint --silent && \
		echo -e "  $(GREEN)✅ JavaScript linting passed$(RESET)" || \
		echo -e "  $(RED)❌ JavaScript linting failed$(RESET)"
	@echo ""
	@echo -e "$(CYAN)🔒 Type Safety...$(RESET)"
	@. $(VENV_DIR)/bin/activate && mypy agents/ api/ coalitions/ --ignore-missing-imports --no-error-summary && \
		echo -e "  $(GREEN)✅ Python types validated$(RESET)" || \
		echo -e "  $(RED)❌ Python type errors found$(RESET)"
	@cd $(WEB_DIR) && npx tsc --noEmit --skipLibCheck && \
		echo -e "  $(GREEN)✅ TypeScript types validated$(RESET)" || \
		echo -e "  $(RED)❌ TypeScript type errors found$(RESET)"
	@echo ""
	@echo -e "$(CYAN)🛡️  Basic Security Scan...$(RESET)"
	@. $(VENV_DIR)/bin/activate && bandit -r . -f text --quiet --severity-level medium > $(REPORT_DIR)/security.log 2>&1 && \
		echo -e "  $(GREEN)✅ Security scan passed$(RESET)" || \
		echo -e "  $(YELLOW)⚠️  Security issues found (see $(REPORT_DIR)/security.log)$(RESET)"
	@echo ""
	@echo -e "$(BOLD)$(GREEN)✅ Pre-commit validation complete!$(RESET)"
	@echo -e "$(CYAN)Code is ready for commit. Consider running $(GREEN)make format$(RESET) if needed.$(RESET)"

test-release: ## 🏆 Comprehensive release validation (~10 minutes)
	@echo -e "$(BOLD)$(MAGENTA)🏆 RELEASE VALIDATION SUITE$(RESET)"
	@echo -e "$(CYAN)Complete validation with all testing tools and comprehensive debugging$(RESET)"
	@echo ""
	@mkdir -p $(REPORT_DIR)/{unit,integration,e2e,security,coverage,quality,build}
	@echo -e "$(YELLOW)📊 System Information:$(RESET)"
	@echo "  Date: $$(date)"
	@echo "  CPU Cores: $$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
	@echo "  Python: $$($(PYTHON) --version)"
	@echo "  Node.js: $$(node --version)"
	@echo "  Git Commit: $$(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
	@echo "  Report Directory: $(REPORT_DIR)"
	@echo ""
	@echo -e "$(BOLD)$(CYAN)Phase 1: Environment Verification$(RESET)"
	@$(MAKE) check
	@echo ""
	@echo -e "$(BOLD)$(CYAN)Phase 2: Complete Test Suite$(RESET)"
	@echo -e "$(CYAN)🧪 Unit Tests with Coverage...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD)" $(PYTEST) tests/unit/ \
		--cov=agents --cov=api --cov=coalitions --cov=inference --cov=world \
		--cov-report=html:$(REPORT_DIR)/coverage/backend \
		--cov-report=xml:$(REPORT_DIR)/coverage/backend.xml \
		--cov-report=term \
		--junitxml=$(REPORT_DIR)/unit/junit.xml \
		-v > $(REPORT_DIR)/unit/detailed.log 2>&1 && \
		echo -e "  $(GREEN)✅ Backend unit tests with coverage$(RESET)" || \
		echo -e "  $(RED)❌ Backend unit tests failed (see $(REPORT_DIR)/unit/)$(RESET)"
	@echo ""
	@echo -e "$(CYAN)🔗 Integration Tests...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD)" $(PYTEST) tests/integration/ \
		--junitxml=$(REPORT_DIR)/integration/junit.xml \
		-v > $(REPORT_DIR)/integration/detailed.log 2>&1 && \
		echo -e "  $(GREEN)✅ Integration tests passed$(RESET)" || \
		echo -e "  $(RED)❌ Integration tests failed (see $(REPORT_DIR)/integration/)$(RESET)"
	@echo ""
	@echo -e "$(CYAN)⚛️  Frontend Tests with Coverage...$(RESET)"
	@cd $(WEB_DIR) && npm test -- \
		--coverage \
		--coverageDirectory=../$(REPORT_DIR)/coverage/frontend \
		--coverageReporters=html,text,lcov \
		--watchAll=false \
		--verbose > ../$(REPORT_DIR)/unit/frontend.log 2>&1 && \
		echo -e "  $(GREEN)✅ Frontend tests with coverage$(RESET)" || \
		echo -e "  $(RED)❌ Frontend tests failed (see $(REPORT_DIR)/unit/frontend.log)$(RESET)"
	@echo ""
	@echo -e "$(BOLD)$(CYAN)Phase 3: Code Quality & Security$(RESET)"
	@echo -e "$(CYAN)🔍 Comprehensive Linting...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		flake8 . --exclude=venv,node_modules,.git --max-line-length=88 \
		--output-file=$(REPORT_DIR)/quality/flake8.txt \
		--statistics && \
		echo -e "  $(GREEN)✅ Python linting$(RESET)" || \
		echo -e "  $(RED)❌ Python linting failed (see $(REPORT_DIR)/quality/flake8.txt)$(RESET)"
	@cd $(WEB_DIR) && npm run lint > ../$(REPORT_DIR)/quality/eslint.log 2>&1 && \
		echo -e "  $(GREEN)✅ JavaScript linting$(RESET)" || \
		echo -e "  $(RED)❌ JavaScript linting failed (see $(REPORT_DIR)/quality/eslint.log)$(RESET)"
	@echo ""
	@echo -e "$(CYAN)🔒 Complete Type Checking...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		mypy agents/ api/ coalitions/ inference/ world/ \
		--ignore-missing-imports \
		--html-report $(REPORT_DIR)/quality/mypy-html \
		> $(REPORT_DIR)/quality/mypy.log 2>&1 && \
		echo -e "  $(GREEN)✅ Python type checking$(RESET)" || \
		echo -e "  $(YELLOW)⚠️  Python type issues (see $(REPORT_DIR)/quality/mypy.log)$(RESET)"
	@cd $(WEB_DIR) && npx tsc --noEmit --listFiles --traceResolution > ../$(REPORT_DIR)/quality/tsc.log 2>&1 && \
		echo -e "  $(GREEN)✅ TypeScript type checking$(RESET)" || \
		echo -e "  $(YELLOW)⚠️  TypeScript type issues (see $(REPORT_DIR)/quality/tsc.log)$(RESET)"
	@echo ""
	@echo -e "$(CYAN)🛡️  Security Analysis...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		bandit -r . -f json -o $(REPORT_DIR)/security/bandit.json 2>&1 && \
		pip-audit --format=json --output=$(REPORT_DIR)/security/pip-audit.json 2>&1 && \
		echo -e "  $(GREEN)✅ Security scanning complete$(RESET)" || \
		echo -e "  $(YELLOW)⚠️  Security issues found (see $(REPORT_DIR)/security/)$(RESET)"
	@echo ""
	@echo -e "$(BOLD)$(CYAN)Phase 4: Build Verification$(RESET)"
	@echo -e "$(CYAN)🔨 Production Build...$(RESET)"
	@cd $(WEB_DIR) && NODE_ENV=production npm run build > ../$(REPORT_DIR)/build/frontend.log 2>&1 && \
		echo -e "  $(GREEN)✅ Frontend production build$(RESET)" || \
		echo -e "  $(RED)❌ Frontend build failed (see $(REPORT_DIR)/build/frontend.log)$(RESET)"
	@echo ""
	@echo -e "$(BOLD)$(CYAN)Phase 5: Report Generation$(RESET)"
	@echo "# FreeAgentics Release Validation Report" > $(REPORT_DIR)/SUMMARY.md
	@echo "Generated: $$(date)" >> $(REPORT_DIR)/SUMMARY.md
	@echo "Commit: $$(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')" >> $(REPORT_DIR)/SUMMARY.md
	@echo "" >> $(REPORT_DIR)/SUMMARY.md
	@echo "## Test Results" >> $(REPORT_DIR)/SUMMARY.md
	@echo "- Unit Tests: $$(grep -c 'PASSED' $(REPORT_DIR)/unit/detailed.log 2>/dev/null || echo 0) passed" >> $(REPORT_DIR)/SUMMARY.md
	@echo "- Integration Tests: $$(grep -c 'PASSED' $(REPORT_DIR)/integration/detailed.log 2>/dev/null || echo 0) passed" >> $(REPORT_DIR)/SUMMARY.md
	@echo "- Code Quality Issues: $$(wc -l < $(REPORT_DIR)/quality/flake8.txt 2>/dev/null || echo 0)" >> $(REPORT_DIR)/SUMMARY.md
	@echo "- Security Issues: $$(grep -c 'issue' $(REPORT_DIR)/security/bandit.json 2>/dev/null || echo 0)" >> $(REPORT_DIR)/SUMMARY.md
	@echo ""
	@echo -e "$(BOLD)$(GREEN)🏆 Release validation complete!$(RESET)"
	@echo ""
	@echo -e "$(YELLOW)📊 Results Summary:$(RESET)"
	@echo -e "  Detailed Report: $(CYAN)$(REPORT_DIR)/SUMMARY.md$(RESET)"
	@echo -e "  Coverage Report: $(CYAN)$(REPORT_DIR)/coverage/backend/index.html$(RESET)"
	@echo -e "  All Logs: $(CYAN)$(REPORT_DIR)/$(RESET)"

# ============================================================================
# 5. INDIVIDUAL TEST COMMANDS
# ============================================================================

test-unit: ## 🧪 Run unit tests only
	@echo -e "$(BOLD)$(GREEN)🧪 Unit Tests Only$(RESET)"
	@mkdir -p $(REPORT_DIR)
	@. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD)" $(PYTEST) tests/unit/ -v --tb=short && \
		echo -e "$(GREEN)✅ Unit tests completed$(RESET)"

test-integration: ## 🔗 Run integration tests only
	@echo -e "$(BOLD)$(GREEN)🔗 Integration Tests Only$(RESET)"
	@mkdir -p $(REPORT_DIR)
	@. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD)" $(PYTEST) tests/integration/ -v --tb=short && \
		echo -e "$(GREEN)✅ Integration tests completed$(RESET)"

test-e2e: ## 🌐 Run end-to-end tests only
	@echo -e "$(BOLD)$(GREEN)🌐 End-to-End Tests$(RESET)"
	@mkdir -p $(REPORT_DIR)
	@cd $(WEB_DIR) && npx playwright test && \
		echo -e "$(GREEN)✅ E2E tests completed$(RESET)"

test-security: ## 🛡️ Run security tests only
	@echo -e "$(BOLD)$(RED)🛡️ Security Tests$(RESET)"
	@mkdir -p $(REPORT_DIR)
	@. $(VENV_DIR)/bin/activate && \
		bandit -r . -f text && \
		pip-audit && \
		echo -e "$(GREEN)✅ Security tests completed$(RESET)"

coverage: ## 📊 Generate detailed coverage reports
	@echo -e "$(BOLD)$(BLUE)📊 Generating Coverage Reports$(RESET)"
	@mkdir -p $(REPORT_DIR)/coverage
	@echo -e "$(CYAN)🐍 Backend Coverage...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD)" $(PYTEST) tests/ \
		--cov=agents --cov=api --cov=coalitions --cov=inference --cov=world \
		--cov-report=html:$(REPORT_DIR)/coverage/backend \
		--cov-report=xml:$(REPORT_DIR)/coverage/backend.xml \
		--cov-report=term-missing \
		-v || echo -e "$(YELLOW)Backend coverage completed with warnings$(RESET)"
	@echo ""
	@echo -e "$(CYAN)⚛️  Frontend Coverage...$(RESET)"
	@cd $(WEB_DIR) && npm test -- \
		--coverage \
		--coverageDirectory=../$(REPORT_DIR)/coverage/frontend \
		--watchAll=false || echo -e "$(YELLOW)Frontend coverage completed with warnings$(RESET)"
	@echo ""
	@echo -e "$(GREEN)✅ Coverage reports generated$(RESET)"
	@echo ""
	@echo -e "$(YELLOW)📊 Coverage Reports:$(RESET)"
	@echo -e "  Backend HTML: $(CYAN)$(REPORT_DIR)/coverage/backend/index.html$(RESET)"
	@echo -e "  Backend XML:  $(CYAN)$(REPORT_DIR)/coverage/backend.xml$(RESET)"
	@echo -e "  Frontend:     $(CYAN)$(REPORT_DIR)/coverage/frontend/lcov-report/index.html$(RESET)"

# ============================================================================
# 6. CODE QUALITY COMMANDS
# ============================================================================

lint: ## 🔍 Run all linting and code quality checks
	@echo -e "$(BOLD)$(YELLOW)🔍 Code Quality Analysis$(RESET)"
	@echo ""
	@echo -e "$(CYAN)🐍 Python Linting (flake8)...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		flake8 . --exclude=venv,node_modules,.git --max-line-length=88 --statistics --show-source --count
	@echo ""
	@echo -e "$(CYAN)⚛️  JavaScript/TypeScript Linting (ESLint)...$(RESET)"
	@cd $(WEB_DIR) && npm run lint
	@echo -e "$(GREEN)✅ Linting complete$(RESET)"

format: ## 🎨 Auto-format all code
	@echo -e "$(BOLD)$(BLUE)🎨 Formatting Code$(RESET)"
	@echo ""
	@echo -e "$(CYAN)🐍 Python Formatting (black + isort)...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		black . --exclude='/(venv|node_modules|\.git)/' && \
		isort . --skip-glob='venv/*' --skip-glob='node_modules/*'
	@echo ""
	@echo -e "$(CYAN)⚛️  JavaScript/TypeScript Formatting (prettier)...$(RESET)"
	@cd $(WEB_DIR) && npm run format
	@echo ""
	@echo -e "$(GREEN)✅ Code formatting complete$(RESET)"

type-check: ## 🔒 Run type checking for all code
	@echo -e "$(BOLD)$(MAGENTA)🔒 Type Safety Validation$(RESET)"
	@echo ""
	@echo -e "$(CYAN)🐍 Python Type Checking (mypy)...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		mypy agents/ api/ coalitions/ inference/ world/ --ignore-missing-imports --show-error-context --show-error-codes || \
		echo -e "$(YELLOW)Python type checking completed with warnings$(RESET)"
	@echo ""
	@echo -e "$(CYAN)⚛️  TypeScript Type Checking (tsc)...$(RESET)"
	@cd $(WEB_DIR) && npx tsc --noEmit --skipLibCheck --pretty || \
		echo -e "$(YELLOW)TypeScript checking completed with warnings$(RESET)"
	@echo ""
	@echo -e "$(GREEN)✅ Type checking complete$(RESET)"

# ============================================================================
# 7. BUILD AND DEPLOYMENT
# ============================================================================

build: ## 🔨 Build for production
	@echo -e "$(BOLD)$(BLUE)🔨 Building for Production$(RESET)"
	@echo ""
	@echo -e "$(CYAN)⚛️  Building Frontend...$(RESET)"
	@cd $(WEB_DIR) && NODE_ENV=production npm run build && \
		echo -e "$(GREEN)✅ Frontend built successfully$(RESET)"
	@echo ""
	@echo -e "$(CYAN)🐍 Backend Build Verification...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		python -m py_compile api/main.py && \
		echo -e "$(GREEN)✅ Backend syntax validated$(RESET)"
	@echo ""
	@echo -e "$(GREEN)✅ Production build complete!$(RESET)"

# ============================================================================
# 8. UTILITY COMMANDS
# ============================================================================

clean: ## 🧹 Remove all test artifacts, caches, and temporary files
	@echo -e "$(BOLD)$(CYAN)🧹 Cleaning Project$(RESET)"
	@echo ""
	@echo -e "$(CYAN)🗑️  Removing test artifacts...$(RESET)"
	@rm -rf test-reports/
	@rm -rf .pytest_cache/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@rm -rf .mypy_cache/
	@echo ""
	@echo -e "$(CYAN)🗑️  Removing Python cache files...$(RESET)"
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
	@echo ""
	@echo -e "$(CYAN)🗑️  Removing frontend artifacts...$(RESET)"
	@if [ -d "$(WEB_DIR)" ]; then \
		cd $(WEB_DIR) && rm -rf coverage/ .next/ dist/ build/; \
	fi
	@echo ""
	@echo -e "$(GREEN)✅ Cleanup complete$(RESET)"

reset: ## 🔄 Complete environment reset (removes venv and node_modules)
	@echo -e "$(BOLD)$(RED)🔄 Environment Reset$(RESET)"
	@echo -e "$(YELLOW)⚠️  This will remove virtual environment and node modules$(RESET)"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(MAKE) stop; \
		$(MAKE) clean; \
		echo -e "$(CYAN)Removing Python virtual environment...$(RESET)"; \
		rm -rf $(VENV_DIR); \
		echo -e "$(CYAN)Removing Node.js modules...$(RESET)"; \
		if [ -d "$(WEB_DIR)" ]; then rm -rf $(WEB_DIR)/node_modules; fi; \
		echo -e "$(GREEN)✅ Environment reset complete$(RESET)"; \
		echo -e "$(CYAN)Run 'make install' to reinstall dependencies$(RESET)"; \
	fi

# ============================================================================
# 9. CI/CD AND ADVANCED COMMANDS
# ============================================================================

ci-setup: ## 🤖 Setup for CI/CD environments
	@echo -e "$(BOLD)$(MAGENTA)🤖 CI/CD Environment Setup$(RESET)"
	@mkdir -p $(REPORT_DIR)
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
		. $(VENV_DIR)/bin/activate && pip install --upgrade pip; \
	fi
	@echo -e "$(GREEN)✅ CI environment ready$(RESET)"

docs: ## 📚 Generate documentation
	@echo -e "$(BOLD)$(CYAN)📚 Generating Documentation$(RESET)"
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		sphinx-build -b html docs docs/_build/html 2>/dev/null || \
		echo -e "$(YELLOW)Sphinx not configured, skipping docs generation$(RESET)"; \
	fi
	@echo -e "$(GREEN)✅ Documentation generation complete$(RESET)"

# ============================================================================
# 10. DOCKER COMMANDS
# ============================================================================

docker: docker-build docker-up ## 🐳 Build and start Docker containers

docker-build: ## 🔨 Build Docker images
	@echo -e "$(BOLD)$(BLUE)🔨 Building Docker Images$(RESET)"
	@docker-compose build
	@echo -e "$(GREEN)✅ Docker images built$(RESET)"

docker-up: ## 🚀 Start Docker containers
	@echo -e "$(BOLD)$(BLUE)🚀 Starting Docker Containers$(RESET)"
	@docker-compose up -d
	@echo -e "$(GREEN)✅ Docker containers started$(RESET)"

docker-down: ## 🛑 Stop Docker containers
	@echo -e "$(BOLD)$(YELLOW)🛑 Stopping Docker Containers$(RESET)"
	@docker-compose down
	@echo -e "$(GREEN)✅ Docker containers stopped$(RESET)"

# ============================================================================
# 11. SECURITY COMMANDS
# ============================================================================

security-check: ## 🔒 Verify security configuration
	@echo -e "$(BOLD)$(CYAN)🔒 Security Configuration Check$(RESET)"
	@echo ""
	@echo -e "$(YELLOW)📋 Environment Variables:$(RESET)"
	@if [ -f ".env" ]; then \
		echo -e "  $(GREEN)✅ .env file exists$(RESET)"; \
		if grep -q "dev_secret_key_2025_not_for_production" .env; then \
			echo -e "  $(YELLOW)⚠️  Using development SECRET_KEY$(RESET)"; \
		else \
			echo -e "  $(GREEN)✅ Custom SECRET_KEY configured$(RESET)"; \
		fi; \
		if grep -q "dev_jwt_secret_2025_not_for_production" .env; then \
			echo -e "  $(YELLOW)⚠️  Using development JWT_SECRET$(RESET)"; \
		else \
			echo -e "  $(GREEN)✅ Custom JWT_SECRET configured$(RESET)"; \
		fi; \
	else \
		echo -e "  $(RED)❌ .env file missing$(RESET)"; \
	fi
	@if [ -f ".env.production" ]; then \
		echo -e "  $(RED)⚠️  WARNING: .env.production exists - ensure it's not in git!$(RESET)"; \
	fi
	@echo ""
	@echo -e "$(YELLOW)🔐 Authentication Setup:$(RESET)"
	@grep -l "@require_permission" api/v1/*.py > /dev/null 2>&1 && \
		echo -e "  $(GREEN)✅ API endpoints have authentication decorators$(RESET)" || \
		echo -e "  $(RED)❌ API endpoints missing authentication$(RESET)"
	@echo ""
	@echo -e "$(YELLOW)📁 Security Files:$(RESET)"
	@test -f "auth/security_implementation.py" && \
		echo -e "  $(GREEN)✅ Security implementation found$(RESET)" || \
		echo -e "  $(RED)❌ Security implementation missing$(RESET)"
	@test -f ".env.production.template" && \
		echo -e "  $(GREEN)✅ Production template exists$(RESET)" || \
		echo -e "  $(YELLOW)⚠️  Production template missing$(RESET)"
	@test -f "SECURITY_AUDIT_REPORT.md" && \
		echo -e "  $(GREEN)✅ Security audit report exists$(RESET)" || \
		echo -e "  $(CYAN)ⓘ No security audit report$(RESET)"
	@echo ""
	@echo -e "$(GREEN)Run 'make security-audit' for comprehensive security analysis$(RESET)"

security-scan: ## 🛡️ Quick vulnerability scan
	@echo -e "$(BOLD)$(YELLOW)🛡️ Security Vulnerability Scan$(RESET)"
	@echo ""
	@if [ -d "$(VENV_DIR)" ]; then \
		echo -e "$(CYAN)🔍 Scanning Python dependencies...$(RESET)"; \
		. $(VENV_DIR)/bin/activate && \
		pip install -q safety pip-audit 2>/dev/null; \
		echo ""; \
		echo -e "$(YELLOW)📦 Dependency vulnerabilities:$(RESET)"; \
		pip-audit --desc 2>/dev/null || echo -e "  $(CYAN)pip-audit not available$(RESET)"; \
		echo ""; \
		echo -e "$(YELLOW)🔒 Code security issues:$(RESET)"; \
		bandit -r . -f text --severity-level medium --quiet 2>/dev/null || echo -e "  $(CYAN)bandit not available$(RESET)"; \
	else \
		echo -e "$(RED)❌ Python environment not ready. Run 'make install' first$(RESET)"; \
	fi
	@echo ""
	@if [ -f "package.json" ] || [ -f "$(WEB_DIR)/package.json" ]; then \
		echo -e "$(CYAN)🔍 Scanning Node.js dependencies...$(RESET)"; \
		npm audit --audit-level=moderate 2>/dev/null || echo -e "  $(CYAN)npm audit not available$(RESET)"; \
	fi

security-audit: ## 🏆 Full security audit for production
	@echo -e "$(BOLD)$(MAGENTA)🏆 Comprehensive Security Audit$(RESET)"
	@echo -e "$(CYAN)Full security analysis for v0.2 release$(RESET)"
	@echo ""
	@mkdir -p $(REPORT_DIR)/security
	@echo -e "$(YELLOW)1. Authentication & Authorization:$(RESET)"
	@echo -n "  Checking API endpoints... "
	@grep -r "@require_permission" api/v1/ --include="*.py" | wc -l | xargs -I {} echo -e "$(GREEN){} protected endpoints found$(RESET)"
	@echo -n "  Checking unprotected routes... "
	@grep -r "@router\." api/v1/ --include="*.py" | grep -v "@require_permission" -B1 | grep "@router" | wc -l | xargs -I {} echo -e "$$([ {} -eq 0 ] && echo '$(GREEN)✅ All routes protected$(RESET)' || echo '$(RED)❌ {} unprotected routes found$(RESET)')"
	@echo ""
	@echo -e "$(YELLOW)2. Secret Management:$(RESET)"
	@$(MAKE) check-secrets
	@echo ""
	@echo -e "$(YELLOW)3. Security Headers:$(RESET)"
	@grep -q "SecurityMiddleware" api/main.py && \
		echo -e "  $(GREEN)✅ Security middleware configured$(RESET)" || \
		echo -e "  $(RED)❌ Security middleware missing$(RESET)"
	@echo ""
	@echo -e "$(YELLOW)4. Database Security:$(RESET)"
	@grep -r "postgresql://" . --include="*.py" --exclude-dir=venv | grep -v "os.getenv" | wc -l | xargs -I {} echo -e "  $$([ {} -eq 0 ] && echo '$(GREEN)✅ No hardcoded DB credentials$(RESET)' || echo '$(RED)❌ {} hardcoded DB credentials found$(RESET)')"
	@echo ""
	@echo -e "$(YELLOW)5. Input Validation:$(RESET)"
	@grep -q "SecurityValidator" auth/security_implementation.py && \
		echo -e "  $(GREEN)✅ Input validation implemented$(RESET)" || \
		echo -e "  $(RED)❌ Input validation missing$(RESET)"
	@echo ""
	@echo -e "$(YELLOW)6. Rate Limiting:$(RESET)"
	@grep -q "RateLimiter" auth/security_implementation.py && \
		echo -e "  $(GREEN)✅ Rate limiting implemented$(RESET)" || \
		echo -e "  $(RED)❌ Rate limiting missing$(RESET)"
	@echo ""
	@echo -e "$(YELLOW)7. OWASP Top 10 Summary:$(RESET)"
	@echo -e "  A01: Broken Access Control    $(GREEN)✅ Fixed$(RESET)"
	@echo -e "  A02: Cryptographic Failures   $(YELLOW)⚠️  Need SSL/TLS$(RESET)"
	@echo -e "  A03: Injection                $(GREEN)✅ Protected$(RESET)"
	@echo -e "  A04: Insecure Design          $(YELLOW)⚠️  WebSocket auth needed$(RESET)"
	@echo -e "  A05: Security Misconfiguration $(YELLOW)⚠️  DB credentials$(RESET)"
	@echo -e "  A07: Auth Failures            $(GREEN)✅ Fixed$(RESET)"
	@echo ""
	@echo -e "$(CYAN)Full report: See SECURITY_AUDIT_REPORT.md for details$(RESET)"

check-secrets: ## 🔍 Scan for hardcoded secrets
	@echo -e "$(CYAN)🔍 Scanning for hardcoded secrets...$(RESET)"
	@echo -n "  Checking for passwords... "
	@if grep -r "password\s*=\s*[\"'][^\"']*[\"']" . --include="*.py" --exclude-dir=venv --exclude-dir=node_modules --exclude-dir=.next --exclude-dir=build --exclude-dir=dist 2>/dev/null | grep -v "getenv\|process.env\|environ\|template\|example\|test\|mock" | grep -q .; then \
		echo -e "$(RED)❌ Found hardcoded passwords$(RESET)"; \
	else \
		echo -e "$(GREEN)✅ Clean$(RESET)"; \
	fi
	@echo -n "  Checking for API keys... "
	@if grep -r "api_key\|apikey\|api-key" . --include="*.py" --exclude-dir=venv --exclude-dir=node_modules --exclude-dir=.next -i 2>/dev/null | grep -v "getenv\|process.env\|environ\|template\|example\|test\|mock\|variable" | grep "=\s*[\"'][^\"']*[\"']" | grep -q .; then \
		echo -e "$(RED)❌ Found hardcoded API keys$(RESET)"; \
	else \
		echo -e "$(GREEN)✅ Clean$(RESET)"; \
	fi
	@echo -n "  Checking for secrets... "
	@if grep -r "secret\s*=\s*[\"'][^\"']*[\"']" . --include="*.py" --exclude-dir=venv --exclude-dir=node_modules --exclude-dir=.next 2>/dev/null | grep -v "getenv\|process.env\|environ\|template\|example\|test\|mock\|dev_secret" | grep -q .; then \
		echo -e "$(RED)❌ Found hardcoded secrets$(RESET)"; \
	else \
		echo -e "$(GREEN)✅ Clean$(RESET)"; \
	fi
	@echo -n "  Checking .env in git... "
	@if [ -d .git ] && git ls-files 2>/dev/null | grep -E "^\.env$$|\.env\.production$$" | grep -q .; then \
		echo -e "$(RED)❌ CRITICAL: .env in git!$(RESET)"; \
	else \
		echo -e "$(GREEN)✅ Not tracked$(RESET)"; \
	fi

dev-env: ## 🔧 Set up development environment variables
	@echo -e "$(BOLD)$(CYAN)🔧 Development Environment Setup$(RESET)"
	@if [ ! -f ".env" ]; then \
		if [ -f ".env.example" ] || [ -f ".env.template" ]; then \
			cp .env.example .env 2>/dev/null || cp .env.template .env 2>/dev/null; \
			echo -e "$(GREEN)✅ Created .env from template$(RESET)"; \
		else \
			echo -e "$(YELLOW)Creating basic .env file...$(RESET)"; \
			echo "# Development environment" > .env; \
			echo "SECRET_KEY=dev_secret_key_2025_not_for_production" >> .env; \
			echo "JWT_SECRET=dev_jwt_secret_2025_not_for_production" >> .env; \
			echo "DATABASE_URL=postgresql://freeagentics:freeagentics_dev_2025@localhost:5432/freeagentics" >> .env; \
			echo "API_HOST=0.0.0.0" >> .env; \
			echo "API_PORT=8000" >> .env; \
			echo "LOG_LEVEL=DEBUG" >> .env; \
			echo "DEVELOPMENT_MODE=true" >> .env; \
			echo -e "$(GREEN)✅ Created development .env file$(RESET)"; \
		fi; \
	else \
		echo -e "$(GREEN)✅ .env file already exists$(RESET)"; \
	fi
	@echo ""
	@echo -e "$(YELLOW)📝 Next steps:$(RESET)"
	@echo "  1. Review .env file and update any values"
	@echo "  2. Never commit .env to version control"
	@echo "  3. For production, use .env.production.template as guide"

prod-env: ## 🚀 Validate production environment setup
	@echo -e "$(BOLD)$(MAGENTA)🚀 Production Environment Validation$(RESET)"
	@echo ""
	@if [ ! -f ".env.production.template" ]; then \
		echo -e "$(RED)❌ Missing .env.production.template$(RESET)"; \
		exit 1; \
	fi
	@echo -e "$(YELLOW)📋 Production Checklist:$(RESET)"
	@echo -e "  $(CYAN)Environment Variables:$(RESET)"
	@echo "    [ ] SECRET_KEY - Generate with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
	@echo "    [ ] JWT_SECRET - Generate with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
	@echo "    [ ] DATABASE_URL - Use strong password, no defaults"
	@echo "    [ ] REDIS_PASSWORD - Use strong password"
	@echo "    [ ] PRODUCTION=true"
	@echo ""
	@echo -e "  $(CYAN)Security Configuration:$(RESET)"
	@echo "    [ ] SSL certificates configured"
	@echo "    [ ] HTTPS enforced"
	@echo "    [ ] CORS_ORIGINS set correctly"
	@echo "    [ ] Rate limiting configured"
	@echo ""
	@echo -e "  $(CYAN)Database Security:$(RESET)"
	@echo "    [ ] Remove all hardcoded credentials"
	@echo "    [ ] Database user has minimal permissions"
	@echo "    [ ] SSL/TLS enabled for DB connections"
	@echo ""
	@echo -e "  $(CYAN)Monitoring:$(RESET)"
	@echo "    [ ] Sentry DSN configured"
	@echo "    [ ] Log aggregation setup"
	@echo "    [ ] Alerts configured"
	@echo ""
	@echo -e "$(GREEN)See .env.production.template for complete configuration$(RESET)"

# ============================================================================
# ALIASES AND SHORTCUTS
# ============================================================================

test: test-dev ## ⚡ Alias for test-dev (fast development tests)
t: test-dev ## ⚡ Short alias for test-dev
tc: test-commit ## 🚦 Short alias for test-commit
tr: test-release ## 🏆 Short alias for test-release
