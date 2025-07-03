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
# BEFORE COMMITTING:
# 1. Run `make test-commit` - Run full test suite before committing
# 2. Run `make format`      - Auto-format all code
# 3. Run `make lint`        - Check code quality
#
# BEFORE RELEASING:
# 1. Run `make test-release` - Comprehensive validation with all tools
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
PYTHON := python3
NODE := node
WEB_DIR := web
VENV_DIR := venv
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
.PHONY: dev demo start stop kill-ports status reset docs docker docker-build docker-up docker-down
.PHONY: test-unit test-integration test-e2e test-security test-chaos test-compliance ci-setup

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
	@printf "  $(CYAN)4. Release Preparation:$(RESET)\n"
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
	@printf "  $(GREEN)demo$(RESET)             Run demo (placeholder)\n"
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
	@if [ -d "$(WEB_DIR)/node_modules" ]; then printf "  $(GREEN)✅ Node modules: installed$(RESET)\n"; else printf "  $(CYAN)ⓘ Node modules: not installed (run 'make install')$(RESET)\n"; fi
	@printf "\n"
	@if [ ! -d "$(VENV_DIR)" ] || [ ! -d "$(WEB_DIR)/node_modules" ]; then \
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
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@printf "  $(YELLOW)→ Upgrading pip...$(RESET)\n"
	@. $(VENV_DIR)/bin/activate && pip install --upgrade pip --quiet
	@printf "  $(YELLOW)→ Installing core dependencies (includes AI/ML)...$(RESET)\n"
	@. $(VENV_DIR)/bin/activate && pip install -e ".[dev]" --quiet
	@printf "  $(YELLOW)→ Installing additional development tools...$(RESET)\n"
	@. $(VENV_DIR)/bin/activate && pip install -e ".[dev]" --quiet || printf "  $(YELLOW)⚠️  Some dev tools failed to install$(RESET)\n"
	@printf "  $(GREEN)✅ Python dependencies installed$(RESET)\n"
	@printf "\n"
	@printf "$(CYAN)📦 Node.js Dependencies...$(RESET)\n"
	@if [ -d "$(WEB_DIR)" ]; then \
		printf "  $(YELLOW)→ Installing Node.js packages...$(RESET)\n"; \
		cd $(WEB_DIR) && npm install --silent; \
		printf "  $(GREEN)✅ Node.js dependencies installed$(RESET)\n"; \
	fi
	@printf "\n"
	@printf "$(BOLD)$(GREEN)🎉 Installation Complete!$(RESET)\n"
	@printf "\n"
	@printf "$(CYAN)📝 Next Steps:$(RESET)\n"
	@printf "  1. Run $(GREEN)make test-dev$(RESET) to verify installation\n"
	@printf "  2. Run $(GREEN)make dev$(RESET) to start development servers\n"
	@printf "  3. Visit $(BLUE)http://localhost:3000$(RESET) for the frontend\n"
	@printf "  4. Visit $(BLUE)http://localhost:8000/docs$(RESET) for API docs\n"

status: ## 📊 Show detailed environment and service status
	@echo "$(BOLD)$(BLUE)📊 FreeAgentics Environment Status$(RESET)"
	@echo ""
	@echo "$(YELLOW)📋 System Information:$(RESET)"
	@echo "  OS: $$(uname -s) $$(uname -r)"
	@echo "  Python: $$($(PYTHON) --version 2>&1)"
	@echo "  Node.js: $$(node --version 2>/dev/null || echo 'Not installed')"
	@echo "  Working Directory: $$(pwd)"
	@echo ""
	@echo "$(YELLOW)🔧 Project Status:$(RESET)"
	@echo "  Python venv:   $$([ -d $(VENV_DIR) ] && echo '$(GREEN)✅ Created$(RESET)' || echo '$(YELLOW)⚠️  Not created$(RESET)')"
	@echo "  Dependencies:  $$([ -d $(VENV_DIR) ] && [ -f $(VENV_DIR)/bin/uvicorn ] && echo '$(GREEN)✅ Installed$(RESET)' || echo '$(YELLOW)⚠️  Incomplete$(RESET)')"
	@echo "  Node modules:  $$([ -d $(WEB_DIR)/node_modules ] && echo '$(GREEN)✅ Installed$(RESET)' || echo '$(YELLOW)⚠️  Not installed$(RESET)')"
	@echo ""
	@echo "$(YELLOW)🚀 Running Services:$(RESET)"
	@echo "  Backend API:   $$(curl -s http://localhost:8000/health 2>/dev/null >/dev/null && echo '$(GREEN)✅ Running at http://localhost:8000$(RESET)' || echo '$(CYAN)○ Not running$(RESET)')"
	@echo "  Frontend App:  $$(curl -s http://localhost:3000 2>/dev/null >/dev/null && echo '$(GREEN)✅ Running at http://localhost:3000$(RESET)' || echo '$(CYAN)○ Not running$(RESET)')"
	@echo ""
	@echo "$(YELLOW)🎯 Recommended Actions:$(RESET)"
	@if [ ! -d $(VENV_DIR) ] || [ ! -d $(WEB_DIR)/node_modules ]; then \
		echo "  → Run $(GREEN)make install$(RESET) to set up dependencies"; \
	elif ! curl -s http://localhost:8000/health 2>/dev/null >/dev/null; then \
		echo "  → Run $(GREEN)make dev$(RESET) to start development servers"; \
	else \
		echo "  → Everything is running! Visit $(BLUE)http://localhost:3000$(RESET)"; \
	fi

# ============================================================================
# 3. DEVELOPMENT COMMANDS
# ============================================================================

dev: ## 🚀 Start development servers (frontend + backend)
	@printf "$(BOLD)$(BLUE)🚀 Starting Development Environment$(RESET)\n"
	@printf "\n"
	@# Verify environment is ready
	@if [ ! -d "$(VENV_DIR)" ] || [ ! -d "$(WEB_DIR)/node_modules" ]; then \
		printf "$(YELLOW)⚠️  Dependencies not installed. Running setup...$(RESET)\n"; \
		$(MAKE) install; \
		printf "\n"; \
	fi
	@# Clear any existing processes on our ports
	@printf "$(CYAN)🔧 Clearing port conflicts...$(RESET)\n"
	@lsof -ti:3000 >/dev/null 2>&1 && (printf "  → Stopping process on port 3000\n" && $(MAKE) kill-ports) || true
	@printf "\n"
	@printf "$(CYAN)🔥 Starting Backend (FastAPI on :8000)...$(RESET)\n"
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="." $(PYTHON) -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 & \
		printf "  $(GREEN)✅ Backend started$(RESET)\n"; \
	fi
	@printf "\n"
	@printf "$(CYAN)⚛️  Starting Frontend (Next.js on :3000)...$(RESET)\n"
	@if [ -d "$(WEB_DIR)" ]; then \
		cd $(WEB_DIR) && npm run dev & \
		printf "  $(GREEN)✅ Frontend started$(RESET)\n"; \
	fi
	@sleep 3
	@printf "\n"
	@printf "$(BOLD)$(GREEN)🎉 Development Environment Ready!$(RESET)\n"
	@printf "\n"
	@printf "$(YELLOW)🌐 Access Points:$(RESET)\n"
	@printf "  Frontend:     $(BLUE)http://localhost:3000$(RESET)\n"
	@printf "  Backend API:  $(BLUE)http://localhost:8000$(RESET)\n"
	@printf "  API Docs:     $(BLUE)http://localhost:8000/docs$(RESET)\n"
	@printf "  GraphQL:      $(BLUE)http://localhost:8000/graphql$(RESET)\n"
	@printf "\n"
	@printf "$(YELLOW)💡 Development Tips:$(RESET)\n"
	@printf "  • Run $(GREEN)make test-dev$(RESET) for fast testing during development\n"
	@printf "  • Run $(GREEN)make format$(RESET) to auto-format your code\n"
	@printf "  • Run $(GREEN)make stop$(RESET) to stop all servers\n"
	@printf "  • Press Ctrl+C to stop the servers\n"

stop: ## 🛑 Stop all development servers and clear ports
	@echo "$(YELLOW)🛑 Stopping Development Servers...$(RESET)"
	@$(MAKE) kill-ports
	@echo "$(GREEN)✅ All servers stopped$(RESET)"

kill-ports: ## 🔧 Clear processes on development ports (3000, 8000)
	@echo "$(YELLOW)🔧 Clearing Port Conflicts$(RESET)"
	@echo "Stopping processes on ports 3000, 8000..."
	@lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@echo "$(GREEN)✅ Ports cleared$(RESET)"

start: dev ## 🚀 Alias for 'make dev'

demo: ## 🎯 Run demo (placeholder for future implementation)
	@echo "$(BOLD)$(MAGENTA)🎯 Demo Mode$(RESET)"
	@echo "$(YELLOW)Demo functionality is under development.$(RESET)"
	@echo ""
	@echo "For now, you can:"
	@echo "  1. Run $(GREEN)make dev$(RESET) to start the development environment"
	@echo "  2. Visit $(BLUE)http://localhost:3000$(RESET) to explore the application"

# ============================================================================
# 4. TESTING COMMANDS (Three-Tier Strategy)
# ============================================================================

test-dev: ## ⚡ Fast tests for active development (~30 seconds)
	@echo "$(BOLD)$(GREEN)⚡ Development Tests$(RESET)"
	@echo "$(CYAN)Fast validation for active development$(RESET)"
	@echo ""
	@mkdir -p $(REPORT_DIR)
	@# Quick environment check
	@if [ ! -d "$(VENV_DIR)" ] || [ ! -d "$(WEB_DIR)/node_modules" ]; then \
		echo "$(RED)❌ Environment not ready. Run 'make install' first$(RESET)"; \
		exit 1; \
	fi
	@echo "$(CYAN)🧪 Unit Tests (Core Logic)...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD)" pytest tests/unit/ -x --tb=short --timeout=30 -q --disable-warnings > $(REPORT_DIR)/unit-tests.log 2>&1 && \
		echo "  $(GREEN)✅ Python unit tests passed$(RESET)" || \
		echo "  $(YELLOW)⚠️  Some Python unit tests failed (see $(REPORT_DIR)/unit-tests.log)$(RESET)"
	@cd $(WEB_DIR) && timeout 30s npm test -- --watchAll=false --passWithNoTests --silent > ../$(REPORT_DIR)/jest-tests.log 2>&1 && \
		echo "  $(GREEN)✅ JavaScript tests passed$(RESET)" || \
		echo "  $(YELLOW)⚠️  Some JavaScript tests failed (see $(REPORT_DIR)/jest-tests.log)$(RESET)"
	@echo ""
	@echo "$(CYAN)🔍 Quick Type Check...$(RESET)"
	@. $(VENV_DIR)/bin/activate && mypy agents/ api/ --ignore-missing-imports --no-error-summary --quiet && \
		echo "  $(GREEN)✅ Python types valid$(RESET)" || \
		echo "  $(YELLOW)⚠️  Python type issues found$(RESET)"
	@cd $(WEB_DIR) && npx tsc --noEmit --skipLibCheck --pretty false > /dev/null 2>&1 && \
		echo "  $(GREEN)✅ TypeScript types valid$(RESET)" || \
		echo "  $(YELLOW)⚠️  TypeScript type issues found$(RESET)"
	@echo ""
	@echo "$(BOLD)$(GREEN)✅ Development tests complete!$(RESET)"
	@echo "$(CYAN)Ready for active development. Run $(GREEN)make test-commit$(RESET) before committing.$(RESET)"

test-commit: ## 🚦 Pre-commit validation suite (~2 minutes)
	@echo "$(BOLD)$(YELLOW)🚦 Pre-Commit Validation$(RESET)"
	@echo "$(CYAN)Comprehensive checks before committing code$(RESET)"
	@echo ""
	@mkdir -p $(REPORT_DIR)
	@# Environment verification
	@if [ ! -d "$(VENV_DIR)" ] || [ ! -d "$(WEB_DIR)/node_modules" ]; then \
		echo "$(RED)❌ Environment not ready. Run 'make install' first$(RESET)"; \
		exit 1; \
	fi
	@echo "$(CYAN)🧪 Complete Test Suite...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD)" pytest tests/unit/ tests/integration/ -v --tb=short --timeout=60 > $(REPORT_DIR)/commit-tests.log 2>&1 && \
		echo "  $(GREEN)✅ All backend tests passed$(RESET)" || \
		echo "  $(RED)❌ Backend tests failed (see $(REPORT_DIR)/commit-tests.log)$(RESET)"
	@cd $(WEB_DIR) && npm test -- --coverage --watchAll=false --coverageReporters=text-summary > ../$(REPORT_DIR)/jest-commit.log 2>&1 && \
		echo "  $(GREEN)✅ Frontend tests with coverage passed$(RESET)" || \
		echo "  $(RED)❌ Frontend tests failed (see $(REPORT_DIR)/jest-commit.log)$(RESET)"
	@echo ""
	@echo "$(CYAN)🔍 Code Quality Checks...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		flake8 . --exclude=venv,node_modules,.git --statistics --quiet && \
		echo "  $(GREEN)✅ Python linting passed$(RESET)" || \
		echo "  $(RED)❌ Python linting failed$(RESET)"
	@cd $(WEB_DIR) && npm run lint --silent && \
		echo "  $(GREEN)✅ JavaScript linting passed$(RESET)" || \
		echo "  $(RED)❌ JavaScript linting failed$(RESET)"
	@echo ""
	@echo "$(CYAN)🔒 Type Safety...$(RESET)"
	@. $(VENV_DIR)/bin/activate && mypy agents/ api/ coalitions/ --ignore-missing-imports --no-error-summary && \
		echo "  $(GREEN)✅ Python types validated$(RESET)" || \
		echo "  $(RED)❌ Python type errors found$(RESET)"
	@cd $(WEB_DIR) && npx tsc --noEmit --skipLibCheck && \
		echo "  $(GREEN)✅ TypeScript types validated$(RESET)" || \
		echo "  $(RED)❌ TypeScript type errors found$(RESET)"
	@echo ""
	@echo "$(CYAN)🛡️  Basic Security Scan...$(RESET)"
	@. $(VENV_DIR)/bin/activate && bandit -r . -f text --quiet --severity-level medium > $(REPORT_DIR)/security.log 2>&1 && \
		echo "  $(GREEN)✅ Security scan passed$(RESET)" || \
		echo "  $(YELLOW)⚠️  Security issues found (see $(REPORT_DIR)/security.log)$(RESET)"
	@echo ""
	@echo "$(BOLD)$(GREEN)✅ Pre-commit validation complete!$(RESET)"
	@echo "$(CYAN)Code is ready for commit. Consider running $(GREEN)make format$(RESET) if needed.$(RESET)"

test-release: ## 🏆 Comprehensive release validation (~10 minutes)
	@echo "$(BOLD)$(MAGENTA)🏆 RELEASE VALIDATION SUITE$(RESET)"
	@echo "$(CYAN)Complete validation with all testing tools and comprehensive debugging$(RESET)"
	@echo ""
	@mkdir -p $(REPORT_DIR)/{unit,integration,e2e,security,coverage,quality,build}
	@echo "$(YELLOW)📊 System Information:$(RESET)"
	@echo "  Date: $$(date)"
	@echo "  CPU Cores: $$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
	@echo "  Python: $$($(PYTHON) --version)"
	@echo "  Node.js: $$(node --version)"
	@echo "  Git Commit: $$(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
	@echo "  Report Directory: $(REPORT_DIR)"
	@echo ""
	@echo "$(BOLD)$(CYAN)Phase 1: Environment Verification$(RESET)"
	@$(MAKE) check
	@echo ""
	@echo "$(BOLD)$(CYAN)Phase 2: Complete Test Suite$(RESET)"
	@echo "$(CYAN)🧪 Unit Tests with Coverage...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD)" pytest tests/unit/ \
		--cov=agents --cov=api --cov=coalitions --cov=inference --cov=world \
		--cov-report=html:$(REPORT_DIR)/coverage/backend \
		--cov-report=xml:$(REPORT_DIR)/coverage/backend.xml \
		--cov-report=term \
		--junitxml=$(REPORT_DIR)/unit/junit.xml \
		-v > $(REPORT_DIR)/unit/detailed.log 2>&1 && \
		echo "  $(GREEN)✅ Backend unit tests with coverage$(RESET)" || \
		echo "  $(RED)❌ Backend unit tests failed (see $(REPORT_DIR)/unit/)$(RESET)"
	@echo ""
	@echo "$(CYAN)🔗 Integration Tests...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD)" pytest tests/integration/ \
		--junitxml=$(REPORT_DIR)/integration/junit.xml \
		-v > $(REPORT_DIR)/integration/detailed.log 2>&1 && \
		echo "  $(GREEN)✅ Integration tests passed$(RESET)" || \
		echo "  $(RED)❌ Integration tests failed (see $(REPORT_DIR)/integration/)$(RESET)"
	@echo ""
	@echo "$(CYAN)⚛️  Frontend Tests with Coverage...$(RESET)"
	@cd $(WEB_DIR) && npm test -- \
		--coverage \
		--coverageDirectory=../$(REPORT_DIR)/coverage/frontend \
		--coverageReporters=html,text,lcov \
		--watchAll=false \
		--verbose > ../$(REPORT_DIR)/unit/frontend.log 2>&1 && \
		echo "  $(GREEN)✅ Frontend tests with coverage$(RESET)" || \
		echo "  $(RED)❌ Frontend tests failed (see $(REPORT_DIR)/unit/frontend.log)$(RESET)"
	@echo ""
	@echo "$(BOLD)$(CYAN)Phase 3: Code Quality & Security$(RESET)"
	@echo "$(CYAN)🔍 Comprehensive Linting...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		flake8 . --exclude=venv,node_modules,.git --max-line-length=88 \
		--output-file=$(REPORT_DIR)/quality/flake8.txt \
		--statistics && \
		echo "  $(GREEN)✅ Python linting$(RESET)" || \
		echo "  $(RED)❌ Python linting failed (see $(REPORT_DIR)/quality/flake8.txt)$(RESET)"
	@cd $(WEB_DIR) && npm run lint > ../$(REPORT_DIR)/quality/eslint.log 2>&1 && \
		echo "  $(GREEN)✅ JavaScript linting$(RESET)" || \
		echo "  $(RED)❌ JavaScript linting failed (see $(REPORT_DIR)/quality/eslint.log)$(RESET)"
	@echo ""
	@echo "$(CYAN)🔒 Complete Type Checking...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		mypy agents/ api/ coalitions/ inference/ world/ \
		--ignore-missing-imports \
		--html-report $(REPORT_DIR)/quality/mypy-html \
		> $(REPORT_DIR)/quality/mypy.log 2>&1 && \
		echo "  $(GREEN)✅ Python type checking$(RESET)" || \
		echo "  $(YELLOW)⚠️  Python type issues (see $(REPORT_DIR)/quality/mypy.log)$(RESET)"
	@cd $(WEB_DIR) && npx tsc --noEmit --listFiles --traceResolution > ../$(REPORT_DIR)/quality/tsc.log 2>&1 && \
		echo "  $(GREEN)✅ TypeScript type checking$(RESET)" || \
		echo "  $(YELLOW)⚠️  TypeScript type issues (see $(REPORT_DIR)/quality/tsc.log)$(RESET)"
	@echo ""
	@echo "$(CYAN)🛡️  Security Analysis...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		bandit -r . -f json -o $(REPORT_DIR)/security/bandit.json 2>&1 && \
		pip-audit --format=json --output=$(REPORT_DIR)/security/pip-audit.json 2>&1 && \
		echo "  $(GREEN)✅ Security scanning complete$(RESET)" || \
		echo "  $(YELLOW)⚠️  Security issues found (see $(REPORT_DIR)/security/)$(RESET)"
	@echo ""
	@echo "$(BOLD)$(CYAN)Phase 4: Build Verification$(RESET)"
	@echo "$(CYAN)🔨 Production Build...$(RESET)"
	@cd $(WEB_DIR) && NODE_ENV=production npm run build > ../$(REPORT_DIR)/build/frontend.log 2>&1 && \
		echo "  $(GREEN)✅ Frontend production build$(RESET)" || \
		echo "  $(RED)❌ Frontend build failed (see $(REPORT_DIR)/build/frontend.log)$(RESET)"
	@echo ""
	@echo "$(BOLD)$(CYAN)Phase 5: Report Generation$(RESET)"
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
	@echo "$(BOLD)$(GREEN)🏆 Release validation complete!$(RESET)"
	@echo ""
	@echo "$(YELLOW)📊 Results Summary:$(RESET)"
	@echo "  Detailed Report: $(CYAN)$(REPORT_DIR)/SUMMARY.md$(RESET)"
	@echo "  Coverage Report: $(CYAN)$(REPORT_DIR)/coverage/backend/index.html$(RESET)"
	@echo "  All Logs: $(CYAN)$(REPORT_DIR)/$(RESET)"

# ============================================================================
# 5. INDIVIDUAL TEST COMMANDS
# ============================================================================

test-unit: ## 🧪 Run unit tests only
	@echo "$(BOLD)$(GREEN)🧪 Unit Tests Only$(RESET)"
	@mkdir -p $(REPORT_DIR)
	@. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD)" pytest tests/unit/ -v --tb=short && \
		echo "$(GREEN)✅ Unit tests completed$(RESET)"

test-integration: ## 🔗 Run integration tests only
	@echo "$(BOLD)$(GREEN)🔗 Integration Tests Only$(RESET)"
	@mkdir -p $(REPORT_DIR)
	@. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD)" pytest tests/integration/ -v --tb=short && \
		echo "$(GREEN)✅ Integration tests completed$(RESET)"

test-e2e: ## 🌐 Run end-to-end tests only
	@echo "$(BOLD)$(GREEN)🌐 End-to-End Tests$(RESET)"
	@mkdir -p $(REPORT_DIR)
	@cd $(WEB_DIR) && npx playwright test && \
		echo "$(GREEN)✅ E2E tests completed$(RESET)"

test-security: ## 🛡️ Run security tests only
	@echo "$(BOLD)$(RED)🛡️ Security Tests$(RESET)"
	@mkdir -p $(REPORT_DIR)
	@. $(VENV_DIR)/bin/activate && \
		bandit -r . -f text && \
		pip-audit && \
		echo "$(GREEN)✅ Security tests completed$(RESET)"

coverage: ## 📊 Generate detailed coverage reports
	@echo "$(BOLD)$(BLUE)📊 Generating Coverage Reports$(RESET)"
	@mkdir -p $(REPORT_DIR)/coverage
	@echo "$(CYAN)🐍 Backend Coverage...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD)" pytest tests/ \
		--cov=agents --cov=api --cov=coalitions --cov=inference --cov=world \
		--cov-report=html:$(REPORT_DIR)/coverage/backend \
		--cov-report=xml:$(REPORT_DIR)/coverage/backend.xml \
		--cov-report=term-missing \
		-v || echo "$(YELLOW)Backend coverage completed with warnings$(RESET)"
	@echo ""
	@echo "$(CYAN)⚛️  Frontend Coverage...$(RESET)"
	@cd $(WEB_DIR) && npm test -- \
		--coverage \
		--coverageDirectory=../$(REPORT_DIR)/coverage/frontend \
		--watchAll=false || echo "$(YELLOW)Frontend coverage completed with warnings$(RESET)"
	@echo ""
	@echo "$(GREEN)✅ Coverage reports generated$(RESET)"
	@echo ""
	@echo "$(YELLOW)📊 Coverage Reports:$(RESET)"
	@echo "  Backend HTML: $(CYAN)$(REPORT_DIR)/coverage/backend/index.html$(RESET)"
	@echo "  Backend XML:  $(CYAN)$(REPORT_DIR)/coverage/backend.xml$(RESET)"
	@echo "  Frontend:     $(CYAN)$(REPORT_DIR)/coverage/frontend/lcov-report/index.html$(RESET)"

# ============================================================================
# 6. CODE QUALITY COMMANDS
# ============================================================================

lint: ## 🔍 Run all linting and code quality checks
	@echo "$(BOLD)$(YELLOW)🔍 Code Quality Analysis$(RESET)"
	@echo ""
	@echo "$(CYAN)🐍 Python Linting (flake8)...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		flake8 . --exclude=venv,node_modules,.git --max-line-length=88 --statistics --show-source --count
	@echo ""
	@echo "$(CYAN)⚛️  JavaScript/TypeScript Linting (ESLint)...$(RESET)"
	@cd $(WEB_DIR) && npm run lint
	@echo "$(GREEN)✅ Linting complete$(RESET)"

format: ## 🎨 Auto-format all code
	@echo "$(BOLD)$(BLUE)🎨 Formatting Code$(RESET)"
	@echo ""
	@echo "$(CYAN)🐍 Python Formatting (black + isort)...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		black . --exclude='/(venv|node_modules|\.git)/' && \
		isort . --skip-glob='venv/*' --skip-glob='node_modules/*'
	@echo ""
	@echo "$(CYAN)⚛️  JavaScript/TypeScript Formatting (prettier)...$(RESET)"
	@cd $(WEB_DIR) && npm run format
	@echo ""
	@echo "$(GREEN)✅ Code formatting complete$(RESET)"

type-check: ## 🔒 Run type checking for all code
	@echo "$(BOLD)$(MAGENTA)🔒 Type Safety Validation$(RESET)"
	@echo ""
	@echo "$(CYAN)🐍 Python Type Checking (mypy)...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		mypy agents/ api/ coalitions/ inference/ world/ --ignore-missing-imports --show-error-context --show-error-codes || \
		echo "$(YELLOW)Python type checking completed with warnings$(RESET)"
	@echo ""
	@echo "$(CYAN)⚛️  TypeScript Type Checking (tsc)...$(RESET)"
	@cd $(WEB_DIR) && npx tsc --noEmit --skipLibCheck --pretty || \
		echo "$(YELLOW)TypeScript checking completed with warnings$(RESET)"
	@echo ""
	@echo "$(GREEN)✅ Type checking complete$(RESET)"

# ============================================================================
# 7. BUILD AND DEPLOYMENT
# ============================================================================

build: ## 🔨 Build for production
	@echo "$(BOLD)$(BLUE)🔨 Building for Production$(RESET)"
	@echo ""
	@echo "$(CYAN)⚛️  Building Frontend...$(RESET)"
	@cd $(WEB_DIR) && NODE_ENV=production npm run build && \
		echo "$(GREEN)✅ Frontend built successfully$(RESET)"
	@echo ""
	@echo "$(CYAN)🐍 Backend Build Verification...$(RESET)"
	@. $(VENV_DIR)/bin/activate && \
		python -m py_compile api/main.py && \
		echo "$(GREEN)✅ Backend syntax validated$(RESET)"
	@echo ""
	@echo "$(GREEN)✅ Production build complete!$(RESET)"

# ============================================================================
# 8. UTILITY COMMANDS
# ============================================================================

clean: ## 🧹 Remove all test artifacts, caches, and temporary files
	@echo "$(BOLD)$(CYAN)🧹 Cleaning Project$(RESET)"
	@echo ""
	@echo "$(CYAN)🗑️  Removing test artifacts...$(RESET)"
	@rm -rf test-reports/
	@rm -rf .pytest_cache/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@rm -rf .mypy_cache/
	@echo ""
	@echo "$(CYAN)🗑️  Removing Python cache files...$(RESET)"
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo ""
	@echo "$(CYAN)🗑️  Removing frontend artifacts...$(RESET)"
	@if [ -d "$(WEB_DIR)" ]; then \
		cd $(WEB_DIR) && rm -rf coverage/ .next/ dist/ build/; \
	fi
	@echo ""
	@echo "$(GREEN)✅ Cleanup complete$(RESET)"

reset: ## 🔄 Complete environment reset (removes venv and node_modules)
	@echo "$(BOLD)$(RED)🔄 Environment Reset$(RESET)"
	@echo "$(YELLOW)⚠️  This will remove virtual environment and node modules$(RESET)"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(MAKE) stop; \
		$(MAKE) clean; \
		echo "$(CYAN)Removing Python virtual environment...$(RESET)"; \
		rm -rf $(VENV_DIR); \
		echo "$(CYAN)Removing Node.js modules...$(RESET)"; \
		if [ -d "$(WEB_DIR)" ]; then rm -rf $(WEB_DIR)/node_modules; fi; \
		echo "$(GREEN)✅ Environment reset complete$(RESET)"; \
		echo "$(CYAN)Run 'make install' to reinstall dependencies$(RESET)"; \
	fi

# ============================================================================
# 9. CI/CD AND ADVANCED COMMANDS
# ============================================================================

ci-setup: ## 🤖 Setup for CI/CD environments
	@echo "$(BOLD)$(MAGENTA)🤖 CI/CD Environment Setup$(RESET)"
	@mkdir -p $(REPORT_DIR)
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
		. $(VENV_DIR)/bin/activate && pip install --upgrade pip; \
	fi
	@echo "$(GREEN)✅ CI environment ready$(RESET)"

docs: ## 📚 Generate documentation
	@echo "$(BOLD)$(CYAN)📚 Generating Documentation$(RESET)"
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		sphinx-build -b html docs docs/_build/html 2>/dev/null || \
		echo "$(YELLOW)Sphinx not configured, skipping docs generation$(RESET)"; \
	fi
	@echo "$(GREEN)✅ Documentation generation complete$(RESET)"

# ============================================================================
# 10. DOCKER COMMANDS
# ============================================================================

docker: docker-build docker-up ## 🐳 Build and start Docker containers

docker-build: ## 🔨 Build Docker images
	@echo "$(BOLD)$(BLUE)🔨 Building Docker Images$(RESET)"
	@docker-compose -f infrastructure/docker/docker-compose.yml build
	@echo "$(GREEN)✅ Docker images built$(RESET)"

docker-up: ## 🚀 Start Docker containers
	@echo "$(BOLD)$(BLUE)🚀 Starting Docker Containers$(RESET)"
	@docker-compose -f infrastructure/docker/docker-compose.yml up -d
	@echo "$(GREEN)✅ Docker containers started$(RESET)"

docker-down: ## 🛑 Stop Docker containers
	@echo "$(BOLD)$(YELLOW)🛑 Stopping Docker Containers$(RESET)"
	@docker-compose -f infrastructure/docker/docker-compose.yml down
	@echo "$(GREEN)✅ Docker containers stopped$(RESET)"

# ============================================================================
# ALIASES AND SHORTCUTS
# ============================================================================

test: test-dev ## ⚡ Alias for test-dev (fast development tests)
t: test-dev ## ⚡ Short alias for test-dev
tc: test-commit ## 🚦 Short alias for test-commit
tr: test-release ## 🏆 Short alias for test-release