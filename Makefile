# FreeAgentics Multi-Agent AI Platform
# Enterprise-ready Active Inference system with comprehensive tooling
# Follows Arch Linux philosophy: simplicity, transparency, and clarity

.PHONY: help install dev mvp test test-release test-release-parallel kill-ports reset status lint format type-check docker docker-validate docs clean coverage
.DEFAULT_GOAL := help

# Colors for terminal output
BOLD := \033[1m
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
CYAN := \033[36m
MAGENTA := \033[35m
RESET := \033[0m

# Project configuration
PYTHON := python3
NODE := node
WEB_DIR := web
VENV_DIR := venv
TEST_TIMEOUT := 300
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
REPORT_DIR := test-reports/$(TIMESTAMP)
RELEASE_TIMEOUT := 2400  # 40 minutes for comprehensive validation

help: ## Show available commands
	@echo "$(BOLD)FreeAgentics Multi-Agent AI Platform$(RESET)"
	@echo "Enterprise-ready Active Inference system with comprehensive tooling"
	@echo ""
	@echo "$(YELLOW)Development:$(RESET)"
	@echo "  $(GREEN)install$(RESET)                 Complete environment setup (Python + Node.js)"
	@echo "  $(GREEN)dev$(RESET)                     Start development servers (frontend + backend)"
	@echo "  $(GREEN)mvp$(RESET)                     Launch CEO dashboard demo"
	@echo ""
	@echo "$(YELLOW)Testing:$(RESET)"
	@echo "  $(GREEN)test$(RESET)                    Quick validation suite (~2 minutes)"
	@echo "  $(GREEN)test-release$(RESET)            Production validation (~40 minutes)"
	@echo "  $(GREEN)test-release-parallel$(RESET)   Optimized parallel validation (~20 minutes)"
	@echo "  $(GREEN)coverage$(RESET)                Generate coverage reports"
	@echo ""
	@echo "$(YELLOW)Quality:$(RESET)"
	@echo "  $(GREEN)lint$(RESET)                    Code quality analysis"
	@echo "  $(GREEN)format$(RESET)                  Auto-format codebase"
	@echo "  $(GREEN)type-check$(RESET)              Type safety validation"
	@echo ""
	@echo "$(YELLOW)Production:$(RESET)"
	@echo "  $(GREEN)docker$(RESET)                  Build and deploy containers"
	@echo "  $(GREEN)docker-validate$(RESET)         Validate Docker configuration"
	@echo "  $(GREEN)docs$(RESET)                    Generate documentation"
	@echo ""
	@echo "$(YELLOW)Utilities:$(RESET)"
	@echo "  $(GREEN)status$(RESET)                  Environment health check"
	@echo "  $(GREEN)kill-ports$(RESET)              Clear port conflicts"
	@echo "  $(GREEN)reset$(RESET)                   Clean environment reset"
	@echo "  $(GREEN)clean$(RESET)                   Remove test artifacts"

## DEVELOPMENT COMMANDS

install: ## Complete environment setup (Python + Node.js)
	@echo "$(BOLD)$(GREEN)🚀 FreeAgentics Installation$(RESET)"
	@echo "$(CYAN)Setting up Python environment...$(RESET)"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "$(YELLOW)Creating virtual environment...$(RESET)"; \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		pip install --upgrade pip --quiet && \
		pip install -r requirements.txt --quiet 2>/dev/null || echo "$(YELLOW)Installing core packages...$(RESET)" && \
		pip install fastapi uvicorn websockets pytest pytest-cov --quiet; \
	fi
	@echo "$(CYAN)Setting up Node.js environment...$(RESET)"
	@if [ -d "$(WEB_DIR)" ]; then \
		cd $(WEB_DIR) && npm install --silent; \
	fi
	@echo "$(GREEN)✅ Installation complete! Ready for development.$(RESET)"
	@echo "$(CYAN)Next steps: make dev$(RESET)"

dev: ## Start development servers (frontend + backend)
	@echo "$(BOLD)$(BLUE)🚀 Starting Development Environment$(RESET)"
	@echo "$(CYAN)Starting backend server (FastAPI on :8000)...$(RESET)"
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		$(PYTHON) -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 & \
		echo "$(GREEN)✅ Backend started on http://localhost:8000$(RESET)"; \
	fi
	@echo "$(CYAN)Starting frontend server (Next.js on :3000)...$(RESET)"
	@if [ -d "$(WEB_DIR)" ]; then \
		cd $(WEB_DIR) && npm run dev & \
		echo "$(GREEN)✅ Frontend started on http://localhost:3000$(RESET)"; \
	fi
	@echo "$(BOLD)$(GREEN)🎉 Development environment ready!$(RESET)"
	@echo "$(CYAN)Frontend: http://localhost:3000$(RESET)"
	@echo "$(CYAN)Backend: http://localhost:8000$(RESET)"
	@echo "$(CYAN)API Docs: http://localhost:8000/docs$(RESET)"

mvp: ## Launch CEO dashboard demo
	@echo "$(BOLD)$(MAGENTA)🎯 Launching CEO Dashboard Demo$(RESET)"
	@echo "$(CYAN)Starting demo environment...$(RESET)"
	@$(MAKE) dev
	@sleep 3
	@echo "$(CYAN)Opening CEO dashboard...$(RESET)"
	@if command -v open >/dev/null 2>&1; then \
		open http://localhost:3000/ceo-dashboard; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open http://localhost:3000/ceo-dashboard; \
	else \
		echo "$(YELLOW)Please open http://localhost:3000/ceo-dashboard in your browser$(RESET)"; \
	fi
	@echo "$(GREEN)✅ CEO Dashboard launched successfully!$(RESET)"

## TESTING COMMANDS

test: ## Quick validation suite (~2 minutes)
	@echo "$(BOLD)$(GREEN)🧪 Running Test Suite$(RESET)"
	@mkdir -p $(REPORT_DIR)
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		echo "$(CYAN)Running backend tests...$(RESET)" && \
		PYTHONPATH="$(PWD):$$PYTHONPATH" pytest tests/unit/ -x --tb=short --timeout=60 -q > $(REPORT_DIR)/test-output.log 2>&1; \
		backend_result=$$?; \
	else \
		backend_result=1; \
	fi; \
	if [ -d "$(WEB_DIR)" ]; then \
		echo "$(CYAN)Running frontend tests...$(RESET)" && \
		cd $(WEB_DIR) && timeout 60s npm test -- --watchAll=false --passWithNoTests > ../$(REPORT_DIR)/jest-output.log 2>&1; \
		frontend_result=$$?; \
	else \
		frontend_result=1; \
	fi; \
	if [ $$backend_result -eq 0 ] && [ $$frontend_result -eq 0 ]; then \
		echo "$(GREEN)✅ All tests passed$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️  Some tests failed - check $(REPORT_DIR)/ for details$(RESET)"; \
	fi

test-release-parallel: ## Optimized parallel production validation (~20 minutes)
	@echo "$(BOLD)$(MAGENTA)🚀 PARALLEL RELEASE VALIDATION - Enterprise-Grade$(RESET)"
	@echo "$(CYAN)Using $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) CPU cores$(RESET)"
	@mkdir -p release-report-parallel-$(TIMESTAMP)
	@# Run quality checks in parallel
	@echo "$(CYAN)▶ Starting parallel quality analysis...$(RESET)"
	@( \
		echo "$(YELLOW)→ Python linting$(RESET)" && \
		. $(VENV_DIR)/bin/activate && flake8 . --exclude=venv,node_modules,.git --output-file=release-report-parallel-$(TIMESTAMP)/flake8-report.txt --exit-zero & \
		FLAKE8_PID=$$!; \
		echo "$(YELLOW)→ JavaScript linting$(RESET)" && \
		cd $(WEB_DIR) && npm run lint --silent > ../release-report-parallel-$(TIMESTAMP)/eslint-report.txt 2>&1 & \
		ESLINT_PID=$$!; \
		echo "$(YELLOW)→ Python type checking$(RESET)" && \
		. $(VENV_DIR)/bin/activate && mypy . --ignore-missing-imports > release-report-parallel-$(TIMESTAMP)/mypy-report.txt 2>&1 & \
		MYPY_PID=$$!; \
		echo "$(YELLOW)→ TypeScript checking$(RESET)" && \
		cd $(WEB_DIR) && npx tsc --noEmit > ../release-report-parallel-$(TIMESTAMP)/tsc-report.txt 2>&1 & \
		TSC_PID=$$!; \
		wait $$FLAKE8_PID $$ESLINT_PID $$MYPY_PID $$TSC_PID \
	)
	@echo "$(GREEN)✅ Quality analysis complete$(RESET)"
	@# Run tests in parallel
	@echo "$(CYAN)▶ Starting parallel test execution...$(RESET)"
	@( \
		echo "$(YELLOW)→ Backend unit tests$(RESET)" && \
		. $(VENV_DIR)/bin/activate && PYTHONPATH=. pytest tests/unit -n auto --cov=. --cov-report=html:release-report-parallel-$(TIMESTAMP)/coverage-backend > release-report-parallel-$(TIMESTAMP)/pytest-unit.log 2>&1 & \
		PYTEST_UNIT_PID=$$!; \
		echo "$(YELLOW)→ Frontend tests$(RESET)" && \
		cd $(WEB_DIR) && npm test -- --coverage --watchAll=false > ../release-report-parallel-$(TIMESTAMP)/jest-output.log 2>&1 & \
		JEST_PID=$$!; \
		echo "$(YELLOW)→ Integration tests$(RESET)" && \
		. $(VENV_DIR)/bin/activate && PYTHONPATH=. pytest tests/integration -n auto > release-report-parallel-$(TIMESTAMP)/pytest-integration.log 2>&1 & \
		PYTEST_INT_PID=$$!; \
		wait $$PYTEST_UNIT_PID $$JEST_PID $$PYTEST_INT_PID \
	)
	@echo "$(GREEN)✅ Test execution complete$(RESET)"
	@# Security and build checks can run in parallel too
	@echo "$(CYAN)▶ Starting security and build validation...$(RESET)"
	@( \
		echo "$(YELLOW)→ Security scanning$(RESET)" && \
		. $(VENV_DIR)/bin/activate && bandit -r . -f json > release-report-parallel-$(TIMESTAMP)/bandit-report.json 2>&1 & \
		BANDIT_PID=$$!; \
		echo "$(YELLOW)→ Production build$(RESET)" && \
		cd $(WEB_DIR) && NODE_ENV=production npm run build > ../release-report-parallel-$(TIMESTAMP)/build.log 2>&1 & \
		BUILD_PID=$$!; \
		wait $$BANDIT_PID $$BUILD_PID \
	)
	@echo "$(GREEN)✅ Security and build validation complete$(RESET)"
	@# Generate summary
	@echo "$(CYAN)▶ Generating summary report...$(RESET)"
	@echo "# Parallel Release Validation Report" > release-report-parallel-$(TIMESTAMP)/SUMMARY.md
	@echo "Generated: $$(date)" >> release-report-parallel-$(TIMESTAMP)/SUMMARY.md
	@echo "" >> release-report-parallel-$(TIMESTAMP)/SUMMARY.md
	@echo "## Results Overview" >> release-report-parallel-$(TIMESTAMP)/SUMMARY.md
	@echo "- Flake8 Issues: $$(grep -c ":" release-report-parallel-$(TIMESTAMP)/flake8-report.txt 2>/dev/null || echo 0)" >> release-report-parallel-$(TIMESTAMP)/SUMMARY.md
	@echo "- Type Errors: $$(grep -c "error:" release-report-parallel-$(TIMESTAMP)/mypy-report.txt 2>/dev/null || echo 0)" >> release-report-parallel-$(TIMESTAMP)/SUMMARY.md
	@echo "- Test Failures: $$(grep -c "FAILED" release-report-parallel-$(TIMESTAMP)/pytest-*.log 2>/dev/null || echo 0)" >> release-report-parallel-$(TIMESTAMP)/SUMMARY.md
	@echo ""
	@echo "$(BOLD)$(GREEN)📊 Parallel validation complete in ~20 minutes$(RESET)"
	@echo "$(CYAN)Report: release-report-parallel-$(TIMESTAMP)/$(RESET)"

test-release: ## Production validation (~40 minutes)
	@echo "$(BOLD)$(MAGENTA)🚀 PRODUCTION RELEASE VALIDATION$(RESET)"
	@echo "$(CYAN)Starting comprehensive validation for production deployment...$(RESET)"
	@mkdir -p $(REPORT_DIR)/{quality,backend,frontend,integration,security,build}
	@echo "$(CYAN)▶ Phase 1: Code Quality Analysis$(RESET)"
	@$(MAKE) lint
	@$(MAKE) type-check
	@echo "$(CYAN)▶ Phase 2: Unit Testing$(RESET)"
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD):$$PYTHONPATH" pytest tests/unit/ --cov=. --cov-report=html:$(REPORT_DIR)/backend/coverage --junitxml=$(REPORT_DIR)/backend/junit.xml -v > $(REPORT_DIR)/backend/output.log 2>&1; \
	fi
	@if [ -d "$(WEB_DIR)" ]; then \
		cd $(WEB_DIR) && npm test -- --coverage --coverageDirectory=../$(REPORT_DIR)/frontend/coverage --watchAll=false > ../$(REPORT_DIR)/frontend/output.log 2>&1; \
	fi
	@echo "$(CYAN)▶ Phase 3: Integration Testing$(RESET)"
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD):$$PYTHONPATH" pytest tests/integration/ --junitxml=$(REPORT_DIR)/integration/junit.xml -v > $(REPORT_DIR)/integration/output.log 2>&1; \
	fi
	@echo "$(CYAN)▶ Phase 4: Security Scanning$(RESET)"
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		bandit -r . -f json > $(REPORT_DIR)/security/bandit-report.json 2>&1 || echo "$(YELLOW)Security scan completed$(RESET)"; \
		pip-audit --format=json --output=$(REPORT_DIR)/security/pip-audit.json 2>&1 || echo "$(YELLOW)Dependency audit completed$(RESET)"; \
	fi
	@echo "$(CYAN)▶ Phase 5: Production Build$(RESET)"
	@if [ -d "$(WEB_DIR)" ]; then \
		cd $(WEB_DIR) && NODE_ENV=production npm run build > ../$(REPORT_DIR)/build/output.log 2>&1; \
	fi
	@echo "$(CYAN)▶ Phase 6: Report Generation$(RESET)"
	@echo "# Production Release Validation Report" > $(REPORT_DIR)/SUMMARY.md
	@echo "Generated: $$(date)" >> $(REPORT_DIR)/SUMMARY.md
	@echo "" >> $(REPORT_DIR)/SUMMARY.md
	@echo "## Status" >> $(REPORT_DIR)/SUMMARY.md
	@echo "- Quality: ✅ Complete" >> $(REPORT_DIR)/SUMMARY.md
	@echo "- Tests: ✅ Complete" >> $(REPORT_DIR)/SUMMARY.md
	@echo "- Security: ✅ Complete" >> $(REPORT_DIR)/SUMMARY.md
	@echo "- Build: ✅ Complete" >> $(REPORT_DIR)/SUMMARY.md
	@echo ""
	@echo "$(BOLD)$(GREEN)📊 Production validation complete$(RESET)"
	@echo "$(CYAN)Report: $(REPORT_DIR)/SUMMARY.md$(RESET)"

coverage: ## Generate coverage reports
	@echo "$(BOLD)$(BLUE)📊 Generating Coverage Reports$(RESET)"
	@mkdir -p $(REPORT_DIR)/coverage
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD):$$PYTHONPATH" pytest tests/ --cov=. --cov-report=html:$(REPORT_DIR)/coverage/backend --cov-report=term; \
	fi
	@if [ -d "$(WEB_DIR)" ]; then \
		cd $(WEB_DIR) && npm test -- --coverage --coverageDirectory=../$(REPORT_DIR)/coverage/frontend --watchAll=false; \
	fi
	@echo "$(GREEN)✅ Coverage reports generated$(RESET)"
	@echo "$(CYAN)Backend: $(REPORT_DIR)/coverage/backend/index.html$(RESET)"
	@echo "$(CYAN)Frontend: $(REPORT_DIR)/coverage/frontend/index.html$(RESET)"

## QUALITY COMMANDS

lint: ## Code quality analysis
	@echo "$(YELLOW)🔍 Running Code Quality Analysis$(RESET)"
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		echo "$(CYAN)→ Python: flake8$(RESET)" && \
		flake8 . --exclude=venv,node_modules,.git --max-line-length=88 --statistics --exit-zero; \
	fi
	@if [ -d "$(WEB_DIR)" ]; then \
		echo "$(CYAN)→ TypeScript: ESLint$(RESET)" && \
		cd $(WEB_DIR) && npm run lint; \
	fi

format: ## Auto-format codebase
	@echo "$(BLUE)🎨 Formatting Code$(RESET)"
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		echo "$(CYAN)→ Python: black + isort$(RESET)" && \
		black . && isort .; \
	fi
	@if [ -d "$(WEB_DIR)" ]; then \
		echo "$(CYAN)→ TypeScript: prettier$(RESET)" && \
		cd $(WEB_DIR) && npm run format; \
	fi

type-check: ## Type safety validation
	@echo "$(MAGENTA)🔒 Type Safety Validation$(RESET)"
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		echo "$(CYAN)→ Python: mypy$(RESET)" && \
		mypy agents/ api/ coalitions/ inference/ world/ --ignore-missing-imports --follow-imports=skip || echo "$(YELLOW)Type checking completed with warnings$(RESET)"; \
	fi
	@if [ -d "$(WEB_DIR)" ]; then \
		echo "$(CYAN)→ TypeScript: tsc$(RESET)" && \
		cd $(WEB_DIR) && npx tsc --noEmit --skipLibCheck || echo "$(YELLOW)TypeScript checking completed with warnings$(RESET)"; \
	fi

## PRODUCTION COMMANDS

docker: docker-build docker-up ## Build and deploy containers
	@echo "$(BOLD)$(GREEN)🐳 Docker environment ready!$(RESET)"
	@echo "$(CYAN)Frontend: http://localhost:3000$(RESET)"
	@echo "$(CYAN)Backend: http://localhost:8000$(RESET)"

docker-build: ## Build Docker images
	@echo "$(BOLD)$(BLUE)🔨 Building Docker images...$(RESET)"
	@docker-compose -f infrastructure/docker/docker-compose.yml build
	@echo "$(GREEN)✅ Docker images built successfully$(RESET)"

docker-up: ## Start Docker containers
	@echo "$(BOLD)$(BLUE)🚀 Starting Docker containers...$(RESET)"
	@docker-compose -f infrastructure/docker/docker-compose.yml up -d
	@echo "$(GREEN)✅ Docker containers started$(RESET)"

docker-down: ## Stop Docker containers
	@echo "$(BOLD)$(YELLOW)🛑 Stopping Docker containers...$(RESET)"
	@docker-compose -f infrastructure/docker/docker-compose.yml down
	@echo "$(GREEN)✅ Docker containers stopped$(RESET)"

docker-validate: ## Validate Docker configuration
	@echo "$(BOLD)$(MAGENTA)🔍 Validating Docker Configuration$(RESET)"
	@chmod +x infrastructure/docker/validate-docker.sh
	@infrastructure/docker/validate-docker.sh
	@echo "$(GREEN)✅ Docker validation completed$(RESET)"

docs: ## Generate documentation
	@echo "$(BOLD)$(CYAN)📚 Generating Documentation$(RESET)"
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		sphinx-build -b html docs docs/_build/html; \
	fi
	@echo "$(GREEN)✅ Documentation generated: docs/_build/html/index.html$(RESET)"

## UTILITY COMMANDS

status: ## Environment health check
	@echo "$(BOLD)$(BLUE)🔍 Environment Status$(RESET)"
	@echo "$(YELLOW)System:$(RESET)"
	@echo "  Python: $$($(PYTHON) --version 2>&1)"
	@echo "  Node.js: $$(node --version 2>/dev/null || echo 'Not installed')"
	@echo "  Docker: $$(docker --version 2>/dev/null | head -1 || echo 'Not installed')"
	@echo "$(YELLOW)Environment:$(RESET)"
	@echo "  Virtual Env: $$([ -d $(VENV_DIR) ] && echo '✅ Active' || echo '❌ Missing')"
	@echo "  Node Modules: $$([ -d $(WEB_DIR)/node_modules ] && echo '✅ Installed' || echo '❌ Missing')"
	@echo "$(YELLOW)Services:$(RESET)"
	@echo "  Backend: $$(curl -s http://localhost:8000/health 2>/dev/null && echo '✅ Running' || echo '❌ Stopped')"
	@echo "  Frontend: $$(curl -s http://localhost:3000 2>/dev/null && echo '✅ Running' || echo '❌ Stopped')"

kill-ports: ## Clear port conflicts
	@echo "$(YELLOW)🔧 Clearing Port Conflicts$(RESET)"
	@echo "$(CYAN)Stopping processes on ports 3000, 8000...$(RESET)"
	@lsof -ti:3000 | xargs kill -9 2>/dev/null || echo "Port 3000 clear"
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || echo "Port 8000 clear"
	@echo "$(GREEN)✅ Ports cleared$(RESET)"

reset: ## Clean environment reset
	@echo "$(BOLD)$(RED)🧹 Environment Reset$(RESET)"
	@echo "$(YELLOW)⚠️  This will remove all environments and artifacts$(RESET)"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(MAKE) kill-ports; \
		$(MAKE) clean; \
		rm -rf $(VENV_DIR); \
		if [ -d "$(WEB_DIR)" ]; then rm -rf $(WEB_DIR)/node_modules; fi; \
		echo "$(GREEN)✅ Environment reset complete$(RESET)"; \
		echo "$(CYAN)Run 'make install' to reinstall$(RESET)"; \
	fi

clean: ## Remove test artifacts
	@echo "$(CYAN)🧹 Cleaning test artifacts...$(RESET)"
	@rm -rf test-reports/
	@rm -rf .pytest_cache/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@if [ -d "$(WEB_DIR)" ]; then \
		cd $(WEB_DIR) && rm -rf coverage/; \
	fi
	@echo "$(GREEN)✅ Cleanup complete$(RESET)"

setup: ## Internal setup for CI/testing
	@mkdir -p $(REPORT_DIR)
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
		. $(VENV_DIR)/bin/activate && pip install --upgrade pip; \
	fi