# FreeAgentics Makefile - Simple and Direct

.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
SHELL := bash
.DEFAULT_GOAL := help

# Configuration
VENV_DIR := venv
PYTHON := $(VENV_DIR)/bin/python
PYTEST := $(VENV_DIR)/bin/pytest
WEB_DIR := web

.PHONY: help install dev test stop kill-ports status clean reset

help: ## Show available commands
	@echo "FreeAgentics - Quick Start:"
	@echo ""
	@echo "  make install    Install all dependencies"
	@echo "  make dev        Start development servers"
	@echo "  make test       Run tests"
	@echo "  make stop       Stop all servers"
	@echo "  make status     Check environment status"
	@echo "  make clean      Clean build artifacts"
	@echo "  make reset      Full reset (removes dependencies)"
	@echo ""
	@echo "Run 'make install' then 'make dev' to get started."


install: ## Install all dependencies
	@echo "Installing dependencies..."
	@# Create Python virtual environment
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating Python virtual environment..."; \
		python3 -m venv $(VENV_DIR); \
	fi
	@# Install Python dependencies
	@echo "Installing Python packages..."
	@. $(VENV_DIR)/bin/activate && pip install --upgrade pip --quiet
	@if [ -f "requirements.txt" ]; then \
		. $(VENV_DIR)/bin/activate && pip install -r requirements.txt --quiet; \
	elif [ -f "pyproject.toml" ]; then \
		. $(VENV_DIR)/bin/activate && pip install -e ".[dev]" --quiet; \
	fi
	@# Install Node.js dependencies
	@echo "Installing Node.js packages..."
	@if [ -f "$(WEB_DIR)/package.json" ]; then \
		cd $(WEB_DIR) && npm install --silent; \
	elif [ -f "package.json" ]; then \
		npm install --silent; \
	fi
	@# Copy environment file if needed
	@if [ ! -f ".env" ] && [ -f ".env.development" ]; then \
		echo "Creating .env from template..."; \
		cp .env.development .env; \
	fi
	@echo ""
	@echo "✅ Installation complete! Run 'make dev' to start."

status: ## Check environment status
	@echo "Environment Status:"
	@echo ""
	@echo "System:"
	@echo "  Python: $$(python3 --version 2>&1 || echo 'Not found')"
	@echo "  Node.js: $$(node --version 2>/dev/null || echo 'Not found')"
	@echo ""
	@echo "Dependencies:"
	@echo "  Python venv: $$([ -d $(VENV_DIR) ] && echo '✅ Installed' || echo '❌ Not installed')"
	@echo "  Node modules: $$([ -d $(WEB_DIR)/node_modules ] && echo '✅ Installed' || echo '❌ Not installed')"
	@echo "  Environment: $$([ -f .env ] && echo '✅ Configured' || echo '❌ Not configured')"
	@echo ""
	@echo "Services:"
	@echo "  Backend: $$(curl -s http://localhost:8000/health 2>/dev/null >/dev/null && echo '✅ Running (http://localhost:8000)' || echo '⭕ Not running')"
	@echo "  Frontend: $$(curl -s http://localhost:3000 2>/dev/null >/dev/null && echo '✅ Running (http://localhost:3000)' || echo '⭕ Not running')"
	@echo ""
	@if [ ! -d $(VENV_DIR) ] || [ ! -d $(WEB_DIR)/node_modules ]; then \
		echo "Next step: Run 'make install'"; \
	elif ! curl -s http://localhost:8000/health 2>/dev/null >/dev/null; then \
		echo "Next step: Run 'make dev'"; \
	else \
		echo "All systems operational!"; \
	fi

dev: kill-ports ## Start development servers
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Error: Python environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "Starting development servers..."
	@$(PYTHON) scripts/dev.py

stop: ## Stop all servers
	@echo "Stopping servers..."
	@$(MAKE) kill-ports
	@echo "All servers stopped."

kill-ports: ## Kill processes on ports 3000 and 8000
	@echo "Killing processes on development ports..."
	@lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	@lsof -ti:3001 | xargs kill -9 2>/dev/null || true
	@lsof -ti:3002 | xargs kill -9 2>/dev/null || true
	@lsof -ti:3003 | xargs kill -9 2>/dev/null || true
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@sleep 2


test: ## Run tests
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Error: Python environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "Running tests..."
	@# Python tests
	@. $(VENV_DIR)/bin/activate && \
		PYTHONPATH="$(PWD)" $(PYTEST) tests/ -v --tb=short || echo "Some tests failed"
	@# Frontend tests if available
	@if [ -f "$(WEB_DIR)/package.json" ]; then \
		cd $(WEB_DIR) && npm test -- --watchAll=false --passWithNoTests || echo "Some frontend tests failed"; \
	fi




clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	@rm -rf test-reports/ .pytest_cache/ .coverage htmlcov/ .mypy_cache/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@if [ -d "$(WEB_DIR)" ]; then \
		cd $(WEB_DIR) && rm -rf coverage/ .next/ dist/ build/; \
	fi
	@# Clean SQLite database files
	@rm -f *.db *.db-journal
	@echo "Clean complete."

reset: ## Full reset (removes dependencies)
	@echo "This will remove all installed dependencies and databases."
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(MAKE) stop; \
		$(MAKE) clean; \
		echo "Removing virtual environment..."; \
		rm -rf $(VENV_DIR); \
		echo "Removing node modules..."; \
		if [ -d "$(WEB_DIR)" ]; then rm -rf $(WEB_DIR)/node_modules; fi; \
		rm -f .env; \
		echo "Reset complete. Run 'make install' to start fresh."; \
	fi

