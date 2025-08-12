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

.PHONY: help check install dev test stop kill-ports status clean reset

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
	@echo "Database Commands:"
	@echo "  make db-setup-docker     Setup PostgreSQL with Docker"
	@echo "  make db-setup-local      Setup PostgreSQL locally"
	@echo "  make db-migrate          Run database migrations"
	@echo "  make db-check            Test database connection"
	@echo "  make db-status           Show database health status"
	@echo "  make db-troubleshoot     Run comprehensive diagnostics"
	@echo "  make db-backup           Create database backup"
	@echo "  make db-restore          Restore from backup"
	@echo "  make db-reset            Reset database (DESTRUCTIVE)"
	@echo ""
	@echo "Quality Gates:"
	@echo "  make lint                Run code linting (Ruff)"
	@echo "  make typecheck           Run type checking (mypy)"
	@echo "  make complexity          Check code complexity (Radon)"
	@echo "  make security            Security vulnerability scan (Safety)"
	@echo "  make quality             Run all quality checks"
	@echo "  make test-onboarding     Validate clean installation process"
	@echo ""
	@echo "Coverage Analysis:"
	@echo "  make coverage-dev        Fast development coverage"
	@echo "  make coverage-ci         Comprehensive CI coverage"
	@echo "  make coverage-baseline   Establish coverage baseline"
	@echo "  make coverage-report     Generate coverage report"
	@echo "  make coverage-clean      Clean coverage artifacts"
	@echo ""
	@echo "Run 'make install' then 'make dev' to get started."

check: ## Check environment prerequisites
	@echo "✓ Checking Python..."
	@python --version
	@echo "✓ Checking pip..."
	@pip --version
	@echo "✓ Environment check passed!"

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
	@rm -rf test-reports/ .pytest_cache/ .mypy_cache/
	@# Clean coverage artifacts
	@rm -rf htmlcov* coverage*.json coverage*.xml .coverage* coverage-data.json
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

# Database setup commands
.PHONY: db-setup-docker db-setup-local db-migrate db-check db-troubleshoot db-status db-reset db-backup db-restore

## Setup PostgreSQL with Docker
db-setup-docker:
	@echo "Setting up PostgreSQL with pgvector using Docker..."
	@bash scripts/setup-db-docker.sh

## Setup PostgreSQL locally
db-setup-local:
	@echo "Setting up PostgreSQL with pgvector locally..."
	@bash scripts/setup-db-local.sh

## Run database migrations
db-migrate:
	@echo "Running database migrations..."
	@if command -v alembic &> /dev/null; then \
		alembic upgrade head; \
	else \
		echo "Alembic not found. Please install dependencies first: make install"; \
		exit 1; \
	fi

## Check database connection
db-check:
	@echo "Checking database connection..."
	@python -c "import os, psycopg2; conn = psycopg2.connect(os.getenv('DATABASE_URL', 'postgresql://freeagentics:freeagentics_dev@localhost:5432/freeagentics')); print('✅ Database connection successful!'); conn.close()" 2>/dev/null || echo "❌ Database connection failed. Check your DATABASE_URL in .env"

## Run comprehensive database diagnostics
db-troubleshoot:
	@echo "Running database troubleshooting diagnostics..."
	@bash scripts/db-troubleshoot.sh

## Show database status and health
db-status:
	@echo "=== Database Status ==="
	@echo "Container Status:"
	@docker-compose -f docker-compose.db.yml ps 2>/dev/null || echo "❌ Docker Compose not available"
	@echo ""
	@echo "Connection Test:"
	@$(MAKE) db-check
	@echo ""
	@echo "Migration Status:"
	@if command -v alembic &> /dev/null; then \
		alembic current 2>/dev/null || echo "❌ Migration status unavailable"; \
	else \
		echo "❌ Alembic not installed"; \
	fi
	@echo ""
	@echo "Quick Diagnostics:"
	@bash scripts/db-troubleshoot.sh 2>/dev/null | head -20 || echo "❌ Troubleshoot script failed"

## Reset database (DESTRUCTIVE - removes all data)
db-reset:
	@echo "⚠️  This will remove ALL database data!"
	@read -p "Are you sure? Type 'yes' to continue: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		echo "Stopping services..."; \
		$(MAKE) stop; \
		echo "Removing database..."; \
		docker-compose -f docker-compose.db.yml down -v; \
		docker volume rm freeagentics_postgres_data 2>/dev/null || true; \
		echo "Restarting database..."; \
		$(MAKE) db-setup-docker; \
		echo "Running migrations..."; \
		$(MAKE) db-migrate; \
		echo "✅ Database reset complete!"; \
	else \
		echo "Reset cancelled."; \
	fi

## Backup database
db-backup:
	@echo "Creating database backup..."
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	backup_file="backup_$${timestamp}.sql"; \
	database_url=$${DATABASE_URL:-"postgresql://freeagentics:freeagentics_dev@localhost:5432/freeagentics"}; \
	if pg_dump "$$database_url" > "$$backup_file"; then \
		echo "✅ Backup created: $$backup_file"; \
	else \
		echo "❌ Backup failed"; \
		exit 1; \
	fi

## Restore database from backup
db-restore:
	@echo "Available backup files:"
	@ls -la backup_*.sql 2>/dev/null || echo "No backup files found"
	@echo ""
	@read -p "Enter backup filename (or press Enter to cancel): " backup_file; \
	if [ -n "$$backup_file" ] && [ -f "$$backup_file" ]; then \
		echo "⚠️  This will replace all current database data!"; \
		read -p "Continue? Type 'yes' to proceed: " confirm; \
		if [ "$$confirm" = "yes" ]; then \
			database_url=$${DATABASE_URL:-"postgresql://freeagentics:freeagentics_dev@localhost:5432/freeagentics"}; \
			if psql "$$database_url" < "$$backup_file"; then \
				echo "✅ Database restored from $$backup_file"; \
			else \
				echo "❌ Restore failed"; \
				exit 1; \
			fi; \
		else \
			echo "Restore cancelled."; \
		fi; \
	else \
		echo "Invalid or missing backup file."; \
	fi

# Quality Gates Commands
.PHONY: lint typecheck complexity security quality

lint: ## Run code linting
	@echo "Running Ruff linting..."
	@. $(VENV_DIR)/bin/activate && python -m ruff check --config=pyproject.toml

typecheck: ## Run type checking
	@echo "Running mypy type checking..."
	@. $(VENV_DIR)/bin/activate && python -m mypy --config-file=pyproject.toml

complexity: ## Check code complexity with Radon
	@echo "Running Radon complexity analysis..."
	@. $(VENV_DIR)/bin/activate && radon cc --min=C . --exclude="node_modules,venv,.git,.next,__pycache__,test-onboarding" || true

security: ## Run security vulnerability scan
	@echo "Running Safety security scan..."
	@. $(VENV_DIR)/bin/activate && safety scan || true

quality: lint typecheck complexity security ## Run all quality checks
	@echo "All quality checks completed."

test-onboarding: ## Validate clean installation and onboarding process
	@echo "Running onboarding validation..."
	@python scripts/test_onboarding_validation.py

# Coverage Analysis Commands
.PHONY: coverage-dev coverage-ci coverage-baseline coverage-report coverage-clean

coverage-dev: ## Fast development coverage analysis
	@echo "Running development coverage analysis..."
	@. $(VENV_DIR)/bin/activate && ./scripts/coverage-dev.sh

coverage-ci: ## Comprehensive CI coverage validation
	@echo "Running CI coverage validation..."
	@. $(VENV_DIR)/bin/activate && ./scripts/coverage-ci.sh

coverage-baseline: ## Establish or update coverage baseline
	@echo "Updating coverage baseline..."
	@. $(VENV_DIR)/bin/activate && python scripts/coverage-baseline.py --compare

coverage-report: ## Generate detailed coverage report
	@echo "Generating coverage report..."
	@. $(VENV_DIR)/bin/activate && python scripts/coverage-check.py --output=coverage-detailed.json
	@echo "📄 Detailed report: coverage-detailed.json"
	@if [ -f "htmlcov/index.html" ]; then \
		echo "🌐 HTML report: htmlcov/index.html"; \
	fi

coverage-clean: ## Clean coverage artifacts
	@echo "Cleaning coverage artifacts..."
	@rm -rf htmlcov* coverage*.json coverage*.xml .coverage* coverage-data.json
	@echo "Coverage artifacts cleaned."

# Performance Benchmarking Commands
.PHONY: bench bench-gate bench-baseline

bench: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	@. $(VENV_DIR)/bin/activate && python benchmarks/simple_benchmark_runner.py

bench-gate: ## Check performance against gates (for CI)
	@echo "Checking performance gates..."
	@if [ ! -f "latest_benchmark_results.json" ]; then \
		echo "No benchmark results found. Run 'make bench' first."; \
		exit 1; \
	fi
	@. $(VENV_DIR)/bin/activate && python benchmarks/performance_gate.py latest_benchmark_results.json

bench-baseline: ## Update performance baseline
	@echo "Updating performance baseline..."
	@. $(VENV_DIR)/bin/activate && python performance_baseline_establishment.py
