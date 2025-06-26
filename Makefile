# FreeAgentics Development Makefile
# Expert Committee: Robert C. Martin, Kent Beck, Rich Hickey, Conor Heins

.PHONY: setup dev clean test test-full lint format help
.DEFAULT_GOAL := help

# Colors for terminal output
BOLD := \033[1m
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

## Development Environment
setup: ## Set up full development environment (dependencies, database, etc.)
	@echo "$(BOLD)$(BLUE)Setting up FreeAgentics development environment...$(RESET)"
	@./infrastructure/scripts/development/setup-dev-environment.sh
	@echo "$(BOLD)$(GREEN)Installing pre-commit hooks...$(RESET)"
	@pip install -r requirements-dev.txt
	@pre-commit install --install-hooks
	@cd web && npm install && npx husky install
	@echo "$(BOLD)$(GREEN)Pre-commit hooks installed successfully!$(RESET)"

dev: ## Start development environment (frontend + backend)
	@echo "$(BOLD)$(GREEN)Starting development environment...$(RESET)"
	@./infrastructure/scripts/development/start-dev.sh

dev-docker: ## Start development environment with Docker
	@echo "$(BOLD)$(GREEN)Starting Docker development environment...$(RESET)"
	@docker-compose -f config/environments/development/docker-compose.yml up -d

## Quality & Testing
test: ## Run basic test suite
	@echo "$(BOLD)$(YELLOW)Running test suite...$(RESET)"
	@cd web && npm run test
	@python -m pytest tests/ -v

test-full: ## Run full test suite with verbose output and coverage [user preference]
	@echo "$(BOLD)$(YELLOW)Running full test suite with maximum verbosity...$(RESET)"
	@cd web && npm run test:coverage
	@python -m pytest tests/ -vvv --tb=long --cov=. --cov-report=html --cov-report=term

type-check: ## Run type checking for both TypeScript and Python [user preference]
	@echo "$(BOLD)$(BLUE)Running type checks...$(RESET)"
	@cd web && npx tsc --noEmit --pretty
	@mypy --verbose --show-traceback --show-error-context --show-column-numbers --show-error-codes --pretty --show-absolute-path .

lint: ## Run linting for all code
	@echo "$(BOLD)$(BLUE)Running linters...$(RESET)"
	@cd web && npm run lint:strict
	@black . --check --line-length=100
	@isort . --check-only --line-length=100
	@flake8 --config config/.flake8 .

format: ## Format all code
	@echo "$(BOLD)$(GREEN)Formatting code...$(RESET)"
	@cd web && npm run format
	@black --line-length=100 .
	@isort --line-length=100 .

quality: ## Run all quality checks (type-check + lint + test-full)
	@echo "$(BOLD)$(BLUE)Running full quality suite...$(RESET)"
	@$(MAKE) type-check
	@$(MAKE) lint
	@$(MAKE) test-full

## Pre-commit Hooks
install-hooks: ## Install and update pre-commit hooks
	@echo "$(BOLD)$(GREEN)Installing pre-commit hooks...$(RESET)"
	@pip install -r requirements-dev.txt
	@pre-commit install --install-hooks
	@pre-commit autoupdate
	@cd web && npm install && npx husky install
	@echo "$(BOLD)$(GREEN)All hooks installed and updated!$(RESET)"

validate-hooks: ## Run all pre-commit hooks on all files
	@echo "$(BOLD)$(BLUE)Running pre-commit hooks validation...$(RESET)"
	@pre-commit run --all-files

hooks-update: ## Update all pre-commit hook versions
	@echo "$(BOLD)$(YELLOW)Updating pre-commit hooks...$(RESET)"
	@pre-commit autoupdate
	@pre-commit run --all-files

## Database
db-setup: ## Set up database
	@echo "$(BOLD)$(BLUE)Setting up database...$(RESET)"
	@./infrastructure/scripts/setup-database.sh

db-reset: ## Reset database
	@echo "$(BOLD)$(YELLOW)Resetting database...$(RESET)"
	@./infrastructure/scripts/development/reset-database.sh

## Cleanup
clean: ## Clean all build artifacts and caches
	@echo "$(BOLD)$(RED)Cleaning build artifacts...$(RESET)"
	@cd web && npm run clean
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .coverage htmlcov/ coverage/

## Docker
docker-build: ## Build all Docker images
	@echo "$(BOLD)$(BLUE)Building Docker images...$(RESET)"
	@docker-compose -f config/environments/development/docker-compose.yml build

docker-down: ## Stop and remove Docker containers
	@echo "$(BOLD)$(YELLOW)Stopping Docker environment...$(RESET)"
	@docker-compose -f config/environments/development/docker-compose.yml down

docker-clean: ## Clean Docker containers and volumes
	@echo "$(BOLD)$(RED)Cleaning Docker environment...$(RESET)"
	@docker-compose -f config/environments/development/docker-compose.yml down -v
	@docker system prune -f

## Help
help: ## Show this help message
	@echo "$(BOLD)FreeAgentics Development Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Quick Start:$(RESET)"
	@echo "  $(GREEN)make setup$(RESET)     - Set up everything for first time"
	@echo "  $(GREEN)make dev$(RESET)       - Start development environment"
	@echo "  $(GREEN)make quality$(RESET)   - Run all quality checks before commit"
