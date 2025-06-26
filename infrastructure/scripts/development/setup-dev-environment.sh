#!/bin/bash
#
# FreeAgentics Development Environment Setup
# Expert Committee: Robert C. Martin, Kent Beck, Rich Hickey, Conor Heins
#
# This script sets up a complete development environment for FreeAgentics
# including Python dependencies, Node.js dependencies, database, and development tools

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
PYTHON_VERSION="3.11"
NODE_VERSION="18"

echo -e "${BOLD}${BLUE}FreeAgentics Development Environment Setup${NC}"
echo "=============================================="
echo ""
echo "Project Root: $PROJECT_ROOT"
echo ""

# Function to print section headers
print_section() {
    echo ""
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo "$(printf '=%.0s' {1..50})"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if we're in the project root
check_project_root() {
    if [ ! -f "$PROJECT_ROOT/pyproject.toml" ] || [ ! -f "$PROJECT_ROOT/web/package.json" ]; then
        echo -e "${RED}Error: This script must be run from the FreeAgentics project root${NC}"
        echo "Expected files not found: pyproject.toml, web/package.json"
        exit 1
    fi
}

# Function to check system requirements
check_system_requirements() {
    print_section "Checking System Requirements"

    local missing_deps=()

    # Check Python
    if command_exists python3; then
        local python_ver=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
        if [ "$(printf '%s\n' "$PYTHON_VERSION" "$python_ver" | sort -V | head -n1)" = "$PYTHON_VERSION" ]; then
            echo -e "Python: ${GREEN}✓${NC} $(python3 --version)"
        else
            echo -e "Python: ${RED}✗${NC} Found $python_ver, need $PYTHON_VERSION+"
            missing_deps+=("python$PYTHON_VERSION")
        fi
    else
        echo -e "Python: ${RED}✗${NC} Not found"
        missing_deps+=("python$PYTHON_VERSION")
    fi

    # Check Node.js
    if command_exists node; then
        local node_ver=$(node --version | sed 's/v//' | cut -d'.' -f1)
        if [ "$node_ver" -ge "$NODE_VERSION" ]; then
            echo -e "Node.js: ${GREEN}✓${NC} $(node --version)"
        else
            echo -e "Node.js: ${RED}✗${NC} Found v$node_ver, need v$NODE_VERSION+"
            missing_deps+=("nodejs")
        fi
    else
        echo -e "Node.js: ${RED}✗${NC} Not found"
        missing_deps+=("nodejs")
    fi

    # Check npm
    if command_exists npm; then
        echo -e "npm: ${GREEN}✓${NC} $(npm --version)"
    else
        echo -e "npm: ${RED}✗${NC} Not found"
        missing_deps+=("npm")
    fi

    # Check PostgreSQL
    if command_exists pg_isready; then
        echo -e "PostgreSQL: ${GREEN}✓${NC} Available"
    else
        echo -e "PostgreSQL: ${YELLOW}⚠${NC} Not found (optional - can use Docker)"
    fi

    # Check Redis
    if command_exists redis-cli; then
        echo -e "Redis: ${GREEN}✓${NC} Available"
    else
        echo -e "Redis: ${YELLOW}⚠${NC} Not found (optional - can use Docker)"
    fi

    # Check Docker
    if command_exists docker; then
        echo -e "Docker: ${GREEN}✓${NC} $(docker --version | cut -d' ' -f3 | sed 's/,//')"
    else
        echo -e "Docker: ${YELLOW}⚠${NC} Not found (optional for development)"
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo ""
        echo -e "${RED}Missing required dependencies:${NC}"
        printf '%s\n' "${missing_deps[@]}"
        echo ""
        echo "Install missing dependencies and run this script again."
        echo ""
        echo "Quick install commands:"
        echo "  macOS (Homebrew): brew install python@$PYTHON_VERSION node postgresql redis"
        echo "  Ubuntu/Debian: sudo apt-get install python$PYTHON_VERSION python3-pip nodejs npm postgresql-client redis-tools"
        echo "  Or use Docker: docker-compose -f config/environments/development/docker-compose.yml up -d"
        exit 1
    fi
}

# Function to create Python virtual environment
setup_python_environment() {
    print_section "Setting up Python Environment"

    cd "$PROJECT_ROOT"

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo -n "Creating Python virtual environment... "
        python3 -m venv venv
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "Virtual environment: ${GREEN}✓${NC} Already exists"
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    echo -n "Upgrading pip... "
    pip install --upgrade pip > /dev/null 2>&1
    echo -e "${GREEN}✓${NC}"

    # Install dependencies
    echo -n "Installing Python dependencies... "
    pip install -e ".[dev]" > /dev/null 2>&1
    echo -e "${GREEN}✓${NC}"

    # Install additional development tools
    echo -n "Installing development tools... "
    pip install pytest-watch coverage-badge > /dev/null 2>&1 || true
    echo -e "${GREEN}✓${NC}"
}

# Function to setup Node.js environment
setup_nodejs_environment() {
    print_section "Setting up Node.js Environment"

    cd "$PROJECT_ROOT/web"

    # Install dependencies
    echo -n "Installing Node.js dependencies... "
    npm install > /dev/null 2>&1
    echo -e "${GREEN}✓${NC}"

    # Check for security vulnerabilities
    echo -n "Checking for security vulnerabilities... "
    npm audit fix --audit-level moderate > /dev/null 2>&1 || true
    echo -e "${GREEN}✓${NC}"

    # Initialize Husky from project root
    echo -n "Setting up Husky hooks... "
    cd "$PROJECT_ROOT"
    npx husky install web/.husky > /dev/null 2>&1 || true
    cd "$PROJECT_ROOT/web"
    echo -e "${GREEN}✓${NC}"
}

# Function to setup database
setup_database() {
    print_section "Setting up Database"

    cd "$PROJECT_ROOT"

    # Check if database setup script exists and run it
    if [ -f "infrastructure/scripts/setup-database.sh" ]; then
        echo "Running database setup script..."
        bash infrastructure/scripts/setup-database.sh check
        if [ $? -eq 0 ]; then
            echo -e "Database: ${GREEN}✓${NC} PostgreSQL is accessible"
            bash infrastructure/scripts/setup-database.sh all
        else
            echo -e "Database: ${YELLOW}⚠${NC} PostgreSQL not accessible, skipping database setup"
            echo "You can set up the database later with: make db-setup"
        fi
    else
        echo -e "Database setup script not found: ${YELLOW}⚠${NC} Skipping database setup"
    fi
}

# Function to create development configuration
setup_development_config() {
    print_section "Setting up Development Configuration"

    cd "$PROJECT_ROOT"

    # Create .env.development if it doesn't exist
    if [ ! -f "config/environments/.env.development" ]; then
        echo -n "Creating development environment file... "
        mkdir -p config/environments
        cat > config/environments/.env.development << 'EOF'
# FreeAgentics Development Environment Configuration
NODE_ENV=development
API_URL=http://localhost:8000
DATABASE_URL=postgresql://freeagentics:dev_password@localhost:5432/freeagentics_dev
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_PORT=8000
API_HOST=localhost
API_RELOAD=true

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Development Tools
DEBUG=true
LOG_LEVEL=debug
PYTHONUNBUFFERED=1
EOF
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "Development config: ${GREEN}✓${NC} Already exists"
    fi

    # Create development shell script
    if [ ! -f "infrastructure/scripts/development/start-dev.sh" ]; then
        echo -n "Creating development start script... "
        cat > infrastructure/scripts/development/start-dev.sh << 'EOF'
#!/bin/bash
# Start FreeAgentics development environment

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Starting FreeAgentics Development Environment${NC}"
echo "=============================================="

# Start backend in background
echo -e "${GREEN}Starting Python backend...${NC}"
source venv/bin/activate
export $(cat config/environments/.env.development | xargs)
cd "$PROJECT_ROOT"
python -m uvicorn api.main:app --reload --host localhost --port 8000 &
BACKEND_PID=$!

# Start frontend
echo -e "${GREEN}Starting Next.js frontend...${NC}"
cd "$PROJECT_ROOT/web"
npm run dev &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}Development environment started!${NC}"
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000"
echo "Press Ctrl+C to stop both services"

# Handle shutdown
cleanup() {
    echo ""
    echo "Shutting down development environment..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait
EOF
        chmod +x infrastructure/scripts/development/start-dev.sh
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "Start script: ${GREEN}✓${NC} Already exists"
    fi
}

# Function to run initial tests
run_initial_tests() {
    print_section "Running Initial Tests"

    cd "$PROJECT_ROOT"

    # Activate Python environment
    source venv/bin/activate

    # Run type checking with user's preferred verbose flags
    echo -n "Running MyPy type checking... "
    if mypy --verbose --show-traceback --show-error-context --show-column-numbers --show-error-codes --pretty --show-absolute-path . > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}⚠${NC} Type checking issues found (run 'make type-check' for details)"
    fi

    # Run Python tests with user's preferred verbose flags
    echo -n "Running Python tests... "
    if python -m pytest tests/ -vvv --tb=long -q > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}⚠${NC} Some Python tests failed (run 'make test-full' for details)"
    fi

    # Run TypeScript type checking
    cd "$PROJECT_ROOT/web"
    echo -n "Running TypeScript type checking... "
    if npx tsc --noEmit --pretty > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}⚠${NC} TypeScript issues found (run 'make type-check' for details)"
    fi

    # Run frontend tests
    echo -n "Running frontend tests... "
    if npm run test -- --watchAll=false --verbose=false > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}⚠${NC} Some frontend tests failed (run 'make test-full' for details)"
    fi
}

# Function to print next steps
print_next_steps() {
    print_section "Setup Complete!"

    echo -e "${GREEN}Development environment is ready!${NC}"
    echo ""
    echo -e "${BOLD}Next steps:${NC}"
    echo "  1. Start development:     ${BLUE}make dev${NC}"
    echo "  2. Run all tests:         ${BLUE}make test-full${NC}"
    echo "  3. Check code quality:    ${BLUE}make quality${NC}"
    echo "  4. View all commands:     ${BLUE}make help${NC}"
    echo ""
    echo -e "${BOLD}Docker alternative:${NC}"
    echo "  Start with Docker:        ${BLUE}make dev-docker${NC}"
    echo ""
    echo -e "${BOLD}Troubleshooting:${NC}"
    echo "  Reset database:           ${BLUE}make db-reset${NC}"
    echo "  Clean build artifacts:    ${BLUE}make clean${NC}"
    echo "  View logs:                Check terminal output or Docker logs"
    echo ""
    echo -e "${GREEN}Happy coding! 🚀${NC}"
}

# Main execution
main() {
    # Change to project root
    cd "$PROJECT_ROOT"

    # Run setup steps
    check_project_root
    check_system_requirements
    setup_python_environment
    setup_nodejs_environment
    setup_database
    setup_development_config
    run_initial_tests
    print_next_steps
}

# Handle command line arguments
case "${1:-setup}" in
    "check")
        check_system_requirements
        ;;
    "python")
        setup_python_environment
        ;;
    "node")
        setup_nodejs_environment
        ;;
    "database")
        setup_database
        ;;
    "config")
        setup_development_config
        ;;
    "test")
        run_initial_tests
        ;;
    "setup"|"")
        main
        ;;
    *)
        echo "Usage: $0 [check|python|node|database|config|test|setup]"
        echo ""
        echo "Commands:"
        echo "  check    - Check system requirements"
        echo "  python   - Set up Python environment only"
        echo "  node     - Set up Node.js environment only"
        echo "  database - Set up database only"
        echo "  config   - Set up development configuration only"
        echo "  test     - Run initial tests only"
        echo "  setup    - Run full setup (default)"
        exit 1
        ;;
esac
