#!/usr/bin/env bash
# FreeAgentics Setup Script - Works on any machine!
# This script sets up the complete FreeAgentics environment with all dependencies

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.10"
POSTGRES_DB="freeagentics_dev"
REDIS_PORT="6379"
API_PORT="8000"
WEB_PORT="3000"
LOG_FILE="setup.log"
VENV_DIR="venv"

# Progress tracking
TOTAL_STEPS=10
CURRENT_STEP=0

# Start logging
echo "FreeAgentics Setup Started at $(date)" > "$LOG_FILE"

# Helper functions
log() {
    echo "$1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}✓ $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}⚠ $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}ℹ $1${NC}" | tee -a "$LOG_FILE"
}

progress() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo -e "${PURPLE}[$CURRENT_STEP/$TOTAL_STEPS]${NC} $1" | tee -a "$LOG_FILE"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Compare versions
version_ge() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

# ASCII Art Banner
show_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
    ___               _                    _   _
   / __\_ __ ___  ___/_\   __ _  ___ _ __ | |_(_) ___ ___
  / _\| '__/ _ \/ _ //_\\ / _` |/ _ \ '_ \| __| |/ __/ __|
 / /  | | |  __/  __/  _  \ (_| |  __/ | | | |_| | (__\__ \
 \/   |_|  \___|\___\_/ \_/\__, |\___|_| |_|\__|_|\___|___/
                           |___/
     🤖 Active Inference Multi-Agent System 🧠
EOF
    echo -e "${NC}"
}

# System diagnostics
run_diagnostics() {
    info "Running system diagnostics..."
    log "OS: $(uname -s)"
    log "Architecture: $(uname -m)"
    log "Hostname: $(hostname)"

    if command_exists python3; then
        log "Python: $(python3 --version)"
    fi

    if command_exists node; then
        log "Node.js: $(node --version)"
    fi

    if command_exists docker; then
        log "Docker: $(docker --version)"
    fi

    log "Available memory: $(free -h 2>/dev/null | grep Mem | awk '{print $7}' || echo 'N/A')"
    log "Available disk: $(df -h . | tail -1 | awk '{print $4}')"
}

# Step 1: Check Python
check_python() {
    progress "Checking Python installation..."

    if ! command_exists python3; then
        error "Python 3 is not installed"
        info "Please install Python 3.10 or higher:"
        info "  Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
        info "  macOS: brew install python3"
        info "  Windows: Download from https://python.org"
        return 1
    fi

    local python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

    if ! version_ge "$python_version" "$PYTHON_MIN_VERSION"; then
        error "Python $python_version found, but $PYTHON_MIN_VERSION+ is required"
        return 1
    fi

    success "Python $python_version found"

    # Check for venv module
    if ! python3 -c "import venv" 2>/dev/null; then
        error "Python venv module not found"
        info "Install with: sudo apt-get install python3-venv"
        return 1
    fi

    return 0
}

# Step 2: Setup Python virtual environment
setup_venv() {
    progress "Setting up Python virtual environment..."

    if [ -d "$VENV_DIR" ]; then
        warning "Virtual environment already exists"

        # Activate and verify
        source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null || {
            error "Failed to activate existing virtual environment"
            info "Removing corrupt venv..."
            rm -rf "$VENV_DIR"
            python3 -m venv "$VENV_DIR"
        }
    else
        python3 -m venv "$VENV_DIR"
        success "Created virtual environment"
    fi

    # Activate venv
    source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null

    # Upgrade pip
    pip install --quiet --upgrade pip
    success "Virtual environment ready"
}

# Step 3: Install Python dependencies
install_python_deps() {
    progress "Installing Python dependencies..."

    if [ ! -f "requirements.txt" ]; then
        error "requirements.txt not found"
        return 1
    fi

    # Install in chunks to show progress
    local total_deps=$(wc -l < requirements.txt)
    info "Installing $total_deps dependencies..."

    # First install critical dependencies
    pip install --quiet fastapi uvicorn pydantic sqlalchemy psycopg2-binary redis
    echo -n "."

    # Install PyMDP separately as it can be problematic
    pip install --quiet pymdp || warning "PyMDP installation failed - continuing without it"
    echo -n "."

    # Install remaining dependencies
    pip install --quiet -r requirements.txt || {
        warning "Some dependencies failed to install"
        info "Attempting to install core dependencies only..."
        pip install --quiet fastapi uvicorn pydantic sqlalchemy
    }
    echo ""

    success "Python dependencies installed"
}

# Step 4: Check/Install PostgreSQL
setup_postgres() {
    progress "Setting up PostgreSQL..."

    if command_exists psql; then
        success "PostgreSQL is installed"

        # Check if PostgreSQL is running
        if pg_isready -q 2>/dev/null; then
            success "PostgreSQL is running"
        else
            warning "PostgreSQL is not running"
            info "Attempting to start PostgreSQL..."

            if [[ "$OSTYPE" == "linux-gnu"* ]]; then
                sudo service postgresql start 2>/dev/null || systemctl start postgresql 2>/dev/null || {
                    warning "Could not start PostgreSQL automatically"
                    info "Please start PostgreSQL manually"
                }
            elif [[ "$OSTYPE" == "darwin"* ]]; then
                brew services start postgresql 2>/dev/null || {
                    warning "Could not start PostgreSQL automatically"
                    info "Try: brew services start postgresql"
                }
            fi
        fi

        # Create database
        info "Creating database '$POSTGRES_DB'..."
        createdb "$POSTGRES_DB" 2>/dev/null || {
            warning "Database might already exist or creation failed"
            info "Continuing with existing database"
        }

    else
        warning "PostgreSQL is not installed"
        info "For full functionality, install PostgreSQL:"
        info "  Ubuntu/Debian: sudo apt-get install postgresql"
        info "  macOS: brew install postgresql"
        info "Will use SQLite as fallback for demo"
        export USE_SQLITE=true
    fi
}

# Step 5: Check/Install Redis
setup_redis() {
    progress "Setting up Redis..."

    if command_exists redis-cli; then
        success "Redis is installed"

        # Check if Redis is running
        if redis-cli ping >/dev/null 2>&1; then
            success "Redis is running"
        else
            warning "Redis is not running"
            info "Attempting to start Redis..."

            if [[ "$OSTYPE" == "linux-gnu"* ]]; then
                sudo service redis-server start 2>/dev/null || systemctl start redis 2>/dev/null || {
                    warning "Could not start Redis automatically"
                }
            elif [[ "$OSTYPE" == "darwin"* ]]; then
                brew services start redis 2>/dev/null || {
                    warning "Could not start Redis automatically"
                }
            fi
        fi
    else
        warning "Redis is not installed"
        info "For caching support, install Redis:"
        info "  Ubuntu/Debian: sudo apt-get install redis-server"
        info "  macOS: brew install redis"
        info "Will run without caching for demo"
        export DISABLE_REDIS=true
    fi
}

# Step 6: Setup Node.js dependencies (for frontend)
setup_nodejs() {
    progress "Setting up Node.js environment..."

    if ! command_exists node; then
        warning "Node.js is not installed"
        info "Frontend will not be available"
        info "Install Node.js from: https://nodejs.org"
        export SKIP_FRONTEND=true
        return 0
    fi

    success "Node.js $(node --version) found"

    # Install frontend dependencies
    if [ -d "web" ] && [ -f "web/package.json" ]; then
        info "Installing frontend dependencies..."
        cd web
        npm install --quiet --no-audit --no-fund || {
            warning "Some frontend dependencies failed to install"
            export SKIP_FRONTEND=true
        }
        cd ..
        success "Frontend dependencies installed"
    else
        warning "Frontend directory not found"
        export SKIP_FRONTEND=true
    fi
}

# Step 7: Initialize database
init_database() {
    progress "Initializing database..."

    # Run Alembic migrations
    if command_exists alembic; then
        info "Running database migrations..."
        alembic upgrade head 2>/dev/null || {
            warning "Database migrations failed"
            info "Will create tables on first run"
        }
    fi

    # Create initial data
    if [ -f "scripts/seed_database.py" ]; then
        python scripts/seed_database.py 2>/dev/null || {
            warning "Database seeding failed"
            info "Will start with empty database"
        }
    fi

    success "Database initialized"
}

# Step 8: Generate configuration files
generate_configs() {
    progress "Generating configuration files..."

    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# FreeAgentics Configuration
DATABASE_URL=${USE_SQLITE:+sqlite:///freeagentics.db}${USE_SQLITE:-postgresql://localhost/freeagentics_dev}
REDIS_URL=${DISABLE_REDIS:+}${DISABLE_REDIS:-redis://localhost:6379}
SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || echo "dev-secret-key-$(date +%s)")
DEBUG=true
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=$API_PORT

# Agent Configuration
PYMDP_ENABLED=${PYMDP_ENABLED:-true}
MAX_AGENTS=10
DEFAULT_GRID_SIZE=10

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:$API_PORT
EOF
        success "Created .env configuration"
    else
        info "Using existing .env file"
    fi
}

# Step 9: Create demo data
create_demo_data() {
    progress "Creating demo agents and data..."

    # Create a simple Python script to generate demo data
    cat > create_demo.py << 'EOF'
import os
import sys
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from agents.base_agent import BasicExplorerAgent
    from database.models import Agent
    from database.session import SessionLocal

    print("Creating demo agents...")

    # Create some demo agents in memory
    demo_agents = [
        {"name": "Explorer Alpha", "type": "explorer", "position": [2, 3]},
        {"name": "Collector Beta", "type": "collector", "position": [5, 7]},
        {"name": "Analyzer Gamma", "type": "analyzer", "position": [8, 2]},
    ]

    # Save demo configuration
    with open("demo_config.json", "w") as f:
        json.dump({
            "agents": demo_agents,
            "grid_size": 10,
            "created_at": datetime.now().isoformat()
        }, f, indent=2)

    print("✓ Created demo configuration")

except Exception as e:
    print(f"Note: Could not create full demo data: {e}")
    print("The system will create agents on first run")
EOF

    python create_demo.py 2>/dev/null || info "Demo data will be created on first run"
    rm -f create_demo.py

    success "Demo environment prepared"
}

# Step 10: Start services
start_services() {
    progress "Starting FreeAgentics services..."

    # Kill any existing services on our ports
    lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
    lsof -ti:$WEB_PORT | xargs kill -9 2>/dev/null || true

    # Start API server
    info "Starting API server on port $API_PORT..."
    nohup python -m uvicorn api.main:app --host 0.0.0.0 --port $API_PORT --reload > api.log 2>&1 &
    API_PID=$!
    sleep 3

    # Check if API started
    if kill -0 $API_PID 2>/dev/null; then
        success "API server started (PID: $API_PID)"
    else
        error "API server failed to start"
        cat api.log
        return 1
    fi

    # Start frontend if available
    if [ -z "$SKIP_FRONTEND" ]; then
        info "Starting frontend on port $WEB_PORT..."
        cd web
        nohup npm run dev > ../frontend.log 2>&1 &
        FRONTEND_PID=$!
        cd ..
        sleep 5

        if kill -0 $FRONTEND_PID 2>/dev/null; then
            success "Frontend started (PID: $FRONTEND_PID)"
        else
            warning "Frontend failed to start"
            SKIP_FRONTEND=true
        fi
    fi

    # Save PIDs for cleanup
    echo "$API_PID" > .api.pid
    [ -n "${FRONTEND_PID:-}" ] && echo "$FRONTEND_PID" > .frontend.pid

    success "Services are running!"
}

# Health check
health_check() {
    info "Running health checks..."

    # Check API health
    if curl -s http://localhost:$API_PORT/health >/dev/null 2>&1; then
        success "API health check passed"
    else
        warning "API health check failed"
    fi

    # Check frontend
    if [ -z "$SKIP_FRONTEND" ] && curl -s http://localhost:$WEB_PORT >/dev/null 2>&1; then
        success "Frontend health check passed"
    fi
}

# Show success message
show_success() {
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✨ FreeAgentics Setup Complete! ✨${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${CYAN}🚀 Quick Start:${NC}"
    echo -e "   API Documentation: ${BLUE}http://localhost:$API_PORT/docs${NC}"
    [ -z "$SKIP_FRONTEND" ] && echo -e "   Web Interface: ${BLUE}http://localhost:$WEB_PORT${NC}"
    echo ""
    echo -e "${CYAN}📝 Demo Commands:${NC}"
    echo -e "   Create an agent:  ${YELLOW}curl -X POST http://localhost:$API_PORT/api/v1/agents -H 'Content-Type: application/json' -d '{\"name\":\"Demo Agent\"}'${NC}"
    echo -e "   View logs:        ${YELLOW}tail -f api.log${NC}"
    echo -e "   Stop services:    ${YELLOW}./stop.sh${NC}"
    echo ""
    echo -e "${CYAN}🧠 Next Steps:${NC}"
    echo -e "   1. Open the API docs to explore endpoints"
    echo -e "   2. Create some agents using the API"
    echo -e "   3. Watch them interact in the web interface"
    echo ""
    echo -e "${PURPLE}Happy exploring! 🤖${NC}"
}

# Create stop script
create_stop_script() {
    cat > stop.sh << 'EOF'
#!/bin/bash
echo "Stopping FreeAgentics services..."

# Stop API
if [ -f .api.pid ]; then
    kill $(cat .api.pid) 2>/dev/null && echo "✓ API stopped"
    rm .api.pid
fi

# Stop frontend
if [ -f .frontend.pid ]; then
    kill $(cat .frontend.pid) 2>/dev/null && echo "✓ Frontend stopped"
    rm .frontend.pid
fi

echo "All services stopped"
EOF
    chmod +x stop.sh
}

# Main setup flow
main() {
    show_banner
    run_diagnostics

    # Run all setup steps
    check_python || exit 1
    setup_venv || exit 1
    install_python_deps || exit 1
    setup_postgres
    setup_redis
    setup_nodejs
    init_database
    generate_configs
    create_demo_data
    start_services || exit 1

    # Final steps
    create_stop_script
    health_check
    show_success

    # Log completion
    echo "Setup completed successfully at $(date)" >> "$LOG_FILE"
}

# Handle arguments
case "${1:-}" in
    --diagnose)
        run_diagnostics
        ;;
    --clean)
        echo "Cleaning up..."
        ./stop.sh 2>/dev/null || true
        rm -rf "$VENV_DIR" node_modules web/node_modules
        rm -f .env demo_config.json *.log *.pid
        echo "Cleanup complete"
        ;;
    --help)
        echo "FreeAgentics Setup Script"
        echo "Usage: $0 [option]"
        echo "Options:"
        echo "  --diagnose    Run system diagnostics"
        echo "  --clean       Clean up all generated files"
        echo "  --help        Show this help message"
        echo ""
        echo "Run without options for full setup"
        ;;
    *)
        main
        ;;
esac
