#!/bin/bash

# CogniticNet Demo Environment - Run Demo Script
# This script orchestrates the execution of the demo including launching services and creating agents

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/demo-run.log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Demo configuration
DEMO_CONFIG_FILE="$PROJECT_ROOT/config/demo.json"
BACKEND_PORT=8000
FRONTEND_PORT=3000
REDIS_PORT=6379
DEMO_AGENT_COUNT=50
DEMO_COALITION_COUNT=10
AUTO_CREATE_AGENTS=true
OPEN_BROWSER=true

# Demo modes
DEMO_MODE="full"  # full, minimal, api-only, frontend-only
AGENT_MODE="standard"  # standard, spec-mode, pseudo-mode, build-mode, test-mode

# Process tracking
BACKEND_PID=""
FRONTEND_PID=""
REDIS_PID=""
CLEANUP_PERFORMED=false

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"

    case "$level" in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
        "DEMO")
            echo -e "${BOLD}[DEMO]${NC} $message"
            ;;
    esac
}

# Clear previous log
mkdir -p "$(dirname "$LOG_FILE")"
> "$LOG_FILE"

log "INFO" "Starting CogniticNet demo environment..."
log "INFO" "Project root: $PROJECT_ROOT"

# Function to load demo configuration
load_demo_config() {
    log "INFO" "Loading demo configuration..."

    if [[ -f "$DEMO_CONFIG_FILE" ]]; then
        log "SUCCESS" "Found demo configuration: $DEMO_CONFIG_FILE"

        # Extract values using simple grep/sed (avoiding jq dependency)
        if command -v python3 >/dev/null 2>&1; then
            DEMO_AGENT_COUNT=$(python3 -c "import json; print(json.load(open('$DEMO_CONFIG_FILE'))['agents']['default_count'])" 2>/dev/null || echo "50")
            DEMO_COALITION_COUNT=$(python3 -c "import json; print(json.load(open('$DEMO_CONFIG_FILE'))['coalitions']['default_count'])" 2>/dev/null || echo "10")
        fi

        log "INFO" "Demo agent count: $DEMO_AGENT_COUNT"
        log "INFO" "Demo coalition count: $DEMO_COALITION_COUNT"
    else
        log "WARN" "Demo configuration file not found, using defaults"
        log "INFO" "Default agent count: $DEMO_AGENT_COUNT"
        log "INFO" "Default coalition count: $DEMO_COALITION_COUNT"
    fi
}

# Function to check if port is available
check_port() {
    local port="$1"
    local service="$2"

    if command -v lsof >/dev/null 2>&1 && lsof -i ":$port" >/dev/null 2>&1; then
        log "WARN" "Port $port is already in use (needed for $service)"
        return 1
    else
        log "SUCCESS" "Port $port is available for $service"
        return 0
    fi
}

# Function to start Redis
start_redis() {
    log "INFO" "Starting Redis server..."

    # Check if Redis is already running
    if command -v redis-cli >/dev/null 2>&1 && redis-cli ping >/dev/null 2>&1; then
        log "SUCCESS" "Redis server is already running"
        return 0
    fi

    if ! command -v redis-server >/dev/null 2>&1; then
        log "ERROR" "Redis server not found. Please install Redis."
        return 1
    fi

    # Start Redis server
    if redis-server --daemonize yes --port "$REDIS_PORT" >/dev/null 2>&1; then
        sleep 2
        if redis-cli -p "$REDIS_PORT" ping >/dev/null 2>&1; then
            log "SUCCESS" "Redis server started on port $REDIS_PORT"
            return 0
        else
            log "ERROR" "Redis server failed to start properly"
            return 1
        fi
    else
        log "ERROR" "Failed to start Redis server"
        return 1
    fi
}

# Function to start backend
start_backend() {
    log "INFO" "Starting backend server..."

    # Check port availability
    if ! check_port "$BACKEND_PORT" "backend"; then
        log "ERROR" "Backend port $BACKEND_PORT is not available"
        return 1
    fi

    # Look for backend entry points
    local backend_cmd=""

    if [[ -f "$PROJECT_ROOT/backend/main.py" ]]; then
        backend_cmd="cd '$PROJECT_ROOT/backend' && python main.py"
    elif [[ -f "$PROJECT_ROOT/main.py" ]]; then
        backend_cmd="cd '$PROJECT_ROOT' && python main.py"
    elif [[ -f "$PROJECT_ROOT/backend/app.py" ]]; then
        backend_cmd="cd '$PROJECT_ROOT/backend' && python app.py"
    elif [[ -f "$PROJECT_ROOT/app.py" ]]; then
        backend_cmd="cd '$PROJECT_ROOT' && python app.py"
    elif [[ -f "$PROJECT_ROOT/backend/server.py" ]]; then
        backend_cmd="cd '$PROJECT_ROOT/backend' && python server.py"
    else
        log "WARN" "No backend entry point found, trying uvicorn..."
        if command -v uvicorn >/dev/null 2>&1; then
            backend_cmd="cd '$PROJECT_ROOT' && uvicorn main:app --host 0.0.0.0 --port $BACKEND_PORT --reload"
        else
            log "ERROR" "No backend entry point found and uvicorn not available"
            return 1
        fi
    fi

    log "INFO" "Backend command: $backend_cmd"

    # Start backend in background
    eval "$backend_cmd" > "$PROJECT_ROOT/logs/backend.log" 2>&1 &
    BACKEND_PID=$!

    # Wait for backend to start
    log "INFO" "Waiting for backend to start..."
    local max_attempts=30
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s "http://localhost:$BACKEND_PORT/health" >/dev/null 2>&1 || \
           curl -s "http://localhost:$BACKEND_PORT/" >/dev/null 2>&1; then
            log "SUCCESS" "Backend server started on port $BACKEND_PORT (PID: $BACKEND_PID)"
            return 0
        fi

        sleep 2
        attempt=$((attempt + 1))

        # Check if process is still running
        if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
            log "ERROR" "Backend process died unexpectedly"
            log "ERROR" "Check backend logs: cat $PROJECT_ROOT/logs/backend.log"
            return 1
        fi
    done

    log "ERROR" "Backend failed to start within timeout"
    log "ERROR" "Check backend logs: cat $PROJECT_ROOT/logs/backend.log"
    return 1
}

# Function to start frontend
start_frontend() {
    log "INFO" "Starting frontend server..."

    # Check port availability
    if ! check_port "$FRONTEND_PORT" "frontend"; then
        log "ERROR" "Frontend port $FRONTEND_PORT is not available"
        return 1
    fi

    # Look for frontend directory and entry points
    local frontend_cmd=""
    local frontend_dir="$PROJECT_ROOT"

    if [[ -d "$PROJECT_ROOT/frontend" ]]; then
        frontend_dir="$PROJECT_ROOT/frontend"
    fi

    # Check for Next.js
    if [[ -f "$frontend_dir/next.config.js" ]] || [[ -f "$frontend_dir/next.config.ts" ]]; then
        frontend_cmd="cd '$frontend_dir' && npm run dev"
    # Check for React (Create React App)
    elif [[ -f "$frontend_dir/App.js" ]] || [[ -f "$frontend_dir/App.tsx" ]]; then
        frontend_cmd="cd '$frontend_dir' && npm start"
    # Check for Vite
    elif [[ -f "$frontend_dir/vite.config.js" ]] || [[ -f "$frontend_dir/vite.config.ts" ]]; then
        frontend_cmd="cd '$frontend_dir' && npm run dev"
    # Check for package.json scripts
    elif [[ -f "$frontend_dir/package.json" ]]; then
        if grep -q '"dev"' "$frontend_dir/package.json"; then
            frontend_cmd="cd '$frontend_dir' && npm run dev"
        elif grep -q '"start"' "$frontend_dir/package.json"; then
            frontend_cmd="cd '$frontend_dir' && npm start"
        else
            log "WARN" "No recognized frontend start script found"
            return 1
        fi
    else
        log "WARN" "No frontend configuration found - skipping frontend startup"
        return 0
    fi

    log "INFO" "Frontend command: $frontend_cmd"

    # Check if node_modules exists
    if [[ ! -d "$frontend_dir/node_modules" ]]; then
        log "WARN" "Frontend dependencies not installed, installing now..."
        cd "$frontend_dir"
        if command -v yarn >/dev/null 2>&1; then
            yarn install
        else
            npm install
        fi
        cd "$PROJECT_ROOT"
    fi

    # Start frontend in background
    eval "$frontend_cmd" > "$PROJECT_ROOT/logs/frontend.log" 2>&1 &
    FRONTEND_PID=$!

    # Wait for frontend to start
    log "INFO" "Waiting for frontend to start..."
    local max_attempts=30
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s "http://localhost:$FRONTEND_PORT" >/dev/null 2>&1; then
            log "SUCCESS" "Frontend server started on port $FRONTEND_PORT (PID: $FRONTEND_PID)"
            return 0
        fi

        sleep 2
        attempt=$((attempt + 1))

        # Check if process is still running
        if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
            log "ERROR" "Frontend process died unexpectedly"
            log "ERROR" "Check frontend logs: cat $PROJECT_ROOT/logs/frontend.log"
            return 1
        fi
    done

    log "ERROR" "Frontend failed to start within timeout"
    log "ERROR" "Check frontend logs: cat $PROJECT_ROOT/logs/frontend.log"
    return 1
}

# Function to create demo agents with different modes
create_demo_agents() {
    if [[ "$AUTO_CREATE_AGENTS" != "true" ]]; then
        log "INFO" "Skipping automatic agent creation"
        return 0
    fi

    log "DEMO" "Creating demo agents in $AGENT_MODE mode..."

    local agent_script="$PROJECT_ROOT/scripts/demo/create-demo-agents.py"

    if [[ ! -f "$agent_script" ]]; then
        log "ERROR" "Demo agent creation script not found: $agent_script"
        return 1
    fi

    log "INFO" "Creating $DEMO_AGENT_COUNT demo agents..."

    # Wait a bit for backend to be fully ready
    sleep 5

    # Prepare agent creation command based on mode
    local agent_cmd_args=(
        --count "$DEMO_AGENT_COUNT"
        --base-url "http://localhost:$BACKEND_PORT"
        --batch-size 10
    )

    # Add mode-specific arguments
    case "$AGENT_MODE" in
        "spec-mode")
            agent_cmd_args+=(--mode spec --specification-based)
            log "INFO" "Running in specification mode - agents created from predefined specs"
            ;;
        "pseudo-mode")
            agent_cmd_args+=(--mode pseudo --simulation-only)
            log "INFO" "Running in pseudo mode - simulated agent creation without full backend"
            ;;
        "build-mode")
            agent_cmd_args+=(--mode build --progressive-build)
            log "INFO" "Running in build mode - incremental agent creation and testing"
            ;;
        "test-mode")
            agent_cmd_args+=(--mode test --validation-suite)
            log "INFO" "Running in test mode - agent creation with comprehensive validation"
            ;;
        "standard")
            log "INFO" "Running in standard mode - full agent creation"
            ;;
        *)
            log "WARN" "Unknown agent mode: $AGENT_MODE, using standard mode"
            ;;
    esac

    if python3 "$agent_script" "${agent_cmd_args[@]}"; then
        log "SUCCESS" "Demo agents created successfully in $AGENT_MODE mode"
        return 0
    else
        log "ERROR" "Failed to create demo agents in $AGENT_MODE mode"
        log "INFO" "You can create agents manually later with:"
        log "INFO" "  python3 $agent_script ${agent_cmd_args[*]}"
        return 1
    fi
}

# Function to start demo based on mode
start_demo_by_mode() {
    case "$DEMO_MODE" in
        "full")
            log "INFO" "Starting full demo environment (Redis + Backend + Frontend + Agents)"
            start_redis
            start_backend || { log "ERROR" "Failed to start backend"; cleanup; exit 1; }
            start_frontend || log "WARN" "Failed to start frontend - API will still be available"
            create_demo_agents
            ;;
        "minimal")
            log "INFO" "Starting minimal demo environment (Backend + Agents only)"
            start_backend || { log "ERROR" "Failed to start backend"; cleanup; exit 1; }
            create_demo_agents
            ;;
        "api-only")
            log "INFO" "Starting API-only demo environment (Backend only)"
            start_backend || { log "ERROR" "Failed to start backend"; cleanup; exit 1; }
            ;;
        "frontend-only")
            log "INFO" "Starting frontend-only demo environment"
            start_frontend || { log "ERROR" "Failed to start frontend"; cleanup; exit 1; }
            ;;
        *)
            log "ERROR" "Unknown demo mode: $DEMO_MODE"
            exit 1
            ;;
    esac
}

# Function to open browser
open_browser() {
    if [[ "$OPEN_BROWSER" != "true" ]]; then
        return 0
    fi

    log "INFO" "Opening browser..."

    local url="http://localhost:$FRONTEND_PORT"

    if command -v open >/dev/null 2>&1; then
        # macOS
        open "$url" >/dev/null 2>&1 &
    elif command -v xdg-open >/dev/null 2>&1; then
        # Linux
        xdg-open "$url" >/dev/null 2>&1 &
    elif command -v start >/dev/null 2>&1; then
        # Windows
        start "$url" >/dev/null 2>&1 &
    else
        log "INFO" "Could not auto-open browser. Please visit: $url"
        return 0
    fi

    log "SUCCESS" "Browser opened to $url"
}

# Function to display demo status
display_demo_status() {
    echo ""
    echo "========================================"
    echo "    CogniticNet Demo Environment"
    echo "========================================"
    echo ""

    log "SUCCESS" "Demo environment is running!"
    echo ""

    log "INFO" "Service URLs:"
    log "INFO" "  Frontend:  http://localhost:$FRONTEND_PORT"
    log "INFO" "  Backend:   http://localhost:$BACKEND_PORT"
    if [[ -n "$BACKEND_PID" ]]; then
        log "INFO" "  API Docs:  http://localhost:$BACKEND_PORT/docs"
    fi
    echo ""

    log "INFO" "Service Status:"
    if [[ -n "$REDIS_PID" ]] || redis-cli ping >/dev/null 2>&1; then
        log "SUCCESS" "  ✓ Redis server running"
    else
        log "ERROR" "  ✗ Redis server not running"
    fi

    if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        log "SUCCESS" "  ✓ Backend server running (PID: $BACKEND_PID)"
    else
        log "ERROR" "  ✗ Backend server not running"
    fi

    if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        log "SUCCESS" "  ✓ Frontend server running (PID: $FRONTEND_PID)"
    else
        log "WARN" "  ✗ Frontend server not running"
    fi

    echo ""
    log "INFO" "Demo Actions:"
    log "INFO" "  Create agents: python3 scripts/demo/create-demo-agents.py --count 20"
    log "INFO" "  View logs:     tail -f logs/demo-run.log"
    log "INFO" "  Stop demo:     Press Ctrl+C"
    echo ""
}

# Function to cleanup processes
cleanup() {
    if [[ "$CLEANUP_PERFORMED" == "true" ]]; then
        return
    fi

    CLEANUP_PERFORMED=true

    log "INFO" "Cleaning up demo environment..."

    # Stop frontend
    if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        log "INFO" "Stopping frontend server (PID: $FRONTEND_PID)..."
        kill "$FRONTEND_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$FRONTEND_PID" 2>/dev/null || true
    fi

    # Stop backend
    if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        log "INFO" "Stopping backend server (PID: $BACKEND_PID)..."
        kill "$BACKEND_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$BACKEND_PID" 2>/dev/null || true
    fi

    # Note: We don't stop Redis as it might be used by other processes

    log "SUCCESS" "Demo environment stopped"
    echo ""
}

# Function to wait for user interruption
wait_for_interruption() {
    log "INFO" "Demo is running. Press Ctrl+C to stop."
    echo ""

    # Wait for SIGINT (Ctrl+C) or SIGTERM
    trap 'cleanup; exit 0' SIGINT SIGTERM

    # Monitor processes and keep running
    while true; do
        # Check if critical processes are still running
        if [[ -n "$BACKEND_PID" ]] && ! kill -0 "$BACKEND_PID" 2>/dev/null; then
            log "ERROR" "Backend process died unexpectedly"
            break
        fi

        if [[ -n "$FRONTEND_PID" ]] && ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
            log "WARN" "Frontend process died unexpectedly"
        fi

        sleep 5
    done
}

# Main execution
main() {
    echo ""
    echo "========================================"
    echo "      CogniticNet Demo Launcher"
    echo "========================================"
    echo ""

    log "INFO" "Demo mode: $DEMO_MODE"
    log "INFO" "Agent mode: $AGENT_MODE"

    # Load configuration
    load_demo_config

    # Start services based on mode
    start_demo_by_mode

    # Open browser (if applicable)
    if [[ "$DEMO_MODE" == "full" || "$DEMO_MODE" == "frontend-only" ]]; then
        open_browser
    fi

    # Display status
    display_demo_status

    # Wait for user to stop
    wait_for_interruption

    # Cleanup
    cleanup
}

# Function to display help
show_help() {
    echo "CogniticNet Demo Environment Launcher"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h              Show this help message"
    echo "  --version, -v           Show version information"
    echo "  --no-agents            Skip automatic agent creation"
    echo "  --no-browser           Don't open browser automatically"
    echo "  --agent-count <count>  Number of agents to create (default: $DEMO_AGENT_COUNT)"
    echo "  --backend-port <port>  Backend port (default: $BACKEND_PORT)"
    echo "  --frontend-port <port> Frontend port (default: $FRONTEND_PORT)"
    echo ""
    echo "Demo Modes:"
    echo "  --mode <mode>          Demo execution mode:"
    echo "    full                 Complete environment (Redis + Backend + Frontend + Agents)"
    echo "    minimal              Backend and agents only"
    echo "    api-only             Backend API server only"
    echo "    frontend-only        Frontend application only"
    echo ""
    echo "Agent Command Modes:"
    echo "  --agent-mode <mode>    Agent creation mode:"
    echo "    standard             Standard agent creation (default)"
    echo "    spec-mode            Specification-based agent creation"
    echo "    pseudo-mode          Simulated agent creation (no backend required)"
    echo "    build-mode           Progressive build and test mode"
    echo "    test-mode            Comprehensive validation mode"
    echo ""
    echo "This script starts the CogniticNet demo environment according to the specified mode."
    echo "The demo will run until you press Ctrl+C."
    echo ""
}

# Handle script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        "--help"|"-h")
            show_help
            exit 0
            ;;
        "--version"|"-v")
            echo "CogniticNet Demo Environment Launcher v1.0.0"
            exit 0
            ;;
        "--no-agents")
            AUTO_CREATE_AGENTS=false
            log "INFO" "Automatic agent creation disabled"
            shift
            ;;
        "--no-browser")
            OPEN_BROWSER=false
            log "INFO" "Automatic browser opening disabled"
            shift
            ;;
        "--agent-count")
            DEMO_AGENT_COUNT="$2"
            log "INFO" "Demo agent count set to: $DEMO_AGENT_COUNT"
            shift 2
            ;;
        "--backend-port")
            BACKEND_PORT="$2"
            log "INFO" "Backend port set to: $BACKEND_PORT"
            shift 2
            ;;
        "--frontend-port")
            FRONTEND_PORT="$2"
            log "INFO" "Frontend port set to: $FRONTEND_PORT"
            shift 2
            ;;
        "--mode")
            DEMO_MODE="$2"
            log "INFO" "Demo mode set to: $DEMO_MODE"
            shift 2
            ;;
        "--agent-mode")
            AGENT_MODE="$2"
            log "INFO" "Agent mode set to: $AGENT_MODE"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Run main function
main
