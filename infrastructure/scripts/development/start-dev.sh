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
# Load environment variables, filtering out comments and empty lines
set -a
source config/environments/.env.development
set +a
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
