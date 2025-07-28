#!/usr/bin/env bash
# FreeAgentics Quick Demo Script
# Minimal setup for immediate demonstration

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 FreeAgentics Quick Demo Setup${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Python
if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}❌ Python 3 is required${NC}"
    exit 1
fi

# Create minimal virtual environment
if [ ! -d "demo_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv demo_venv
fi

# Activate venv
source demo_venv/bin/activate 2>/dev/null || source demo_venv/Scripts/activate 2>/dev/null

# Install minimal dependencies
echo "Installing core dependencies..."
pip install --quiet --upgrade pip
pip install --quiet fastapi uvicorn pydantic sqlalchemy aiosqlite httpx

# Create minimal .env
if [ ! -f ".env" ]; then
    cat > .env << EOF
DATABASE_URL=sqlite+aiosqlite:///demo.db
SECRET_KEY=demo-secret-key
DEBUG=true
LOG_LEVEL=INFO
PYMDP_ENABLED=false
DISABLE_REDIS=true
EOF
fi

# Start API with demo mode
echo -e "${GREEN}Starting API server...${NC}"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait for API to start
sleep 5

# Create demo agents using the API
echo -e "${YELLOW}Creating demo agents...${NC}"

# Function to create an agent
create_agent() {
    local name=$1
    local type=$2
    curl -s -X POST http://localhost:8000/api/v1/agents \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"$name\",\"type\":\"$type\",\"config\":{\"grid_size\":10}}" \
        >/dev/null 2>&1 || true
}

# Create several demo agents
create_agent "Explorer Alpha" "explorer"
create_agent "Collector Beta" "collector" 
create_agent "Analyzer Gamma" "analyzer"
create_agent "Scout Delta" "explorer"

echo -e "${GREEN}✅ Demo setup complete!${NC}"
echo ""
echo -e "${BLUE}📋 Try these commands:${NC}"
echo ""
echo "List all agents:"
echo -e "${YELLOW}curl http://localhost:8000/api/v1/agents${NC}"
echo ""
echo "Get agent details:"
echo -e "${YELLOW}curl http://localhost:8000/api/v1/agents/1${NC}"
echo ""
echo "Make agent take action:"
echo -e "${YELLOW}curl -X POST http://localhost:8000/api/v1/agents/1/act${NC}"
echo ""
echo "View API documentation:"
echo -e "${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "${RED}Press Ctrl+C to stop the demo${NC}"

# Keep running
wait $API_PID