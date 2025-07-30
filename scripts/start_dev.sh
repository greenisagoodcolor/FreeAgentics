#!/bin/bash
# Start development servers in background

echo "🚀 Starting FreeAgentics Development Environment..."

# Kill any existing processes
echo "Cleaning up old processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
sleep 2

# Start backend
echo "Starting backend..."
cd /home/green/FreeAgentics
venv/bin/python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Start frontend
echo "Starting frontend..."
cd /home/green/FreeAgentics/web
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for services
echo "Waiting for services to start..."
sleep 5

# Check status
cd /home/green/FreeAgentics
venv/bin/python scripts/verify_setup.py

echo ""
echo "To view logs:"
echo "  Backend:  tail -f backend.log"
echo "  Frontend: tail -f frontend.log"
echo ""
echo "To stop all services: make stop"