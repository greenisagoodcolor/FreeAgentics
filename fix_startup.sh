#!/bin/bash
# Quick fix script for startup issues

echo "🔧 Fixing startup issues..."

# Ensure we're in the project directory
cd "$(dirname "$0")"

# Create __init__.py files if missing
touch core/__init__.py
touch auth/__init__.py
touch api/__init__.py
touch api/v1/__init__.py

echo "✅ Created missing __init__.py files"

# Set PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "✅ Set PYTHONPATH to include project root"

# Kill any existing processes on ports
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:3001 | xargs kill -9 2>/dev/null || true
lsof -ti:3002 | xargs kill -9 2>/dev/null || true
lsof -ti:3003 | xargs kill -9 2>/dev/null || true
echo "✅ Cleared port conflicts"

echo ""
echo "🚀 Now run: make dev"
echo ""
echo "Or manually with venv:"
echo "  source venv/bin/activate"
echo "  make dev"