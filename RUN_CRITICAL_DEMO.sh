#!/bin/bash
# 🚨 CRITICAL BUSINESS DEMO LAUNCHER 🚨
# This demonstrates all 3 features needed to save the company

echo "========================================"
echo "🌟 FREEAGENTICS CRITICAL FEATURES DEMO 🌟"
echo "========================================"
echo ""
echo "This will demonstrate:"
echo "1. ✅ GMN Parser - Converting specifications to PyMDP models"
echo "2. ✅ Knowledge Graph Backend - Storing agent knowledge"
echo "3. ✅ End-to-End Pipeline - Full integration working"
echo ""
echo "Starting demo..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the critical demo
python examples/demo_full_pipeline.py

echo ""
echo "========================================"
echo "✅ Demo complete! All features working!"
echo "========================================"