#!/bin/bash
# Quick load test script for FreeAgentics database

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=== FreeAgentics Database Load Test ==="
echo ""

# Check if PostgreSQL is running
if ! docker ps | grep -q freeagentics-postgres; then
    echo "ERROR: PostgreSQL container 'freeagentics-postgres' is not running!"
    echo "Please start it with: docker-compose up -d postgres"
    exit 1
fi

echo "✓ PostgreSQL container is running"
echo ""

# Run infrastructure test first
echo "1. Testing infrastructure..."
python -m tests.db_infrastructure.test_infrastructure
if [ $? -ne 0 ]; then
    echo "ERROR: Infrastructure test failed!"
    exit 1
fi

echo ""
echo "2. Running quick load test (10 seconds, 5 threads)..."
python -m tests.db_infrastructure.load_test --test quick

echo ""
echo "Load test completed!"
echo ""
echo "For more extensive testing, try:"
echo "  - Full load test:   python -m tests.db_infrastructure.load_test --test load --duration 60 --threads 20"
echo "  - Stress test:      python -m tests.db_infrastructure.load_test --test stress --threads 50"
echo "  - View table counts: python -m tests.db_infrastructure.db_reset counts"
