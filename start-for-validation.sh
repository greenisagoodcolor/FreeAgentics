#!/bin/bash

# Start application with production environment for validation
set -a  # Automatically export all variables
source .env.production
set +a  # Disable automatic export

# Start the application
python3 main.py &
APP_PID=$!

echo "Application started with PID: $APP_PID"
echo "Waiting for application to start..."

# Wait for application to be ready
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Application is ready!"
        exit 0
    fi
    sleep 1
done

echo "Application failed to start within 30 seconds"
kill $APP_PID 2>/dev/null
exit 1
