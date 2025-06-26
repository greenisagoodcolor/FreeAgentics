#!/bin/bash

# Start CogniticNet in local development mode

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Starting CogniticNet Local Development Environment...${NC}"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "No .env file found. Setting up local environment..."
    ./scripts/env-setup.sh local
fi

# Start services
echo -e "${BLUE}Starting Docker services...${NC}"
docker compose -f docker-compose.yml -f environments/docker/docker-compose.local.yml up

echo -e "${GREEN}Development environment stopped.${NC}"
