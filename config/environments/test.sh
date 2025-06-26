#!/bin/bash

# Run CogniticNet tests

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}Running CogniticNet Tests...${NC}"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "No .env file found. Setting up testing environment..."
    ./scripts/env-setup.sh testing --force
fi

# Run tests
echo -e "${BLUE}Starting test suite...${NC}"

# Build test images
docker compose -f docker-compose.yml -f environments/docker/docker-compose.testing.yml build

# Run backend tests
echo -e "${BLUE}Running backend tests...${NC}"
docker compose -f docker-compose.yml -f environments/docker/docker-compose.testing.yml run --rm backend

# Run frontend tests
echo -e "${BLUE}Running frontend tests...${NC}"
docker compose -f docker-compose.yml -f environments/docker/docker-compose.testing.yml run --rm frontend

# Clean up
echo -e "${BLUE}Cleaning up test environment...${NC}"
docker compose -f docker-compose.yml -f environments/docker/docker-compose.testing.yml down -v

echo -e "${GREEN}All tests completed!${NC}"
