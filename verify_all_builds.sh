#!/bin/bash

# BUILD-DOCTOR Agent #4 Build Verification Script
# This script verifies all build systems are working correctly

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== FreeAgentics Build System Verification ===${NC}"
echo -e "Date: $(date)"
echo -e "Working Directory: $(pwd)"
echo ""

# Track overall status
OVERALL_STATUS=0

# Function to check command result
check_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2 - SUCCESS${NC}"
    else
        echo -e "${RED}❌ $2 - FAILED${NC}"
        OVERALL_STATUS=1
    fi
}

# 1. Frontend Build Check
echo -e "${YELLOW}1. FRONTEND BUILD (npm run build)${NC}"
if [ -f "web/package.json" ]; then
    npm run build:frontend > build_frontend.log 2>&1
    check_status $? "Frontend build"

    # Check for build output
    if [ -d "web/.next" ]; then
        echo -e "${GREEN}   ✓ Next.js build output found in web/.next${NC}"
    else
        echo -e "${RED}   ✗ Next.js build output NOT found${NC}"
        OVERALL_STATUS=1
    fi
else
    echo -e "${RED}   ✗ web/package.json not found${NC}"
    OVERALL_STATUS=1
fi
echo ""

# 2. Backend Build Check
echo -e "${YELLOW}2. BACKEND BUILD (make build)${NC}"
make build > build_backend.log 2>&1
check_status $? "Backend build (syntax validation)"

# Additional Python compilation check
echo -e "   Checking Python compilation..."
python3 -m py_compile api/main.py 2>/dev/null
check_status $? "Python main.py compilation"
echo ""

# 3. Docker Build Check
echo -e "${YELLOW}3. DOCKER BUILD${NC}"
if command -v docker &> /dev/null && docker info &> /dev/null; then
    # Check if docker-compose is available
    if command -v docker-compose &> /dev/null; then
        echo -e "   Building backend-dev service..."
        timeout 300 docker-compose build backend-dev > build_docker.log 2>&1
        check_status $? "Docker backend-dev build"
    else
        echo -e "${YELLOW}   ⚠ docker-compose not available, trying docker build${NC}"
        timeout 300 docker build -t freeagentics-backend-dev --target development . > build_docker.log 2>&1
        check_status $? "Docker build"
    fi
else
    echo -e "${YELLOW}   ⚠ Docker not available or not running${NC}"
fi
echo ""

# 4. Build Script Analysis
echo -e "${YELLOW}4. BUILD SCRIPTS STATUS${NC}"
echo -e "   Package.json scripts:"
if [ -f "package.json" ]; then
    echo -e "   - build: $(grep -o '"build":.*' package.json | head -1)"
    echo -e "   - build:frontend: $(grep -o '"build:frontend":.*' package.json | head -1)"
fi

echo -e "   Makefile targets:"
if [ -f "Makefile" ]; then
    echo -e "   - build target: $(grep -E '^build:' Makefile | head -1)"
    echo -e "   - docker-build: $(grep -E '^docker-build:' Makefile | head -1 || echo 'NOT FOUND')"
fi
echo ""

# 5. Dependencies Check
echo -e "${YELLOW}5. DEPENDENCIES STATUS${NC}"
echo -e "   Frontend dependencies:"
if [ -d "web/node_modules" ]; then
    echo -e "${GREEN}   ✓ web/node_modules exists${NC}"
else
    echo -e "${RED}   ✗ web/node_modules missing - run npm install${NC}"
fi

echo -e "   Backend dependencies:"
if [ -d "venv" ]; then
    echo -e "${GREEN}   ✓ Python venv exists${NC}"
else
    echo -e "${YELLOW}   ⚠ Python venv not found${NC}"
fi
echo ""

# 6. Build Warnings/Errors Summary
echo -e "${YELLOW}6. BUILD WARNINGS/ERRORS SUMMARY${NC}"
if [ -f "build_frontend.log" ]; then
    echo -e "   Frontend warnings:"
    grep -i "warning" build_frontend.log | head -3 || echo "   No warnings found"
fi

if [ -f "build_docker.log" ]; then
    echo -e "   Docker warnings:"
    grep -i "warn" build_docker.log | head -3 || echo "   No warnings found"
fi
echo ""

# 7. Final Report
echo -e "${BLUE}=== BUILD VERIFICATION SUMMARY ===${NC}"
if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}✅ ALL BUILDS SUCCESSFUL!${NC}"
    echo -e "   - Frontend: WORKING"
    echo -e "   - Backend: WORKING"
    echo -e "   - Docker: WORKING"
else
    echo -e "${RED}❌ SOME BUILDS FAILED${NC}"
    echo -e "   Check the individual log files for details:"
    echo -e "   - build_frontend.log"
    echo -e "   - build_backend.log"
    echo -e "   - build_docker.log"
fi

echo ""
echo -e "Build verification completed at $(date)"

exit $OVERALL_STATUS
