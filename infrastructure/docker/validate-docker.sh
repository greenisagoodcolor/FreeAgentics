#!/bin/bash
# FreeAgentics Docker Validation Script
# Comprehensive validation of Docker infrastructure

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo -e "${MAGENTA}🚀 FreeAgentics Docker Validation${NC}"
echo -e "${CYAN}Project Root: ${PROJECT_ROOT}${NC}"
echo ""

# Function to check command existence
check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ $1 is available${NC}"
        return 0
    else
        echo -e "${RED}❌ $1 is not available${NC}"
        return 1
    fi
}

# Function to validate file existence
check_file() {
    if [[ -f "$1" ]]; then
        echo -e "${GREEN}✅ $(basename "$1") exists${NC}"
        return 0
    else
        echo -e "${RED}❌ $(basename "$1") missing${NC}"
        return 1
    fi
}

# Function to validate Docker Compose syntax
validate_compose() {
    echo -e "${BLUE}🔍 Validating Docker Compose syntax...${NC}"
    
    if docker-compose -f "${SCRIPT_DIR}/docker-compose.yml" config > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Docker Compose syntax is valid${NC}"
        return 0
    else
        echo -e "${RED}❌ Docker Compose syntax error${NC}"
        docker-compose -f "${SCRIPT_DIR}/docker-compose.yml" config
        return 1
    fi
}

# Function to validate Dockerfile syntax
validate_dockerfile() {
    local dockerfile="$1"
    local service_name="$2"
    
    echo -e "${BLUE}🔍 Validating ${service_name} Dockerfile...${NC}"
    
    if docker build -f "${dockerfile}" --target production -t "freeagentics-${service_name}:validate" "${PROJECT_ROOT}" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ ${service_name} Dockerfile builds successfully${NC}"
        # Clean up test image
        docker rmi "freeagentics-${service_name}:validate" > /dev/null 2>&1 || true
        return 0
    else
        echo -e "${RED}❌ ${service_name} Dockerfile build failed${NC}"
        return 1
    fi
}

# Function to test container orchestration
test_orchestration() {
    echo -e "${BLUE}🚀 Testing container orchestration...${NC}"
    
    # Create .env file if it doesn't exist
    if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
        cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
        echo -e "${YELLOW}⚠️  Created .env from .env.example - please configure passwords${NC}"
    fi
    
    # Start services
    if docker-compose -f "${SCRIPT_DIR}/docker-compose.yml" up -d; then
        echo -e "${GREEN}✅ Services started successfully${NC}"
        
        # Wait for health checks
        echo -e "${BLUE}⏳ Waiting for services to be healthy...${NC}"
        sleep 30
        
        # Check service health
        local all_healthy=true
        for service in postgres redis api web; do
            if docker-compose -f "${SCRIPT_DIR}/docker-compose.yml" ps | grep -q "${service}.*healthy"; then
                echo -e "${GREEN}✅ ${service} is healthy${NC}"
            else
                echo -e "${RED}❌ ${service} is not healthy${NC}"
                all_healthy=false
            fi
        done
        
        # Clean up
        docker-compose -f "${SCRIPT_DIR}/docker-compose.yml" down
        
        if $all_healthy; then
            echo -e "${GREEN}🎉 All services are healthy!${NC}"
            return 0
        else
            echo -e "${RED}💥 Some services failed health checks${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Failed to start services${NC}"
        return 1
    fi
}

# Main validation
main() {
    local exit_code=0
    
    echo -e "${BLUE}📋 Checking prerequisites...${NC}"
    check_command "docker" || exit_code=1
    check_command "docker-compose" || exit_code=1
    
    echo ""
    echo -e "${BLUE}📁 Checking Docker files...${NC}"
    check_file "${SCRIPT_DIR}/docker-compose.yml" || exit_code=1
    check_file "${SCRIPT_DIR}/Dockerfile.api" || exit_code=1
    check_file "${SCRIPT_DIR}/Dockerfile.web" || exit_code=1
    check_file "${PROJECT_ROOT}/.dockerignore" || exit_code=1
    check_file "${PROJECT_ROOT}/.env.example" || exit_code=1
    
    echo ""
    validate_compose || exit_code=1
    
    # Check if Docker daemon is running
    if docker info > /dev/null 2>&1; then
        echo ""
        echo -e "${BLUE}🔨 Testing Docker builds...${NC}"
        validate_dockerfile "${SCRIPT_DIR}/Dockerfile.api" "api" || exit_code=1
        validate_dockerfile "${SCRIPT_DIR}/Dockerfile.web" "web" || exit_code=1
        
        echo ""
        echo -e "${BLUE}🧪 Testing full orchestration...${NC}"
        test_orchestration || exit_code=1
    else
        echo -e "${YELLOW}⚠️  Docker daemon not running - skipping build tests${NC}"
    fi
    
    echo ""
    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}🎉 All Docker validation checks passed!${NC}"
        echo -e "${CYAN}Ready for production deployment with: make docker${NC}"
    else
        echo -e "${RED}💥 Some validation checks failed${NC}"
        echo -e "${YELLOW}Please fix the issues above before deploying${NC}"
    fi
    
    return $exit_code
}

# Run main function
main "$@"