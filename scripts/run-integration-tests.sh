#!/bin/bash
# Run integration tests with test containers
# This script provides nemesis-level testing infrastructure

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.test.yml"
TEST_TIMEOUT=300
CLEANUP=true
VERBOSE=false
COVERAGE=true
PARALLEL=false
SPECIFIC_TEST=""

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -t, --timeout SEC    Test timeout in seconds (default: 300)"
    echo "  -n, --no-cleanup     Don't cleanup containers after tests"
    echo "  -v, --verbose        Verbose output"
    echo "  -c, --no-coverage    Disable coverage reporting"
    echo "  -p, --parallel       Run tests in parallel"
    echo "  -s, --specific TEST  Run specific test file or pattern"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all integration tests"
    echo "  $0 --parallel                         # Run tests in parallel"
    echo "  $0 --specific test_gnn_llm            # Run specific test pattern"
    echo "  $0 --no-cleanup --verbose             # Keep containers running with verbose output"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -t|--timeout)
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        -n|--no-cleanup)
            CLEANUP=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--no-coverage)
            COVERAGE=false
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -s|--specific)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Function to cleanup containers
cleanup_containers() {
    echo -e "\n${YELLOW}Cleaning up test containers...${NC}"
    docker-compose -f "$COMPOSE_FILE" down -v --remove-orphans
}

# Function to check if services are healthy
check_services_health() {
    local max_attempts=30
    local attempt=0

    echo -e "${BLUE}Waiting for services to be healthy...${NC}"

    while [ $attempt -lt $max_attempts ]; do
        local all_healthy=true

        # Check each service
        for service in test-postgres test-redis test-rabbitmq test-elasticsearch test-minio; do
            if ! docker-compose -f "$COMPOSE_FILE" ps | grep -q "${service}.*healthy"; then
                all_healthy=false
                break
            fi
        done

        if [ "$all_healthy" = true ]; then
            echo -e "${GREEN}All services are healthy!${NC}"
            return 0
        fi

        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done

    echo -e "\n${RED}Services failed to become healthy in time${NC}"
    return 1
}

# Function to create test database schema
setup_test_database() {
    echo -e "${BLUE}Setting up test database schema...${NC}"

    docker-compose -f "$COMPOSE_FILE" exec -T test-postgres psql -U test_user -d freeagentics_test <<EOF
-- Create necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- Create test schema
CREATE SCHEMA IF NOT EXISTS test;

-- Grant permissions
GRANT ALL ON SCHEMA test TO test_user;
EOF

    echo -e "${GREEN}Database schema setup complete${NC}"
}

# Function to run integration tests
run_tests() {
    echo -e "\n${BLUE}Running integration tests...${NC}"

    # Source test environment if it exists
    if [ -f "${PROJECT_ROOT}/.env.test" ]; then
        echo -e "${BLUE}Loading test environment from .env.test...${NC}"
        set -a
        source "${PROJECT_ROOT}/.env.test"
        set +a
    fi

    # Build pytest command
    local pytest_cmd="pytest tests/integration/"

    # Add verbose flag
    if [ "$VERBOSE" = true ]; then
        pytest_cmd="$pytest_cmd -vv"
    else
        pytest_cmd="$pytest_cmd -v"
    fi

    # Add coverage flags
    if [ "$COVERAGE" = true ]; then
        pytest_cmd="$pytest_cmd --cov=. --cov-report=html --cov-report=term"
    fi

    # Add parallel flag
    if [ "$PARALLEL" = true ]; then
        pytest_cmd="$pytest_cmd -n auto"
    fi

    # Add timeout
    pytest_cmd="$pytest_cmd --timeout=$TEST_TIMEOUT"

    # Add specific test if provided
    if [ -n "$SPECIFIC_TEST" ]; then
        pytest_cmd="$pytest_cmd -k $SPECIFIC_TEST"
    fi

    # Run tests in container with environment
    docker-compose -f "$COMPOSE_FILE" run --rm \
        --env-file "${PROJECT_ROOT}/.env.test" \
        test-runner $pytest_cmd
}

# Main execution
main() {
    echo -e "${BLUE}FreeAgentics Integration Test Runner${NC}"
    echo -e "${BLUE}=====================================${NC}\n"

    # Change to project root
    cd "$PROJECT_ROOT"

    # Trap to ensure cleanup on exit
    if [ "$CLEANUP" = true ]; then
        trap cleanup_containers EXIT
    fi

    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running${NC}"
        exit 1
    fi

    # Check if docker-compose is available
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Error: docker-compose is not installed${NC}"
        exit 1
    fi

    # Stop any existing test containers
    echo -e "${YELLOW}Stopping any existing test containers...${NC}"
    docker-compose -f "$COMPOSE_FILE" down -v --remove-orphans 2>/dev/null || true

    # Start test containers
    echo -e "\n${BLUE}Starting test containers...${NC}"
    docker-compose -f "$COMPOSE_FILE" up -d --build

    # Wait for services to be healthy
    if ! check_services_health; then
        echo -e "${RED}Failed to start services${NC}"
        docker-compose -f "$COMPOSE_FILE" logs
        exit 1
    fi

    # Setup test database
    setup_test_database

    # Run the tests
    if run_tests; then
        echo -e "\n${GREEN}✓ Integration tests passed!${NC}"

        if [ "$COVERAGE" = true ]; then
            echo -e "\n${BLUE}Coverage report available at: htmlcov/index.html${NC}"
        fi

        exit 0
    else
        echo -e "\n${RED}✗ Integration tests failed!${NC}"

        if [ "$CLEANUP" = false ]; then
            echo -e "\n${YELLOW}Containers are still running. To view logs:${NC}"
            echo "  docker-compose -f $COMPOSE_FILE logs"
            echo -e "\n${YELLOW}To stop containers manually:${NC}"
            echo "  docker-compose -f $COMPOSE_FILE down -v"
        fi

        exit 1
    fi
}

# Run main function
main
