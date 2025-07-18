#!/bin/bash
#
# PostgreSQL Quick Start Script for FreeAgentics
# This script provides a quick way to get PostgreSQL up and running
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== FreeAgentics PostgreSQL Quick Start ===${NC}"
echo ""

# Function to check Docker
check_docker() {
    if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        echo -e "${GREEN}✓ Docker and Docker Compose are installed${NC}"
        return 0
    else
        echo -e "${RED}✗ Docker or Docker Compose not found${NC}"
        return 1
    fi
}

# Function to check existing PostgreSQL
check_existing_postgres() {
    if docker-compose ps postgres 2>/dev/null | grep -q "Up"; then
        echo -e "${YELLOW}⚠ PostgreSQL container is already running${NC}"
        return 0
    fi
    return 1
}

# Function to setup with Docker
setup_docker() {
    echo -e "${YELLOW}Setting up PostgreSQL with Docker...${NC}"

    # Check if .env.production exists
    if [ ! -f "$PROJECT_ROOT/.env.production" ]; then
        echo -e "${YELLOW}Creating .env.production file...${NC}"
        ./setup-postgresql.sh --docker
    fi

    # Start PostgreSQL container
    echo -e "${YELLOW}Starting PostgreSQL container...${NC}"
    cd "$PROJECT_ROOT"
    docker-compose up -d postgres

    # Wait for PostgreSQL to be ready
    echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if docker-compose exec -T postgres pg_isready -U freeagentics &>/dev/null; then
            echo -e "${GREEN}✓ PostgreSQL is ready${NC}"
            break
        fi
        echo -n "."
        sleep 1
        ((attempt++))
    done

    if [ $attempt -gt $max_attempts ]; then
        echo -e "${RED}✗ PostgreSQL failed to start${NC}"
        docker-compose logs postgres
        exit 1
    fi
}

# Function to run migrations
run_migrations() {
    echo -e "${YELLOW}Running database migrations...${NC}"

    cd "$PROJECT_ROOT"

    # Check if we're using Docker
    if docker-compose ps postgres 2>/dev/null | grep -q "Up"; then
        # Run migrations in Docker
        docker-compose run --rm migration
    else
        # Run migrations locally
        source .env.production
        alembic upgrade head
    fi

    echo -e "${GREEN}✓ Migrations completed${NC}"
}

# Function to apply indexes
apply_indexes() {
    echo -e "${YELLOW}Applying database indexes...${NC}"

    cd "$PROJECT_ROOT"

    if docker-compose ps postgres 2>/dev/null | grep -q "Up"; then
        # Run in Docker
        docker-compose exec -T backend python scripts/apply_database_indexes.py
    else
        # Run locally
        python scripts/apply_database_indexes.py
    fi

    echo -e "${GREEN}✓ Indexes applied${NC}"
}

# Function to load seed data
load_seed_data() {
    read -p "Load seed data? (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Loading seed data...${NC}"

        cd "$PROJECT_ROOT"

        if docker-compose ps postgres 2>/dev/null | grep -q "Up"; then
            # Run in Docker
            docker-compose exec -T backend python scripts/seed_database.py
        else
            # Run locally
            python scripts/seed_database.py
        fi

        echo -e "${GREEN}✓ Seed data loaded${NC}"
    fi
}

# Function to test connection
test_connection() {
    echo -e "${YELLOW}Testing database connection...${NC}"

    cd "$PROJECT_ROOT"

    if docker-compose ps postgres 2>/dev/null | grep -q "Up"; then
        # Test in Docker
        docker-compose exec -T backend python scripts/test_database_connection.py
    else
        # Test locally
        python scripts/test_database_connection.py
    fi
}

# Function to show connection info
show_connection_info() {
    echo ""
    echo -e "${BLUE}=== Connection Information ===${NC}"

    if [ -f "$PROJECT_ROOT/.env.production" ]; then
        source "$PROJECT_ROOT/.env.production"
        echo -e "Database URL: ${GREEN}$DATABASE_URL${NC}"
        echo ""
        echo "To connect with psql:"
        echo -e "${YELLOW}psql $DATABASE_URL${NC}"
        echo ""
        echo "To connect from application:"
        echo -e "${YELLOW}source .env.production${NC}"
    fi

    if docker-compose ps postgres 2>/dev/null | grep -q "Up"; then
        echo ""
        echo "Docker commands:"
        echo -e "${YELLOW}docker-compose logs -f postgres${NC}     # View logs"
        echo -e "${YELLOW}docker-compose exec postgres psql -U freeagentics${NC}  # Connect to DB"
        echo -e "${YELLOW}docker-compose stop postgres${NC}        # Stop PostgreSQL"
        echo -e "${YELLOW}docker-compose down${NC}                 # Stop and remove containers"
    fi
}

# Main flow
main() {
    cd "$PROJECT_ROOT"

    # Check if Docker is available
    if check_docker; then
        if ! check_existing_postgres; then
            setup_docker
        else
            echo -e "${GREEN}Using existing PostgreSQL container${NC}"
        fi

        # Run setup steps
        run_migrations
        apply_indexes
        load_seed_data
        test_connection

    else
        echo ""
        echo -e "${YELLOW}Docker not found. Please install Docker or set up PostgreSQL manually.${NC}"
        echo ""
        echo "Manual setup instructions:"
        echo "1. Install PostgreSQL locally"
        echo "2. Run: ./scripts/setup-postgresql.sh"
        echo "3. Run: alembic upgrade head"
        echo "4. Run: python scripts/apply_database_indexes.py"
        echo ""
        exit 1
    fi

    # Show connection info
    show_connection_info

    echo ""
    echo -e "${GREEN}=== PostgreSQL Setup Complete! ===${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Start the application: docker-compose up -d"
    echo "2. View logs: docker-compose logs -f"
    echo "3. Access API: http://localhost:8000"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Quick start script for PostgreSQL setup"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --reset        Reset database (WARNING: destroys all data)"
        echo "  --status       Show current database status"
        exit 0
        ;;
    --reset)
        echo -e "${RED}WARNING: This will destroy all data!${NC}"
        read -p "Are you sure? (yes/no): " response
        if [ "$response" = "yes" ]; then
            docker-compose down -v
            rm -f "$PROJECT_ROOT/.env.production"
            echo -e "${YELLOW}Database reset. Run script again to set up fresh.${NC}"
        fi
        exit 0
        ;;
    --status)
        if docker-compose ps postgres 2>/dev/null | grep -q "Up"; then
            echo -e "${GREEN}PostgreSQL is running${NC}"
            docker-compose exec postgres pg_isready -U freeagentics
            docker-compose exec postgres psql -U freeagentics -c "SELECT version();"
        else
            echo -e "${RED}PostgreSQL is not running${NC}"
        fi
        exit 0
        ;;
esac

# Run main setup
main
