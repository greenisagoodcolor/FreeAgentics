#!/bin/bash
#
# Database setup script for CogniticNet
#
# This script helps set up the PostgreSQL database for development
# It can be run locally or in Docker environments

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-cogniticnet}"
DB_PASSWORD="${DB_PASSWORD:-dev_password}"
DB_NAME="${DB_NAME:-cogniticnet_dev}"
ENVIRONMENT="${ENVIRONMENT:-development}"

echo -e "${GREEN}FreeAgentics Database Setup${NC}"
echo "=============================="
echo ""

# Function to check if PostgreSQL is running
check_postgres() {
    echo -n "Checking PostgreSQL connection... "
    if pg_isready -h "$DB_HOST" -p "$DB_PORT" -q; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        echo -e "${RED}PostgreSQL is not running or not accessible${NC}"
        echo ""
        echo "Options:"
        echo "1. Start PostgreSQL locally:"
        echo "   - macOS: brew services start postgresql"
        echo "   - Linux: sudo systemctl start postgresql"
        echo ""
        echo "2. Use Docker:"
        echo "   docker-compose -f environments/development/docker-compose.yml up -d postgres"
        echo ""
        return 1
    fi
}

# Function to create database user and database
setup_database() {
    echo "Setting up database..."

    # Try to create user (might already exist)
    echo -n "Creating user '$DB_USER'... "
    if PGPASSWORD=postgres psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';" 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}User already exists${NC}"
    fi

    # Create database
    echo -n "Creating database '$DB_NAME'... "
    if PGPASSWORD=postgres psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;" 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}Database already exists${NC}"
    fi

    # Grant privileges
    echo -n "Granting privileges... "
    PGPASSWORD=postgres psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;" 2>/dev/null
    echo -e "${GREEN}✓${NC}"
}

# Function to run migrations
run_migrations() {
    echo ""
    echo "Running database migrations..."

    export DATABASE_URL="postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"

    # Check if alembic is available and migrations exist
    if [ -f "alembic.ini" ] && [ -d "alembic" ]; then
        echo -n "Running Alembic migrations... "
        alembic upgrade head
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}No migrations found - skipping${NC}"
    fi
}

# Function to seed data
seed_data() {
    echo ""
    echo "Seeding database with $ENVIRONMENT data..."

    export DATABASE_URL="postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"

    # Check if seeding script exists
    if [ -f "infrastructure/scripts/seed-data.py" ]; then
        echo -n "Seeding database... "
        python infrastructure/scripts/seed-data.py
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}No seed data script found - skipping${NC}"
    fi
}

# Main execution
main() {
    # Check if we're in the project root
    if [ ! -f "pyproject.toml" ] || [ ! -d "web" ]; then
        echo -e "${RED}Error: This script must be run from the FreeAgentics project root${NC}"
        exit 1
    fi

    # Check PostgreSQL
    if ! check_postgres; then
        exit 1
    fi

    # Parse command line arguments
    case "${1:-all}" in
        "check")
            echo "PostgreSQL is accessible"
            ;;
        "create")
            setup_database
            ;;
        "migrate")
            run_migrations
            ;;
        "seed")
            seed_data
            ;;
        "all"|"")
            setup_database
            run_migrations
            if [ "${DATABASE_SEED_DATA:-true}" = "true" ]; then
                seed_data
            fi
            echo ""
            echo -e "${GREEN}Database setup complete!${NC}"
            ;;
        *)
            echo "Usage: $0 [check|create|migrate|seed|all]"
            echo ""
            echo "Commands:"
            echo "  check   - Check PostgreSQL connection"
            echo "  create  - Create database and user"
            echo "  migrate - Run database migrations"
            echo "  seed    - Seed database with test data"
            echo "  all     - Run all steps (default)"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
