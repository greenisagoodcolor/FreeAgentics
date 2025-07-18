#!/bin/bash
#
# PostgreSQL Setup Script for FreeAgentics
# This script helps set up PostgreSQL for production use
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
DB_NAME="freeagentics"
DB_USER="freeagentics"
DB_HOST="localhost"
DB_PORT="5432"

echo -e "${GREEN}=== FreeAgentics PostgreSQL Setup ===${NC}"

# Function to generate secure password
generate_password() {
    openssl rand -hex 32
}

# Function to check if PostgreSQL is installed
check_postgres_installed() {
    if command -v psql &> /dev/null; then
        echo -e "${GREEN}✓ PostgreSQL is installed${NC}"
        psql --version
        return 0
    else
        echo -e "${RED}✗ PostgreSQL is not installed${NC}"
        return 1
    fi
}

# Function to check if PostgreSQL is running
check_postgres_running() {
    if pg_isready -h "$DB_HOST" -p "$DB_PORT" &> /dev/null; then
        echo -e "${GREEN}✓ PostgreSQL is running${NC}"
        return 0
    else
        echo -e "${RED}✗ PostgreSQL is not running${NC}"
        return 1
    fi
}

# Function to create database and user
setup_database() {
    local db_password="$1"

    echo -e "${YELLOW}Creating database and user...${NC}"

    # Create SQL script
    cat > /tmp/setup_freeagentics.sql <<EOF
-- Create user if not exists
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = '${DB_USER}') THEN
        CREATE USER ${DB_USER} WITH PASSWORD '${db_password}';
    END IF;
END
\$\$;

-- Create database if not exists
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DB_NAME}') THEN
        CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};
    END IF;
END
\$\$;

-- Grant all privileges
GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};

-- Connect to the database and set up extensions
\c ${DB_NAME}

-- Create extensions if they don't exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
EOF

    # Execute SQL script
    sudo -u postgres psql < /tmp/setup_freeagentics.sql

    # Clean up
    rm -f /tmp/setup_freeagentics.sql

    echo -e "${GREEN}✓ Database and user created successfully${NC}"
}

# Function to create .env.production file
create_env_file() {
    local db_password="$1"
    local redis_password="$2"
    local secret_key="$3"
    local jwt_secret="$4"

    cat > "$PROJECT_ROOT/.env.production" <<EOF
# Production Environment Configuration
# Generated on $(date)

# PostgreSQL Database Configuration
DATABASE_URL=postgresql://${DB_USER}:${db_password}@${DB_HOST}:${DB_PORT}/${DB_NAME}

# For Docker Compose deployment
POSTGRES_PASSWORD=${db_password}
POSTGRES_USER=${DB_USER}
POSTGRES_DB=${DB_NAME}

# Redis Cache Configuration
REDIS_URL=redis://:${redis_password}@localhost:6379/0
REDIS_PASSWORD=${redis_password}

# Application Security Keys
SECRET_KEY=${secret_key}
JWT_SECRET=${jwt_secret}

# JWT Token Expiration
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Application Environment
ENVIRONMENT=production
PRODUCTION=true
LOG_LEVEL=INFO
DEBUG=false
DEBUG_SQL=false

# Connection Pool Settings
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=10
DB_POOL_RECYCLE=1800
EOF

    chmod 600 "$PROJECT_ROOT/.env.production"
    echo -e "${GREEN}✓ Created .env.production file${NC}"
}

# Function to test database connection
test_connection() {
    local db_password="$1"

    echo -e "${YELLOW}Testing database connection...${NC}"

    if PGPASSWORD="$db_password" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT version();" &> /dev/null; then
        echo -e "${GREEN}✓ Database connection successful${NC}"
        return 0
    else
        echo -e "${RED}✗ Database connection failed${NC}"
        return 1
    fi
}

# Main setup flow
main() {
    # Check if running with Docker
    if [ "${DOCKER_SETUP:-false}" = "true" ]; then
        echo -e "${YELLOW}Docker setup mode - skipping local PostgreSQL checks${NC}"
    else
        # Check PostgreSQL installation
        if ! check_postgres_installed; then
            echo -e "${YELLOW}Please install PostgreSQL first:${NC}"
            echo "Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
            echo "RHEL/CentOS: sudo yum install postgresql postgresql-server postgresql-contrib"
            echo "macOS: brew install postgresql"
            exit 1
        fi

        # Check if PostgreSQL is running
        if ! check_postgres_running; then
            echo -e "${YELLOW}Starting PostgreSQL...${NC}"
            if command -v systemctl &> /dev/null; then
                sudo systemctl start postgresql
            else
                echo -e "${RED}Please start PostgreSQL manually${NC}"
                exit 1
            fi
        fi
    fi

    # Generate secure passwords
    echo -e "${YELLOW}Generating secure passwords...${NC}"
    DB_PASSWORD=$(generate_password)
    REDIS_PASSWORD=$(generate_password)
    SECRET_KEY=$(generate_password)
    JWT_SECRET=$(generate_password)

    # Create secrets directory
    mkdir -p "$PROJECT_ROOT/secrets"
    chmod 700 "$PROJECT_ROOT/secrets"

    # Save passwords to secrets directory
    echo "$DB_PASSWORD" > "$PROJECT_ROOT/secrets/postgres_password.txt"
    echo "$REDIS_PASSWORD" > "$PROJECT_ROOT/secrets/redis_password.txt"
    echo "$SECRET_KEY" > "$PROJECT_ROOT/secrets/secret_key.txt"
    echo "$JWT_SECRET" > "$PROJECT_ROOT/secrets/jwt_secret.txt"
    chmod 600 "$PROJECT_ROOT/secrets"/*.txt

    # Set up database (skip for Docker)
    if [ "${DOCKER_SETUP:-false}" != "true" ]; then
        setup_database "$DB_PASSWORD"

        # Test connection
        if ! test_connection "$DB_PASSWORD"; then
            echo -e "${RED}Setup completed but connection test failed${NC}"
            echo "Please check PostgreSQL logs for details"
            exit 1
        fi
    fi

    # Create .env.production file
    create_env_file "$DB_PASSWORD" "$REDIS_PASSWORD" "$SECRET_KEY" "$JWT_SECRET"

    echo -e "${GREEN}=== PostgreSQL Setup Complete ===${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run migrations: cd $PROJECT_ROOT && alembic upgrade head"
    echo "2. Load seed data: cd $PROJECT_ROOT && python scripts/seed_database.py"
    echo "3. Test the setup: cd $PROJECT_ROOT && python scripts/test_database_connection.py"
    echo ""
    echo "For Docker deployment:"
    echo "docker-compose --profile prod up -d"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            DOCKER_SETUP=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --docker    Setup for Docker deployment (skip local PostgreSQL)"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main setup
main
