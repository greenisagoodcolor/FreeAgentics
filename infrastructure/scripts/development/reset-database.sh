#!/bin/bash
#
# FreeAgentics Database Reset Script
#
# This script completely resets the development database

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Default values
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-freeagentics}"
DB_PASSWORD="${DB_PASSWORD:-dev_password}"
DB_NAME="${DB_NAME:-freeagentics_dev}"

echo -e "${BOLD}${RED}FreeAgentics Database Reset${NC}"
echo "============================="
echo ""
echo -e "${YELLOW}WARNING: This will completely destroy and recreate the database!${NC}"
echo "Database: $DB_NAME on $DB_HOST:$DB_PORT"
echo ""

# Confirmation prompt
read -p "Are you sure you want to reset the database? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Database reset cancelled."
    exit 0
fi

echo ""
echo -e "${BOLD}${BLUE}Resetting database...${NC}"

# Drop database
echo -n "Dropping database '$DB_NAME'... "
if PGPASSWORD=postgres psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -c "DROP DATABASE IF EXISTS $DB_NAME;" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC} Failed to drop database"
    exit 1
fi

# Drop user
echo -n "Dropping user '$DB_USER'... "
if PGPASSWORD=postgres psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -c "DROP USER IF EXISTS $DB_USER;" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}⚠${NC} User may not exist"
fi

echo ""
echo -e "${BOLD}${BLUE}Recreating database...${NC}"

# Run the main database setup script
if [ -f "infrastructure/scripts/setup-database.sh" ]; then
    bash infrastructure/scripts/setup-database.sh all
    echo ""
    echo -e "${GREEN}Database reset complete!${NC}"
else
    echo -e "${RED}Database setup script not found!${NC}"
    exit 1
fi
