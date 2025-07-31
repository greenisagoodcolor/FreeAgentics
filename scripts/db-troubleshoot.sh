#!/bin/bash
# Comprehensive database diagnostics and troubleshooting for FreeAgentics
# Usage: bash scripts/db-troubleshoot.sh [--fix] [--verbose]

set -e

# Configuration
FIX_MODE=false
VERBOSE=false
DATABASE_URL="${DATABASE_URL:-postgresql://freeagentics:freeagentics_dev@localhost:5432/freeagentics}"

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --fix)
            FIX_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "FreeAgentics Database Troubleshooting Script"
            echo ""
            echo "Usage: $0 [--fix] [--verbose]"
            echo ""
            echo "Options:"
            echo "  --fix      Attempt to automatically fix common issues"
            echo "  --verbose  Show detailed diagnostic information"
            echo "  --help     Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  DATABASE_URL  Database connection string (default: Docker setup)"
            exit 0
            ;;
    esac
done

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}ℹ${NC} $1"; }
log_success() { echo -e "${GREEN}✅${NC} $1"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}❌${NC} $1"; }

# Utility functions
check_command() {
    command -v "$1" >/dev/null 2>&1
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    local fix_command="${3:-}"
    
    echo -n "Testing $test_name... "
    
    if eval "$test_command" >/dev/null 2>&1; then
        log_success "$test_name OK"
        return 0
    else
        log_error "$test_name FAILED"
        
        if [ "$FIX_MODE" = true ] && [ -n "$fix_command" ]; then
            log_info "Attempting to fix $test_name..."
            if eval "$fix_command"; then
                log_success "$test_name FIXED"
                return 0
            else
                log_error "Failed to fix $test_name"
                return 1
            fi
        fi
        return 1
    fi
}

echo "=== FreeAgentics Database Diagnostics ==="
echo "Database URL: ${DATABASE_URL}"
echo "Fix mode: ${FIX_MODE}"
echo "Verbose: ${VERBOSE}"
echo ""

# Test 1: Check required commands
log_info "1. Checking required commands..."
MISSING_COMMANDS=()

for cmd in docker docker-compose psql; do
    if ! check_command "$cmd"; then
        MISSING_COMMANDS+=("$cmd")
        log_error "$cmd not found"
    else
        log_success "$cmd available"
    fi
done

if [ ${#MISSING_COMMANDS[@]} -gt 0 ]; then
    log_error "Missing required commands: ${MISSING_COMMANDS[*]}"
    echo "Please install missing commands and try again."
    exit 1
fi

# Test 2: Docker container status
log_info "2. Checking Docker container status..."
if docker-compose -f docker-compose.db.yml ps | grep -q "Up"; then
    log_success "PostgreSQL container is running"
    CONTAINER_RUNNING=true
else
    log_error "PostgreSQL container is not running"
    CONTAINER_RUNNING=false
    
    if [ "$FIX_MODE" = true ]; then
        log_info "Starting PostgreSQL container..."
        docker-compose -f docker-compose.db.yml up -d
        sleep 10
        if docker-compose -f docker-compose.db.yml ps | grep -q "Up"; then
            log_success "Container started successfully"
            CONTAINER_RUNNING=true
        else
            log_error "Failed to start container"
        fi
    fi
fi

# Test 3: Port availability
log_info "3. Checking port 5432 availability..."
if netstat -an 2>/dev/null | grep -q ":5432.*LISTEN" || ss -an 2>/dev/null | grep -q ":5432.*LISTEN"; then
    log_success "Port 5432 is listening"
else
    log_error "Port 5432 is not available"
fi

# Test 4: Database connection
log_info "4. Testing database connection..."
if psql "$DATABASE_URL" -c "SELECT 1;" >/dev/null 2>&1; then
    log_success "Database connection successful"
    CONNECTION_OK=true
else
    log_error "Database connection failed"
    CONNECTION_OK=false
    
    if [ "$VERBOSE" = true ]; then
        echo "Connection error details:"
        psql "$DATABASE_URL" -c "SELECT 1;" 2>&1 || true
    fi
fi

# Test 5: PostgreSQL version and health
if [ "$CONNECTION_OK" = true ]; then
    log_info "5. Checking PostgreSQL version and health..."
    
    PG_VERSION=$(psql "$DATABASE_URL" -t -c "SELECT version();" 2>/dev/null | head -1)
    if [ -n "$PG_VERSION" ]; then
        log_success "PostgreSQL version: $(echo $PG_VERSION | cut -d' ' -f1-2)"
    else
        log_error "Could not retrieve PostgreSQL version"
    fi
    
    # Check if database is accepting connections
    if psql "$DATABASE_URL" -c "SELECT pg_is_in_recovery();" >/dev/null 2>&1; then
        log_success "PostgreSQL is healthy and accepting connections"
    else
        log_error "PostgreSQL health check failed"
    fi
else
    log_warning "5. Skipping PostgreSQL health check (no connection)"
fi

# Test 6: pgvector extension
if [ "$CONNECTION_OK" = true ]; then
    log_info "6. Checking pgvector extension..."
    
    # Check if extension is available
    if psql "$DATABASE_URL" -t -c "SELECT count(*) FROM pg_available_extensions WHERE name = 'vector';" 2>/dev/null | grep -q "1"; then
        log_success "pgvector extension is available"
        
        # Check if extension is installed
        if psql "$DATABASE_URL" -t -c "SELECT count(*) FROM pg_extension WHERE extname = 'vector';" 2>/dev/null | grep -q "1"; then
            log_success "pgvector extension is installed"
            
            # Test vector functionality
            if psql "$DATABASE_URL" -c "SELECT '[1,2,3]'::vector;" >/dev/null 2>&1; then
                log_success "pgvector functionality is working"
            else
                log_error "pgvector functionality test failed"
            fi
        else
            log_error "pgvector extension is not installed"
            
            if [ "$FIX_MODE" = true ]; then
                log_info "Installing pgvector extension..."
                if psql "$DATABASE_URL" -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null 2>&1; then
                    log_success "pgvector extension installed"
                else
                    log_error "Failed to install pgvector extension"
                fi
            fi
        fi
    else
        log_error "pgvector extension is not available"
        log_warning "Make sure you're using the pgvector/pgvector Docker image"
    fi
else
    log_warning "6. Skipping pgvector check (no connection)"
fi

# Test 7: Alembic migration status
log_info "7. Checking Alembic migration status..."
if check_command alembic; then
    CURRENT_REVISION=$(alembic current 2>/dev/null | grep -o '[a-f0-9]\{12\}' | head -1)
    if [ -n "$CURRENT_REVISION" ]; then
        log_success "Current migration revision: $CURRENT_REVISION"
        
        # Check if migrations are up to date
        HEAD_REVISION=$(alembic heads 2>/dev/null | grep -o '[a-f0-9]\{12\}' | head -1)
        if [ "$CURRENT_REVISION" = "$HEAD_REVISION" ]; then
            log_success "Database migrations are up to date"
        else
            log_warning "Database migrations are not up to date"
            log_info "Current: $CURRENT_REVISION, Head: $HEAD_REVISION"
            
            if [ "$FIX_MODE" = true ]; then
                log_info "Running database migrations..."
                if alembic upgrade head >/dev/null 2>&1; then
                    log_success "Database migrations completed"
                else
                    log_error "Database migration failed"
                fi
            fi
        fi
    else
        log_error "Could not determine current migration revision"
    fi
else
    log_error "Alembic not found - install with: pip install alembic"
fi

# Test 8: Database schema
if [ "$CONNECTION_OK" = true ]; then
    log_info "8. Checking database schema..."
    
    TABLES=$(psql "$DATABASE_URL" -t -c "\dt" 2>/dev/null | wc -l)
    if [ "$TABLES" -gt 0 ]; then
        log_success "Database tables exist ($TABLES tables found)"
        
        # Check key tables
        for table in agents coalitions knowledge_nodes knowledge_edges; do
            if psql "$DATABASE_URL" -c "\d $table" >/dev/null 2>&1; then
                log_success "Table '$table' exists"
            else
                log_warning "Table '$table' missing"
            fi
        done
    else
        log_error "No database tables found"
        
        if [ "$FIX_MODE" = true ]; then
            log_info "Creating database schema..."
            if alembic upgrade head >/dev/null 2>&1; then
                log_success "Database schema created"
            else
                log_error "Failed to create database schema"
            fi
        fi
    fi
else
    log_warning "8. Skipping schema check (no connection)"
fi

# Test 9: Database permissions
if [ "$CONNECTION_OK" = true ]; then
    log_info "9. Checking database permissions..."
    
    # Test basic operations
    TEST_TABLE="test_permissions_$(date +%s)"
    
    if psql "$DATABASE_URL" -c "CREATE TABLE $TEST_TABLE (id SERIAL PRIMARY KEY);" >/dev/null 2>&1; then
        log_success "CREATE permission OK"
        
        if psql "$DATABASE_URL" -c "INSERT INTO $TEST_TABLE DEFAULT VALUES;" >/dev/null 2>&1; then
            log_success "INSERT permission OK"
            
            if psql "$DATABASE_URL" -c "SELECT * FROM $TEST_TABLE;" >/dev/null 2>&1; then
                log_success "SELECT permission OK"
            else
                log_error "SELECT permission failed"
            fi
        else
            log_error "INSERT permission failed"
        fi
        
        # Clean up test table
        psql "$DATABASE_URL" -c "DROP TABLE $TEST_TABLE;" >/dev/null 2>&1
    else
        log_error "CREATE permission failed"
    fi
else
    log_warning "9. Skipping permissions check (no connection)"
fi

# Test 10: Performance check
if [ "$CONNECTION_OK" = true ]; then
    log_info "10. Checking database performance..."
    
    # Check active connections
    ACTIVE_CONNECTIONS=$(psql "$DATABASE_URL" -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null)
    if [ -n "$ACTIVE_CONNECTIONS" ]; then
        log_success "Active connections: $ACTIVE_CONNECTIONS"
        
        if [ "$ACTIVE_CONNECTIONS" -gt 50 ]; then
            log_warning "High number of active connections"
        fi
    fi
    
    # Check database size
    DB_SIZE=$(psql "$DATABASE_URL" -t -c "SELECT pg_size_pretty(pg_database_size('freeagentics'));" 2>/dev/null)
    if [ -n "$DB_SIZE" ]; then
        log_success "Database size: $(echo $DB_SIZE | xargs)"
    fi
    
    # Simple performance test
    START_TIME=$(date +%s%N)
    psql "$DATABASE_URL" -c "SELECT count(*) FROM pg_stat_activity;" >/dev/null 2>&1
    END_TIME=$(date +%s%N)
    QUERY_TIME=$(( (END_TIME - START_TIME) / 1000000 ))
    
    if [ "$QUERY_TIME" -lt 100 ]; then
        log_success "Query performance: ${QUERY_TIME}ms (good)"
    elif [ "$QUERY_TIME" -lt 500 ]; then
        log_warning "Query performance: ${QUERY_TIME}ms (acceptable)"
    else
        log_error "Query performance: ${QUERY_TIME}ms (slow)"
    fi
else
    log_warning "10. Skipping performance check (no connection)"
fi

# Verbose diagnostics
if [ "$VERBOSE" = true ]; then
    echo ""
    log_info "=== Verbose Diagnostics ==="
    
    if [ "$CONTAINER_RUNNING" = true ]; then
        echo ""
        echo "Docker container details:"
        docker-compose -f docker-compose.db.yml ps
        echo ""
        echo "Recent container logs:"
        docker-compose -f docker-compose.db.yml logs --tail=20 postgres
    fi
    
    if [ "$CONNECTION_OK" = true ]; then
        echo ""
        echo "Database configuration:"
        psql "$DATABASE_URL" -c "SHOW shared_buffers; SHOW work_mem; SHOW max_connections;" 2>/dev/null
        
        echo ""
        echo "Database statistics:"
        psql "$DATABASE_URL" -c "SELECT * FROM pg_stat_database WHERE datname = 'freeagentics';" 2>/dev/null
        
        echo ""
        echo "Installed extensions:"
        psql "$DATABASE_URL" -c "SELECT extname, extversion FROM pg_extension ORDER BY extname;" 2>/dev/null
    fi
fi

echo ""
log_info "=== Summary ==="

if [ "$CONNECTION_OK" = true ]; then
    log_success "Database is operational"
    echo "Next steps:"
    echo "  - Run 'make dev' to start the application"
    echo "  - Visit http://localhost:3000 to access the frontend"
    echo "  - Use 'make db-check' for quick connection tests"
else
    log_error "Database has issues that need attention"
    echo "Next steps:"
    echo "  - Run with --fix flag to attempt automatic repairs"
    echo "  - Check Docker is running: 'docker --version'"
    echo "  - Start database: 'docker-compose -f docker-compose.db.yml up -d'"
    echo "  - Check logs: 'docker-compose -f docker-compose.db.yml logs postgres'"
fi

echo ""
echo "For more help, see docs/DATABASE_SETUP.md"