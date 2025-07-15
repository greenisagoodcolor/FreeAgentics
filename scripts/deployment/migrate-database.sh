#!/bin/bash
# Database migration script for FreeAgentics

set -euo pipefail

# Configuration
ENVIRONMENT="${1:-production}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_BEFORE_MIGRATION="${BACKUP_BEFORE_MIGRATION:-true}"
DRY_RUN="${DRY_RUN:-false}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
log() {
    echo -e "${GREEN}[MIGRATION]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $*" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $*"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $*"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is required but not installed"
        exit 1
    fi
    
    # Check docker-compose
    if ! command -v docker-compose &> /dev/null; then
        error "docker-compose is required but not installed"
        exit 1
    fi
    
    # Check environment file
    ENV_FILE="$PROJECT_ROOT/.env.$ENVIRONMENT"
    if [[ ! -f "$ENV_FILE" ]]; then
        error "Environment file not found: $ENV_FILE"
        exit 1
    fi
    
    # Check compose file
    COMPOSE_FILE="$PROJECT_ROOT/docker-compose.$ENVIRONMENT.yml"
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        warning "Environment-specific compose file not found, using production"
        COMPOSE_FILE="$PROJECT_ROOT/docker-compose.production.yml"
    fi
    
    log "Prerequisites check passed ✓"
}

# Backup database
backup_database() {
    if [[ "$BACKUP_BEFORE_MIGRATION" != "true" ]]; then
        warning "Skipping database backup (BACKUP_BEFORE_MIGRATION=false)"
        return
    fi
    
    log "Creating database backup..."
    
    BACKUP_DIR="/var/backups/freeagentics/migrations"
    mkdir -p "$BACKUP_DIR"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="$BACKUP_DIR/pre_migration_${ENVIRONMENT}_${TIMESTAMP}.sql.gz"
    
    # Get database connection details
    DB_HOST=$(grep DATABASE_HOST "$ENV_FILE" | cut -d'=' -f2 || echo "postgres")
    DB_PORT=$(grep DATABASE_PORT "$ENV_FILE" | cut -d'=' -f2 || echo "5432")
    DB_NAME=$(grep DATABASE_NAME "$ENV_FILE" | cut -d'=' -f2 || echo "freeagentics")
    DB_USER=$(grep DATABASE_USER "$ENV_FILE" | cut -d'=' -f2 || echo "freeagentics")
    
    # Create backup
    docker-compose -f "$COMPOSE_FILE" exec -T postgres \
        pg_dump -U "$DB_USER" -d "$DB_NAME" --no-owner --clean --if-exists | \
        gzip > "$BACKUP_FILE"
    
    # Verify backup
    if [[ -f "$BACKUP_FILE" ]] && [[ -s "$BACKUP_FILE" ]]; then
        SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
        log "Database backup created: $BACKUP_FILE ($SIZE)"
        
        # Test backup integrity
        if gunzip -t "$BACKUP_FILE" 2>/dev/null; then
            log "Backup integrity verified ✓"
        else
            error "Backup file is corrupted!"
            exit 1
        fi
    else
        error "Backup creation failed!"
        exit 1
    fi
}

# Check current migration status
check_migration_status() {
    log "Checking current migration status..."
    
    # Get current revision
    CURRENT_REV=$(docker-compose -f "$COMPOSE_FILE" run --rm api \
        alembic current 2>/dev/null | grep -oE '[a-f0-9]{12}' | head -1 || echo "none")
    
    if [[ "$CURRENT_REV" == "none" ]]; then
        warning "No migrations have been applied yet"
    else
        info "Current revision: $CURRENT_REV"
        
        # Show revision details
        docker-compose -f "$COMPOSE_FILE" run --rm api \
            alembic show "$CURRENT_REV" 2>/dev/null || true
    fi
    
    # Check for pending migrations
    log "Checking for pending migrations..."
    PENDING=$(docker-compose -f "$COMPOSE_FILE" run --rm api \
        alembic history -r "$CURRENT_REV:head" 2>/dev/null | grep -v "$CURRENT_REV" || echo "")
    
    if [[ -z "$PENDING" ]]; then
        info "No pending migrations"
        return 1
    else
        info "Pending migrations:"
        echo "$PENDING"
        return 0
    fi
}

# Generate migration SQL
generate_migration_sql() {
    log "Generating migration SQL for review..."
    
    SQL_FILE="$PROJECT_ROOT/migration_preview_$(date +%Y%m%d_%H%M%S).sql"
    
    docker-compose -f "$COMPOSE_FILE" run --rm api \
        alembic upgrade head --sql > "$SQL_FILE" 2>/dev/null
    
    if [[ -s "$SQL_FILE" ]]; then
        info "Migration SQL generated: $SQL_FILE"
        echo -e "\n${BLUE}=== Migration Preview ===${NC}"
        head -50 "$SQL_FILE"
        if [[ $(wc -l < "$SQL_FILE") -gt 50 ]]; then
            echo "... (truncated, see full file for complete SQL)"
        fi
        echo -e "${BLUE}========================${NC}\n"
    else
        warning "No SQL generated (database may be up to date)"
        rm -f "$SQL_FILE"
    fi
}

# Run migrations
run_migrations() {
    if [[ "$DRY_RUN" == "true" ]]; then
        warning "DRY RUN MODE - Not applying migrations"
        generate_migration_sql
        return
    fi
    
    log "Applying database migrations..."
    
    # Run with output capture for better error handling
    MIGRATION_OUTPUT=$(mktemp)
    
    if docker-compose -f "$COMPOSE_FILE" run --rm api \
        alembic upgrade head 2>&1 | tee "$MIGRATION_OUTPUT"; then
        
        log "Migrations applied successfully ✓"
        
        # Get new revision
        NEW_REV=$(docker-compose -f "$COMPOSE_FILE" run --rm api \
            alembic current 2>/dev/null | grep -oE '[a-f0-9]{12}' | head -1)
        
        info "New revision: $NEW_REV"
        
    else
        error "Migration failed!"
        echo -e "\n${RED}=== Migration Error Output ===${NC}"
        cat "$MIGRATION_OUTPUT"
        echo -e "${RED}=============================${NC}\n"
        
        rm -f "$MIGRATION_OUTPUT"
        exit 1
    fi
    
    rm -f "$MIGRATION_OUTPUT"
}

# Verify migration success
verify_migration() {
    log "Verifying migration success..."
    
    # Check alembic version table
    VERSION_CHECK=$(docker-compose -f "$COMPOSE_FILE" run --rm api \
        python -c "
from database.session import engine
from sqlalchemy import text
with engine.connect() as conn:
    result = conn.execute(text('SELECT version_num FROM alembic_version'))
    print(result.scalar())
" 2>/dev/null || echo "error")
    
    if [[ "$VERSION_CHECK" == "error" ]]; then
        error "Could not verify migration version"
        exit 1
    fi
    
    info "Alembic version table shows: $VERSION_CHECK"
    
    # Run basic schema validation
    log "Running schema validation..."
    
    SCHEMA_CHECK=$(docker-compose -f "$COMPOSE_FILE" run --rm api \
        python -c "
from database import models
from database.session import engine
from sqlalchemy import inspect

inspector = inspect(engine)
tables = inspector.get_table_names()
print(f'Found {len(tables)} tables')
for table in sorted(tables):
    print(f'  - {table}')
" 2>&1) || {
        error "Schema validation failed"
        echo "$SCHEMA_CHECK"
        exit 1
    }
    
    echo "$SCHEMA_CHECK"
    
    # Test database operations
    log "Testing database operations..."
    
    TEST_RESULT=$(docker-compose -f "$COMPOSE_FILE" run --rm api \
        python -c "
from database.session import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    # Test read
    result = db.execute(text('SELECT 1'))
    print('✓ Read test passed')
    
    # Test transaction
    db.execute(text('BEGIN'))
    db.execute(text('SELECT 1'))
    db.execute(text('ROLLBACK'))
    print('✓ Transaction test passed')
    
finally:
    db.close()
" 2>&1) || {
        error "Database operation tests failed"
        echo "$TEST_RESULT"
        exit 1
    }
    
    echo "$TEST_RESULT"
    
    log "Migration verification completed ✓"
}

# Create migration
create_migration() {
    local message="$1"
    
    if [[ -z "$message" ]]; then
        error "Migration message is required"
        echo "Usage: $0 create \"migration message\""
        exit 1
    fi
    
    log "Creating new migration: $message"
    
    # Generate migration files
    docker-compose -f "$COMPOSE_FILE" run --rm api \
        alembic revision --autogenerate -m "$message"
    
    # Show generated files
    NEW_MIGRATION=$(find "$PROJECT_ROOT/alembic/versions" -name "*.py" -mmin -1 | head -1)
    
    if [[ -n "$NEW_MIGRATION" ]]; then
        log "Migration created: $NEW_MIGRATION"
        echo -e "\n${BLUE}=== Generated Migration ===${NC}"
        cat "$NEW_MIGRATION"
        echo -e "${BLUE}==========================${NC}\n"
        
        warning "Please review the migration before applying!"
    else
        error "Migration creation failed"
        exit 1
    fi
}

# Rollback migration
rollback_migration() {
    local target="${1:-1}"
    
    warning "Rolling back migration..."
    
    # Backup current state first
    backup_database
    
    if [[ "$target" =~ ^[0-9]+$ ]]; then
        # Rollback by number of revisions
        log "Rolling back $target revision(s)..."
        docker-compose -f "$COMPOSE_FILE" run --rm api \
            alembic downgrade "-$target"
    else
        # Rollback to specific revision
        log "Rolling back to revision: $target"
        docker-compose -f "$COMPOSE_FILE" run --rm api \
            alembic downgrade "$target"
    fi
    
    # Show new status
    check_migration_status
}

# Main execution
main() {
    case "${2:-migrate}" in
        "create")
            check_prerequisites
            create_migration "$3"
            ;;
        "rollback")
            check_prerequisites
            rollback_migration "$3"
            ;;
        "status")
            check_prerequisites
            check_migration_status
            ;;
        "migrate"|"")
            check_prerequisites
            
            echo -e "\n${GREEN}=== Database Migration Tool ===${NC}"
            echo "Environment: $ENVIRONMENT"
            echo "Dry Run: $DRY_RUN"
            echo -e "${GREEN}==============================${NC}\n"
            
            # Check if migrations are needed
            if ! check_migration_status; then
                log "Database is already up to date"
                exit 0
            fi
            
            # Confirm before proceeding
            if [[ "$DRY_RUN" != "true" ]] && [[ "${FORCE:-false}" != "true" ]]; then
                echo -e "\n${YELLOW}This will apply migrations to the $ENVIRONMENT database.${NC}"
                echo -e "${YELLOW}Do you want to continue? (yes/no)${NC}"
                read -r response
                if [[ "$response" != "yes" ]]; then
                    log "Migration cancelled by user"
                    exit 0
                fi
            fi
            
            # Run migration steps
            backup_database
            run_migrations
            verify_migration
            
            echo -e "\n${GREEN}=== Migration Complete ===${NC}"
            log "All migrations applied successfully! ✓"
            ;;
        *)
            echo "Usage: $0 [environment] [command] [args]"
            echo "Commands:"
            echo "  migrate  - Apply pending migrations (default)"
            echo "  create   - Create a new migration"
            echo "  rollback - Rollback migrations"
            echo "  status   - Show migration status"
            exit 1
            ;;
    esac
}

# Handle arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --force)
            FORCE="true"
            shift
            ;;
        --no-backup)
            BACKUP_BEFORE_MIGRATION="false"
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Run main
main "$@"