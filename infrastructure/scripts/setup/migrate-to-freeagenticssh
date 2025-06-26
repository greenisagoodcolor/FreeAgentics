#!/bin/bash

# FreeAgentics Migration Script (Bash version)
# Preserves git history while transforming CogniticNet to FreeAgentics
# Lead: Martin Fowler

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DRY_RUN=${1:-false}
LOG_FILE="migration_log.txt"

# Functions
log() {
    echo -e "${GREEN}[MIGRATE]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

git_mv_safe() {
    local source=$1
    local dest=$2

    if [ "$DRY_RUN" = "true" ]; then
        log "DRY RUN: git mv $source $dest"
        return 0
    fi

    if [ ! -e "$source" ]; then
        warning "Source not found: $source"
        return 1
    fi

    # Create destination directory if needed
    local dest_dir=$(dirname "$dest")
    mkdir -p "$dest_dir"

    # Perform git mv
    if git mv "$source" "$dest" 2>/dev/null; then
        log "Moved: $source → $dest"
        return 0
    else
        error "Failed to move: $source"
        return 1
    fi
}

# Main migration function
migrate() {
    log "Starting FreeAgentics migration..."
    log "Timestamp: $(date)"

    # Create rollback tag
    if [ "$DRY_RUN" != "true" ]; then
        if ! git tag pre-freeagentics-migration 2>/dev/null; then
            warning "Tag pre-freeagentics-migration already exists"
        else
            log "Created rollback tag: pre-freeagentics-migration"
        fi
    fi

    # Stage 1: Core Agent Structure
    log "\nStage 1: Core Agent Structure"
    log "----------------------------------------"

    # Basic agent to base
    if [ -d "src/agents/basic_agent" ]; then
        git_mv_safe "src/agents/basic_agent" "agents/base"
    fi

    # Agent conversation
    if [ -f "src/agents/agent_conversation.py" ]; then
        git_mv_safe "src/agents/agent_conversation.py" "agents/base/communication.py"
    fi

    # Create agent type directories if they don't exist
    for agent in explorer merchant scholar guardian; do
        mkdir -p "agents/$agent"
    done

    # Stage 2: Active Inference Engine
    log "\nStage 2: Active Inference Engine"
    log "----------------------------------------"

    # Move active inference
    if [ -d "src/agents/active_inference" ]; then
        mkdir -p "inference/engine"

        # Move specific files with renaming
        [ -f "src/agents/active_inference/inference.py" ] && \
            git_mv_safe "src/agents/active_inference/inference.py" "inference/engine/active-inference.py"

        [ -f "src/agents/active_inference/belief_update.py" ] && \
            git_mv_safe "src/agents/active_inference/belief_update.py" "inference/engine/belief-update.py"

        [ -f "src/agents/active_inference/policy_learning.py" ] && \
            git_mv_safe "src/agents/active_inference/policy_learning.py" "inference/engine/policy-selection.py"

        # Move remaining files
        for file in src/agents/active_inference/*.py; do
            if [ -f "$file" ]; then
                basename=$(basename "$file")
                git_mv_safe "$file" "inference/engine/$basename"
            fi
        done
    fi

    # Move GNN
    if [ -d "src/gnn" ]; then
        git_mv_safe "src/gnn" "inference/gnn"
    fi

    # Move LLM
    if [ -d "src/llm" ]; then
        git_mv_safe "src/llm" "inference/llm"
    fi

    # Stage 3: Coalition & World
    log "\nStage 3: Coalition & World"
    log "----------------------------------------"

    # Move coalition
    if [ -d "src/agents/coalition" ]; then
        mkdir -p "coalitions/formation"
        mkdir -p "coalitions/contracts"

        [ -f "src/agents/coalition/coalition_criteria.py" ] && \
            git_mv_safe "src/agents/coalition/coalition_criteria.py" "coalitions/formation/preference-matching.py"

        [ -f "src/agents/coalition/business_opportunities.py" ] && \
            git_mv_safe "src/agents/coalition/business_opportunities.py" "coalitions/contracts/resource-sharing.py"

        # Move remaining coalition files
        for file in src/agents/coalition/*.py; do
            if [ -f "$file" ]; then
                basename=$(basename "$file")
                git_mv_safe "$file" "coalitions/formation/$basename"
            fi
        done
    fi

    # Move world
    if [ -d "src/world" ]; then
        mkdir -p "world/grid"

        [ -f "src/world/h3_world.py" ] && \
            git_mv_safe "src/world/h3_world.py" "world/grid/hex-world.py"

        # Move other world files
        for file in src/world/*.py; do
            if [ -f "$file" ] && [ "$file" != "src/world/h3_world.py" ]; then
                basename=$(basename "$file")
                git_mv_safe "$file" "world/$basename"
            fi
        done
    fi

    # Move spatial
    if [ -d "src/spatial" ]; then
        mkdir -p "world/grid"

        [ -f "src/spatial/spatial_api.py" ] && \
            git_mv_safe "src/spatial/spatial_api.py" "world/grid/spatial-index.py"

        # Move other spatial files
        for file in src/spatial/*.py; do
            if [ -f "$file" ] && [ "$file" != "src/spatial/spatial_api.py" ]; then
                basename=$(basename "$file")
                git_mv_safe "$file" "world/grid/$basename"
            fi
        done
    fi

    # Stage 4: API & Frontend
    log "\nStage 4: API & Frontend"
    log "----------------------------------------"

    # Move API
    if [ -d "app/api" ]; then
        git_mv_safe "app/api" "api/rest"
    fi

    # Move frontend components
    if [ -d "app/components" ]; then
        git_mv_safe "app/components" "web/src/components"
    fi

    if [ -d "src/hooks" ]; then
        git_mv_safe "src/hooks" "web/src/hooks"
    fi

    if [ -d "src/lib" ]; then
        git_mv_safe "src/lib" "web/src/lib"
    fi

    if [ -d "src/contexts" ]; then
        git_mv_safe "src/contexts" "web/src/contexts"
    fi

    # Move app pages to web
    for item in app/*; do
        if [ -f "$item" ] || [ -d "$item" ]; then
            basename=$(basename "$item")
            if [[ "$basename" != "api" && "$basename" != "components" ]]; then
                git_mv_safe "$item" "web/src/$basename"
            fi
        fi
    done

    # Stage 5: Infrastructure & Config
    log "\nStage 5: Infrastructure & Config"
    log "----------------------------------------"

    # Move Docker files
    if [ -f "environments/demo/docker-compose.yml" ]; then
        mkdir -p "infrastructure/docker"
        git_mv_safe "environments/demo/docker-compose.yml" "infrastructure/docker/docker-compose.yml"
    fi

    if [ -f "environments/demo/Dockerfile.web" ]; then
        git_mv_safe "environments/demo/Dockerfile.web" "infrastructure/docker/Dockerfile.web"
    fi

    # Move environment configs
    if [ -d "environments" ]; then
        for env in environments/*; do
            if [ -d "$env" ]; then
                envname=$(basename "$env")
                if [ -f "$env/env.$envname" ]; then
                    mkdir -p "config/environments"
                    git_mv_safe "$env/env.$envname" "config/environments/$envname.yml"
                fi
            fi
        done
    fi

    # Move tests
    if [ -d "src/tests" ]; then
        git_mv_safe "src/tests" "tests"
    fi

    # Move documentation
    if [ -d "doc" ]; then
        git_mv_safe "doc" "docs"
    fi

    # Rename scripts
    if [ -f "scripts/cogniticnet-cli.js" ]; then
        git_mv_safe "scripts/cogniticnet-cli.js" "scripts/freeagentics-cli.js"
    fi

    # Update references in files
    log "\nUpdating file references..."
    if [ "$DRY_RUN" != "true" ]; then
        # Find and replace in all relevant files
        find . -type f \( -name "*.py" -o -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.jsx" -o -name "*.md" -o -name "*.yml" -o -name "*.yaml" -o -name "*.json" \) \
            -not -path "./.git/*" -not -path "./node_modules/*" -not -path "./venv/*" \
            -exec sed -i.bak \
            -e 's/CogniticNet/FreeAgentics/g' \
            -e 's/cogniticnet/freeagentics/g' \
            -e 's/cogneticnet/freeagentics/g' \
            -e 's/COGNITICNET/FREEAGENTICS/g' {} \;

        # Remove backup files
        find . -name "*.bak" -delete

        log "Updated all file references"
    fi

    # Commit final changes
    if [ "$DRY_RUN" != "true" ]; then
        git add -A
        git commit -m "Complete FreeAgentics migration: Updated all references" || true
    fi

    log "\nMigration complete!"
    log "Check $LOG_FILE for full details"
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    error "Not in a git repository!"
    exit 1
fi

# Main execution
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [--dry-run]"
    echo "  --dry-run: Show what would be done without making changes"
    exit 0
fi

# Initialize log file
echo "FreeAgentics Migration Log - $(date)" > "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

# Run migration
migrate

# Show summary
echo -e "\n${GREEN}Migration Summary:${NC}"
echo "- Log file: $LOG_FILE"
echo "- Rollback tag: pre-freeagentics-migration"
echo "- To rollback: git reset --hard pre-freeagentics-migration"
