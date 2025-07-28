#!/bin/bash
# FreeAgentics Directory Structure Migration Script
# Purpose: Clean up the confusing directory structure by consolidating to a single clear layout

set -euo pipefail

echo "=== FreeAgentics Directory Structure Migration ==="
echo "This script will reorganize the confusing directory structure"
echo ""

# Check if we're in the right directory
if [ ! -d "freeagentics2-nemesis" ]; then
    echo "ERROR: Must run from /home/green/FreeAgentics/"
    exit 1
fi

# Create backup
echo "Step 1: Creating backup..."
BACKUP_DIR=".backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "Backing up critical directories to $BACKUP_DIR/"

# Only backup the key directories, not the huge venv directories
for dir in freeagentics2-nemesis freeagentics2-fresh new-dev-test; do
    if [ -d "$dir" ]; then
        echo "Backing up $dir (excluding venv and node_modules)..."
        rsync -av --exclude='venv' --exclude='node_modules' --exclude='.git' "$dir" "$BACKUP_DIR/" || true
    fi
done

echo ""
echo "Step 2: Creating new structure..."

# Create the new clean structure
mkdir -p src
mkdir -p tests 
mkdir -p scripts
mkdir -p docker
mkdir -p .archived

echo ""
echo "Step 3: Moving active code from freeagentics2-nemesis..."

# Move the active development code to the root
if [ -d "freeagentics2-nemesis" ]; then
    # Move source directories
    for dir in agents api auth coalitions database inference knowledge knowledge_graph llm services websocket world; do
        if [ -d "freeagentics2-nemesis/$dir" ]; then
            echo "Moving $dir to src/"
            mv "freeagentics2-nemesis/$dir" "src/" 2>/dev/null || cp -r "freeagentics2-nemesis/$dir" "src/"
        fi
    done
    
    # Move web directory specially (it's the frontend)
    if [ -d "freeagentics2-nemesis/web" ]; then
        echo "Moving web frontend..."
        mv "freeagentics2-nemesis/web" ./ 2>/dev/null || cp -r "freeagentics2-nemesis/web" ./
    fi
    
    # Move test directory
    if [ -d "freeagentics2-nemesis/tests" ]; then
        echo "Moving tests..."
        rsync -av "freeagentics2-nemesis/tests/" "tests/"
    fi
    
    # Move scripts
    if [ -d "freeagentics2-nemesis/scripts" ]; then
        echo "Moving scripts..."
        rsync -av "freeagentics2-nemesis/scripts/" "scripts/"
    fi
    
    # Move docker files
    for file in Dockerfile docker-compose.yml docker-compose.production.yml; do
        if [ -f "freeagentics2-nemesis/$file" ]; then
            echo "Moving $file..."
            cp "freeagentics2-nemesis/$file" ./
        fi
    done
    
    # Move key config files
    for file in requirements.txt requirements-dev.txt pyproject.toml pytest.ini .env.example alembic.ini; do
        if [ -f "freeagentics2-nemesis/$file" ]; then
            echo "Moving $file..."
            cp "freeagentics2-nemesis/$file" ./
        fi
    done
    
    # Use the Makefile from nemesis as it's the most recent
    if [ -f "freeagentics2-nemesis/Makefile" ]; then
        echo "Updating Makefile..."
        cp "freeagentics2-nemesis/Makefile" ./Makefile.new
        echo "# Note: Review Makefile.new and replace current Makefile if appropriate"
    fi
fi

echo ""
echo "Step 4: Creating deprecation notices..."

# Create deprecation READMEs
for dir in freeagentics2 freeagentics2-fresh freeagentics2-nemesis new-dev-test; do
    if [ -d "$dir" ]; then
        cat > "$dir/DEPRECATED_README.md" << 'EOF'
# ⚠️ DEPRECATED DIRECTORY ⚠️

This directory is DEPRECATED and should not be used.

## Why is this deprecated?

The FreeAgentics project previously had multiple confusing directory structures:
- `freeagentics2/` - Old development version
- `freeagentics2-fresh/` - Staging/test environment  
- `freeagentics2-nemesis/` - Was the active development
- `new-dev-test/` - Test environment

This caused massive confusion for new developers.

## Where is the current code?

All active development has been consolidated to the root directory:
- `/home/green/FreeAgentics/src/` - All source code
- `/home/green/FreeAgentics/tests/` - All tests
- `/home/green/FreeAgentics/web/` - Frontend code

## What should I do?

1. Use the root directory structure
2. Run `make` commands from `/home/green/FreeAgentics/`
3. Do NOT use any code from this deprecated directory

This directory will be moved to `.archived/` in the next cleanup phase.
EOF
    fi
done

echo ""
echo "Step 5: Summary of changes..."
echo ""
echo "✅ Created backup in $BACKUP_DIR/"
echo "✅ Moved active code to clean structure at root"
echo "✅ Added deprecation notices to old directories"
echo ""
echo "⚠️  IMPORTANT NEXT STEPS:"
echo "1. Review the new structure"
echo "2. Update all import paths in Python files"
echo "3. Update paths in configuration files"
echo "4. Test with 'make install' and 'make test'"
echo "5. Commit the changes"
echo "6. Eventually move deprecated directories to .archived/"
echo ""
echo "Migration complete!"