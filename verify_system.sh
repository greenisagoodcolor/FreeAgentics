#!/bin/bash
# FreeAgentics System Verification Script
# Purpose: Help new developers understand the current directory structure

echo "==================================="
echo "FreeAgentics Directory Structure Verification"
echo "==================================="
echo ""

# Check current directory
echo "📍 Current directory: $(pwd)"
echo ""

# Show the confusing structure
echo "📁 Directory Structure Overview:"
echo "--------------------------------"
ls -la | grep -E "^d.*freeagentics|^-.*README" | awk '{print $9}' | while read dir; do
    if [[ -d "$dir" ]]; then
        file_count=$(find "$dir" -name "*.py" -o -name "*.ts" -o -name "*.tsx" 2>/dev/null | grep -v node_modules | grep -v venv | wc -l)
        echo "  📂 $dir/ ($file_count source files)"
    else
        echo "  📄 $dir"
    fi
done

echo ""
echo "🔍 Checking Git Repositories:"
echo "--------------------------------"
# Check main repo
echo "Main directory git remote:"
git remote -v 2>/dev/null | head -1 || echo "  No git repository"

# Check subdirectories
for dir in freeagentics2 freeagentics2-fresh freeagentics2-nemesis; do
    if [[ -d "$dir/.git" ]]; then
        echo ""
        echo "$dir/ git remote:"
        (cd "$dir" && git remote -v 2>/dev/null | head -1)
    fi
done

echo ""
echo "✅ Active Development Directory:"
echo "--------------------------------"
echo "👉 freeagentics2-nemesis/ is the ACTIVE directory"
echo ""

# Check if key files exist in the active directory
echo "🔧 Checking freeagentics2-nemesis/ contents:"
if [[ -d "freeagentics2-nemesis" ]]; then
    cd freeagentics2-nemesis
    echo "  ✓ Makefile exists: $(test -f Makefile && echo 'YES' || echo 'NO')"
    echo "  ✓ requirements.txt exists: $(test -f requirements.txt && echo 'YES' || echo 'NO')"
    echo "  ✓ package.json exists: $(test -f web/package.json && echo 'YES' || echo 'NO')"
    echo "  ✓ API directory exists: $(test -d api && echo 'YES' || echo 'NO')"
    echo "  ✓ Tests directory exists: $(test -d tests && echo 'YES' || echo 'NO')"
    cd ..
fi

echo ""
echo "📚 Documentation:"
echo "--------------------------------"
echo "1. Read URGENT_DIRECTORY_STRUCTURE_README.md for details"
echo "2. Use freeagentics2-nemesis/ for ALL development"
echo "3. Ignore other freeagentics* directories"
echo ""
echo "🚀 Quick Start Commands:"
echo "--------------------------------"
echo "cd freeagentics2-nemesis"
echo "make install"
echo "make dev"
echo ""
echo "==================================="