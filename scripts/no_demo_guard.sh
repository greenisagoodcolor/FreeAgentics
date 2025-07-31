#!/usr/bin/env bash
# Guard script to prevent demo references from entering the codebase

set -euo pipefail

echo "🔍 Checking for forbidden 'demo' references..."

# Exclude certain files and directories from the check
EXCLUSIONS=(
    "CHANGELOG.md"
    "README.md"
    "CLAUDE.md"
    "scripts/no_demo_guard.sh"
    "tests/characterization/"
    "docs/archive/"
    "examples/"
    "*.log"
    "*.md"
)

# Build exclusion pattern for grep
GREP_EXCLUSIONS=""
for exclusion in "${EXCLUSIONS[@]}"; do
    GREP_EXCLUSIONS="${GREP_EXCLUSIONS} --exclude=${exclusion}"
done

# Search for demo references
if git grep -I --line-number -e '\bdemo\b' -- ':!CHANGELOG.md' ':!README.md' ':!CLAUDE.md' ':!scripts/no_demo_guard.sh' ':!tests/characterization/' ':!docs/archive/' ':!examples/' | grep -vE 'tests?/' | head -10; then
    echo ""
    echo "🚫 Found forbidden token 'demo' in codebase!"
    echo ""
    echo "The 'Kill-All-Demo v2' policy prohibits any 'demo' references in active code."
    echo "Only 'dev' mode is allowed for external developer testing."
    echo ""
    echo "Please replace 'demo' with 'dev' in the above files."
    exit 1
else
    echo "✅ No forbidden 'demo' references found!"
    echo "🎯 Codebase is demo-free and ready for external dev testing."
fi
