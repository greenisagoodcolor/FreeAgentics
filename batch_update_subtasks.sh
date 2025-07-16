#!/bin/bash

# Batch update script for all remaining subtasks
# This script systematically adds the comprehensive cleanup process to all subtasks

echo "Starting comprehensive cleanup integration for all subtasks..."

# Common cleanup addition text
CLEANUP_TEXT="Add comprehensive cleanup process: **COMPREHENSIVE CLEANUP PROCESS (MANDATORY)**: Before marking this subtask complete, perform comprehensive repository cleanup using ultrathink methodology: 1. **ULTRATHINK RESEARCH & PLANNING** (30 min): Re-read all 1051 lines of CLAUDE.md to refresh methodology, analyze current repository state and identify cleanup opportunities, create systematic cleanup plan with validation checkpoints. 2. **REPOSITORY CLEANUP** (45 min): Scan and remove tech debt: unused imports, dead code, obsolete files; Delete old files: *.tmp, *.backup, *.old, old test reports, cache files; Consolidate directories: merge duplicated structures, organize by function; Remove obsolete documentation and outdated information; Clean build artifacts: __pycache__, *.pyc, dist/, build/, node_modules/. 3. **DOCUMENTATION CONSOLIDATION** (30 min): Update and consolidate documentation into README structure; Minimize separate documents by merging related content; Create clear documentation order for new developers; Ensure logical onboarding path with numbered steps; Archive obsolete documentation rather than deleting. 4. **CODE QUALITY RESOLUTION** (60 min): Fix ALL type errors comprehensively using ultrathink approach; Resolve ALL pre-commit hook issues (zero tolerance policy); Ensure ALL automated checks pass: make format && make test && make lint; Document and fix any red flags in code quality checks; Validate security baseline compliance. 5. **GIT WORKFLOW** (15 min): Execute proper git workflow: git add ., git commit -m [cleanup] Comprehensive cleanup for subtask SUBTASK_ID, git push; Use conventional commit messages with clear scope; Validate all changes are properly committed and pushed. **VALIDATION REQUIREMENTS**: ALL automated checks must pass (make format && make test && make lint); ZERO type errors allowed; ZERO pre-commit hook failures; Clean git working directory; Documentation consolidated and organized; Repository size optimized. **FAILURE PROTOCOL**: If ANY quality check fails: STOP IMMEDIATELY - do not continue with other tasks; FIX ALL ISSUES - address every ❌ until everything is ✅ green; VERIFY THE FIX - re-run failed command to confirm resolution; CONTINUE CLEANUP - return to cleanup process; NEVER IGNORE - zero tolerance policy for quality issues. Use the cleanup tools: ./run_cleanup.sh for full process or ./validate_cleanup.py for validation only."

# Task 1 remaining subtasks
for subtask in 1.4 1.5 1.6 1.7 1.8 1.9 1.10; do
    echo "Updating subtask $subtask..."
    SUBTASK_CLEANUP_TEXT=$(echo "$CLEANUP_TEXT" | sed "s/SUBTASK_ID/$subtask/g")
    task-master update-subtask --id="$subtask" --prompt="$SUBTASK_CLEANUP_TEXT"
done

# Task 2 subtasks
for subtask in 2.1 2.2 2.3 2.4 2.5 2.6; do
    echo "Updating subtask $subtask..."
    SUBTASK_CLEANUP_TEXT=$(echo "$CLEANUP_TEXT" | sed "s/SUBTASK_ID/$subtask/g")
    task-master update-subtask --id="$subtask" --prompt="$SUBTASK_CLEANUP_TEXT"
done

# Task 3 subtasks (need to check structure)
for subtask in 3.1 3.2 3.3 3.4 3.5 3.6 3.7; do
    echo "Updating subtask $subtask..."
    SUBTASK_CLEANUP_TEXT=$(echo "$CLEANUP_TEXT" | sed "s/SUBTASK_ID/$subtask/g")
    task-master update-subtask --id="$subtask" --prompt="$SUBTASK_CLEANUP_TEXT"
done

echo "Batch update of initial subtasks completed!"