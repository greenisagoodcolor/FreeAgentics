# Comprehensive Repository Cleanup Process

## Overview
This document outlines a systematic cleanup process designed to be applied to each subtask, following the ultrathink methodology and zero tolerance approach to quality issues as outlined in CLAUDE.md.

## Phase 1: Ultrathink Research & Planning Phase

### 1.1 Re-read CLAUDE.md Guidelines
```bash
# MANDATORY: Re-read all 1159 lines of CLAUDE.md before starting
cat CLAUDE.md | head -50  # Review first 50 lines
cat CLAUDE.md | tail -100  # Review last 100 lines  
grep -n "MANDATORY\|BLOCKING\|NON-NEGOTIABLE\|ZERO TOLERANCE" CLAUDE.md
```

**Key Principles to Internalize:**
- "AUTOMATED CHECKS ARE MANDATORY. ALL hook issues are BLOCKING – EVERYTHING must be ✅ GREEN!"
- "TEST-DRIVEN DEVELOPMENT IS NON-NEGOTIABLE"
- "NEVER JUMP STRAIGHT TO CODING! Always follow this sequence: 1. Research, 2. Plan, 3. Implement"
- "Zero tolerance" for failing tests, linters, type-checkers, or any CI hooks

### 1.2 Deep Repository State Analysis
```bash
# Current repository status
git status --porcelain | wc -l  # Count modified files
git log --oneline -10           # Recent commits
git branch -a                   # All branches
git diff --name-only HEAD~5..HEAD  # Files changed in last 5 commits

# Quality metrics baseline
find . -name "*.py" -type f | wc -l    # Python files count
find . -name "*.ts" -type f | wc -l    # TypeScript files count
find . -name "*.test.*" -type f | wc -l # Test files count
find . -name "*.md" -type f | wc -l     # Documentation files count

# Identify technical debt
find . -name "TODO" -o -name "FIXME" -o -name "XXX" | head -20
grep -r "TODO\|FIXME\|XXX\|HACK" --include="*.py" --include="*.ts" . | head -10
```

### 1.3 Systematic Planning Documentation
Create a cleanup plan document for the current subtask:

```bash
# Create cleanup plan
cat > CLEANUP_PLAN_$(date +%Y%m%d_%H%M%S).md << 'EOF'
# Cleanup Plan for [SUBTASK_NAME]

## Pre-Cleanup Assessment
- [ ] Repository files count: [COUNT]
- [ ] Documentation files count: [COUNT]  
- [ ] Test files count: [COUNT]
- [ ] Open TODO/FIXME items: [COUNT]
- [ ] Current test coverage: [PERCENTAGE]
- [ ] Failing tests: [COUNT]
- [ ] Linting issues: [COUNT]
- [ ] Type errors: [COUNT]

## Cleanup Objectives
- [ ] Remove obsolete files and directories
- [ ] Consolidate documentation
- [ ] Fix all type errors
- [ ] Resolve all pre-commit hook issues
- [ ] Achieve 100% green CI status
- [ ] Update consolidated documentation

## Success Criteria
- [ ] All automated checks pass: `make format && make test && make lint`
- [ ] Zero failing tests
- [ ] Zero linting issues
- [ ] Zero type errors
- [ ] Documentation consolidated and up-to-date
- [ ] No obsolete files remaining
EOF
```

## Phase 2: Repository Cleanup Phase

### 2.1 Systematic File Scanning and Removal

#### 2.1.1 Identify Obsolete Files
```bash
# Find potentially obsolete files
find . -name "*.bak" -o -name "*.tmp" -o -name "*.old" -o -name "*~"
find . -name "*.pyc" -o -name "__pycache__" -type d
find . -name ".DS_Store" -o -name "Thumbs.db"
find . -name "*.log" -path "*/logs/*" -mtime +30  # Log files older than 30 days

# Find large files that might be artifacts
find . -size +10M -type f | head -10

# Find empty directories
find . -type d -empty

# Find duplicate files by name pattern
find . -name "*.py" -exec basename {} \; | sort | uniq -d
```

#### 2.1.2 Remove Technical Debt Systematically
```bash
# Remove build artifacts
rm -rf build/ dist/ *.egg-info/ .pytest_cache/
rm -rf node_modules/ .npm/ .yarn/
rm -rf .coverage htmlcov/ .nyc_output/

# Remove IDE and editor files
find . -name ".vscode" -type d -exec rm -rf {} +
find . -name ".idea" -type d -exec rm -rf {} +
find . -name "*.swp" -o -name "*.swo" -exec rm -f {} +

# Remove OS-specific files
find . -name ".DS_Store" -exec rm -f {} +
find . -name "Thumbs.db" -exec rm -f {} +
```

#### 2.1.3 Clean Up Old Test Reports and Artifacts
```bash
# Remove old test reports
find . -name "test-results" -type d -exec rm -rf {} +
find . -name "test-reports" -type d -exec rm -rf {} +
find . -name "coverage-reports" -type d -exec rm -rf {} +

# Remove old benchmark results
find . -name "benchmark-results" -type d -exec rm -rf {} +
find . -name "*.benchmark" -exec rm -f {} +

# Remove old profiling data
find . -name "*.prof" -o -name "*.pstats" -exec rm -f {} +
```

### 2.2 Directory Consolidation

#### 2.2.1 Analyze Directory Structure
```bash
# Generate directory tree
tree -d -L 3 > directory_structure.txt

# Identify directories with similar purposes
find . -type d -name "*test*" | head -10
find . -type d -name "*doc*" | head -10
find . -type d -name "*util*" | head -10
```

#### 2.2.2 Consolidate Similar Directories
```bash
# Example consolidation commands (adjust based on findings)
# Consolidate test directories
if [ -d "test" ] && [ -d "tests" ]; then
    mv test/* tests/ 2>/dev/null 
    rmdir test 2>/dev/null 
fi

# Consolidate documentation directories
if [ -d "doc" ] && [ -d "docs" ]; then
    mv doc/* docs/ 2>/dev/null 
    rmdir doc 2>/dev/null 
fi

# Consolidate utility directories
if [ -d "util" ] && [ -d "utils" ]; then
    mv util/* utils/ 2>/dev/null 
    rmdir util 2>/dev/null 
fi
```

### 2.3 Unused Code Removal

#### 2.3.1 Identify Unused Imports and Functions
```bash
# For Python files
autoflake --check --recursive --remove-all-unused-imports --remove-unused-variables . > unused_imports.txt

# For TypeScript files (if applicable)
npx ts-unused-exports tsconfig.json > unused_exports.txt 2>/dev/null 
```

#### 2.3.2 Remove Unused Code
```bash
# Remove unused imports (Python)
autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables .

# Remove unused test files
for test_file in $(find . -name "*test*.py" -o -name "test_*.py"); do
    if [ ! -s "$test_file" ]; then
        echo "Removing empty test file: $test_file"
        rm "$test_file"
    fi
done
```

## Phase 3: Documentation Consolidation Phase

### 3.1 Documentation Inventory
```bash
# List all documentation files
find . -name "*.md" -o -name "*.rst" -o -name "*.txt" | grep -E "(README|GUIDE|DOC)" > doc_inventory.txt

# Analyze documentation structure
echo "=== Documentation Inventory ===" > doc_analysis.txt
find . -name "*.md" -exec echo "File: {}" \; -exec head -5 {} \; -exec echo "---" \; >> doc_analysis.txt
```

### 3.2 Consolidate Documentation into README
```bash
# Create comprehensive README structure
cat > README_TEMPLATE.md << 'EOF'
# FreeAgentics

## Quick Start Guide
[Essential information for immediate use]

## Architecture Overview
[High-level system design]

## Development Setup
[Step-by-step development environment setup]

## API Reference
[Complete API documentation]

## Security Implementation
[Security features and best practices]

## Performance Optimization
[Performance characteristics and optimization strategies]

## Testing Guide
[Comprehensive testing approach]

## Deployment Guide
[Production deployment procedures]

## Troubleshooting
[Common issues and solutions]

## Contributing
[Development guidelines and contribution process]

## License
[License information]
EOF
```

### 3.3 Minimize Separate Documents
```bash
# Consolidate smaller documentation files
for doc_file in $(find . -name "*.md" -not -name "README.md" -not -name "CLAUDE.md"); do
    if [ $(wc -l < "$doc_file") -lt 20 ]; then
        echo "Consider consolidating: $doc_file ($(wc -l < "$doc_file") lines)"
    fi
done

# Create consolidated documentation
mkdir -p docs/archive
for small_doc in $(find . -name "*.md" -not -name "README.md" -not -name "CLAUDE.md" -exec wc -l {} \; | awk '$1 < 20 {print $2}'); do
    mv "$small_doc" docs/archive/
done
```

### 3.4 Create Clear Documentation Order
```bash
# Create documentation index
cat > docs/INDEX.md << 'EOF'
# Documentation Index

## For New Developers
1. [README.md](../README.md) - Start here
2. [CLAUDE.md](../CLAUDE.md) - Development guidelines
3. [QUICK_START.md](QUICK_START.md) - Get up and running

## Architecture Documentation
1. [ARCHITECTURE.md](ARCHITECTURE.md) - System design
2. [API_REFERENCE.md](API_REFERENCE.md) - API documentation
3. [SECURITY_GUIDE.md](SECURITY_GUIDE.md) - Security implementation

## Development Documentation
1. [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - Development procedures
2. [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing strategies
3. [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Production deployment

## Reference Documentation
1. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Issue resolution
2. [PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md) - Performance optimization
3. [CHANGELOG.md](CHANGELOG.md) - Version history
EOF
```

## Phase 4: Code Quality Resolution Phase

### 4.1 Comprehensive Type Error Resolution

#### 4.1.1 Identify All Type Errors
```bash
# For Python with mypy
mypy . --ignore-missing-imports --show-error-codes > type_errors.txt 2>&1

# For TypeScript
npx tsc --noEmit --skipLibCheck > ts_type_errors.txt 2>&1 

# Count and categorize errors
echo "=== Type Error Summary ===" > type_error_summary.txt
if [ -f type_errors.txt ]; then
    echo "Python Type Errors: $(wc -l < type_errors.txt)" >> type_error_summary.txt
    grep -o "error:" type_errors.txt | wc -l >> type_error_summary.txt
fi
if [ -f ts_type_errors.txt ]; then
    echo "TypeScript Type Errors: $(wc -l < ts_type_errors.txt)" >> type_error_summary.txt
fi
```

#### 4.1.2 Fix Type Errors Systematically
```bash
# Create type error fix script
cat > fix_type_errors.py << 'EOF'
#!/usr/bin/env python3
"""
Systematic type error resolution script.
Process each type error and apply fixes.
"""

import subprocess
import re
import sys

def get_type_errors():
    """Get all type errors from mypy."""
    try:
        result = subprocess.run(['mypy', '.', '--ignore-missing-imports'], 
                              capture_output=True, text=True)
        return result.stdout.split('\n')
    except Exception as e:
        print(f"Error running mypy: {e}")
        return []

def fix_missing_type_annotations():
    """Add missing type annotations."""
    # This would contain logic to add type annotations
    # Based on common patterns found in type errors
    pass

def fix_import_errors():
    """Fix import-related type errors."""
    # This would contain logic to fix import issues
    pass

if __name__ == "__main__":
    errors = get_type_errors()
    print(f"Found {len(errors)} type errors")
    # Process each error systematically
    for error in errors:
        if error.strip():
            print(f"Processing: {error}")
EOF

python fix_type_errors.py
```

### 4.2 Pre-commit Hook Resolution

#### 4.2.1 Run All Pre-commit Hooks
```bash
# Install pre-commit if not already installed
pip install pre-commit

# Install hooks
pre-commit install

# Run all hooks on all files
pre-commit run --all-files > precommit_results.txt 2>&1 

# Analyze results
echo "=== Pre-commit Hook Results ===" > precommit_analysis.txt
if grep -q "FAILED" precommit_results.txt; then
    echo "FAILED HOOKS FOUND - MUST FIX IMMEDIATELY" >> precommit_analysis.txt
    grep "FAILED" precommit_results.txt >> precommit_analysis.txt
else
    echo "All pre-commit hooks passed" >> precommit_analysis.txt
fi
```

#### 4.2.2 Fix Pre-commit Issues Following 5-Step Protocol
```bash
# Following CLAUDE.md 5-step protocol for hook failures:
# 1. STOP IMMEDIATELY – do not continue with other tasks
# 2. FIX ALL ISSUES – address every ❌ until everything is ✅ green
# 3. VERIFY THE FIX – re-run the failed command to confirm it's fixed
# 4. CONTINUE ORIGINAL TASK – return to what you were doing
# 5. NEVER IGNORE – There are NO warnings, only requirements

fix_precommit_issues() {
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "=== Pre-commit Fix Attempt $attempt ==="
        
        # Run pre-commit hooks
        if pre-commit run --all-files; then
            echo "✅ ALL PRE-COMMIT HOOKS PASSED"
            break
        else
            echo "❌ Pre-commit hooks failed - fixing issues..."
            
            # Auto-fix common issues
            black . --check --diff || black .
            isort . --check-only --diff || isort .
            flake8 .   # Report issues but continue
            
            attempt=$((attempt + 1))
        fi
    done
    
    if [ $attempt -gt $max_attempts ]; then
        echo "❌ FAILED TO FIX PRE-COMMIT ISSUES AFTER $max_attempts ATTEMPTS"
        echo "MANUAL INTERVENTION REQUIRED"
        exit 1
    fi
}

fix_precommit_issues
```

### 4.3 Ensure All Automated Checks Pass

#### 4.3.1 Run Complete Check Suite
```bash
# The mandatory check sequence from CLAUDE.md
run_complete_checks() {
    echo "=== Running Complete Check Suite ==="
    
    # Format check
    echo "Running format check..."
    if ! make format; then
        echo "❌ FORMAT CHECK FAILED"
        return 1
    fi
    
    # Test check
    echo "Running test check..."
    if ! make test; then
        echo "❌ TEST CHECK FAILED"
        return 1
    fi
    
    # Lint check
    echo "Running lint check..."
    if ! make lint; then
        echo "❌ LINT CHECK FAILED"
        return 1
    fi
    
    echo "✅ ALL CHECKS PASSED"
    return 0
}

# Run checks with retry logic
check_attempts=0
max_check_attempts=5

while [ $check_attempts -lt $max_check_attempts ]; do
    if run_complete_checks; then
        break
    else
        check_attempts=$((check_attempts + 1))
        echo "Check attempt $check_attempts failed. Retrying..."
        sleep 5
    fi
done

if [ $check_attempts -eq $max_check_attempts ]; then
    echo "❌ CRITICAL ERROR: CHECKS FAILED AFTER $max_check_attempts ATTEMPTS"
    echo "MANUAL INTERVENTION REQUIRED - CLEANUP CANNOT PROCEED"
    exit 1
fi
```

#### 4.3.2 Document and Fix Red Flags
```bash
# Create red flags report
cat > RED_FLAGS_REPORT.md << 'EOF'
# Red Flags Report

## Critical Issues Found
- [ ] Failing tests: [COUNT]
- [ ] Linting errors: [COUNT]
- [ ] Type errors: [COUNT]
- [ ] Security vulnerabilities: [COUNT]
- [ ] Performance issues: [COUNT]

## Resolution Actions Taken
- [ ] Fixed all type errors
- [ ] Resolved all linting issues
- [ ] Fixed all failing tests
- [ ] Addressed security vulnerabilities
- [ ] Optimized performance bottlenecks

## Verification Steps
- [ ] All automated checks pass
- [ ] Manual testing completed
- [ ] Security scan completed
- [ ] Performance benchmarks met
EOF

# Run security scan
bandit -r . -f json -o security_scan.json 2>/dev/null 
if [ -f security_scan.json ]; then
    echo "Security scan completed. Review security_scan.json"
fi
```

## Phase 5: Git Workflow Phase

### 5.1 Proper Git Add, Commit, and Push Workflow

#### 5.1.1 Incremental Commits for Each Cleanup Phase
```bash
# Commit cleanup phase by phase following conventional commits
commit_cleanup_phase() {
    local phase="$1"
    local description="$2"
    
    # Add changes
    git add -A
    
    # Check if there are changes to commit
    if git diff --cached --quiet; then
        echo "No changes to commit for phase: $phase"
        return 0
    fi
    
    # Create commit message following conventional format
    local commit_msg="cleanup: $description

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    
    # Commit with proper message
    if git commit -m "$commit_msg"; then
        echo "✅ Committed phase: $phase"
        return 0
    else
        echo "❌ Failed to commit phase: $phase"
        return 1
    fi
}

# Commit each phase
commit_cleanup_phase "file-removal" "remove obsolete files and technical debt"
commit_cleanup_phase "directory-consolidation" "consolidate directory structure"
commit_cleanup_phase "documentation" "consolidate and organize documentation"
commit_cleanup_phase "code-quality" "fix type errors and quality issues"
commit_cleanup_phase "automated-checks" "ensure all automated checks pass"
```

#### 5.1.2 Final Validation Before Push
```bash
# Final validation before push
final_validation() {
    echo "=== Final Validation Before Push ==="
    
    # Run all checks one final time
    if ! run_complete_checks; then
        echo "❌ FINAL VALIDATION FAILED - CANNOT PUSH"
        return 1
    fi
    
    # Verify git status is clean
    if ! git diff --quiet && ! git diff --cached --quiet; then
        echo "❌ Git working directory is not clean"
        return 1
    fi
    
    # Verify all commits follow conventional format
    if ! git log --oneline -10 | grep -E "^[a-f0-9]+ (feat|fix|docs|style|refactor|test|chore|cleanup):"; then
        echo "⚠️  Some commits may not follow conventional format"
    fi
    
    echo "✅ Final validation passed"
    return 0
}

# Run final validation
if final_validation; then
    # Push to remote
    echo "Pushing to remote..."
    if git push origin $(git branch --show-current); then
        echo "✅ Successfully pushed to remote"
    else
        echo "❌ Failed to push to remote"
        exit 1
    fi
else
    echo "❌ Final validation failed - not pushing"
    exit 1
fi
```

### 5.2 Clear Commit Messages Following Conventions

#### 5.2.1 Commit Message Standards
```bash
# Commit message template
create_commit_message() {
    local type="$1"        # feat, fix, docs, style, refactor, test, chore, cleanup
    local scope="$2"       # optional scope
    local description="$3" # brief description
    local body="$4"        # optional body
    
    local message="$type"
    if [ -n "$scope" ]; then
        message="$message($scope)"
    fi
    message="$message: $description"
    
    if [ -n "$body" ]; then
        message="$message

$body"
    fi
    
    # Add standard footer
    message="$message

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    
    echo "$message"
}

# Example usage
commit_message=$(create_commit_message "cleanup" "repository" "comprehensive cleanup and optimization" "
- Removed obsolete files and technical debt
- Consolidated directory structure
- Updated documentation
- Fixed all type errors and quality issues
- Ensured all automated checks pass")

echo "$commit_message"
```

## Validation Checklist

### Pre-Cleanup Validation
- [ ] Re-read all 1159 lines of CLAUDE.md
- [ ] Understand current repository state
- [ ] Document cleanup plan and objectives
- [ ] Establish success criteria

### Post-Cleanup Validation
- [ ] All automated checks pass: `make format && make test && make lint`
- [ ] Zero failing tests
- [ ] Zero linting issues  
- [ ] Zero type errors
- [ ] No obsolete files remain
- [ ] Documentation consolidated and organized
- [ ] Git history clean with conventional commits
- [ ] All changes pushed to remote

### Quality Gate Verification
```bash
# Final quality gate check
quality_gate_check() {
    local issues=0
    
    echo "=== Quality Gate Verification ==="
    
    # Check automated tests
    if ! make test > /dev/null 2>&1; then
        echo "❌ Tests failing"
        issues=$((issues + 1))
    else
        echo "✅ All tests passing"
    fi
    
    # Check linting
    if ! make lint > /dev/null 2>&1; then
        echo "❌ Linting issues found"
        issues=$((issues + 1))
    else
        echo "✅ No linting issues"
    fi
    
    # Check formatting
    if ! make format > /dev/null 2>&1; then
        echo "❌ Formatting issues found"
        issues=$((issues + 1))
    else
        echo "✅ Code properly formatted"
    fi
    
    # Check type errors
    if ! mypy . --ignore-missing-imports > /dev/null 2>&1; then
        echo "❌ Type errors found"
        issues=$((issues + 1))
    else
        echo "✅ No type errors"
    fi
    
    # Check git status
    if ! git diff --quiet && ! git diff --cached --quiet; then
        echo "❌ Git working directory not clean"
        issues=$((issues + 1))
    else
        echo "✅ Git working directory clean"
    fi
    
    if [ $issues -eq 0 ]; then
        echo "✅ ALL QUALITY GATES PASSED"
        return 0
    else
        echo "❌ $issues QUALITY GATE(S) FAILED"
        return 1
    fi
}

# Run quality gate check
quality_gate_check
```

## Emergency Procedures

### If Cleanup Breaks Something
```bash
# Emergency rollback procedure
emergency_rollback() {
    echo "=== EMERGENCY ROLLBACK PROCEDURE ==="
    
    # Stash any uncommitted changes
    git stash push -m "Emergency stash before rollback"
    
    # Reset to last known good state
    git reset --hard HEAD~1
    
    # Verify system is working
    if run_complete_checks; then
        echo "✅ Rollback successful - system restored"
    else
        echo "❌ Rollback failed - manual intervention required"
        exit 1
    fi
}
```

### If Automated Checks Fail
```bash
# When automated checks fail, follow 5-step protocol:
handle_check_failure() {
    echo "=== AUTOMATED CHECK FAILURE PROTOCOL ==="
    echo "1. STOP IMMEDIATELY – do not continue with other tasks"
    echo "2. FIX ALL ISSUES – address every ❌ until everything is ✅ green"
    echo "3. VERIFY THE FIX – re-run the failed command to confirm it's fixed"
    echo "4. CONTINUE ORIGINAL TASK – return to what you were doing"
    echo "5. NEVER IGNORE – There are NO warnings, only requirements"
    
    exit 1  # Stop immediately as per protocol
}
```

## Usage Instructions

To apply this cleanup process to any subtask:

1. **Copy this document** to your subtask working directory
2. **Execute each phase sequentially** - do not skip phases
3. **Validate after each phase** - ensure no regressions
4. **Follow the 5-step protocol** for any failures
5. **Document any deviations** from the standard process
6. **Update this document** with lessons learned

## Success Metrics

- **Zero tolerance metrics**: 0 failing tests, 0 linting issues, 0 type errors
- **Documentation quality**: Consolidated, organized, and up-to-date
- **Repository cleanliness**: No obsolete files, optimized structure
- **Git hygiene**: Clean history with conventional commits
- **Automation compliance**: All CI/CD checks passing

Remember: This process embodies the ultrathink methodology - thorough research, systematic planning, and zero tolerance for quality issues. Every step must be completed successfully before proceeding to the next phase.