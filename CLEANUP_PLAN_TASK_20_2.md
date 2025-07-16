# Comprehensive Cleanup Plan for Task 20.2

## Current State Analysis
Date: 2025-07-16
Task: 20.2 - Profile and Optimize Memory Usage

### Repository State Snapshot
- **Python cache files**: 2,881 files (__pycache__, *.pyc, *.pyo)
- **Backup files**: 6+ files (*.backup, *.bak, *.old)
- **Log files**: 10+ files (*.log, *.out)
- **Empty directories**: 6+ directories
- **Build artifacts**: Potentially in dist/, build/, htmlcov/
- **Test reports**: Multiple in .archive/test_reports/

### Tech Debt Identified
1. Excessive Python cache files accumulation
2. Backup files not properly managed
3. Log files scattered across repository
4. Empty directories from removed features
5. Old test reports in archive

## Cleanup Plan

### Phase 2: Repository Cleanup (45 min)
**Objective**: Remove all unnecessary files and optimize repository structure

#### Step 1: Python Cache Cleanup (5 min)
```bash
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type d -name "__pycache__" -exec rm -rf {} +
```

#### Step 2: Backup File Management (5 min)
- Review each backup file for necessity
- Move critical backups to .archive/backups/
- Delete obsolete backups
```bash
find . -name "*.backup" -o -name "*.bak" -o -name "*.old"
```

#### Step 3: Log File Cleanup (5 min)
- Keep only recent production validation logs
- Archive important logs to .archive/logs/
- Delete development logs
```bash
find . -name "*.log" -mtime +7 -delete
```

#### Step 4: Build Artifacts Removal (5 min)
```bash
rm -rf dist/ build/ htmlcov/ .pytest_cache/
rm -rf .coverage* .mypy_cache/
```

#### Step 5: Empty Directory Cleanup (5 min)
```bash
find . -type d -empty -delete
```

#### Step 6: Node Modules Optimization (10 min)
- Check if node_modules is accidentally committed
- Clean npm cache if needed
```bash
cd web && npm cache clean --force
```

#### Step 7: Test Reports Archive (10 min)
- Keep only last 3 test report sets
- Compress older reports

### Phase 3: Documentation Consolidation (30 min)
**Objective**: Organize and consolidate documentation

#### Documentation Structure
```
docs/
├── README.md (Main entry point)
├── ARCHITECTURE.md (System architecture)
├── DEVELOPMENT.md (Development guide)
├── DEPLOYMENT.md (Deployment guide)
├── api/ (API documentation)
├── security/ (Security guides)
├── operations/ (Operational runbooks)
└── archived/ (Old documentation)
```

#### Consolidation Tasks
1. Merge duplicate security documentation
2. Update README with clear navigation
3. Archive obsolete documentation
4. Create clear onboarding path
5. Update CLAUDE.md with Task 20.2 learnings

### Phase 4: Code Quality Resolution (60 min)
**Objective**: Ensure all quality checks pass

#### Quality Checklist
- [ ] Run `make format` - Fix all formatting issues
- [ ] Run `make test` - Ensure all tests pass
- [ ] Run `make lint` - Fix all linting issues
- [ ] Fix all type errors
- [ ] Resolve pre-commit hook issues
- [ ] Security vulnerability check

#### 5-Step Protocol for Failures
1. STOP IMMEDIATELY when issue found
2. FIX ALL ISSUES in the category
3. VERIFY THE FIX by re-running
4. CONTINUE to next check
5. NEVER IGNORE any issue

### Phase 5: Git Workflow (15 min)
**Objective**: Properly commit all changes

#### Git Commands
```bash
# Stage all changes
git add .

# Commit with conventional message
git commit -m "cleanup: comprehensive repository cleanup for task 20.2

- Removed 2,881 Python cache files
- Cleaned backup and log files
- Removed empty directories
- Consolidated documentation
- Fixed all quality issues
- Optimized repository structure"

# Push changes
git push
```

## Validation Checkpoints

### Pre-Cleanup
- [x] Document current state
- [x] Create cleanup plan
- [ ] Backup critical files

### During Cleanup
- [ ] Verify each deletion
- [ ] Test after major changes
- [ ] Keep cleanup log

### Post-Cleanup
- [ ] All tests passing
- [ ] No type errors
- [ ] Clean git status
- [ ] Repository size reduced
- [ ] Documentation organized

## Rollback Procedures

If issues occur:
1. `git stash` - Save current changes
2. `git reset --hard HEAD` - Revert to last commit
3. Review what went wrong
4. Apply fixes incrementally
5. Re-run validation

## Success Criteria

- Zero Python cache files
- No unnecessary backup files
- Organized documentation structure
- All quality checks passing (green)
- Clean git working directory
- Repository size optimized
- Task 20.2 learnings documented in CLAUDE.md