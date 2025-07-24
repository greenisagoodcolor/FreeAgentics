# Cleanup Tools Quick Reference

## Overview

This directory contains a comprehensive cleanup system designed to systematically apply repository cleanup to each subtask following CLAUDE.md guidelines.

## Files Created

### 1. `COMPREHENSIVE_CLEANUP_PROCESS.md`

- **Purpose**: Detailed documentation of the 5-phase cleanup process
- **Usage**: Read this to understand the complete methodology
- **Key Features**:
  - Phase-by-phase breakdown
  - Specific commands for each step
  - Validation checkpoints
  - Emergency procedures
  - Success metrics

### 2. `validate_cleanup.py`

- **Purpose**: Automated validation script to verify cleanup completion
- **Usage**: `python3 validate_cleanup.py`
- **Key Features**:
  - Validates all automated checks pass
  - Checks for type errors
  - Verifies pre-commit hooks
  - Validates documentation consolidation
  - Generates comprehensive JSON report

### 3. `run_cleanup.sh`

- **Purpose**: Master orchestration script for the entire cleanup process
- **Usage**: `./run_cleanup.sh`
- **Key Features**:
  - Executes all 5 phases automatically
  - Follows CLAUDE.md zero tolerance approach
  - Includes error handling and rollback
  - Creates proper git commits for each phase
  - Provides colored output and progress tracking

## Quick Start

### For Any Subtask Cleanup

1. **Copy cleanup tools to your subtask directory**:

   ```bash
   cp COMPREHENSIVE_CLEANUP_PROCESS.md your_subtask_dir/
   cp validate_cleanup.py your_subtask_dir/
   cp run_cleanup.sh your_subtask_dir/
   cd your_subtask_dir/
   ```

2. **Run the full cleanup process**:

   ```bash
   ./run_cleanup.sh
   ```

3. **Or run validation only**:
   ```bash
   ./run_cleanup.sh --validate
   ```

### Manual Step-by-Step Process

If you prefer to run cleanup manually:

1. **Read CLAUDE.md guidelines** (mandatory):

   ```bash
   # Re-read all 1159 lines before starting
   cat CLAUDE.md | head -50
   grep -n "MANDATORY\|BLOCKING\|NON-NEGOTIABLE" CLAUDE.md
   ```

2. **Phase 1: Research & Planning**:

   ```bash
   # Analyze repository state
   git status --porcelain | wc -l
   find . -name "*.py" -type f | wc -l
   find . -name "*test*.py" | wc -l
   ```

3. **Phase 2: Repository Cleanup**:

   ```bash
   # Remove obsolete files
   rm -rf build/ dist/ *.egg-info/ .pytest_cache/
   find . -name "*.pyc" -delete
   find . -name "__pycache__" -type d -exec rm -rf {} +
   ```

4. **Phase 3: Documentation Consolidation**:

   ```bash
   # Consolidate documentation
   mkdir -p docs/archive
   # Move small docs to archive
   ```

5. **Phase 4: Code Quality Resolution**:

   ```bash
   # Fix all issues following zero tolerance approach
   make format && make test && make lint
   mypy . --ignore-missing-imports
   pre-commit run --all-files
   ```

6. **Phase 5: Git Workflow**:
   ```bash
   # Validate and commit
   python3 validate_cleanup.py
   git add -A && git commit -m "cleanup: comprehensive cleanup completed"
   ```

## Command Reference

### Master Cleanup Script

```bash
./run_cleanup.sh              # Run full cleanup process
./run_cleanup.sh --validate   # Run validation only
./run_cleanup.sh --help       # Show help
```

### Validation Script

```bash
python3 validate_cleanup.py   # Run all validation checks
python3 validate_cleanup.py --help  # Show help
```

### Key Validation Checks

- ✅ All automated checks pass (`make format && make test && make lint`)
- ✅ Zero type errors (`mypy . --ignore-missing-imports`)
- ✅ Pre-commit hooks pass (`pre-commit run --all-files`)
- ✅ No obsolete files remain
- ✅ Documentation consolidated
- ✅ Git working directory clean
- ✅ Conventional commit messages
- ✅ Adequate test coverage
- ✅ Security baseline met
- ✅ Performance baseline met

## Integration with Subtasks

### Before Starting Any Subtask

1. Copy cleanup tools to subtask directory
2. Run initial validation: `./run_cleanup.sh --validate`
3. Document current state

### During Subtask Development

1. Follow TDD and quality practices from CLAUDE.md
2. Run validation periodically: `python3 validate_cleanup.py`
3. Fix issues immediately (zero tolerance approach)

### After Completing Subtask

1. Run full cleanup: `./run_cleanup.sh`
2. Verify all validations pass
3. Commit with proper conventional format
4. Update documentation with lessons learned

## Error Handling

### If Cleanup Fails

The scripts follow CLAUDE.md 5-step protocol:

1. **STOP IMMEDIATELY** - do not continue with other tasks
2. **FIX ALL ISSUES** - address every ❌ until everything is ✅ green
3. **VERIFY THE FIX** - re-run the failed command to confirm it's fixed
4. **CONTINUE ORIGINAL TASK** - return to what you were doing
5. **NEVER IGNORE** - There are NO warnings, only requirements

### Emergency Rollback

```bash
# If cleanup breaks something
git stash push -m "Emergency stash before rollback"
git reset --hard HEAD~1
```

## Quality Gates

### Mandatory Checks (Zero Tolerance)

- ❌ **BLOCKING**: Failing tests
- ❌ **BLOCKING**: Linting errors
- ❌ **BLOCKING**: Type errors
- ❌ **BLOCKING**: Pre-commit hook failures
- ❌ **BLOCKING**: Security vulnerabilities
- ❌ **BLOCKING**: Uncommitted changes

### Success Criteria

- ✅ All automated checks pass
- ✅ Documentation consolidated and up-to-date
- ✅ Repository optimized and clean
- ✅ Git history follows conventional commits
- ✅ All quality metrics meet thresholds

## File Structure After Cleanup

```
repository/
├── README.md                           # Consolidated main documentation
├── CLAUDE.md                          # Development guidelines
├── COMPREHENSIVE_CLEANUP_PROCESS.md    # Cleanup process documentation
├── validate_cleanup.py                # Validation script
├── run_cleanup.sh                     # Master cleanup script
├── docs/
│   ├── INDEX.md                       # Documentation index
│   ├── ARCHITECTURE.md                # System design
│   ├── DEVELOPMENT_GUIDE.md           # Development procedures
│   └── archive/                       # Archived small docs
├── src/                               # Source code
├── tests/                             # Test code
└── cleanup_validation_report.json     # Latest validation report
```

## Best Practices

### 1. Always Follow the Sequence

- Never skip phases
- Complete each phase before moving to next
- Validate after each phase

### 2. Zero Tolerance Approach

- Fix all issues immediately
- No warnings ignored
- All checks must pass

### 3. Documentation

- Update CLAUDE.md with lessons learned
- Document any deviations from standard process
- Maintain validation reports

### 4. Git Hygiene

- Use conventional commit messages
- Commit each phase separately
- Keep working directory clean

## Troubleshooting

### Common Issues

**1. Pre-commit hooks failing**

```bash
# Fix formatting issues
black .
isort .
# Re-run hooks
pre-commit run --all-files
```

**2. Type errors**

```bash
# Check specific errors
mypy . --ignore-missing-imports
# Fix type annotations as needed
```

**3. Test failures**

```bash
# Run tests with verbose output
python -m pytest -v
# Fix failing tests immediately
```

**4. Make commands not found**

```bash
# Check if Makefile exists
ls -la Makefile
# Run commands manually if needed
python -m pytest
flake8 .
```

## Success Metrics

After successful cleanup:

- **Zero** failing tests
- **Zero** linting issues
- **Zero** type errors
- **Zero** security vulnerabilities
- **Zero** uncommitted changes
- **100%** automated checks passing
- **Consolidated** documentation
- **Optimized** repository structure

## Integration with CI/CD

The cleanup tools integrate with CI/CD pipelines:

- Validation script can be run in CI
- All checks must pass before deployment
- Quality gates enforced automatically
- Comprehensive reporting available

## Next Steps

After successful cleanup:

1. Review validation report (`cleanup_validation_report.json`)
2. Update any project-specific configurations
3. Share lessons learned with team
4. Schedule regular cleanup maintenance
5. Continue with subtask development

Remember: This cleanup process embodies the **ultrathink methodology** - thorough research, systematic planning, and zero tolerance for quality issues. Every step must be completed successfully before proceeding to the next phase.
