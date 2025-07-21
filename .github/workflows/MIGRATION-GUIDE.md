# 🔄 Pipeline Migration Guide

## Overview

This guide helps migrate from multiple scattered workflow files to a single, unified pipeline that follows Martin Fowler and Jessica Kerr's principles.

## Current State Analysis

### Existing Workflow Files to Consolidate:
1. `ci.yml` - Basic CI/CD pipeline
2. `unified-pipeline.yml` - Comprehensive pipeline (good foundation)
3. `security-tests.yml` - Security testing workflow
4. `security-ci.yml` - Security CI workflow
5. `security-scan.yml` - Security scanning workflow
6. `security-scanning.yml` - Another security scanning workflow
7. `coverage.yml` - Code coverage workflow
8. `performance.yml` - Performance testing
9. `performance-benchmarks.yml` - Performance benchmarking
10. `performance-monitoring.yml` - Performance monitoring
11. `performance-regression-check.yml` - Performance regression checks
12. `production-deployment.yml` - Production deployment workflow
13. `tdd-validation.yml` - TDD validation workflow
14. `docker-multiarch.yml` - Docker multi-arch builds

### Issues with Current Setup:
- **Redundancy**: Multiple workflows doing similar tasks
- **Lack of Coordination**: No clear dependencies between workflows
- **Bypass Mechanisms**: Some workflows may have skip conditions
- **Limited Visibility**: No unified view of pipeline state
- **Inconsistent Standards**: Different approaches across workflows

## Migration Strategy

### Phase 1: Preparation (Week 1)
1. **Audit Dependencies**
   - Review each workflow for unique features
   - Document any custom scripts or tools
   - Identify environment-specific configurations

2. **Backup Current State**
   ```bash
   # Create backup of current workflows
   mkdir -p .github/workflows/.archive
   cp .github/workflows/*.yml .github/workflows/.archive/
   ```

3. **Test New Pipeline**
   - Deploy `main-pipeline.yml` alongside existing workflows
   - Run in parallel to validate functionality
   - Compare results and timing

### Phase 2: Gradual Migration (Week 2-3)

1. **Disable Redundant Workflows**
   ```yaml
   # Add to top of each redundant workflow
   name: "[DEPRECATED] Original Workflow Name"
   on:
     workflow_dispatch: # Only manual trigger during migration
   ```

2. **Consolidation Order**:
   - First: Merge all security workflows → main-pipeline security stages
   - Second: Merge performance workflows → main-pipeline performance stage
   - Third: Merge deployment workflows → main-pipeline deployment stages
   - Last: Disable original ci.yml

3. **Update Branch Protection Rules**
   - Update required status checks to use new pipeline jobs
   - Remove old workflow requirements

### Phase 3: Cutover (Week 4)

1. **Final Validation**
   - Run A/B testing between old and new pipelines
   - Verify all quality gates are enforced
   - Confirm no bypass mechanisms exist

2. **Archive Old Workflows**
   ```bash
   # Move deprecated workflows to archive
   mv .github/workflows/ci.yml .github/workflows/.archive/
   mv .github/workflows/security-*.yml .github/workflows/.archive/
   mv .github/workflows/performance-*.yml .github/workflows/.archive/
   mv .github/workflows/coverage.yml .github/workflows/.archive/
   mv .github/workflows/tdd-validation.yml .github/workflows/.archive/
   mv .github/workflows/production-deployment.yml .github/workflows/.archive/
   mv .github/workflows/docker-multiarch.yml .github/workflows/.archive/
   ```

3. **Update Documentation**
   - Update README with new pipeline information
   - Update contribution guidelines
   - Update deployment documentation

## Configuration Mapping

### Environment Variables
Consolidate all environment variables into the main pipeline:

```yaml
env:
  # From various workflows
  PYTHON_VERSION: "3.12"  # Upgraded from 3.11
  NODE_VERSION: "18"
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
  # Add any missing environment variables here
```

### Secrets Management
Ensure all required secrets are available:
- `GITHUB_TOKEN` (automatic)
- Any deployment-specific secrets
- Third-party integration tokens

### Service Dependencies
Consolidated service definitions:

```yaml
services:
  postgres:
    image: postgres:15  # Upgraded from postgres:14
    env:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: freeagentics_test
    options: >-
      --health-cmd pg_isready
      --health-interval 10s
      --health-timeout 5s
      --health-retries 5
    ports:
      - 5432:5432
  
  redis:
    image: redis:7-alpine
    options: >-
      --health-cmd "redis-cli ping"
      --health-interval 10s
      --health-timeout 5s
      --health-retries 5
    ports:
      - 6379:6379
```

## Validation Checklist

### Pre-Migration
- [ ] All workflows documented
- [ ] Dependencies identified
- [ ] Test environment ready
- [ ] Rollback plan prepared

### During Migration
- [ ] Security stages working
- [ ] Performance benchmarks running
- [ ] Test coverage maintained
- [ ] Build artifacts generated
- [ ] Deployments successful

### Post-Migration
- [ ] All quality gates enforced
- [ ] No skip mechanisms
- [ ] Pipeline visibility improved
- [ ] Metrics being collected
- [ ] Team trained on new pipeline

## Benefits After Migration

1. **Single Source of Truth**: One pipeline file to maintain
2. **Clear Dependencies**: Explicit stage dependencies
3. **No Bypass Options**: All quality gates mandatory
4. **Better Visibility**: Unified pipeline view with visual representation
5. **Consistent Standards**: Same approach across all stages
6. **Faster Feedback**: Optimized stage execution
7. **Comprehensive Metrics**: Full pipeline observability

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Check if all required tools are installed in the new pipeline
   - Verify all environment variables are set

2. **Failed Tests**
   - Compare test commands between old and new workflows
   - Check for missing test configurations

3. **Deployment Issues**
   - Verify deployment credentials
   - Check environment-specific configurations

4. **Performance Differences**
   - Compare resource allocations
   - Check for parallel vs sequential execution

### Rollback Procedure

If issues arise:
1. Re-enable original workflows from archive
2. Disable main-pipeline.yml
3. Update branch protection rules
4. Investigate and fix issues
5. Retry migration

## Long-term Maintenance

### Regular Reviews
- Monthly: Review pipeline metrics
- Quarterly: Update dependencies
- Annually: Major version upgrades

### Continuous Improvement
- Monitor pipeline execution times
- Optimize slow stages
- Update quality thresholds
- Incorporate new security tools

### Documentation
- Keep PIPELINE-ARCHITECTURE.md updated
- Document any customizations
- Maintain runbooks for common issues

## Support

For questions or issues during migration:
- Create an issue with label `pipeline-migration`
- Reference this guide and specific step
- Include relevant logs and configurations

---

*Migration Guide Version 1.0 - Following Martin Fowler's CI/CD principles and Jessica Kerr's system thinking approach*