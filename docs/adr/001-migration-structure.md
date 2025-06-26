# ADR-001: Project Structure Migration from CogniticNet to FreeAgentics

## Status

Superceded by [ADR-002: Canonical Directory Structure](002-canonical-directory-structure.md)

## Context

This document records the historical context and execution of the initial migration from the legacy `CogniticNet` structure to the new `FreeAgentics` domain-driven structure. The original codebase had accumulated significant technical debt, hindering development.

**This document is for historical purposes only.** For the definitive, up-to-date specification of the project's architecture, please see **[ADR-002: Canonical Directory Structure](002-canonical-directory-structure.md)**.

### Key Problems Identified

1. **Inconsistent Naming**: Mixed references (CogniticNet, cogniticnet, cogneticnet)
2. **Poor Structure**: Frontend/backend coupling, scattered configuration, buried core logic
3. **Gaming Terminology**: PlayerAgent, NPCAgent, EnemyAgent instead of professional multi-agent terms
4. **Missing Standards**: No ADRs, SECURITY.md, GOVERNANCE.md, or proper documentation
5. **Test Organization**: Tests separated from source, making maintenance difficult

## Decision

Transform the entire codebase structure from CogniticNet to FreeAgentics following domain-driven design principles with clear architectural boundaries. The final, canonical structure is detailed in [ADR-002](002-canonical-directory-structure.md).

## Implementation Summary

The migration was executed in a staged approach to minimize risk and preserve git history:

### Stage 1-2: Core Agent Structure & Active Inference

- Moved `agents/basic_agent` → `agents/base`
- Moved `agents/active_inference` → `inference/engine`
- Moved `gnn` → `inference/gnn`
- Moved `llm` → `inference/llm`

### Stage 3-5: Coalition, World, API, Frontend, Tests

- Moved `agents/coalition` → `coalitions/`
- Moved `world` + `spatial` → `world/`
- Moved `app/api` → `api/rest`
- Moved `app/components` + frontend libs → `web/`
- Moved `tests` → `tests/`
- Moved `doc` → `docs/`

### Import Path Updates

- Created `update_import_paths.py` script
- Updated 316 imports across 68 files
- Updated all CogniticNet references to FreeAgentics
- Modified configuration files (tsconfig.json, package.json, pyproject.toml)

### Validation

- Created `validate_migration.py` script
- Verified git history preservation
- Confirmed all critical files moved correctly
- Validated no broken imports remain
- Total: 602 files, 7.36 MB successfully migrated

## Consequences

### Positive

1. **Clear Domain Boundaries**: Core domains are now properly separated
2. **Professional Structure**: Aligns with industry standards and DDD principles
3. **Improved Developer Experience**: Intuitive file locations and naming
4. **Investment Ready**: Professional structure demonstrates technical maturity
5. **Git History Preserved**: All file history maintained for traceability

### Negative

1. **Short-term Disruption**: Developers need to learn new structure
2. **Build Updates Required**: CI/CD pipelines need path updates
3. **Documentation Updates**: All docs referencing old paths need updates

### Mitigation

- Comprehensive migration scripts created for reproducibility
- Rollback tag created: `pre-freeagentics-migration`
- Validation scripts ensure no data loss
- Clear documentation of new structure

## Lessons Learned

1. **Staged Migration Works**: Breaking migration into logical stages reduced risk
2. **Automation Critical**: Scripts for import updates saved hours of manual work
3. **Validation Essential**: Automated validation caught issues early
4. **Git History Valuable**: Preserving history maintains project continuity

## References

- Migration Audit: `/MIGRATION_AUDIT.md`
- Migration Scripts: `/migrate_to_freeagentics.py`, `/migrate_to_freeagentics.sh`
- Validation Report: `/VALIDATION_REPORT.json`
- PRD: `/.taskmaster/docs/prd.txt`
- **Canonical Structure**: [ADR-002: Canonical Directory Structure](002-canonical-directory-structure.md)

## Appendix: Before/After Structure

This section provides a high-level overview of the structural change. For a detailed breakdown of the final structure, refer to [ADR-002](002-canonical-directory-structure.md).

### Before (CogniticNet)

```
CogniticNet/
├── app/                # Mixed frontend/backend
│   ├── api/           # API routes mixed with UI
│   └── components/    # UI components
├──                # Buried backend logic
│   ├── agents/        # Agent code
│   ├── tests/         # Separated tests
│   └── world/         # Environment
├── doc/               # Inconsistent docs
└── environments/      # Misplaced config
```

### After (FreeAgentics)

```
freeagentics/
├── agents/            # Clear agent domain
├── inference/         # Active Inference domain
├── coalitions/        # Coalition domain
├── world/            # Environment domain
├── api/              # Clean API layer
├── web/              # Pure frontend
├── infrastructure/   # Deployment concerns
├── tests/            # Centralized testing
└── docs/             # Consistent docs
```

---

_Decision made by: Martin Fowler (Lead) and Expert Committee_
_Date: 2025-06-18_
_Version: 1.0 (Historical)_
