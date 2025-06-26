# ADR-002: Naming Conventions and Code Standards

## Status

Accepted and Implemented

## Context

Following the successful migration from CogniticNet to FreeAgentics (ADR-001), the codebase exhibited significant naming inconsistencies that hindered readability, maintainability, and professional presentation. The expert committee identified this as a critical Day 2 priority for the 10-day transformation sprint.

### Issues Identified

1. **Inconsistent File Naming**: Mix of camelCase, snake_case, and kebab-case
2. **Gaming Terminology**: PlayerAgent, NPCAgent, spawn() - unprofessional for enterprise software
3. **TypeScript Conventions**: Components using kebab-case instead of PascalCase
4. **Missing Standards**: No interface prefixing, inconsistent method naming
5. **Legacy References**: Remaining "CogniticNet" references throughout codebase

## Decision

Implement comprehensive naming conventions across all languages and establish automated enforcement through tooling.

### Naming Standards Adopted

#### File Naming

- **Python**:
  - **Modules** (imported by other code): snake_case (`belief_update.py`)
  - **Standalone scripts** (not imported): kebab-case (`audit-naming.py`)
  - **Test files** (using importlib): kebab-case (`test-agent-creation.py`)
  - **Special files**: Preserve standard names (`__init__.py`, `setup.py`)
- **TypeScript Components**: PascalCase (`AgentDashboard.tsx`)
- **TypeScript Utilities**: camelCase (`apiClient.ts`)
- **TypeScript Hooks**: camelCase with 'use' prefix (`useAgentState.ts`)
- **Configuration**: kebab-case (`docker-compose.yml`)

#### Code Conventions

**Python**:

```python
class ExplorerAgent(BaseAgent):  # PascalCase classes
    def update_beliefs(self):     # snake_case methods
        MAX_ITERATIONS = 100      # UPPER_SNAKE_CASE constants
```

**TypeScript**:

```typescript
interface IAgent {
  // 'I' prefix for domain interfaces
  beliefState: BeliefState; // camelCase properties
}

const AgentCreator: React.FC = () => {
  // PascalCase components
  const handleSubmit = () => {}; // 'handle' prefix for events
  const MAX_AGENTS = 1000; // UPPER_SNAKE_CASE constants
};
```

#### Prohibited Terms

All gaming terminology replaced with professional multi-agent system terms:

- PlayerAgent → ExplorerAgent
- NPCAgent → AutonomousAgent
- spawn() → initialize()
- GameWorld → Environment

### Python File Naming Rationale

**CRITICAL ARCHITECTURAL CONSIDERATION**: Python's import system does not support hyphens in module names when using standard import statements. Modules with hyphens require complex workarounds using `importlib.util.spec_from_file_location()`, which would necessitate rewriting hundreds of import statements throughout the codebase.

**The Solution**: Distinguish between different Python file types:

1. **Modules** (files imported by other Python code): Must use `snake_case` to maintain normal import functionality
   - Example: `agents/base/state_manager.py` can be imported as `from .state_manager import AgentStateManager`

2. **Standalone Scripts** (executable files not imported): May use `kebab-case` for consistency with other languages
   - Example: `scripts/audit-naming.py` is executed directly, not imported

3. **Test Files** (using importlib for dynamic imports): May use `kebab-case`
   - Example: `tests/unit/test-agent-creation.py` uses `importlib.util.spec_from_file_location()`

This approach maintains **both** the professional kebab-case aesthetic for non-imported files **and** the functional requirements of Python's import system for modules.

## Implementation

### Phase 1: Documentation (Completed)

- Created comprehensive CONTRIBUTING.md with human-readable standards
- Created machine-readable naming-conventions.json for tooling
- Established clear examples and anti-patterns

### Phase 2: Audit (Completed)

- Developed audit-naming.py script
- Scanned 523 files, found 301 violations
- Categorized by severity and type
- Generated detailed reports

### Phase 3: Automated Fixes (Completed)

- Developed fix-naming.py script
- Fixed 36 high-priority violations:
  - 5 prohibited term replacements
  - 6 Python file renames
  - 10 TypeScript component renames
  - 5 configuration file renames
  - 10 code convention fixes
- Used git mv to preserve history
- Updated all import references

### Phase 4: Python Naming Correction (Current)

- **Discovery**: Original "kebab-case for all Python files" broke import system
- **Analysis**: 143+ Python files with underscores, hundreds of import statements affected
- **Solution**: Implemented differentiated naming based on file usage patterns
- **Reverted**: Module renames back to snake_case to restore functionality
- **Updated**: ADR-004 with architectural rationale and proper exceptions

### Phase 5: Documentation Updates (Planned)

- Update all documentation to reflect corrected conventions
- Update CONTRIBUTING.md with Python naming exceptions
- Update tooling to enforce differentiated Python naming

## Consequences

### Positive

1. **Professional Appearance**: No gaming terminology, enterprise-ready naming
2. **Consistency**: Single standard across all languages
3. **Tooling Support**: Machine-readable format enables automation
4. **Developer Experience**: Clear patterns reduce cognitive load
5. **Investment Ready**: Shows attention to code quality

### Negative

1. **Learning Curve**: Developers must adapt to new conventions
2. **Ongoing Maintenance**: Need to monitor and fix violations
3. **Import Updates**: File renames require import updates

### Mitigations

- Comprehensive documentation in CONTRIBUTING.md
- Automated tooling for detection and fixes
- Pre-commit hooks (to be implemented)
- Regular audits in CI/CD pipeline

## Metrics

### Before

- 301 naming violations
- 6 prohibited gaming terms
- 111 incorrectly named Python files
- 71 incorrectly named TypeScript components

### After Phase 3

- 265 violations remaining (mostly lower priority)
- 0 prohibited gaming terms
- Consistent file naming in critical paths
- Professional terminology throughout

## Next Steps

1. Complete systematic audit with corrected Python naming rules
2. Update tooling (audit-naming.py, fix-naming.py) to enforce differentiated Python naming
3. Update CONTRIBUTING.md and naming-conventions.json with corrected standards
4. Implement pre-commit hooks with proper Python file type detection
5. Add CI/CD enforcement for differentiated naming conventions
6. Regular audits to prevent regression

## References

- [CONTRIBUTING.md](/CONTRIBUTING.md) - Human-readable conventions
- [naming-conventions.json](/docs/standards/naming-conventions.json) - Machine-readable rules
- [audit-naming.py](/scripts/audit-naming.py) - Audit tool
- [fix-naming.py](/scripts/fix-naming.py) - Automated fixes

## Revision History

**Version 1.1** (2025-06-21): Critical Python naming correction
- Added architectural rationale for Python file naming exceptions
- Corrected "kebab-case for all Python files" to differentiated naming
- Distinguished modules (snake_case) vs scripts (kebab-case) vs tests (kebab-case)
- Updated implementation phases and next steps

**Version 1.0** (2025-06-18): Initial naming convention establishment
- Established baseline naming standards across all languages
- Replaced gaming terminology with professional terms
- Implemented initial tooling and audit processes

---

_Original Decision: Robert Martin (Clean Code Lead)_
_Python Naming Correction: Task Master AI Analysis_
_Date: 2025-06-21_
_Version: 1.1_
