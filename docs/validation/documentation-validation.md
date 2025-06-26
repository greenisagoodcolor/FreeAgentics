# Documentation Validation Guide

This guide provides instructions and checklists for validating that documentation adheres to the architectural decisions outlined in ADR-002 (Canonical Directory Structure), ADR-003 (Dependency Rules), and ADR-004 (Naming Conventions).

## ADR-002: Canonical Directory Structure Validation

### Documentation Location Checklist

All documentation must be located in the appropriate directories according to ADR-002:

- [ ] General project documentation is in the `docs/` directory
- [ ] Architecture documentation is in the `docs/architecture/` directory
- [ ] Architecture Decision Records (ADRs) are in the `docs/architecture/decisions/` directory
- [ ] API documentation is in the `docs/api/` directory
- [ ] Tutorial documentation is in the `docs/tutorials/` directory
- [ ] User guides are in the `docs/guides/` directory
- [ ] No documentation files exist outside of the `docs/` directory

### Directory Structure Reference

```
docs/
├── api/                # API documentation
│   ├── index.md        # API documentation index
│   ├── rest-api.md     # REST API documentation
│   ├── gnn-api.md      # GNN API documentation
│   ├── agents-api.md   # Agents API documentation
│   └── openapi.yml     # OpenAPI specification
├── architecture/       # Architecture documentation
│   ├── index.md        # Architecture documentation index
│   ├── diagrams.md     # Architecture diagrams
│   └── decisions/      # Architecture Decision Records
│       ├── 001-migration-structure.md
│       ├── 002-canonical-directory-structure.md
│       ├── 003-dependency-rules.md
│       └── 004-naming-conventions.md
├── tutorials/          # Tutorial documentation
│   ├── index.md        # Tutorial index
│   ├── creating-an-agent.md
│   ├── coalition-formation.md
│   └── edge-deployment.md
├── guides/             # User guides
│   ├── index.md        # User guides index
│   ├── installation.md
│   └── configuration.md
├── glossary.md         # Glossary of technical terms
└── index.md            # Documentation root index
```

### Validation Script

You can use the following script to validate the documentation structure:

```bash
#!/bin/bash

# Check that all documentation is in the docs/ directory
echo "Checking for documentation files outside of docs/..."
DOC_FILES_OUTSIDE=$(find . -type f -not -path "./docs/*" -not -path "./.git/*" -name "*.md" | wc -l)
if [ $DOC_FILES_OUTSIDE -gt 0 ]; then
    echo "ERROR: Found $DOC_FILES_OUTSIDE documentation files outside of docs/ directory"
    find . -type f -not -path "./docs/*" -not -path "./.git/*" -name "*.md" | sort
else
    echo "OK: All documentation files are in the docs/ directory"
fi

# Check that ADRs are in the correct directory
echo "Checking ADR locations..."
ADR_FILES_OUTSIDE=$(find ./docs -type f -name "*adr*.md" -not -path "./docs/architecture/decisions/*" | wc -l)
if [ $ADR_FILES_OUTSIDE -gt 0 ]; then
    echo "ERROR: Found $ADR_FILES_OUTSIDE ADR files outside of docs/architecture/decisions/ directory"
    find ./docs -type f -name "*adr*.md" -not -path "./docs/architecture/decisions/*" | sort
else
    echo "OK: All ADR files are in the docs/architecture/decisions/ directory"
fi

# Check that API documentation is in the correct directory
echo "Checking API documentation locations..."
API_FILES_OUTSIDE=$(find ./docs -type f -name "*api*.md" -not -path "./docs/api/*" | wc -l)
if [ $API_FILES_OUTSIDE -gt 0 ]; then
    echo "ERROR: Found $API_FILES_OUTSIDE API documentation files outside of docs/api/ directory"
    find ./docs -type f -name "*api*.md" -not -path "./docs/api/*" | sort
else
    echo "OK: All API documentation files are in the docs/api/ directory"
fi

# Check that tutorial documentation is in the correct directory
echo "Checking tutorial documentation locations..."
TUTORIAL_FILES_OUTSIDE=$(find ./docs -type f -name "*tutorial*.md" -not -path "./docs/tutorials/*" | wc -l)
if [ $TUTORIAL_FILES_OUTSIDE -gt 0 ]; then
    echo "ERROR: Found $TUTORIAL_FILES_OUTSIDE tutorial files outside of docs/tutorials/ directory"
    find ./docs -type f -name "*tutorial*.md" -not -path "./docs/tutorials/*" | sort
else
    echo "OK: All tutorial files are in the docs/tutorials/ directory"
fi

echo "Documentation structure validation complete"
```

## ADR-003: Dependency Rules Validation

### Documentation Dependency Checklist

Documentation should accurately reflect the dependency rules defined in ADR-003:

- [ ] Documentation correctly describes the Core Domain as independent from Interface and Infrastructure layers
- [ ] Documentation correctly describes the Interface layer as dependent on the Core Domain
- [ ] Documentation correctly describes the Infrastructure layer as potentially dependent on any layer
- [ ] Architecture diagrams correctly show the dependency flow (inward toward Core Domain)
- [ ] No documentation suggests or encourages violating the dependency rules
- [ ] Examples in documentation follow the dependency rules

### Dependency Rules Reference

1. **Core Domain Independence**:
   - Core Domain modules (`agents/`, `inference/`, `coalitions/`, `world/`) must not import from Interface or Infrastructure layers
   - Documentation should not suggest or show examples of Core Domain modules importing from these layers

2. **Interface Layer Dependencies**:
   - Interface modules (`api/`, `web/`) can import from the Core Domain
   - Documentation should show correct import patterns for Interface modules

3. **Infrastructure Dependencies**:
   - Infrastructure modules can import from any layer
   - Documentation should clarify when and how Infrastructure modules should interact with other layers

### Validation Checklist

For each documentation file, check:

- [ ] Import examples follow the dependency rules
- [ ] Architecture diagrams show correct dependency flow
- [ ] No examples or instructions violate dependency rules
- [ ] Explanations of the architecture correctly describe the dependency constraints

## ADR-004: Naming Conventions Validation

### Documentation Naming Checklist

Documentation should follow and correctly describe the naming conventions defined in ADR-004:

- [ ] Documentation file names follow kebab-case convention (e.g., `dependency-validation-guide.md`)
- [ ] Documentation correctly describes Python file naming conventions (kebab-case)
- [ ] Documentation correctly describes TypeScript component naming conventions (PascalCase)
- [ ] Documentation correctly describes TypeScript utility naming conventions (camelCase)
- [ ] Documentation correctly describes configuration file naming conventions (kebab-case)
- [ ] Code examples in documentation follow the naming conventions for their respective languages
- [ ] No documentation uses or encourages the use of prohibited terms (gaming terminology)

### Naming Conventions Reference

1. **File Naming**:
   - Python files: kebab-case (e.g., `belief-update.py`)
   - TypeScript Components: PascalCase (e.g., `AgentDashboard.tsx`)
   - TypeScript Utilities: camelCase (e.g., `apiClient.ts`)
   - Configuration: kebab-case (e.g., `docker-compose.yml`)

2. **Code Conventions**:
   - Python:
     - Classes: PascalCase (e.g., `ExplorerAgent`)
     - Methods: snake_case (e.g., `update_beliefs`)
     - Constants: UPPER_SNAKE_CASE (e.g., `MAX_ITERATIONS`)
   - TypeScript:
     - Interfaces: 'I' prefix with PascalCase (e.g., `IAgent`)
     - Properties: camelCase (e.g., `beliefState`)
     - Components: PascalCase (e.g., `AgentCreator`)
     - Event handlers: 'handle' prefix (e.g., `handleSubmit`)
     - Constants: UPPER_SNAKE_CASE (e.g., `MAX_AGENTS`)

3. **Prohibited Terms**:
   - ❌ PlayerAgent → ✅ ExplorerAgent
   - ❌ NPCAgent → ✅ AutonomousAgent
   - ❌ spawn() → ✅ initialize()
   - ❌ GameWorld → ✅ Environment

### Validation Script

You can use the following script to validate documentation naming conventions:

```bash
#!/bin/bash

# Check documentation file naming conventions
echo "Checking documentation file naming conventions..."
NON_KEBAB_CASE_FILES=$(find ./docs -type f -name "*.md" | grep -v "^[a-z0-9-]\+\.md$" | wc -l)
if [ $NON_KEBAB_CASE_FILES -gt 0 ]; then
    echo "ERROR: Found $NON_KEBAB_CASE_FILES documentation files not following kebab-case convention"
    find ./docs -type f -name "*.md" | grep -v "^[a-z0-9-]\+\.md$" | sort
else
    echo "OK: All documentation files follow kebab-case convention"
fi

# Check for prohibited terms in documentation
echo "Checking for prohibited terms in documentation..."
PROHIBITED_TERMS=("PlayerAgent" "NPCAgent" "spawn(" "GameWorld")
for term in "${PROHIBITED_TERMS[@]}"; do
    COUNT=$(grep -r "$term" --include="*.md" ./docs | wc -l)
    if [ $COUNT -gt 0 ]; then
        echo "ERROR: Found prohibited term '$term' $COUNT times in documentation"
        grep -r "$term" --include="*.md" ./docs
    else
        echo "OK: Prohibited term '$term' not found in documentation"
    fi
done

echo "Documentation naming convention validation complete"
```

## Combined Validation Process

To ensure comprehensive validation of documentation against all three ADRs, follow these steps:

1. **Run the validation scripts** provided above to automatically check for common issues
2. **Manually review documentation** using the checklists provided
3. **Fix any issues** identified by the validation scripts or manual review
4. **Update examples** in documentation to ensure they follow the conventions and rules
5. **Review architecture diagrams** to ensure they correctly represent the dependency rules
6. **Check cross-references** between documentation files to ensure consistency

## Example Issues and Fixes

### Directory Structure Issue

**Issue**: API documentation file found in the wrong location:
```
docs/guides/api-usage.md
```

**Fix**: Move the file to the correct location:
```bash
mkdir -p docs/api
mv docs/guides/api-usage.md docs/api/usage.md
```

### Dependency Rule Issue

**Issue**: Documentation example shows Core Domain importing from Interface layer:
```python
# Example in docs/tutorials/creating-an-agent.md
from api.rest.endpoints import AgentEndpoint  # Violates dependency rule

class ExplorerAgent(BaseAgent):
    # ...
```

**Fix**: Update the example to follow dependency rules:
```python
# Corrected example
from agents.base.agent_interface import AgentInterface  # Follows dependency rules

class ExplorerAgent(BaseAgent):
    # ...
```

### Naming Convention Issue

**Issue**: Documentation uses prohibited gaming terminology:
```markdown
When you spawn a new agent, it will appear in the GameWorld...
```

**Fix**: Update the text to use approved terminology:
```markdown
When you initialize a new agent, it will appear in the Environment...
```

## Conclusion

Regular validation of documentation against ADR-002, ADR-003, and ADR-004 ensures consistency and adherence to the project's architectural decisions. Use the provided checklists and scripts to maintain high-quality documentation that correctly reflects the project's structure, dependencies, and naming conventions.
