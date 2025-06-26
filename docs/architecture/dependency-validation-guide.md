# Dependency Validation Guide

This guide provides practical tools and methods for validating architectural dependencies according to ADR-003 (Dependency Rules).

## Quick Reference: Dependency Rules

### Core Principle
All dependencies MUST flow inward toward the domain core:

```
Infrastructure → Interface → Application → Domain
```

### Directory Dependency Matrix

| From Layer | Can Import From | Cannot Import From |
|------------|-----------------|-------------------|
| `agents/`, `inference/`, `coalitions/`, `world/` (Domain) | Only other domain modules | `api/`, `web/`, `infrastructure/`, `config/` |
| `api/`, `web/` (Interface) | Domain modules | `infrastructure/`, `config/` (directly) |
| `infrastructure/`, `config/` (Infrastructure) | Any layer | N/A (outermost layer) |

## Validation Tools

### 1. Automated Dependency Checker

```python
#!/usr/bin/env python3
"""
Dependency validation script for FreeAgentics architecture.
Run: python scripts/validate-dependencies.py
"""

import ast
import os
import sys
from pathlib import Path
from typing import Set, Dict, List

class DependencyChecker:
    """Validates architectural dependency rules per ADR-003."""

    DOMAIN_DIRS = {'agents', 'inference', 'coalitions', 'world'}
    INTERFACE_DIRS = {'api', 'web'}
    INFRASTRUCTURE_DIRS = {'infrastructure', 'config', 'data'}

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.violations = []

    def check_file(self, file_path: Path) -> List[str]:
        """Check a single Python file for dependency violations."""
        violations = []

        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())

            imports = self._extract_imports(tree)
            file_layer = self._get_layer(file_path)

            for import_path in imports:
                import_layer = self._get_layer_from_import(import_path)

                if not self._is_valid_dependency(file_layer, import_layer):
                    violations.append(
                        f"VIOLATION: {file_path} ({file_layer}) imports from "
                        f"{import_path} ({import_layer})"
                    )

        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")

        return violations

    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract all import statements from AST."""
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)

        return imports

    def _get_layer(self, file_path: Path) -> str:
        """Determine which architectural layer a file belongs to."""
        relative_path = file_path.relative_to(self.project_root)
        top_dir = relative_path.parts[0]

        if top_dir in self.DOMAIN_DIRS:
            return 'domain'
        elif top_dir in self.INTERFACE_DIRS:
            return 'interface'
        elif top_dir in self.INFRASTRUCTURE_DIRS:
            return 'infrastructure'
        else:
            return 'unknown'

    def _get_layer_from_import(self, import_path: str) -> str:
        """Determine layer from import path."""
        if any(import_path.startswith(d) for d in self.DOMAIN_DIRS):
            return 'domain'
        elif any(import_path.startswith(d) for d in self.INTERFACE_DIRS):
            return 'interface'
        elif any(import_path.startswith(d) for d in self.INFRASTRUCTURE_DIRS):
            return 'infrastructure'
        else:
            return 'external'  # Third-party or stdlib

    def _is_valid_dependency(self, from_layer: str, to_layer: str) -> bool:
        """Check if dependency is valid per ADR-003."""
        if to_layer == 'external':
            return True  # External dependencies are always allowed

        # Domain can only depend on domain
        if from_layer == 'domain':
            return to_layer == 'domain'

        # Interface can depend on domain
        if from_layer == 'interface':
            return to_layer in ('domain', 'interface')

        # Infrastructure can depend on anything
        if from_layer == 'infrastructure':
            return True

        return False

    def validate_project(self) -> bool:
        """Validate entire project for dependency violations."""
        print("🔍 Validating architectural dependencies per ADR-003...")

        python_files = list(self.project_root.rglob("*.py"))
        total_violations = 0

        for file_path in python_files:
            violations = self.check_file(file_path)
            total_violations += len(violations)

            for violation in violations:
                print(f"❌ {violation}")

        if total_violations == 0:
            print("✅ All dependencies are valid!")
            return True
        else:
            print(f"❌ Found {total_violations} dependency violations")
            return False

if __name__ == "__main__":
    checker = DependencyChecker(".")
    is_valid = checker.validate_project()
    sys.exit(0 if is_valid else 1)
```

### 2. Pre-commit Hook Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-dependencies
        name: Validate Architectural Dependencies
        entry: python scripts/validate-dependencies.py
        language: system
        pass_filenames: false
        always_run: true

      - id: validate-naming
        name: Validate Naming Conventions
        entry: python scripts/audit-naming.py --strict
        language: system
        pass_filenames: false
        always_run: true
```

### 3. IDE Configuration

#### VS Code Settings (`.vscode/settings.json`)

```json
{
  "python.analysis.extraPaths": [
    "./agents",
    "./inference",
    "./coalitions",
    "./world"
  ],
  "python.analysis.diagnosticSeverityOverrides": {
    "reportMissingImports": "error"
  },
  "eslint.validate": [
    "javascript",
    "typescript",
    "typescriptreact"
  ],
  "files.associations": {
    "*.py": "python",
    "*.ts": "typescript",
    "*.tsx": "typescriptreact"
  }
}
```

## Common Dependency Patterns

### ✅ Valid Dependencies

```python
# agents/explorer/explorer.py (Domain)
from agents.base.agent import Agent              # Domain → Domain ✅
from inference.engine.active_inference import AI # Domain → Domain ✅
import numpy as np                               # Domain → External ✅

# api/rest/agents/route.ts (Interface)
import { Agent } from '../../../agents/base/agent'     # Interface → Domain ✅
import { CoalitionService } from '../coalitions/service' # Interface → Interface ✅

# infrastructure/database/connection.py (Infrastructure)
from agents.base.agent import Agent              # Infrastructure → Domain ✅
from api.security.auth import AuthMiddleware     # Infrastructure → Interface ✅
```

### ❌ Invalid Dependencies

```python
# agents/explorer/explorer.py (Domain)
from api.rest.client import APIClient           # Domain → Interface ❌
from infrastructure.database.models import DB   # Domain → Infrastructure ❌

# api/rest/agents/route.ts (Interface)
from infrastructure.database.connection import db # Interface → Infrastructure ❌
```

## Dependency Injection Patterns

### Correct Approach: Dependency Inversion

```python
# agents/base/interfaces.py (Domain)
from abc import ABC, abstractmethod
from typing import List, Dict

class IAgentRepository(ABC):
    """Interface for agent persistence (defined in domain)."""

    @abstractmethod
    async def save_agent(self, agent: Agent) -> None:
        pass

    @abstractmethod
    async def get_agent(self, agent_id: str) -> Agent:
        pass

# agents/explorer/explorer.py (Domain)
class ExplorerAgent(Agent):
    """Domain entity that depends only on abstractions."""

    def __init__(self, repository: IAgentRepository):
        self.repository = repository  # Depends on interface, not implementation

    async def save_state(self):
        await self.repository.save_agent(self)

# infrastructure/database/agent_repository.py (Infrastructure)
class PostgresAgentRepository(IAgentRepository):
    """Concrete implementation in infrastructure layer."""

    def __init__(self, db_connection):
        self.db = db_connection

    async def save_agent(self, agent: Agent) -> None:
        # Implementation details...
        pass
```

## Enforcement in CI/CD

### GitHub Actions Workflow

```yaml
# .github/workflows/architecture-validation.yml
name: Architecture Validation

on: [push, pull_request]

jobs:
  validate-architecture:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt

      - name: Validate Dependencies
        run: |
          python scripts/validate-dependencies.py

      - name: Validate Naming Conventions
        run: |
          python scripts/audit-naming.py --strict

      - name: Generate Architecture Report
        run: |
          python scripts/generate-architecture-report.py

      - name: Upload Architecture Report
        uses: actions/upload-artifact@v3
        with:
          name: architecture-report
          path: architecture-report.html
```

## Troubleshooting Common Issues

### Issue: Domain Module Needs External Service

**Problem**: Agent needs to send notifications, but can't depend on infrastructure.

**Solution**: Use dependency injection with interfaces.

```python
# ❌ Wrong - Direct dependency
from infrastructure.email.service import EmailService

class Agent:
    def notify(self):
        EmailService().send("notification")  # Violates dependency rule

# ✅ Right - Dependency injection
from abc import ABC, abstractmethod

class INotificationService(ABC):
    @abstractmethod
    def send_notification(self, message: str) -> None:
        pass

class Agent:
    def __init__(self, notification_service: INotificationService):
        self.notifications = notification_service

    def notify(self):
        self.notifications.send_notification("notification")
```

### Issue: Circular Dependencies

**Problem**: Two domain modules depend on each other.

**Solution**: Extract shared abstractions or use events.

```python
# ❌ Wrong - Circular dependency
# agents/explorer.py
from coalitions.coalition import Coalition  # A → B

# coalitions/coalition.py
from agents.explorer import Explorer        # B → A (Circular!)

# ✅ Right - Shared abstractions
# agents/base/interfaces.py
class ICoalitionMember(ABC):
    pass

# coalitions/base/interfaces.py
class ICoalition(ABC):
    def add_member(self, member: ICoalitionMember):
        pass

# agents/explorer.py
from agents.base.interfaces import ICoalitionMember
from coalitions.base.interfaces import ICoalition

class Explorer(ICoalitionMember):
    def join_coalition(self, coalition: ICoalition):
        coalition.add_member(self)
```

## Reference Tools

- **Dependency Checker**: `python scripts/validate-dependencies.py`
- **Architecture Visualization**: `python scripts/generate-dependency-graph.py`
- **Naming Audit**: `python scripts/audit-naming.py`
- **Pre-commit Setup**: `pre-commit install`

## Related Documents

- [ADR-003: Dependency Rules](docs/architecture/decisions/003-dependency-rules.md)
- [ADR-004: Naming Conventions](docs/architecture/decisions/004-naming-conventions.md)
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
