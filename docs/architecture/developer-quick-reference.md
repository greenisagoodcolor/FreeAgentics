# Developer Quick Reference: Dependency & Naming Rules

This is a concise reference for FreeAgentics architectural rules. For detailed explanations, see the full ADR documents.

## 🏗️ Dependency Rules (ADR-003)

### Core Rule
**Dependencies MUST flow inward**: `Infrastructure → Interface → Domain`

### Quick Check
```python
# ✅ ALLOWED - Domain importing Domain
from agents.base.agent import Agent
from inference.engine.active_inference import AI

# ✅ ALLOWED - Interface importing Domain
from agents.explorer.explorer import Explorer

# ❌ FORBIDDEN - Domain importing Interface
from api.rest.client import APIClient

# ❌ FORBIDDEN - Domain importing Infrastructure
from infrastructure.database.models import AgentModel
```

### Layer Definitions
- **Domain**: `agents/`, `inference/`, `coalitions/`, `world/`
- **Interface**: `api/`, `web/`
- **Infrastructure**: `infrastructure/`, `config/`, `data/`

### Common Solutions
**Problem**: Domain needs database access
**Solution**: Use dependency injection
```python
# Define interface in domain
class IAgentRepository(ABC):
    @abstractmethod
    async def save_agent(self, agent: Agent) -> None: pass

# Implement in infrastructure
class PostgresAgentRepository(IAgentRepository):
    async def save_agent(self, agent: Agent) -> None:
        # Implementation...
```

## 📛 Naming Conventions (ADR-004)

### File Naming

| Type | Convention | Example |
|------|------------|---------|
| Python files | kebab-case | `belief-update.py` |
| TypeScript components | PascalCase | `AgentDashboard.tsx` |
| TypeScript utilities | camelCase | `apiClient.ts` |
| Configuration | kebab-case | `docker-compose.yml` |

### Code Naming

#### Python
```python
class ExplorerAgent(BaseAgent):     # PascalCase classes
    def update_beliefs(self):        # snake_case methods
        MAX_ENERGY = 100            # UPPER_SNAKE_CASE constants
        agent_id = "12345"          # snake_case variables
```

#### TypeScript
```typescript
interface IAgent {                  // 'I' prefix for domain interfaces
    beliefState: BeliefState;       // camelCase properties
}

const AgentCreator: React.FC = () => {  // PascalCase components
    const handleSubmit = () => {};      // 'handle' prefix for events
    const MAX_AGENTS = 1000;           // UPPER_SNAKE_CASE constants
};
```

### Prohibited Terms
❌ **Gaming terminology** → ✅ **Professional terms**
- PlayerAgent → ExplorerAgent
- NPCAgent → AutonomousAgent
- spawn() → initialize()
- GameWorld → Environment

## 🛠️ Validation Tools

### Quick Commands
```bash
# Check dependencies
python scripts/validate-dependencies.py

# Check naming conventions
python scripts/audit-naming.py --strict

# Generate reports
python scripts/validate-dependencies.py --html-report dependency-report.html
```

### Pre-commit Setup
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🚨 Common Violations & Fixes

### 1. Domain Importing Infrastructure
**❌ Wrong:**
```python
# agents/explorer/explorer.py
from infrastructure.database.connection import db
```

**✅ Right:**
```python
# agents/base/interfaces.py
class IAgentRepository(ABC):
    @abstractmethod
    async def save_agent(self, agent: Agent) -> None: pass

# agents/explorer/explorer.py
class Explorer(Agent):
    def __init__(self, repository: IAgentRepository):
        self.repository = repository
```

### 2. Circular Dependencies
**❌ Wrong:**
```python
# agents/explorer.py
from coalitions.coalition import Coalition

# coalitions/coalition.py
from agents.explorer import Explorer
```

**✅ Right:**
```python
# agents/base/interfaces.py
class ICoalitionMember(ABC): pass

# coalitions/base/interfaces.py
class ICoalition(ABC): pass

# Use interfaces instead of concrete classes
```

### 3. Incorrect File Naming
**❌ Wrong:**
- `beliefUpdate.py` (camelCase Python file)
- `agent-dashboard.tsx` (kebab-case React component)
- `AgentAPI.py` (PascalCase Python file)

**✅ Right:**
- `belief-update.py`
- `AgentDashboard.tsx`
- `agent-api.py`

## 📋 Checklist for New Code

Before committing, verify:

- [ ] All imports follow dependency flow rules
- [ ] File names follow conventions for language/type
- [ ] Class names use PascalCase
- [ ] Method names use snake_case (Python) or camelCase (TypeScript)
- [ ] No gaming terminology used
- [ ] Interfaces defined in domain layer for external dependencies
- [ ] No circular import dependencies

## 🔍 Quick Debug

**Dependency violation?**
1. Check which layer your file is in
2. Check which layer you're importing from
3. Ensure dependencies flow inward only

**Naming violation?**
1. Check file extension and use appropriate convention
2. Verify class/method naming matches language standards
3. Replace any gaming terminology

**Need help?**
- Full rules: [ADR-003](docs/architecture/decisions/003-dependency-rules.md), [ADR-004](docs/architecture/decisions/004-naming-conventions.md)
- Validation guide: [dependency-validation-guide.md](docs/architecture/dependency-validation-guide.md)
- Run: `python scripts/validate-dependencies.py --help`
