# Contributing to FreeAgentics

Welcome to the FreeAgentics project! This guide will help you understand our development practices, naming conventions, and contribution process.

## Table of Contents

- [Getting Started](#getting-started)
- [Naming Conventions](#naming-conventions)
- [Development Workflow](#development-workflow)
- [Code Quality Standards](#code-quality-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Git
- Docker (optional)

### Setup

1. Fork and clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Install Node dependencies: `cd web && npm install`
6. Install pre-commit hooks: `pre-commit install`

## Naming Conventions

Our naming conventions are designed for clarity, consistency, and professionalism. They are enforced automatically via linting and pre-commit hooks.

### File Naming

#### Python Files
- **Format**: `kebab-case.py`
- **Examples**: `agent-manager.py`, `belief-update.py`, `active-inference.py`
- **Test files**: `test-agent-manager.py`, `test-belief-update.py`

```bash
✅ Good
agents/belief-update.py
tests/test-coalition-formation.py

❌ Bad
agents/beliefUpdate.py
agents/belief_update.py
tests/TestCoalitionFormation.py
```

#### TypeScript/JavaScript Files

**React Components** (PascalCase):
```bash
✅ Good
components/AgentDashboard.tsx
components/BeliefVisualizer.tsx

❌ Bad
components/agent-dashboard.tsx
components/belief_visualizer.tsx
```

**Utilities and Services** (camelCase):
```bash
✅ Good
utils/apiClient.ts
services/beliefProcessor.ts

❌ Bad
utils/api-client.ts
services/belief_processor.ts
```

**React Hooks** (camelCase with 'use' prefix):
```bash
✅ Good
hooks/useAgentState.ts
hooks/useBeliefUpdate.ts

❌ Bad
hooks/AgentState.ts
hooks/use-belief-update.ts
```

#### Configuration Files
- **Format**: `kebab-case.ext`
- **Examples**: `docker-compose.yml`, `jest-config.js`, `eslint-config.js`

### Code Naming

#### Python Code

**Classes** (PascalCase):
```python
✅ Good
class ExplorerAgent:
class BeliefState:
class CoalitionManager:

❌ Bad
class explorerAgent:
class belief_state:
class coalition_manager:
```

**Functions and Methods** (snake_case):
```python
✅ Good
def update_beliefs(self):
def calculate_free_energy():
def initialize_agent_pool():

❌ Bad
def updateBeliefs(self):
def calculateFreeEnergy():
def InitializeAgentPool():
```

**Variables** (snake_case):
```python
✅ Good
belief_state = {}
learning_rate = 0.01
agent_pool = []

❌ Bad
beliefState = {}
learningRate = 0.1
AgentPool = []
```

**Constants** (UPPER_SNAKE_CASE):
```python
✅ Good
MAX_BELIEF_PRECISION = 1e-6
DEFAULT_LEARNING_RATE = 0.1
COALITION_THRESHOLD = 0.8

❌ Bad
max_belief_precision = 1e-6
defaultLearningRate = 0.1
```

#### TypeScript Code

**Interfaces** (PascalCase with 'I' prefix for domain interfaces):
```typescript
✅ Good
interface IAgent {
  id: string;
  beliefState: BeliefState;
}

interface UserPreferences {  // No 'I' for simple data structures
  theme: string;
}

❌ Bad
interface agent {
  id: string;
}

interface IUserPreferences {  // Don't use 'I' for simple data
  theme: string;
}
```

**Functions** (camelCase):
```typescript
✅ Good
function calculateBeliefEntropy(beliefs: Belief[]): number
function updateAgentState(agent: IAgent): void

❌ Bad
function CalculateBeliefEntropy(beliefs: Belief[]): number
function update_agent_state(agent: IAgent): void
```

**Event Handlers** (camelCase with 'handle' prefix):
```typescript
✅ Good
const handleCreateAgent = () => {}
const handleBeliefUpdate = (belief: Belief) => {}

❌ Bad
const createAgent = () => {}
const onBeliefUpdate = (belief: Belief) => {}
```

**React Hooks** (camelCase with 'use' prefix):
```typescript
✅ Good
const useAgentStore = () => {}
const useBeliefState = (agentId: string) => {}

❌ Bad
const AgentStore = () => {}
const getBeliefState = (agentId: string) => {}
```

### Database Schema

**Tables** (snake_case, plural):
```sql
✅ Good
agents
belief_states
coalition_formations

❌ Bad
Agent
belief_state
coalitionFormation
```

**Columns** (snake_case):
```sql
✅ Good
agent_id
created_at
belief_precision

❌ Bad
agentId
createdAt
beliefPrecision
```

### API Endpoints

**REST Endpoints** (kebab-case resources):
```
✅ Good
GET /api/v1/agents
POST /api/v1/agents
GET /api/v1/agents/:id/beliefs
PUT /api/v1/coalition-formations/:id

❌ Bad
GET /api/v1/Agents
POST /api/v1/agent
GET /api/v1/agents/:id/beliefStates
```

### Git Conventions

**Branch Names**:
```bash
✅ Good
feature/agent-coalition-formation
fix/belief-update-precision
refactor/api-client-structure
docs/naming-conventions

❌ Bad
feature/AgentCoalitionFormation
fix_belief_update
MyNewFeature
```

**Commit Messages** (Conventional Commits):
```bash
✅ Good
feat(agents): add coalition formation algorithm
fix(api): resolve belief state synchronization issue
docs(readme): update installation instructions
refactor(core): extract belief update logic

❌ Bad
Added coalition stuff
Fixed bug
Updated readme
Refactoring
```

### Prohibited Terms

These gaming terms have been replaced with professional alternatives:

| Prohibited | Professional Alternative | Context |
|------------|-------------------------|---------|
| `PlayerAgent` | `ExplorerAgent` | Agent that explores environment |
| `NPCAgent` | `AutonomousAgent` | Agent with autonomous behavior |
| `EnemyAgent` | `CompetitiveAgent` | Agent in competitive scenarios |
| `GameWorld` | `Environment` | The execution environment |
| `spawn()` | `initialize()` | Creating new instances |
| `respawn()` | `reset()` | Resetting to initial state |

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Follow naming conventions
- Write tests for new functionality
- Update documentation as needed

### 3. Run Quality Checks
```bash
# Python formatting and linting
black .
flake8 .

# TypeScript/JavaScript formatting and linting
cd web
npm run lint
npm run format

# Run tests
pytest
cd web && npm test
```

### 4. Commit Changes
```bash
git add .
git commit -m "feat(module): add new functionality"
```

### 5. Push and Create PR
```bash
git push origin feature/your-feature-name
# Create pull request via GitHub UI
```

## Code Quality Standards

### Automated Enforcement

Our pre-commit hooks automatically check:
- ✅ Naming convention compliance
- ✅ Code formatting (Black for Python, Prettier for TypeScript)
- ✅ Linting (flake8 for Python, ESLint for TypeScript)
- ✅ No prohibited gaming terminology
- ✅ Import organization
- ✅ Basic security checks

### Manual Review Points

During code review, we also check:
- 📋 Architectural compliance (see ADRs in `docs/architecture/`)
- 📋 Test coverage and quality
- 📋 Documentation updates
- 📋 Performance considerations
- 📋 Security implications

## Testing

### Python Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test-agent-manager.py
```

### TypeScript Tests
```bash
cd web

# Run all tests
npm test

# Run tests in watch mode
npm test -- --watch

# Run specific test file
npm test -- AgentDashboard.test.tsx
```

### Integration Tests
```bash
# Run full integration test suite
npm run test:integration
```

## Submitting Changes

### Pull Request Requirements

Before submitting a PR, ensure:

1. ✅ **Naming Conventions**: All new code follows our naming standards
2. ✅ **Tests Pass**: All existing and new tests pass
3. ✅ **Linting**: No linting errors or warnings
4. ✅ **Documentation**: Updated relevant documentation
5. ✅ **ADR Compliance**: Changes align with architectural decisions
6. ✅ **Security**: No security vulnerabilities introduced

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows naming conventions
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No prohibited terms used
```

## Getting Help

- 📖 **Documentation**: Check `docs/` directory
- 🏗️ **Architecture**: See `docs/architecture/decisions/`
- 🐛 **Issues**: Create a GitHub issue
- 💬 **Questions**: Start a GitHub discussion

## Enforcement Tools

Our naming conventions are enforced by:

- **Pre-commit hooks**: Automatic checking before commits
- **CI/CD pipeline**: Checks on every pull request
- **Linting tools**: ESLint, flake8, and custom validators
- **Automated scripts**: `scripts/check-prohibited-terms.py`, `scripts/fix-naming.py`

For machine-readable naming rules, see: [`docs/standards/naming-conventions.json`](docs/standards/naming-conventions.json)

## 8. Automated Naming Enforcement

### Pre-commit Hooks Setup

The project uses pre-commit hooks to automatically enforce naming conventions and prevent violations from being committed:

```bash
# Install pre-commit (if not already installed)
pip install pre-commit

# Install the hooks in your repository
pre-commit install

# Run hooks on all files (optional, for testing)
pre-commit run --all-files
```

### Naming Enforcement Workflow

The pre-commit system includes several automated checks:

#### 1. **Enhanced Naming Convention Check**
- **Script**: `scripts/enhanced-check-naming.py`
- **Purpose**: Validates file names and code content against ADR-004 conventions
- **Scope**: All Python, TypeScript, JavaScript files
- **Failure Action**: Provides specific violation details with suggestions

#### 2. **Prohibited Terms Check**
- **Script**: `scripts/enhanced-prohibited-terms.py`
- **Purpose**: Prevents gaming terminology and old project names
- **Scope**: All source files
- **Failure Action**: Lists violations with recommended replacements

#### 3. **Architectural Dependencies Check**
- **Script**: `scripts/validate-dependencies.py`
- **Purpose**: Enforces ADR-003 architectural dependency rules
- **Scope**: Python files
- **Failure Action**: Shows violating imports and correct alternatives

#### 4. **File Naming Validation**
- **Script**: `scripts/check-file-naming.py`
- **Purpose**: Validates file names match language-specific conventions
- **Scope**: All source files
- **Failure Action**: Shows incorrect file names with suggested corrections

### Handling Hook Failures

When pre-commit hooks fail, follow this workflow:

#### 1. **Review Violations**
The hooks provide detailed output showing:
- Specific files with violations
- Line numbers (when applicable)
- Expected vs. actual naming patterns
- Suggested fixes

#### 2. **Automatic Fixes**
For many violations, use the automated fix script:
```bash
# Preview fixes (dry run)
python3 scripts/fix-naming.py --dry-run --priority

# Apply fixes automatically
python3 scripts/fix-naming.py --apply --priority
```

#### 3. **Manual Fixes**
For complex violations or syntax errors:
- Fix prohibited terms manually using provided suggestions
- Correct file names according to conventions
- Resolve syntax errors that prevent AST parsing

#### 4. **Batch Renaming**
For large-scale file renaming:
```bash
# Plan renames (dry run)
python3 scripts/batch-rename.py --python-only

# Execute renames with reference updates
python3 scripts/integrated-refactor-pipeline.py --apply --language python
```

### Troubleshooting Pre-commit Issues

#### **Python Path Issues**
If hooks fail with "Python not found":
```bash
# Ensure python3 is in PATH or update hook configuration
which python3
```

#### **Permission Issues**
If scripts aren't executable:
```bash
chmod +x scripts/*.py
```

#### **Hook Installation Issues**
If pre-commit fails to install:
```bash
# Clean and reinstall
pre-commit clean
pre-commit install
```

#### **Dependency Issues**
If hooks fail due to missing dependencies:
```bash
# Install project dependencies
pip install -r requirements.txt

# For development dependencies
pip install -r requirements-dev.txt
```

### Bypassing Hooks (Emergency Only)

In rare cases where you need to commit despite hook failures:
```bash
# Skip all hooks (use sparingly)
git commit --no-verify -m "Emergency commit"

# Skip specific hook
SKIP=enhanced-naming-check git commit -m "Skip naming check"
```

**Note**: Bypassed violations must be fixed in a follow-up commit.

### IDE Integration

#### **VS Code Integration**
The `.vscode/settings.json` includes:
- ESLint configuration for TypeScript naming rules
- Python formatting settings aligned with conventions
- File association patterns for naming validation

#### **Editor Configuration**
The `.editorconfig` file enforces:
- Consistent indentation
- File encoding (UTF-8)
- Line ending normalization
- Trailing whitespace removal

### Naming Convention Quick Reference

#### **File Naming**
- **Python**: `kebab-case.py` (e.g., `agent-manager.py`)
- **TypeScript Components**: `PascalCase.tsx` (e.g., `AgentList.tsx`)
- **TypeScript Utilities**: `camelCase.ts` (e.g., `apiUtils.ts`)
- **Configuration**: `kebab-case.json` (e.g., `test-config.json`)

#### **Code Naming**
- **Classes**: `PascalCase` (e.g., `AgentManager`)
- **Functions/Methods**: `snake_case` (Python) / `camelCase` (TypeScript)
- **Variables**: `snake_case` (Python) / `camelCase` (TypeScript)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_AGENTS`)

#### **Prohibited Terms**
❌ **Don't Use**: `PlayerAgent`, `NPCAgent`, `spawn`, `GameWorld`, `CogniticNet`
✅ **Use Instead**: `ExplorerAgent`, `AutonomousAgent`, `initialize`, `Environment`, `FreeAgentics`

### Advanced Workflow Tools

#### **AST-Based Refactoring**
For complex refactoring with automated reference updates:
```bash
# Complete refactoring pipeline
python3 scripts/integrated-refactor-pipeline.py --apply

# AST-only reference updates
python3 scripts/ast-refactor-enhanced.py --apply --add-mapping old_file.py new_file.py
```

#### **Complexity Analysis**
Before major refactoring:
```bash
python3 scripts/audit-naming.py > naming-violations.json
```

#### **Git Integration**
All renaming scripts preserve git history:
- Use `git mv` when possible
- Maintain file tracking across renames
- Generate atomic commits for clean history

This automated enforcement ensures consistent code quality and prevents naming violations from entering the codebase.

---

Thank you for contributing to FreeAgentics! 🚀
