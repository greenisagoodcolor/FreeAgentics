# Detailed Architectural Violations Analysis

## 1. Domain-Infrastructure Coupling Violations

### Violation 1: Agent Coordinator Database Dependency

**File**: `/agents/enhanced_agent_coordinator.py:21`

```python
from database.enhanced_connection_manager import get_enhanced_db_manager
```

**Clean Architecture Principle Violated**: Dependency Rule - Dependencies should point inward
**Layer**: Domain (agents) → Infrastructure (database) ❌

**Correct Implementation**:

```python
# agents/interfaces/persistence.py
from abc import ABC, abstractmethod

class IAgentRepository(ABC):
    @abstractmethod
    async def save_agent(self, agent: Agent) -> None:
        pass

    @abstractmethod
    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        pass

# agents/enhanced_agent_coordinator.py
class EnhancedAgentCoordinator:
    def __init__(self, repository: IAgentRepository):
        self.repository = repository  # Dependency injection
```

### Violation 2: API Layer Direct Domain Implementation Import

**Files**:

- `/api/v1/agents.py:75`
- `/api/v1/graphql_resolvers.py:12`

```python
from agents.agent_manager import AgentManager
agent_manager = AgentManager()  # Direct instantiation
```

**Violation**: Interface Segregation & Dependency Inversion
**Impact**: API layer tightly coupled to specific implementation

**Correct Implementation**:

```python
# api/v1/agents.py
from typing import Protocol

class IAgentService(Protocol):
    def create_agent(self, config: dict) -> dict:
        ...

# Inject through FastAPI dependency injection
@router.post("/agents")
async def create_agent(
    config: AgentConfig,
    agent_service: IAgentService = Depends(get_agent_service)
):
    return await agent_service.create_agent(config.dict())
```

## 2. Framework Bleeding Into Domain

### Violation 3: PyMDP Framework Coupling in Domain Entity

**File**: `/agents/base_agent.py`

```python
class ActiveInferenceAgent(ABC):
    def __init__(self, agent_id: str, name: str, config: Optional[Dict[str, Any]] = None):
        # ...
        self.pymdp_agent = None  # Framework-specific field
        # ...
        if PYMDP_AVAILABLE and self.config.get("use_pymdp", True):
            self._initialize_pymdp()  # Framework initialization in domain
```

**Violations**:

- Domain entity aware of specific framework (PyMDP)
- Configuration logic in domain entity
- Infrastructure concerns in constructor

**Correct Implementation**:

```python
# domain/agents/active_inference_agent.py
class ActiveInferenceAgent:
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.beliefs = Beliefs()
        self.preferences = Preferences()

    def update_beliefs(self, observation: Observation) -> None:
        # Pure domain logic
        pass

# infrastructure/pymdp/pymdp_adapter.py
class PyMDPAdapter(IInferenceEngine):
    def __init__(self, pymdp_agent):
        self.pymdp_agent = pymdp_agent

    def compute_beliefs(self, observation: Observation) -> Beliefs:
        # Adapt PyMDP to domain interface
        pass
```

## 3. Missing Architectural Layers

### Violation 4: No Application Service Layer

**Current Structure**:

```
api/v1/agents.py → agents/agent_manager.py → database/
```

**Missing**: Application services to orchestrate use cases

**Correct Structure**:

```
api/v1/agents.py
    ↓ (uses)
application/services/agent_service.py
    ↓ (orchestrates)
domain/agents/agent.py
    ↓ (via interface)
infrastructure/repositories/agent_repository.py
```

**Implementation**:

```python
# application/services/agent_service.py
class AgentService:
    def __init__(
        self,
        agent_repository: IAgentRepository,
        event_publisher: IEventPublisher,
        belief_engine: IBeliefEngine
    ):
        self._repository = agent_repository
        self._publisher = event_publisher
        self._belief_engine = belief_engine

    async def create_agent(self, command: CreateAgentCommand) -> AgentCreatedEvent:
        # Orchestrate domain logic
        agent = Agent.create(command.name, command.template)

        # Use infrastructure through interfaces
        await self._repository.save(agent)
        await self._publisher.publish(AgentCreatedEvent(agent.id))

        return agent
```

## 4. Data Transfer Object (DTO) Violations

### Violation 5: Database Models Used as API Responses

**File**: `/api/v1/agents.py`

```python
from database.models import Agent as AgentModel

@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str, db: Session = Depends(get_db)):
    agent = db.query(AgentModel).filter(AgentModel.id == agent_id).first()
    return agent  # Returning ORM model directly
```

**Problems**:

- Exposes internal database structure
- Changes to DB schema break API contract
- No control over serialization

**Correct Implementation**:

```python
# api/v1/models/responses.py
class AgentResponse(BaseModel):
    id: str
    name: str
    status: str
    created_at: datetime

# api/v1/agents.py
@router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    agent_service: IAgentService = Depends(get_agent_service)
):
    agent = await agent_service.get_agent(agent_id)
    return AgentResponse.from_domain(agent)
```

## 5. Cross-Cutting Concerns in Domain

### Violation 6: Logging and Monitoring in Domain Logic

**File**: `/agents/base_agent.py`

```python
import logging
logger = logging.getLogger(__name__)

class ActiveInferenceAgent(ABC):
    def __init__(self, ...):
        # ...
        logger.info(f"Created agent {self.agent_id}")  # Infrastructure concern

        # Performance monitoring in domain
        self.performance_optimizer = PerformanceOptimizer()
        self.performance_metrics: Dict[str, Any] = {}
```

**Correct Implementation**:

```python
# domain/agents/agent.py
class Agent:
    def __init__(self, agent_id: AgentId, name: AgentName):
        self.id = agent_id
        self.name = name
        # Pure domain logic only

# infrastructure/decorators/monitoring.py
def monitor_performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        # Log and monitor at infrastructure level
        logger.info(f"Executed {func.__name__} in {duration}s")
        metrics.record(func.__name__, duration)

        return result
    return wrapper

# application/services/agent_service.py
class AgentService:
    @monitor_performance
    async def create_agent(self, command: CreateAgentCommand):
        # Service method with cross-cutting concerns applied via decorator
        pass
```

## 6. Test Boundary Violations

### Violation 7: Tests Depend on Infrastructure

**File**: `/tests/unit/test_base_agent.py`

```python
with patch.dict(
    "sys.modules",
    {
        "pymdp": MagicMock(),
        "pymdp.utils": MagicMock(),
        "pymdp.agent": MagicMock(),
    },
):
    from agents.base_agent import ActiveInferenceAgent, AgentConfig
```

**Problems**:

- Unit tests require mocking infrastructure
- Tests are brittle and framework-dependent
- Cannot test domain logic in isolation

**Correct Implementation**:

```python
# tests/unit/domain/test_agent.py
from domain.agents import Agent, AgentId, AgentName

def test_agent_creation():
    # No infrastructure dependencies needed
    agent = Agent(
        agent_id=AgentId("123"),
        name=AgentName("TestAgent")
    )

    assert agent.id.value == "123"
    assert agent.name.value == "TestAgent"
    assert agent.is_active is False

# tests/integration/test_agent_service.py
@pytest.mark.integration
async def test_agent_service_creates_agent():
    # Integration test with real infrastructure
    repository = InMemoryAgentRepository()
    service = AgentService(repository)

    agent = await service.create_agent(
        CreateAgentCommand(name="TestAgent", template="basic")
    )

    assert await repository.exists(agent.id)
```

## Summary of Architecture Fitness Functions

To prevent future violations, implement these automated checks:

```python
# architecture_tests.py
import ast
import os

def test_no_database_imports_in_domain():
    """Domain layer should not import from database package."""
    domain_files = glob.glob("domain/**/*.py", recursive=True)

    for file in domain_files:
        with open(file) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("database"), \
                        f"{file} imports from database layer"

def test_api_uses_interfaces_not_implementations():
    """API layer should depend on abstractions."""
    api_files = glob.glob("api/**/*.py", recursive=True)

    for file in api_files:
        with open(file) as f:
            content = f.read()

        # Should import from interfaces
        assert "from domain.interfaces" in content or \
               "from application.interfaces" in content

        # Should not import concrete implementations
        assert "from agents.agent_manager import AgentManager" not in content
```

These architecture fitness functions should run in CI/CD to prevent regression.
