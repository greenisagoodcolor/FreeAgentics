# FreeAgentics Interface Contracts

## Overview
This document defines the exact interface contracts between FreeAgentics components to enable proper integration.

## 1. LLM Provider Interfaces

### ILLMProvider (Existing)
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class GenerationRequest:
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False

@dataclass
class GenerationResponse:
    text: str
    model: str
    provider: ProviderType
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0

class ILLMProvider(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from prompt"""
        pass
```

### IGMNGenerator (Needs Implementation)
```python
class IGMNGenerator(ABC):
    @abstractmethod
    def prompt_to_gmn(
        self, 
        user_prompt: str,
        agent_type: str = "explorer",
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """Convert natural language prompt to GMN specification"""
        pass
    
    @abstractmethod
    def validate_gmn(self, gmn_spec: str) -> Tuple[bool, List[str]]:
        """Validate GMN specification, return (is_valid, errors)"""
        pass
    
    @abstractmethod
    def refine_gmn(
        self, 
        gmn_spec: str, 
        feedback: str
    ) -> str:
        """Refine GMN based on validation feedback"""
        pass
```

## 2. GMN Parser Interfaces

### GMNParser (Existing)
```python
@dataclass
class GMNNode:
    id: str
    type: GMNNodeType
    properties: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class GMNGraph:
    nodes: Dict[str, GMNNode]
    edges: List[GMNEdge]
    metadata: Dict[str, Any]

class GMNParser:
    def parse(self, gmn_spec: Union[str, Dict[str, Any]]) -> GMNGraph:
        """Parse GMN specification into graph"""
        pass
    
    def to_pymdp_model(self, graph: GMNGraph) -> Dict[str, Any]:
        """Convert GMN graph to PyMDP model specification"""
        # Returns: {
        #     "num_states": List[int],
        #     "num_obs": List[int],
        #     "num_actions": List[int],
        #     "A": List[np.ndarray],  # Observation models
        #     "B": List[np.ndarray],  # Transition models
        #     "C": List[np.ndarray],  # Preferences
        #     "D": List[np.ndarray],  # Initial beliefs
        #     "llm_integration": List[Dict]
        # }
        pass
```

## 3. Agent Factory Interface (Needs Implementation)

### IAgentFactory
```python
from pymdp import Agent
from typing import Union

class IAgentFactory(ABC):
    @abstractmethod
    def create_from_gmn_model(
        self,
        model: Dict[str, Any],
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Agent:
        """Create PyMDP agent from GMN model output"""
        pass
    
    @abstractmethod
    def validate_model(self, model: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate model before agent creation"""
        pass
    
    @abstractmethod
    def create_from_arrays(
        self,
        A: Union[np.ndarray, List[np.ndarray]],
        B: Union[np.ndarray, List[np.ndarray]],
        C: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        D: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        **kwargs
    ) -> Agent:
        """Create agent from raw arrays"""
        pass
```

## 4. PyMDP Adapter Interface (Existing)

### PyMDPCompatibilityAdapter
```python
class PyMDPCompatibilityAdapter:
    def sample_action(self, pymdp_agent: Agent) -> int:
        """Convert numpy array action to int"""
        pass
    
    def infer_policies(
        self, 
        pymdp_agent: Agent
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Get (q_pi, G) policy arrays"""
        pass
    
    def infer_states(
        self,
        pymdp_agent: Agent,
        observation: Union[int, List[int], NDArray[Any]]
    ) -> List[NDArray[np.floating]]:
        """Update beliefs given observation"""
        pass
```

## 5. Belief Extraction Interface (Needs Implementation)

### IBeliefExtractor
```python
@dataclass
class BeliefState:
    factor_beliefs: List[np.ndarray]  # Beliefs per state factor
    timestamp: datetime
    entropy: float
    most_likely_states: List[int]
    metadata: Dict[str, Any]

class IBeliefExtractor(ABC):
    @abstractmethod
    def extract_beliefs(self, agent: Agent) -> BeliefState:
        """Extract current belief state from agent"""
        pass
    
    @abstractmethod
    def get_belief_trajectory(
        self, 
        agent: Agent,
        history_length: int = 10
    ) -> List[BeliefState]:
        """Get historical belief trajectory"""
        pass
```

## 6. Knowledge Graph Interfaces

### KnowledgeGraph (Existing)
```python
class KnowledgeGraph:
    def add_node(self, node: KnowledgeNode) -> bool:
        """Add node to graph"""
        pass
    
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update existing node"""
        pass
    
    def get_neighbors(
        self, 
        node_id: str, 
        edge_type: Optional[EdgeType] = None
    ) -> List[str]:
        """Get neighboring nodes"""
        pass
```

### IBeliefKGBridge (Needs Implementation)
```python
class IBeliefKGBridge(ABC):
    @abstractmethod
    def belief_to_nodes(
        self,
        belief_state: BeliefState,
        agent_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[KnowledgeNode]:
        """Convert belief state to knowledge nodes"""
        pass
    
    @abstractmethod
    def observation_to_nodes(
        self,
        observation: Union[int, List[int], Dict[str, Any]],
        agent_id: str,
        timestamp: datetime
    ) -> List[KnowledgeNode]:
        """Convert observation to knowledge nodes"""
        pass
    
    @abstractmethod
    def update_kg_from_agent(
        self,
        agent: Agent,
        agent_id: str,
        kg: KnowledgeGraph
    ) -> Dict[str, Any]:
        """Update knowledge graph from agent state"""
        # Returns: {
        #     "nodes_added": int,
        #     "nodes_updated": int,
        #     "edges_added": int
        # }
        pass
```

## 7. Preference Update Interface (Needs Implementation)

### IPreferenceUpdater
```python
class IPreferenceUpdater(ABC):
    @abstractmethod
    def kg_insights_to_preferences(
        self,
        kg: KnowledgeGraph,
        agent_id: str,
        current_C: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate new preferences from KG insights"""
        pass
    
    @abstractmethod
    def apply_preference_update(
        self,
        agent: Agent,
        new_C: Union[np.ndarray, List[np.ndarray]]
    ) -> bool:
        """Apply new preferences to agent"""
        pass
```

## 8. WebSocket Message Contracts

### Agent State Update
```typescript
interface AgentStateUpdate {
    type: "agent_state_update";
    agent_id: string;
    timestamp: string;
    data: {
        position?: number[];
        beliefs?: number[][];
        action?: number;
        observation?: number;
        metadata?: Record<string, any>;
    };
}
```

### Knowledge Graph Update
```typescript
interface KnowledgeGraphUpdate {
    type: "kg_update";
    timestamp: string;
    data: {
        nodes_added: NodeData[];
        nodes_updated: NodeData[];
        edges_added: EdgeData[];
        graph_stats: {
            total_nodes: number;
            total_edges: number;
        };
    };
}
```

## 9. API Endpoint Contracts

### Create Agent from GMN
```typescript
// POST /api/v1/agents/from-gmn
interface CreateAgentRequest {
    prompt: string;
    agent_type?: "explorer" | "trader" | "coordinator";
    gmn_spec?: string;  // Optional: provide GMN directly
    metadata?: Record<string, any>;
}

interface CreateAgentResponse {
    agent_id: string;
    status: "created" | "failed";
    gmn_spec: string;
    model_dimensions: {
        num_states: number[];
        num_obs: number[];
        num_actions: number[];
    };
    errors?: string[];
}
```

### Update Agent Preferences
```typescript
// POST /api/v1/agents/{agent_id}/preferences
interface UpdatePreferencesRequest {
    source: "manual" | "kg_insights" | "learning";
    preferences: number[] | number[][];
    metadata?: Record<string, any>;
}

interface UpdatePreferencesResponse {
    success: boolean;
    previous_preferences: number[] | number[][];
    new_preferences: number[] | number[][];
    impact_metrics?: {
        policy_divergence: number;
        expected_free_energy_change: number;
    };
}
```

## Implementation Priority

1. **Critical Path** (Must implement first):
   - `IGMNGenerator` - Without this, no LLM→GMN flow
   - `IAgentFactory` - Without this, no GMN→Agent flow
   - `IBeliefExtractor` - Without this, no Agent→KG flow

2. **Integration Layer** (Implement second):
   - `IBeliefKGBridge` - Connects agents to knowledge graph
   - WebSocket contracts - Enables real-time updates

3. **Enhancement Layer** (Implement last):
   - `IPreferenceUpdater` - Closes the feedback loop
   - Advanced API endpoints - Enables full control

## Testing Requirements

Each interface requires:
1. Unit tests for all methods
2. Integration tests with adjacent components
3. Error case coverage
4. Performance benchmarks
5. Contract validation tests

## Versioning Strategy

All interfaces should follow semantic versioning:
- Major version: Breaking changes to method signatures
- Minor version: New methods added
- Patch version: Bug fixes, no API changes

Example:
```python
class IAgentFactory_v1_0_0(ABC):
    # Initial version
    
class IAgentFactory_v1_1_0(IAgentFactory_v1_0_0):
    # Adds new methods, backward compatible
    
class IAgentFactory_v2_0_0(ABC):
    # Breaking changes to existing methods
```