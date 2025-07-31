# FreeAgentics Full System Product Requirements Document

## Executive Summary

FreeAgentics is an advanced multi-agent system that enables autonomous AI agents to collaborate, reason, and solve complex problems using Active Inference principles. The system combines LLM-powered conversations, Generative Model Networks (GMN), PyMDP integration, and dynamic knowledge graphs to create a continuously learning and adapting AI ecosystem.

## System Overview

### Core Capabilities

1. **Multi-Agent Conversations**: Agents with different roles collaborate to solve problems
2. **GMN Generation**: Natural language prompts are converted to formal GMN specifications
3. **Active Inference via PyMDP**: Agents use Bayesian inference for decision-making
4. **Knowledge Graph Evolution**: Dynamic graph updates based on agent interactions
5. **Continuous Planning**: Agents iteratively plan, execute, and refine strategies

### Architecture Components

- **Frontend**: React-based UI with real-time WebSocket updates
- **Backend**: FastAPI services with async processing
- **Agent Engine**: PyMDP-based active inference implementation
- **LLM Integration**: Multi-provider support (OpenAI, Anthropic, etc.)
- **Knowledge Store**: Graph database for agent knowledge

## Detailed Requirements

### 1. Agent Conversation System

#### 1.1 Agent Creation from Prompts

- **Input**: User provides natural language prompt describing a goal or problem
- **Processing**:
  - System analyzes prompt to determine optimal agent roles
  - Creates 2-5 specialized agents (Advocate, Analyst, Critic, Creative, Moderator)
  - Each agent receives unique personality and system prompt
- **Output**: Multiple agents ready for collaborative conversation

#### 1.2 Multi-Turn Conversations

- **Conversation Flow**:
  - Agents take turns responding to the prompt and each other
  - Each response considers previous context
  - Conversation continues for configured number of turns (5-20)
- **Real-time Updates**:
  - WebSocket broadcasts each agent message
  - Frontend displays messages in conversation window
  - Users see agents "thinking" and responding

#### 1.3 Conversation Management

- **Session Tracking**: Each conversation has unique ID
- **Message History**: Full conversation history stored and retrievable
- **Conversation Controls**: Start, pause, stop, and replay conversations

### 2. GMN (Generative Model Network) Integration

#### 2.1 GMN Generation from Natural Language

- **LLM-Powered Translation**:
  ```
  User Prompt → LLM Analysis → GMN Specification
  ```
- **GMN Structure**:
  ```json
  {
    "name": "agent_name",
    "states": ["exploring", "planning", "executing"],
    "observations": ["success", "obstacle", "opportunity"],
    "actions": ["move", "analyze", "collaborate"],
    "parameters": {
      "A": [[observation_model]],
      "B": [[[transition_model]]],
      "C": [[preferences]],
      "D": [[initial_beliefs]]
    }
  }
  ```

#### 2.2 GMN Validation

- **Syntax Validation**: Ensure valid JSON structure
- **Semantic Validation**: Verify probability distributions sum to 1.0
- **Fallback Templates**: Pre-defined GMNs for common scenarios

#### 2.3 GMN Evolution

- **Learning from Experience**: GMN parameters update based on outcomes
- **Agent Specialization**: GMNs become more refined over time
- **Cross-Agent Learning**: Successful strategies propagate between agents

### 3. PyMDP Active Inference Engine

#### 3.1 Belief State Management

- **Initial Beliefs**: Set from GMN D parameter
- **Belief Updates**: Bayesian inference on new observations
- **Multi-Factor Beliefs**: Support for complex state spaces

#### 3.2 Policy Selection

- **Expected Free Energy**: Minimize uncertainty and maximize preferences
- **Planning Horizon**: Look-ahead for 3-10 steps
- **Action Selection**: Probabilistic selection based on policy values

#### 3.3 Active Inference Loop

```
1. Observe current state
2. Update beliefs using Bayes rule
3. Calculate expected free energy for policies
4. Select and execute action
5. Repeat
```

### 4. Knowledge Graph System

#### 4.1 Graph Structure

- **Nodes**:
  - Concepts discovered by agents
  - Agent states and beliefs
  - Environmental entities
  - Goals and sub-goals
- **Edges**:
  - Relationships between concepts
  - Causal connections
  - Temporal sequences
  - Influence weights

#### 4.2 LLM-Powered Updates

- **Entity Extraction**: LLM analyzes agent conversations for new entities
- **Relationship Discovery**: Identify connections between concepts
- **Graph Enrichment**: Add properties and metadata to nodes/edges

#### 4.3 Graph Queries

- **Semantic Search**: Find related concepts
- **Path Finding**: Discover connections between disparate ideas
- **Subgraph Extraction**: Focus on relevant knowledge domains

### 5. Continuous Planning & Execution

#### 5.1 Planning Cycle

```
1. Agents analyze current knowledge graph
2. Identify gaps or opportunities
3. Formulate new hypotheses/strategies
4. Execute plans through conversation/action
5. Observe outcomes and update knowledge
6. Repeat cycle
```

#### 5.2 Multi-Agent Coordination

- **Coalition Formation**: Agents with complementary skills team up
- **Task Delegation**: Distribute work based on agent capabilities
- **Consensus Building**: Resolve conflicts through structured debate

#### 5.3 Adaptive Strategies

- **Success Metrics**: Track which approaches work
- **Strategy Evolution**: Refine plans based on feedback
- **Emergent Behaviors**: Allow for unexpected solutions

## Technical Implementation

### API Endpoints

#### Agent Conversations

```
POST /api/v1/agent-conversations
{
  "prompt": "string",
  "agent_count": 2-5,
  "conversation_turns": 5-20,
  "llm_provider": "openai|anthropic",
  "model": "gpt-3.5-turbo|claude-3"
}

Response:
{
  "conversation_id": "uuid",
  "agents": [...],
  "messages": [...],
  "status": "completed"
}
```

#### GMN Generation

```
POST /api/v1/gmn/generate
{
  "prompt": "string",
  "agent_type": "explorer|analyst|creative"
}

Response:
{
  "gmn_spec": {...},
  "validation_status": "valid"
}
```

#### Knowledge Graph Operations

```
GET /api/v1/knowledge-graph
POST /api/v1/knowledge-graph/update
GET /api/v1/knowledge-graph/query
```

### WebSocket Events

#### Message Types

- `conversation_message`: New agent message
- `agent_created`: Agent spawned
- `knowledge_update`: Graph modified
- `belief_update`: Agent belief changed
- `plan_generated`: New plan created

### Database Schema

#### Agents Table

```sql
CREATE TABLE agents (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    type VARCHAR(50),
    gmn_spec JSONB,
    belief_state JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

#### Conversations Table

```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    prompt TEXT,
    agent_ids UUID[],
    message_count INT,
    status VARCHAR(50),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

#### Knowledge Graph Tables

```sql
CREATE TABLE kg_nodes (
    id UUID PRIMARY KEY,
    label VARCHAR(255),
    type VARCHAR(50),
    properties JSONB,
    created_by UUID REFERENCES agents(id),
    created_at TIMESTAMP
);

CREATE TABLE kg_edges (
    id UUID PRIMARY KEY,
    source_id UUID REFERENCES kg_nodes(id),
    target_id UUID REFERENCES kg_nodes(id),
    relationship VARCHAR(255),
    weight FLOAT,
    properties JSONB
);
```

## User Experience Flow

### Initial Interaction

1. User enters goal: "Create a Jakarta accelerator for green chemistry"
2. System creates specialized agents (Business Strategist, Chemist, Market Analyst)
3. Agents begin conversation, visible in real-time
4. Knowledge graph populates with discovered concepts

### Ongoing Interaction

1. User can ask follow-up questions
2. Agents incorporate new information
3. Plans are refined based on feedback
4. Knowledge graph grows and connections strengthen

### Output & Results

1. Actionable recommendations from agent consensus
2. Visual knowledge graph showing discovered relationships
3. Downloadable reports and plans
4. API access for integration with other systems

## Performance Requirements

### Response Times

- Agent message generation: < 2 seconds
- Knowledge graph updates: < 500ms
- WebSocket message delivery: < 100ms

### Scalability

- Support 100 concurrent conversations
- Knowledge graphs up to 10,000 nodes
- Message history retention: 30 days

### Reliability

- 99.9% uptime for core services
- Graceful degradation when LLM providers fail
- Automatic recovery from transient errors

## Security & Privacy

### Data Protection

- Encrypted storage for sensitive conversations
- API key rotation every 90 days
- User-level access controls

### Audit Trail

- Log all agent actions and decisions
- Track knowledge graph modifications
- Maintain conversation history for compliance

## Future Enhancements

### Phase 2

- Visual agent simulation environments
- Multi-modal inputs (images, documents)
- External tool integration (calculators, databases)

### Phase 3

- Distributed agent networks across organizations
- Federated learning from multiple deployments
- Agent marketplace for specialized capabilities

## Success Metrics

### Technical KPIs

- Average conversation completion rate: > 95%
- GMN generation success rate: > 90%
- Knowledge graph query response time: < 200ms

### Business KPIs

- User engagement: > 10 conversations per user per month
- Knowledge graph growth: > 100 new nodes per day
- Agent recommendation adoption rate: > 60%

## Implementation Timeline

### Month 1: Foundation

- Complete agent conversation system
- Basic GMN generation
- Simple knowledge graph

### Month 2: Intelligence

- PyMDP integration
- Advanced planning algorithms
- LLM-powered knowledge extraction

### Month 3: Polish

- Performance optimization
- UI/UX improvements
- Production deployment

## Conclusion

FreeAgentics represents a paradigm shift in AI collaboration, moving from single-model interactions to rich multi-agent ecosystems. By combining conversational AI, active inference, and dynamic knowledge graphs, the system enables emergent intelligence that continuously learns and adapts to solve complex real-world problems.
