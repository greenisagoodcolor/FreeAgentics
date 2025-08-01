# FreeAgentics Developer Guide

## 🚀 Complete LLM→GMN→PyMDP→Knowledge Graph Cycle

FreeAgentics implements a complete cognitive architecture where:

1. **Natural language goals** are converted to **GMN specifications** via LLMs
2. **GMN specs** create **PyMDP Active Inference agents**
3. **PyMDP agents** take actions and update their **beliefs**
4. **Agent actions** update the **knowledge graph**
5. **Knowledge graph** provides context for the **next LLM generation**

## 🎯 Quick Start: Full Experience

### 1. Basic Agent Conversation (Works Now!)

```bash
# Start the system
make dev

# Create a multi-agent conversation
curl -X POST "http://localhost:8000/api/v1/agent-conversations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Discuss strategies for sustainable energy",
    "agent_count": 3,
    "conversation_turns": 5
  }'
```

### 2. GMN Generation from Natural Language

#### Option A: Demo Endpoint (No API Key Required)

```bash
# Create agent with simplified GMN
curl -X POST "http://localhost:8000/api/v1/prompts/demo" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create an agent that explores unknown environments and learns optimal paths",
    "agent_name": "PathFinder"
  }'
```

#### Option B: Full LLM Generation (Requires API Key)

```bash
# First, add your OpenAI/Anthropic API key in settings
# Then use the full prompt endpoint
curl -X POST "http://localhost:8000/api/v1/prompts" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create an agent that balances exploration and exploitation in resource gathering",
    "agent_name": "ResourceOptimizer",
    "llm_provider": "openai"
  }'
```

### 3. Knowledge Graph Integration

The knowledge graph automatically updates as agents interact:

```bash
# View current knowledge graph
curl "http://localhost:8000/api/knowledge-graph"

# Returns nodes for agents, beliefs, goals, observations, and actions
```

## 🔄 The Complete Cycle in Action

### Step 1: User Enters Goal

In the UI, enter a goal prompt like:

- "Create agents to design a sustainable city"
- "Build a team to optimize supply chain logistics"
- "Develop strategies for personalized education"

### Step 2: LLM Generates GMN

The system converts your goal into a GMN graph with:

- **State nodes**: Different states the agent can be in
- **Observation nodes**: What the agent can perceive
- **Action nodes**: What the agent can do
- **Belief nodes**: Agent's probabilistic beliefs
- **Preference nodes**: What the agent values

### Step 3: PyMDP Active Inference

Agents use Active Inference to:

- Maintain beliefs about the world
- Minimize surprise (free energy)
- Take actions that fulfill preferences
- Learn from observations

### Step 4: Knowledge Graph Updates

Every agent action updates the knowledge graph with:

- New observations
- Updated beliefs
- Action outcomes
- Free energy values
- Agent relationships

### Step 5: Feedback Loop

The knowledge graph provides context for future conversations:

- Agents remember past interactions
- Learning accumulates over time
- Collective intelligence emerges

## 🛠️ Technical Details

### GMN Graph Structure

```json
{
  "nodes": [
    { "id": "s1", "type": "state", "properties": { "name": "exploring" } },
    { "id": "o1", "type": "observation", "properties": { "name": "new_area" } },
    { "id": "a1", "type": "action", "properties": { "name": "move_forward" } },
    {
      "id": "b1",
      "type": "belief",
      "properties": { "distribution": [0.3, 0.7] }
    },
    { "id": "p1", "type": "preference", "properties": { "values": [0.1, 0.9] } }
  ],
  "edges": [
    { "source": "s1", "target": "o1", "type": "influences" },
    { "source": "a1", "target": "s1", "type": "influences" }
  ]
}
```

### PyMDP Integration

- Agents have `use_pymdp: True` in conversation service
- Beliefs update via Bayesian inference
- Actions selected to minimize expected free energy
- Knowledge graph stores belief trajectories

### Real-time Updates

- WebSocket broadcasts agent actions
- Frontend shows knowledge graph evolution
- Conversations update in real-time

## 🔧 Configuration

### API Keys (Optional)

Add in Settings modal or `.env`:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Advanced Settings

- **GMN Validation**: Strict mode requires valid graph structure
- **PyMDP Planning**: Configure planning horizon (default: 3)
- **Knowledge Graph**: Persistence across sessions (coming soon)

## 🎮 Try It Now!

1. **Start a conversation** with multiple agents
2. **Watch the knowledge graph** update in real-time
3. **Create new agents** from natural language goals
4. **See beliefs evolve** as agents interact
5. **Experience emergent intelligence** from the collective

The system is ready for experimentation. While some features like persistent knowledge graphs and complex GMN validation are still evolving, the core cognitive loop is fully functional!

## 🚨 Known Limitations

- GMN generation from LLM requires specific graph format
- Knowledge graph is currently in-memory (resets on restart)
- Complex GMN validation may be strict

But these don't prevent you from experiencing the full Active Inference multi-agent system today!
