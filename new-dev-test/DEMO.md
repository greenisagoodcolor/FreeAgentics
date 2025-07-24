# FreeAgentics v1.0.0-alpha+ Demo Guide

> **Living Multi-Agent AI Platform with Active Inference**
>
> This guide demonstrates the complete end-to-end flow:
> **Goal Prompt → LLM → GMN → PyMDP → Agent → Knowledge Graph → Visualization**

## 🚀 Quick Start

### Prerequisites

1. **API Keys** (set in environment):

   ```bash
   export OPENAI_API_KEY="your-key-here"      # Optional
   export ANTHROPIC_API_KEY="your-key-here"   # Optional
   ```

2. **Start the System**:

   ```bash
   # Development mode
   docker-compose up -d

   # Or production mode
   docker-compose -f docker-compose.production.yml up -d
   ```

3. **Verify Services**:

   ```bash
   # Check health
   curl http://localhost:8000/health

   # Check metrics
   curl http://localhost:8000/api/v1/metrics
   ```

## 🎯 Core Functionality Demo

### 1. Create Agent from Natural Language Prompt

```bash
# Create an explorer agent using the /prompts endpoint
curl -X POST http://localhost:8000/api/v1/prompts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "prompt": "Create an agent that explores a grid world to find hidden rewards while avoiding obstacles",
    "agent_name": "explorer_bot"
  }'
```

**Expected Response**:

```json
{
  "agent_id": "agent_abc123",
  "agent_name": "explorer_bot",
  "gmn_spec": {
    "name": "explorer_bot",
    "states": ["exploring", "found_target", "avoiding_obstacle"],
    "observations": ["empty", "target", "obstacle", "boundary"],
    "actions": ["move_up", "move_down", "move_left", "move_right", "stay"],
    "parameters": { ... }
  },
  "pymdp_model": { ... },
  "status": "active",
  "llm_provider_used": "openai",
  "generation_time_ms": 1523.4
}
```

### 2. Monitor Agent in Real-Time via WebSocket

```javascript
// Connect to WebSocket for live updates
const ws = new WebSocket("ws://localhost:8000/api/v1/ws");

ws.onopen = () => {
  // Authenticate
  ws.send(
    JSON.stringify({
      type: "auth",
      data: { token: AUTH_TOKEN },
    }),
  );

  // Subscribe to agent events
  ws.send(
    JSON.stringify({
      type: "subscribe",
      data: { agent_id: "agent_abc123" },
    }),
  );
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(`Agent ${msg.agent_id}: ${msg.type}`, msg.data);
  // Types: agent_created, agent_step, belief_update, etc.
};
```

### 3. Step Agent Through Environment

```bash
# Send observation to agent
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "agent_id": "agent_abc123",
    "observation": {
      "type": "grid_cell",
      "content": "empty",
      "position": [5, 3]
    }
  }'
```

**Response includes**:

- Selected action (e.g., "move_right")
- Updated beliefs
- Free energy value
- Execution time

### 4. Query Knowledge Graph

```bash
# Get agent's knowledge graph
curl http://localhost:8000/api/v1/knowledge/graphs/agent_abc123 \
  -H "Authorization: Bearer $AUTH_TOKEN"

# Query specific patterns
curl -X POST http://localhost:8000/api/v1/knowledge/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "graph_id": "agent_abc123",
    "query_type": "path",
    "start_node": "observation_1",
    "end_node": "action_5"
  }'
```

### 5. Export Knowledge Graph

```bash
# Export as JSON
curl http://localhost:8000/api/v1/knowledge/graphs/agent_abc123/export?format=json \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -o agent_knowledge.json

# Export as GraphML
curl http://localhost:8000/api/v1/knowledge/graphs/agent_abc123/export?format=graphml \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -o agent_knowledge.graphml
```

## 🎨 Frontend Visualization

Open http://localhost:3000 in your browser to see:

1. **Agent Panel** (left): List of active agents with status
2. **Conversation** (center): Real-time agent thoughts and actions
3. **Knowledge Graph** (right): Interactive D3 force-directed graph
4. **Grid World** (bottom): H3 hexagonal grid showing agent positions

## 📊 Example Scenarios

### Scenario 1: Multi-Agent Exploration

```python
# Create multiple agents with different strategies
agents = []
for strategy in ["explorer", "cautious", "goal_directed"]:
    response = create_agent(
        prompt=f"Create a {strategy} agent for grid navigation",
        name=f"{strategy}_bot"
    )
    agents.append(response['agent_id'])

# Run them simultaneously
for _ in range(100):
    observations = get_world_state()
    for agent_id in agents:
        step_agent(agent_id, observations[agent_id])
    time.sleep(0.1)  # 10 steps per second
```

### Scenario 2: Knowledge Transfer

```python
# Export knowledge from experienced agent
expert_kg = export_knowledge_graph("expert_agent_id")

# Import into new agent
new_agent = create_agent("Create a learning agent")
import_knowledge_graph(new_agent['agent_id'], expert_kg)
```

## 🔍 Monitoring & Metrics

### Prometheus Metrics

- `freeagentics_agent_inference_duration_seconds`: Time per inference step
- `freeagentics_agent_free_energy`: Current free energy per agent
- `freeagentics_kg_nodes_total`: Total knowledge graph nodes
- `freeagentics_llm_requests_total`: LLM API calls
- `freeagentics_websocket_connections`: Active WebSocket connections

### Grafana Dashboard

Access at http://localhost:3001 (default login: admin/admin)

## 🧪 Advanced Features

### Custom GMN Specifications

```json
{
  "prompt": "Create agent with custom GMN",
  "gmn_override": {
    "states": ["state1", "state2", "state3"],
    "observations": ["obs1", "obs2"],
    "actions": ["action1", "action2"],
    "parameters": {
      "A": [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]],
      "B": [...],
      "C": [[0.1, 0.9]],
      "D": [[0.6, 0.3, 0.1]]
    }
  }
}
```

### Batch Operations

```bash
# Create multiple agents at once
curl -X POST http://localhost:8000/api/v1/agents/batch \
  -H "Content-Type: application/json" \
  -d '{
    "agents": [
      {"prompt": "Explorer agent", "name": "explorer1"},
      {"prompt": "Guardian agent", "name": "guardian1"},
      {"prompt": "Forager agent", "name": "forager1"}
    ]
  }'
```

## 🛠️ Troubleshooting

### Common Issues

1. **"No LLM providers available"**

   - Ensure API keys are set in environment
   - Check `docker-compose logs api`

2. **WebSocket connection fails**

   - Verify authentication token
   - Check CORS settings if connecting from different origin

3. **Agent not moving**
   - Check free energy values (might be in local minimum)
   - Verify observation format matches GMN specification

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
docker-compose restart api
```

## 📚 Further Reading

- [Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md)
- [API Reference](docs/api/API_REFERENCE.md)
- [Active Inference Theory](docs/ACTIVE_INFERENCE_GUIDE.md)
- [GMN Specification](docs/GMN_SPECIFICATION.md)

---

**🎉 Congratulations!** You've successfully demonstrated the FreeAgentics platform with:

- ✅ Natural language → Agent creation
- ✅ Active Inference decision making
- ✅ Real-time monitoring via WebSocket
- ✅ Knowledge Graph persistence
- ✅ Multi-agent coordination

For production deployment, see [DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md).
