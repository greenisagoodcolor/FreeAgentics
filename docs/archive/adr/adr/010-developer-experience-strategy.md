# ADR-010: Developer Experience and Tooling Strategy

## Status

Accepted

## Context

FreeAgentics aims to democratize autonomous AI agent development by providing an exceptional developer experience. The framework must be approachable for beginners while remaining powerful for experts. This requires comprehensive tooling, clear documentation, intuitive APIs, and smooth development workflows.

## Decision

We will implement a comprehensive developer experience strategy that prioritizes simplicity, discoverability, and productivity while maintaining the power and flexibility required for advanced use cases.

## Developer Experience Principles

### 1. Progressive Disclosure

- Simple start with minimal code
- Gradual complexity with advanced features discoverable as needed
- Sensible defaults that work out-of-the-box
- Expert escape hatches for full customization

### 2. Discoverability

- IDE integration with type hints and auto-completion
- Interactive examples with Jupyter notebooks
- Visual debugging with real-time agent state visualization
- Comprehensive documentation with multiple learning paths

## Developer Tooling Stack

### 1. Command Line Interface (CLI)

#### FreeAgentics CLI Tool

```bash
# Installation and setup
$ pip install freeagentics
$ freeagentics init my-agent-project
$ cd my-agent-project
$ freeagentics dev  # Start development server

# Agent management
$ freeagentics agent create --name Explorer --type curious
$ freeagentics agent list
$ freeagentics agent inspect agent_123

# Simulation management
$ freeagentics sim create --agents 100 --world-size large
$ freeagentics sim run --id sim_456 --steps 1000
$ freeagentics sim visualize --id sim_456

# Coalition operations
$ freeagentics coalition list
$ freeagentics coalition analyze --id coal_789
$ freeagentics coalition export --id coal_789 --format json

# Development utilities
$ freeagentics test --coverage
$ freeagentics lint --fix
$ freeagentics benchmark --quick
$ freeagentics deploy --target edge
```

#### CLI Implementation

```python
# cli/commands/agent.py
import click
from rich.console import Console
from rich.table import Table

@click.group()
def agent():
    """Agent management commands."""
    pass

@agent.command()
@click.option('--name', required=True, help='Agent name')
@click.option('--type', default='Explorer', help='Agent type')
@click.option('--personality', help='Personality traits (e.g., curious:0.9,social:0.7)')
def create(name: str, type: str, personality: str):
    """Create a new agent with specified characteristics."""
    console = Console()

    try:
        # Parse personality if provided
        traits = {}
        if personality:
            for trait in personality.split(','):
                key, value = trait.split(':')
                traits[key] = float(value)

        # Create agent
        agent = Agent.create(type, name=name, personality=traits)

        console.print(f"✨ Created {type} agent '{name}' with ID: {agent.id}")
        console.print(f"📊 Personality: {format_personality(traits)}")
        console.print(f"🎯 Starting location: {agent.location}")
        console.print(f"⚡ Energy: {agent.energy}/100")

    except Exception as e:
        console.print(f"❌ Error creating agent: {e}", style="bold red")
```

### 2. Development Dashboard

#### Web-based Development Environment

```typescript
// web/dashboard/development/page.tsx
'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { AgentVisualization } from '@/components/agent-visualization'
import { CoalitionNetwork } from '@/components/coalition-network'
import { PerformanceMetrics } from '@/components/performance-metrics'

export default function DevelopmentDashboard() {
  const [agents, setAgents] = useState([])
  const [coalitions, setCoalitions] = useState([])
  const [metrics, setMetrics] = useState({})

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 p-6">
      <Card className="col-span-full">
        <CardHeader>
          <CardTitle>🚀 FreeAgentics Development Environment</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <button
              onClick={() => createAgent()}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Create Agent
            </button>
            <button
              onClick={() => startSimulation()}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            >
              Start Simulation
            </button>
            <button
              onClick={() => exportCoalition()}
              className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
            >
              Export Coalition
            </button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>🤖 Active Agents</CardTitle>
        </CardHeader>
        <CardContent>
          <AgentVisualization agents={agents} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>🤝 Coalition Network</CardTitle>
        </CardHeader>
        <CardContent>
          <CoalitionNetwork coalitions={coalitions} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>📊 Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <PerformanceMetrics metrics={metrics} />
        </CardContent>
      </Card>
    </div>
  )
}
```

### 3. Interactive Development Environment

#### Jupyter Integration

```python
# notebooks/examples/getting_started.ipynb
"""
FreeAgentics Interactive Tutorial

This notebook demonstrates core concepts through hands-on examples.
"""

# Cell 1: Installation and imports
!pip install freeagentics
from freeagentics import Agent, World, Coalition
from freeagentics.visualization import AgentVisualizer, BeliefPlotter
import matplotlib.pyplot as plt

# Cell 2: Create your first agent
explorer = Agent.create("Explorer", name="Alice")
print(f"Created agent: {explorer.name}")
print(f"Initial beliefs: {explorer.beliefs.tolist()}")

# Cell 3: Interactive belief visualization
visualizer = BeliefPlotter(explorer)
visualizer.show_interactive()  # Creates interactive Plotly widget

# Cell 4: Multi-agent simulation
world = World(grid_size=20)
agents = [
    Agent.create("Explorer", name=f"Explorer_{i}")
    for i in range(5)
]
for agent in agents:
    world.add_agent(agent)

# Cell 5: Real-time simulation with visualization
@interact(steps=(1, 100, 1))
def simulate_and_visualize(steps=10):
    for _ in range(steps):
        world.step()

    fig = world.visualize()
    fig.show()

    return {
        'active_agents': len(world.agents),
        'coalitions_formed': len(world.coalitions),
        'total_energy': sum(agent.energy for agent in world.agents)
    }
```

### 4. API Development Tools

#### API Explorer and Playground

```typescript
// web/api-playground/page.tsx
'use client'

import { useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { CodeEditor } from '@/components/code-editor'
import { APIResponse } from '@/components/api-response'

export default function APIPlayground() {
  const [endpoint, setEndpoint] = useState('/api/v1/agents')
  const [method, setMethod] = useState('GET')
  const [requestBody, setRequestBody] = useState('{}')
  const [response, setResponse] = useState(null)

  const sendRequest = async () => {
    try {
      const res = await fetch(endpoint, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: method !== 'GET' ? requestBody : undefined
      })
      const data = await res.json()
      setResponse({ status: res.status, data })
    } catch (error) {
      setResponse({ error: error.message })
    }
  }

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">🔧 FreeAgentics API Playground</h1>

      <Tabs defaultValue="rest">
        <TabsList>
          <TabsTrigger value="rest">REST API</TabsTrigger>
          <TabsTrigger value="websocket">WebSocket</TabsTrigger>
          <TabsTrigger value="graphql">GraphQL</TabsTrigger>
        </TabsList>

        <TabsContent value="rest" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Endpoint</label>
                <input
                  value={endpoint}
                  onChange={(e) => setEndpoint(e.target.value)}
                  className="w-full p-2 border rounded"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Method</label>
                <select
                  value={method}
                  onChange={(e) => setMethod(e.target.value)}
                  className="w-full p-2 border rounded"
                >
                  <option value="GET">GET</option>
                  <option value="POST">POST</option>
                  <option value="PUT">PUT</option>
                  <option value="DELETE">DELETE</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Request Body</label>
                <CodeEditor
                  value={requestBody}
                  onChange={setRequestBody}
                  language="json"
                />
              </div>

              <button
                onClick={sendRequest}
                className="w-full py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Send Request
              </button>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Response</label>
              <APIResponse response={response} />
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
```

### 5. Visual Debugging Tools

#### Agent State Visualizer

```python
# tools/debugger/visualizer.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class AgentDebugger:
    """Visual debugging tool for agent states and decisions."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.history = []

    def capture_state(self):
        """Capture current agent state for visualization."""
        state = {
            'timestamp': time.time(),
            'beliefs': self.agent.beliefs.copy(),
            'energy': self.agent.energy,
            'location': self.agent.location,
            'last_action': self.agent.last_action,
            'free_energy': self.agent.calculate_free_energy()
        }
        self.history.append(state)

    def visualize_beliefs_over_time(self):
        """Create interactive visualization of belief evolution."""
        if not self.history:
            return

        timestamps = [state['timestamp'] for state in self.history]
        belief_states = [state['beliefs'] for state in self.history]

        fig = go.Figure()

        # Add traces for each belief dimension
        for i in range(len(belief_states[0])):
            values = [beliefs[i] for beliefs in belief_states]
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines+markers',
                name=f'Belief {i}',
                hovertemplate=f'Belief {i}: %{{y:.3f}}<br>Time: %{{x}}<extra></extra>'
            ))

        fig.update_layout(
            title=f"Belief Evolution - {self.agent.name}",
            xaxis_title="Time",
            yaxis_title="Belief Strength",
            hovermode='x unified'
        )

        return fig

    def visualize_decision_process(self, step: int):
        """Visualize the Active Inference decision process."""
        if step >= len(self.history):
            return

        state = self.history[step]

        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Current Beliefs", "Free Energy by Action",
                "Energy Level", "Location History"
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "indicator"}, {"type": "scatter"}]
            ]
        )

        # Belief distribution
        fig.add_trace(
            go.Bar(
                x=[f"State {i}" for i in range(len(state['beliefs']))],
                y=state['beliefs'],
                name="Beliefs"
            ),
            row=1, col=1
        )

        # Free energy by potential action
        actions = ['explore', 'exploit', 'cooperate', 'rest']
        free_energies = [
            self.agent.calculate_free_energy_for_action(action)
            for action in actions
        ]
        fig.add_trace(
            go.Bar(
                x=actions,
                y=free_energies,
                name="Free Energy"
            ),
            row=1, col=2
        )

        # Energy gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=state['energy'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Energy"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=1
        )

        # Location history
        locations = [s['location'] for s in self.history[:step+1]]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(locations))),
                y=[hash(loc) % 100 for loc in locations],  # Simple hash for demo
                mode='lines+markers',
                name="Movement"
            ),
            row=2, col=2
        )

        fig.update_layout(
            title=f"Agent Debug View - Step {step}",
            showlegend=False
        )

        return fig
```

### 6. Documentation and Learning Resources

#### Documentation Structure

```
docs/
├── getting-started/
│   ├── installation.md
│   ├── first-agent.md
│   ├── basic-concepts.md
│   └── your-first-coalition.md
├── tutorials/
│   ├── active-inference-explained.md
│   ├── building-agent-behaviors.md
│   ├── coalition-strategies.md
│   └── edge-deployment.md
├── api-reference/
│   ├── agents/
│   ├── coalitions/
│   ├── world/
│   └── utilities/
├── examples/
│   ├── notebooks/
│   ├── code-samples/
│   └── case-studies/
└── advanced/
    ├── performance-optimization.md
    ├── custom-inference-engines.md
    └── extending-freeagentics.md
```

#### Interactive Documentation

```python
# docs/interactive/active_inference_demo.py
"""
Interactive Active Inference Demonstration

This example shows how belief updating works in real-time.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from freeagentics import Agent

st.title("🧠 Active Inference Interactive Demo")
st.markdown("Explore how agents update their beliefs based on observations.")

# Sidebar controls
st.sidebar.header("Agent Configuration")
agent_type = st.sidebar.selectbox("Agent Type", ["Explorer", "Guardian", "Merchant"])
curiosity = st.sidebar.slider("Curiosity", 0.0, 1.0, 0.7)
social = st.sidebar.slider("Social Tendency", 0.0, 1.0, 0.5)

# Create agent
agent = Agent.create(
    agent_type,
    personality={'curiosity': curiosity, 'social': social}
)

# Simulation controls
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Random Observation"):
        agent.observe(np.random.randint(0, 5))
        st.rerun()

with col2:
    if st.button("Positive Event"):
        agent.observe_reward(1.0)
        st.rerun()

with col3:
    if st.button("Reset Agent"):
        agent.reset()
        st.rerun()

# Visualization
st.subheader("Current Belief State")
fig = go.Figure(data=go.Bar(
    x=[f"State {i}" for i in range(len(agent.beliefs))],
    y=agent.beliefs,
    marker_color='lightblue'
))
fig.update_layout(
    title="Agent's Belief Distribution",
    xaxis_title="World States",
    yaxis_title="Belief Strength"
)
st.plotly_chart(fig, use_container_width=True)

# Action selection
st.subheader("Action Selection")
actions = agent.get_available_actions()
action_probs = agent.get_action_probabilities()

fig2 = go.Figure(data=go.Bar(
    x=actions,
    y=action_probs,
    marker_color='lightgreen'
))
fig2.update_layout(
    title="Action Selection Probabilities",
    xaxis_title="Actions",
    yaxis_title="Selection Probability"
)
st.plotly_chart(fig2, use_container_width=True)

# Explanation
st.subheader("What's Happening?")
st.markdown(f"""
- **Current Energy**: {agent.energy}/100
- **Last Action**: {agent.last_action or 'None'}
- **Free Energy**: {agent.calculate_free_energy():.3f}

The agent uses Active Inference to:
1. Maintain beliefs about the world state
2. Predict outcomes of potential actions
3. Select actions that minimize free energy (surprise)
""")
```

## Architectural Compliance

### Directory Structure (ADR-002)

- CLI tools in `tools/cli/`
- Web dashboard in `web/dashboard/`
- Documentation in `docs/`
- Developer utilities in `tools/dev/`

### Dependency Rules (ADR-003)

- Developer tools depend on core domain through interfaces
- No core domain dependencies on development tools
- Development infrastructure isolated from production code

### Naming Conventions (ADR-004)

- CLI commands use kebab-case: `freeagentics agent create`
- Tool files use kebab-case: `development-server.py`
- Dashboard components use PascalCase: `AgentVisualizer`

## Implementation Strategy

### Phase 1: Core Developer Tools

1. Basic CLI with agent/simulation commands
2. Simple web dashboard
3. Jupyter notebook examples
4. API documentation

### Phase 2: Advanced Tooling

5. Visual debugging tools
6. Interactive documentation
7. API playground
8. Performance profiling tools

### Phase 3: IDE Integration

9. VS Code extension
10. TypeScript definitions
11. Language server protocol
12. Code generators

### Phase 4: Community Tools

13. Example gallery
14. Community contributions
15. Plugin system
16. Marketplace

## Success Metrics

### Developer Onboarding

- **Time to First Agent**: <5 minutes from install to running agent
- **Time to First Coalition**: <15 minutes with guided tutorial
- **Documentation Satisfaction**: >4.5/5 in developer surveys
- **Example Completion Rate**: >90% for getting started examples

### Developer Productivity

- **Development Velocity**: Measurable improvement in agent development time
- **Error Resolution Time**: Built-in debugging reduces troubleshooting time
- **Feature Discovery**: Developers find and use advanced features naturally
- **Community Contributions**: Active community submitting examples and tools

## Testing Strategy

### Developer Tool Testing

- CLI command integration tests
- Web dashboard E2E tests
- Documentation example verification
- API playground functionality tests

### User Experience Testing

- Developer onboarding flow testing
- Documentation usability studies
- Tool performance benchmarking
- Community feedback integration

## Consequences

### Positive

- Accelerated developer adoption
- Reduced learning curve for complex concepts
- Higher developer satisfaction and retention
- Strong community ecosystem

### Negative

- Additional maintenance overhead for tools
- Increased complexity in development pipeline
- Documentation maintenance requirements
- Support burden for developer tools

### Risks and Mitigations

- **Risk**: Tools becoming outdated as core library evolves
  - **Mitigation**: Automated tool testing in CI/CD pipeline
- **Risk**: Developer tools introducing performance overhead
  - **Mitigation**: Optional tool activation, performance monitoring
- **Risk**: Documentation becoming stale
  - **Mitigation**: Documentation as code, automated example testing

## Related Decisions

- ADR-002: Canonical Directory Structure
- ADR-003: Dependency Rules
- ADR-007: Testing Strategy Architecture
- ADR-008: API and Interface Layer Architecture

This ADR ensures FreeAgentics provides world-class developer experience that makes AI agent development accessible, productive, and enjoyable while maintaining architectural integrity.
