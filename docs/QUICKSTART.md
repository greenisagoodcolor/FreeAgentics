# FreeAgentics Quick Start

> Get running with autonomous AI agents in 5 minutes

## Prerequisites

- Python 3.9+ with pip
- Node.js 18+ with npm
- 4GB+ RAM recommended

## Installation

```bash
# Clone and enter directory
git clone https://github.com/your-org/freeagentics.git
cd freeagentics

# Install dependencies
pip install -e .
cd web && npm install && cd ..

# Start development server
npm run dev
```

Visit **<http://localhost:3000>** to access the FreeAgentics platform.

## Create Your First Agent (2 minutes)

### Option 1: Web Interface

1. Navigate to <http://localhost:3000/agents>
2. Click "Create Agent"
3. Select "Explorer" template
4. Adjust personality sliders (curiosity: 0.8, caution: 0.3)
5. Click "Create Agent"

### Option 2: Python Code

```python
from freeagentics import Agent, World

# Create an autonomous explorer
explorer = Agent.create("Explorer",
                       name="Scout",
                       personality={'curiosity': 0.8, 'caution': 0.3})

# Create a world and add the agent
world = World(grid_size=20, resource_density=0.3)
world.add_agent(explorer)

# Run simulation
for step in range(50):
    world.step()
    print(f"Step {step}: Agent at {explorer.location}")
```

## Watch Your Agent in Action

1. Go to **<http://localhost:3000/world>**
2. See your agent moving autonomously based on Active Inference
3. Monitor belief states at **<http://localhost:3000/active-inference-demo>**

## Next Steps

- **Learn More**: [User Guide](USER-GUIDE.md) - Complete feature walkthrough
- **Develop**: [Developer Guide](DEVELOPER-GUIDE.md) - Setup development environment
- **Deploy**: [Deployment Guide](DEPLOYMENT.md) - Production deployment options
- **Understand**: [Architecture](ARCHITECTURE.md) - Technical deep dive

## Troubleshooting

- **Port 3000 in use**: Change port with `PORT=3001 npm run dev`
- **Python package issues**: Use virtual environment: `python -m venv venv && source venv/bin/activate`
- **Node version issues**: Use Node 18+ or install via [nvm](https://github.com/nvm-sh/nvm)

For more help, see [Troubleshooting Guide](TROUBLESHOOTING.md).

---

_You now have autonomous AI agents running! 🎉_
