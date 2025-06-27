# Getting Started with FreeAgentics

Welcome to FreeAgentics! This guide will help you get up and running with the agent simulator platform.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Your First Agent](#your-first-agent)
4. [Running a Simulation](#running-a-simulation)
5. [Understanding the UI](#understanding-the-ui)
6. [Next Steps](#next-steps)

## Prerequisites

Before you begin, ensure you have the following installed:

### Required Software

- **Node.js** (v18.0.0 or higher)

  ```bash
  # Check version
  node --version

  # Install via nvm (recommended)
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
  nvm install 18
  nvm use 18
  ```

- **Python** (v3.9 or higher)

  ```bash
  # Check version
  python --version

  # Install via pyenv (recommended)
  curl https://pyenv.run | bash
  pyenv install 3.9.0
  pyenv global 3.9.0
  ```

- **Git**

  ```bash
  # Check if installed
  git --version
  ```

### Optional Software

- **Docker** (for containerized deployment)
- **PostgreSQL** (if not using Docker)
- **Redis** (for message queuing)

### System Requirements

- **RAM**: Minimum 4GB, 8GB recommended
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+ recommended)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/freeagentics.git
cd freeagentics
```

### 2. Install Dependencies

#### Frontend Dependencies

```bash
npm install
```

#### Backend Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your settings
# Important variables to configure:
# - DATABASE_URL: PostgreSQL connection string
# - REDIS_URL: Redis connection string
# - API_KEY: Your API key for LLM services
# - PORT: Port for the web server (default: 3000)
```

### 4. Initialize Database

```bash
# Run database migrations
npm run db:migrate

# Seed with example data (optional)
npm run db:seed
```

### 5. Start the Application

#### Development Mode

```bash
# Start all services
npm run dev

# Or start services separately:
# Terminal 1: Frontend
npm run dev:frontend

# Terminal 2: Backend
npm run dev:backend

# Terminal 3: Worker processes
npm run dev:worker
```

#### Production Mode

```bash
# Build the application
npm run build

# Start production server
npm start
```

### 6. Verify Installation

Open your browser and navigate to:

- Frontend: `http://localhost:3000`
- API: `http://localhost:3000/api`
- Documentation: `http://localhost:3000/docs`

You should see the FreeAgentics dashboard.

## Your First Agent

Let's create your first agent! We'll create a simple Explorer agent.

### Using the Web Interface

1. **Navigate to Agent Creator**
   - Click "Agents" in the navigation bar
   - Click "Create New Agent" button

2. **Configure Basic Information**
   - **Name**: "Explorer Alpha"
   - **Class**: Select "Explorer"
   - **Starting Position**: Leave as default or click on the map

3. **Set Personality Traits**
   Use the sliders to set:
   - **Openness**: 80 (high curiosity)
   - **Conscientiousness**: 70 (organized exploration)
   - **Extraversion**: 60 (moderate social interaction)
   - **Agreeableness**: 70 (cooperative)
   - **Neuroticism**: 30 (stable under pressure)

4. **Generate Backstory** (Optional)
   - Click "Generate Backstory"
   - The AI will create a unique history for your agent

5. **Create Agent**
   - Review the GNN model preview
   - Click "Create Agent"
   - Your agent will appear on the world map!

### Using the API

```python
import requests

# API endpoint
url = "http://localhost:3000/api/agents/create"

# Agent configuration
agent_data = {
    "name": "Explorer Beta",
    "class": "explorer",
    "personality": {
        "openness": 0.8,
        "conscientiousness": 0.7,
        "extraversion": 0.6,
        "agreeableness": 0.7,
        "neuroticism": 0.3
    },
    "position": {
        "x": 10,
        "y": 10
    }
}

# Create agent
response = requests.post(url, json=agent_data)
agent = response.json()

print(f"Created agent: {agent['id']}")
```

## Running a Simulation

### Quick Start Simulation

1. **Access Simulation Control**
   - Go to the main dashboard
   - Click "Start Simulation" button

2. **Configure Simulation**
   - **Speed**: 1x (real-time) to 10x (fast)
   - **Duration**: Set cycles or run indefinitely
   - **Auto-pause**: Enable for specific events

3. **Monitor Progress**
   - Watch agents move on the hexagonal grid
   - View real-time statistics
   - Check message logs
   - Monitor resource levels

### Simulation Scenarios

Try these pre-built scenarios:

#### Resource Scarcity

```bash
npm run scenario:scarcity
```

Agents must cooperate to survive limited resources.

#### Trade Network

```bash
npm run scenario:trade
```

Merchants establish trade routes between settlements.

#### Knowledge Quest

```bash
npm run scenario:knowledge
```

Scholars work together to solve puzzles.

#### Territory Defense

```bash
npm run scenario:defense
```

Guardians protect against environmental threats.

## Understanding the UI

### Main Dashboard

The dashboard is divided into several sections:

#### 1. World View (Center)

- **Hexagonal Grid**: H3-based world representation
- **Agent Icons**: Different colors/shapes for each class
- **Resources**: Shown as colored dots
- **Terrain**: Different shades indicate terrain types

**Controls**:

- **Pan**: Click and drag
- **Zoom**: Mouse wheel or pinch
- **Select**: Click on agents or cells
- **Multi-select**: Shift+click

#### 2. Agent Panel (Left)

- **Agent List**: All active agents
- **Filters**: Filter by class, status, etc.
- **Quick Actions**: Pause, inspect, or control agents

**Agent Status Indicators**:

- 🟢 Active and healthy
- 🟡 Low resources
- 🔴 Critical state
- ⚫ Inactive

#### 3. Statistics Panel (Right)

- **Population**: Agent count by class
- **Resources**: Global resource levels
- **Knowledge**: Collective knowledge nodes
- **Trade**: Economic activity

**Graphs**:

- Population over time
- Resource distribution
- Knowledge growth
- Trade volume

#### 4. Message Log (Bottom)

- **All Messages**: Complete communication log
- **Filters**: By type, sender, or content
- **Search**: Find specific messages

**Message Types**:

- 💬 Text communication
- 🤝 Trade offers
- 📚 Knowledge sharing
- ⚠️ Warnings
- 🎯 Coordination

### Agent Inspector

Double-click any agent to open the inspector:

#### Overview Tab

- Basic information
- Current status
- Resource levels
- Recent actions

#### Personality Tab

- Personality traits
- Behavioral tendencies
- Learning progress

#### Knowledge Tab

- Knowledge graph visualization
- Known locations
- Discovered patterns
- Shared knowledge

#### History Tab

- Movement history
- Communication log
- Trade history
- Important events

### World Editor

Access via "World" → "Edit Mode":

#### Terrain Tools

- **Brush**: Paint terrain types
- **Fill**: Fill connected areas
- **Generate**: Procedural generation

#### Resource Placement

- **Add**: Click to place resources
- **Remove**: Right-click to remove
- **Scatter**: Random distribution

#### Spawn Points

- Set agent spawn locations
- Create spawn zones
- Configure spawn rules

## Next Steps

### Tutorials

1. **[Creating Custom Agents](agent_creator_guide.md)**
   - Advanced personality configuration
   - Custom GNN models
   - Behavioral scripting

2. **[World Building](world_building_guide.md)**
   - Designing environments
   - Resource economics
   - Environmental challenges

3. **[Active Inference Basics](../active_inference_guide.md)**
   - Understanding the theory
   - Practical applications
   - Tuning parameters

### Examples

Explore the `examples/` directory:

```bash
# Basic agent creation
python examples/basic_agent.py

# Multi-agent scenario
python examples/multi_agent_scenario.py

# Custom GNN model
python examples/custom_gnn_model.py

# API integration
python examples/api_client.py
```

### Advanced Topics

- **[Performance Optimization](performance_guide.md)**
- **[Deployment Options](deployment_guide.md)**
- **[API Reference](../api/rest_api.md)**
- **[Contributing](../../CONTRIBUTING.md)**

## Troubleshooting

### Common Issues

#### Port Already in Use

```bash
# Find process using port 3000
lsof -i :3000  # macOS/Linux
netstat -ano | findstr :3000  # Windows

# Kill the process or use different port
PORT=3001 npm run dev
```

#### Database Connection Failed

```bash
# Check PostgreSQL is running
pg_isready

# Verify connection string in .env
# Format: postgresql://user:password@localhost:5432/freeagentics
```

#### Module Not Found

```bash
# Clear caches and reinstall
rm -rf node_modules package-lock.json
npm install

# For Python
pip install -r requirements.txt --force-reinstall
```

#### Performance Issues

- Reduce number of agents
- Lower simulation speed
- Disable visual effects
- Use production build

### Getting Help

- **Documentation**: Check our comprehensive docs
- **GitHub Issues**: Report bugs or request features
- **Discussions**: Join community discussions
- **Discord**: Real-time chat with developers

## Summary

You've successfully:

- ✅ Installed FreeAgentics
- ✅ Created your first agent
- ✅ Run a simulation
- ✅ Explored the UI

Welcome to the FreeAgentics community! We're excited to see what you'll create.

---

**Ready to dive deeper?** Check out the [Agent Creator Guide](agent_creator_guide.md) to learn about advanced agent configuration and custom behaviors.
