# FreeAgentics Demo Guide

## 🚀 Quick Start (30 seconds)

For the fastest demo experience:

```bash
# Clone and run
git clone <repository>
cd freeagentics
./demo.sh
```

Open: http://localhost:8000/docs for API documentation
Open: http://localhost:8000/demo for web visualization

## 🎯 Demo Highlights

### 1. Active Inference Agents

- **Real PyMDP integration** with fallback to simplified Active Inference
- **Belief updates** based on observations
- **Free energy minimization** driving action selection
- **Emergent coordination** without explicit communication

### 2. Agent Types

- 🔵 **Explorer**: High exploration drive, seeks unknown areas
- 🟢 **Collector**: Resource-focused, maximizes collection efficiency
- 🟣 **Analyzer**: Balanced approach, considers all factors
- 🟠 **Scout**: Fast movement, covers ground quickly

### 3. Interactive Features

- **Web interface** at `/demo` with real-time visualization
- **API endpoints** for programmatic control
- **Belief visualization** showing agent uncertainty
- **Performance metrics** tracking success rates

## 🛠 Full Setup (Production Ready)

For a complete installation:

```bash
./setup.sh
```

This script:

- ✅ Checks Python 3.10+ installation
- ✅ Creates virtual environment
- ✅ Installs all dependencies (PyMDP, FastAPI, etc.)
- ✅ Sets up PostgreSQL/SQLite database
- ✅ Configures Redis caching (optional)
- ✅ Starts all services with health checks
- ✅ Creates demo agents and world

## 🎮 Demo Scenarios

### Scenario 1: Basic Exploration

1. Start the demo
2. Watch agents explore the grid world
3. Observe belief updates (colored overlays)
4. Notice how they avoid walls and dangers

### Scenario 2: Resource Competition

1. Add multiple collector agents
2. Watch them compete for resources
3. See emergent coordination patterns
4. Track performance metrics

### Scenario 3: Active Inference in Action

1. Focus on a single agent
2. Observe its belief state (uncertainty overlay)
3. Watch free energy minimization
4. See how it balances exploration vs exploitation

## 📊 Technical Details

### Active Inference Implementation

- **Generative Model**: World state representation
- **Belief Updates**: Bayesian inference on observations
- **Policy Selection**: Expected free energy minimization
- **Action Execution**: Environmental interaction

### Architecture

```
Frontend (React/HTML5) ←→ API (FastAPI) ←→ Agents (PyMDP)
                                    ↓
                              Knowledge Graph
                                    ↓
                            Database (PostgreSQL)
```

## 🔧 API Usage Examples

### Create an Agent

```bash
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"name": "Demo Agent", "type": "explorer"}'
```

### Get Agent Status

```bash
curl http://localhost:8000/api/v1/agents/1
```

### Make Agent Act

```bash
curl -X POST http://localhost:8000/api/v1/agents/1/act
```

### Update Agent Beliefs

```bash
curl -X POST http://localhost:8000/api/v1/inference/update_beliefs \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "1", "observation": {"position": [5, 5]}}'
```

## 🧠 Understanding Active Inference

### Key Concepts Demonstrated

1. **Free Energy Minimization**

   - Agents act to minimize surprise
   - Balance between accuracy and complexity
   - Observable in action selection patterns

2. **Belief Updating**

   - Bayesian inference on new observations
   - Uncertainty reduction over time
   - Visual representation as transparency

3. **Epistemic vs Pragmatic Value**

   - Exploration for information (epistemic)
   - Exploitation for rewards (pragmatic)
   - Dynamic balance based on agent type

4. **Emergent Coordination**
   - No explicit communication
   - Shared environment leads to coordination
   - Collective intelligence from individual rationality

## 🎨 Visualization Features

### Real-time Displays

- **Agent positions** with smooth movement
- **Belief overlays** showing uncertainty
- **Resource collection** with particle effects
- **Performance metrics** updating live

### Interactive Elements

- **Click agents** to inspect their state
- **Hover cells** to see world information
- **Control simulation** speed and state
- **Add/remove agents** dynamically

## 🔍 Troubleshooting

### Common Issues

**Demo won't start?**

- Check Python 3.10+ is installed
- Ensure no other services on port 8000
- Try `./demo.sh` for minimal setup

**PyMDP not working?**

- Demo works without PyMDP (simplified Active Inference)
- Install with: `pip install pymdp`
- Check logs for specific errors

**Database errors?**

- Demo uses SQLite by default
- PostgreSQL optional for full features
- Check `DATABASE_URL` in `.env`

**Frontend not loading?**

- Static files served from `/static/`
- Web demo at `/demo` endpoint
- Check browser console for errors

## 📈 Performance Metrics

The demo tracks and displays:

- **Resources collected** per agent
- **Goals reached** (high-value targets)
- **Steps taken** (efficiency measure)
- **Average free energy** (surprise level)
- **Exploration coverage** (area covered)

## 🎯 Next Steps

After the demo:

1. **Explore the API** documentation at `/docs`
2. **Modify agent parameters** in the code
3. **Create custom worlds** with different layouts
4. **Implement new agent types** with unique behaviors
5. **Scale up** with more agents and larger worlds

## 🤖 Technical Implementation Notes

- **Modular design** allows easy extension
- **Production-ready** error handling and logging
- **Cross-platform** compatibility (Linux, macOS, Windows)
- **Docker support** for containerized deployment
- **Comprehensive testing** with 723+ test cases

The demo showcases a fully functional Active Inference multi-agent system ready for research, development, and production use.
