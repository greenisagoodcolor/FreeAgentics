# FreeAgentics

> **Multi-agent AI platform implementing Active Inference for autonomous, mathematically-principled intelligent systems**

## Quick Start

### Option 1: Demo Mode (Zero Setup - Recommended)

Experience FreeAgentics immediately without any configuration:

```bash
git clone https://github.com/greenisagoodcolor/FreeAgentics.git
cd FreeAgentics
make install
make dev
```

🎯 **That's it!** The system will start up and guide you through each step:

#### Step-by-Step Developer Onboarding

**Step 1: Installation (2-3 minutes)**
```bash
$ make install
```
You'll see:
- ✅ Python virtual environment created
- ✅ Python dependencies installed (FastAPI, PyMDP, etc.)
- ✅ Node modules installed (Next.js, React, etc.)
- ✅ Development tools configured

**Step 2: Start Development Environment (30 seconds)**
```bash
$ make dev
```
You'll see:
- 🔥 Backend starting on http://localhost:8000
- ⚛️  Frontend starting on http://localhost:3000
- ✅ WebSocket connections established
- 📊 In-memory database ready

**Step 3: Access the Application**
Open http://localhost:3000 in your browser. You should see:
- 🎨 Clean UI with dark theme
- 💬 Prompt bar at the bottom
- 📊 Empty metrics panel (no agents yet)
- 🌐 Empty knowledge graph visualization

**Step 4: Create Your First Agent**
Type in the prompt bar: `"Create an agent to explore the environment"`

You'll observe:
1. **Conversation starts** - Two agents (Advocate & Analyst) appear
2. **Real-time updates** - Messages stream as agents discuss
3. **Agent creation** - A new explorer agent appears in the grid
4. **Knowledge graph updates** - Nodes and connections form

**Step 5: Explore Core Features**
- **Multi-Agent Chat**: Watch the Advocate and Analyst discuss your request
- **Grid World**: See your explorer agent move around the environment
- **Knowledge Graph**: Click nodes to see agent beliefs and relationships
- **Metrics Panel**: Monitor agent performance and system health

**Demo Features Ready:**

- ✅ Create and manage Active Inference agents
- ✅ Watch agents explore the grid world
- ✅ View the knowledge graph build in real-time
- ✅ Test the conversation interface
- ✅ Explore all UI components

### Option 2: Development Mode (Real AI)

For real OpenAI responses and persistent data:

**Step 1: Configure API Keys**
```bash
cp .env.example .env
```

**Step 2: Edit .env file**
```bash
# Add your OpenAI API key:
OPENAI_API_KEY=sk-your-key-here

# Optional: Add PostgreSQL for persistence
DATABASE_URL=postgresql://user:password@localhost/freeagentics
```

**Step 3: Restart with Real Providers**
```bash
make dev
```

You'll notice:
- 🤖 Real AI responses instead of mock data
- 💾 Persistent database (if configured)
- 🧠 Actual LLM-generated agent behaviors
- 📈 More sophisticated knowledge graph growth

### Testing the System

**Example Prompts to Try:**

1. **Basic Agent Creation** (Demo Mode)
   - `"Create an agent to explore the environment"`
   - Expected: Explorer agent appears and starts moving

2. **Business Planning** (Requires API Key)
   - `"Help me create a sustainable business plan"`
   - Expected: Agents discuss and analyze business strategies

3. **Theoretical Discussion** (Best with API Key)
   - `"Have two agents discuss active inference theory"`
   - Expected: Deep conversation about mathematical principles

4. **Complex Task** (Requires API Key)
   - `"Design a multi-agent system for climate monitoring"`
   - Expected: Multiple specialized agents created with specific roles

### What You Should See Working

✅ **WebSocket Connections**: Real-time bidirectional communication
✅ **Agent Conversations**: Natural dialogue between AI agents
✅ **Knowledge Graph Growth**: Nodes and edges forming as agents interact
✅ **Grid World Actions**: Agents moving and exploring autonomously
✅ **Belief Updates**: Agent mental states evolving based on observations
✅ **Goal-Directed Behavior**: Agents following user-specified objectives

### Troubleshooting

**Issue: Port conflicts**
```bash
make kill-ports  # Kill processes on ports 3000 and 8000
make dev         # Restart
```

**Issue: Dependencies missing**
```bash
make clean       # Remove all dependencies
make install     # Fresh install
make dev         # Start again
```

**Issue: WebSocket errors**
```bash
make status      # Check service health
# Look for "WebSocket: Connected" status
```

**Issue: Not sure what's happening**
```bash
make logs        # View backend logs
# Check for error messages or warnings
```

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What is FreeAgentics?

FreeAgentics creates AI agents using Active Inference - a mathematical framework from cognitive science. Unlike chatbots or scripted AI, our agents make decisions by minimizing free energy, leading to emergent, intelligent behavior.

### 🚀 Complete LLM→GMN→PyMDP→Knowledge Graph Cycle

FreeAgentics implements a complete cognitive architecture where:

1. **Natural language goals** are converted to **GMN specifications** via LLMs
2. **GMN specs** create **PyMDP Active Inference agents**
3. **PyMDP agents** take actions and update their **beliefs**
4. **Agent actions** update the **knowledge graph**
5. **Knowledge graph** provides context for the **next LLM generation**

### 🎯 Key Features

- **Multi-Agent Conversations**: Watch AI agents discuss and collaborate in real-time
- **Active Inference**: Agents use PyMDP for probabilistic reasoning and decision-making
- **Knowledge Graph**: Live visualization of agent beliefs, goals, and relationships
- **GMN Generation**: Convert natural language into formal agent specifications
- **Real-time Updates**: WebSocket integration for instant feedback
- **Zero Setup Demo**: Experience everything without API keys or configuration

## Requirements

- Python 3.9+
- Node.js 18+
- Git

## Basic Commands

```bash
make install    # Install all dependencies
make dev        # Start development servers
make test       # Run tests
make stop       # Stop all servers
make status     # Check environment status
make clean      # Clean build artifacts
make reset      # Full reset (removes dependencies)
```

## 🔄 The Complete Cognitive Cycle

### 1. Create Agents from Natural Language

**Demo Mode (No API Key)**:

```bash
curl -X POST "http://localhost:8000/api/v1/prompts/demo" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create an agent that explores unknown environments",
    "agent_name": "Explorer"
  }'
```

**With LLM (Requires API Key)**:

```bash
curl -X POST "http://localhost:8000/api/v1/prompts" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create an agent that balances exploration and exploitation",
    "agent_name": "Optimizer",
    "llm_provider": "openai"
  }'
```

### 2. Multi-Agent Conversations

Start a conversation between multiple Active Inference agents:

```bash
curl -X POST "http://localhost:8000/api/v1/agent-conversations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Discuss strategies for sustainable energy",
    "agent_count": 3,
    "conversation_turns": 5
  }'
```

### 3. View the Knowledge Graph

The knowledge graph automatically updates as agents interact:

```bash
curl "http://localhost:8000/api/knowledge-graph"
```

### 4. The Feedback Loop

1. **Goal Prompt** → LLM generates GMN specification
2. **GMN** → Creates PyMDP Active Inference model
3. **PyMDP** → Agent takes actions based on beliefs
4. **Actions** → Update knowledge graph
5. **Knowledge Graph** → Provides context for next iteration

This creates a continuous learning loop where agents become more intelligent over time!

## Configuration

### Zero-Setup Demo Mode (Default)

FreeAgentics automatically detects when no configuration is provided and switches to demo mode:

- **SQLite in-memory database** - No installation needed
- **Demo WebSocket endpoint** - Auto-connects to `/api/v1/ws/demo`
- **Mock LLM providers** - Realistic AI responses without API keys
- **In-memory caching** - No Redis required
- **Auto-generated auth tokens** - Skip complex authentication setup
- **Real-time WebSocket** - Full functionality including live updates

### Custom Configuration

Copy the comprehensive example file and customize as needed:

```bash
cp .env.example .env
# Edit .env with your preferences
```

**Key Settings:**

```bash
# For real AI responses
OPENAI_API_KEY=sk-your-key-here

# For persistent data
DATABASE_URL=postgresql://user:pass@host:port/database

# For production caching
REDIS_URL=redis://localhost:6379/0
```

The `.env.example` file includes detailed documentation for all 100+ available settings.

### PostgreSQL + pgvector Setup (Optional)

For production with vector storage:

```bash
# Install PostgreSQL with pgvector extension
# Ubuntu/Debian:
sudo apt install postgresql postgresql-contrib
sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Set DATABASE_URL in .env:
DATABASE_URL=postgresql://username:password@localhost:5432/freeagentics
```

**Note**: SQLite works fine for development and small deployments.

## Project Structure

```
/
├── agents/          # Active Inference agents
├── api/             # FastAPI backend
├── web/             # Next.js frontend
├── inference/       # PyMDP integration
├── database/        # SQLAlchemy models
└── tests/           # Test suite
```

## Troubleshooting

### Quick Diagnostics

```bash
make status        # Check environment and service status
make kill-ports    # Stop conflicting processes
make clean         # Remove build artifacts
make install       # Reinstall dependencies
make dev          # Start fresh
```

### WebSocket Connection Issues

- **Connection refused**: Check `NEXT_PUBLIC_WS_URL` in `.env` (leave empty for demo mode)
- **Authentication errors**: Demo mode doesn't require auth. For dev mode, ensure valid JWT token
- **Connection drops**: Check browser console, enable debug logging with `ENABLE_WEBSOCKET_LOGGING=true`
- **Testing WebSocket**: `wscat -c ws://localhost:8000/api/v1/ws/demo`

See [WebSocket API Documentation](docs/api/WEBSOCKET_API.md#debugging-websocket-connections) and [WebSocket Testing Guide](docs/WEBSOCKET_TESTING_GUIDE.md) for detailed debugging.

### Common Issues

**Service Won't Start:**

```bash
# Check if ports are in use
make kill-ports && make dev

# Verify dependencies
make status

# Full reset if needed
make reset && make install && make dev
```

**Frontend Not Loading:**

- Ensure backend is running: http://localhost:8000/health
- Check frontend port: usually http://localhost:3000
- Look for port conflicts in terminal output

**API/Database Errors:**

- Demo mode should work without any setup
- If using custom config, verify `.env` file settings
- Check logs in terminal for specific error messages

**Performance Issues:**

- Demo mode uses in-memory database (data resets on restart)
- For persistent data, set `DATABASE_URL` in `.env` file
- Reduce `MAX_AGENTS_PER_USER` in `.env` if needed

### Getting Help

1. Check `make status` output
2. Look for error messages in terminal
3. Verify http://localhost:8000/health returns OK
4. Try demo mode first (no configuration needed)

## Documentation

- [Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md)
- [API Reference](docs/api/API_REFERENCE.md)
- [Environment Setup](ENVIRONMENT_SETUP.md)
- [Contributing Guide](CONTRIBUTING.md)

## License

MIT License - see [LICENSE](LICENSE) file for details.
