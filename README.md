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

🎯 **That's it!** Open http://localhost:3000 and start exploring:

- **Mock AI responses** - No API keys needed
- **In-memory database** - No setup required
- **Real-time updates** - Full WebSocket functionality
- **Agent communication** - See multi-agent conversations

**Demo Features Ready:**

- Create and manage Active Inference agents
- Watch agents explore the grid world
- View the knowledge graph build in real-time
- Test the conversation interface
- Explore all UI components

### Option 2: Development Mode (Real AI)

For real OpenAI responses and persistent data:

```bash
# After running the demo mode steps above
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-your-key-here
# Optionally set DATABASE_URL for PostgreSQL

make dev  # Restart with real providers
```

### 3. Test the System

Try these example prompts in the UI:

- **Demo Mode**: "Create an agent to explore the environment"
- **With API Key**: "Help me create a sustainable business plan"
- **Multi-Agent**: "Have two agents discuss active inference theory"

### Troubleshooting

- **Port conflicts**: Run `make kill-ports` then `make dev`
- **Dependencies missing**: Run `make clean && make install`
- **Not working**: Check `make status` for diagnostics

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What is FreeAgentics?

FreeAgentics creates AI agents using Active Inference - a mathematical framework from cognitive science. Unlike chatbots or scripted AI, our agents make decisions by minimizing free energy, leading to emergent, intelligent behavior.

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
