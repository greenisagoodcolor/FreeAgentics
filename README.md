# FreeAgentics

> **Multi-agent AI platform implementing Active Inference for autonomous, mathematically-principled intelligent systems**

## Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/greenisagoodcolor/FreeAgentics.git
cd FreeAgentics
make install
```

### 2. Start Development Servers
```bash
make dev        # This will start servers and keep running - press Ctrl+C to stop
```

✅ **Almost there!** After running `make dev`, you'll see:
- Backend API running at http://localhost:8000
- Frontend app running at http://localhost:3000
- The terminal will show logs from both servers (this is normal)

### 3. Enable Agent Conversations (Required)
🔑 **Critical Step**: To enable AI agents to converse with each other, you must add your OpenAI API key:

1. Open http://localhost:3000 in your browser
2. Click the **Settings** button (gear icon) in the top-right corner
3. Paste your OpenAI API key (get one at https://platform.openai.com/api-keys)
4. Click **Save**

**Without an API key, agents cannot communicate!** The system will show errors if you try to use it without configuring your key first.

### 4. Test Agent Conversations
Once your API key is saved, try these prompts:
- "Help me create a sustainable coffee shop business plan"
- "Can my agents talk to each other about active inference?"
- "Design an AI-powered learning platform"

**Note**: `make dev` keeps running to serve your application. Open a new terminal for other commands.

If you encounter any issues, run `make status` in a new terminal to diagnose.

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

The system auto-configures for demo mode:
- Uses SQLite if no PostgreSQL is configured
- Mock LLM providers work without API keys
- Demo WebSocket connections for real-time updates
- Creates `.env` from `.env.development` if missing

**🎯 Zero Setup Required**: The app works immediately with mock data and providers.

For production with real APIs:
```bash
cp .env.example .env
# Edit .env and add your API keys and database URL
```

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

**Common Issues:**
- **Port conflicts**: Run `make kill-ports` then `make dev`
- **Dependencies missing**: Run `make clean && make install`
- **Build errors**: Check TypeScript compilation with `cd web && npm run build`
- **Frontend won't load**: Verify backend is running at http://localhost:8000/health
- **Agent errors**: App works in demo mode without API keys - add real keys in Settings

**Quick Fixes:**
```bash
make kill-ports    # Stop all services
make clean         # Clean build artifacts  
make install       # Reinstall dependencies
make dev          # Restart everything
```

If you're still having issues, the demo mode should work immediately without any external dependencies.


## Documentation

- [Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md)
- [API Reference](docs/api/API_REFERENCE.md)
- [Environment Setup](ENVIRONMENT_SETUP.md)
- [Contributing Guide](CONTRIBUTING.md)



## License

MIT License - see [LICENSE](LICENSE) file for details.
