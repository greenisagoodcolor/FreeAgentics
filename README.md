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

The system auto-configures on first run:
- Uses SQLite if no PostgreSQL is configured
- Creates `.env` from `.env.development` template
- Generates development auth tokens

For production setup with PostgreSQL:
```bash
cp .env.development .env
# Edit .env and set DATABASE_URL
```

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

- **Port already in use**: Run `make kill-ports`
- **Module not found**: Run `make clean && make install`
- **Database issues**: Check `make status` output
- **Frontend not loading**: Check console for errors at http://localhost:3000


## Documentation

- [Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md)
- [API Reference](docs/api/API_REFERENCE.md)
- [Environment Setup](ENVIRONMENT_SETUP.md)
- [Contributing Guide](CONTRIBUTING.md)



## License

MIT License - see [LICENSE](LICENSE) file for details.
