# FreeAgentics

> **Multi-agent AI platform implementing Active Inference for autonomous, mathematically-principled intelligent systems**

## Quick Start

```bash
git clone https://github.com/greenisagoodcolor/FreeAgentics.git
cd FreeAgentics
make install
make dev
```

✅ **That's it!** You should see:
- Backend API running at http://localhost:8000
- Frontend app running at http://localhost:3000
- Auto-configured database (SQLite if no PostgreSQL found)

If you encounter any issues, run `make status` to diagnose.

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
