# FreeAgentics - Unified Development Environment

## 🚀 60-Second Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/freeagentics.git
cd freeagentics

# 2. Install dependencies
make install

# 3. Start development environment
make dev
```

That's it! The platform automatically detects your environment and configures itself:

- **No DATABASE_URL?** → Demo mode with SQLite in-memory + mock LLM + auto-generated JWT
- **DATABASE_URL set?** → Development mode with real database + standard auth
- **PRODUCTION=true?** → Production mode with strict security

## 🎯 What Just Happened?

When you run `make dev`, the unified provider system:

1. **Detects Mode** - Checks environment variables to determine demo/dev/prod mode
2. **Selects Providers** - Automatically configures:
   - Database: PostgreSQL → SQLite → In-memory
   - Cache: Redis → In-memory dictionary  
   - LLM: OpenAI/Anthropic → Mock responses
   - Auth: Standard JWT → Auto-generated dev token
3. **Starts Services** - Launches backend (port 8000) and frontend (port 3000)

## 🔑 Authentication in Demo Mode

In demo mode, authentication is automatic:

```bash
# Get your dev token and configuration
curl http://localhost:8000/api/v1/dev-config

# Response includes:
{
  "mode": "demo",
  "auth": {
    "token": "eyJ...",  # Auto-generated admin token
    "user": {
      "role": "admin",
      "permissions": ["create_agent", "view_agents", ...]
    }
  }
}
```

The frontend automatically uses this token. For API testing:

```bash
TOKEN=$(curl -s http://localhost:8000/api/v1/dev-config | jq -r .auth.token)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/agents
```

## 🔧 Progressive Enhancement

Start simple, add services as needed:

```bash
# Start with demo mode (no external dependencies)
make dev

# Add real database
export DATABASE_URL=postgresql://user:pass@localhost/mydb
make dev

# Add Redis for caching
export REDIS_URL=redis://localhost:6379
make dev

# Add real LLM
export OPENAI_API_KEY=sk-...
make dev

# Production mode
export PRODUCTION=true
export DATABASE_URL=postgresql://prod-connection-string
make dev
```

## 📊 Provider Status

The system logs its configuration on startup:

```
🎯 Provider Mode: DEMO
  📦 Database: SQLite (in-memory)
  💾 Cache: In-memory dictionary
  🤖 LLM: Mock responses
  🔑 Auth: Auto-generated dev token
```

## 🛠️ Architecture

### Provider Factory Pattern

All external dependencies use a unified provider interface:

```python
from core.providers import get_database, get_rate_limiter, get_llm

# Automatically returns the right implementation
db = get_database()       # PostgreSQL or SQLite
cache = get_rate_limiter() # Redis or in-memory
llm = get_llm()           # OpenAI or mock
```

### Key Files

- `core/providers.py` - Provider factory implementation
- `auth/dev_auth.py` - Development authentication
- `api/v1/dev_config.py` - Development configuration endpoint
- `scripts/dev.py` - Unified development launcher

## 🧪 Testing

The unified system includes comprehensive tests:

```bash
# Unit tests for provider selection
pytest tests/unit/test_unified_dev_mode.py

# Integration tests for end-to-end flow
pytest tests/integration/test_dev_mode_e2e.py

# Full test suite
make test-dev
```

## 🚨 Common Issues

### "403 Forbidden" Errors
- **Cause**: Missing or invalid authentication token
- **Fix**: In demo mode, tokens are auto-injected. Check `/api/v1/dev-config`

### "Database not available"
- **Cause**: No DATABASE_URL and SQLite initialization failed
- **Fix**: The system should auto-fallback to in-memory SQLite. Check logs.

### "Rate limit exceeded"
- **Cause**: Too many requests to an endpoint
- **Fix**: Demo mode has lenient limits. Wait a moment or restart.

## 📚 Migration from Old System

If you were using the old split demo/dev approach:

1. **`make demo`** → Just use `make dev` (auto-detects demo mode)
2. **Demo-specific code** → Removed, providers handle differences
3. **`DEMO_MODE` checks** → Use `ProviderMode.get_mode()` instead
4. **Manual token creation** → Automatic in demo mode

## 🔒 Security Notes

- Dev tokens are for **local development only**
- In production, always use proper authentication
- Demo mode disables security features - never expose publicly
- All providers maintain the same API surface for security