# Testing the Unified Development Environment

## Quick Test Instructions

Follow these steps to test the new unified development environment:

### 1. Clone and Setup (60 seconds)

```bash
# Clone the repository
git clone https://github.com/greenisagoodcolor/FreeAgentics.git
cd FreeAgentics

# Install dependencies (Python + Node.js)
make install
```

### 2. Start in Demo Mode (No External Dependencies)

```bash
# Start the development environment
make dev
```

**What to expect:**

- You'll see: `🎯 Provider Mode: DEMO`
- Backend starts on http://localhost:8000
- Frontend starts on http://localhost:3000
- No database or API keys required!

### 3. Verify Demo Mode Works

#### Test 1: Check Dev Configuration

```bash
# In a new terminal, get the dev config
curl http://localhost:8000/api/v1/dev-config | jq
```

**Expected output:**

```json
{
  "mode": "demo",
  "auth": {
    "token": "eyJ...",  // Auto-generated token
    "user": {
      "role": "admin",
      "permissions": ["create_agent", "view_agents", ...]
    }
  },
  "message": "🎯 Running in demo mode..."
}
```

#### Test 2: Use the API with Auto Token

```bash
# Extract the token
TOKEN=$(curl -s http://localhost:8000/api/v1/dev-config | jq -r .auth.token)

# List agents (should work without manual auth setup!)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/agents
```

**Expected:** `[]` (empty array)

#### Test 3: Create an Agent

```bash
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Agent",
    "template": "basic-explorer",
    "parameters": {"description": "My first agent"}
  }'
```

**Expected:** Returns created agent with ID

#### Test 4: Frontend UI

1. Open http://localhost:3000 in your browser
2. The UI should load without authentication errors
3. You should be able to create and view agents

### 4. Progressive Enhancement Tests

#### Add a Real Database

```bash
# Stop the current dev server (Ctrl+C)

# Set database URL (SQLite file)
export DATABASE_URL=sqlite:///test.db

# Restart
make dev
```

**What changes:**

- You'll see: `🔧 Provider Mode: DEVELOPMENT`
- Data persists between restarts
- Auth is required (no auto-token)

#### Add Redis

```bash
# If you have Redis running locally
export REDIS_URL=redis://localhost:6379
make dev
```

**What changes:**

- Rate limiting uses Redis instead of memory
- Better performance under load

### 5. Verify Tests Pass

```bash
# Run the new unified dev mode tests
pytest tests/unit/test_unified_dev_mode.py -v

# Run integration tests
pytest tests/integration/test_dev_mode_e2e.py -v

# Run quick development tests
make test-dev
```

### 6. Key Things to Verify

✅ **No 403 Errors in Demo Mode**

- All API endpoints should work with the auto-generated token
- No manual authentication setup required

✅ **Clean Provider Detection**

- Check the console output shows correct provider mode
- Verify SQLite is used when no DATABASE_URL

✅ **Frontend Works Immediately**

- UI loads without configuration
- Can create/view agents without setup

✅ **Same Code Paths**

- No special "demo mode" bypasses
- Just different provider implementations

### Common Issues & Solutions

**Port Already in Use**

```bash
# The dev script auto-cleans ports, but if needed:
make kill-ports
```

**Module Import Errors**

```bash
# Ensure you ran install
make install
```

**Frontend Not Starting**

```bash
# Check web directory
cd web && npm install
```

### What Success Looks Like

1. ✅ Single `make dev` command starts everything
2. ✅ No authentication setup needed in demo mode
3. ✅ API endpoints return data, not 403 errors
4. ✅ Frontend connects and shows agents
5. ✅ Clear logs showing which providers are active

### Cleanup

```bash
# Stop all services
make stop

# Remove test database (if created)
rm -f test.db

# Full reset
make reset
```

## Report Results

Please test and report:

1. Did `make dev` work on first try?
2. Could you access the API without auth setup?
3. Did the frontend load and work?
4. Any errors or confusion points?

The goal is "clone → make install → make dev → working UI" with zero configuration!
