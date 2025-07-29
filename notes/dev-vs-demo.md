# Dev vs Demo Comparison Analysis

## Current State Analysis

### Make Targets

**`make demo`:**
- Sets `unset DATABASE_URL` to trigger demo mode  
- Calls `make dev` internally
- No other differences

**`make dev`:**
- Verifies Python venv exists
- Checks for node_modules
- Kills port conflicts (3000, 8000)
- Starts backend with `.env` loaded
- Starts frontend with `npm run dev`

### Demo Mode Detection

**Backend (`database/session.py`):**
```python
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    warnings.warn("⚠️ Running in demo mode without database...")
    DATABASE_URL = None  # Explicitly set to None for demo mode
```

**WebSocket (`api/v1/websocket.py`):**
```python
DEMO_MODE = os.getenv("DATABASE_URL") is None
```

### Key Dependencies & Failures

1. **Database Access (403 source #1)**
   - `/api/agents` requires `get_db()` dependency
   - `get_db()` fails if `SessionLocal` is None (demo mode)
   - Results in 500 error, not 403

2. **Authentication (403 source #2)**
   - All agent endpoints have `@require_permission` decorator
   - Requires valid JWT token via `get_current_user`
   - No token = 401, invalid permissions = 403
   - No bypass for demo mode

3. **Rate Limiting (potential 403 source #3)**
   - Configured in `config/rate_limiting.py`
   - Uses Redis if available
   - Falls back to in-memory if no Redis

### Missing Routes
- Frontend expects routes that may not exist
- Need to verify all UI-referenced endpoints exist

### Current Problems

1. **No unified provider injection** - demo detection scattered across files
2. **Auth is mandatory** - no dev/demo bypass, no auto-generated tokens
3. **Database required** - no SQLite fallback implemented despite comments
4. **WebSocket demo mode** - only WS has proper demo handling at `/api/v1/ws/demo`
5. **Multiple README files** - confusing onboarding experience