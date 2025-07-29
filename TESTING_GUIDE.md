# Local Testing Guide for E2E Demo

## Changes Made

1. **KnowledgeGraphView.tsx** - Added node click functionality to open details side-sheet
2. **Created test files**:
   - `/tests/unit/test_metrics_endpoint.py` - Tests for /metrics endpoint
   - Updated `/web/__tests__/components/main/KnowledgeGraphView.test.tsx` - Added node click test
3. **Documentation**:
   - `/notes/first-run-gaps.md` - Analysis of missing pieces and implementation status

## How to Test Locally

### 1. Start the Backend (Demo Mode)

```bash
# From project root
cd /home/green/freeagentics

# Create virtual environment if not exists
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Start backend in demo mode (no DATABASE_URL)
make dev
# or
python -m api.main
```

The backend should start on `http://localhost:8000`

### 2. Start the Frontend

In a new terminal:

```bash
# Navigate to web directory
cd /home/green/freeagentics/web

# Install dependencies if needed
npm install

# Start development server
npm run dev
```

The frontend should start on `http://localhost:3001`

### 3. Test Each Component

#### A. Test Metrics Endpoint
```bash
# In a new terminal
curl http://localhost:8000/metrics
```

You should see Prometheus-formatted metrics including:
- `agent_spawn_total`
- `kg_node_total`
- Other system metrics

#### B. Test WebSocket Demo Connection
1. Open browser console at http://localhost:3001
2. You should see "WebSocket connected" message
3. The demo endpoint doesn't require authentication

#### C. Test PromptBar
1. Go to http://localhost:3001/main
2. Type "Explore grid" in the prompt bar
3. Verify:
   - Entry appears in history
   - Settings drawer opens (if implemented)

#### D. Test AgentCreator
1. Click "Create Agent" button
2. Fill in agent details
3. Verify agent appears in list
4. Try deleting an agent

#### E. Test KnowledgeGraph (New Feature!)
1. Create some agents or interact with the system
2. Look for the Knowledge Graph panel
3. When nodes appear, **click on a node**
4. Verify a side sheet opens showing:
   - Node type (with color badge)
   - Node label
   - Node ID
   - Position (if available)

#### F. Test SimulationGrid
1. Create at least one agent
2. Start simulation with play button
3. Verify:
   - Agents appear as colored circles
   - Energy indicators show
   - Agents move smoothly on the grid
   - Hover over agent shows tooltip

#### G. Test Conversation Window
1. Submit prompts
2. Verify 3-message cycle appears:
   - Goal message
   - GMN (Generative Model Network) message
   - Result message

### 4. Run Tests

#### Backend Tests
```bash
# From project root
pytest tests/unit/test_metrics_endpoint.py -v
```

#### Frontend Tests
```bash
# From web directory
cd web
npm test -- --testPathPattern=KnowledgeGraphView
npm test -- --testPathPattern=SimulationGrid
```

### 5. Verify Demo Mode Features

- System runs without PostgreSQL database
- WebSocket connects without authentication
- Mock LLM provider returns reasonable responses
- All UI components function in demo mode

## Expected Results

✅ All components load without errors
✅ WebSocket connects successfully
✅ Can create/delete agents
✅ Knowledge graph shows nodes and clicking opens details
✅ Simulation grid animates smoothly
✅ Metrics endpoint returns Prometheus format
✅ No authentication required in demo mode

## Troubleshooting

1. **Port conflicts**: Make sure 8000 and 3001 are free
2. **WebSocket 403**: Check you're using the demo endpoint
3. **Module not found**: Run `pip install -e .` in project root
4. **Frontend errors**: Run `npm install` in web directory

## Git Commands (when working)

```bash
# Stage all changes
git add -A

# Commit with descriptive message
git commit -m "feat: wire KnowledgeGraph node click for details sheet (demo happy-path)

- Add Sheet component to show node details on click
- Update tests to verify sheet functionality
- Document E2E demo components in first-run-gaps.md
- Verify /metrics endpoint returns Prometheus format

Kent: Tests are minimal and behavior-focused
Martin: Clear separation between viz and UI logic  
Charity: D3 errors logged, interaction errors need enhancement"

# Push to remote
git push origin main
```