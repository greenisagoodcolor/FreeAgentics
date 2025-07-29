# FreeAgentics Demo Quick Start Guide

## 🚀 One-Command Demo Setup

Get FreeAgentics running in demo mode in under 2 minutes:

```bash
# Clone and enter the project
git clone https://github.com/greenisagoodcolor/freeagentics.git
cd freeagentics

# Install everything (backend + frontend)
make install

# Run in demo mode (no database required!)
make demo
```

That's it! The system will start with:
- ✅ Backend API on http://localhost:8000
- ✅ Frontend UI on http://localhost:3001
- ✅ Demo WebSocket (no auth required)
- ✅ Mock LLM provider (no API keys needed)

## 📋 What You'll See

Open http://localhost:3001 in your browser. You'll see a single-page application with:

```
┌─────────────────────────────────────────────────────┐
│                   PromptBar (Top)                   │
├─────────────────┬─────────────────┬─────────────────┤
│  AgentCreator   │  Conversation   │  KnowledgeGraph │
│     Panel       │     Window      │      View       │
├─────────────────┴─────────────────┴─────────────────┤
│              SimulationGrid (Full Width)            │
├─────────────────────────────────────────────────────┤
│                 MetricsFooter                       │
└─────────────────────────────────────────────────────┘
```

## 🎮 Try These Features

1. **Create an Agent**
   - Click "Create Agent" in the left panel
   - Give it a name like "Explorer 1"
   - Click "Create"

2. **Submit a Prompt**
   - Type in the top prompt bar: "Explore the grid and find resources"
   - Press Enter

3. **Watch the Knowledge Graph**
   - See nodes appear as agents process information
   - **Click any node** to see details in a side panel (new feature!)

4. **Start the Simulation**
   - Press the Play button in the Simulation Grid
   - Watch agents move on the hex grid
   - Hover over agents to see their status

5. **Check Metrics**
   - Visit http://localhost:8000/metrics for Prometheus metrics
   - See counters like `agent_spawn_total` and `kg_node_total`

## 🛠️ Troubleshooting

### "Command not found: make"
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install

# Or use direct commands:
python3 -m venv venv
source venv/bin/activate
pip install -e .
cd web && npm install && cd ..
python -m api.main & cd web && npm run dev
```

### "Port already in use"
```bash
# Kill processes on ports
lsof -ti:8000 | xargs kill -9  # Backend
lsof -ti:3001 | xargs kill -9  # Frontend
```

### "Module not found"
```bash
# Ensure you're in virtual environment
source venv/bin/activate
pip install -e .
```

## 🔍 What's Happening Under the Hood

When you run `make demo`, it:

1. **Detects Demo Mode** - No DATABASE_URL means demo mode
2. **Starts Mock Services** - In-memory database, mock LLM provider
3. **Opens WebSocket** - `/api/v1/ws/demo` (no auth required)
4. **Loads UI** - All components on single page at localhost:3001

## 📚 Next Steps

- **Explore Code**: Start with `/agents/base_agent.py` for Active Inference
- **Run Tests**: `make test` to see all tests pass
- **Full Pipeline Demo**: `python examples/demo_full_pipeline.py`
- **API Docs**: http://localhost:8000/docs (auto-generated)

## 🚨 Demo Limitations

Demo mode uses:
- In-memory storage (data lost on restart)
- Mock LLM responses (not real AI)
- Simplified agent behaviors
- No authentication required

For production features, set up PostgreSQL and real LLM providers.

---

**Questions?** Check `/docs` or open an issue on GitHub!