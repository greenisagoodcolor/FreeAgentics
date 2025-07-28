# FreeAgentics Demo Video Script

## Duration: 5 minutes

### Opening (30 seconds)
**[Screen: Terminal with clean background]**

**Narrator:** "In just 30 seconds, I'll show you a working Active Inference multi-agent system that demonstrates cutting-edge AI research in action."

**[Type commands]**
```bash
git clone <repository> && cd freeagentics
./demo.sh
```

**Narrator:** "One command. That's all it takes."

### Setup Phase (45 seconds)
**[Screen: Demo script running with colored output]**

**Narrator:** "Watch as the system automatically sets up Python environments, installs dependencies, and creates our Active Inference agents. No configuration needed - it just works."

**[Show log output highlighting key setup steps]**
- ✓ Creating virtual environment...
- ✓ Installing core dependencies...
- ✓ Starting API server...
- ✓ Creating demo agents...

**Narrator:** "The system is running in demo mode - no database setup, no complex configuration. Perfect for rapid prototyping and demonstration."

### API Documentation (45 seconds)
**[Screen: Browser opening to http://localhost:8000/docs]**

**Narrator:** "Here's our FastAPI documentation - a complete Active Inference API with 98 endpoints. Notice the clean, professional interface and comprehensive coverage."

**[Scroll through key endpoints]**
- Agent management
- Active Inference operations  
- Real-time monitoring
- Knowledge graph integration

**Narrator:** "Each endpoint is fully documented with examples. This isn't a mockup - it's production-ready code."

### Web Visualization (2 minutes)
**[Screen: Browser to http://localhost:8000/demo]**

**Narrator:** "But here's where it gets impressive. A real-time Active Inference visualization built from scratch."

**[Show the demo interface loading]**

**Narrator:** "Four types of agents - Explorers, Collectors, Analyzers, and Scouts - each with different behavioral priorities based on Karl Friston's Active Inference theory."

**[Click Start button]**

**Narrator:** "Watch them move. Each step they take is driven by free energy minimization - the mathematical principle underlying biological intelligence."

**[Point to belief overlays]**

**Narrator:** "These colored overlays represent agent beliefs. As they explore, their uncertainty decreases. This is genuine Active Inference, not a simulation."

**[Show real-time statistics]**

**Narrator:** "Notice the statistics updating in real-time - resources collected, goals reached, free energy levels. You can see the agents learning and adapting."

**[Add a new agent]**

**Narrator:** "I can add new agents dynamically. Watch how they coordinate without explicit communication - emergent behavior from individual rationality."

**[Speed up simulation]**

**Narrator:** "Speed it up to see long-term patterns. Collectors focus on resources, Explorers seek unknown areas, Analyzers balance both strategies."

### Technical Depth (45 seconds)
**[Screen: API calls in terminal]**

**Narrator:** "Behind this interface is a complete API. Let me show you..."

**[Show curl commands]**
```bash
curl http://localhost:8000/api/v1/agents
curl -X POST http://localhost:8000/api/v1/agents/1/act
```

**Narrator:** "Real agents responding to real API calls. The beliefs update, the free energy minimizes, the learning happens - all through standard REST endpoints."

**[Show belief data]**

**Narrator:** "Look at the raw data - belief matrices, uncertainty quantification, action probabilities. This is the mathematics of consciousness made accessible."

### Real-World Applications (30 seconds)
**[Screen: Code repository view]**

**Narrator:** "This system isn't just a demo. It's a platform for serious AI research with applications in:"

**[Show file structure]**
- Autonomous robotics
- Adaptive user interfaces  
- Distributed systems
- Cognitive science research

**Narrator:** "Over 5000 tests, comprehensive documentation, production-ready error handling, security implementation."

### Closing (15 seconds)
**[Screen: Back to running demo]**

**Narrator:** "From zero to running Active Inference system in 30 seconds. From simple demo to research platform with endless possibilities."

**[Show GitHub repository URL]**

**Narrator:** "This is FreeAgentics - where artificial minds come to life."

---

## Technical Notes for Video Production

### Screen Recording Setup
- Use 1920x1080 resolution
- Record terminal in dark theme for better visibility
- Browser should be in fullscreen mode for demo sections
- Ensure text is large enough to read at 720p

### Audio Requirements
- Clear, professional narration
- Background music: subtle, modern, tech-focused
- No distracting sound effects
- Good quality microphone essential

### Timing Notes
- Pause for visual impact after major moments
- Allow time for viewers to read error messages (if any)
- Speed can be increased during repetitive setup portions
- Keep the energy high throughout

### Backup Plans
- Have pre-recorded segments ready in case of live demo failures
- Static screenshots for any problematic sections
- Pre-populated data if agent creation is slow

### Post-Production
- Add text overlays for key technical terms
- Highlight important UI elements with subtle animations
- Include captions for accessibility
- Compress for web delivery without quality loss

This script creates a compelling 5-minute journey from "git clone" to a sophisticated AI system running in real-time, perfect for convincing stakeholders of the project's viability and potential.