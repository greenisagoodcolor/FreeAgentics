# FreeAgentics Demo Environment

A fully-featured demonstration environment showcasing the FreeAgentics Agent Simulator platform. This environment runs in Docker and provides an accelerated, pre-populated simulation perfect for presentations, testing, and exploration.

## 🚀 Quick Start

```bash
# Start the demo environment
./scripts/demo/start-demo.sh

# Access the demo at
http://localhost:8080
```

## 📋 Features

### Pre-Populated Demo Data

- **4 Demo Agents**: Explorer, Merchant, Scholar, and Guardian classes
- **Varied Readiness Levels**: From training to deployment-ready
- **Rich History**: Pre-seeded interactions, discoveries, and achievements

### Automated Scenarios

1. **Explorer Discovery**: Watch agents autonomously explore and discover resources
2. **Merchant Trading**: See dynamic negotiations and market analysis
3. **Scholar Research**: Observe knowledge creation and theory formulation
4. **Guardian Patrol**: Experience security protocols and threat response
5. **Multi-Agent Collaboration**: Marvel at complex teamwork and goal achievement

### Enhanced Visualization

- **10x Simulation Speed**: See days of agent activity in minutes
- **Real-Time Updates**: WebSocket-powered live data streaming
- **Visual Highlights**: Color-coded events and achievements
- **Celebration Animations**: Special effects for major milestones

## 🏗️ Architecture

```
demo-nginx (8080)
    ├── demo-web (3030) - Next.js application in demo mode
    ├── demo-monitor (3031) - Real-time monitoring dashboard
    │
    ├── demo-simulator - Accelerated agent simulation engine
    ├── demo-scenarios - Automated scenario runner
    │
    ├── demo-db (5433) - PostgreSQL with demo data
    └── demo-redis (6380) - Redis for real-time events
```

## 🛠️ Services

### demo-web

The main FreeAgentics application running in demo mode with:

- Enhanced UI animations
- Demo-specific features enabled
- Pre-configured for optimal presentation

### demo-simulator

Python service that:

- Runs agents at 10x normal speed
- Simulates realistic agent behaviors
- Generates continuous activity

### demo-scenarios

Orchestrates compelling demonstration scenarios:

- Runs every 5 minutes (configurable)
- Showcases different agent capabilities
- Creates engaging narratives

### demo-monitor

Dedicated monitoring dashboard showing:

- Real-time agent activities
- Scenario progress
- System metrics
- Event timeline

## 📊 Demo Scenarios

### Explorer Discovery (3 minutes)

Demonstrates autonomous exploration and resource discovery:

- Dynamic pathfinding
- Resource identification
- Knowledge sharing

### Merchant Trade (4 minutes)

Shows economic simulation capabilities:

- Market analysis
- Price negotiation
- Trade execution
- Profit tracking

### Scholar Research (5 minutes)

Highlights knowledge creation:

- Data collection
- Pattern analysis
- Theory formulation
- Knowledge dissemination

### Guardian Patrol (4 minutes)

Displays security and protection behaviors:

- Territory establishment
- Threat detection
- Response coordination
- Incident reporting

### Multi-Agent Collaboration (6 minutes)

The grand finale showing teamwork:

- Team formation
- Role assignment
- Coordinated execution
- Shared success

## 🎮 Usage

### Starting the Demo

```bash
./scripts/demo/start-demo.sh
```

### Monitoring

```bash
# View all logs
docker-compose -f docker/demo/docker-compose.yml logs -f

# View specific service
docker-compose -f docker/demo/docker-compose.yml logs -f demo-simulator
```

### Stopping

```bash
./scripts/demo/stop-demo.sh
```

### Resetting

```bash
# Complete reset (deletes all data)
./scripts/demo/reset-demo.sh
```

## 🔧 Configuration

### Environment Variables

Edit `docker/demo/docker-compose.yml` to adjust:

- `SIMULATION_SPEED`: How fast agents act (default: 10x)
- `SCENARIO_INTERVAL`: Time between scenarios (default: 300s)
- `AUTO_PLAY`: Auto-start scenarios (default: true)

### Demo Data

- Initial agents: `docker/demo/seed-demo-data.sql`
- Scenarios: `docker/demo/scenario_runner.py`
- UI config: `docker/demo/demo-config.json`

## 📈 Customization

### Adding New Scenarios

1. Add scenario to `seed-demo-data.sql`
2. Implement handler in `scenario_runner.py`
3. Update `demo-config.json`

### Modifying Agents

1. Edit agent data in `seed-demo-data.sql`
2. Adjust stats for desired readiness levels
3. Rebuild with `docker-compose build`

### UI Enhancements

1. Edit `demo-config.json` for colors/animations
2. Modify notification settings
3. Adjust highlight thresholds

## 🐛 Troubleshooting

### Services Not Starting

```bash
# Check service status
docker-compose -f docker/demo/docker-compose.yml ps

# View detailed logs
docker-compose -f docker/demo/docker-compose.yml logs [service-name]
```

### Database Connection Issues

```bash
# Check database is running
docker exec -it freeagentics-demo-db psql -U demo -d freeagentics_demo

# Verify data is loaded
SELECT COUNT(*) FROM agents.agents;
```

### Performance Issues

- Reduce `SIMULATION_SPEED` to lower values
- Increase `SCENARIO_INTERVAL` for fewer scenarios
- Check Docker resource allocation

## 🎯 Demo Script Suggestions

### For Investors (15 minutes)

1. Start with Character Creator - show personality sliders
2. Run Explorer Discovery - highlight autonomous behavior
3. Show Knowledge Graph - demonstrate learning
4. Run Multi-Agent Collaboration - showcase emergence
5. Display Readiness Panel - show deployment readiness
6. Export agent package - demonstrate hardware deployment

### For Technical Audience (30 minutes)

1. Deep dive into GNN architecture
2. Show real-time model updates
3. Demonstrate Active Inference in action
4. Explore knowledge graph relationships
5. Review readiness evaluation metrics
6. Discuss hardware optimization

### For General Audience (10 minutes)

1. Quick agent creation
2. Watch automated scenarios
3. Show agent conversations
4. Highlight achievements
5. Celebrate readiness milestones

## 🚀 Advanced Features

### WebSocket Events

Connect to WebSocket for real-time updates:

```javascript
const ws = new WebSocket("ws://localhost:8080/ws");
ws.on("message", (data) => {
  const event = JSON.parse(data);
  console.log("Event:", event);
});
```

### Redis Pub/Sub

Monitor events directly:

```bash
docker exec -it freeagentics-demo-redis redis-cli
> SUBSCRIBE demo:events:all
```

### Direct Database Access

```bash
psql postgresql://demo:demo123@localhost:5433/freeagentics_demo
```

## 📝 Notes

- Demo data resets on container restart (unless using volumes)
- Scenarios loop continuously when `SCENARIO_LOOP=true`
- Agent improvements are accelerated for demonstration
- Some random events added for variety
- WebSocket connections required for real-time features

## 🤝 Contributing

To improve the demo environment:

1. Test changes locally first
2. Document new features
3. Update demo scripts if needed
4. Ensure backward compatibility

---

Happy Demonstrating! 🎉
