# FreeAgentics Web Interface

## Overview

The FreeAgentics web interface provides a real-time dashboard for interacting with Active Inference agents. Built with Next.js and TypeScript, it offers a modern, responsive UI for agent creation, monitoring, and knowledge graph visualization.

## Key Features

- **Prompt-Based Agent Creation**: Natural language interface for creating agents
- **Real-Time Updates**: WebSocket integration for live agent status and metrics
- **Knowledge Graph Visualization**: Interactive 3D visualization of agent knowledge
- **Agent Monitoring**: Live dashboards showing agent beliefs, actions, and performance
- **Conversation History**: Track iterative refinements and agent evolution

## Architecture

### Frontend Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS with custom design tokens
- **State Management**: React hooks with WebSocket integration
- **Visualization**: Custom WebGL-based graph rendering

### Core Components

```
web/
├── app/                      # Next.js app directory
│   ├── page.tsx             # Main prompt interface
│   ├── agents/              # Agent management pages
│   └── dashboard/           # System dashboard
├── components/              # Reusable components
│   ├── prompt-interface.tsx # Main prompt UI
│   ├── agent-visualization.tsx # Agent state viewer
│   ├── knowledge-graph-view.tsx # KG visualization
│   └── suggestions-list.tsx # AI suggestions
├── hooks/                   # Custom React hooks
│   ├── use-prompt-processor.ts # Prompt pipeline hook
│   └── use-agent-conversation.ts # Conversation state
└── lib/                     # Utilities and clients
    ├── api-client.ts       # REST API integration
    └── websocket-client.ts # WebSocket connection
```

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- Running FreeAgentics backend (port 8000)

### Installation

```bash
# Install dependencies
npm install

# Copy environment template
cp .env.example .env.local

# Configure API endpoint in .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

### Development

```bash
# Start development server
npm run dev

# Open http://localhost:3000
```

### Production Build

```bash
# Build for production
npm run build

# Start production server
npm start
```

## Core Functionality

### 1. Prompt Interface

The main interface allows users to:

- Enter natural language prompts to create agents
- View real-time pipeline progress
- See generated GMN specifications
- Receive intelligent suggestions for refinements

Example workflow:

```typescript
// User enters prompt
"Create an explorer agent for a 5x5 grid world"

// System processes through pipeline:
1. GMN Generation (LLM converts to formal spec)
2. Validation (Ensures GMN is valid)
3. Agent Creation (PyMDP agent instantiated)
4. KG Update (Agent beliefs added to graph)
5. Suggestions (Next actions recommended)
```

### 2. Real-Time Updates

WebSocket integration provides:

- Pipeline progress notifications
- Agent state changes
- Knowledge graph updates
- System events and alerts

### 3. Agent Visualization

Interactive views showing:

- Current belief states
- Action probabilities
- Free energy landscapes
- Historical trajectories

### 4. Knowledge Graph Explorer

3D visualization features:

- Node clustering by type
- Relationship strength visualization
- Interactive navigation
- Real-time updates as agents learn

## API Integration

### REST Endpoints

The frontend integrates with these key endpoints:

```typescript
// Submit prompt for processing
POST /api/v1/prompts
{
  prompt: string,
  conversation_id?: string,
  iteration_count?: number
}

// Get agent details
GET /api/v1/agents/{agent_id}

// Get knowledge graph data
GET /api/v1/knowledge/search
```

### WebSocket Events

Real-time updates via WebSocket:

```typescript
// Subscribe to events
ws.send({
  type: "subscribe",
  event_types: ["pipeline:*", "agent:*", "knowledge_graph:*"],
});

// Receive updates
ws.on("message", (event) => {
  switch (event.type) {
    case "pipeline_progress":
      updatePipelineUI(event.data);
      break;
    case "agent_created":
      showNewAgent(event.data);
      break;
    case "knowledge_graph_updated":
      refreshVisualization(event.data);
      break;
  }
});
```

## Environment Variables

| Variable              | Description      | Default                 |
| --------------------- | ---------------- | ----------------------- |
| `NEXT_PUBLIC_API_URL` | Backend API URL  | `http://localhost:8000` |
| `NEXT_PUBLIC_WS_URL`  | WebSocket URL    | `ws://localhost:8000`   |
| `NEXT_PUBLIC_ENV`     | Environment name | `development`           |

## Performance Optimization

- **Code Splitting**: Automatic route-based splitting
- **Image Optimization**: Next.js Image component
- **Lazy Loading**: Components loaded on demand
- **WebSocket Pooling**: Efficient connection management
- **Memoization**: React.memo for expensive renders

## Security Features

- **CSRF Protection**: Built into Next.js
- **XSS Prevention**: Automatic escaping
- **CSP Headers**: Content Security Policy
- **Authentication**: JWT token management
- **HTTPS**: Enforced in production

## Testing

```bash
# Run unit tests
npm test

# Run integration tests
npm run test:integration

# Run E2E tests
npm run test:e2e

# Generate coverage report
npm run test:coverage
```

## Deployment

### Docker

```bash
# Build Docker image
docker build -t freeagentics-web .

# Run container
docker run -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=https://api.freeagentics.com \
  freeagentics-web
```

### Kubernetes

See `/k8s/frontend-deployment.yaml` for Kubernetes configuration.

## Troubleshooting

### Common Issues

**White screen on load**

- Check browser console for errors
- Verify API URL configuration
- Ensure backend is running

**WebSocket connection failed**

- Check WebSocket URL configuration
- Verify authentication token
- Check browser WebSocket support

**Slow performance**

- Enable production mode
- Check network latency
- Monitor bundle size

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
