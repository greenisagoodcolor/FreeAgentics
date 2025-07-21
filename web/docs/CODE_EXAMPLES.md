# FreeAgentics Frontend Code Examples

This document provides comprehensive code examples for using the FreeAgentics frontend components and APIs.

## Table of Contents

1. [API Client Usage](#api-client-usage)
2. [Agent Chat Components](#agent-chat-components)
3. [Knowledge Graph Visualization](#knowledge-graph-visualization)
4. [WebSocket Integration](#websocket-integration)
5. [Error Handling](#error-handling)
6. [Memory Viewer Components](#memory-viewer-components)

## API Client Usage

### Basic API Client Setup

```typescript
import { ApiClient, getApiClient } from '@/lib/api-client';

// Get the singleton API client
const apiClient = getApiClient();

// Or create a new instance
const customApiClient = new ApiClient('https://your-api-endpoint.com');
```

### Creating and Managing Agents

```typescript
import { AgentConfig } from '@/lib/api-client';

// Create a new agent
const agentConfig: AgentConfig = {
  name: 'Assistant Bot',
  template: 'conversational',
  parameters: {
    temperature: 0.7,
    maxTokens: 150
  },
  use_pymdp: true,
  planning_horizon: 3
};

try {
  const newAgent = await apiClient.createAgent(agentConfig);
  console.log('Agent created:', newAgent);
} catch (error) {
  console.error('Failed to create agent:', error);
}

// List all agents
const agents = await apiClient.getAgents();
console.log('Available agents:', agents);

// Get specific agent
const agent = await apiClient.getAgent('agent-id');

// Update agent
const updatedAgent = await apiClient.updateAgent('agent-id', {
  name: 'Updated Assistant Bot'
});

// Delete agent
await apiClient.deleteAgent('agent-id');
```

### Health Checks and Monitoring

```typescript
// Check API health
const healthStatus = await apiClient.getHealth();
console.log('API Status:', healthStatus);

// Get agent metrics
const metrics = await apiClient.getAgentMetrics('agent-id');
console.log('Agent performance:', metrics);
```

## Agent Chat Components

### Basic AgentChat Component

```tsx
import React from 'react';
import { AgentChat } from '@/components/conversation/AgentChat';
import { Agent } from '@/lib/types';

const ChatExample: React.FC = () => {
  const agent: Agent = {
    id: 'agent-1',
    name: 'Assistant',
    template: 'conversational',
    status: 'active',
    pymdp_config: {},
    beliefs: {},
    preferences: {},
    metrics: {},
    parameters: {},
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    inference_count: 0,
  };

  const handleMessageSent = (message: string) => {
    console.log('Message sent:', message);
  };

  const handleAgentResponse = (response: string) => {
    console.log('Agent response:', response);
  };

  return (
    <div className="chat-container">
      <AgentChat
        agent={agent}
        initialMessages={[
          {
            id: '1',
            content: 'Hello! How can I help you today?',
            sender: 'agent',
            timestamp: new Date(),
          }
        ]}
        onMessageSent={handleMessageSent}
        onAgentResponse={handleAgentResponse}
        enableDirectChat={true}
        showTypingIndicator={true}
      />
    </div>
  );
};

export default ChatExample;
```

### Conversation Panel with Multiple Agents

```tsx
import React, { useState } from 'react';
import { ConversationPanel } from '@/components/conversation/ConversationPanel';
import { Agent } from '@/lib/types';

const MultiAgentChat: React.FC = () => {
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);

  const agents: Agent[] = [
    {
      id: 'agent-1',
      name: 'Research Assistant',
      template: 'research',
      status: 'active',
      // ... other properties
    },
    {
      id: 'agent-2',
      name: 'Creative Writer',
      template: 'creative',
      status: 'active',
      // ... other properties
    }
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div className="lg:col-span-1">
        <h3 className="text-lg font-semibold mb-4">Available Agents</h3>
        {agents.map(agent => (
          <button
            key={agent.id}
            onClick={() => setSelectedAgent(agent)}
            className={`w-full p-3 mb-2 rounded-lg text-left transition-colors ${
              selectedAgent?.id === agent.id
                ? 'bg-blue-100 border-blue-300'
                : 'bg-gray-50 hover:bg-gray-100'
            }`}
          >
            <div className="font-medium">{agent.name}</div>
            <div className="text-sm text-gray-600">{agent.template}</div>
          </button>
        ))}
      </div>
      
      <div className="lg:col-span-2">
        {selectedAgent ? (
          <ConversationPanel
            agents={[selectedAgent]}
            onAgentSelect={setSelectedAgent}
            enableSearch={true}
            enableExport={true}
          />
        ) : (
          <div className="flex items-center justify-center h-64 text-gray-500">
            Select an agent to start chatting
          </div>
        )}
      </div>
    </div>
  );
};

export default MultiAgentChat;
```

## Knowledge Graph Visualization

### Basic Graph Rendering

```tsx
import React, { useEffect, useRef } from 'react';
import { GraphRenderingManager, createGraphRenderer } from '@/lib/graph-rendering';
import { GraphData } from '@/lib/graph-rendering/types';

const KnowledgeGraphView: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<GraphRenderingManager | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Create graph renderer
    rendererRef.current = createGraphRenderer(containerRef.current, {
      renderingConfig: {
        width: 800,
        height: 600,
        theme: 'light',
        enablePan: true,
        enableZoom: true,
        showFPS: true
      }
    });

    // Sample graph data
    const graphData: GraphData = {
      nodes: [
        {
          id: '1',
          label: 'Agent A',
          type: 'agent',
          x: 100,
          y: 100,
          radius: 20,
          color: '#3b82f6'
        },
        {
          id: '2',
          label: 'Knowledge Base',
          type: 'knowledge',
          x: 200,
          y: 150,
          radius: 15,
          color: '#10b981'
        }
      ],
      edges: [
        {
          id: 'edge-1',
          source: '1',
          target: '2',
          type: 'accesses',
          width: 2,
          color: '#6b7280'
        }
      ]
    };

    // Load data and start simulation
    rendererRef.current.loadData(graphData);
    rendererRef.current.startSimulation();

    // Cleanup
    return () => {
      if (rendererRef.current) {
        rendererRef.current.destroy();
      }
    };
  }, []);

  const handleNodeClick = (nodeId: string) => {
    console.log('Node clicked:', nodeId);
  };

  useEffect(() => {
    if (rendererRef.current) {
      rendererRef.current.on('nodeClick', handleNodeClick);
    }
  }, []);

  return (
    <div className="knowledge-graph-container">
      <div className="graph-controls mb-4">
        <button
          onClick={() => rendererRef.current?.fitToViewport()}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Fit to View
        </button>
        <button
          onClick={() => rendererRef.current?.centerGraph()}
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 ml-2"
        >
          Center Graph
        </button>
      </div>
      <div
        ref={containerRef}
        className="graph-container border border-gray-300 rounded-lg"
        style={{ width: '100%', height: '600px' }}
      />
    </div>
  );
};

export default KnowledgeGraphView;
```

### D3 Force Layout Integration

```typescript
import { D3ForceLayoutEngine } from '@/lib/knowledge-graph/layout/d3-force-layout';
import { RenderableGraphData, RenderingConfig } from '@/lib/knowledge-graph/types';

// Create D3 force layout engine
const layoutEngine = new D3ForceLayoutEngine();

const config: RenderingConfig = {
  forces: {
    charge: -300,
    link: 1,
    center: 0.1,
    collide: 0.7
  },
  width: 800,
  height: 600
};

// Compute layout
const layoutData = await layoutEngine.compute(graphData, config);

// Add custom forces
layoutEngine.addForce('radial', d3.forceRadial(100, 400, 300));

// Fix specific nodes
layoutEngine.fixNode('node-1', { x: 400, y: 300 });

// Control simulation
layoutEngine.setAlpha(0.5); // Reheat simulation
layoutEngine.restart();
```

## WebSocket Integration

### Basic WebSocket Client Usage

```typescript
import { getWebSocketClient } from '@/lib/websocket-client';

const wsClient = getWebSocketClient();

// Connect to WebSocket
await wsClient.connect();

// Subscribe to events
const unsubscribe = wsClient.subscribe('agent_chat_message', (message) => {
  console.log('New chat message:', message);
});

// Send message
wsClient.send({
  type: 'chat_message',
  data: {
    agent_id: 'agent-1',
    message: 'Hello from client!'
  }
});

// Check connection status
const status = wsClient.getConnectionState();
console.log('Connection status:', status);

// Cleanup
unsubscribe();
wsClient.disconnect();
```

### React Hook for WebSocket

```tsx
import React, { useEffect, useState } from 'react';
import { getWebSocketClient } from '@/lib/websocket-client';

const useWebSocket = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<any[]>([]);

  useEffect(() => {
    const wsClient = getWebSocketClient();

    const connect = async () => {
      try {
        await wsClient.connect();
        setIsConnected(true);

        // Subscribe to messages
        const unsubscribe = wsClient.subscribe('agent_chat_message', (message) => {
          setMessages(prev => [...prev, message]);
        });

        return unsubscribe;
      } catch (error) {
        console.error('WebSocket connection failed:', error);
        setIsConnected(false);
      }
    };

    const unsubscribe = connect();

    return () => {
      if (unsubscribe) {
        unsubscribe.then(unsub => unsub?.());
      }
      wsClient.disconnect();
      setIsConnected(false);
    };
  }, []);

  const sendMessage = (message: any) => {
    const wsClient = getWebSocketClient();
    wsClient.send(message);
  };

  return { isConnected, messages, sendMessage };
};

// Usage in component
const ChatComponent: React.FC = () => {
  const { isConnected, messages, sendMessage } = useWebSocket();

  const handleSendMessage = () => {
    sendMessage({
      type: 'chat_message',
      data: { message: 'Hello!' }
    });
  };

  return (
    <div>
      <div>Status: {isConnected ? 'Connected' : 'Disconnected'}</div>
      <div>Messages: {messages.length}</div>
      <button onClick={handleSendMessage}>Send Message</button>
    </div>
  );
};
```

## Error Handling

### Enhanced Error Boundary

```tsx
import React from 'react';
import { EnhancedErrorBoundary } from '@/components/EnhancedErrorBoundary';

const App: React.FC = () => {
  const handleError = (error: Error, errorInfo: React.ErrorInfo) => {
    console.error('Application error:', error, errorInfo);
    // Send to error tracking service
  };

  const handleRetry = () => {
    window.location.reload();
  };

  return (
    <EnhancedErrorBoundary
      onError={handleError}
      fallback={({ error, retry, reset }) => (
        <div className="error-container p-8 text-center">
          <h2 className="text-2xl font-bold text-red-600 mb-4">
            Something went wrong
          </h2>
          <p className="text-gray-600 mb-4">
            {error?.message || 'An unexpected error occurred'}
          </p>
          <div className="space-x-4">
            <button
              onClick={retry}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Try Again
            </button>
            <button
              onClick={reset}
              className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
            >
              Reset
            </button>
          </div>
        </div>
      )}
    >
      <YourMainAppComponent />
    </EnhancedErrorBoundary>
  );
};
```

### API Error Handling

```typescript
import { errorHandler } from '@/lib/error-handling';

const makeApiCall = async () => {
  try {
    const response = await apiClient.getAgents();
    return response;
  } catch (error) {
    // Handle with error handler
    errorHandler.handleApiError(error);
    
    // Or handle specific error types
    if (error.status === 401) {
      // Redirect to login
      window.location.href = '/login';
    } else if (error.status === 500) {
      // Show user-friendly error
      toast.error('Server error. Please try again later.');
    }
    
    throw error;
  }
};
```

## Memory Viewer Components

### Biography Viewer

```tsx
import React from 'react';
import { BiographyView } from '@/components/memory-viewer/BiographyView';

const AgentBiographyExample: React.FC = () => {
  const biographyData = {
    agent_id: 'agent-1',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-15T12:30:00Z',
    total_interactions: 150,
    key_experiences: [
      {
        id: '1',
        title: 'First successful task completion',
        description: 'Successfully helped user with complex query',
        timestamp: '2024-01-05T10:00:00Z',
        importance: 'high'
      }
    ],
    personality_traits: {
      helpfulness: 0.9,
      curiosity: 0.8,
      patience: 0.85
    },
    learned_preferences: {
      communication_style: 'detailed',
      response_length: 'medium',
      tone: 'friendly'
    }
  };

  return (
    <div className="biography-container max-w-4xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6">Agent Biography</h2>
      <BiographyView
        data={biographyData}
        onUpdate={(updatedData) => {
          console.log('Biography updated:', updatedData);
        }}
        enableEdit={true}
        showMetrics={true}
      />
    </div>
  );
};

export default AgentBiographyExample;
```

### Knowledge Graph Integration

```tsx
import React, { useState } from 'react';
import { KnowledgeView } from '@/components/memory-viewer/KnowledgeView';
import { GraphView } from '@/components/memory-viewer/GraphView';

const KnowledgeMemoryExample: React.FC = () => {
  const [viewMode, setViewMode] = useState<'list' | 'graph'>('list');

  const knowledgeData = {
    concepts: [
      {
        id: '1',
        name: 'Machine Learning',
        type: 'domain',
        confidence: 0.95,
        connections: ['2', '3']
      },
      {
        id: '2',
        name: 'Neural Networks',
        type: 'technique',
        confidence: 0.88,
        connections: ['1', '3']
      }
    ],
    relationships: [
      {
        id: 'rel-1',
        source: '1',
        target: '2',
        type: 'includes',
        strength: 0.9
      }
    ]
  };

  return (
    <div className="knowledge-memory-container">
      <div className="controls mb-4">
        <button
          onClick={() => setViewMode('list')}
          className={`px-4 py-2 rounded-l ${
            viewMode === 'list' ? 'bg-blue-500 text-white' : 'bg-gray-200'
          }`}
        >
          List View
        </button>
        <button
          onClick={() => setViewMode('graph')}
          className={`px-4 py-2 rounded-r ${
            viewMode === 'graph' ? 'bg-blue-500 text-white' : 'bg-gray-200'
          }`}
        >
          Graph View
        </button>
      </div>

      {viewMode === 'list' ? (
        <KnowledgeView
          data={knowledgeData}
          onConceptSelect={(concept) => {
            console.log('Concept selected:', concept);
          }}
          enableFiltering={true}
          showConfidence={true}
        />
      ) : (
        <GraphView
          data={knowledgeData}
          onNodeClick={(nodeId) => {
            console.log('Node clicked:', nodeId);
          }}
          layout="force"
          interactive={true}
        />
      )}
    </div>
  );
};

export default KnowledgeMemoryExample;
```

## Best Practices

### 1. Error Boundary Placement

```tsx
// Wrap route components
<EnhancedErrorBoundary>
  <AgentDashboard />
</EnhancedErrorBoundary>

// Wrap data-fetching components
<EnhancedErrorBoundary>
  <AgentList />
</EnhancedErrorBoundary>
```

### 2. Loading States

```tsx
import { Skeleton } from '@/components/Skeleton';
import { LoadingState } from '@/components/LoadingState';

const AgentComponent: React.FC = () => {
  const [loading, setLoading] = useState(true);

  if (loading) {
    return <LoadingState message="Loading agents..." />;
  }

  return <div>Agent content</div>;
};
```

### 3. Responsive Design

```tsx
const ResponsiveLayout: React.FC = () => {
  return (
    <div className="container mx-auto px-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Content */}
      </div>
    </div>
  );
};
```

### 4. Performance Optimization

```tsx
import { memo, useCallback, useMemo } from 'react';

const OptimizedComponent = memo(({ data, onUpdate }) => {
  const processedData = useMemo(() => {
    return data.map(item => processItem(item));
  }, [data]);

  const handleUpdate = useCallback((id: string) => {
    onUpdate(id);
  }, [onUpdate]);

  return <div>{/* Component content */}</div>;
});
```

This documentation provides practical examples for using all major frontend components and APIs in the FreeAgentics platform. Each example includes proper TypeScript types, error handling, and follows React best practices.