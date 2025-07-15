# Code Examples

This document provides comprehensive code examples for integrating with the FreeAgentics API in multiple programming languages.

## Table of Contents

1. [Python Examples](#python-examples)
2. [JavaScript/TypeScript Examples](#javascripttypescript-examples)
3. [Go Examples](#go-examples)
4. [Java Examples](#java-examples)
5. [C# Examples](#c-examples)
6. [PHP Examples](#php-examples)
7. [Ruby Examples](#ruby-examples)
8. [Swift Examples](#swift-examples)
9. [cURL Examples](#curl-examples)

## Python Examples

### Basic Authentication and Agent Creation

```python
import requests
import json
from datetime import datetime, timedelta
import time

class FreeAgenticsClient:
    def __init__(self, base_url="https://api.freeagentics.com"):
        self.base_url = base_url
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None
        
    def login(self, username, password):
        """Login and obtain tokens"""
        response = requests.post(
            f"{self.base_url}/api/v1/login",
            json={"username": username, "password": password}
        )
        
        if response.status_code == 200:
            data = response.json()
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            self.token_expiry = datetime.now() + timedelta(minutes=14)
            return True
        else:
            raise Exception(f"Login failed: {response.json()}")
    
    def _get_headers(self):
        """Get authenticated headers"""
        if not self.access_token or datetime.now() >= self.token_expiry:
            self._refresh_token()
        
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    def _refresh_token(self):
        """Refresh access token"""
        if not self.refresh_token:
            raise Exception("No refresh token available")
        
        response = requests.post(
            f"{self.base_url}/api/v1/refresh",
            json={"refresh_token": self.refresh_token}
        )
        
        if response.status_code == 200:
            data = response.json()
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            self.token_expiry = datetime.now() + timedelta(minutes=14)
        else:
            raise Exception(f"Token refresh failed: {response.json()}")
    
    def create_agent(self, name, template, parameters=None):
        """Create a new agent"""
        data = {
            "name": name,
            "template": template,
            "parameters": parameters or {},
            "use_pymdp": True,
            "planning_horizon": 3
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/agents",
            headers=self._get_headers(),
            json=data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Agent creation failed: {response.json()}")
    
    def run_inference(self, agent_id, query, context=None, parameters=None):
        """Run inference with an agent"""
        data = {
            "agent_id": agent_id,
            "query": query,
            "context": context or {},
            "parameters": parameters or {}
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/inference",
            headers=self._get_headers(),
            json=data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Inference failed: {response.json()}")
    
    def get_inference_result(self, inference_id):
        """Get inference result"""
        response = requests.get(
            f"{self.base_url}/api/v1/inference/{inference_id}",
            headers=self._get_headers()
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get inference result: {response.json()}")
    
    def wait_for_inference(self, inference_id, timeout=300):
        """Wait for inference to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = self.get_inference_result(inference_id)
            
            if result["status"] == "completed":
                return result
            elif result["status"] == "failed":
                raise Exception(f"Inference failed: {result.get('error', 'Unknown error')}")
            
            time.sleep(1)
        
        raise Exception(f"Inference timed out after {timeout} seconds")

# Usage example
if __name__ == "__main__":
    client = FreeAgenticsClient()
    
    # Login
    client.login("your_username", "your_password")
    
    # Create agent
    agent = client.create_agent(
        name="Research Assistant",
        template="research_v2",
        parameters={
            "temperature": 0.7,
            "max_tokens": 2048
        }
    )
    
    print(f"Created agent: {agent['id']}")
    
    # Run inference
    inference = client.run_inference(
        agent_id=agent["id"],
        query="What are the latest trends in AI?",
        context={"focus": "machine learning", "year": "2024"}
    )
    
    print(f"Started inference: {inference['inference_id']}")
    
    # Wait for result
    result = client.wait_for_inference(inference["inference_id"])
    print(f"Analysis: {result['result']['analysis']}")
```

### WebSocket Client Implementation

```python
import asyncio
import websockets
import json
from typing import Dict, List, Callable, Optional

class WebSocketClient:
    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token
        self.websocket = None
        self.subscriptions = set()
        self.message_handlers = {}
        self.running = False
        
    async def connect(self):
        """Connect to WebSocket"""
        uri = f"{self.url}?token={self.token}"
        
        try:
            self.websocket = await websockets.connect(uri)
            self.running = True
            
            # Start message handling loop
            await self._handle_messages()
            
        except Exception as e:
            print(f"Connection failed: {e}")
            raise
    
    async def _handle_messages(self):
        """Handle incoming messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self._process_message(data)
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
        except Exception as e:
            print(f"Message handling error: {e}")
    
    async def _process_message(self, message: Dict):
        """Process received message"""
        message_type = message.get("type")
        
        if message_type in self.message_handlers:
            handler = self.message_handlers[message_type]
            await handler(message.get("data", {}))
        else:
            print(f"Unhandled message type: {message_type}")
    
    def on_message(self, message_type: str, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type] = handler
    
    async def subscribe(self, events: List[str], agents: Optional[List[str]] = None):
        """Subscribe to events"""
        message = {
            "type": "subscribe",
            "data": {
                "events": events,
                "agents": agents or []
            }
        }
        
        await self.websocket.send(json.dumps(message))
        self.subscriptions.update(events)
    
    async def send_command(self, agent_id: str, command: str, parameters: Dict = None):
        """Send command to agent"""
        message = {
            "type": "command",
            "data": {
                "agent_id": agent_id,
                "command": command,
                "parameters": parameters or {}
            }
        }
        
        await self.websocket.send(json.dumps(message))
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        self.running = False
        if self.websocket:
            await self.websocket.close()

# Usage example
async def main():
    client = WebSocketClient("wss://api.freeagentics.com/api/v1/ws", "your_token")
    
    # Register handlers
    async def on_agent_status(data):
        print(f"Agent {data['agent_id']} status: {data['new_status']}")
    
    async def on_metrics(data):
        print(f"CPU: {data['system']['cpu_usage']}%")
        print(f"Memory: {data['system']['memory_usage']}%")
    
    client.on_message("agent_status", on_agent_status)
    client.on_message("metrics_update", on_metrics)
    
    # Connect and subscribe
    await client.connect()
    await client.subscribe(["agent_status", "metrics_update"])
    
    # Keep connection alive
    try:
        await asyncio.sleep(3600)  # 1 hour
    except KeyboardInterrupt:
        print("Disconnecting...")
    finally:
        await client.disconnect()

# Run WebSocket client
# asyncio.run(main())
```

### Knowledge Graph Operations

```python
class KnowledgeGraphClient:
    def __init__(self, client: FreeAgenticsClient):
        self.client = client
    
    def search_entities(self, query: str, entity_type: str = None, limit: int = 20):
        """Search knowledge graph entities"""
        params = {"q": query, "limit": limit}
        if entity_type:
            params["type"] = entity_type
        
        response = requests.get(
            f"{self.client.base_url}/api/v1/knowledge/search",
            headers=self.client._get_headers(),
            params=params
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Search failed: {response.json()}")
    
    def create_entity(self, entity_type: str, label: str, properties: Dict):
        """Create new entity"""
        data = {
            "type": entity_type,
            "label": label,
            "properties": properties
        }
        
        response = requests.post(
            f"{self.client.base_url}/api/v1/knowledge/entities",
            headers=self.client._get_headers(),
            json=data
        )
        
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Entity creation failed: {response.json()}")
    
    def create_relationship(self, source_id: str, target_id: str, 
                          relationship_type: str, properties: Dict = None):
        """Create relationship between entities"""
        data = {
            "source_id": source_id,
            "target_id": target_id,
            "relationship_type": relationship_type,
            "properties": properties or {}
        }
        
        response = requests.post(
            f"{self.client.base_url}/api/v1/knowledge/relationships",
            headers=self.client._get_headers(),
            json=data
        )
        
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Relationship creation failed: {response.json()}")
    
    def build_knowledge_graph(self, entities_data: List[Dict], relationships_data: List[Dict]):
        """Build knowledge graph from structured data"""
        created_entities = {}
        
        # Create entities
        for entity_data in entities_data:
            entity = self.create_entity(
                entity_type=entity_data["type"],
                label=entity_data["label"],
                properties=entity_data.get("properties", {})
            )
            created_entities[entity_data["id"]] = entity["id"]
        
        # Create relationships
        for rel_data in relationships_data:
            source_id = created_entities[rel_data["source"]]
            target_id = created_entities[rel_data["target"]]
            
            self.create_relationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=rel_data["type"],
                properties=rel_data.get("properties", {})
            )
        
        return created_entities

# Usage example
kg_client = KnowledgeGraphClient(client)

# Search for AI-related entities
results = kg_client.search_entities("artificial intelligence", limit=10)
print(f"Found {len(results['results'])} entities")

# Create new entity
entity = kg_client.create_entity(
    entity_type="technology",
    label="Large Language Models",
    properties={
        "description": "AI models trained on vast amounts of text data",
        "category": "AI/ML",
        "importance": "high"
    }
)
```

## JavaScript/TypeScript Examples

### Modern Fetch API Client

```typescript
interface AgentConfig {
  name: string;
  template: string;
  parameters?: Record<string, any>;
  gmn_spec?: string;
  use_pymdp?: boolean;
  planning_horizon?: number;
}

interface InferenceRequest {
  agent_id: string;
  query: string;
  context?: Record<string, any>;
  parameters?: Record<string, any>;
}

class FreeAgenticsClient {
  private baseUrl: string;
  private accessToken: string | null = null;
  private refreshToken: string | null = null;
  private tokenExpiry: Date | null = null;

  constructor(baseUrl: string = 'https://api.freeagentics.com') {
    this.baseUrl = baseUrl;
  }

  async login(username: string, password: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/v1/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
    });

    if (!response.ok) {
      throw new Error(`Login failed: ${await response.text()}`);
    }

    const data = await response.json();
    this.accessToken = data.access_token;
    this.refreshToken = data.refresh_token;
    this.tokenExpiry = new Date(Date.now() + 14 * 60 * 1000); // 14 minutes
  }

  private async getHeaders(): Promise<Record<string, string>> {
    if (!this.accessToken || (this.tokenExpiry && new Date() >= this.tokenExpiry)) {
      await this.refreshTokens();
    }

    return {
      'Authorization': `Bearer ${this.accessToken}`,
      'Content-Type': 'application/json',
    };
  }

  private async refreshTokens(): Promise<void> {
    if (!this.refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await fetch(`${this.baseUrl}/api/v1/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ refresh_token: this.refreshToken }),
    });

    if (!response.ok) {
      throw new Error(`Token refresh failed: ${await response.text()}`);
    }

    const data = await response.json();
    this.accessToken = data.access_token;
    this.refreshToken = data.refresh_token;
    this.tokenExpiry = new Date(Date.now() + 14 * 60 * 1000);
  }

  async createAgent(config: AgentConfig): Promise<any> {
    const headers = await this.getHeaders();
    const response = await fetch(`${this.baseUrl}/api/v1/agents`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        ...config,
        use_pymdp: config.use_pymdp ?? true,
        planning_horizon: config.planning_horizon ?? 3,
      }),
    });

    if (!response.ok) {
      throw new Error(`Agent creation failed: ${await response.text()}`);
    }

    return response.json();
  }

  async runInference(request: InferenceRequest): Promise<any> {
    const headers = await this.getHeaders();
    const response = await fetch(`${this.baseUrl}/api/v1/inference`, {
      method: 'POST',
      headers,
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Inference failed: ${await response.text()}`);
    }

    return response.json();
  }

  async getInferenceResult(inferenceId: string): Promise<any> {
    const headers = await this.getHeaders();
    const response = await fetch(`${this.baseUrl}/api/v1/inference/${inferenceId}`, {
      method: 'GET',
      headers,
    });

    if (!response.ok) {
      throw new Error(`Failed to get inference result: ${await response.text()}`);
    }

    return response.json();
  }

  async waitForInference(inferenceId: string, timeout: number = 300000): Promise<any> {
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const result = await this.getInferenceResult(inferenceId);

      if (result.status === 'completed') {
        return result;
      } else if (result.status === 'failed') {
        throw new Error(`Inference failed: ${result.error || 'Unknown error'}`);
      }

      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    throw new Error(`Inference timed out after ${timeout}ms`);
  }

  async batchInference(requests: InferenceRequest[], parallel: boolean = true): Promise<any> {
    const headers = await this.getHeaders();
    const response = await fetch(`${this.baseUrl}/api/v1/batch-inference`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ requests, parallel }),
    });

    if (!response.ok) {
      throw new Error(`Batch inference failed: ${await response.text()}`);
    }

    return response.json();
  }
}

// Usage example
async function main() {
  const client = new FreeAgenticsClient();
  
  try {
    // Login
    await client.login('your_username', 'your_password');
    
    // Create agent
    const agent = await client.createAgent({
      name: 'TypeScript Agent',
      template: 'research_v2',
      parameters: {
        temperature: 0.7,
        max_tokens: 2048,
      },
    });
    
    console.log(`Created agent: ${agent.id}`);
    
    // Run inference
    const inference = await client.runInference({
      agent_id: agent.id,
      query: 'Analyze TypeScript adoption trends',
      context: {
        focus: 'web development',
        timeframe: '2024',
      },
    });
    
    console.log(`Started inference: ${inference.inference_id}`);
    
    // Wait for result
    const result = await client.waitForInference(inference.inference_id);
    console.log(`Analysis: ${result.result.analysis}`);
    
  } catch (error) {
    console.error('Error:', error);
  }
}

// main();
```

### React Hook for FreeAgentics

```tsx
import { useState, useEffect, useCallback } from 'react';

interface Agent {
  id: string;
  name: string;
  status: string;
  template: string;
  created_at: string;
  inference_count: number;
}

interface UseFreeAgenticsProps {
  username: string;
  password: string;
}

export const useFreeAgentics = ({ username, password }: UseFreeAgenticsProps) => {
  const [client, setClient] = useState<FreeAgenticsClient | null>(null);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Initialize client
  useEffect(() => {
    const initClient = async () => {
      try {
        const newClient = new FreeAgenticsClient();
        await newClient.login(username, password);
        setClient(newClient);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Authentication failed');
      } finally {
        setLoading(false);
      }
    };

    initClient();
  }, [username, password]);

  // Fetch agents
  const fetchAgents = useCallback(async () => {
    if (!client) return;

    try {
      const response = await fetch(`${client.baseUrl}/api/v1/agents`, {
        headers: await client.getHeaders(),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch agents');
      }

      const data = await response.json();
      setAgents(data.agents);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch agents');
    }
  }, [client]);

  // Create agent
  const createAgent = useCallback(async (config: AgentConfig) => {
    if (!client) throw new Error('Client not initialized');

    const agent = await client.createAgent(config);
    setAgents(prev => [...prev, agent]);
    return agent;
  }, [client]);

  // Run inference
  const runInference = useCallback(async (request: InferenceRequest) => {
    if (!client) throw new Error('Client not initialized');

    return client.runInference(request);
  }, [client]);

  // Load agents on client initialization
  useEffect(() => {
    if (client) {
      fetchAgents();
    }
  }, [client, fetchAgents]);

  return {
    client,
    agents,
    loading,
    error,
    createAgent,
    runInference,
    fetchAgents,
  };
};

// Usage in React component
const AgentDashboard: React.FC = () => {
  const {
    agents,
    loading,
    error,
    createAgent,
    runInference,
  } = useFreeAgentics({
    username: 'your_username',
    password: 'your_password',
  });

  const handleCreateAgent = async () => {
    try {
      await createAgent({
        name: 'New Agent',
        template: 'research_v2',
        parameters: { temperature: 0.7 },
      });
    } catch (err) {
      console.error('Failed to create agent:', err);
    }
  };

  const handleRunInference = async (agentId: string) => {
    try {
      const result = await runInference({
        agent_id: agentId,
        query: 'Analyze current market trends',
      });
      console.log('Inference started:', result);
    } catch (err) {
      console.error('Failed to run inference:', err);
    }
  };

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <h1>Agent Dashboard</h1>
      <button onClick={handleCreateAgent}>Create Agent</button>
      
      <div>
        {agents.map(agent => (
          <div key={agent.id}>
            <h3>{agent.name}</h3>
            <p>Status: {agent.status}</p>
            <p>Inferences: {agent.inference_count}</p>
            <button onClick={() => handleRunInference(agent.id)}>
              Run Inference
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};
```

## Go Examples

### Basic Client Implementation

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

type Client struct {
    BaseURL      string
    AccessToken  string
    RefreshToken string
    TokenExpiry  time.Time
    HTTPClient   *http.Client
}

type LoginRequest struct {
    Username string `json:"username"`
    Password string `json:"password"`
}

type TokenResponse struct {
    AccessToken  string `json:"access_token"`
    RefreshToken string `json:"refresh_token"`
    TokenType    string `json:"token_type"`
    User         User   `json:"user"`
}

type User struct {
    UserID      string   `json:"user_id"`
    Username    string   `json:"username"`
    Role        string   `json:"role"`
    Permissions []string `json:"permissions"`
}

type AgentConfig struct {
    Name            string                 `json:"name"`
    Template        string                 `json:"template"`
    Parameters      map[string]interface{} `json:"parameters,omitempty"`
    GMNSpec         string                 `json:"gmn_spec,omitempty"`
    UsePyMDP        bool                   `json:"use_pymdp"`
    PlanningHorizon int                    `json:"planning_horizon"`
}

type Agent struct {
    ID             string                 `json:"id"`
    Name           string                 `json:"name"`
    Template       string                 `json:"template"`
    Status         string                 `json:"status"`
    CreatedAt      time.Time              `json:"created_at"`
    LastActive     *time.Time             `json:"last_active,omitempty"`
    InferenceCount int                    `json:"inference_count"`
    Parameters     map[string]interface{} `json:"parameters"`
}

type InferenceRequest struct {
    AgentID    string                 `json:"agent_id"`
    Query      string                 `json:"query"`
    Context    map[string]interface{} `json:"context,omitempty"`
    Parameters map[string]interface{} `json:"parameters,omitempty"`
}

type InferenceResponse struct {
    InferenceID         string     `json:"inference_id"`
    AgentID             string     `json:"agent_id"`
    Status              string     `json:"status"`
    CreatedAt           time.Time  `json:"created_at"`
    EstimatedCompletion *time.Time `json:"estimated_completion,omitempty"`
}

func NewClient(baseURL string) *Client {
    return &Client{
        BaseURL:    baseURL,
        HTTPClient: &http.Client{Timeout: 30 * time.Second},
    }
}

func (c *Client) Login(username, password string) error {
    loginReq := LoginRequest{
        Username: username,
        Password: password,
    }
    
    jsonData, err := json.Marshal(loginReq)
    if err != nil {
        return err
    }
    
    resp, err := c.HTTPClient.Post(
        c.BaseURL+"/api/v1/login",
        "application/json",
        bytes.NewBuffer(jsonData),
    )
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return fmt.Errorf("login failed with status: %d", resp.StatusCode)
    }
    
    var tokenResp TokenResponse
    if err := json.NewDecoder(resp.Body).Decode(&tokenResp); err != nil {
        return err
    }
    
    c.AccessToken = tokenResp.AccessToken
    c.RefreshToken = tokenResp.RefreshToken
    c.TokenExpiry = time.Now().Add(14 * time.Minute)
    
    return nil
}

func (c *Client) refreshToken() error {
    refreshReq := map[string]string{
        "refresh_token": c.RefreshToken,
    }
    
    jsonData, err := json.Marshal(refreshReq)
    if err != nil {
        return err
    }
    
    resp, err := c.HTTPClient.Post(
        c.BaseURL+"/api/v1/refresh",
        "application/json",
        bytes.NewBuffer(jsonData),
    )
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return fmt.Errorf("token refresh failed with status: %d", resp.StatusCode)
    }
    
    var tokenResp TokenResponse
    if err := json.NewDecoder(resp.Body).Decode(&tokenResp); err != nil {
        return err
    }
    
    c.AccessToken = tokenResp.AccessToken
    c.RefreshToken = tokenResp.RefreshToken
    c.TokenExpiry = time.Now().Add(14 * time.Minute)
    
    return nil
}

func (c *Client) makeRequest(method, path string, body interface{}) (*http.Response, error) {
    // Check if token needs refresh
    if time.Now().After(c.TokenExpiry) {
        if err := c.refreshToken(); err != nil {
            return nil, err
        }
    }
    
    var reqBody *bytes.Buffer
    if body != nil {
        jsonData, err := json.Marshal(body)
        if err != nil {
            return nil, err
        }
        reqBody = bytes.NewBuffer(jsonData)
    }
    
    req, err := http.NewRequest(method, c.BaseURL+path, reqBody)
    if err != nil {
        return nil, err
    }
    
    req.Header.Set("Authorization", "Bearer "+c.AccessToken)
    req.Header.Set("Content-Type", "application/json")
    
    return c.HTTPClient.Do(req)
}

func (c *Client) CreateAgent(config AgentConfig) (*Agent, error) {
    config.UsePyMDP = true
    config.PlanningHorizon = 3
    
    resp, err := c.makeRequest("POST", "/api/v1/agents", config)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("agent creation failed with status: %d", resp.StatusCode)
    }
    
    var agent Agent
    if err := json.NewDecoder(resp.Body).Decode(&agent); err != nil {
        return nil, err
    }
    
    return &agent, nil
}

func (c *Client) RunInference(req InferenceRequest) (*InferenceResponse, error) {
    resp, err := c.makeRequest("POST", "/api/v1/inference", req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("inference failed with status: %d", resp.StatusCode)
    }
    
    var inferenceResp InferenceResponse
    if err := json.NewDecoder(resp.Body).Decode(&inferenceResp); err != nil {
        return nil, err
    }
    
    return &inferenceResp, nil
}

func (c *Client) GetInferenceResult(inferenceID string) (map[string]interface{}, error) {
    resp, err := c.makeRequest("GET", "/api/v1/inference/"+inferenceID, nil)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("failed to get inference result with status: %d", resp.StatusCode)
    }
    
    var result map[string]interface{}
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, err
    }
    
    return result, nil
}

func (c *Client) WaitForInference(inferenceID string, timeout time.Duration) (map[string]interface{}, error) {
    deadline := time.Now().Add(timeout)
    
    for time.Now().Before(deadline) {
        result, err := c.GetInferenceResult(inferenceID)
        if err != nil {
            return nil, err
        }
        
        status, ok := result["status"].(string)
        if !ok {
            return nil, fmt.Errorf("invalid response format")
        }
        
        switch status {
        case "completed":
            return result, nil
        case "failed":
            errorMsg, _ := result["error"].(string)
            return nil, fmt.Errorf("inference failed: %s", errorMsg)
        }
        
        time.Sleep(1 * time.Second)
    }
    
    return nil, fmt.Errorf("inference timed out after %v", timeout)
}

func main() {
    client := NewClient("https://api.freeagentics.com")
    
    // Login
    if err := client.Login("your_username", "your_password"); err != nil {
        fmt.Printf("Login failed: %v\n", err)
        return
    }
    
    // Create agent
    agent, err := client.CreateAgent(AgentConfig{
        Name:     "Go Agent",
        Template: "research_v2",
        Parameters: map[string]interface{}{
            "temperature": 0.7,
            "max_tokens":  2048,
        },
    })
    if err != nil {
        fmt.Printf("Agent creation failed: %v\n", err)
        return
    }
    
    fmt.Printf("Created agent: %s\n", agent.ID)
    
    // Run inference
    inference, err := client.RunInference(InferenceRequest{
        AgentID: agent.ID,
        Query:   "What are the benefits of Go programming language?",
        Context: map[string]interface{}{
            "focus": "performance",
            "year":  "2024",
        },
    })
    if err != nil {
        fmt.Printf("Inference failed: %v\n", err)
        return
    }
    
    fmt.Printf("Started inference: %s\n", inference.InferenceID)
    
    // Wait for result
    result, err := client.WaitForInference(inference.InferenceID, 5*time.Minute)
    if err != nil {
        fmt.Printf("Failed to get result: %v\n", err)
        return
    }
    
    if resultData, ok := result["result"].(map[string]interface{}); ok {
        if analysis, ok := resultData["analysis"].(string); ok {
            fmt.Printf("Analysis: %s\n", analysis)
        }
    }
}
```

## Java Examples

### Spring Boot Integration

```java
package com.example.freeagentics;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

@SpringBootApplication
public class FreeAgenticsApplication {
    public static void main(String[] args) {
        SpringApplication.run(FreeAgenticsApplication.class, args);
    }
}

@Configuration
class Config {
    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
    
    @Bean
    public ObjectMapper objectMapper() {
        return new ObjectMapper();
    }
}

@Service
class FreeAgenticsService {
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    private final String baseUrl = "https://api.freeagentics.com";
    
    private String accessToken;
    private String refreshToken;
    private LocalDateTime tokenExpiry;
    
    public FreeAgenticsService(RestTemplate restTemplate, ObjectMapper objectMapper) {
        this.restTemplate = restTemplate;
        this.objectMapper = objectMapper;
    }
    
    public void login(String username, String password) throws Exception {
        Map<String, String> loginData = new HashMap<>();
        loginData.put("username", username);
        loginData.put("password", password);
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        
        HttpEntity<Map<String, String>> entity = new HttpEntity<>(loginData, headers);
        
        ResponseEntity<Map> response = restTemplate.postForEntity(
            baseUrl + "/api/v1/login", 
            entity, 
            Map.class
        );
        
        if (response.getStatusCode() == HttpStatus.OK) {
            Map<String, Object> responseBody = response.getBody();
            this.accessToken = (String) responseBody.get("access_token");
            this.refreshToken = (String) responseBody.get("refresh_token");
            this.tokenExpiry = LocalDateTime.now().plusMinutes(14);
        } else {
            throw new RuntimeException("Login failed: " + response.getStatusCode());
        }
    }
    
    private void refreshTokenIfNeeded() throws Exception {
        if (tokenExpiry != null && LocalDateTime.now().isAfter(tokenExpiry)) {
            Map<String, String> refreshData = new HashMap<>();
            refreshData.put("refresh_token", refreshToken);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<Map<String, String>> entity = new HttpEntity<>(refreshData, headers);
            
            ResponseEntity<Map> response = restTemplate.postForEntity(
                baseUrl + "/api/v1/refresh", 
                entity, 
                Map.class
            );
            
            if (response.getStatusCode() == HttpStatus.OK) {
                Map<String, Object> responseBody = response.getBody();
                this.accessToken = (String) responseBody.get("access_token");
                this.refreshToken = (String) responseBody.get("refresh_token");
                this.tokenExpiry = LocalDateTime.now().plusMinutes(14);
            } else {
                throw new RuntimeException("Token refresh failed: " + response.getStatusCode());
            }
        }
    }
    
    private HttpHeaders getAuthHeaders() throws Exception {
        refreshTokenIfNeeded();
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setBearerAuth(accessToken);
        
        return headers;
    }
    
    public Map<String, Object> createAgent(AgentConfig config) throws Exception {
        HttpHeaders headers = getAuthHeaders();
        HttpEntity<AgentConfig> entity = new HttpEntity<>(config, headers);
        
        ResponseEntity<Map> response = restTemplate.postForEntity(
            baseUrl + "/api/v1/agents", 
            entity, 
            Map.class
        );
        
        if (response.getStatusCode() == HttpStatus.OK) {
            return response.getBody();
        } else {
            throw new RuntimeException("Agent creation failed: " + response.getStatusCode());
        }
    }
    
    public Map<String, Object> runInference(InferenceRequest request) throws Exception {
        HttpHeaders headers = getAuthHeaders();
        HttpEntity<InferenceRequest> entity = new HttpEntity<>(request, headers);
        
        ResponseEntity<Map> response = restTemplate.postForEntity(
            baseUrl + "/api/v1/inference", 
            entity, 
            Map.class
        );
        
        if (response.getStatusCode() == HttpStatus.OK) {
            return response.getBody();
        } else {
            throw new RuntimeException("Inference failed: " + response.getStatusCode());
        }
    }
    
    public Map<String, Object> getInferenceResult(String inferenceId) throws Exception {
        HttpHeaders headers = getAuthHeaders();
        HttpEntity<?> entity = new HttpEntity<>(headers);
        
        ResponseEntity<Map> response = restTemplate.exchange(
            baseUrl + "/api/v1/inference/" + inferenceId,
            HttpMethod.GET,
            entity,
            Map.class
        );
        
        if (response.getStatusCode() == HttpStatus.OK) {
            return response.getBody();
        } else {
            throw new RuntimeException("Failed to get inference result: " + response.getStatusCode());
        }
    }
    
    public CompletableFuture<Map<String, Object>> waitForInference(String inferenceId, int timeoutSeconds) {
        return CompletableFuture.supplyAsync(() -> {
            long startTime = System.currentTimeMillis();
            long timeout = timeoutSeconds * 1000L;
            
            while (System.currentTimeMillis() - startTime < timeout) {
                try {
                    Map<String, Object> result = getInferenceResult(inferenceId);
                    String status = (String) result.get("status");
                    
                    if ("completed".equals(status)) {
                        return result;
                    } else if ("failed".equals(status)) {
                        throw new RuntimeException("Inference failed: " + result.get("error"));
                    }
                    
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException("Interrupted while waiting for inference");
                } catch (Exception e) {
                    throw new RuntimeException("Error while waiting for inference", e);
                }
            }
            
            throw new RuntimeException("Inference timed out after " + timeoutSeconds + " seconds");
        });
    }
}

// DTOs
class AgentConfig {
    private String name;
    private String template;
    private Map<String, Object> parameters;
    private String gmnSpec;
    private boolean usePymdp = true;
    private int planningHorizon = 3;
    
    // Getters and setters
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    
    public String getTemplate() { return template; }
    public void setTemplate(String template) { this.template = template; }
    
    public Map<String, Object> getParameters() { return parameters; }
    public void setParameters(Map<String, Object> parameters) { this.parameters = parameters; }
    
    public String getGmnSpec() { return gmnSpec; }
    public void setGmnSpec(String gmnSpec) { this.gmnSpec = gmnSpec; }
    
    public boolean isUsePymdp() { return usePymdp; }
    public void setUsePymdp(boolean usePymdp) { this.usePymdp = usePymdp; }
    
    public int getPlanningHorizon() { return planningHorizon; }
    public void setPlanningHorizon(int planningHorizon) { this.planningHorizon = planningHorizon; }
}

class InferenceRequest {
    private String agentId;
    private String query;
    private Map<String, Object> context;
    private Map<String, Object> parameters;
    
    // Getters and setters
    public String getAgentId() { return agentId; }
    public void setAgentId(String agentId) { this.agentId = agentId; }
    
    public String getQuery() { return query; }
    public void setQuery(String query) { this.query = query; }
    
    public Map<String, Object> getContext() { return context; }
    public void setContext(Map<String, Object> context) { this.context = context; }
    
    public Map<String, Object> getParameters() { return parameters; }
    public void setParameters(Map<String, Object> parameters) { this.parameters = parameters; }
}

@RestController
@RequestMapping("/api/agents")
class AgentController {
    private final FreeAgenticsService freeAgenticsService;
    
    public AgentController(FreeAgenticsService freeAgenticsService) {
        this.freeAgenticsService = freeAgenticsService;
    }
    
    @PostMapping
    public ResponseEntity<Map<String, Object>> createAgent(@RequestBody AgentConfig config) {
        try {
            Map<String, Object> agent = freeAgenticsService.createAgent(config);
            return ResponseEntity.ok(agent);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(Map.of("error", e.getMessage()));
        }
    }
    
    @PostMapping("/{agentId}/inference")
    public ResponseEntity<CompletableFuture<Map<String, Object>>> runInference(
            @PathVariable String agentId,
            @RequestBody Map<String, Object> request) {
        try {
            InferenceRequest inferenceRequest = new InferenceRequest();
            inferenceRequest.setAgentId(agentId);
            inferenceRequest.setQuery((String) request.get("query"));
            inferenceRequest.setContext((Map<String, Object>) request.get("context"));
            inferenceRequest.setParameters((Map<String, Object>) request.get("parameters"));
            
            Map<String, Object> inference = freeAgenticsService.runInference(inferenceRequest);
            String inferenceId = (String) inference.get("inference_id");
            
            CompletableFuture<Map<String, Object>> result = 
                freeAgenticsService.waitForInference(inferenceId, 300);
            
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(CompletableFuture.completedFuture(Map.of("error", e.getMessage())));
        }
    }
}
```

## C# Examples

### .NET Core Client

```csharp
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace FreeAgentics.Client
{
    public class FreeAgenticsClient
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<FreeAgenticsClient> _logger;
        private readonly string _baseUrl;
        
        private string _accessToken;
        private string _refreshToken;
        private DateTime _tokenExpiry;
        
        public FreeAgenticsClient(HttpClient httpClient, ILogger<FreeAgenticsClient> logger, string baseUrl = "https://api.freeagentics.com")
        {
            _httpClient = httpClient;
            _logger = logger;
            _baseUrl = baseUrl;
        }
        
        public async Task<bool> LoginAsync(string username, string password)
        {
            try
            {
                var loginData = new { username, password };
                var json = JsonSerializer.Serialize(loginData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                var response = await _httpClient.PostAsync($"{_baseUrl}/api/v1/login", content);
                
                if (response.IsSuccessStatusCode)
                {
                    var responseData = await response.Content.ReadAsStringAsync();
                    var tokenResponse = JsonSerializer.Deserialize<TokenResponse>(responseData);
                    
                    _accessToken = tokenResponse.AccessToken;
                    _refreshToken = tokenResponse.RefreshToken;
                    _tokenExpiry = DateTime.Now.AddMinutes(14);
                    
                    _logger.LogInformation("Login successful");
                    return true;
                }
                else
                {
                    _logger.LogError($"Login failed with status: {response.StatusCode}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Login failed");
                return false;
            }
        }
        
        private async Task RefreshTokenIfNeededAsync()
        {
            if (DateTime.Now >= _tokenExpiry)
            {
                var refreshData = new { refresh_token = _refreshToken };
                var json = JsonSerializer.Serialize(refreshData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                var response = await _httpClient.PostAsync($"{_baseUrl}/api/v1/refresh", content);
                
                if (response.IsSuccessStatusCode)
                {
                    var responseData = await response.Content.ReadAsStringAsync();
                    var tokenResponse = JsonSerializer.Deserialize<TokenResponse>(responseData);
                    
                    _accessToken = tokenResponse.AccessToken;
                    _refreshToken = tokenResponse.RefreshToken;
                    _tokenExpiry = DateTime.Now.AddMinutes(14);
                }
                else
                {
                    throw new Exception("Token refresh failed");
                }
            }
        }
        
        private async Task<HttpRequestMessage> CreateAuthenticatedRequestAsync(HttpMethod method, string path, object content = null)
        {
            await RefreshTokenIfNeededAsync();
            
            var request = new HttpRequestMessage(method, $"{_baseUrl}{path}");
            request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _accessToken);
            
            if (content != null)
            {
                var json = JsonSerializer.Serialize(content);
                request.Content = new StringContent(json, Encoding.UTF8, "application/json");
            }
            
            return request;
        }
        
        public async Task<Agent> CreateAgentAsync(AgentConfig config)
        {
            try
            {
                var request = await CreateAuthenticatedRequestAsync(HttpMethod.Post, "/api/v1/agents", config);
                var response = await _httpClient.SendAsync(request);
                
                if (response.IsSuccessStatusCode)
                {
                    var responseData = await response.Content.ReadAsStringAsync();
                    var agent = JsonSerializer.Deserialize<Agent>(responseData);
                    
                    _logger.LogInformation($"Agent created: {agent.Id}");
                    return agent;
                }
                else
                {
                    throw new Exception($"Agent creation failed with status: {response.StatusCode}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Agent creation failed");
                throw;
            }
        }
        
        public async Task<InferenceResponse> RunInferenceAsync(InferenceRequest request)
        {
            try
            {
                var httpRequest = await CreateAuthenticatedRequestAsync(HttpMethod.Post, "/api/v1/inference", request);
                var response = await _httpClient.SendAsync(httpRequest);
                
                if (response.IsSuccessStatusCode)
                {
                    var responseData = await response.Content.ReadAsStringAsync();
                    var inferenceResponse = JsonSerializer.Deserialize<InferenceResponse>(responseData);
                    
                    _logger.LogInformation($"Inference started: {inferenceResponse.InferenceId}");
                    return inferenceResponse;
                }
                else
                {
                    throw new Exception($"Inference failed with status: {response.StatusCode}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Inference failed");
                throw;
            }
        }
        
        public async Task<InferenceResult> GetInferenceResultAsync(string inferenceId)
        {
            try
            {
                var request = await CreateAuthenticatedRequestAsync(HttpMethod.Get, $"/api/v1/inference/{inferenceId}");
                var response = await _httpClient.SendAsync(request);
                
                if (response.IsSuccessStatusCode)
                {
                    var responseData = await response.Content.ReadAsStringAsync();
                    var result = JsonSerializer.Deserialize<InferenceResult>(responseData);
                    
                    return result;
                }
                else
                {
                    throw new Exception($"Failed to get inference result with status: {response.StatusCode}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to get inference result");
                throw;
            }
        }
        
        public async Task<InferenceResult> WaitForInferenceAsync(string inferenceId, int timeoutSeconds = 300)
        {
            var startTime = DateTime.Now;
            var timeout = TimeSpan.FromSeconds(timeoutSeconds);
            
            while (DateTime.Now - startTime < timeout)
            {
                var result = await GetInferenceResultAsync(inferenceId);
                
                if (result.Status == "completed")
                {
                    return result;
                }
                else if (result.Status == "failed")
                {
                    throw new Exception($"Inference failed: {result.Error}");
                }
                
                await Task.Delay(1000);
            }
            
            throw new TimeoutException($"Inference timed out after {timeoutSeconds} seconds");
        }
    }
    
    // Data models
    public class TokenResponse
    {
        public string AccessToken { get; set; }
        public string RefreshToken { get; set; }
        public string TokenType { get; set; }
        public User User { get; set; }
    }
    
    public class User
    {
        public string UserId { get; set; }
        public string Username { get; set; }
        public string Role { get; set; }
        public string[] Permissions { get; set; }
    }
    
    public class AgentConfig
    {
        public string Name { get; set; }
        public string Template { get; set; }
        public Dictionary<string, object> Parameters { get; set; }
        public string GmnSpec { get; set; }
        public bool UsePymdp { get; set; } = true;
        public int PlanningHorizon { get; set; } = 3;
    }
    
    public class Agent
    {
        public string Id { get; set; }
        public string Name { get; set; }
        public string Template { get; set; }
        public string Status { get; set; }
        public DateTime CreatedAt { get; set; }
        public DateTime? LastActive { get; set; }
        public int InferenceCount { get; set; }
        public Dictionary<string, object> Parameters { get; set; }
    }
    
    public class InferenceRequest
    {
        public string AgentId { get; set; }
        public string Query { get; set; }
        public Dictionary<string, object> Context { get; set; }
        public Dictionary<string, object> Parameters { get; set; }
    }
    
    public class InferenceResponse
    {
        public string InferenceId { get; set; }
        public string AgentId { get; set; }
        public string Status { get; set; }
        public DateTime CreatedAt { get; set; }
        public DateTime? EstimatedCompletion { get; set; }
    }
    
    public class InferenceResult
    {
        public string InferenceId { get; set; }
        public string Status { get; set; }
        public Dictionary<string, object> Result { get; set; }
        public double? ProcessingTime { get; set; }
        public DateTime? CompletedAt { get; set; }
        public string Error { get; set; }
    }
    
    // ASP.NET Core service registration
    public static class ServiceCollectionExtensions
    {
        public static IServiceCollection AddFreeAgentics(this IServiceCollection services, string baseUrl = "https://api.freeagentics.com")
        {
            services.AddHttpClient<FreeAgenticsClient>(client =>
            {
                client.Timeout = TimeSpan.FromSeconds(30);
            });
            
            services.AddSingleton<FreeAgenticsClient>(provider =>
            {
                var httpClient = provider.GetRequiredService<HttpClient>();
                var logger = provider.GetRequiredService<ILogger<FreeAgenticsClient>>();
                return new FreeAgenticsClient(httpClient, logger, baseUrl);
            });
            
            return services;
        }
    }
    
    // Usage example
    public class Program
    {
        public static async Task Main(string[] args)
        {
            var host = Host.CreateDefaultBuilder(args)
                .ConfigureServices((context, services) =>
                {
                    services.AddFreeAgentics();
                })
                .Build();
            
            var client = host.Services.GetRequiredService<FreeAgenticsClient>();
            
            // Login
            if (await client.LoginAsync("your_username", "your_password"))
            {
                // Create agent
                var agent = await client.CreateAgentAsync(new AgentConfig
                {
                    Name = "C# Agent",
                    Template = "research_v2",
                    Parameters = new Dictionary<string, object>
                    {
                        ["temperature"] = 0.7,
                        ["max_tokens"] = 2048
                    }
                });
                
                Console.WriteLine($"Created agent: {agent.Id}");
                
                // Run inference
                var inference = await client.RunInferenceAsync(new InferenceRequest
                {
                    AgentId = agent.Id,
                    Query = "What are the advantages of .NET Core?",
                    Context = new Dictionary<string, object>
                    {
                        ["focus"] = "performance",
                        ["year"] = "2024"
                    }
                });
                
                Console.WriteLine($"Started inference: {inference.InferenceId}");
                
                // Wait for result
                var result = await client.WaitForInferenceAsync(inference.InferenceId);
                
                if (result.Result.ContainsKey("analysis"))
                {
                    Console.WriteLine($"Analysis: {result.Result["analysis"]}");
                }
            }
        }
    }
}
```

## cURL Examples

### Basic Authentication Flow

```bash
#!/bin/bash

# Configuration
BASE_URL="https://api.freeagentics.com"
USERNAME="your_username"
PASSWORD="your_password"

# Login and get tokens
echo "Logging in..."
LOGIN_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"$USERNAME\", \"password\": \"$PASSWORD\"}")

# Extract access token
ACCESS_TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.access_token')
REFRESH_TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.refresh_token')

if [ "$ACCESS_TOKEN" == "null" ]; then
  echo "Login failed"
  exit 1
fi

echo "Login successful"

# Create agent
echo "Creating agent..."
CREATE_AGENT_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/agents" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "cURL Agent",
    "template": "research_v2",
    "parameters": {
      "temperature": 0.7,
      "max_tokens": 2048
    },
    "use_pymdp": true,
    "planning_horizon": 3
  }')

# Extract agent ID
AGENT_ID=$(echo "$CREATE_AGENT_RESPONSE" | jq -r '.id')

if [ "$AGENT_ID" == "null" ]; then
  echo "Agent creation failed"
  exit 1
fi

echo "Agent created: $AGENT_ID"

# Run inference
echo "Running inference..."
INFERENCE_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/inference" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"agent_id\": \"$AGENT_ID\",
    \"query\": \"What are the benefits of using cURL for API testing?\",
    \"context\": {
      \"focus\": \"command line tools\",
      \"year\": \"2024\"
    }
  }")

# Extract inference ID
INFERENCE_ID=$(echo "$INFERENCE_RESPONSE" | jq -r '.inference_id')

if [ "$INFERENCE_ID" == "null" ]; then
  echo "Inference failed"
  exit 1
fi

echo "Inference started: $INFERENCE_ID"

# Wait for inference completion
echo "Waiting for inference completion..."
while true; do
  RESULT_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/inference/$INFERENCE_ID" \
    -H "Authorization: Bearer $ACCESS_TOKEN")
  
  STATUS=$(echo "$RESULT_RESPONSE" | jq -r '.status')
  
  if [ "$STATUS" == "completed" ]; then
    echo "Inference completed!"
    ANALYSIS=$(echo "$RESULT_RESPONSE" | jq -r '.result.analysis')
    echo "Analysis: $ANALYSIS"
    break
  elif [ "$STATUS" == "failed" ]; then
    echo "Inference failed"
    ERROR=$(echo "$RESULT_RESPONSE" | jq -r '.error')
    echo "Error: $ERROR"
    break
  else
    echo "Status: $STATUS"
    sleep 2
  fi
done

# Get agent metrics
echo "Getting agent metrics..."
METRICS_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/agents/$AGENT_ID/metrics" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

echo "Agent metrics:"
echo "$METRICS_RESPONSE" | jq '.'
```

### Batch Operations

```bash
#!/bin/bash

# Batch inference example
echo "Running batch inference..."
BATCH_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/batch-inference" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"requests\": [
      {
        \"agent_id\": \"$AGENT_ID\",
        \"query\": \"Analyze market trends in AI\"
      },
      {
        \"agent_id\": \"$AGENT_ID\",
        \"query\": \"Analyze market trends in blockchain\"
      },
      {
        \"agent_id\": \"$AGENT_ID\",
        \"query\": \"Analyze market trends in quantum computing\"
      }
    ],
    \"parallel\": true
  }")

echo "Batch inference response:"
echo "$BATCH_RESPONSE" | jq '.'
```

### Knowledge Graph Operations

```bash
#!/bin/bash

# Search knowledge graph
echo "Searching knowledge graph..."
SEARCH_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/knowledge/search" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -G \
  -d "q=machine learning" \
  -d "limit=10" \
  -d "type=concept")

echo "Search results:"
echo "$SEARCH_RESPONSE" | jq '.'

# Create knowledge entity
echo "Creating knowledge entity..."
ENTITY_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/knowledge/entities" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "concept",
    "label": "Neural Networks",
    "properties": {
      "description": "Computing systems inspired by biological neural networks",
      "category": "AI/ML",
      "importance": "high"
    }
  }')

ENTITY_ID=$(echo "$ENTITY_RESPONSE" | jq -r '.id')
echo "Created entity: $ENTITY_ID"

# Create relationship
echo "Creating relationship..."
RELATIONSHIP_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/knowledge/relationships" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"source_id\": \"$ENTITY_ID\",
    \"target_id\": \"existing_entity_123\",
    \"relationship_type\": \"related_to\",
    \"properties\": {
      \"strength\": 0.9,
      \"confidence\": 0.95
    }
  }")

echo "Relationship created:"
echo "$RELATIONSHIP_RESPONSE" | jq '.'
```

### System Monitoring

```bash
#!/bin/bash

# Get system status
echo "Getting system status..."
STATUS_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/system/status" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

echo "System status:"
echo "$STATUS_RESPONSE" | jq '.'

# Get system metrics
echo "Getting system metrics..."
METRICS_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/system/metrics" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

echo "System metrics:"
echo "$METRICS_RESPONSE" | jq '.'

# Get active alerts
echo "Getting active alerts..."
ALERTS_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/monitoring/alerts" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

echo "Active alerts:"
echo "$ALERTS_RESPONSE" | jq '.'
```

### Token Refresh

```bash
#!/bin/bash

# Function to refresh token
refresh_token() {
  echo "Refreshing token..."
  REFRESH_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/refresh" \
    -H "Content-Type: application/json" \
    -d "{\"refresh_token\": \"$REFRESH_TOKEN\"}")
  
  NEW_ACCESS_TOKEN=$(echo "$REFRESH_RESPONSE" | jq -r '.access_token')
  NEW_REFRESH_TOKEN=$(echo "$REFRESH_RESPONSE" | jq -r '.refresh_token')
  
  if [ "$NEW_ACCESS_TOKEN" != "null" ]; then
    ACCESS_TOKEN="$NEW_ACCESS_TOKEN"
    REFRESH_TOKEN="$NEW_REFRESH_TOKEN"
    echo "Token refreshed successfully"
  else
    echo "Token refresh failed"
    exit 1
  fi
}

# Use before long-running operations
refresh_token
```

These examples provide comprehensive coverage of the FreeAgentics API in multiple programming languages, showing both basic usage and advanced patterns. Each example includes proper error handling, authentication management, and real-world usage scenarios.

For more examples and the latest updates, check the [GitHub repository](https://github.com/freeagentics/examples) and [API documentation](API_REFERENCE.md).