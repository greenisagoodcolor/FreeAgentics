# FreeAgentics Developer Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Authentication](#authentication)
3. [SDK Usage](#sdk-usage)
   - [Python SDK](#python-sdk)
   - [JavaScript/TypeScript SDK](#javascripttypescript-sdk)
   - [Go SDK](#go-sdk)
   - [Java SDK](#java-sdk)
   - [.NET SDK](#net-sdk)
4. [Integration Patterns](#integration-patterns)
5. [Best Practices](#best-practices)
6. [Common Use Cases](#common-use-cases)
7. [Troubleshooting](#troubleshooting)

## Quick Start

Get up and running with FreeAgentics in minutes.

### 1. Create an Account

Sign up at [https://freeagentics.com/signup](https://freeagentics.com/signup) or via API:

```bash
curl -X POST https://api.freeagentics.com/api/v1/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "myusername",
    "email": "user@example.com",
    "password": "securepassword123"
  }'
```

### 2. Install SDK

Choose your preferred language:

```bash
# Python
pip install freeagentics

# JavaScript/TypeScript
npm install @freeagentics/sdk

# Go
go get github.com/freeagentics/go-sdk

# Java
<dependency>
  <groupId>com.freeagentics</groupId>
  <artifactId>freeagentics-sdk</artifactId>
  <version>0.1.0</version>
</dependency>

# .NET
dotnet add package FreeAgentics.SDK
```

### 3. Initialize Client

```python
from freeagentics import FreeAgenticsClient

client = FreeAgenticsClient(
    username="myusername",
    password="securepassword123"
)

# Create your first agent
agent = client.agents.create(
    name="My First Agent",
    template="research_v2"
)

# Run inference
result = client.inference.run(
    agent_id=agent.id,
    query="Analyze recent market trends"
)

print(result.analysis)
```

## Authentication

### Token Management

FreeAgentics uses JWT tokens with automatic refresh handling.

#### Manual Token Management

```python
# Python Example
import requests
from datetime import datetime, timedelta

class TokenManager:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None

    def login(self):
        response = requests.post(
            "https://api.freeagentics.com/api/v1/login",
            json={"username": self.username, "password": self.password}
        )
        data = response.json()
        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        # Token expires in 15 minutes
        self.token_expiry = datetime.now() + timedelta(minutes=14)

    def get_token(self):
        if not self.access_token or datetime.now() >= self.token_expiry:
            self.refresh()
        return self.access_token

    def refresh(self):
        response = requests.post(
            "https://api.freeagentics.com/api/v1/refresh",
            json={"refresh_token": self.refresh_token}
        )
        data = response.json()
        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        self.token_expiry = datetime.now() + timedelta(minutes=14)
```

#### Using Client Fingerprinting

For enhanced security, include a client fingerprint:

```javascript
// JavaScript Example
const crypto = require('crypto');

class SecureClient {
    constructor() {
        // Generate unique client fingerprint
        this.fingerprint = crypto.randomBytes(32).toString('hex');
    }

    async makeRequest(endpoint, options = {}) {
        const response = await fetch(`https://api.freeagentics.com/api/v1${endpoint}`, {
            ...options,
            headers: {
                ...options.headers,
                'Authorization': `Bearer ${this.accessToken}`,
                'X-Client-Fingerprint': this.fingerprint,
                'X-Client-Version': '1.0.0'
            }
        });

        return response.json();
    }
}
```

## SDK Usage

### Python SDK

#### Installation and Setup

```bash
pip install freeagentics
```

#### Basic Usage

```python
from freeagentics import FreeAgenticsClient
from freeagentics.exceptions import RateLimitError, AuthenticationError

# Initialize client
client = FreeAgenticsClient(
    username="your_username",
    password="your_password",
    # Optional configuration
    base_url="https://api.freeagentics.com",
    timeout=30,
    max_retries=3
)

# Error handling
try:
    agents = client.agents.list()
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except AuthenticationError:
    print("Authentication failed")
```

#### Agent Management

```python
# Create agent with custom parameters
agent = client.agents.create(
    name="Market Analyzer",
    template="research_v2",
    parameters={
        "temperature": 0.7,
        "max_tokens": 2048,
        "focus_areas": ["finance", "technology"]
    },
    gmn_spec="""
    # GMN Specification for Market Analysis
    states: [analyzing, summarizing, idle]
    observations: [market_data, news_feeds]
    actions: [analyze, summarize, wait]
    """,
    use_pymdp=True,
    planning_horizon=3
)

# Update agent
agent = client.agents.update(
    agent_id=agent.id,
    parameters={"temperature": 0.8}
)

# Get agent metrics
metrics = client.agents.get_metrics(agent.id)
print(f"Total inferences: {metrics.total_inferences}")
print(f"Average response time: {metrics.avg_response_time}s")

# Control agent lifecycle
client.agents.start(agent.id)
client.agents.stop(agent.id)
```

#### Running Inference

```python
# Simple inference
result = client.inference.run(
    agent_id=agent.id,
    query="What are the key trends in AI technology?"
)

# Inference with context
result = client.inference.run(
    agent_id=agent.id,
    query="Analyze Q4 performance",
    context={
        "company": "TechCorp",
        "metrics": ["revenue", "growth", "market_share"],
        "compare_to": "Q3"
    },
    parameters={
        "temperature": 0.5,  # More focused responses
        "max_tokens": 1000
    }
)

# Batch inference
batch_results = client.inference.batch([
    {
        "agent_id": agent.id,
        "query": "Analyze AAPL stock"
    },
    {
        "agent_id": agent.id,
        "query": "Analyze GOOGL stock"
    }
], parallel=True)

# Async inference with callback
def on_complete(result):
    print(f"Inference completed: {result.inference_id}")
    print(f"Result: {result.analysis}")

client.inference.run_async(
    agent_id=agent.id,
    query="Long-running analysis task",
    callback=on_complete
)
```

#### Knowledge Graph Operations

```python
# Search knowledge graph
results = client.knowledge.search(
    query="machine learning",
    limit=10,
    entity_type="concept"
)

# Create entity
entity = client.knowledge.create_entity(
    type="concept",
    label="Quantum Computing",
    properties={
        "description": "Computing using quantum mechanical phenomena",
        "category": "technology",
        "related_fields": ["physics", "computer science"]
    }
)

# Create relationship
relationship = client.knowledge.create_relationship(
    source_id=entity.id,
    target_id="existing_entity_123",
    relationship_type="related_to",
    properties={"strength": 0.8}
)

# Build knowledge subgraph
subgraph = client.knowledge.get_subgraph(
    entity_id=entity.id,
    depth=2,  # Two levels of relationships
    relationship_types=["related_to", "derived_from"]
)
```

#### Real-time Monitoring

```python
# WebSocket connection for real-time updates
async def monitor_agents():
    async with client.websocket() as ws:
        # Subscribe to events
        await ws.subscribe(
            events=["agent_status", "metrics"],
            agents=[agent.id]
        )

        # Handle messages
        async for message in ws:
            if message.type == "agent_status":
                print(f"Agent {message.data.agent_id} is now {message.data.status}")
            elif message.type == "metrics_update":
                print(f"CPU: {message.data.cpu_usage}%")

# Run monitor
import asyncio
asyncio.run(monitor_agents())
```

### JavaScript/TypeScript SDK

#### Installation

```bash
npm install @freeagentics/sdk
# or
yarn add @freeagentics/sdk
```

#### TypeScript Setup

```typescript
import { FreeAgenticsClient, Agent, InferenceResult } from '@freeagentics/sdk';

const client = new FreeAgenticsClient({
    username: 'your_username',
    password: 'your_password',
    // Optional type-safe configuration
    config: {
        baseUrl: 'https://api.freeagentics.com',
        timeout: 30000,
        retryAttempts: 3
    }
});
```

#### Async/Await Pattern

```typescript
// Modern async/await usage
async function analyzeMarket(): Promise<void> {
    try {
        // Create agent
        const agent: Agent = await client.agents.create({
            name: 'Market Analyzer',
            template: 'research_v2',
            parameters: {
                temperature: 0.7
            }
        });

        // Run inference
        const result: InferenceResult = await client.inference.run({
            agentId: agent.id,
            query: 'Analyze tech sector performance',
            context: {
                timeframe: '2024-Q1',
                sectors: ['software', 'hardware']
            }
        });

        console.log('Analysis:', result.analysis);
        console.log('Confidence:', result.confidence);

    } catch (error) {
        if (error.code === 'RATE_LIMITED') {
            console.log(`Rate limited. Retry after ${error.retryAfter}s`);
        } else {
            console.error('Error:', error.message);
        }
    }
}
```

#### Promise Chaining

```javascript
// Promise-based approach
client.agents.create({
    name: 'Research Agent',
    template: 'research_v2'
})
.then(agent => {
    return client.inference.run({
        agentId: agent.id,
        query: 'Latest AI developments'
    });
})
.then(result => {
    console.log('Result:', result);
})
.catch(error => {
    console.error('Error:', error);
});
```

#### Real-time WebSocket

```typescript
// TypeScript WebSocket example
import { WebSocketClient, Message } from '@freeagentics/sdk';

class AgentMonitor {
    private ws: WebSocketClient;

    constructor(private token: string) {
        this.ws = new WebSocketClient(token);
    }

    async start(): Promise<void> {
        await this.ws.connect();

        // Subscribe to events
        await this.ws.subscribe({
            events: ['agent_status', 'metrics'],
            agents: ['agent_123', 'agent_456']
        });

        // Handle messages
        this.ws.on('message', (message: Message) => {
            switch (message.type) {
                case 'agent_status':
                    this.handleStatusChange(message.data);
                    break;
                case 'metrics_update':
                    this.updateDashboard(message.data);
                    break;
                case 'error':
                    console.error('WebSocket error:', message.data);
                    break;
            }
        });

        // Handle connection events
        this.ws.on('disconnect', () => {
            console.log('Disconnected, attempting reconnect...');
        });
    }

    private handleStatusChange(data: any): void {
        console.log(`Agent ${data.agent_id} changed to ${data.status}`);
    }

    private updateDashboard(data: any): void {
        // Update UI with metrics
        document.getElementById('cpu-usage').textContent = `${data.cpu_usage}%`;
    }
}
```

### Go SDK

#### Installation

```bash
go get github.com/freeagentics/go-sdk
```

#### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/freeagentics/go-sdk"
)

func main() {
    // Create client
    client, err := freeagentics.NewClient(
        freeagentics.WithCredentials("username", "password"),
        freeagentics.WithTimeout(30),
    )
    if err != nil {
        log.Fatal(err)
    }

    ctx := context.Background()

    // Create agent
    agent, err := client.Agents.Create(ctx, &freeagentics.CreateAgentRequest{
        Name:     "Go Agent",
        Template: "research_v2",
        Parameters: map[string]interface{}{
            "temperature": 0.7,
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    // Run inference
    result, err := client.Inference.Run(ctx, &freeagentics.InferenceRequest{
        AgentID: agent.ID,
        Query:   "Analyze Go ecosystem growth",
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Analysis: %s\n", result.Analysis)
}
```

#### Concurrent Operations

```go
// Concurrent inference example
func runConcurrentInferences(client *freeagentics.Client, agentID string, queries []string) {
    ctx := context.Background()
    results := make(chan *freeagentics.InferenceResult, len(queries))
    errors := make(chan error, len(queries))

    // Launch goroutines
    for _, query := range queries {
        go func(q string) {
            result, err := client.Inference.Run(ctx, &freeagentics.InferenceRequest{
                AgentID: agentID,
                Query:   q,
            })
            if err != nil {
                errors <- err
                return
            }
            results <- result
        }(query)
    }

    // Collect results
    for i := 0; i < len(queries); i++ {
        select {
        case result := <-results:
            fmt.Printf("Result: %+v\n", result)
        case err := <-errors:
            fmt.Printf("Error: %v\n", err)
        }
    }
}
```

### Java SDK

#### Maven Setup

```xml
<dependency>
    <groupId>com.freeagentics</groupId>
    <artifactId>freeagentics-sdk</artifactId>
    <version>0.1.0</version>
</dependency>
```

#### Basic Usage

```java
import com.freeagentics.FreeAgenticsClient;
import com.freeagentics.models.*;
import com.freeagentics.exceptions.*;

public class Example {
    public static void main(String[] args) {
        // Initialize client
        FreeAgenticsClient client = FreeAgenticsClient.builder()
            .username("your_username")
            .password("your_password")
            .timeout(30)
            .build();

        try {
            // Create agent
            Agent agent = client.agents().create(
                CreateAgentRequest.builder()
                    .name("Java Agent")
                    .template("research_v2")
                    .parameter("temperature", 0.7)
                    .build()
            );

            // Run inference
            InferenceResult result = client.inference().run(
                InferenceRequest.builder()
                    .agentId(agent.getId())
                    .query("Analyze Java ecosystem")
                    .build()
            );

            System.out.println("Result: " + result.getAnalysis());

        } catch (RateLimitException e) {
            System.err.println("Rate limited: " + e.getRetryAfter());
        } catch (FreeAgenticsException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}
```

#### Async Operations

```java
import java.util.concurrent.CompletableFuture;

// Async inference
CompletableFuture<InferenceResult> future = client.inference().runAsync(
    InferenceRequest.builder()
        .agentId(agentId)
        .query("Long running analysis")
        .build()
);

// Handle completion
future.thenAccept(result -> {
    System.out.println("Completed: " + result.getAnalysis());
}).exceptionally(throwable -> {
    System.err.println("Failed: " + throwable.getMessage());
    return null;
});
```

### .NET SDK

#### NuGet Installation

```bash
dotnet add package FreeAgentics.SDK
```

#### C# Usage

```csharp
using FreeAgentics;
using FreeAgentics.Models;
using System;
using System.Threading.Tasks;

class Program
{
    static async Task Main(string[] args)
    {
        // Create client
        var client = new FreeAgenticsClient(
            username: "your_username",
            password: "your_password",
            options: new ClientOptions
            {
                BaseUrl = "https://api.freeagentics.com",
                Timeout = TimeSpan.FromSeconds(30)
            }
        );

        try
        {
            // Create agent
            var agent = await client.Agents.CreateAsync(new CreateAgentRequest
            {
                Name = "C# Agent",
                Template = "research_v2",
                Parameters = new Dictionary<string, object>
                {
                    ["temperature"] = 0.7
                }
            });

            // Run inference
            var result = await client.Inference.RunAsync(new InferenceRequest
            {
                AgentId = agent.Id,
                Query = "Analyze .NET ecosystem"
            });

            Console.WriteLine($"Analysis: {result.Analysis}");
            Console.WriteLine($"Confidence: {result.Confidence}");
        }
        catch (RateLimitException ex)
        {
            Console.WriteLine($"Rate limited. Retry after {ex.RetryAfter} seconds");
        }
        catch (FreeAgenticsException ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }
}
```

#### LINQ Integration

```csharp
// Query agents using LINQ
var activeAgents = await client.Agents
    .Where(a => a.Status == AgentStatus.Active)
    .OrderBy(a => a.CreatedAt)
    .Take(10)
    .ToListAsync();

// Process results
var tasks = activeAgents.Select(agent =>
    client.Inference.RunAsync(new InferenceRequest
    {
        AgentId = agent.Id,
        Query = "Status check"
    })
);

var results = await Task.WhenAll(tasks);
```

## Integration Patterns

### Microservices Architecture

```python
# Service integration example
from freeagentics import FreeAgenticsClient
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
client = FreeAgenticsClient(
    username=os.getenv('FREEAGENTICS_USER'),
    password=os.getenv('FREEAGENTICS_PASS')
)

@app.route('/analyze', methods=['POST'])
async def analyze():
    data = request.json

    try:
        # Use FreeAgentics for analysis
        result = await client.inference.run(
            agent_id=os.getenv('AGENT_ID'),
            query=data['query'],
            context=data.get('context', {})
        )

        return jsonify({
            'success': True,
            'analysis': result.analysis,
            'confidence': result.confidence
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

### Event-Driven Architecture

```javascript
// Event-driven integration with message queue
const { FreeAgenticsClient } = require('@freeagentics/sdk');
const amqp = require('amqplib');

class InferenceProcessor {
    constructor() {
        this.client = new FreeAgenticsClient({
            username: process.env.FREEAGENTICS_USER,
            password: process.env.FREEAGENTICS_PASS
        });
    }

    async start() {
        const connection = await amqp.connect('amqp://localhost');
        const channel = await connection.createChannel();

        await channel.assertQueue('inference_requests');

        // Process messages
        channel.consume('inference_requests', async (msg) => {
            const request = JSON.parse(msg.content.toString());

            try {
                // Run inference
                const result = await this.client.inference.run({
                    agentId: request.agentId,
                    query: request.query
                });

                // Publish result
                await channel.publish('', 'inference_results',
                    Buffer.from(JSON.stringify({
                        requestId: request.id,
                        result: result
                    }))
                );

                channel.ack(msg);
            } catch (error) {
                console.error('Processing error:', error);
                channel.nack(msg, false, true); // Requeue
            }
        });
    }
}
```

### Batch Processing Pipeline

```python
# Batch processing with progress tracking
import asyncio
from typing import List, Dict
from freeagentics import FreeAgenticsClient

class BatchProcessor:
    def __init__(self, client: FreeAgenticsClient):
        self.client = client
        self.progress = {}

    async def process_batch(self,
                          agent_id: str,
                          items: List[Dict],
                          concurrent_limit: int = 5):
        """Process items in batches with concurrency control."""

        semaphore = asyncio.Semaphore(concurrent_limit)
        tasks = []

        for i, item in enumerate(items):
            task = self._process_item(agent_id, item, i, semaphore)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            'successful': sum(1 for r in results if not isinstance(r, Exception)),
            'failed': sum(1 for r in results if isinstance(r, Exception)),
            'results': results
        }

    async def _process_item(self, agent_id: str, item: Dict,
                          index: int, semaphore: asyncio.Semaphore):
        async with semaphore:
            try:
                # Update progress
                self.progress[index] = 'processing'

                result = await self.client.inference.run_async(
                    agent_id=agent_id,
                    query=item['query'],
                    context=item.get('context', {})
                )

                self.progress[index] = 'completed'
                return result

            except Exception as e:
                self.progress[index] = 'failed'
                return e

    def get_progress(self):
        """Get current batch processing progress."""
        total = len(self.progress)
        completed = sum(1 for status in self.progress.values()
                       if status == 'completed')
        return {
            'total': total,
            'completed': completed,
            'percentage': (completed / total * 100) if total > 0 else 0
        }
```

### Webhook Integration

```typescript
// Webhook receiver with signature verification
import express from 'express';
import crypto from 'crypto';

const app = express();
app.use(express.raw({ type: 'application/json' }));

const WEBHOOK_SECRET = process.env.WEBHOOK_SECRET;

function verifySignature(payload: Buffer, signature: string, timestamp: string): boolean {
    const message = `${timestamp}.${payload}`;
    const expected = crypto
        .createHmac('sha256', WEBHOOK_SECRET)
        .update(message)
        .digest('hex');

    return crypto.timingSafeEqual(
        Buffer.from(expected),
        Buffer.from(signature)
    );
}

app.post('/webhook', (req, res) => {
    const signature = req.headers['x-freeagentics-signature'] as string;
    const timestamp = req.headers['x-freeagentics-timestamp'] as string;

    // Verify signature
    if (!verifySignature(req.body, signature, timestamp)) {
        return res.status(401).send('Invalid signature');
    }

    // Check timestamp (prevent replay attacks)
    const requestTime = parseInt(timestamp);
    const currentTime = Math.floor(Date.now() / 1000);
    if (Math.abs(currentTime - requestTime) > 300) { // 5 minutes
        return res.status(401).send('Request too old');
    }

    // Process webhook
    const event = JSON.parse(req.body.toString());

    switch (event.event) {
        case 'agent.created':
            handleAgentCreated(event.data);
            break;
        case 'inference.completed':
            handleInferenceCompleted(event.data);
            break;
        // ... handle other events
    }

    res.status(200).send('OK');
});
```

## Best Practices

### 1. Error Handling

Always implement comprehensive error handling:

```python
from freeagentics.exceptions import (
    RateLimitError,
    AuthenticationError,
    ValidationError,
    NetworkError
)
import time

def robust_inference(client, agent_id, query, max_retries=3):
    """Run inference with automatic retry logic."""

    for attempt in range(max_retries):
        try:
            return client.inference.run(
                agent_id=agent_id,
                query=query
            )

        except RateLimitError as e:
            # Wait for rate limit to reset
            wait_time = e.retry_after or 60
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)

        except NetworkError as e:
            # Exponential backoff for network errors
            wait_time = 2 ** attempt
            print(f"Network error. Retrying in {wait_time}s...")
            time.sleep(wait_time)

        except ValidationError as e:
            # Don't retry validation errors
            print(f"Validation error: {e}")
            raise

        except AuthenticationError:
            # Re-authenticate and retry
            client.authenticate()

    raise Exception(f"Failed after {max_retries} attempts")
```

### 2. Connection Pooling

Reuse connections for better performance:

```python
# Use connection pooling
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)

# Mount adapter with connection pool
adapter = HTTPAdapter(
    pool_connections=10,
    pool_maxsize=10,
    max_retries=retry_strategy
)
session.mount("https://", adapter)

# Use session in client
client = FreeAgenticsClient(
    username="user",
    password="pass",
    session=session  # Reuse connections
)
```

### 3. Caching Strategy

Implement intelligent caching:

```typescript
// Cache with TTL
class CachedClient {
    private cache = new Map<string, { data: any, expires: number }>();
    private client: FreeAgenticsClient;

    constructor(client: FreeAgenticsClient) {
        this.client = client;
    }

    async getAgent(agentId: string, ttl: number = 300000): Promise<Agent> {
        const cacheKey = `agent:${agentId}`;
        const cached = this.cache.get(cacheKey);

        if (cached && cached.expires > Date.now()) {
            return cached.data;
        }

        const agent = await this.client.agents.get(agentId);
        this.cache.set(cacheKey, {
            data: agent,
            expires: Date.now() + ttl
        });

        return agent;
    }

    invalidateAgent(agentId: string): void {
        this.cache.delete(`agent:${agentId}`);
    }
}
```

### 4. Logging and Monitoring

Implement comprehensive logging:

```python
import logging
from functools import wraps
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('freeagentics')

def log_performance(func):
    """Decorator to log function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise

    return wrapper

@log_performance
def analyze_data(client, agent_id, data):
    """Analyze data with performance logging."""
    return client.inference.run(
        agent_id=agent_id,
        query=f"Analyze: {data}"
    )
```

### 5. Security Best Practices

```python
# Secure credential management
import os
from cryptography.fernet import Fernet

class SecureCredentials:
    def __init__(self):
        # Use environment variable for encryption key
        key = os.environ.get('ENCRYPTION_KEY')
        if not key:
            raise ValueError("ENCRYPTION_KEY not set")

        self.cipher = Fernet(key.encode())

    def encrypt_password(self, password: str) -> bytes:
        """Encrypt password for storage."""
        return self.cipher.encrypt(password.encode())

    def decrypt_password(self, encrypted: bytes) -> str:
        """Decrypt password for use."""
        return self.cipher.decrypt(encrypted).decode()

    @staticmethod
    def get_client():
        """Get client with secure credentials."""
        # Never hardcode credentials
        username = os.environ.get('FREEAGENTICS_USER')
        password = os.environ.get('FREEAGENTICS_PASS')

        if not username or not password:
            raise ValueError("Credentials not configured")

        return FreeAgenticsClient(
            username=username,
            password=password,
            # Use secure connection settings
            verify_ssl=True,
            timeout=30
        )
```

## Common Use Cases

### 1. Automated Research Assistant

```python
class ResearchAssistant:
    def __init__(self, client: FreeAgenticsClient):
        self.client = client
        self.agent_id = None

    async def setup(self):
        """Create and configure research agent."""
        agent = await self.client.agents.create(
            name="Research Assistant",
            template="research_v2",
            parameters={
                "temperature": 0.7,
                "expertise": ["technology", "science", "business"]
            }
        )
        self.agent_id = agent.id
        await self.client.agents.start(agent.id)

    async def research_topic(self, topic: str, depth: str = "comprehensive"):
        """Research a topic with specified depth."""

        # Build research query
        query = f"""
        Research the topic: {topic}

        Depth: {depth}
        Include:
        - Current state and trends
        - Key players and innovations
        - Future predictions
        - Relevant statistics
        """

        result = await self.client.inference.run(
            agent_id=self.agent_id,
            query=query,
            parameters={
                "max_tokens": 4096 if depth == "comprehensive" else 2048
            }
        )

        return {
            "topic": topic,
            "research": result.analysis,
            "confidence": result.confidence,
            "sources": result.metadata.get("sources", [])
        }
```

### 2. Real-time Data Analysis Pipeline

```javascript
class DataAnalysisPipeline {
    constructor(client) {
        this.client = client;
        this.agents = {};
    }

    async initialize() {
        // Create specialized agents
        this.agents.preprocessor = await this.client.agents.create({
            name: 'Data Preprocessor',
            template: 'data_processing_v1'
        });

        this.agents.analyzer = await this.client.agents.create({
            name: 'Data Analyzer',
            template: 'analysis_v2'
        });

        this.agents.reporter = await this.client.agents.create({
            name: 'Report Generator',
            template: 'reporting_v1'
        });
    }

    async processDataStream(dataStream) {
        const pipeline = new Stream.Transform({
            objectMode: true,
            async transform(chunk, encoding, callback) {
                try {
                    // Preprocess
                    const preprocessed = await this.client.inference.run({
                        agentId: this.agents.preprocessor.id,
                        query: 'Clean and normalize data',
                        context: { data: chunk }
                    });

                    // Analyze
                    const analysis = await this.client.inference.run({
                        agentId: this.agents.analyzer.id,
                        query: 'Analyze patterns and anomalies',
                        context: { data: preprocessed.result }
                    });

                    // Generate report
                    const report = await this.client.inference.run({
                        agentId: this.agents.reporter.id,
                        query: 'Generate executive summary',
                        context: { analysis: analysis.result }
                    });

                    callback(null, report.result);
                } catch (error) {
                    callback(error);
                }
            }.bind(this)
        });

        return dataStream.pipe(pipeline);
    }
}
```

### 3. Multi-Agent Collaboration System

```python
class CollaborationSystem:
    def __init__(self, client: FreeAgenticsClient):
        self.client = client
        self.agents = {}
        self.knowledge_base = {}

    async def create_team(self, team_config):
        """Create a team of specialized agents."""

        for role, config in team_config.items():
            agent = await self.client.agents.create(
                name=f"{role} Agent",
                template=config['template'],
                parameters=config.get('parameters', {}),
                gmn_spec=config.get('gmn_spec')
            )
            self.agents[role] = agent

    async def collaborative_task(self, task_description):
        """Execute task using multiple agents collaboratively."""

        # Phase 1: Understanding
        understanding = await self.client.inference.run(
            agent_id=self.agents['analyst'].id,
            query=f"Break down this task: {task_description}"
        )

        # Phase 2: Planning
        plan = await self.client.inference.run(
            agent_id=self.agents['planner'].id,
            query="Create execution plan",
            context={
                "task_breakdown": understanding.result,
                "available_agents": list(self.agents.keys())
            }
        )

        # Phase 3: Parallel execution
        subtasks = plan.result.get('subtasks', [])
        results = await asyncio.gather(*[
            self._execute_subtask(subtask)
            for subtask in subtasks
        ])

        # Phase 4: Integration
        final_result = await self.client.inference.run(
            agent_id=self.agents['integrator'].id,
            query="Integrate results into cohesive output",
            context={"subtask_results": results}
        )

        return final_result

    async def _execute_subtask(self, subtask):
        """Execute individual subtask with appropriate agent."""

        agent_role = subtask.get('assigned_to')
        agent_id = self.agents[agent_role].id

        return await self.client.inference.run(
            agent_id=agent_id,
            query=subtask['description'],
            context=subtask.get('context', {})
        )
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Authentication Failures

**Problem**: 401 Unauthorized errors

**Solutions**:
```python
# Check token expiration
if client.is_token_expired():
    client.refresh_token()

# Verify credentials
try:
    client.authenticate()
except AuthenticationError as e:
    print(f"Auth failed: {e}")
    # Check username/password
    # Verify account is active
    # Check for 2FA requirements
```

#### 2. Rate Limiting

**Problem**: 429 Too Many Requests

**Solutions**:
```python
# Implement backoff strategy
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def make_request(client, *args, **kwargs):
    return client.inference.run(*args, **kwargs)

# Use rate limit information
response = client.agents.list()
remaining = response.headers.get('X-RateLimit-Remaining')
reset_time = response.headers.get('X-RateLimit-Reset')

if int(remaining) < 10:
    wait_until = datetime.fromtimestamp(int(reset_time))
    print(f"Rate limit low. Waiting until {wait_until}")
```

#### 3. Connection Issues

**Problem**: Network timeouts or connection errors

**Solutions**:
```javascript
// Implement connection retry
class ResilientClient {
    constructor(config) {
        this.config = {
            ...config,
            retryOptions: {
                retries: 3,
                minTimeout: 1000,
                maxTimeout: 5000,
                randomize: true
            }
        };
    }

    async makeRequest(endpoint, options) {
        const operation = retry.operation(this.config.retryOptions);

        return new Promise((resolve, reject) => {
            operation.attempt(async (currentAttempt) => {
                try {
                    const result = await this._request(endpoint, options);
                    resolve(result);
                } catch (error) {
                    if (operation.retry(error)) {
                        console.log(`Retry attempt ${currentAttempt}`);
                        return;
                    }
                    reject(operation.mainError());
                }
            });
        });
    }
}
```

#### 4. Large Response Handling

**Problem**: Memory issues with large responses

**Solutions**:
```python
# Stream large responses
async def stream_large_inference(client, agent_id, query):
    """Handle large inference responses with streaming."""

    # Use streaming endpoint
    async with client.inference.stream(
        agent_id=agent_id,
        query=query,
        stream=True
    ) as response:
        full_result = []

        async for chunk in response:
            # Process chunk immediately
            process_chunk(chunk)

            # Store if needed
            full_result.append(chunk)

            # Free memory periodically
            if len(full_result) > 100:
                await flush_to_storage(full_result)
                full_result = []

        return full_result
```

#### 5. Debugging WebSocket Connections

**Problem**: WebSocket disconnections or missed messages

**Solutions**:
```typescript
// Enhanced WebSocket debugging
class DebugWebSocket {
    constructor(url: string, token: string) {
        this.url = url;
        this.token = token;
        this.reconnectAttempts = 0;
    }

    connect() {
        console.log(`Connecting to ${this.url}`);

        this.ws = new WebSocket(`${this.url}?token=${this.token}`);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;

            // Send ping every 30 seconds
            this.pingInterval = setInterval(() => {
                if (this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: 'ping' }));
                }
            }, 30000);
        };

        this.ws.onmessage = (event) => {
            console.log('Received:', event.data);
            try {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            } catch (e) {
                console.error('Parse error:', e);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = (event) => {
            console.log(`WebSocket closed: ${event.code} - ${event.reason}`);
            clearInterval(this.pingInterval);

            // Reconnect with backoff
            if (this.reconnectAttempts < 5) {
                const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
                console.log(`Reconnecting in ${delay}ms`);
                setTimeout(() => this.connect(), delay);
                this.reconnectAttempts++;
            }
        };
    }
}
```

### Performance Optimization

#### 1. Connection Reuse

```python
# Global connection pool
from concurrent.futures import ThreadPoolExecutor
import threading

class ConnectionPool:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.clients = []
        self.executor = ThreadPoolExecutor(max_workers=10)

        # Pre-create clients
        for _ in range(5):
            client = FreeAgenticsClient(
                username=os.getenv('FREEAGENTICS_USER'),
                password=os.getenv('FREEAGENTICS_PASS')
            )
            self.clients.append(client)

    def get_client(self):
        """Get available client from pool."""
        return self.clients.pop() if self.clients else self._create_client()

    def return_client(self, client):
        """Return client to pool."""
        if len(self.clients) < 10:
            self.clients.append(client)
```

#### 2. Batch Request Optimization

```javascript
// Optimize batch requests
class BatchOptimizer {
    constructor(client, options = {}) {
        this.client = client;
        this.batchSize = options.batchSize || 10;
        this.queue = [];
        this.processing = false;
    }

    async addRequest(request) {
        return new Promise((resolve, reject) => {
            this.queue.push({ request, resolve, reject });

            if (!this.processing) {
                this.processQueue();
            }
        });
    }

    async processQueue() {
        this.processing = true;

        while (this.queue.length > 0) {
            // Take batch
            const batch = this.queue.splice(0, this.batchSize);

            try {
                // Process batch in parallel
                const results = await this.client.inference.batch(
                    batch.map(item => item.request),
                    { parallel: true }
                );

                // Resolve promises
                batch.forEach((item, index) => {
                    item.resolve(results[index]);
                });

            } catch (error) {
                // Reject all in batch
                batch.forEach(item => item.reject(error));
            }

            // Brief pause between batches
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        this.processing = false;
    }
}
```

### Monitoring and Debugging Tools

#### 1. Request Logger

```python
# Comprehensive request logging
import json
from datetime import datetime

class RequestLogger:
    def __init__(self, log_file='freeagentics_requests.log'):
        self.log_file = log_file

    def log_request(self, method, endpoint, data, response, duration):
        """Log API request details."""

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'endpoint': endpoint,
            'request_data': data,
            'response_status': response.status_code,
            'response_headers': dict(response.headers),
            'duration_ms': duration * 1000,
            'success': response.status_code < 400
        }

        # Log rate limit info
        if 'X-RateLimit-Remaining' in response.headers:
            log_entry['rate_limit'] = {
                'remaining': response.headers['X-RateLimit-Remaining'],
                'limit': response.headers.get('X-RateLimit-Limit'),
                'reset': response.headers.get('X-RateLimit-Reset')
            }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
```

#### 2. Performance Monitor

```typescript
// Performance monitoring
class PerformanceMonitor {
    private metrics: Map<string, number[]> = new Map();

    recordMetric(name: string, value: number): void {
        if (!this.metrics.has(name)) {
            this.metrics.set(name, []);
        }

        const values = this.metrics.get(name)!;
        values.push(value);

        // Keep last 1000 values
        if (values.length > 1000) {
            values.shift();
        }
    }

    getStats(name: string): object {
        const values = this.metrics.get(name) || [];

        if (values.length === 0) {
            return { count: 0 };
        }

        const sorted = [...values].sort((a, b) => a - b);

        return {
            count: values.length,
            min: sorted[0],
            max: sorted[sorted.length - 1],
            avg: values.reduce((a, b) => a + b) / values.length,
            p50: sorted[Math.floor(values.length * 0.5)],
            p95: sorted[Math.floor(values.length * 0.95)],
            p99: sorted[Math.floor(values.length * 0.99)]
        };
    }

    // Use with client
    async monitoredRequest(client: any, operation: string, fn: Function) {
        const start = Date.now();

        try {
            const result = await fn();
            const duration = Date.now() - start;

            this.recordMetric(`${operation}_duration`, duration);
            this.recordMetric(`${operation}_success`, 1);

            return result;
        } catch (error) {
            const duration = Date.now() - start;

            this.recordMetric(`${operation}_duration`, duration);
            this.recordMetric(`${operation}_error`, 1);

            throw error;
        }
    }
}
```

## Additional Resources

- **API Reference**: [Full API documentation](API_REFERENCE.md)
- **Example Repository**: https://github.com/freeagentics/examples
- **Community Forum**: https://community.freeagentics.com
- **Video Tutorials**: https://freeagentics.com/tutorials
- **Support**: support@freeagentics.com

## Updates and Changelog

Stay updated with the latest changes:

- **RSS Feed**: https://freeagentics.com/api/changelog.rss
- **Email Updates**: Subscribe at https://freeagentics.com/updates
- **Release Notes**: https://github.com/freeagentics/releases
