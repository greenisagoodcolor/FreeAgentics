# FreeAgentics API Documentation

Welcome to the FreeAgentics API documentation! This comprehensive guide will help you integrate with our multi-agent AI platform with Active Inference capabilities.

## 📚 Documentation Overview

This documentation includes:

- **[API Reference](API_REFERENCE.md)** - Complete API specification with all endpoints, authentication, and rate limiting
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Quick start guide, SDK usage, integration patterns, and best practices
- **[WebSocket API](WEBSOCKET_API.md)** - Real-time communication documentation with examples
- **[Code Examples](CODE_EXAMPLES.md)** - Comprehensive examples in multiple programming languages
- **[OpenAPI Specification](openapi.yaml)** - Machine-readable API specification
- **[Postman Collection](collections/FreeAgentics_API.postman_collection.json)** - Ready-to-use API collection
- **[Insomnia Collection](collections/FreeAgentics_API.insomnia_collection.json)** - Alternative API collection

## 🚀 Quick Start

### 1. Get Your API Credentials

1. Sign up at [https://freeagentics.com/signup](https://freeagentics.com/signup)
1. Verify your email address
1. Note your username and password for API authentication

### 2. Install SDK (Optional)

Choose your preferred language:

```bash
# Python
pip install freeagentics

# JavaScript/TypeScript
npm install @freeagentics/sdk

# Go
go get github.com/freeagentics/go-sdk
```

### 3. Make Your First API Call

```bash
# Login to get access token
curl -X POST https://api.freeagentics.com/api/v1/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'

# Create your first agent
curl -X POST https://api.freeagentics.com/api/v1/agents \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My First Agent",
    "template": "research_v2",
    "parameters": {"temperature": 0.7}
  }'

# Run inference
curl -X POST https://api.freeagentics.com/api/v1/inference \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "AGENT_ID",
    "query": "What are the latest AI trends?"
  }'
```

## 🔐 Authentication

FreeAgentics uses JWT tokens for authentication:

1. **Login** to get access and refresh tokens
1. **Include** access token in `Authorization: Bearer <token>` header
1. **Refresh** tokens when they expire (every 15 minutes)

```python
import requests

# Login
response = requests.post('https://api.freeagentics.com/api/v1/login',
                        json={'username': 'your_username', 'password': 'your_password'})
tokens = response.json()
access_token = tokens['access_token']

# Use token in subsequent requests
headers = {'Authorization': f'Bearer {access_token}'}
```

## 🚦 Rate Limits

| Environment | Auth Endpoints | API Endpoints | WebSocket |
|-------------|----------------|---------------|-----------|
| Production | 3/min | 60/min | 100/min |
| Development | 10/min | 200/min | 500/min |

Rate limit headers are included in all responses:

- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset timestamp

## 🧠 Core Concepts

### Agents

AI agents are the core processing units. Each agent:

- Has a specific template (research, analysis, conversation, etc.)
- Can be configured with custom parameters
- Supports Active Inference with PyMDP integration
- Maintains state and learning capabilities

### Inference

Inference is the process of running queries through agents:

- Submit queries with optional context
- Get real-time progress updates
- Receive structured results with confidence scores
- Support for batch processing

### Knowledge Graph

Build and query interconnected knowledge:

- Create entities and relationships
- Search semantic knowledge
- Build domain-specific knowledge bases
- Enable agent reasoning over structured data

## 🛠️ SDKs and Tools

### Official SDKs

- **Python**: Full-featured SDK with async support
- **JavaScript/TypeScript**: Modern SDK with Promise/async support
- **Go**: Concurrent SDK with proper error handling
- **Java**: Spring Boot integration ready
- **C#**: .NET Core compatible

### Development Tools

- **Postman Collection**: Pre-configured API requests
- **Insomnia Collection**: Alternative API client setup
- **OpenAPI Spec**: For code generation and documentation
- **WebSocket Client**: Real-time monitoring tools

## 🔌 Integration Patterns

### REST API

Standard HTTP API for basic operations:

- Agent management
- Inference execution
- Knowledge graph operations
- System monitoring

### WebSocket API

Real-time bidirectional communication:

- Live agent status updates
- Streaming inference results
- System metrics monitoring
- Interactive agent control

### GraphQL API

Flexible data querying:

- Custom data selection
- Nested resource fetching
- Real-time subscriptions
- Efficient bulk operations

### Webhooks

Event-driven notifications:

- Agent lifecycle events
- Inference completion alerts
- System health notifications
- Custom event routing

## 🏗️ Architecture Examples

### Microservices Integration

```python
# Flask microservice example
from flask import Flask, request, jsonify
from freeagentics import FreeAgenticsClient

app = Flask(__name__)
client = FreeAgenticsClient(username='service_user', password='service_pass')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    result = client.inference.run(
        agent_id='analysis_agent',
        query=data['query'],
        context=data.get('context', {})
    )
    return jsonify(result)
```

### Event-Driven Architecture

```javascript
// Node.js event processor
const { FreeAgenticsClient } = require('@freeagentics/sdk');
const client = new FreeAgenticsClient(config);

// WebSocket for real-time events
client.websocket.on('inference_completed', (data) => {
    // Process completed inference
    handleInferenceResult(data);
});

// Queue-based processing
processQueue.on('inference_request', async (request) => {
    const result = await client.inference.run(request);
    await publishResult(result);
});
```

### Batch Processing

```python
# Batch inference pipeline
import asyncio
from freeagentics import FreeAgenticsClient

async def process_batch(items):
    client = FreeAgenticsClient()

    # Process items concurrently
    tasks = [
        client.inference.run_async(
            agent_id='batch_agent',
            query=item['query'],
            context=item.get('context', {})
        )
        for item in items
    ]

    results = await asyncio.gather(*tasks)
    return results
```

## 📊 Monitoring and Observability

### System Metrics

- CPU and memory usage
- Agent performance metrics
- Inference throughput
- Error rates and latency

### Real-time Monitoring

```python
# WebSocket monitoring
async def monitor_system():
    async with client.websocket() as ws:
        await ws.subscribe(['system_metrics', 'agent_status'])

        async for message in ws:
            if message.type == 'system_metrics':
                update_dashboard(message.data)
            elif message.type == 'agent_status':
                log_agent_change(message.data)
```

### Health Checks

```bash
# System health endpoint
curl https://api.freeagentics.com/api/v1/system/status

# Agent-specific health
curl https://api.freeagentics.com/api/v1/agents/AGENT_ID/metrics
```

## 🔒 Security Best Practices

### Authentication Security

- Use secure token storage
- Implement token refresh logic
- Monitor for authentication failures
- Use client fingerprinting for enhanced security

### API Security

- Validate all inputs
- Implement rate limiting
- Use HTTPS for all communications
- Monitor for suspicious activity

### Data Protection

- Encrypt sensitive data
- Implement proper access controls
- Log security events
- Regular security audits

## 🚨 Error Handling

### Standard Error Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request was invalid",
    "details": {
      "field": "name",
      "reason": "Required field missing"
    }
  },
  "request_id": "req_12345"
}
```

### Common Error Codes

- `INVALID_REQUEST` (400): Request validation failed
- `UNAUTHORIZED` (401): Authentication required
- `FORBIDDEN` (403): Insufficient permissions
- `NOT_FOUND` (404): Resource not found
- `RATE_LIMITED` (429): Rate limit exceeded
- `INTERNAL_ERROR` (500): Server error

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def robust_api_call():
    return client.inference.run(agent_id='agent', query='test')
```

## 📈 Performance Optimization

### Connection Pooling

```python
# Reuse HTTP connections
import requests
from requests.adapters import HTTPAdapter

session = requests.Session()
adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
session.mount('https://', adapter)

client = FreeAgenticsClient(session=session)
```

### Batch Operations

```python
# Batch multiple inferences
results = client.inference.batch([
    {'agent_id': 'agent1', 'query': 'query1'},
    {'agent_id': 'agent2', 'query': 'query2'}
], parallel=True)
```

### Caching Strategies

```python
# Cache agent instances
from functools import lru_cache

@lru_cache(maxsize=100)
def get_agent(agent_id):
    return client.agents.get(agent_id)
```

## 🧪 Testing

### Unit Testing

```python
import unittest
from unittest.mock import Mock, patch

class TestFreeAgenticsIntegration(unittest.TestCase):
    def setUp(self):
        self.client = FreeAgenticsClient()

    @patch('freeagentics.client.requests.post')
    def test_create_agent(self, mock_post):
        mock_post.return_value.json.return_value = {'id': 'test_agent'}

        agent = self.client.agents.create(
            name='Test Agent',
            template='research_v2'
        )

        self.assertEqual(agent['id'], 'test_agent')
```

### Integration Testing

```python
# Test with real API
def test_full_inference_flow():
    client = FreeAgenticsClient()
    client.login('test_user', 'test_pass')

    # Create agent
    agent = client.agents.create(
        name='Integration Test Agent',
        template='research_v2'
    )

    # Run inference
    result = client.inference.run(
        agent_id=agent['id'],
        query='Test query'
    )

    # Verify result
    assert result['status'] == 'processing'
```

## 📖 Advanced Usage

### Custom Agent Templates

```python
# Create agent with custom GMN specification
gmn_spec = """
states: [analyzing, summarizing, idle]
observations: [text_input, context_data]
actions: [analyze, summarize, wait]
"""

agent = client.agents.create(
    name='Custom Agent',
    template='custom_v1',
    gmn_spec=gmn_spec,
    use_pymdp=True,
    planning_horizon=5
)
```

### Knowledge Graph Integration

```python
# Build domain knowledge
entity = client.knowledge.create_entity(
    type='concept',
    label='Machine Learning',
    properties={'domain': 'AI', 'complexity': 'high'}
)

# Connect to related concepts
client.knowledge.create_relationship(
    source_id=entity['id'],
    target_id='neural_networks_entity',
    relationship_type='includes',
    properties={'strength': 0.9}
)
```

### Multi-Agent Coordination

```python
# Coordinate multiple agents
async def coordinate_agents():
    # Create specialized agents
    analyst = client.agents.create(name='Analyst', template='analysis_v1')
    researcher = client.agents.create(name='Researcher', template='research_v2')

    # Parallel processing
    analysis_task = client.inference.run_async(
        agent_id=analyst['id'],
        query='Analyze data trends'
    )

    research_task = client.inference.run_async(
        agent_id=researcher['id'],
        query='Research market conditions'
    )

    # Combine results
    analysis_result = await analysis_task
    research_result = await research_task

    # Synthesize with coordinator agent
    coordinator = client.agents.create(name='Coordinator', template='synthesis_v1')
    final_result = await client.inference.run_async(
        agent_id=coordinator['id'],
        query='Synthesize findings',
        context={
            'analysis': analysis_result,
            'research': research_result
        }
    )

    return final_result
```

## 🔧 Troubleshooting

### Common Issues

#### Connection Problems

```python
# Check connectivity
try:
    response = requests.get('https://api.freeagentics.com/api/v1/system/status')
    print(f"API Status: {response.status_code}")
except requests.exceptions.ConnectionError:
    print("Cannot connect to API")
```

#### Authentication Issues

```python
# Debug authentication
if not client.is_authenticated():
    print("Authentication failed - check credentials")
    # Re-authenticate
    client.login(username, password)
```

#### Rate Limiting

```python
# Handle rate limits
try:
    result = client.inference.run(...)
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
    time.sleep(e.retry_after)
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Client debug mode
client = FreeAgenticsClient(debug=True)
```

## 📚 Resources

### Documentation

- [API Reference](API_REFERENCE.md) - Complete API specification
- [Developer Guide](DEVELOPER_GUIDE.md) - Integration guide and best practices
- [WebSocket API](WEBSOCKET_API.md) - Real-time API documentation
- [Code Examples](CODE_EXAMPLES.md) - Multi-language examples

### Tools

- [Postman Collection](collections/FreeAgentics_API.postman_collection.json)
- [Insomnia Collection](collections/FreeAgentics_API.insomnia_collection.json)
- [OpenAPI Specification](openapi.yaml)

### Community

- **GitHub**: [https://github.com/freeagentics](https://github.com/freeagentics)
- **Discord**: [https://discord.gg/freeagentics](https://discord.gg/freeagentics)
- **Forum**: [https://community.freeagentics.com](https://community.freeagentics.com)

### Support

- **Email**: support@freeagentics.com
- **Status Page**: https://status.freeagentics.com
- **Documentation**: https://docs.freeagentics.com

## 🗺️ Roadmap

### Coming Soon

- GraphQL subscriptions for real-time updates
- Enhanced agent templates and customization
- Advanced knowledge graph capabilities
- Multi-language agent support
- Federated learning capabilities

### Request Features

Have a feature request? Let us know:

- GitHub Issues: [https://github.com/freeagentics/api/issues](https://github.com/freeagentics/api/issues)
- Feature Requests: [https://freeagentics.com/feature-requests](https://freeagentics.com/feature-requests)

______________________________________________________________________

## 📄 License

This documentation is licensed under [MIT License](LICENSE).

## 🤝 Contributing

We welcome contributions to our documentation! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

______________________________________________________________________

**Ready to get started?** Check out our [Developer Guide](DEVELOPER_GUIDE.md) for detailed integration instructions and examples!
