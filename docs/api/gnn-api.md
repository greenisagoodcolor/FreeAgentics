# GNN Processing Core API Documentation

## Overview

The GNN Processing Core API provides RESTful endpoints for processing graphs using various Graph Neural Network architectures. The API supports node classification, graph classification, and link prediction tasks.

## Base URL

```
https://api.freeagentics.ai/api/gnn
```

## Authentication

The API requires authentication using either:

- **API Key**: Include in the `x-api-key` header
- **Session**: Valid user session (for web app users)

```bash
curl -H "x-api-key: YOUR_API_KEY" https://api.freeagentics.ai/api/gnn/models
```

## Rate Limiting

- **Default**: 10 requests per minute per API key/IP
- **Premium**: 100 requests per minute (contact support)

Rate limit headers:

- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset timestamp

## Endpoints

### 1. List Available Models

Get information about available GNN architectures and their parameters.

**Endpoint**: `GET /api/gnn/models`

**Response**:

```json
{
  "models": [
    {
      "id": "gcn",
      "name": "Graph Convolutional Network (GCN)",
      "description": "Classic GNN architecture...",
      "tasks": ["node_classification", "graph_classification"],
      "parameters": {
        "hidden_dims": {
          "type": "array",
          "description": "Hidden layer dimensions",
          "default": [64, 32]
        }
      }
    }
  ],
  "taskDescriptions": {
    "node_classification": {
      "name": "Node Classification",
      "description": "Predict labels for individual nodes"
    }
  }
}
```

### 2. Process Graph

Submit a graph for processing with a specified model and task.

**Endpoint**: `POST /api/gnn/process`

**Request Body**:

```json
{
  "graph": {
    "nodes": [
      {
        "id": "node1",
        "features": {
          "degree": 3,
          "pagerank": 0.15,
          "category": "A"
        },
        "position": {
          "lat": 40.7128,
          "lon": -74.006
        }
      }
    ],
    "edges": [
      {
        "source": "node1",
        "target": "node2",
        "weight": 0.8,
        "type": "connection"
      }
    ]
  },
  "model": {
    "architecture": "gcn",
    "task": "node_classification",
    "config": {
      "hidden_dims": [128, 64],
      "dropout": 0.5
    }
  },
  "options": {
    "batch_size": 32,
    "return_embeddings": true,
    "return_attention_weights": false
  }
}
```

**Response**:

```json
{
  "success": true,
  "jobId": "job_123456",
  "status": "queued",
  "message": "Graph processing initiated successfully",
  "estimatedTime": 30,
  "links": {
    "status": "/api/gnn/jobs/job_123456",
    "results": "/api/gnn/jobs/job_123456/results"
  }
}
```

### 3. Check Job Status

Monitor the status of a processing job.

**Endpoint**: `GET /api/gnn/jobs/{jobId}`

**Response**:

```json
{
  "jobId": "job_123456",
  "status": "processing",
  "progress": 0.45,
  "createdAt": "2024-01-18T10:00:00Z",
  "updatedAt": "2024-01-18T10:01:30Z",
  "metadata": {
    "graphNodes": 100,
    "graphEdges": 250,
    "modelArchitecture": "gcn",
    "task": "node_classification"
  },
  "links": {
    "self": "/api/gnn/jobs/job_123456",
    "results": null,
    "cancel": "/api/gnn/jobs/job_123456"
  }
}
```

**Status Values**:

- `queued`: Job is waiting to be processed
- `processing`: Job is currently being processed
- `completed`: Job finished successfully
- `failed`: Job failed with error
- `cancelled`: Job was cancelled

### 4. Get Job Results

Retrieve the results of a completed job.

**Endpoint**: `GET /api/gnn/jobs/{jobId}/results`

**Response** (Node Classification):

```json
{
  "jobId": "job_123456",
  "status": "completed",
  "task": "node_classification",
  "model": {
    "architecture": "gcn",
    "config": {
      "hidden_dims": [128, 64],
      "dropout": 0.5
    }
  },
  "results": {
    "predictions": {
      "nodes": ["node1", "node2", "node3"],
      "classes": [0, 1, 0],
      "probabilities": [
        [0.9, 0.1],
        [0.2, 0.8],
        [0.85, 0.15]
      ]
    },
    "embeddings": {
      "node1": [0.12, -0.34, ...],
      "node2": [0.45, 0.23, ...]
    },
    "metrics": {
      "accuracy": 0.92,
      "precision": 0.89,
      "recall": 0.91,
      "f1Score": 0.90
    }
  },
  "metadata": {
    "processingTime": 28.5,
    "graphStats": {
      "nodes": 100,
      "edges": 250
    },
    "timestamp": "2024-01-18T10:02:00Z"
  }
}
```

### 5. Cancel Job

Cancel a queued or processing job.

**Endpoint**: `DELETE /api/gnn/jobs/{jobId}`

**Response**:

```json
{
  "success": true,
  "message": "Job job_123456 cancelled successfully",
  "jobId": "job_123456"
}
```

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "Error type",
  "message": "Human-readable error message",
  "details": {
    "field": "Additional error context"
  }
}
```

**Common Error Codes**:

- `400`: Bad Request - Invalid input data
- `401`: Unauthorized - Missing or invalid authentication
- `404`: Not Found - Resource doesn't exist
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error

## Examples

### Example 1: Node Classification on Social Network

```bash
curl -X POST https://api.freeagentics.ai/api/gnn/process \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "graph": {
      "nodes": [
        {"id": "user1", "features": {"followers": 100, "posts": 50}},
        {"id": "user2", "features": {"followers": 200, "posts": 30}},
        {"id": "user3", "features": {"followers": 50, "posts": 100}}
      ],
      "edges": [
        {"source": "user1", "target": "user2"},
        {"source": "user2", "target": "user3"}
      ]
    },
    "model": {
      "architecture": "gat",
      "task": "node_classification"
    }
  }'
```

### Example 2: Graph Classification for Molecules

```bash
curl -X POST https://api.freeagentics.ai/api/gnn/process \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "graph": {
      "nodes": [
        {"id": "C1", "features": {"element": "C", "charge": 0}},
        {"id": "O1", "features": {"element": "O", "charge": -0.5}},
        {"id": "H1", "features": {"element": "H", "charge": 0.1}}
      ],
      "edges": [
        {"source": "C1", "target": "O1", "type": "double"},
        {"source": "C1", "target": "H1", "type": "single"}
      ]
    },
    "model": {
      "architecture": "gin",
      "task": "graph_classification",
      "config": {
        "hidden_dims": [64, 64, 32]
      }
    }
  }'
```

### Example 3: Link Prediction

```bash
curl -X POST https://api.freeagentics.ai/api/gnn/process \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "graph": {
      "nodes": [
        {"id": "A", "features": {"type": 1}},
        {"id": "B", "features": {"type": 2}},
        {"id": "C", "features": {"type": 1}},
        {"id": "D", "features": {"type": 3}}
      ],
      "edges": [
        {"source": "A", "target": "B"},
        {"source": "B", "target": "C"}
      ]
    },
    "model": {
      "architecture": "sage",
      "task": "link_prediction",
      "config": {
        "aggregation": "mean"
      }
    }
  }'
```

## Best Practices

1. **Graph Size**: For graphs with >10,000 nodes, consider using GraphSAGE with sampling
2. **Feature Normalization**: Normalize numerical features before submission
3. **Batch Processing**: Use appropriate batch sizes based on graph size
4. **Model Selection**: Use "auto" architecture for automatic model selection
5. **Error Handling**: Implement exponential backoff for retries
6. **Job Monitoring**: Poll status endpoint every 5-10 seconds

## SDK Support

Official SDKs available for:

- Python: `pip install freeagentics-gnn`
- JavaScript/TypeScript: `npm install @freeagentics/gnn`
- Java: Maven package available
- Go: `go get github.com/freeagentics/gnn-go`

## Changelog

### v1.0.0 (2024-01-18)

- Initial release with GCN, GAT, GraphSAGE, and GIN support
- Support for node classification, graph classification, and link prediction
- Asynchronous job processing
- Comprehensive error handling and rate limiting
