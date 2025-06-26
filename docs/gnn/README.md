# GMN (Generative Model Notation) System

> **Important**: This documentation covers **Generative Model Notation (GMN)** for PyMDP mathematical models. For Graph Neural Network components, see the `inference/gnn/layers.py` module.

FreeAgentics uses **Generative Model Notation (GMN)** to define Active Inference agent models in a human-readable format. GMN provides a bridge between natural language specifications and mathematical PyMDP implementations.

## Overview

The GMN system consists of several key components:

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│              │  │              │  │              │  │              │
│ GMN Parser   │  │ Feature      │  │ Edge Processor   │  │
│              │  │ Extractor    │  │              │  │ Model Mapper │
│ Parses .gmn  │  │ Extracts     │  │ Processes    │  │ Maps to      │
│ files into   │  │ node/edge    │  │ relationships│  │ PyMDP models │
│ structured   │  │ features     │  │ between      │  │ for Active   │
│ format       │  │ from graphs  │  │ components   │  │ Inference    │
│              │  │              │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

## Quick Start

```python
from inference.gnn.parser import GMNParser
from inference.gnn.executor import GMNExecutor
from inference.gnn.validator import GMNValidator

# Parse a mathematical model definition
parser = GMNParser()
model_def = parser.parse_file("models/base/explorer.gmn.md")

# Validate the mathematical model
validator = GMNValidator()
validation_result = validator.validate(model_def)

# Execute Active Inference
executor = GMNExecutor()
result = executor.execute_inference(model_def, observation)
```

## Components

### 1. GMN Parser

The parser reads and validates `.gmn.md` model definition files:

```python
from inference.gnn.parser import GMNParser

parser = GMNParser()
model = parser.parse_file("path/to/model.gmn.md")
```

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Getting Started](#getting-started)
3. [Core Components](#core-components)
4. [API Reference](../api/gnn-api.md)
5. [Configuration Guide](#configuration-guide)
6. [Examples](#examples)
7. [Performance Guidelines](#performance-guidelines)
8. [Monitoring & Operations](#monitoring--operations)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Architecture Overview

The GNN Processing Core consists of several interconnected components:

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ REST API    │  │ Job Manager  │  │ Authentication   │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Processing Core                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ GNN Parser  │  │ Feature      │  │ Edge Processor   │  │
│  │             │  │ Extractor    │  │                  │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ GNN Layers  │  │ Model Mapper │  │ Batch Processor  │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 Infrastructure Layer                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Monitoring  │  │ Metrics      │  │ Alerting         │  │
│  │             │  │ Collector    │  │                  │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

- **Multiple GNN Architectures**: Support for GCN, GAT, GraphSAGE, GIN, and EdgeConv
- **Automatic Model Selection**: Intelligent selection based on graph properties
- **Batch Processing**: Efficient handling of multiple graphs
- **Real-time Monitoring**: Comprehensive metrics and alerting
- **Production Ready**: Rate limiting, authentication, and error handling

## Getting Started

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install additional dependencies for monitoring
pip install flask flask-cors matplotlib
```

### Quick Start

```python
from inference.gnn.parser import GNNParser
from inference.gnn.feature_extractor import NodeFeatureExtractor
from inference.gnn.layers import GNNStack
from inference.gnn.batch_processor import GraphBatchProcessor

# Parse a GNN model definition
parser = GNNParser()
model_def = parser.parse_file("models/base/explorer.gmn.md")

# Create feature extractor
extractor = NodeFeatureExtractor()

# Build model
model = GNNStack(
    input_dim=model_def.metadata.get('input_dim', 32),
    hidden_dims=[64, 32],
    output_dim=model_def.metadata.get('output_dim', 16),
    architecture='gcn'
)

# Process graphs in batches
processor = GraphBatchProcessor()
batched_data = processor.create_batch(graphs)
```

## Core Components

### 1. GNN Parser

The parser reads and validates `.gmn.md` model definition files:

```python
from inference.gnn.parser import GMNParser

parser = GMNParser()
model = parser.parse_file("path/to/model.gmn.md")

# Access model components
print(model.metadata)
print(model.behavior)
print(model.perception)
```

### 2. Feature Extractor

Extracts and normalizes node features from various data types:

```python
from inference.gnn.feature_extractor import NodeFeatureExtractor

extractor = NodeFeatureExtractor()

# Configure feature types
config = {
    'spatial_features': ['lat', 'lon'],
    'categorical_features': ['type', 'category'],
    'numerical_features': ['value', 'score'],
    'normalization': 'standard'
}

# Extract features
result = extractor.extract_features(nodes, config)
features = result.features  # torch.Tensor
```

### 3. Edge Processor

Handles edge features and graph connectivity:

```python
from inference.gnn.edge_processor import EdgeProcessor

processor = EdgeProcessor()

# Process edges
edge_batch = processor.create_edge_batch(edges)
edge_index = processor.to_edge_index(edges)
```

### 4. GNN Layers

Multiple GNN architectures with a unified interface:

```python
from inference.gnn.layers import GNNStack

# Create a GNN model
model = GNNStack(
    input_dim=32,
    hidden_dims=[64, 64, 32],
    output_dim=16,
    architecture='gat',  # 'gcn', 'sage', 'gin', 'edgeconv'
    dropout=0.5,
    num_heads=4  # For GAT
)

# Forward pass
output = model(x, edge_index)
```

### 5. Model Mapper

Automatically selects the best architecture:

```python
from inference.gnn.model_mapper import GraphToModelMapper

mapper = GraphToModelMapper()

# Map graph to model
mapping_result = mapper.map_graph_to_model(
    graph_data,
    task_type='node_classification'
)

print(f"Selected architecture: {mapping_result['architecture']}")
print(f"Recommended config: {mapping_result['config']}")
```

### 6. Batch Processor

Efficient batch processing for multiple graphs:

```python
from inference.gnn.batch_processor import GraphBatchProcessor

processor = GraphBatchProcessor()

# Create batches
batches = processor.create_batch(
    graphs,
    batch_size=32,
    shuffle=True
)

# Process batch
for batch in batches:
    output = model(batch.x, batch.edge_index, batch.batch)
```

## Configuration Guide

### Environment Variables

```bash
# Core settings
GNN_MAX_BATCH_SIZE=64
GNN_DEFAULT_ARCHITECTURE=auto
GNN_DEVICE=cuda  # or cpu

# Performance settings
GNN_NUM_WORKERS=4
GNN_PREFETCH_FACTOR=2
GNN_PIN_MEMORY=true

# Monitoring settings
GNN_METRICS_ENABLED=true
GNN_METRICS_DB_PATH=/var/lib/gnn/metrics.db
GNN_ALERT_EMAIL=admin@example.com
```

### Model Configuration

Models can be configured via the API or configuration files:

```json
{
  "architecture": "gat",
  "config": {
    "hidden_dims": [128, 64, 32],
    "num_heads": 4,
    "dropout": 0.6,
    "attention_dropout": 0.6,
    "activation": "elu"
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 5e-4,
    "epochs": 200,
    "early_stopping_patience": 20
  }
}
```

## Examples

### Example 1: Node Classification

```python
import torch
from inference.gnn.layers import GNNStack
from inference.gnn.feature_extractor import NodeFeatureExtractor

# Prepare data
nodes = [
    {'id': 0, 'features': {'degree': 3, 'pagerank': 0.15}},
    {'id': 1, 'features': {'degree': 2, 'pagerank': 0.10}},
    # ... more nodes
]

edges = [(0, 1), (1, 2), (2, 0)]

# Extract features
extractor = NodeFeatureExtractor()
features = extractor.extract_features(nodes, {
    'numerical_features': ['degree', 'pagerank']
})

# Create model
model = GNNStack(
    input_dim=2,
    hidden_dims=[16, 8],
    output_dim=3,  # 3 classes
    architecture='gcn'
)

# Forward pass
edge_index = torch.tensor(edges, dtype=torch.long).t()
output = model(features.features, edge_index)
predictions = output.argmax(dim=1)
```

### Example 2: Graph Classification

```python
from inference.gnn.batch_processor import GraphBatchProcessor, GraphData

# Create graph data
graphs = []
for i in range(10):
    graph = GraphData(
        node_features=torch.randn(20, 32),
        edge_index=torch.randint(0, 20, (2, 40)),
        graph_label=torch.tensor([i % 2])  # Binary classification
    )
    graphs.append(graph)

# Batch processing
processor = GraphBatchProcessor()
batch = processor.create_batch(graphs)

# Model with global pooling
model = GNNStack(
    input_dim=32,
    hidden_dims=[64, 32],
    output_dim=2,
    architecture='gin',
    global_pool='mean'
)

# Forward pass
output = model(batch.x, batch.edge_index, batch.batch)
```

### Example 3: Using the API

```python
import requests

# Process a graph
response = requests.post(
    'http://localhost:5000/api/gnn/process',
    headers={'x-api-key': 'your-api-key'},
    json={
        'graph': {
            'nodes': [
                {'id': 'A', 'features': {'value': 1.0}},
                {'id': 'B', 'features': {'value': 2.0}}
            ],
            'edges': [
                {'source': 'A', 'target': 'B', 'weight': 0.5}
            ]
        },
        'model': {
            'architecture': 'gcn',
            'task': 'node_classification'
        }
    }
)

job_id = response.json()['jobId']

# Check status
status = requests.get(
    f'http://localhost:5000/api/gnn/jobs/{job_id}',
    headers={'x-api-key': 'your-api-key'}
)

# Get results
results = requests.get(
    f'http://localhost:5000/api/gnn/jobs/{job_id}/results',
    headers={'x-api-key': 'your-api-key'}
)
```

## Performance Guidelines

### Graph Size Recommendations

| Graph Size     | Recommended Architecture | Batch Size | Notes                           |
| -------------- | ------------------------ | ---------- | ------------------------------- |
| < 1K nodes     | GCN, GAT                 | 64-128     | Fast processing                 |
| 1K-10K nodes   | GraphSAGE                | 16-32      | Use sampling                    |
| 10K-100K nodes | GraphSAGE with sampling  | 4-8        | Increase sampling neighbors     |
| > 100K nodes   | Custom sampling strategy | 1-4        | Consider distributed processing |

### Memory Optimization

1. **Use Sparse Tensors**: For graphs with low density
2. **Enable Gradient Checkpointing**: For deep models
3. **Batch Processing**: Process multiple small graphs together
4. **Feature Dimensionality**: Keep features compact

### Performance Tuning

```python
# Optimize batch processing
processor = GraphBatchProcessor(
    max_nodes_per_batch=10000,  # Limit total nodes
    padding_strategy='minimal',   # Reduce padding overhead
    device='cuda'
)

# Use mixed precision training
from torch.cuda.amp import autocast

with autocast():
    output = model(x, edge_index)
```

## Monitoring & Operations

### Starting the Monitoring Dashboard

```python
from inference.gnn.monitoring_dashboard import create_dashboard

# Create and start dashboard
dashboard = create_dashboard(host='0.0.0.0', port=5000)
dashboard.start()

# Access at http://localhost:5000
```

### Setting Up Alerts

```python
from inference.gnn.alerting import get_alert_manager, AlertRule, AlertType, AlertSeverity

manager = get_alert_manager()

# Add custom rule
rule = AlertRule(
    name="high_gpu_memory",
    alert_type=AlertType.RESOURCE_USAGE,
    metric="gpu_memory_mb",
    threshold=14000,  # 14GB
    comparison="gt",
    severity=AlertSeverity.WARNING
)

manager.add_rule(rule)
manager.start()
```

### Metrics Collection

```python
from inference.gnn.metrics_collector import get_metrics_collector, GraphMetrics

collector = get_metrics_collector()

# Collect metrics
metrics = GraphMetrics(
    graph_id="graph_001",
    num_nodes=1000,
    num_edges=5000,
    avg_degree=10.0,
    density=0.01,
    processing_time=1.23,
    model_architecture="GCN",
    task_type="node_classification",
    success=True
)

collector.collect_graph_metrics(metrics)
```

## Best Practices

### 1. Model Selection

- Use `auto` architecture for automatic selection
- Consider graph properties:
  - Dense graphs → GCN
  - Heterophilic graphs → GAT
  - Large graphs → GraphSAGE
  - Graph-level tasks → GIN

### 2. Feature Engineering

- Normalize numerical features
- Use one-hot encoding for small categorical sets
- Consider graph structural features (degree, centrality)
- Handle missing values appropriately

### 3. Error Handling

```python
from inference.gnn.monitoring import monitor_performance, log_operation

@monitor_performance("critical_operation")
@log_operation("critical_operation")
def process_important_graph(graph):
    try:
        # Process graph
        result = model(graph)
        return result
    except Exception as e:
        logger.error(f"Failed to process graph: {e}")
        # Graceful degradation
        return fallback_result
```

### 4. Production Deployment

1. **Use Environment Variables**: For configuration
2. **Enable Monitoring**: Always monitor in production
3. **Set Up Alerts**: For critical thresholds
4. **Implement Rate Limiting**: Protect your API
5. **Use Caching**: For repeated computations
6. **Plan for Scaling**: Design for horizontal scaling

## Troubleshooting

### Common Issues

#### Out of Memory Errors

```python
# Reduce batch size
processor = GraphBatchProcessor(max_nodes_per_batch=5000)

# Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(batches):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### Slow Processing

1. Check if GPU is being utilized
2. Profile the code to find bottlenecks
3. Use batch processing for small graphs
4. Enable graph sampling for large graphs

#### Poor Model Performance

1. Check feature normalization
2. Verify graph connectivity
3. Try different architectures
4. Adjust hyperparameters
5. Increase model capacity

### Debug Mode

Enable debug logging:

```python
import logging
from inference.gnn.monitoring import get_logger

logger = get_logger(level=logging.DEBUG)
```

### Getting Help

1. Check the [API documentation](../api/gnn-api.md)
2. Review example notebooks in `examples/`
3. Check monitoring dashboard for system health
4. Review logs in `.logs/` directory

## Advanced Topics

### Custom GNN Layers

```python
from inference.gnn.layers import MessagePassing

class CustomGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Custom message passing logic
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # Define how messages are computed
        return self.lin(x_j)
```

### Distributed Processing

For very large graphs, consider distributed processing:

```python
# Using PyTorch Distributed
import torch.distributed as dist

# Initialize process group
dist.init_process_group(backend='nccl')

# Distribute graph across GPUs
# Implementation depends on specific requirements
```

### Integration with Other Systems

The GNN Processing Core can be integrated with:

1. **Apache Kafka**: For streaming graph data
2. **Redis**: For caching processed results
3. **PostgreSQL**: For storing graph metadata
4. **Elasticsearch**: For graph search capabilities

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on contributing to the GNN Processing Core.

## License

This project is licensed under the MIT License. See [LICENSE.md](../../LICENSE.md) for details.
