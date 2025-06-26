# Getting Started with GNN Processing Core

This tutorial will walk you through the basics of using the GNN Processing Core for graph neural network tasks.

## Prerequisites

- Python 3.8 or higher
- PyTorch 1.12 or higher
- Basic understanding of graph theory and neural networks

## Installation

```bash
# Clone the repository
git clone https://github.com/freeagentics/freeagentics.git
cd freeagentics

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies for visualization
pip install matplotlib networkx
```

## Tutorial 1: Your First Graph Neural Network

Let's start with a simple example of node classification on a small graph.

### Step 1: Create a Simple Graph

```python
import torch
import numpy as np
from inference.gnn.feature_extractor import NodeFeatureExtractor
from inference.gnn.edge_processor import EdgeProcessor
from inference.gnn.layers import GNNStack

# Define nodes with features
nodes = [
    {'id': 0, 'features': {'degree': 3, 'type': 'hub'}},
    {'id': 1, 'features': {'degree': 2, 'type': 'regular'}},
    {'id': 2, 'features': {'degree': 2, 'type': 'regular'}},
    {'id': 3, 'features': {'degree': 1, 'type': 'leaf'}},
]

# Define edges (undirected)
edges = [
    (0, 1), (0, 2), (0, 3),  # Node 0 connects to all others
    (1, 2),  # Node 1 connects to node 2
]

print(f"Created graph with {len(nodes)} nodes and {len(edges)} edges")
```

### Step 2: Extract Features

```python
# Configure feature extraction
feature_config = {
    'numerical_features': ['degree'],
    'categorical_features': ['type'],
    'normalization': 'standard'
}

# Extract features
extractor = NodeFeatureExtractor()
extraction_result = extractor.extract_features(nodes, feature_config)

print(f"Feature shape: {extraction_result.features.shape}")
print(f"Feature names: {extraction_result.feature_names}")
```

### Step 3: Process Edges

```python
# Convert edges to PyTorch format
processor = EdgeProcessor()
edge_index = processor.to_edge_index(edges)

print(f"Edge index shape: {edge_index.shape}")
print(f"Edge index:\n{edge_index}")
```

### Step 4: Create and Run GNN Model

```python
# Create GNN model
model = GNNStack(
    input_dim=extraction_result.features.shape[1],
    hidden_dims=[16, 8],
    output_dim=2,  # Binary classification
    architecture='gcn',
    dropout=0.5
)

# Forward pass
with torch.no_grad():
    output = model(extraction_result.features, edge_index)
    predictions = output.argmax(dim=1)

print(f"Output shape: {output.shape}")
print(f"Predictions: {predictions}")
```

### Step 5: Visualize the Graph (Optional)

```python
import matplotlib.pyplot as plt
import networkx as nx

# Create NetworkX graph for visualization
G = nx.Graph()
G.add_nodes_from(range(len(nodes)))
G.add_edges_from(edges)

# Plot
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue',
        node_size=1000, font_size=16, font_weight='bold')
plt.title("Example Graph")
plt.show()
```

## Tutorial 2: Batch Processing Multiple Graphs

When working with multiple graphs, batch processing is essential for efficiency.

### Step 1: Generate Multiple Graphs

```python
from inference.gnn.batch_processor import GraphData, GraphBatchProcessor
import torch

# Generate 10 random graphs
graphs = []
for i in range(10):
    num_nodes = np.random.randint(5, 15)
    num_edges = np.random.randint(num_nodes, num_nodes * 2)

    # Random features
    node_features = torch.randn(num_nodes, 8)

    # Random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Create GraphData object
    graph = GraphData(
        node_features=node_features,
        edge_index=edge_index,
        graph_label=torch.tensor([i % 3])  # 3-class classification
    )
    graphs.append(graph)

print(f"Created {len(graphs)} graphs")
```

### Step 2: Create Batches

```python
# Initialize batch processor
processor = GraphBatchProcessor()

# Create a single batch
batch = processor.create_batch(graphs, batch_size=5)

print(f"Batch contains {batch.num_graphs} graphs")
print(f"Total nodes in batch: {batch.x.shape[0]}")
print(f"Total edges in batch: {batch.edge_index.shape[1]}")
```

### Step 3: Process Batch with GNN

```python
# Create model for graph classification
model = GNNStack(
    input_dim=8,
    hidden_dims=[32, 16],
    output_dim=3,  # 3 classes
    architecture='gin',  # GIN is good for graph-level tasks
    global_pool='mean'   # Global pooling for graph classification
)

# Forward pass
with torch.no_grad():
    # The batch tensor indicates which node belongs to which graph
    output = model(batch.x, batch.edge_index, batch.batch)
    predictions = output.argmax(dim=1)

print(f"Predictions for {len(predictions)} graphs: {predictions}")
```

## Tutorial 3: Using Different GNN Architectures

Let's explore different GNN architectures and their characteristics.

### Graph Convolutional Network (GCN)

```python
# GCN - Simple and effective for many tasks
gcn_model = GNNStack(
    input_dim=32,
    hidden_dims=[64, 32],
    output_dim=10,
    architecture='gcn',
    dropout=0.5
)

print("GCN Model:")
print(f"  Parameters: {sum(p.numel() for p in gcn_model.parameters())}")
```

### Graph Attention Network (GAT)

```python
# GAT - Uses attention mechanism
gat_model = GNNStack(
    input_dim=32,
    hidden_dims=[64, 32],
    output_dim=10,
    architecture='gat',
    dropout=0.6,
    num_heads=4,  # Multi-head attention
    attention_dropout=0.6
)

print("GAT Model:")
print(f"  Parameters: {sum(p.numel() for p in gat_model.parameters())}")
print(f"  Attention heads: 4")
```

### GraphSAGE

```python
# GraphSAGE - Inductive learning with sampling
sage_model = GNNStack(
    input_dim=32,
    hidden_dims=[64, 32],
    output_dim=10,
    architecture='sage',
    dropout=0.5,
    aggregation='mean'  # Can be 'mean', 'max', 'sum'
)

print("GraphSAGE Model:")
print(f"  Parameters: {sum(p.numel() for p in sage_model.parameters())}")
print(f"  Aggregation: mean")
```

### Comparing Architectures

```python
# Create sample data
x = torch.randn(100, 32)
edge_index = torch.randint(0, 100, (2, 300))

# Time each architecture
import time

for name, model in [('GCN', gcn_model), ('GAT', gat_model), ('SAGE', sage_model)]:
    start = time.time()
    with torch.no_grad():
        _ = model(x, edge_index)
    end = time.time()
    print(f"{name} inference time: {(end - start)*1000:.2f} ms")
```

## Tutorial 4: Automatic Model Selection

The GNN Processing Core can automatically select the best architecture based on your graph.

```python
from inference.gnn.model_mapper import GraphToModelMapper

# Create mapper
mapper = GraphToModelMapper()

# Analyze a graph
graph_data = {
    'num_nodes': 1000,
    'num_edges': 5000,
    'features': torch.randn(1000, 32),
    'edge_index': torch.randint(0, 1000, (2, 5000))
}

# Get recommendation
result = mapper.map_graph_to_model(
    graph_data,
    task_type='node_classification'
)

print(f"Recommended architecture: {result['architecture']}")
print(f"Reason: {result['selection_reason']}")
print(f"Suggested config: {result['config']}")

# Create model based on recommendation
recommended_model = GNNStack(
    input_dim=32,
    output_dim=10,
    architecture=result['architecture'],
    **result['config']
)
```

## Tutorial 5: Monitoring and Performance

### Setting Up Monitoring

```python
from inference.gnn.monitoring import get_monitor, monitor_performance
from inference.gnn.metrics_collector import get_metrics_collector

# Get monitoring instances
monitor = get_monitor()
collector = get_metrics_collector()

# Decorate your processing function
@monitor_performance("graph_processing")
def process_graph(graph_data):
    # Your processing logic
    model = GNNStack(input_dim=32, hidden_dims=[64], output_dim=10)
    output = model(graph_data.x, graph_data.edge_index)
    return output

# Process with monitoring
result = process_graph(batch)

# Get performance statistics
stats = monitor.get_statistics("graph_processing")
print(f"Average processing time: {stats['duration']['mean']:.3f} seconds")
```

### Starting the Dashboard

```python
from inference.gnn.monitoring_dashboard import create_dashboard

# Create and start dashboard
dashboard = create_dashboard(port=5000)
dashboard.start()

print("Monitoring dashboard available at http://localhost:5000")
```

## Tutorial 6: Using the REST API

### Starting the API Server

```bash
# In your terminal
python -m flask run --host=0.0.0.0 --port=8080
```

### Making API Requests

```python
import requests
import json

# Prepare graph data
graph_data = {
    'graph': {
        'nodes': [
            {'id': 'A', 'features': {'value': 1.0, 'type': 'source'}},
            {'id': 'B', 'features': {'value': 2.0, 'type': 'intermediate'}},
            {'id': 'C', 'features': {'value': 3.0, 'type': 'target'}}
        ],
        'edges': [
            {'source': 'A', 'target': 'B', 'weight': 0.5},
            {'source': 'B', 'target': 'C', 'weight': 0.8}
        ]
    },
    'model': {
        'architecture': 'auto',  # Let the system choose
        'task': 'node_classification'
    }
}

# Send request
response = requests.post(
    'http://localhost:8080/api/gnn/process',
    headers={'Content-Type': 'application/json'},
    json=graph_data
)

result = response.json()
print(f"Job ID: {result['jobId']}")
print(f"Status URL: {result['links']['status']}")
```

## Best Practices

### 1. Feature Preprocessing

Always normalize your features:

```python
config = {
    'numerical_features': ['value1', 'value2'],
    'normalization': 'standard',  # or 'minmax', 'robust'
    'handle_missing': 'mean'      # or 'median', 'drop'
}
```

### 2. Handling Large Graphs

For graphs with >10k nodes:

```python
# Use GraphSAGE with sampling
model = GNNStack(
    architecture='sage',
    neighbor_samples=[25, 10],  # Sample 25 neighbors in layer 1, 10 in layer 2
    aggregation='mean'
)

# Use smaller batch sizes
processor = GraphBatchProcessor(
    max_nodes_per_batch=5000
)
```

### 3. Memory Management

```python
# Enable gradient checkpointing for deep models
model = GNNStack(
    hidden_dims=[128, 128, 128, 128],  # Deep model
    checkpoint_gradients=True  # Trade compute for memory
)

# Clear cache periodically
import torch
torch.cuda.empty_cache()
```

## Common Pitfalls and Solutions

### Issue: Out of Memory

```python
# Solution 1: Reduce batch size
smaller_batch = processor.create_batch(graphs, batch_size=1)

# Solution 2: Use CPU for large graphs
model = model.cpu()
data = data.cpu()

# Solution 3: Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(x, edge_index)
```

### Issue: Poor Performance

```python
# Solution 1: Check if features are normalized
print(f"Feature mean: {features.mean()}")
print(f"Feature std: {features.std()}")

# Solution 2: Try different architectures
architectures = ['gcn', 'gat', 'sage', 'gin']
for arch in architectures:
    model = GNNStack(architecture=arch, ...)
    # Evaluate performance
```

### Issue: Disconnected Graphs

```python
# Check graph connectivity
from inference.gnn.edge_processor import EdgeProcessor

processor = EdgeProcessor()
is_connected = processor.check_connectivity(edge_index, num_nodes)

if not is_connected:
    # Add self-loops
    edge_index = processor.add_self_loops(edge_index, num_nodes)
```

## Next Steps

1. Explore the [API Documentation](../../api/gnn-api.md)
2. Check out [Advanced Examples](./advanced-examples.md)
3. Learn about [Model Optimization](./optimization-guide.md)
4. Read about [Production Deployment](./deployment-guide.md)

## Resources

- [GNN Model Format Specification](../../gnn-model-format.md)
- [Performance Benchmarks](../benchmarks.md)
- [Troubleshooting Guide](../troubleshooting.md)
- [Contributing Guidelines](../../../CONTRIBUTING.md)
