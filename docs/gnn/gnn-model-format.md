# GMN Model Format Specification

The `.gmn.md` (Generative Model Notation Markdown) format is a human-readable specification language for defining PyMDP Active Inference models in FreeAgentics. It combines the expressiveness of natural language with mathematical precision for agent behavior modeling.

## Structure

A `.gmn.md` file consists of several sections, each serving a specific purpose in defining the mathematical model:

## File Structure

A `.gmn.md` file consists of several sections, each serving a specific purpose in defining the GNN model:

````markdown
# Model Name

## Metadata

- Version: 1.0.0
- Author: System/User
- Created: 2024-01-01
- Modified: 2024-01-01
- Tags: [explorer, cautious, analytical]

## Description

Natural language description of the model's purpose and behavior.

## Architecture

```gnn
architecture {
  type: "GraphSAGE"
  layers: 3
  hidden_dim: 128
  activation: "relu"
  dropout: 0.2
}
```
````

````

## Parameters

```gnn
parameters {
  learning_rate: 0.001
  optimizer: "adam"
  batch_size: 32
  epochs: 100
}
````

## Active Inference Mapping

```gnn
active_inference {
  beliefs {
    initial: "uniform"
    update_rule: "bayesian"
  }

  preferences {
    exploration: 0.7
    exploitation: 0.3
    risk_tolerance: 0.5
  }

  policies {
    action_selection: "softmax"
    temperature: 1.0
  }
}
```

## Node Features

```gnn
node_features {
  spatial: ["x", "y", "z"]
  temporal: ["timestamp", "duration"]
  categorical: ["type", "status"]
  numerical: ["energy", "resources"]
}
```

## Edge Features

```gnn
edge_features {
  type: "directed"
  attributes: ["weight", "distance", "relationship"]
  dynamic: true
}
```

## Constraints

```gnn
constraints {
  max_nodes: 10000
  max_edges: 50000
  memory_limit: "4GB"
  compute_timeout: 300
}
```

## Validation Rules

```gnn
validation {
  node_degree: {
    min: 1
    max: 100
  }

  graph_connectivity: "connected"

  feature_ranges: {
    energy: [0, 1]
    resources: [0, 1000]
  }
}
```

````

## Section Details

### 1. Metadata Section

The metadata section contains essential information about the model:

```markdown
## Metadata
- Version: Semantic versioning (MAJOR.MINOR.PATCH)
- Author: Creator identification
- Created: ISO 8601 timestamp
- Modified: ISO 8601 timestamp
- Tags: Array of descriptive tags
- Dependencies: Optional list of required models
- License: Optional license information
````

### 2. Description Section

Natural language description explaining:

- Model purpose and goals
- Expected behavior patterns
- Use cases and scenarios
- Integration requirements

Example:

```markdown
## Description

This model implements an Explorer agent with cautious behavior patterns.
The agent prioritizes systematic exploration of unknown territories while
maintaining safety margins. It uses GraphSAGE architecture to aggregate
neighborhood information and make informed decisions about movement and
resource gathering.
```

### 3. Architecture Section

Defines the GNN architecture using structured notation:

```gnn
architecture {
  // Core architecture type
  type: "GCN" | "GAT" | "GraphSAGE" | "Custom"

  // Layer configuration
  layers: integer (1-10)
  hidden_dim: integer (16-512)
  output_dim: integer

  // Activation functions
  activation: "relu" | "tanh" | "sigmoid" | "elu" | "leaky_relu"

  // Regularization
  dropout: float (0.0-0.9)
  batch_norm: boolean
  layer_norm: boolean

  // Attention (for GAT)
  attention_heads: integer (1-16)
  attention_dropout: float (0.0-0.9)

  // Aggregation (for GraphSAGE)
  aggregator: "mean" | "max" | "lstm" | "pool"

  // Custom layers
  custom_layers: [
    {
      name: string
      type: string
      params: object
    }
  ]
}
```

### 4. Parameters Section

Training and optimization parameters:

```gnn
parameters {
  // Optimization
  learning_rate: float (0.0001-0.1)
  optimizer: "adam" | "sgd" | "rmsprop" | "adagrad"
  weight_decay: float (0.0-0.1)

  // Training
  batch_size: integer (1-256)
  epochs: integer (1-1000)
  early_stopping: {
    patience: integer
    min_delta: float
    monitor: string
  }

  // Learning rate scheduling
  lr_scheduler: {
    type: "step" | "exponential" | "cosine" | "plateau"
    params: object
  }

  // Gradient clipping
  gradient_clip: float
}
```

### 5. Active Inference Mapping

Maps GNN outputs to Active Inference framework:

```gnn
active_inference {
  // Belief dynamics
  beliefs {
    initial: "uniform" | "gaussian" | "learned"
    update_rule: "bayesian" | "variational" | "particle_filter"
    precision: float (0.1-10.0)
  }

  // Preference specifications
  preferences {
    exploration: float (0.0-1.0)
    exploitation: float (0.0-1.0)
    risk_tolerance: float (0.0-1.0)
    curiosity: float (0.0-1.0)
    social_weight: float (0.0-1.0)
  }

  // Policy configuration
  policies {
    action_selection: "softmax" | "epsilon_greedy" | "thompson_sampling"
    temperature: float (0.1-10.0)
    epsilon: float (0.0-1.0)
    planning_horizon: integer (1-100)
  }

  // Free energy components
  free_energy {
    complexity_weight: float (0.0-1.0)
    accuracy_weight: float (0.0-1.0)
    pragmatic_weight: float (0.0-1.0)
  }
}
```

### 6. Feature Specifications

#### Node Features

```gnn
node_features {
  // Spatial features
  spatial: ["x", "y", "z", "region", "territory"]

  // Temporal features
  temporal: ["timestamp", "age", "last_update", "duration"]

  // Categorical features
  categorical: {
    type: ["explorer", "merchant", "scholar", "guardian"]
    status: ["active", "idle", "blocked"]
    faction: ["red", "blue", "green", "neutral"]
  }

  // Numerical features
  numerical: {
    energy: { range: [0, 1], default: 1.0 }
    resources: { range: [0, 1000], default: 100 }
    health: { range: [0, 100], default: 100 }
    experience: { range: [0, unlimited], default: 0 }
  }

  // Embedding features
  embeddings: {
    personality: { dim: 16, method: "learned" }
    skills: { dim: 32, method: "pretrained" }
  }
}
```

#### Edge Features

```gnn
edge_features {
  // Edge type
  type: "directed" | "undirected" | "bidirectional"

  // Edge attributes
  attributes: {
    weight: { range: [0, 1], default: 1.0 }
    distance: { range: [0, unlimited], compute: "euclidean" }
    relationship: ["ally", "enemy", "neutral", "trade"]
    strength: { range: [0, 1], decay: 0.1 }
  }

  // Dynamic properties
  dynamic: boolean
  temporal_decay: float
  update_frequency: integer
}
```

### 7. Constraints Section

Resource and performance constraints:

```gnn
constraints {
  // Graph size limits
  max_nodes: integer
  max_edges: integer
  max_degree: integer

  // Resource limits
  memory_limit: string ("1GB", "4GB", etc.)
  compute_timeout: integer (seconds)
  gpu_required: boolean

  // Performance requirements
  min_throughput: integer (graphs/second)
  max_latency: integer (milliseconds)

  // Deployment constraints
  platforms: ["cpu", "cuda", "edge"]
  min_compute: string ("2 cores", "4GB RAM")
}
```

### 8. Validation Rules

Data quality and consistency rules:

```gnn
validation {
  // Graph structure
  graph_connectivity: "connected" | "disconnected" | "any"
  allow_self_loops: boolean
  allow_multi_edges: boolean

  // Node constraints
  node_degree: {
    min: integer
    max: integer
  }

  // Feature validation
  feature_ranges: {
    feature_name: [min, max]
  }

  // Required features
  required_node_features: [string]
  required_edge_features: [string]

  // Custom validation
  custom_rules: [
    {
      name: string
      condition: string
      error_message: string
    }
  ]
}
```

## Extended Syntax

### 1. Conditional Configurations

```gnn
architecture {
  type: "GAT"

  @if (node_count > 1000) {
    layers: 2
    hidden_dim: 64
  } @else {
    layers: 3
    hidden_dim: 128
  }
}
```

### 2. Template Inheritance

```gnn
@extends "base_explorer.gmn.md"

architecture {
  @override
  hidden_dim: 256

  @append
  custom_layers: [
    { name: "spatial_attention", type: "custom_attention" }
  ]
}
```

### 3. Preprocessor Directives

```gnn
@define MAX_NODES 10000
@define DEFAULT_DIM 128

architecture {
  hidden_dim: @DEFAULT_DIM
}

constraints {
  max_nodes: @MAX_NODES
}
```

### 4. Comments and Documentation

```gnn
architecture {
  type: "GraphSAGE"  // Using GraphSAGE for scalability

  /*
   * Layer configuration optimized for sparse graphs
   * Based on empirical testing with similar datasets
   */
  layers: 3
  hidden_dim: 128
}
```

## Personality Templates

FreeAgentics provides predefined personality templates:

### Explorer Template

```gnn
@template explorer_cautious
active_inference {
  preferences {
    exploration: 0.3
    exploitation: 0.7
    risk_tolerance: 0.2
    curiosity: 0.6
  }
}
```

### Merchant Template

```gnn
@template merchant_aggressive
active_inference {
  preferences {
    exploration: 0.5
    exploitation: 0.5
    risk_tolerance: 0.8
    social_weight: 0.9
  }
}
```

## Best Practices

### 1. Naming Conventions

- Model names: `AgentType_Personality_Version.gmn.md`
- Feature names: `snake_case`
- Constants: `UPPER_CASE`
- Templates: `type_personality`

### 2. Version Management

```gnn
## Metadata
- Version: 2.0.0  // Breaking changes
- Previous: 1.3.2
- Migration: "migrate_v1_to_v2.py"
```

### 3. Documentation

- Include clear descriptions for each section
- Document non-obvious parameter choices
- Reference papers or methods used
- Provide usage examples

### 4. Validation

- Define comprehensive validation rules
- Set reasonable constraints
- Include error messages
- Test edge cases

## Example: Complete Explorer Model

````markdown
# Explorer Cautious Model

## Metadata

- Version: 1.0.0
- Author: FreeAgentics Team
- Created: 2024-01-15T10:00:00Z
- Modified: 2024-01-15T10:00:00Z
- Tags: [explorer, cautious, efficient]

## Description

This model implements a cautious explorer agent that prioritizes safety while
systematically exploring unknown territories. It uses GraphSAGE architecture
for efficient neighborhood aggregation and maintains a balance between
exploration and self-preservation.

## Architecture

```gnn
architecture {
  type: "GraphSAGE"
  layers: 3
  hidden_dim: 128
  output_dim: 64
  activation: "relu"
  dropout: 0.2
  aggregator: "mean"
  batch_norm: true
}
```
````

````

## Parameters

```gnn
parameters {
  learning_rate: 0.001
  optimizer: "adam"
  weight_decay: 0.0001
  batch_size: 32
  epochs: 100

  early_stopping: {
    patience: 10
    min_delta: 0.001
    monitor: "val_loss"
  }
}
````

## Active Inference Mapping

```gnn
active_inference {
  beliefs {
    initial: "gaussian"
    update_rule: "variational"
    precision: 2.0
  }

  preferences {
    exploration: 0.3
    exploitation: 0.7
    risk_tolerance: 0.2
    curiosity: 0.6
    social_weight: 0.4
  }

  policies {
    action_selection: "softmax"
    temperature: 0.8
    planning_horizon: 5
  }

  free_energy {
    complexity_weight: 0.4
    accuracy_weight: 0.6
  }
}
```

## Node Features

```gnn
node_features {
  spatial: ["x", "y", "region_id"]
  temporal: ["last_visit", "discovery_time"]

  categorical: {
    status: ["unexplored", "exploring", "explored", "dangerous"]
    terrain: ["plains", "forest", "mountain", "water"]
  }

  numerical: {
    energy: { range: [0, 1], default: 1.0 }
    danger_level: { range: [0, 1], default: 0.0 }
    resources_found: { range: [0, unlimited], default: 0 }
  }
}
```

## Edge Features

```gnn
edge_features {
  type: "directed"

  attributes: {
    distance: { compute: "euclidean" }
    traversal_cost: { range: [0, 10], default: 1.0 }
    safety_score: { range: [0, 1], default: 1.0 }
  }

  dynamic: true
  temporal_decay: 0.05
}
```

## Constraints

```gnn
constraints {
  max_nodes: 5000
  max_edges: 25000
  memory_limit: "2GB"
  compute_timeout: 60
  gpu_required: false
}
```

## Validation Rules

```gnn
validation {
  graph_connectivity: "connected"
  allow_self_loops: false

  node_degree: {
    min: 1
    max: 50
  }

  feature_ranges: {
    energy: [0, 1]
    danger_level: [0, 1]
  }

  required_node_features: ["x", "y", "status", "energy"]
  required_edge_features: ["distance"]
}
```

```

## Error Messages

Common error messages and their meanings:

| Error Code | Message | Description |
|------------|---------|-------------|
| GNN001 | Invalid architecture type | Specified type not in allowed list |
| GNN002 | Feature dimension mismatch | Input/output dimensions don't match |
| GNN003 | Constraint violation | Model exceeds resource constraints |
| GNN004 | Validation rule failed | Data doesn't meet validation criteria |
| GNN005 | Syntax error in GNN block | Malformed GNN notation |
| GNN006 | Missing required section | Required section not found in file |
| GNN007 | Invalid parameter range | Parameter value outside allowed range |
| GNN008 | Circular dependency | Model dependencies form a cycle |
| GNN009 | Version conflict | Incompatible model versions |
| GNN010 | Template not found | Referenced template doesn't exist |

## Migration Guide

When updating model formats:

1. **Increment version** following semantic versioning
2. **Document changes** in model metadata
3. **Provide migration script** for automated updates
4. **Test compatibility** with existing systems
5. **Update documentation** to reflect changes

---

This specification defines the complete format for `.gmn.md` files in FreeAgentics. Models following this format can be parsed, validated, and converted into executable Graph Neural Networks for agent behavior modeling.
```
