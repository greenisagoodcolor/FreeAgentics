# GMN Model Format

This document describes the Generative Model Notation (GMN) format used in FreeAgentics for defining Active Inference agent models using PyMDP mathematical frameworks.

## Overview

GMN provides a structured, human-readable format for specifying mathematical models that can be automatically converted to PyMDP implementations for Active Inference agents.

## File Structure

```markdown
---
model_name: AgentName
version: 1.0
agent_class: Explorer|Merchant|Scholar|Guardian|Custom
description: Brief description of the agent
author: Your Name
created: 2024-01-01
personality:
  openness: 0-100
  conscientiousness: 0-100
  extraversion: 0-100
  agreeableness: 0-100
  neuroticism: 0-100
---

## Beliefs

## Preferences

## Policies

## Connections (optional)

## Initial State (optional)

## Equations (optional)
```

## Sections Explained

### Frontmatter (Required)

The YAML frontmatter contains metadata:

```yaml
---
model_name: CuriousExplorer
version: 1.0
agent_class: Explorer
description: An agent driven by curiosity to map the unknown
author: FreeAgentics Team
created: 2024-01-15
personality:
  openness: 85
  conscientiousness: 45
  extraversion: 60
  agreeableness: 55
  neuroticism: 30
tags:
  - exploration
  - knowledge-seeking
  - social
---
```

**Required fields:**

- `model_name`: Unique identifier for the model
- `version`: Semantic version (major.minor)
- `agent_class`: Base class (Explorer, Merchant, Scholar, Guardian, Custom)

**Optional fields:**

- `description`: One-line summary
- `author`: Creator's name
- `created`: Creation date (YYYY-MM-DD)
- `personality`: Big Five traits (0-100)
- `tags`: Categories for organization

### Beliefs (Required)

Beliefs represent what the agent thinks is true about the world:

```markdown
## Beliefs

- The world contains valuable resources to discover
- Cooperation leads to better outcomes than competition
- Knowledge shared is knowledge multiplied
- Every location has hidden potential
- Other agents can be trusted until proven otherwise
```

**Format:**

- Use bullet points (-, \*, or •)
- Write in first person or general statements
- Keep beliefs concise and clear
- Order by importance

**Good beliefs:**

- ✓ "Exploration reveals new opportunities"
- ✓ "Energy must be conserved for survival"
- ✓ "Patterns in the environment can be learned"

**Poor beliefs:**

- ✗ "Maybe there could be something interesting somewhere"
- ✗ "I think that possibly cooperation might be good sometimes"

### Preferences (Required)

Preferences define what the agent values and how strongly:

```markdown
## Preferences

- curiosity_weight: 0.8
- safety_threshold: 0.3
- social_weight: 0.6
- resource_priority: knowledge
- exploration_radius: 5
- risk_tolerance: 0.7
- learning_rate: 0.2
- memory_capacity: 100
```

**Format:**

- Use `key: value` pairs
- Numeric values typically 0.0-1.0
- Can use integers for counts
- Can use strings for categories

**Common preferences:**

- `curiosity_weight`: Drive to explore (0-1)
- `safety_threshold`: Minimum safe energy (0-1)
- `social_weight`: Importance of social interaction (0-1)
- `resource_priority`: knowledge|energy|materials|social
- `risk_tolerance`: Willingness to take risks (0-1)
- `learning_rate`: Speed of belief updates (0-1)

### Policies (Required)

Policies define decision-making rules:

```markdown
## Policies

- **Explore**: When energy > 40%, move to unexplored adjacent hexes
- **Gather**: When resources detected within 2 hexes, move to collect
- **Communicate**: When another agent is adjacent, initiate conversation
- **Rest**: When energy < 20%, remain stationary to recover
- **Share**: When meeting friendly agents, exchange map information
- **Learn**: After each experience, update beliefs based on outcomes
```

**Format:**

- Use bold for policy names: `**PolicyName**`
- Follow with colon and description
- Include trigger conditions
- Specify concrete actions

**Policy structure:**

```markdown
- **PolicyName**: When [condition], then [action]
```

**Good policies:**

- ✓ `**Explore**: When energy > 50% and no threats detected, move to highest-value unexplored hex`
- ✓ `**Trade**: When meeting merchant class agents, offer resources for information`

**Poor policies:**

- ✗ `**Do Something**: Sometimes do things`
- ✗ `**Move**: Go places`

### Connections (Optional)

Defines relationships between internal variables:

```markdown
## Connections

- beliefs.world_knowledge -> policies.exploration_strategy
- observations.resource_density -> preferences.risk_tolerance
- social.trust_network -> policies.cooperation_threshold
- energy.current -> policies.activity_level
```

**Format:**

- Use arrow notation: `source -> target`
- Can specify connection strength: `source -0.8-> target`
- Group related connections

### Initial State (Optional)

Sets starting values for agent variables:

```markdown
## Initial State

- energy: 100
- knowledge_nodes: 0
- explored_hexes: 1
- social_connections: 0
- resources:
  - materials: 10
  - information: 0
- position: random
- heading: north
```

**Format:**

- Use `key: value` pairs
- Can nest structures with indentation
- Specify units where relevant

### Equations (Optional)

Mathematical relationships for advanced models:

```markdown
## Equations

- free_energy = complexity - accuracy
- exploration_value = curiosity_weight \* uncertainty - risk_cost
- social_benefit = (shared_knowledge \* trust_level) / interaction_cost
- learning_delta = learning_rate \* (observation - prediction)
```

**Format:**

- Use standard mathematical notation
- Define all variables used
- Keep equations simple and computational

## Complete Example

```markdown
---
model_name: AdaptiveExplorer
version: 2.0
agent_class: Explorer
description: An explorer that adapts strategies based on environment
author: Research Team
created: 2024-01-20
personality:
  openness: 80
  conscientiousness: 60
  extraversion: 50
  agreeableness: 70
  neuroticism: 40
---

## Beliefs

- The world has patterns that can be learned
- Cooperation accelerates discovery
- Resource management ensures long-term survival
- Every failure is a learning opportunity
- Diverse strategies lead to better outcomes

## Preferences

- curiosity_weight: 0.75
- safety_threshold: 0.25
- social_weight: 0.6
- resource_priority: balanced
- exploration_method: adaptive
- risk_tolerance: 0.65
- learning_rate: 0.25
- memory_capacity: 150
- cooperation_bias: 0.7

## Policies

- **Adaptive Exploration**: When energy > 35%, choose exploration strategy based on recent success rates
- **Resource Balance**: Maintain minimum 20% reserves of each resource type
- **Social Learning**: When meeting successful agents, adopt their effective strategies
- **Risk Assessment**: Before dangerous actions, calculate risk vs. reward using past experiences
- **Knowledge Sharing**: Share discoveries with agents who have shared with me
- **Pattern Recognition**: Every 10 steps, analyze movement patterns for optimization

## Connections

- beliefs.pattern_recognition -> policies.exploration_strategy
- experiences.success_rate -> preferences.risk_tolerance
- social.shared_knowledge -> beliefs.world_model
- resources.current -> policies.activity_selection

## Initial State

- energy: 100
- knowledge_nodes: 5
- explored_hexes: 1
- known_agents: 0
- strategy_weights:
  - random_walk: 0.33
  - systematic_sweep: 0.33
  - follow_edges: 0.34
- memory:
  - successful_paths: []
  - dangerous_locations: []
  - helpful_agents: []
```

## Validation Rules

FreeAgentics validates GNN models for:

1. **Required Sections**: Must have Beliefs, Preferences, and Policies
2. **Valid YAML**: Frontmatter must parse correctly
3. **Reasonable Values**: Numeric preferences in valid ranges
4. **Policy Structure**: Policies must have conditions and actions
5. **No Conflicts**: Policies shouldn't contradict each other
6. **Computational Feasibility**: Equations must be computable

## Best Practices

### Writing Beliefs

- Start with 3-7 core beliefs
- Make them actionable
- Avoid contradictions
- Express as facts, not uncertainties

### Setting Preferences

- Use 0-1 scale for weights
- Define all referenced preferences
- Balance competing values
- Document unusual values

### Creating Policies

- Order by priority/frequency
- Make conditions specific
- Define clear actions
- Include failure cases

### Model Evolution

- Increment version for changes
- Document major updates
- Preserve backward compatibility
- Test thoroughly

## Advanced Features

### Dynamic Properties

```markdown
## Preferences

- curiosity_weight: dynamic(0.5, 0.9) # Varies based on success
- risk_tolerance: adaptive(experience) # Learns from outcomes
```

### Conditional Policies

```markdown
## Policies

- **Complex Exploration**:
  - If energy > 70%: aggressive_exploration()
  - Elif energy > 40%: balanced_exploration()
  - Else: conservative_exploration()
```

### Meta-Learning

```markdown
## Policies

- **Strategy Evolution**: Every 50 steps, evaluate all strategies and adjust weights based on performance
- **Belief Revision**: When predictions fail > 30%, trigger belief update process
```

## Tools and Utilities

### Validation

```bash
python pipeline/main.py --only-steps 2 --models-dir models/
```

### Visualization

View parsed models in the web interface at `/models`

### Testing

Test models in isolated simulations before full deployment

---

_For more examples, see the [models directory](../../models/) in the repository._
