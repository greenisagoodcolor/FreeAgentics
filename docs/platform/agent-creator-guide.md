# Agent Creator Guide

The Agent Creator is FreeAgentics's visual tool for designing Active Inference agents. This guide covers all features and best practices for creating compelling, functional agents.

## Overview

The Agent Creator provides:

- **Personality Design**: Big Five trait sliders
- **Backstory Generation**: AI-powered narrative creation
- **GNN Model Preview**: Real-time model generation
- **Visual Customization**: Agent appearance options
- **Validation**: Ensure agents are well-formed

## Understanding Agent Personality

### The Big Five Model

FreeAgentics uses the Big Five personality model to generate agent behaviors:

#### Openness (0-100)

- **Low (0-30)**: Traditional, routine-focused, cautious
- **Medium (30-70)**: Balanced curiosity and caution
- **High (70-100)**: Creative, exploratory, risk-taking

_Effect on behavior_: High openness leads to more exploration, trying new strategies, and forming unusual connections.

#### Conscientiousness (0-100)

- **Low (0-30)**: Spontaneous, flexible, reactive
- **Medium (30-70)**: Balanced planning and adaptability
- **High (70-100)**: Organized, goal-oriented, persistent

_Effect on behavior_: High conscientiousness results in systematic resource gathering and long-term planning.

#### Extraversion (0-100)

- **Low (0-30)**: Solitary, independent, observant
- **Medium (30-70)**: Selective social engagement
- **High (70-100)**: Social, communicative, collaborative

_Effect on behavior_: High extraversion increases communication frequency and alliance formation.

#### Agreeableness (0-100)

- **Low (0-30)**: Competitive, skeptical, self-focused
- **Medium (30-70)**: Balanced cooperation and competition
- **High (70-100)**: Cooperative, trusting, helpful

_Effect on behavior_: High agreeableness leads to resource sharing and peaceful conflict resolution.

#### Neuroticism (0-100)

- **Low (0-30)**: Stable, calm, resilient
- **Medium (30-70)**: Moderate emotional responses
- **High (70-100)**: Sensitive, reactive, cautious

_Effect on behavior_: High neuroticism causes more frequent belief updates and defensive behaviors.

## Creating an Agent Step-by-Step

### Step 1: Access the Creator

Navigate to `/agents` or click "Create Agent" from the dashboard.

### Step 2: Basic Information

#### Name Your Agent

- Choose a memorable, unique name
- Consider names that reflect personality
- Avoid special characters

#### Select Agent Class

Choose from base templates:

- **Explorer**: High openness, medium conscientiousness
- **Merchant**: High extraversion, high agreeableness
- **Scholar**: High conscientiousness, low extraversion
- **Guardian**: High conscientiousness, low openness
- **Custom**: Define your own combination

### Step 3: Set Personality Traits

#### Using the Sliders

1. Drag each slider to set trait values
2. Watch the preview update in real-time
3. See how traits affect the GNN model

#### Personality Recipes

**The Pioneer** (Explorer variant)

- Openness: 85
- Conscientiousness: 40
- Extraversion: 60
- Agreeableness: 50
- Neuroticism: 30

**The Diplomat** (Social variant)

- Openness: 60
- Conscientiousness: 70
- Extraversion: 80
- Agreeableness: 85
- Neuroticism: 25

**The Strategist** (Planner variant)

- Openness: 50
- Conscientiousness: 90
- Extraversion: 30
- Agreeableness: 40
- Neuroticism: 35

### Step 4: Generate or Write Backstory

#### AI Generation

1. Click "Generate Backstory"
2. Select a template:

   - **Origin Story**: Where they came from
   - **Defining Moment**: Key life event
   - **Quest**: Current mission
   - **Relationships**: Social history

3. Customize the prompt:

   ```
   Generate a backstory for an agent who is highly curious but
   cautious, values knowledge above resources, and has a mysterious
   past involving ancient ruins.
   ```

#### Manual Writing

Consider including:

- **Origins**: Where/how was the agent created?
- **Motivations**: What drives them?
- **Fears**: What do they avoid?
- **Goals**: What do they seek?
- **Quirks**: Unique behaviors or preferences

### Step 5: Review GNN Model

The generated GNN model includes:

#### Beliefs Section

```markdown
## Beliefs

- The world contains hidden knowledge waiting to be discovered
- Cooperation leads to better outcomes than competition
- Every challenge has a solution if approached correctly
```

#### Preferences Section

```markdown
## Preferences

- curiosity_weight: 0.8
- social_weight: 0.6
- safety_threshold: 0.3
- resource_priority: knowledge
```

#### Policies Section

```markdown
## Policies

- **Explore**: When energy > 40%, move to unexplored areas
- **Communicate**: When meeting others, share discoveries
- **Learn**: After each experience, extract patterns
```

### Step 6: Visual Customization

#### Appearance Options

- **Color Scheme**: Reflects personality
- **Icon**: Visual representation
- **Size**: Based on traits
- **Glow**: Indicates special abilities

#### Auto-Generation

The system suggests appearance based on:

- High openness → Bright, varied colors
- High conscientiousness → Structured patterns
- High extraversion → Larger, more prominent
- High agreeableness → Soft, warm colors
- High neuroticism → Shifting, dynamic effects

### Step 7: Validate and Create

#### Validation Checks

The system ensures:

- ✓ Valid personality values (0-100)
- ✓ Unique agent name
- ✓ Well-formed GNN model
- ✓ No conflicting policies
- ✓ Reasonable resource requirements

#### Common Validation Errors

- **"Policies conflict"**: Adjust trait balance
- **"Invalid GNN syntax"**: Check custom edits
- **"Name already exists"**: Choose unique name

### Step 8: Deploy to World

After creation:

1. Agent appears in the world view
2. Starting position is assigned
3. Initial resources allocated
4. Agent begins following its policies

## Advanced Features

### Custom GNN Editing

Click "Edit GNN Model" to modify directly:

```markdown
---
model_name: CustomExplorer
version: 1.0
agent_class: Explorer
personality:
  openness: 85
  conscientiousness: 45
  extraversion: 60
  agreeableness: 55
  neuroticism: 30
---

## Beliefs

- Knowledge is more valuable than material resources
- The unknown holds more promise than the familiar
- Every agent has something valuable to share

## Preferences

- curiosity_weight: 0.85
- exploration_bonus: 1.5
- social_openness: high
- risk_tolerance: 0.7
- learning_rate: 0.3

## Policies

- **Explore**: When energy > 30%, prioritize unexplored hexes
- **Investigate**: When finding anomalies, spend time studying
- **Share**: When meeting others, exchange map information
- **Rest**: When energy < 20%, find safe location to recover
```

### Personality Templates

Load pre-made personalities:

1. Click "Load Template"
2. Choose from:
   - Research personalities
   - Story archetypes
   - Historical figures
   - Community creations

### Batch Creation

Create multiple related agents:

1. Design base personality
2. Click "Create Variants"
3. Adjust variation parameters:
   - Trait variance (±10-20)
   - Backstory themes
   - Visual variations

## Best Practices

### Balanced Agents

- Avoid extreme values on all traits
- Consider trait interactions
- Test in small simulations first

### Compelling Backstories

- Make them specific but flexible
- Include hooks for interaction
- Leave room for growth

### Effective Policies

- Start simple, add complexity
- Ensure energy management
- Include social policies

### Performance Considerations

- Limit complex calculations in policies
- Use reasonable sensor ranges
- Balance update frequencies

## Troubleshooting

### Agent Not Moving

- Check energy levels
- Verify movement policies
- Ensure valid starting position

### No Communication

- Increase extraversion
- Add communication policies
- Check proximity to others

### Rapid Belief Changes

- Lower neuroticism
- Adjust learning rate
- Add belief stability policies

### Resource Depletion

- Increase conscientiousness
- Add resource management policies
- Balance exploration vs. exploitation

## Examples Gallery

### The Wandering Scholar

_A knowledge-seeking agent who maps the world_

- Openness: 90, Conscientiousness: 70
- Focuses on exploration and documentation
- Shares knowledge freely

### The Merchant Prince

_A social trader building networks_

- Extraversion: 85, Agreeableness: 75
- Creates trade routes and alliances
- Balances profit with reputation

### The Guardian Sentinel

_A protective agent maintaining order_

- Conscientiousness: 85, Neuroticism: 60
- Patrols territories and aids others
- Responds quickly to threats

## Next Steps

After creating agents:

1. **Test in Simulation**: Run small tests
2. **Observe Behaviors**: Watch for emergent patterns
3. **Refine Models**: Adjust based on results
4. **Share Creations**: Contribute to community

---

_For more details, see the [GNN Model Format](../gnn_models/model_format.md) documentation._
