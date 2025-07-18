# Iterative Loop Controller Guide

## Overview

The Iterative Loop Controller is a core component of FreeAgentics that enables multi-turn conversations where each iteration builds upon previous ones. It tracks conversation state, analyzes patterns, and generates intelligent suggestions based on the evolving knowledge graph and agent beliefs.

## Key Features

### 1. Conversation Context Management
- Tracks all prompts, agents, and knowledge graph updates across iterations
- Maintains belief history to analyze convergence patterns
- Preserves suggestion history to avoid repetition

### 2. Intelligent Suggestion Generation
- Context-aware suggestions based on:
  - Current iteration number
  - Belief stability and evolution
  - Knowledge graph connectivity
  - Identified capability gaps
  - Prompt theme analysis
- Suggestions evolve from basic exploration to advanced meta-learning

### 3. Iteration-Specific Constraints
- Dynamically adjusts GMN generation constraints based on:
  - Belief convergence (high stability → increase complexity)
  - Knowledge graph density (low connectivity → encourage links)
  - Prompt evolution pattern (refining vs. pivoting)
  - Iteration count (early → explore, middle → optimize, late → innovate)

### 4. Knowledge Graph Accumulation
- Tracks how the knowledge graph grows across iterations
- Identifies isolated nodes and disconnected clusters
- Suggests ways to improve connectivity

## Usage

### API Integration

The iterative controller is automatically integrated into the prompt processing pipeline:

```python
POST /api/v1/prompts
{
  "prompt": "Create an explorer agent",
  "conversation_id": "optional-existing-id",
  "iteration_count": 1
}
```

Response includes iteration context:
```json
{
  "agent_id": "agent-123",
  "gmn_specification": "...",
  "next_suggestions": [
    "Add goal states to guide behavior",
    "Increase observation diversity"
  ],
  "iteration_context": {
    "iteration_number": 1,
    "total_agents": 1,
    "kg_nodes": 5,
    "conversation_summary": {
      "iteration_count": 1,
      "belief_evolution": {
        "trend": "exploring",
        "stability": 0.2
      }
    }
  }
}
```

### Frontend Integration

The React hook automatically manages iteration state:

```typescript
const {
  submitPrompt,
  iterationContext,
  suggestions,
  resetConversation
} = usePromptProcessor();

// Submit with conversation continuity
await submitPrompt("Add goals to the agent", true);

// Start fresh conversation
resetConversation();
await submitPrompt("Create new agent", true);
```

### Suggestion Evolution Pattern

1. **Iteration 1-2**: Basic exploration
   - "Start with basic exploration to establish environmental understanding"
   - "Add sensory modalities to reduce belief ambiguity"

2. **Iteration 3-4**: Goal-directed behavior
   - "Add goal-directed behavior to guide agent actions"
   - "Define preferences to shape agent objectives"

3. **Iteration 5-6**: Multi-agent coordination
   - "Introduce multi-agent coordination for complex tasks"
   - "Add communication channels between agents"

4. **Iteration 7+**: Advanced capabilities
   - "Consider meta-learning - Let agents adapt their own models"
   - "Focus on knowledge consolidation rather than expansion"

## Architecture

### ConversationContext Class
Maintains state for a single conversation:
- `iteration_count`: Number of prompts processed
- `agent_ids`: List of all agents created
- `kg_node_ids`: Set of all knowledge graph nodes
- `belief_history`: Evolution of agent beliefs
- `suggestion_history`: All generated suggestions
- `prompt_history`: All user prompts

### IterativeController Class
Orchestrates the iterative loop:
- `prepare_iteration_context()`: Prepares context for next iteration
- `generate_intelligent_suggestions()`: Creates context-aware suggestions
- `update_conversation_context()`: Records iteration results
- `_analyze_kg_connectivity()`: Evaluates knowledge graph structure
- `_identify_capability_gaps()`: Finds missing agent capabilities

## Best Practices

1. **Continuous Conversations**: Let users build on previous iterations naturally
2. **Suggestion Following**: Encourage users to explore generated suggestions
3. **Context Reset**: Provide clear option to start fresh conversations
4. **Progress Indicators**: Show iteration number and evolution metrics
5. **Adaptive Complexity**: Start simple and increase complexity gradually

## Testing

Run the iterative loop tests:
```bash
# Unit tests
pytest tests/unit/services/test_iterative_controller.py -v

# Integration tests
pytest tests/integration/test_iterative_loop.py -v

# Interactive demo
python demo_iterative_loop.py
```

## Future Enhancements

1. **Conversation Branching**: Allow exploring multiple paths from same point
2. **Iteration Rollback**: Undo and try different approaches
3. **Suggestion Learning**: Learn from which suggestions users select
4. **Cross-Conversation Learning**: Transfer insights between conversations
5. **Collaborative Iterations**: Multiple users contributing to same conversation