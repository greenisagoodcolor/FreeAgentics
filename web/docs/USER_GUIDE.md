# FreeAgentics User Guide

## Welcome to FreeAgentics

FreeAgentics is a revolutionary platform that allows you to create intelligent AI agents using natural language. Simply describe what you want your agent to do, and our system will create an Active Inference agent that can perceive, think, and act autonomously.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding the Interface](#understanding-the-interface)
3. [Creating Your First Agent](#creating-your-first-agent)
4. [Understanding Agent Behavior](#understanding-agent-behavior)
5. [Working with the Knowledge Graph](#working-with-the-knowledge-graph)
6. [Using Suggestions](#using-suggestions)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### What You Can Do

With FreeAgentics, you can:

- Create intelligent agents using plain English
- Watch agents explore and learn in real-time
- See how agents build knowledge over time
- Refine and improve agents through conversation
- Visualize agent beliefs and decisions

### Accessing the Platform

1. Open your web browser and navigate to the FreeAgentics interface
2. Log in with your credentials
3. You'll see the main prompt interface ready for your first agent

## Understanding the Interface

### Main Components

```
┌─────────────────────────────────────────────────────────┐
│                    Navigation Bar                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────┐  ┌─────────────────────────┐  │
│  │                     │  │                         │  │
│  │   Prompt Input      │  │   Agent Visualization  │  │
│  │                     │  │                         │  │
│  └─────────────────────┘  └─────────────────────────┘  │
│                                                          │
│  ┌─────────────────────┐  ┌─────────────────────────┐  │
│  │                     │  │                         │  │
│  │   Suggestions       │  │   Knowledge Graph      │  │
│  │                     │  │                         │  │
│  └─────────────────────┘  └─────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Key Areas

1. **Prompt Input**: Where you describe your agent in natural language
2. **Agent Visualization**: Real-time view of your agent's state and actions
3. **Suggestions Panel**: AI-generated ideas for improving your agent
4. **Knowledge Graph**: Visual representation of what your agent knows

## Creating Your First Agent

### Step 1: Write a Prompt

Start with a simple, clear description of what you want your agent to do.

**Good Examples:**

- "Create an explorer agent for a 5x5 grid world"
- "Make a trading agent that can buy and sell resources"
- "Build an agent that can navigate a maze and find the exit"

**Tips for Better Prompts:**

- Be specific about the environment
- Mention key behaviors you want
- Include any constraints or goals

### Step 2: Submit Your Prompt

1. Type your prompt in the input field
2. Click "Create Agent" or press Enter
3. Watch the progress indicators as your agent is created

### Step 3: Understanding the Pipeline

When you submit a prompt, the system goes through 6 stages:

1. **GMN Generation** - Converting your words to a formal specification
2. **Validation** - Checking the specification is correct
3. **Agent Creation** - Building the Active Inference agent
4. **Database Storage** - Saving your agent
5. **Knowledge Graph Update** - Recording what your agent knows
6. **Suggestion Generation** - Creating ideas for improvements

Each stage shows a progress indicator and status message.

## Understanding Agent Behavior

### Agent States

Your agent's current state is shown in the visualization panel:

- **Location**: Where the agent is (for spatial agents)
- **Beliefs**: What the agent thinks about its environment
- **Uncertainty**: How confident the agent is
- **Goals**: What the agent is trying to achieve

### Belief Visualization

The belief display shows probability distributions:

```
Location Beliefs:
█████████░░░░░░ 60% - Position (2,3)
███░░░░░░░░░░░░ 20% - Position (2,4)
███░░░░░░░░░░░░ 20% - Position (3,3)
```

Higher bars mean the agent is more certain about that belief.

### Action Selection

Watch how your agent chooses actions:

1. **Exploration** - Agent tries new things to learn
2. **Exploitation** - Agent uses what it knows
3. **Goal-Seeking** - Agent moves toward objectives

The action probabilities show what the agent might do next.

## Working with the Knowledge Graph

### What is the Knowledge Graph?

The knowledge graph is a visual representation of everything your agent knows:

- **Nodes**: Represent concepts, beliefs, or entities
- **Edges**: Show relationships between nodes
- **Colors**: Indicate different types of knowledge
- **Size**: Shows importance or certainty

### Interacting with the Graph

- **Zoom**: Scroll or pinch to zoom in/out
- **Pan**: Click and drag to move around
- **Select**: Click nodes to see details
- **Filter**: Use controls to show/hide node types

### Understanding Node Types

| Color  | Type        | Description                     |
| ------ | ----------- | ------------------------------- |
| Blue   | Belief      | What the agent thinks is true   |
| Green  | Goal        | What the agent wants to achieve |
| Yellow | Observation | What the agent has seen         |
| Red    | Uncertainty | Areas of high uncertainty       |
| Purple | Action      | Possible agent actions          |

## Using Suggestions

### What are Suggestions?

After creating an agent, you'll receive intelligent suggestions for improvements:

- **Behavioral changes** - "Make the agent more curious"
- **Goal additions** - "Add a specific target location"
- **Environment modifications** - "Include obstacles in the world"
- **Multi-agent ideas** - "Create a coordinator agent"

### Applying Suggestions

1. Review the suggestion list
2. Click on a suggestion to use it as your next prompt
3. Modify the suggestion if needed
4. Submit to refine your agent

### Iterative Improvement

Each time you apply a suggestion:

- Your agent becomes more sophisticated
- The system learns from the refinement
- New, more advanced suggestions appear

## Advanced Features

### Conversation Mode

Continue refining your agent through conversation:

```
You: "Create an explorer agent for a 5x5 grid"
System: [Creates basic explorer]

You: "Make it avoid the corners"
System: [Adds corner avoidance behavior]

You: "Now make it search for a hidden treasure"
System: [Adds treasure-seeking goal]
```

### Multi-Agent Systems

Create multiple agents that work together:

1. Create your first agent
2. Use suggestions to create complementary agents
3. Watch them interact in the shared environment

### Custom Parameters

Advanced users can specify detailed parameters:

```
"Create an explorer agent with:
- Planning horizon of 5 steps
- High curiosity (exploration bonus 0.8)
- Preference for moving north"
```

### GMN Specification View

Click "Show GMN" to see the formal specification:

```
WORLD grid_5x5
  STATES locations[25]
  OBSERVATIONS visible_cells[9]
  ACTIONS move[4] = [north, south, east, west]

AGENT explorer
  BELIEFS location_belief[25]
  POLICY active_inference
  PREFERENCE explore_new = 0.8
```

## Troubleshooting

### Common Issues

**"Agent creation failed"**

- Try a simpler prompt first
- Check for typos or unclear descriptions
- Use the template suggestions

**"No visualization appearing"**

- Refresh the page
- Check your internet connection
- Ensure WebSocket connections are allowed

**"Suggestions not relevant"**

- Provide more specific initial prompts
- Try different agent types
- Use the feedback option

### Performance Tips

1. **Start Simple**: Begin with basic agents before complex ones
2. **Use Templates**: Build on proven agent designs
3. **Iterate Gradually**: Make small improvements each time
4. **Monitor Beliefs**: Watch for high uncertainty areas

### Getting Help

- **Documentation**: Check the API reference for technical details
- **Templates**: Use pre-built templates as starting points
- **Community**: Join discussions with other users
- **Support**: Contact support for specific issues

## Best Practices

### Writing Effective Prompts

1. **Be Specific**
   - ❌ "Make an agent"
   - ✅ "Create an explorer agent for a 10x10 grid world"

2. **Include Context**
   - ❌ "Agent that trades"
   - ✅ "Create a trader agent for a market with gold, silver, and copper"

3. **Specify Goals**
   - ❌ "Navigation agent"
   - ✅ "Build an agent that navigates to find the shortest path to the exit"

### Understanding Results

1. **Check Belief Entropy**: Lower entropy means more certainty
2. **Monitor Exploration**: Ensure agents aren't stuck
3. **Validate Goals**: Confirm agents pursue intended objectives
4. **Review Suggestions**: They often reveal overlooked aspects

### Iterative Development

1. Start with core functionality
2. Add complexity gradually
3. Test each iteration
4. Use suggestions wisely
5. Document what works

## Next Steps

Now that you understand the basics:

1. **Experiment**: Try different types of agents
2. **Explore**: Use the visualization tools fully
3. **Learn**: Understand how beliefs drive behavior
4. **Share**: Collaborate with other users
5. **Innovate**: Push the boundaries of what's possible

## Glossary

- **Active Inference**: Mathematical framework for intelligent behavior
- **GMN**: Generalized Modeling Notation - formal agent specification
- **Belief**: Agent's probabilistic understanding of its world
- **Free Energy**: Measure the agent tries to minimize
- **Knowledge Graph**: Network representation of agent knowledge
- **PyMDP**: Python library implementing Active Inference

## Conclusion

FreeAgentics makes cutting-edge AI accessible through natural language. Start simple, iterate often, and watch as your agents develop increasingly sophisticated behaviors. The platform grows with you - from basic agents to complex multi-agent systems.

Happy agent building!
