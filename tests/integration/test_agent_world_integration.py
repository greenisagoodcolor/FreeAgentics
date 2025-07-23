"""Integration test for Active Inference agent in GridWorld."""

from agents.agent_manager import AgentManager
from agents.base_agent import BasicExplorerAgent
from world.grid_world import Agent as GridAgent
from world.grid_world import CellType, GridWorld, GridWorldConfig, Position


def test_explorer_agent_in_gridworld():
    """Test that explorer agent can navigate GridWorld using Active Inference."""
    # Create world
    config = GridWorldConfig(width=5, height=5)
    world = GridWorld(config)

    # Add goal and obstacles
    world.set_cell(Position(4, 4), CellType.GOAL, value=10.0)
    world.set_cell(Position(2, 2), CellType.WALL)
    world.set_cell(Position(2, 3), CellType.WALL)

    # Create Active Inference agent
    ai_agent = BasicExplorerAgent("explorer_1", "Explorer", grid_size=5)
    ai_agent.start()

    # Create corresponding grid agent
    grid_agent = GridAgent("explorer_1", Position(0, 0))
    world.add_agent(grid_agent)

    # Run several steps
    steps_taken = 0
    for step in range(10):
        # Get observation from world
        obs = world.get_observation("explorer_1")

        # Convert to format expected by AI agent
        if obs:
            ai_observation = {
                "position": [obs["agent_position"].x, obs["agent_position"].y],
                "surroundings": obs["local_grid"],
            }

            # AI agent processes observation and selects action
            action = ai_agent.step(ai_observation)
            steps_taken += 1

            # Convert action to position
            current_pos = grid_agent.position
            new_pos = current_pos

            if action == "up":
                new_pos = Position(current_pos.x, current_pos.y - 1)
            elif action == "down":
                new_pos = Position(current_pos.x, current_pos.y + 1)
            elif action == "left":
                new_pos = Position(current_pos.x - 1, current_pos.y)
            elif action == "right":
                new_pos = Position(current_pos.x + 1, current_pos.y)

            # Move agent in world
            world.move_agent("explorer_1", new_pos)

    # Check if agent metrics match actual steps taken
    if hasattr(ai_agent, "metrics"):
        assert ai_agent.metrics["total_observations"] == steps_taken
        assert ai_agent.metrics["total_actions"] == steps_taken

    # Agent should have moved from starting position
    final_pos = world.agents["explorer_1"].position
    assert final_pos != Position(0, 0), "Agent should have explored"

    # Check that beliefs were updated
    if hasattr(ai_agent, "beliefs"):
        assert len(ai_agent.beliefs) > 0, "Agent should have beliefs"

    # Stop agent
    ai_agent.stop()


def test_agent_manager_integration():
    """Test agent manager creates and coordinates agents properly."""
    manager = AgentManager()

    # Create world
    world = manager.create_world(size=10)
    assert world is not None
    assert world.width == 10
    assert world.height == 10

    # Create agent
    agent_id = manager.create_agent("explorer", "Test Explorer")
    assert agent_id in manager.agents

    # Start agent
    success = manager.start_agent(agent_id)
    assert success
    assert manager.agents[agent_id].is_active

    # Stop agent
    success = manager.stop_agent(agent_id)
    assert success
    assert not manager.agents[agent_id].is_active

    # Delete agent
    success = manager.delete_agent(agent_id)
    assert success
    assert agent_id not in manager.agents


def test_multiple_agents_coordination():
    """Test multiple agents can exist in the same world."""
    manager = AgentManager()
    world = manager.create_world(size=10)

    # Create multiple agents
    agent_ids = []
    for i in range(3):
        agent_id = manager.create_agent("explorer", f"Explorer {i}")
        agent_ids.append(agent_id)
        manager.start_agent(agent_id)

    # All agents should be active
    for agent_id in agent_ids:
        assert manager.agents[agent_id].is_active

    # Each agent should have unique position in the world
    positions = set()
    for agent_id in agent_ids:
        # Check position in the world (grid agent position)
        if agent_id in world.agents:
            grid_agent = world.agents[agent_id]
            pos = (grid_agent.position.x, grid_agent.position.y)
            assert pos not in positions, (
                f"Agents should have unique positions, but {agent_id} has duplicate position {pos}"
            )
            positions.add(pos)

    # Clean up
    for agent_id in agent_ids:
        manager.stop_agent(agent_id)
        manager.delete_agent(agent_id)


if __name__ == "__main__":
    print("Running agent-world integration tests...")
    try:
        test_explorer_agent_in_gridworld()
        print("✓ test_explorer_agent_in_gridworld passed")
    except Exception as e:
        print(f"✗ test_explorer_agent_in_gridworld failed: {e}")

    try:
        test_agent_manager_integration()
        print("✓ test_agent_manager_integration passed")
    except Exception as e:
        print(f"✗ test_agent_manager_integration failed: {e}")

    try:
        test_multiple_agents_coordination()
        print("✓ test_multiple_agents_coordination passed")
    except Exception as e:
        print(f"✗ test_multiple_agents_coordination failed: {e}")

    print("All tests completed.")
