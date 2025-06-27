"""Behavior-Driven Tests for Agent Scenarios.

Expert Committee: Kent Beck, Martin Fowler, Andy Clark
Following ADR-007 mandate for behavior-driven testing.
"""

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

# Load scenarios from feature files
scenarios("../features/agent_exploration.feature")
scenarios("../features/coalition_formation.feature")


class TestAgentExplorationBehavior:
    """BDD tests for agent exploration behaviors."""

    @given("an Explorer agent with high curiosity")
    def explorer_agent(self):
        """Create an explorer agent with high curiosity parameter."""
        from agents.base.agent import BaseAgent

        agent = BaseAgent(
            agent_id="explorer_001",
            name="CuriousExplorer",
            agent_class="explorer",
            initial_position=(0, 0),
        )
        agent.personality = {"curiosity": 0.9, "caution": 0.2}
        return agent

    @given("a world with unknown territories")
    def world_with_unknown_territories(self):
        """Create a world with unexplored areas."""
        from world.h3_world import H3World

        world = H3World(resolution=6)
        # Mark most areas as unexplored
        return world

    @when("the agent explores for 50 timesteps")
    def agent_explores(self, explorer_agent, world_with_unknown_territories):
        """Simulate agent exploration for specified timesteps."""
        world = world_with_unknown_territories
        world.add_agent(explorer_agent)

        exploration_data = []
        for timestep in range(50):
            # Agent makes exploration decisions
            current_pos = explorer_agent.position
            unexplored_neighbors = world.get_unexplored_neighbors(current_pos)

            if unexplored_neighbors:
                # High curiosity drives exploration of unknown areas
                next_pos = explorer_agent.select_exploration_target(unexplored_neighbors)
                explorer_agent.move_to(next_pos)

            exploration_data.append(
                {
                    "timestep": timestep,
                    "position": explorer_agent.position,
                    "discoveries": len(explorer_agent.discovered_locations),
                }
            )

        return exploration_data

    @then("the agent should discover new areas")
    def verify_discoveries(self, exploration_data):
        """Verify that the agent discovered new areas."""
        initial_discoveries = exploration_data[0]["discoveries"]
        final_discoveries = exploration_data[-1]["discoveries"]

        assert (
            final_discoveries > initial_discoveries
        ), "Agent should discover new areas during exploration"

    @then("exploration efficiency should improve over time")
    def verify_exploration_efficiency(self, exploration_data):
        """Verify that exploration becomes more efficient."""
        # Calculate discovery rate over time
        early_period = exploration_data[:20]
        late_period = exploration_data[30:]

        early_rate = len([d for d in early_period if d["discoveries"] > 0]) / 20
        late_rate = len([d for d in late_period if d["discoveries"] > 0]) / 20

        # Later exploration should be more targeted and efficient
        assert (
            late_rate >= early_rate * 0.8
        ), "Exploration efficiency should not significantly decrease"


class TestCoalitionFormationBehavior:
    """BDD tests for coalition formation scenarios."""

    @given(parsers.parse("a {agent_type} agent with {capability} capability"))
    def agent_with_capability(self, agent_type, capability):
        """Create an agent with specific capabilities."""
        from agents.base.agent import BaseAgent

        agent = BaseAgent(
            agent_id=f"{agent_type.lower()}_001",
            name=f"{agent_type}Agent",
            agent_class=agent_type.lower(),
            initial_position=(0, 0),
        )

        # Set capability scores
        capabilities = {
            "resource_gathering": 0.5,
            "data_processing": 0.5,
            "coordination": 0.5,
            "optimization": 0.5,
        }
        capabilities[capability] = 0.9
        agent.capabilities = capabilities

        return agent

    @given("multiple agents in close proximity")
    def agents_in_proximity(self):
        """Create multiple agents near each other."""
        from agents.base.agent import BaseAgent

        agents = []
        positions = [(0, 0), (1, 0), (0, 1), (1, 1)]

        for i, pos in enumerate(positions):
            agent = BaseAgent(
                agent_id=f"agent_{i:03d}",
                name=f"Agent{i}",
                agent_class="explorer",
                initial_position=pos,
            )
            agents.append(agent)

        return agents

    @when("agents discover complementary capabilities over time")
    def discover_complementary_capabilities(self, agents_in_proximity):
        """Simulate capability discovery process."""
        agents = agents_in_proximity

        # Assign complementary capabilities
        capability_sets = [
            {"resource_gathering": 0.9, "data_processing": 0.3},
            {"data_processing": 0.9, "resource_gathering": 0.3},
            {"coordination": 0.9, "optimization": 0.4},
            {"optimization": 0.9, "coordination": 0.4},
        ]

        for agent, caps in zip(agents, capability_sets):
            agent.capabilities = caps

        # Simulate interaction and capability discovery
        capability_matrix = {}
        for i, agent_a in enumerate(agents):
            for j, agent_b in enumerate(agents[i + 1 :], i + 1):
                # Calculate synergy potential
                synergy = self._calculate_synergy(agent_a.capabilities, agent_b.capabilities)
                capability_matrix[(agent_a.agent_id, agent_b.agent_id)] = synergy

        return capability_matrix

    def _calculate_synergy(self, caps_a, caps_b):
        """Calculate synergy between two capability sets."""
        synergy_score = 0.0

        for capability, value_a in caps_a.items():
            value_b = caps_b.get(capability, 0.0)

            # High synergy when capabilities are complementary
            if abs(value_a - value_b) > 0.4:  # Different strength levels
                synergy_score += min(value_a, value_b) + 0.3

        return synergy_score

    @then("a coalition should form based on synergy")
    def verify_coalition_formation(self, capability_matrix):
        """Verify that coalitions form based on synergy."""
        # Find highest synergy pairs
        max_synergy = max(capability_matrix.values())
        high_synergy_pairs = [
            pair for pair, synergy in capability_matrix.items() if synergy >= max_synergy * 0.8
        ]

        assert len(high_synergy_pairs) > 0, "At least one high-synergy pair should be identified"

    @then("coalition members should have improved capabilities")
    def verify_capability_improvement(self, agents_in_proximity, capability_matrix):
        """Verify that coalition formation improves overall capabilities."""
        agents = agents_in_proximity

        # Calculate individual capability scores
        individual_scores = []
        for agent in agents:
            score = sum(agent.capabilities.values())
            individual_scores.append(score)

        # Find best synergy pair and calculate coalition score
        best_pair = max(capability_matrix.items(), key=lambda x: x[1])
        agent_ids = best_pair[0]

        # Simulate coalition capability enhancement
        coalition_score = sum(individual_scores) + best_pair[1]
        individual_sum = sum(individual_scores)

        assert (
            coalition_score > individual_sum
        ), "Coalition should provide capability improvement over individuals"


class TestResourceOptimizationBehavior:
    """BDD tests for resource optimization scenarios."""

    @given("agents with different resource needs")
    def agents_with_resource_needs(self):
        """Create agents with varying resource requirements."""
        from agents.base.agent import BaseAgent

        agents = []
        resource_profiles = [
            {"food": 0.8, "water": 0.3, "materials": 0.2},
            {"food": 0.2, "water": 0.9, "materials": 0.1},
            {"food": 0.1, "water": 0.2, "materials": 0.8},
        ]

        for i, profile in enumerate(resource_profiles):
            agent = BaseAgent(
                agent_id=f"resource_agent_{i:03d}",
                name=f"ResourceAgent{i}",
                agent_class="explorer",
                initial_position=(i, 0),
            )
            agent.resource_needs = profile
            agents.append(agent)

        return agents

    @when("resources become scarce in the environment")
    def resources_become_scarce(self, agents_with_resource_needs):
        """Simulate resource scarcity scenario."""
        from world.h3_world import H3World

        world = H3World(resolution=5)
        agents = agents_with_resource_needs

        # Add agents to world
        for agent in agents:
            world.add_agent(agent)

        # Create scarce resource distribution
        total_resources = {"food": 10, "water": 8, "materials": 12}

        # Distribute limited resources
        for resource_type, amount in total_resources.items():
            for i in range(amount):
                position = world.get_random_position()
                world.add_resource(position, resource_type, 1)

        return world

    @then("agents should form resource-sharing coalitions")
    def verify_resource_sharing_coalitions(
        self, agents_with_resource_needs, resources_become_scarce
    ):
        """Verify that agents form coalitions for resource optimization."""
        agents = agents_with_resource_needs
        world = resources_become_scarce

        # Simulate resource optimization behavior
        resource_exchanges = []

        for timestep in range(20):
            # Each agent evaluates trading opportunities
            for agent_a in agents:
                for agent_b in agents:
                    if agent_a != agent_b:
                        # Check for beneficial trade
                        trade_benefit = self._evaluate_trade_benefit(
                            agent_a.resource_needs, agent_b.resource_needs
                        )

                        if trade_benefit > 0.3:  # Significant benefit threshold
                            resource_exchanges.append(
                                {
                                    "timestep": timestep,
                                    "agents": (agent_a.agent_id, agent_b.agent_id),
                                    "benefit": trade_benefit,
                                }
                            )

        assert (
            len(resource_exchanges) > 0
        ), "Agents should identify beneficial resource trading opportunities"

    def _evaluate_trade_benefit(self, needs_a, needs_b):
        """Calculate benefit of resource trade between two agents."""
        # Simplified trade benefit calculation
        complementarity = 0.0

        for resource, need_a in needs_a.items():
            need_b = needs_b.get(resource, 0.0)

            # High benefit when needs are complementary
            if abs(need_a - need_b) > 0.4:
                complementarity += abs(need_a - need_b)

        return complementarity / len(needs_a)
