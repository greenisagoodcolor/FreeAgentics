"""Example tests demonstrating the test data management system.

This file shows how to use builders, factories, fixtures, and generators
to create comprehensive test scenarios.
"""

import pytest

from database.models import Agent
from database.models import AgentStatus as DBAgentStatus
from tests.fixtures import (
    AgentBuilder,
    AgentFactory,
    CoalitionBuilder,
    CoalitionFactory,
    KnowledgeGraphFactory,
    generate_agent_batch,
    generate_knowledge_graph,
    generate_performance_dataset,
)
from tests.fixtures.fixtures import (
    active_agent,
    coalition_with_agents,
    db_session,
    test_engine,
)
from tests.fixtures.schemas import (
    AgentStatus,
    CoalitionStatus,
    PerformanceTestConfigSchema,
)


class TestBuildersExample:
    """Examples of using builders for test data creation."""

    def test_simple_agent_builder(self):
        """Create a simple agent using builder."""
        agent = (
            AgentBuilder()
            .with_name("TestAgent001")
            .with_template("grid_world")
            .active()
            .build()
        )

        assert agent.name == "TestAgent001"
        assert agent.template == "grid_world"
        assert agent.status == AgentStatus.ACTIVE
        assert agent.id is not None  # Auto-generated UUID

    def test_complex_agent_builder(self):
        """Create agent with full configuration."""
        agent = (
            AgentBuilder()
            .with_name("ComplexAgent")
            .as_explorer()  # Pre-configured template
            .with_grid_world_config(grid_size=10, num_actions=5)
            .with_uniform_beliefs(num_states=10)
            .with_position(25.5, 30.2)
            .with_random_metrics()
            .with_exploration_parameters(exploration_rate=0.3)
            .with_inference_history(inference_count=100, total_steps=500)
            .build()
        )

        assert (
            agent.template == "grid_world"
        )  # Overridden by with_grid_world_config
        assert agent.position == [25.5, 30.2]
        assert agent.inference_count == 100
        assert agent.parameters.exploration_rate == 0.3
        assert len(agent.beliefs.state_beliefs) == 10

    def test_coalition_builder_with_objectives(self):
        """Create coalition with multiple objectives."""
        coalition = (
            CoalitionBuilder()
            .with_name("Strategic Alliance")
            .with_description("Multi-objective coalition")
            .as_resource_coalition()
            .with_objective(
                "explore_north",
                "Explore northern territories",
                priority="high",
            )
            .with_objective(
                "defend_base", "Defend home base", priority="critical"
            )
            .with_required_capabilities(
                "exploration", "defense", "communication"
            )
            .with_achieved_objectives("explore_north")
            .with_random_scores()
            .active()
            .build()
        )

        assert coalition.name == "Strategic Alliance"
        assert len(coalition.objectives) == 3  # resource_opt + 2 custom
        assert coalition.objectives["explore_north"].status == "completed"
        assert coalition.objectives["defend_base"].priority == "critical"
        assert "exploration" in coalition.required_capabilities
        assert coalition.performance_score > 0


class TestFactoriesExample:
    """Examples of using factories with database persistence."""

    @pytest.mark.skip(reason="SQLAlchemy enum configuration issue")
    def test_agent_factory_create(self, db_session):
        """Create and persist an agent."""
        agent = AgentFactory.create(
            session=db_session,
            name="FactoryAgent",
            template="resource_collector",
            status=DBAgentStatus.ACTIVE,
        )

        # Verify persistence
        assert agent.id is not None
        db_agent = (
            db_session.query(Agent).filter_by(name="FactoryAgent").first()
        )
        assert db_agent is not None
        assert db_agent.template == "resource_collector"

    @pytest.mark.skip(reason="SQLAlchemy enum configuration issue")
    def test_agent_factory_batch(self, db_session):
        """Create multiple agents efficiently."""
        agents = AgentFactory.create_batch(
            session=db_session,
            count=20,
            template="grid_world",
            status=DBAgentStatus.ACTIVE,
            distribute_positions=True,
            position_bounds={"min": [0, 0], "max": [50, 50]},
        )

        assert len(agents) == 20
        assert all(a.status == AgentStatus.ACTIVE for a in agents)

        # Check position distribution
        positions = [a.position for a in agents if a.position]
        assert len(positions) > 0
        assert all(0 <= p[0] <= 50 and 0 <= p[1] <= 50 for p in positions)

    @pytest.mark.skip(reason="SQLAlchemy enum configuration issue")
    def test_coalition_with_agents(self, db_session):
        """Create coalition with member agents."""
        coalition, agents = CoalitionFactory.create_with_agents(
            session=db_session,
            num_agents=7,
            agent_template="grid_world",
            name="TestCoalition",
            status=CoalitionStatus.ACTIVE,
        )

        assert coalition.name == "TestCoalition"
        assert len(agents) == 7
        assert len(coalition.agents) == 7

        # Verify roles are assigned correctly
        agent_roles = db_session.execute(
            "SELECT role FROM agent_coalition WHERE coalition_id = :cid",
            {"cid": coalition.id},
        ).fetchall()

        roles = [r[0] for r in agent_roles]
        assert roles.count("leader") == 1  # First agent is leader
        assert roles.count("coordinator") >= 1  # Some coordinators
        assert roles.count("member") >= 1  # Rest are members

    def test_knowledge_graph_factory(self, db_session):
        """Create a knowledge graph."""
        result = KnowledgeGraphFactory.create_knowledge_graph(
            session=db_session,
            num_nodes=15,
            connectivity=0.2,
            node_types=["concept", "entity", "observation"],
            edge_types=["relates_to", "causes", "supports"],
        )

        assert len(result["nodes"]) == 15
        assert result["statistics"]["num_edges"] > 0
        assert result["statistics"]["avg_degree"] > 0

        # Verify node types distribution
        node_types = [n.type for n in result["nodes"]]
        assert "concept" in node_types
        assert "entity" in node_types
        assert "observation" in node_types


class TestFixturesExample:
    """Examples of using pytest fixtures."""

    @pytest.mark.skip(reason="SQLAlchemy enum configuration issue")
    def test_agent_fixtures(
        self, active_agent, resource_collector_agent, explorer_agent
    ):
        """Test with various agent fixtures."""
        # Active agent has full configuration
        assert active_agent.status == AgentStatus.ACTIVE
        assert active_agent.beliefs is not None
        assert active_agent.position is not None

        # Template-specific agents
        assert resource_collector_agent.template == "resource_collector"
        assert "collection_radius" in resource_collector_agent.parameters

        assert explorer_agent.template == "explorer"
        assert explorer_agent.parameters.get("exploration_rate", 0) > 0

    @pytest.mark.skip(reason="SQLAlchemy enum configuration issue")
    def test_coalition_fixtures(
        self, coalition_with_agents, resource_coalition
    ):
        """Test coalition fixtures."""
        # Coalition with agents
        assert len(coalition_with_agents.agents) > 0
        assert coalition_with_agents.status == CoalitionStatus.ACTIVE

        # Specialized coalition
        assert resource_coalition.name == "ResourceOptimizers"
        assert any(
            "resource" in obj["description"].lower()
            for obj in resource_coalition.objectives.values()
        )

    def test_knowledge_graph_fixture(self, knowledge_graph_fixture):
        """Test knowledge graph fixture."""
        assert knowledge_graph_fixture["statistics"]["num_nodes"] == 10
        assert knowledge_graph_fixture["statistics"]["connectivity"] > 0

        # Check graph structure
        nodes = knowledge_graph_fixture["nodes"]
        edges = knowledge_graph_fixture["edges"]

        node_ids = {n.id for n in nodes}
        for edge in edges:
            assert edge.source_id in node_ids
            assert edge.target_id in node_ids

    def test_multi_agent_scenario(self, multi_agent_scenario):
        """Test complex scenario fixture."""
        scenario = multi_agent_scenario

        assert len(scenario["agents"]) == 9  # 3 of each type
        assert len(scenario["coalitions"]) == 3

        # Check agent distribution
        assert len(scenario["agent_types"]["resource_collectors"]) == 3
        assert len(scenario["agent_types"]["explorers"]) == 3
        assert len(scenario["agent_types"]["coordinators"]) == 3

        # Check coalitions have appropriate agents
        for coalition in scenario["coalitions"]:
            assert len(coalition.agents) > 0


class TestGeneratorsExample:
    """Examples of using data generators."""

    def test_generate_agent_batch(self):
        """Generate agents without database."""
        agents = generate_agent_batch(
            count=50, template="grid_world", status=AgentStatus.ACTIVE
        )

        assert len(agents) == 50
        assert all(a.template == "grid_world" for a in agents)
        assert all(a.status == AgentStatus.ACTIVE for a in agents)

    def test_generate_knowledge_graph(self):
        """Generate different types of knowledge graphs."""
        # Random graph
        random_graph = generate_knowledge_graph(
            num_nodes=30, connectivity=0.15, graph_type="random"
        )

        assert len(random_graph["nodes"]) == 30
        assert random_graph["properties"]["actual_connectivity"] < 0.2

        # Scale-free graph
        scale_free_graph = generate_knowledge_graph(
            num_nodes=50, graph_type="scale_free"
        )

        assert len(scale_free_graph["nodes"]) == 50
        # Scale-free graphs have high-degree hubs
        assert (
            scale_free_graph["properties"]["max_degree"]
            > scale_free_graph["properties"]["avg_degree"] * 2
        )

    def test_performance_dataset(self):
        """Generate complete performance test dataset."""
        config = PerformanceTestConfigSchema(
            num_agents=100,
            num_coalitions=10,
            num_knowledge_nodes=200,
            knowledge_graph_connectivity=0.05,
            seed=42,  # Reproducible results
        )

        dataset = generate_performance_dataset(config)

        assert len(dataset["agents"]) == 100
        assert len(dataset["coalitions"]) == 10
        assert dataset["knowledge_graph"] is not None
        assert len(dataset["knowledge_graph"]["nodes"]) == 200

        # Check timing information
        assert "agent_generation" in dataset["timing"]
        assert "coalition_generation" in dataset["timing"]
        assert "knowledge_generation" in dataset["timing"]

        # Check statistics
        assert dataset["statistics"]["agent_count"] == 100
        assert "agent_templates" in dataset["statistics"]


class TestValidationExample:
    """Examples of schema validation."""

    def test_invalid_agent_data(self):
        """Test schema validation catches errors."""
        with pytest.raises(ValueError):
            # Name too short
            AgentBuilder().with_name("").build()

        with pytest.raises(ValueError):
            # Invalid template
            AgentBuilder().with_template("invalid_template").build()

        with pytest.raises(ValueError):
            # Invalid position dimensions
            AgentBuilder().with_position(1.0, 2.0, 3.0, 4.0).build()

    def test_probability_normalization(self):
        """Test automatic probability normalization."""
        agent = (
            AgentBuilder()
            .with_beliefs(
                state_beliefs={"s1": 0.3, "s2": 0.3, "s3": 0.3},
                confidence=0.8,  # Sum = 0.9
            )
            .build()
        )

        # Should be normalized to sum to 1.0
        total = sum(agent.beliefs.state_beliefs.values())
        assert abs(total - 1.0) < 0.01

    def test_coalition_capability_validation(self):
        """Test coalition capability validation."""
        with pytest.raises(ValueError):
            # Invalid capability
            CoalitionBuilder().with_required_capabilities(
                "invalid_capability"
            ).build()


class TestPerformanceExample:
    """Examples of performance-oriented test data generation."""

    def test_large_scale_generation(self, db_session):
        """Test generating large datasets efficiently."""
        from tests.fixtures import PerformanceDataFactory
        from tests.fixtures.schemas import PerformanceTestConfigSchema

        config = PerformanceTestConfigSchema(
            num_agents=500,
            num_coalitions=20,
            num_knowledge_nodes=1000,
            batch_size=100,  # Process in batches
        )

        factory = PerformanceDataFactory()
        results = factory.create_performance_scenario(db_session, config)

        # Verify counts
        assert len(results["agents"]) == 500
        assert len(results["coalitions"]) == 20
        assert results["knowledge_graph"]["statistics"]["num_nodes"] == 1000

        # Check performance metrics
        total_time = results["statistics"]["total_creation_time"]
        assert total_time < 30  # Should complete within 30 seconds

        # Check memory efficiency (batch processing)
        assert results["timing"]["agent_creation"] < 10
        assert results["timing"]["coalition_creation"] < 5
        assert results["timing"]["knowledge_creation"] < 15


if __name__ == "__main__":
    # Run specific test class
    pytest.main([__file__, "-v", "-k", "TestBuildersExample"])
