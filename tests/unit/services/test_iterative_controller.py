"""Unit tests for the iterative controller service."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from database.models import Agent
from database.prompt_models import (
    Conversation,
    ConversationStatus,
    KnowledgeGraphUpdate,
    Prompt,
    PromptStatus,
)
from services.iterative_controller import (
    ConversationContext,
    IterativeController,
)


class TestConversationContext:
    """Test the ConversationContext class."""

    def test_initialization(self):
        """Test context initialization."""
        context = ConversationContext("test-conv-id")

        assert context.conversation_id == "test-conv-id"
        assert context.iteration_count == 0
        assert context.agent_ids == []
        assert len(context.kg_node_ids) == 0
        assert context.belief_history == []
        assert context.suggestion_history == []
        assert context.prompt_history == []

    def test_add_iteration(self):
        """Test adding an iteration to context."""
        context = ConversationContext("test-conv-id")

        context.add_iteration(
            prompt="Create an explorer agent",
            agent_id="agent-1",
            gmn_spec="GMN spec here",
            beliefs={"state": [0.5, 0.5]},
            kg_nodes=["node-1", "node-2"],
            suggestions=["Add goals", "Increase exploration"],
        )

        assert context.iteration_count == 1
        assert context.agent_ids == ["agent-1"]
        assert context.kg_node_ids == {"node-1", "node-2"}
        assert len(context.belief_history) == 1
        assert context.belief_history[0] == {"state": [0.5, 0.5]}
        assert context.suggestion_history == [
            ["Add goals", "Increase exploration"]
        ]
        assert context.prompt_history == ["Create an explorer agent"]

    def test_multiple_iterations(self):
        """Test multiple iterations."""
        context = ConversationContext("test-conv-id")

        # First iteration
        context.add_iteration(
            prompt="Create an explorer",
            agent_id="agent-1",
            gmn_spec="GMN 1",
            beliefs={"state": [0.5, 0.5]},
            kg_nodes=["node-1", "node-2"],
            suggestions=["Add goals"],
        )

        # Second iteration
        context.add_iteration(
            prompt="Add goal states",
            agent_id="agent-2",
            gmn_spec="GMN 2",
            beliefs={"state": [0.3, 0.7], "goal": [1.0, 0.0]},
            kg_nodes=["node-2", "node-3"],  # node-2 repeated
            suggestions=["Increase complexity"],
        )

        assert context.iteration_count == 2
        assert context.agent_ids == ["agent-1", "agent-2"]
        assert context.kg_node_ids == {"node-1", "node-2", "node-3"}
        assert len(context.belief_history) == 2
        assert len(context.suggestion_history) == 2

    def test_context_summary(self):
        """Test context summary generation."""
        context = ConversationContext("test-conv-id")

        # Add some iterations
        context.add_iteration(
            prompt="Create explorer agent",
            agent_id="agent-1",
            gmn_spec="GMN 1",
            beliefs={"state": [0.5, 0.5]},
            kg_nodes=["node-1"],
            suggestions=["Add goals"],
        )

        context.add_iteration(
            prompt="Create another explorer",
            agent_id="agent-2",
            gmn_spec="GMN 2",
            beliefs={"state": [0.4, 0.6], "goal": [1.0]},
            kg_nodes=["node-2"],
            suggestions=["Add goals", "Explore more"],
        )

        summary = context.get_context_summary()

        assert summary["iteration_count"] == 2
        assert summary["total_agents"] == 2
        assert summary["kg_nodes"] == 2
        assert "belief_evolution" in summary
        assert "prompt_themes" in summary
        assert "suggestion_patterns" in summary

    def test_belief_evolution_analysis(self):
        """Test belief evolution analysis."""
        context = ConversationContext("test-conv-id")

        # Similar beliefs (converging)
        context.belief_history = [
            {"state": [0.5, 0.5], "goal": [1.0, 0.0]},
            {"state": [0.48, 0.52], "goal": [0.95, 0.05]},
            {"state": [0.49, 0.51], "goal": [0.98, 0.02]},
        ]

        summary = context.get_context_summary()
        evolution = summary["belief_evolution"]

        assert evolution["trend"] == "converging"
        assert evolution["stability"] > 0.5
        assert evolution["total_iterations"] == 3

    def test_theme_extraction(self):
        """Test prompt theme extraction."""
        context = ConversationContext("test-conv-id")

        context.prompt_history = [
            "Create an explorer agent for grid world",
            "Make the explorer more curious",
            "Add exploration rewards to the agent",
        ]

        summary = context.get_context_summary()
        themes = summary["prompt_themes"]

        # Should identify "explorer" and "agent" as common themes
        assert "explorer" in themes or "exploration" in themes
        assert "agent" in themes

    def test_suggestion_patterns(self):
        """Test suggestion pattern analysis."""
        context = ConversationContext("test-conv-id")

        context.suggestion_history = [
            ["Add goal states", "Increase exploration"],
            ["Add goal rewards", "Explore new areas"],
            ["Define preferences", "Add goal objectives"],
        ]

        summary = context.get_context_summary()
        patterns = summary["suggestion_patterns"]

        assert patterns["diversity"] > 0.5  # Reasonably diverse
        assert "goal" in patterns["common_themes"]
        assert patterns["total_suggestions"] == 6


class TestIterativeController:
    """Test the IterativeController class."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        return {
            "knowledge_graph": AsyncMock(),
            "belief_kg_bridge": AsyncMock(),
            "pymdp_adapter": Mock(),
        }

    @pytest.fixture
    def controller(self, mock_dependencies):
        """Create controller instance."""
        return IterativeController(
            knowledge_graph=mock_dependencies["knowledge_graph"],
            belief_kg_bridge=mock_dependencies["belief_kg_bridge"],
            pymdp_adapter=mock_dependencies["pymdp_adapter"],
        )

    @pytest.mark.asyncio
    async def test_get_or_create_context_new(self, controller):
        """Test creating new context."""
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = []

        context = await controller.get_or_create_context("conv-1", mock_db)

        assert context.conversation_id == "conv-1"
        assert context.iteration_count == 0
        assert "conv-1" in controller._conversation_contexts

    @pytest.mark.asyncio
    async def test_get_or_create_context_existing(self, controller):
        """Test getting existing context from cache."""
        # Pre-populate cache
        existing_context = ConversationContext("conv-1")
        existing_context.iteration_count = 3
        controller._conversation_contexts["conv-1"] = existing_context

        mock_db = AsyncMock()
        context = await controller.get_or_create_context("conv-1", mock_db)

        assert context is existing_context
        assert context.iteration_count == 3
        # Should not hit database
        mock_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_prepare_iteration_context(self, controller):
        """Test preparing iteration context."""
        context = ConversationContext("conv-1")
        context.add_iteration(
            prompt="Create explorer",
            agent_id="agent-1",
            gmn_spec="GMN",
            beliefs={"state": [0.5, 0.5]},
            kg_nodes=["node-1"],
            suggestions=["Add goals"],
        )

        # Mock KG context
        controller._get_kg_context = AsyncMock(
            return_value={"nodes": 1, "clusters": 1, "density": 0.0}
        )

        iteration_context = await controller.prepare_iteration_context(
            context, "Add goal states to explorer"
        )

        assert iteration_context["iteration_number"] == 2
        assert "conversation_summary" in iteration_context
        assert "kg_state" in iteration_context
        assert "prompt_analysis" in iteration_context
        assert "constraints" in iteration_context
        assert iteration_context["previous_suggestions"] == ["Add goals"]

    @pytest.mark.asyncio
    async def test_generate_intelligent_suggestions_new_conversation(
        self, controller
    ):
        """Test suggestion generation for new conversation."""
        context = ConversationContext("conv-1")
        mock_agent = Mock()
        mock_db = AsyncMock()

        # Mock methods
        controller._analyze_kg_connectivity = AsyncMock(
            return_value={
                "isolated_nodes": 0,
                "cluster_count": 1,
                "avg_connectivity": 0,
            }
        )
        controller._identify_capability_gaps = AsyncMock(return_value=[])

        suggestions = await controller.generate_intelligent_suggestions(
            "agent-1", mock_agent, context, {"state": [0.5, 0.5]}, mock_db
        )

        assert len(suggestions) > 0
        assert len(suggestions) <= 5
        # Should include exploration suggestion for new conversation
        assert any("exploration" in s.lower() for s in suggestions)

    @pytest.mark.asyncio
    async def test_generate_intelligent_suggestions_mature_conversation(
        self, controller
    ):
        """Test suggestions for mature conversation."""
        context = ConversationContext("conv-1")

        # Simulate 6 iterations
        for i in range(6):
            context.add_iteration(
                prompt=f"Iteration {i}",
                agent_id=f"agent-{i}",
                gmn_spec=f"GMN {i}",
                beliefs={"state": [0.5, 0.5]},
                kg_nodes=[f"node-{i}"],
                suggestions=["Same suggestion"],  # Low diversity
            )

        mock_agent = Mock()
        mock_db = AsyncMock()

        # Mock methods
        controller._analyze_kg_connectivity = AsyncMock(
            return_value={
                "isolated_nodes": 0,
                "cluster_count": 1,
                "avg_connectivity": 2,
            }
        )
        controller._identify_capability_gaps = AsyncMock(
            return_value=["learning"]
        )

        suggestions = await controller.generate_intelligent_suggestions(
            "agent-6", mock_agent, context, {"state": [0.5, 0.5]}, mock_db
        )

        # Should suggest meta-learning for mature conversation
        assert any(
            "meta-learning" in s.lower() or "adapt" in s.lower()
            for s in suggestions
        )
        # Should notice low diversity
        assert any("different approach" in s.lower() for s in suggestions)

    @pytest.mark.asyncio
    async def test_update_conversation_context(self, controller):
        """Test updating conversation context."""
        context = ConversationContext("conv-1")
        controller._conversation_contexts["conv-1"] = context

        kg_updates = [
            Mock(node_id="node-1", applied=True),
            Mock(node_id="node-2", applied=True),
            Mock(node_id="node-3", applied=False),  # Not applied
        ]

        await controller.update_conversation_context(
            context,
            "Create explorer",
            "agent-1",
            "GMN spec",
            {"state": [0.5, 0.5]},
            kg_updates,
            ["Suggestion 1", "Suggestion 2"],
        )

        assert context.iteration_count == 1
        assert context.agent_ids == ["agent-1"]
        assert context.kg_node_ids == {
            "node-1",
            "node-2",
        }  # Only applied nodes
        assert context.prompt_history == ["Create explorer"]

    def test_calculate_prompt_similarity(self, controller):
        """Test prompt similarity calculation."""
        prompt1 = "Create an explorer agent for grid world"
        prompt2 = "Create an explorer agent with goals"
        prompt3 = "Build a trader agent for market"

        # Similar prompts
        similarity1 = controller._calculate_prompt_similarity(prompt1, prompt2)
        assert similarity1 > 0.5

        # Different prompts
        similarity2 = controller._calculate_prompt_similarity(prompt1, prompt3)
        assert similarity2 < 0.5

    def test_identify_clusters(self, controller):
        """Test cluster identification in subgraph."""
        # Disconnected graph with 2 clusters
        subgraph = {
            "nodes": [
                {"id": "n1"},
                {"id": "n2"},
                {"id": "n3"},
                {"id": "n4"},
                {"id": "n5"},
            ],
            "edges": [
                {"source": "n1", "target": "n2"},
                {"source": "n2", "target": "n3"},
                # n4 and n5 form separate cluster
                {"source": "n4", "target": "n5"},
            ],
        }

        clusters = controller._identify_clusters(subgraph)

        assert len(clusters) == 2
        assert {"n1", "n2", "n3"} in clusters
        assert {"n4", "n5"} in clusters

    def test_generate_iteration_constraints(self, controller):
        """Test constraint generation based on context."""
        # Test high stability -> increase complexity
        context_summary = {
            "belief_evolution": {"stability": 0.9},
            "iteration_count": 3,
        }
        kg_context = {"density": 0.5}
        prompt_analysis = {"evolution": "refining"}

        constraints = controller._generate_iteration_constraints(
            context_summary, kg_context, prompt_analysis
        )

        assert constraints["maintain_consistency"] is True
        assert constraints["iteration_specific"]["increase_complexity"] is True
        assert (
            constraints["iteration_specific"]["preserve_core_structure"]
            is True
        )
        assert constraints["iteration_specific"]["focus"] == "optimization"

    @pytest.mark.asyncio
    async def test_analyze_kg_connectivity(self, controller):
        """Test KG connectivity analysis."""
        controller.knowledge_graph.get_subgraph = AsyncMock(
            return_value={
                "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
                "edges": [{"source": "n1", "target": "n2"}],  # n3 is isolated
            }
        )

        analysis = await controller._analyze_kg_connectivity(
            {"n1", "n2", "n3"}, "agent-1"
        )

        assert analysis["isolated_nodes"] == 1  # n3
        assert analysis["cluster_count"] == 2  # n1-n2 cluster and n3 alone
        assert analysis["avg_connectivity"] < 1.0

    def test_extract_prompt_themes(self, controller):
        """Test theme extraction from prompts."""
        prompts = [
            "Create an explorer agent to explore the environment",
            "Add learning capabilities to the explorer",
            "Make the agent learn from exploration",
        ]

        themes = controller._extract_prompt_themes(prompts)

        assert "exploration" in themes
        assert "learning" in themes

    @pytest.mark.asyncio
    async def test_identify_capability_gaps(self, controller):
        """Test capability gap identification."""
        mock_db = AsyncMock()

        # Mock agents with some capabilities
        mock_agents = [
            Mock(
                pymdp_config={
                    "C": [1, 0],  # Has preferences
                    "planning_horizon": 3,  # Has planning
                    "num_controls": [2],  # Has actions
                }
            ),
            Mock(
                pymdp_config={
                    "planning_horizon": 1,  # No planning
                    "num_controls": [3],  # Has actions
                }
            ),
        ]

        mock_db.execute.return_value.scalars.return_value.all.return_value = (
            mock_agents
        )

        gaps = await controller._identify_capability_gaps(
            ["agent-1", "agent-2"], mock_db
        )

        # Should identify missing perception and learning
        assert "perception" in gaps
        assert "learning" in gaps
        assert len(gaps) <= 3  # Limited to top 3
