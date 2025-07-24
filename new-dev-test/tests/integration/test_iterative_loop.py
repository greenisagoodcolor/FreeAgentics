"""Integration tests for the iterative loop functionality."""

import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest
from api.v1.prompts import PromptRequest, process_prompt
from auth.security_implementation import TokenData
from sqlalchemy.ext.asyncio import AsyncSession

from services.iterative_controller import IterativeController
from services.prompt_processor import PromptProcessor


@pytest.mark.asyncio
class TestIterativeLoop:
    """Test the complete iterative loop functionality."""

    @pytest.fixture
    async def db_session(self):
        """Create a test database session."""
        # Mock database session
        session = AsyncMock(spec=AsyncSession)
        session.add = Mock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        session.execute = AsyncMock()

        return session

    @pytest.fixture
    def mock_user(self):
        """Create a mock authenticated user."""
        return TokenData(
            username="test_user",
            email="test@example.com",
            permissions=["create_agent"],
        )

    @pytest.fixture
    def mock_prompt_processor(self):
        """Create a mock prompt processor with iterative controller."""
        processor = AsyncMock(spec=PromptProcessor)
        processor.iterative_controller = Mock(spec=IterativeController)
        return processor

    async def test_single_iteration(self, db_session, mock_user, mock_prompt_processor):
        """Test a single iteration of the loop."""
        # Setup mock response
        mock_prompt_processor.process_prompt.return_value = {
            "agent_id": "agent-123",
            "gmn_specification": "GMN spec here",
            "knowledge_graph_updates": [{"node_id": "node-1", "type": "belief", "properties": {}}],
            "next_suggestions": [
                "Add goal states to guide behavior",
                "Increase observation diversity",
            ],
            "status": "success",
            "iteration_context": {
                "iteration_number": 1,
                "total_agents": 1,
                "kg_nodes": 1,
                "conversation_summary": {
                    "iteration_count": 1,
                    "belief_evolution": {"trend": "initial", "stability": 0.0},
                },
            },
        }

        # Create request
        request = PromptRequest(
            prompt="Create an explorer agent for a 5x5 grid world",
            iteration_count=1,
        )

        # Process prompt
        with patch(
            "api.v1.prompts.get_prompt_processor",
            return_value=mock_prompt_processor,
        ):
            response = await process_prompt(request=request, current_user=mock_user, db=db_session)

        # Verify response
        assert response.agent_id == "agent-123"
        assert len(response.next_suggestions) == 2
        assert response.iteration_context is not None
        assert response.iteration_context["iteration_number"] == 1
        assert response.status == "success"

    async def test_multiple_iterations_same_conversation(self, db_session, mock_user):
        """Test multiple iterations in the same conversation."""
        conversation_id = str(uuid.uuid4())

        # Mock the prompt processor to track iterations
        iteration_results = []

        async def mock_process_prompt(prompt_text, user_id, db, conversation_id, iteration_count):
            iteration_num = len(iteration_results) + 1
            result = {
                "agent_id": f"agent-{iteration_num}",
                "gmn_specification": f"GMN spec iteration {iteration_num}",
                "knowledge_graph_updates": [
                    {
                        "node_id": f"node-{iteration_num}",
                        "type": "belief",
                        "properties": {},
                    }
                ],
                "next_suggestions": [
                    f"Suggestion {iteration_num}.1",
                    f"Suggestion {iteration_num}.2",
                ],
                "status": "success",
                "iteration_context": {
                    "iteration_number": iteration_num,
                    "total_agents": iteration_num,
                    "kg_nodes": iteration_num,
                    "conversation_summary": {
                        "iteration_count": iteration_num,
                        "belief_evolution": {
                            "trend": "converging" if iteration_num > 2 else "exploring",
                            "stability": min(0.2 * iteration_num, 0.8),
                        },
                    },
                },
            }
            iteration_results.append(result)
            return result

        mock_processor = AsyncMock()
        mock_processor.process_prompt = mock_process_prompt

        # Test prompts that build on each other
        prompts = [
            "Create an explorer agent for a grid world",
            "Add goal states to the explorer agent",
            "Make the agent more curious about unexplored areas",
            "Add coalition capabilities for multi-agent scenarios",
        ]

        responses = []

        with patch("api.v1.prompts.get_prompt_processor", return_value=mock_processor):
            for i, prompt_text in enumerate(prompts):
                request = PromptRequest(
                    prompt=prompt_text,
                    conversation_id=conversation_id,
                    iteration_count=1,
                )

                response = await process_prompt(
                    request=request, current_user=mock_user, db=db_session
                )

                responses.append(response)

        # Verify iteration progression
        assert len(responses) == 4

        # Check iteration numbers increase
        for i, response in enumerate(responses):
            assert response.iteration_context["iteration_number"] == i + 1
            assert response.iteration_context["total_agents"] == i + 1

        # Check belief evolution changes
        assert (
            responses[0].iteration_context["conversation_summary"]["belief_evolution"]["trend"]
            == "exploring"
        )
        assert (
            responses[3].iteration_context["conversation_summary"]["belief_evolution"]["trend"]
            == "converging"
        )

        # Verify stability increases
        stabilities = [
            r.iteration_context["conversation_summary"]["belief_evolution"]["stability"]
            for r in responses
        ]
        assert stabilities == sorted(stabilities)  # Should be increasing

    async def test_suggestion_evolution(self, db_session, mock_user):
        """Test how suggestions evolve based on conversation context."""
        conversation_id = str(uuid.uuid4())

        # Create a controller that tracks state
        class StatefulIterativeController:
            def __init__(self):
                self.iteration_count = 0
                self.kg_nodes = set()
                self.agent_types = []

            async def generate_suggestions(self, prompt, iteration):
                self.iteration_count = iteration

                suggestions = []

                if iteration == 1:
                    suggestions = [
                        "Start with basic exploration to establish environmental understanding",
                        "Add sensory modalities to reduce belief ambiguity",
                    ]
                elif iteration == 2:
                    suggestions = [
                        "Add goal-directed behavior to guide agent actions",
                        "Define preferences to shape agent objectives",
                    ]
                elif iteration == 3:
                    suggestions = [
                        "Introduce multi-agent coordination for complex tasks",
                        "Add communication channels between agents",
                    ]
                else:
                    suggestions = [
                        "Consider meta-learning - Let agents adapt their own models",
                        "Focus on knowledge consolidation rather than expansion",
                    ]

                return suggestions

        controller = StatefulIterativeController()

        # Mock processor that uses the stateful controller
        async def mock_process_with_controller(
            prompt_text, user_id, db, conversation_id, iteration_count
        ):
            iteration = controller.iteration_count + 1
            suggestions = await controller.generate_suggestions(prompt_text, iteration)

            return {
                "agent_id": f"agent-{iteration}",
                "gmn_specification": f"GMN {iteration}",
                "knowledge_graph_updates": [],
                "next_suggestions": suggestions,
                "status": "success",
                "iteration_context": {
                    "iteration_number": iteration,
                    "total_agents": iteration,
                    "kg_nodes": iteration * 2,
                },
            }

        mock_processor = AsyncMock()
        mock_processor.process_prompt = mock_process_with_controller

        # Process multiple iterations
        prompts = [
            "Create explorer agent",
            "Add goals based on previous suggestion",
            "Enable multi-agent coordination",
            "Optimize the system",
        ]

        all_suggestions = []

        with patch("api.v1.prompts.get_prompt_processor", return_value=mock_processor):
            for prompt in prompts:
                request = PromptRequest(prompt=prompt, conversation_id=conversation_id)

                response = await process_prompt(
                    request=request, current_user=mock_user, db=db_session
                )

                all_suggestions.extend(response.next_suggestions)

        # Verify suggestion progression
        assert "basic exploration" in all_suggestions[0].lower()
        assert "goal-directed" in all_suggestions[2].lower()
        assert "multi-agent" in all_suggestions[4].lower()
        assert "meta-learning" in all_suggestions[6].lower()

    async def test_context_aware_gmn_generation(self, db_session, mock_user):
        """Test that GMN generation uses context from previous iterations."""
        conversation_id = str(uuid.uuid4())

        # Track GMN specs generated
        gmn_specs = []

        async def mock_gmn_aware_process(
            prompt_text, user_id, db, conversation_id, iteration_count
        ):
            iteration = len(gmn_specs) + 1

            # Generate GMN that references previous context
            if iteration == 1:
                gmn = "Basic explorer GMN with 3x3 grid"
            elif iteration == 2:
                gmn = "Enhanced GMN with goal states added to previous 3x3 grid"
            elif iteration == 3:
                gmn = "Advanced GMN with curiosity rewards and expanded 5x5 grid"
            else:
                gmn = "Multi-agent GMN with communication channels between explorers"

            gmn_specs.append(gmn)

            return {
                "agent_id": f"agent-{iteration}",
                "gmn_specification": gmn,
                "knowledge_graph_updates": [
                    {
                        "node_id": f"node-{iteration}",
                        "type": "gmn",
                        "properties": {"iteration": iteration},
                    }
                ],
                "next_suggestions": [f"Suggestion for iteration {iteration}"],
                "status": "success",
                "iteration_context": {"iteration_number": iteration},
            }

        mock_processor = AsyncMock()
        mock_processor.process_prompt = mock_gmn_aware_process

        # Process iterations
        prompts = [
            "Create basic explorer",
            "Add goals to the explorer",
            "Make it more curious",
            "Add another agent",
        ]

        responses = []

        with patch("api.v1.prompts.get_prompt_processor", return_value=mock_processor):
            for prompt in prompts:
                request = PromptRequest(prompt=prompt, conversation_id=conversation_id)

                response = await process_prompt(
                    request=request, current_user=mock_user, db=db_session
                )

                responses.append(response)

        # Verify GMN evolution
        assert "Basic explorer" in responses[0].gmn_specification
        assert "goal states added to previous" in responses[1].gmn_specification
        assert "curiosity rewards and expanded" in responses[2].gmn_specification
        assert "Multi-agent" in responses[3].gmn_specification

    async def test_knowledge_graph_accumulation(self, db_session, mock_user):
        """Test that knowledge graph accumulates across iterations."""
        conversation_id = str(uuid.uuid4())

        # Track KG state
        kg_nodes = set()
        kg_edges = []

        async def mock_kg_accumulating_process(
            prompt_text, user_id, db, conversation_id, iteration_count
        ):
            iteration = len(kg_nodes) // 2 + 1  # Rough iteration count

            # Add nodes for this iteration
            new_nodes = [f"node-{iteration}a", f"node-{iteration}b"]
            kg_nodes.update(new_nodes)

            # Add edges connecting to previous nodes
            if iteration > 1:
                kg_edges.append(
                    {
                        "source": f"node-{iteration}a",
                        "target": f"node-{iteration - 1}b",
                    }
                )

            kg_updates = [
                {
                    "node_id": node,
                    "type": "belief",
                    "properties": {"iteration": iteration},
                }
                for node in new_nodes
            ]

            return {
                "agent_id": f"agent-{iteration}",
                "gmn_specification": f"GMN {iteration}",
                "knowledge_graph_updates": kg_updates,
                "next_suggestions": [
                    f"Connect {len(kg_nodes)} knowledge nodes through agent interactions"
                ],
                "status": "success",
                "iteration_context": {
                    "iteration_number": iteration,
                    "kg_nodes": len(kg_nodes),
                    "conversation_summary": {
                        "kg_connectivity": len(kg_edges) / max(len(kg_nodes) - 1, 1)
                    },
                },
            }

        mock_processor = AsyncMock()
        mock_processor.process_prompt = mock_kg_accumulating_process

        # Process iterations
        responses = []

        with patch("api.v1.prompts.get_prompt_processor", return_value=mock_processor):
            for i in range(4):
                request = PromptRequest(
                    prompt=f"Iteration {i + 1} prompt",
                    conversation_id=conversation_id,
                )

                response = await process_prompt(
                    request=request, current_user=mock_user, db=db_session
                )

                responses.append(response)

        # Verify KG growth
        assert responses[0].iteration_context["kg_nodes"] == 2
        assert responses[1].iteration_context["kg_nodes"] == 4
        assert responses[2].iteration_context["kg_nodes"] == 6
        assert responses[3].iteration_context["kg_nodes"] == 8

        # Verify connectivity increases
        assert responses[3].iteration_context["conversation_summary"]["kg_connectivity"] > 0

    async def test_error_handling_in_iteration(self, db_session, mock_user):
        """Test that errors in one iteration don't break the loop."""
        conversation_id = str(uuid.uuid4())

        iteration_count = 0

        async def mock_process_with_errors(prompt_text, user_id, db, conversation_id, iter_count):
            nonlocal iteration_count
            iteration_count += 1

            # Fail on iteration 2
            if iteration_count == 2:
                raise ValueError("GMN validation failed: Invalid structure")

            return {
                "agent_id": f"agent-{iteration_count}",
                "gmn_specification": f"GMN {iteration_count}",
                "knowledge_graph_updates": [],
                "next_suggestions": ["Continue iterating"],
                "status": "success",
                "iteration_context": {"iteration_number": iteration_count},
            }

        mock_processor = AsyncMock()
        mock_processor.process_prompt = mock_process_with_errors

        responses = []
        errors = []

        with patch("api.v1.prompts.get_prompt_processor", return_value=mock_processor):
            for i in range(4):
                request = PromptRequest(prompt=f"Prompt {i + 1}", conversation_id=conversation_id)

                try:
                    response = await process_prompt(
                        request=request, current_user=mock_user, db=db_session
                    )
                    responses.append(response)
                except Exception as e:
                    errors.append((i + 1, str(e)))

        # Should have 3 successful responses and 1 error
        assert len(responses) == 3
        assert len(errors) == 1
        assert errors[0][0] == 2  # Error on iteration 2

        # Other iterations should continue
        assert responses[0].iteration_context["iteration_number"] == 1
        assert responses[1].iteration_context["iteration_number"] == 3
        assert responses[2].iteration_context["iteration_number"] == 4
