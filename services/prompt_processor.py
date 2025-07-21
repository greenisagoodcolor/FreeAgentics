"""Prompt processing service that orchestrates the prompt → agent → KG pipeline.

This service handles the core business logic for converting natural language
prompts into active inference agents and updating the knowledge graph.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.agent_manager import AgentManager
from agents.pymdp_adapter import PyMDPCompatibilityAdapter
from database.models import Agent, AgentStatus
from database.prompt_models import (
    Conversation,
    ConversationStatus,
    KnowledgeGraphUpdate,
    Prompt,
    PromptStatus,
)
from inference.active.gmn_parser import GMNParser
from knowledge_graph.graph_engine import KnowledgeGraph
from services.iterative_controller import IterativeController

logger = logging.getLogger(__name__)


class PromptProcessor:
    """Orchestrates the prompt processing pipeline."""

    def __init__(
        self,
        gmn_generator,  # IGMNGenerator instance
        gmn_parser: GMNParser,
        agent_factory,  # IAgentFactory instance
        agent_manager: AgentManager,
        knowledge_graph: KnowledgeGraph,
        belief_kg_bridge,  # IBeliefKGBridge instance
        pymdp_adapter: PyMDPCompatibilityAdapter,
        iterative_controller: Optional[IterativeController] = None,
        websocket_callback: Optional[Callable] = None,
    ):
        """Initialize the prompt processor with required services."""
        self.gmn_generator = gmn_generator
        self.gmn_parser = gmn_parser
        self.agent_factory = agent_factory
        self.agent_manager = agent_manager
        self.knowledge_graph = knowledge_graph
        self.belief_kg_bridge = belief_kg_bridge
        self.pymdp_adapter = pymdp_adapter
        self.websocket_callback = websocket_callback

        # Initialize iterative controller if not provided
        self.iterative_controller = iterative_controller
        if self.iterative_controller is None:
            self.iterative_controller = IterativeController(
                knowledge_graph=self.knowledge_graph,
                belief_kg_bridge=self.belief_kg_bridge,
                pymdp_adapter=self.pymdp_adapter,
            )

    async def process_prompt(
        self,
        prompt_text: str,
        user_id: str,
        db: AsyncSession,
        conversation_id: Optional[str] = None,
        iteration_count: int = 1,
    ) -> Dict[str, Any]:
        """Process a prompt through the full pipeline.

        Args:
            prompt_text: The user's prompt
            user_id: ID of the user making the request
            db: Database session
            conversation_id: Optional existing conversation ID
            iteration_count: Number of refinement iterations

        Returns:
            Response containing agent_id, GMN spec, KG updates, and suggestions

        Raises:
            ValueError: If GMN generation or validation fails
            RuntimeError: If agent creation fails
        """
        start_time = time.time()

        logger.info(f"Processing prompt from user {user_id}: {prompt_text[:100]}...")

        # Get or create conversation
        conversation = await self._get_or_create_conversation(
            db, user_id, conversation_id
        )

        # Create prompt record
        prompt_record = Prompt(
            conversation_id=conversation.id,
            prompt_text=prompt_text,
            iteration_count=iteration_count,
            status=PromptStatus.PROCESSING,
        )
        db.add(prompt_record)
        await db.flush()

        try:
            # Get conversation context from iterative controller
            conversation_context = (
                await self.iterative_controller.get_or_create_context(
                    str(conversation.id), db
                )
            )

            # Prepare iteration-specific context
            iteration_context = (
                await self.iterative_controller.prepare_iteration_context(
                    conversation_context, prompt_text
                )
            )

            # Send initial WebSocket update with iteration info
            await self._send_websocket_update(
                "pipeline_started",
                {
                    "prompt_id": str(prompt_record.id),
                    "prompt_text": prompt_text[:100] + "..."
                    if len(prompt_text) > 100
                    else prompt_text,
                    "user_id": user_id,
                    "conversation_id": conversation.id,
                    "stage": "initialization",
                    "total_stages": 6,
                    "iteration_number": iteration_context["iteration_number"],
                    "conversation_summary": iteration_context["conversation_summary"],
                },
            )

            # Step 1: Generate GMN from prompt with iteration context
            logger.info("Generating GMN specification from prompt...")
            await self._send_websocket_update(
                "pipeline_progress",
                {
                    "prompt_id": str(prompt_record.id),
                    "stage": "gmn_generation",
                    "stage_number": 1,
                    "message": "Generating GMN specification from natural language...",
                    "iteration_context": {
                        "number": iteration_context["iteration_number"],
                        "previous_suggestions": iteration_context[
                            "previous_suggestions"
                        ],
                    },
                },
            )

            # Update conversation context with iteration-specific constraints
            conversation.context.update(iteration_context)

            gmn_spec = await self._generate_gmn(
                prompt_text, conversation.context, iteration_count
            )
            prompt_record.gmn_specification = gmn_spec

            await self._send_websocket_update(
                "gmn_generated",
                {
                    "prompt_id": str(prompt_record.id),
                    "gmn_preview": gmn_spec[:200] + "..."
                    if len(gmn_spec) > 200
                    else gmn_spec,
                    "gmn_length": len(gmn_spec),
                },
            )

            # Step 2: Parse and validate GMN
            logger.info("Parsing and validating GMN specification...")
            await self._send_websocket_update(
                "pipeline_progress",
                {
                    "prompt_id": str(prompt_record.id),
                    "stage": "gmn_validation",
                    "stage_number": 2,
                    "message": "Parsing and validating GMN specification...",
                },
            )

            gmn_graph = self.gmn_parser.parse(gmn_spec)
            pymdp_model = self.gmn_parser.to_pymdp_model(gmn_graph)

            is_valid, errors = await self._validate_model(pymdp_model)
            if not is_valid:
                await self._send_websocket_update(
                    "validation_failed",
                    {
                        "prompt_id": str(prompt_record.id),
                        "errors": errors,
                        "stage": "gmn_validation",
                    },
                )
                raise ValueError(f"GMN validation failed: {', '.join(errors)}")

            await self._send_websocket_update(
                "validation_success",
                {
                    "prompt_id": str(prompt_record.id),
                    "model_dimensions": {
                        "num_states": pymdp_model.get("num_states", []),
                        "num_obs": pymdp_model.get("num_obs", []),
                        "num_actions": pymdp_model.get("num_controls", []),
                    },
                },
            )

            # Step 3: Create PyMDP agent
            logger.info("Creating PyMDP agent from model...")
            await self._send_websocket_update(
                "pipeline_progress",
                {
                    "prompt_id": str(prompt_record.id),
                    "stage": "agent_creation",
                    "stage_number": 3,
                    "message": "Creating active inference agent...",
                },
            )

            agent_id = str(uuid.uuid4())
            pymdp_agent = await self._create_agent(pymdp_model, agent_id, prompt_text)

            await self._send_websocket_update(
                "agent_created",
                {
                    "prompt_id": str(prompt_record.id),
                    "agent_id": agent_id,
                    "agent_type": self._infer_agent_type(prompt_text),
                },
            )

            # Step 4: Store agent in database
            logger.info("Storing agent in database...")
            await self._send_websocket_update(
                "pipeline_progress",
                {
                    "prompt_id": str(prompt_record.id),
                    "stage": "database_storage",
                    "stage_number": 4,
                    "message": "Storing agent in database...",
                },
            )

            db_agent = await self._store_agent(
                db, agent_id, gmn_spec, pymdp_model, prompt_text
            )
            prompt_record.agent_id = db_agent.id

            # Step 5: Update knowledge graph
            logger.info("Updating knowledge graph with agent beliefs...")
            await self._send_websocket_update(
                "pipeline_progress",
                {
                    "prompt_id": str(prompt_record.id),
                    "stage": "knowledge_graph_update",
                    "stage_number": 5,
                    "message": "Updating knowledge graph with agent beliefs...",
                },
            )

            kg_updates = await self._update_knowledge_graph(
                pymdp_agent, agent_id, prompt_record.id, db
            )

            await self._send_websocket_update(
                "knowledge_graph_updated",
                {
                    "prompt_id": str(prompt_record.id),
                    "updates_count": len(kg_updates),
                    "nodes_added": len([u for u in kg_updates if u.applied]),
                },
            )

            # Step 6: Generate next suggestions using iterative controller
            logger.info("Generating intelligent suggestions...")
            await self._send_websocket_update(
                "pipeline_progress",
                {
                    "prompt_id": str(prompt_record.id),
                    "stage": "suggestion_generation",
                    "stage_number": 6,
                    "message": "Generating intelligent context-aware suggestions...",
                },
            )

            # Extract current beliefs for context
            current_beliefs = await self.belief_kg_bridge.extract_beliefs(pymdp_agent)

            # Use iterative controller for intelligent suggestions
            suggestions = (
                await self.iterative_controller.generate_intelligent_suggestions(
                    agent_id,
                    pymdp_agent,
                    conversation_context,
                    current_beliefs,
                    db,
                )
            )

            # Update prompt record
            processing_time = (time.time() - start_time) * 1000
            prompt_record.status = PromptStatus.SUCCESS
            prompt_record.processed_at = datetime.utcnow()
            prompt_record.processing_time_ms = processing_time
            prompt_record.next_suggestions = suggestions
            prompt_record.response_data = {
                "model_dimensions": {
                    "num_states": pymdp_model.get("num_states", []),
                    "num_obs": pymdp_model.get("num_obs", []),
                    "num_actions": pymdp_model.get("num_actions", []),
                }
            }

            # Update conversation
            conversation.agent_ids = list(set(conversation.agent_ids + [agent_id]))
            conversation.updated_at = datetime.utcnow()

            # Update iterative controller context
            await self.iterative_controller.update_conversation_context(
                conversation_context,
                prompt_text,
                agent_id,
                gmn_spec,
                current_beliefs,
                kg_updates,
                suggestions,
            )

            await db.commit()

            logger.info(f"Successfully processed prompt in {processing_time:.2f}ms")

            # Send final success update with iteration info
            await self._send_websocket_update(
                "pipeline_completed",
                {
                    "prompt_id": str(prompt_record.id),
                    "agent_id": agent_id,
                    "processing_time_ms": processing_time,
                    "suggestions": suggestions,
                    "kg_updates_count": len(kg_updates),
                    "status": "success",
                    "iteration_number": conversation_context.iteration_count + 1,
                    "conversation_summary": conversation_context.get_context_summary(),
                },
            )

            return {
                "agent_id": agent_id,
                "gmn_specification": gmn_spec,
                "knowledge_graph_updates": self._format_kg_updates(kg_updates),
                "next_suggestions": suggestions,
                "status": "success",
                "iteration_context": {
                    "iteration_number": conversation_context.iteration_count,
                    "total_agents": len(set(conversation_context.agent_ids)),
                    "kg_nodes": len(conversation_context.kg_node_ids),
                    "conversation_summary": conversation_context.get_context_summary(),
                },
            }

        except ValueError as e:
            # GMN validation error
            logger.error(f"GMN validation error: {str(e)}")
            prompt_record.status = PromptStatus.FAILED
            prompt_record.errors = [str(e)]
            await db.commit()

            await self._send_websocket_update(
                "pipeline_failed",
                {
                    "prompt_id": str(prompt_record.id),
                    "stage": "gmn_validation",
                    "error": str(e),
                    "error_type": "validation_error",
                },
            )
            raise

        except RuntimeError as e:
            # Agent creation error
            logger.error(f"Agent creation error: {str(e)}")
            prompt_record.status = PromptStatus.FAILED
            prompt_record.errors = [str(e)]
            await db.commit()

            await self._send_websocket_update(
                "pipeline_failed",
                {
                    "prompt_id": str(prompt_record.id),
                    "stage": "agent_creation",
                    "error": str(e),
                    "error_type": "runtime_error",
                },
            )
            raise

        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error processing prompt: {str(e)}")
            prompt_record.status = PromptStatus.FAILED
            prompt_record.errors = [f"Unexpected error: {str(e)}"]
            await db.commit()

            await self._send_websocket_update(
                "pipeline_failed",
                {
                    "prompt_id": str(prompt_record.id),
                    "stage": "unknown",
                    "error": str(e),
                    "error_type": "unexpected_error",
                },
            )
            raise RuntimeError(f"Prompt processing failed: {str(e)}")

    async def _get_or_create_conversation(
        self, db: AsyncSession, user_id: str, conversation_id: Optional[str]
    ) -> Conversation:
        """Get existing conversation or create new one."""
        if conversation_id:
            result = await db.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )
            conversation = result.scalar_one_or_none()
            if conversation:
                return conversation

        # Create new conversation
        conversation = Conversation(user_id=user_id, status=ConversationStatus.ACTIVE)
        db.add(conversation)
        await db.flush()
        return conversation

    async def _generate_gmn(
        self, prompt_text: str, context: Dict[str, Any], iteration_count: int
    ) -> str:
        """Generate GMN specification from prompt."""
        # Extract agent type from prompt if possible
        agent_type = self._infer_agent_type(prompt_text)

        # Generate initial GMN
        gmn_spec = await self.gmn_generator.prompt_to_gmn(
            prompt_text,
            agent_type=agent_type,
            constraints=context.get("constraints", {}),
        )

        # Refine through iterations
        for i in range(iteration_count - 1):
            is_valid, errors = await self.gmn_generator.validate_gmn(gmn_spec)
            if is_valid:
                break

            feedback = f"Validation errors: {', '.join(errors)}"
            gmn_spec = await self.gmn_generator.refine_gmn(gmn_spec, feedback)

        return gmn_spec

    async def _validate_model(
        self, pymdp_model: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate PyMDP model before agent creation."""
        return await self.agent_factory.validate_model(pymdp_model)

    async def _create_agent(
        self, pymdp_model: Dict[str, Any], agent_id: str, prompt_text: str
    ):
        """Create PyMDP agent from model."""
        metadata = {
            "created_from_prompt": prompt_text[:200],
            "creation_time": datetime.utcnow().isoformat(),
        }

        return await self.agent_factory.create_from_gmn_model(
            pymdp_model, agent_id=agent_id, metadata=metadata
        )

    async def _store_agent(
        self,
        db: AsyncSession,
        agent_id: str,
        gmn_spec: str,
        pymdp_model: Dict[str, Any],
        prompt_text: str,
    ) -> Agent:
        """Store agent in database."""
        agent = Agent(
            id=agent_id,
            name=self._generate_agent_name(prompt_text),
            template=self._infer_agent_type(prompt_text),
            status=AgentStatus.ACTIVE,
            gmn_spec=gmn_spec,
            pymdp_config=pymdp_model,
            beliefs={},
            preferences=pymdp_model.get("C", {}),
        )
        db.add(agent)
        await db.flush()
        return agent

    async def _update_knowledge_graph(
        self, pymdp_agent, agent_id: str, prompt_id: str, db: AsyncSession
    ) -> List[KnowledgeGraphUpdate]:
        """Update knowledge graph with agent state."""
        try:
            # Update KG from agent
            await self.belief_kg_bridge.update_kg_from_agent(
                pymdp_agent, agent_id, self.knowledge_graph
            )

            # Record updates in database
            kg_updates = []

            # Extract belief state for node creation
            belief_state = await self.belief_kg_bridge.extract_beliefs(pymdp_agent)
            nodes = await self.belief_kg_bridge.belief_to_nodes(
                belief_state, agent_id, context={"prompt_id": prompt_id}
            )

            for node in nodes:
                kg_update = KnowledgeGraphUpdate(
                    prompt_id=prompt_id,
                    node_id=node.id,
                    node_type=node.type,
                    operation="add",
                    properties=node.properties,
                    applied=True,
                )
                db.add(kg_update)
                kg_updates.append(kg_update)

            await db.flush()
            return kg_updates

        except Exception as e:
            logger.warning(f"Knowledge graph update failed: {str(e)}")
            # Record failure but don't fail the whole operation
            kg_update = KnowledgeGraphUpdate(
                prompt_id=prompt_id,
                node_id="error",
                node_type="error",
                operation="failed",
                properties={},
                applied=False,
                error_message=str(e),
            )
            db.add(kg_update)
            await db.flush()
            return []

    async def _generate_suggestions(
        self, pymdp_agent, gmn_graph, context: Dict[str, Any]
    ) -> List[str]:
        """Generate next action suggestions based on agent state."""
        suggestions = []

        # Analyze current beliefs
        beliefs = self.pymdp_adapter.get_beliefs(pymdp_agent)
        entropy = self._calculate_entropy(beliefs)

        if entropy > 0.8:
            suggestions.append("Add more observations to reduce uncertainty")

        # Check for unexplored actions
        if hasattr(pymdp_agent, "action_hist") and len(pymdp_agent.action_hist) < 5:
            suggestions.append("Explore different action sequences")

        # Analyze GMN structure
        if "goal" not in [node.type for node in gmn_graph.nodes.values()]:
            suggestions.append("Add goal states to guide agent behavior")

        if "preference" not in [node.type for node in gmn_graph.nodes.values()]:
            suggestions.append("Define preferences to shape agent objectives")

        # Context-based suggestions
        if context.get("agent_count", 0) > 1:
            suggestions.append("Consider coalition formation with other agents")

        return suggestions[:3]  # Return top 3 suggestions

    def _infer_agent_type(self, prompt_text: str) -> str:
        """Infer agent type from prompt text."""
        prompt_lower = prompt_text.lower()

        if any(word in prompt_lower for word in ["explore", "discover", "search"]):
            return "explorer"
        elif any(word in prompt_lower for word in ["trade", "exchange", "negotiate"]):
            return "trader"
        elif any(word in prompt_lower for word in ["coordinate", "manage", "organize"]):
            return "coordinator"
        else:
            return "general"

    def _generate_agent_name(self, prompt_text: str) -> str:
        """Generate agent name from prompt."""
        # Extract key words
        words = prompt_text.split()[:5]
        name = "_".join(w.lower() for w in words if len(w) > 3)
        return f"agent_{name[:30]}"

    def _format_kg_updates(
        self, kg_updates: List[KnowledgeGraphUpdate]
    ) -> List[Dict[str, Any]]:
        """Format knowledge graph updates for response."""
        return [
            {
                "node_id": update.node_id,
                "type": update.node_type,
                "properties": update.properties,
            }
            for update in kg_updates
            if update.applied
        ]

    def _calculate_entropy(self, beliefs: List) -> float:
        """Calculate belief entropy as measure of uncertainty."""
        # Simplified entropy calculation
        import numpy as np

        total_entropy = 0.0
        for belief in beliefs:
            if isinstance(belief, np.ndarray):
                # Normalize to probabilities
                probs = belief / belief.sum()
                # Calculate entropy
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                total_entropy += entropy

        # Normalize by number of factors
        return total_entropy / len(beliefs) if beliefs else 0.0

    async def _send_websocket_update(self, event_type: str, data: Dict[str, Any]):
        """Send WebSocket update if callback is available."""
        if self.websocket_callback:
            try:
                await self.websocket_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket update: {str(e)}")
