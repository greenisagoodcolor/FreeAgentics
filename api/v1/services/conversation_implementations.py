"""
Conversation Implementation Services

This module provides concrete implementations of the conversation interfaces,
including response generation, turn control, and event publishing.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from api.v1.models.agent_conversation import (
    AgentRole,
    ConversationAggregate,
    ConversationContext,
    ConversationEvent,
    ConversationTurnDomain,
    IAgentResponseGenerator,
    IConversationEventPublisher,
    IConversationRepository,
    ITurnController,
)
from inference.llm.provider_factory import LLMProviderFactory
from inference.llm.provider_interface import GenerationRequest

logger = logging.getLogger(__name__)


class LLMAgentResponseGenerator(IAgentResponseGenerator):
    """Agent response generator using LLM providers."""

    def __init__(self, llm_factory: Optional[LLMProviderFactory] = None):
        """Initialize with LLM provider factory."""
        self.llm_factory = llm_factory or LLMProviderFactory()

    async def generate_response(
        self, agent: AgentRole, context: ConversationContext, timeout_seconds: int = 30
    ) -> str:
        """Generate agent response using LLM provider."""
        try:
            # Get provider manager (assuming a user_id - in real implementation this would come from context)
            provider_manager = self.llm_factory.create_from_config(user_id="system")
            healthy_providers = provider_manager.registry.get_healthy_providers()

            if not healthy_providers:
                # Fallback to mock response if no providers available
                return self._generate_mock_response(agent, context)

            # Create contextual prompt for the agent
            agent_prompt = context.create_agent_prompt(agent)

            # Prepare messages in the format expected by LLM providers
            messages = [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": agent_prompt},
            ]

            # Create generation request
            generation_request = GenerationRequest(
                messages=messages,
                model="gpt-3.5-turbo",  # Default model, should be configurable
                temperature=0.8,  # Higher temperature for varied agent responses
                max_tokens=200,  # Reasonable limit for conversation turns
            )

            # Generate response with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(provider_manager.generate_with_fallback, generation_request),
                timeout=timeout_seconds,
            )

            # Extract text content from response
            if hasattr(response, "text"):
                content = response.text
            elif hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # Clean up the response
            content = content.strip()

            # Remove agent name prefix if present
            if content.startswith(f"{agent.name}:"):
                content = content[len(f"{agent.name}:") :].strip()

            logger.info(f"Generated response for agent {agent.name}: {content[:100]}...")
            return content

        except asyncio.TimeoutError:
            logger.warning(f"Response generation timed out for agent {agent.name}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate response for agent {agent.name}: {e}")
            # Fallback to mock response on error
            return self._generate_mock_response(agent, context)

    def _generate_mock_response(self, agent: AgentRole, context: ConversationContext) -> str:
        """Generate a mock response for testing/fallback purposes."""
        # Simple template-based responses based on agent role
        role_responses = {
            "advocate": f"As {agent.name}, I support this direction and see great potential in what we're discussing about {context.topic}.",
            "critic": f"As {agent.name}, I have some concerns about this approach to {context.topic} that we should consider carefully.",
            "analyst": f"As {agent.name}, let me analyze the key aspects of {context.topic} based on the evidence we have.",
            "creative": f"As {agent.name}, I'd like to suggest an innovative approach to {context.topic} that might work well.",
            "moderator": f"As {agent.name}, let me summarize where we are on {context.topic} and suggest our next steps.",
        }

        # Get response template or use generic one
        template = role_responses.get(
            agent.role.lower(),
            f"As {agent.name}, I'd like to contribute my perspective on {context.topic}.",
        )

        # Add turn-specific variation
        turn_variations = [
            "This is an important point to consider.",
            "I think we should explore this further.",
            "Let me add another perspective here.",
            "Building on what's been said already...",
            "Here's how I see this situation...",
        ]

        turn_number = context.current_turn_number
        variation = turn_variations[turn_number % len(turn_variations)]

        return f"{template} {variation}"


class DefaultTurnController(ITurnController):
    """Default implementation of turn controller."""

    def __init__(self, response_generator: IAgentResponseGenerator):
        """Initialize with response generator."""
        self.response_generator = response_generator

    async def execute_turn(
        self, turn: ConversationTurnDomain, agent: AgentRole, context: ConversationContext
    ) -> ConversationTurnDomain:
        """Execute a single conversation turn."""
        logger.info(f"Executing turn {turn.turn_number} for agent {agent.name}")

        try:
            # Mark turn as started
            turn.mark_started()

            # Generate agent response
            response = await self.response_generator.generate_response(
                agent=agent,
                context=context,
                timeout_seconds=30,  # Should be configurable
            )

            # Mark turn as completed with response
            turn.mark_completed(response)

            logger.info(f"Successfully completed turn {turn.turn_number} for agent {agent.name}")
            return turn

        except Exception as e:
            # Mark turn as failed
            turn.mark_failed(str(e))
            logger.error(f"Turn {turn.turn_number} failed for agent {agent.name}: {e}")
            return turn


class WebSocketEventPublisher(IConversationEventPublisher):
    """Event publisher that broadcasts to WebSocket clients."""

    def __init__(self):
        """Initialize WebSocket event publisher."""
        pass

    async def publish_event(self, event: ConversationEvent) -> None:
        """Publish conversation event via WebSocket."""
        try:
            # Import WebSocket manager (avoiding circular imports)
            from api.v1.websocket import manager

            # Format event for WebSocket broadcast
            websocket_message = {
                "type": "conversation_event",
                "timestamp": event.timestamp.isoformat(),
                "data": {
                    "event_id": event.event_id,
                    "conversation_id": event.conversation_id,
                    "event_type": event.event_type,
                    "data": event.data,
                },
            }

            # Broadcast to all connected clients
            # In a production system, you might want to filter by user permissions
            await manager.broadcast(websocket_message, event_type="conversation_event")

            logger.debug(
                f"Published event {event.event_type} for conversation {event.conversation_id}"
            )

        except Exception as e:
            logger.warning(f"Failed to publish WebSocket event: {e}")
            # Don't raise - event publishing failures shouldn't break conversation flow


class InMemoryConversationRepository(IConversationRepository):
    """In-memory repository for conversation persistence."""

    def __init__(self):
        """Initialize in-memory repository."""
        self._conversations: Dict[str, ConversationAggregate] = {}
        self._user_conversations: Dict[str, List[str]] = {}

    async def save_conversation(self, conversation: ConversationAggregate) -> None:
        """Save conversation to memory."""
        self._conversations[conversation.conversation_id] = conversation

        # Update user conversation index
        user_conversations = self._user_conversations.get(conversation.user_id, [])
        if conversation.conversation_id not in user_conversations:
            user_conversations.append(conversation.conversation_id)
            self._user_conversations[conversation.user_id] = user_conversations

    async def get_conversation(self, conversation_id: str) -> Optional[ConversationAggregate]:
        """Retrieve conversation by ID."""
        return self._conversations.get(conversation_id)

    async def list_active_conversations(self, user_id: str) -> List[ConversationAggregate]:
        """List active conversations for a user."""
        conversation_ids = self._user_conversations.get(user_id, [])
        conversations = []

        for conv_id in conversation_ids:
            conversation = self._conversations.get(conv_id)
            if conversation and conversation.is_active:
                conversations.append(conversation)

        return conversations


# Database repository implementation would go here in a production system
class DatabaseConversationRepository(IConversationRepository):
    """Database-backed repository for conversation persistence."""

    def __init__(self, db_session_factory):
        """Initialize with database session factory."""
        self.db_session_factory = db_session_factory

    async def save_conversation(self, conversation: ConversationAggregate) -> None:
        """Save conversation to database."""
        # Implementation would use SQLAlchemy models to persist conversation state
        # For now, this is a placeholder
        logger.info(f"Would save conversation {conversation.conversation_id} to database")

    async def get_conversation(self, conversation_id: str) -> Optional[ConversationAggregate]:
        """Retrieve conversation from database."""
        # Implementation would query database and reconstruct aggregate
        logger.info(f"Would load conversation {conversation_id} from database")
        return None

    async def list_active_conversations(self, user_id: str) -> List[ConversationAggregate]:
        """List active conversations from database."""
        # Implementation would query for user's active conversations
        logger.info(f"Would list active conversations for user {user_id} from database")
        return []


class CompositeEventPublisher(IConversationEventPublisher):
    """Composite event publisher that publishes to multiple channels."""

    def __init__(self, publishers: List[IConversationEventPublisher]):
        """Initialize with list of event publishers."""
        self.publishers = publishers

    async def publish_event(self, event: ConversationEvent) -> None:
        """Publish event to all configured publishers."""
        # Publish to all publishers concurrently
        tasks = [publisher.publish_event(event) for publisher in self.publishers]

        # Wait for all publishers, but don't fail if some publishers fail
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any publishing failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                publisher_name = type(self.publishers[i]).__name__
                logger.warning(f"Event publisher {publisher_name} failed: {result}")


# Factory functions for dependency injection


def create_default_response_generator() -> IAgentResponseGenerator:
    """Create default agent response generator."""
    return LLMAgentResponseGenerator()


def create_default_turn_controller(
    response_generator: Optional[IAgentResponseGenerator] = None,
) -> ITurnController:
    """Create default turn controller."""
    if response_generator is None:
        response_generator = create_default_response_generator()
    return DefaultTurnController(response_generator)


def create_default_event_publisher() -> IConversationEventPublisher:
    """Create default event publisher."""
    return WebSocketEventPublisher()


def create_default_repository() -> IConversationRepository:
    """Create default conversation repository."""
    return InMemoryConversationRepository()
