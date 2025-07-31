"""
Agent Conversation Repository

This module implements the Repository Pattern for agent conversation database operations
as specified in Task 39.4. It provides clean interfaces for conversation, message,
and agent participation management following Uncle Bob's separation of concerns.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, desc, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from api.v1.schemas.agent_conversation_schemas import (
    ConversationQueryParams,
    ConversationStatusEnum,
    MessageQueryParams,
)
from database.models import (
    Agent,
    AgentConversationMessage,
    AgentConversationSession,
    ConversationStatus,
    agent_conversation_association,
)


class AgentConversationRepository:
    """Repository for agent conversation session operations."""

    def __init__(self, db: Session):
        self.db = db

    async def create_conversation(
        self,
        prompt: str,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        max_turns: int = 5,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> AgentConversationSession:
        """Create a new agent conversation session."""

        conversation = AgentConversationSession(
            prompt=prompt,
            title=title,
            description=description,
            max_turns=max_turns,
            llm_provider=llm_provider,
            llm_model=llm_model,
            config=config or {},
            user_id=user_id,
            status=ConversationStatus.PENDING,
        )

        try:
            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)
            return conversation
        except SQLAlchemyError as e:
            self.db.rollback()
            raise Exception(f"Failed to create conversation: {str(e)}")

    async def get_conversation_by_id(
        self,
        conversation_id: UUID,
        include_agents: bool = False,
        include_messages: bool = False,
    ) -> Optional[AgentConversationSession]:
        """Get conversation by ID with optional related data."""

        query = self.db.query(AgentConversationSession).filter(
            AgentConversationSession.id == conversation_id
        )

        if include_agents:
            query = query.options(selectinload(AgentConversationSession.agents))

        if include_messages:
            query = query.options(selectinload(AgentConversationSession.messages))

        return query.first()

    async def get_conversations(
        self,
        query_params: ConversationQueryParams,
    ) -> tuple[List[AgentConversationSession], int]:
        """Get conversations with filtering and pagination."""

        query = self.db.query(AgentConversationSession)

        # Apply filters
        if query_params.status:
            query = query.filter(AgentConversationSession.status == query_params.status.value)

        if query_params.user_id:
            query = query.filter(AgentConversationSession.user_id == query_params.user_id)

        if query_params.created_after:
            query = query.filter(AgentConversationSession.created_at >= query_params.created_after)

        if query_params.created_before:
            query = query.filter(AgentConversationSession.created_at <= query_params.created_before)

        # Get total count for pagination
        total_count = query.count()

        # Apply pagination and ordering
        conversations = (
            query.order_by(desc(AgentConversationSession.created_at))
            .offset((query_params.page - 1) * query_params.page_size)
            .limit(query_params.page_size)
            .all()
        )

        return conversations, total_count

    async def update_conversation_status(
        self,
        conversation_id: UUID,
        status: ConversationStatusEnum,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> Optional[AgentConversationSession]:
        """Update conversation status and timestamps."""

        conversation = await self.get_conversation_by_id(conversation_id)
        if not conversation:
            return None

        try:
            conversation.status = ConversationStatus(status.value)

            if started_at:
                conversation.started_at = started_at
            elif status == ConversationStatusEnum.ACTIVE and not conversation.started_at:
                conversation.started_at = datetime.now()

            if completed_at:
                conversation.completed_at = completed_at
            elif status in [
                ConversationStatusEnum.COMPLETED,
                ConversationStatusEnum.FAILED,
                ConversationStatusEnum.CANCELLED,
            ]:
                conversation.completed_at = datetime.now()

            self.db.commit()
            self.db.refresh(conversation)
            return conversation
        except SQLAlchemyError as e:
            self.db.rollback()
            raise Exception(f"Failed to update conversation status: {str(e)}")

    async def add_agent_to_conversation(
        self,
        conversation_id: UUID,
        agent_id: UUID,
        role: str = "participant",
    ) -> bool:
        """Add an agent to a conversation."""

        try:
            # Check if agent is already in conversation
            existing = (
                self.db.query(agent_conversation_association)
                .filter(
                    and_(
                        agent_conversation_association.c.agent_id == agent_id,
                        agent_conversation_association.c.conversation_id == conversation_id,
                    )
                )
                .first()
            )

            if existing:
                return False  # Agent already in conversation

            # Insert new association
            self.db.execute(
                agent_conversation_association.insert().values(
                    agent_id=agent_id,
                    conversation_id=conversation_id,
                    role=role,
                    joined_at=datetime.now(),
                    message_count=0,
                )
            )

            # Update conversation agent count
            conversation = await self.get_conversation_by_id(conversation_id)
            if conversation:
                conversation.agent_count = (
                    self.db.query(agent_conversation_association)
                    .filter(agent_conversation_association.c.conversation_id == conversation_id)
                    .count()
                )

            self.db.commit()
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            raise Exception(f"Failed to add agent to conversation: {str(e)}")

    async def remove_agent_from_conversation(
        self,
        conversation_id: UUID,
        agent_id: UUID,
    ) -> bool:
        """Remove an agent from a conversation."""

        try:
            # Update left_at timestamp
            result = (
                self.db.query(agent_conversation_association)
                .filter(
                    and_(
                        agent_conversation_association.c.agent_id == agent_id,
                        agent_conversation_association.c.conversation_id == conversation_id,
                    )
                )
                .update({"left_at": datetime.now()})
            )

            if result == 0:
                return False  # Agent not found in conversation

            # Update conversation agent count (only count active participants)
            conversation = await self.get_conversation_by_id(conversation_id)
            if conversation:
                conversation.agent_count = (
                    self.db.query(agent_conversation_association)
                    .filter(
                        and_(
                            agent_conversation_association.c.conversation_id == conversation_id,
                            agent_conversation_association.c.left_at.is_(None),
                        )
                    )
                    .count()
                )

            self.db.commit()
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            raise Exception(f"Failed to remove agent from conversation: {str(e)}")

    async def delete_conversation(self, conversation_id: UUID) -> bool:
        """Delete a conversation and all related data."""

        conversation = await self.get_conversation_by_id(conversation_id)
        if not conversation:
            return False

        try:
            # SQLAlchemy will cascade delete messages and associations
            self.db.delete(conversation)
            self.db.commit()
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            raise Exception(f"Failed to delete conversation: {str(e)}")


class AgentConversationMessageRepository:
    """Repository for agent conversation message operations."""

    def __init__(self, db: Session):
        self.db = db

    async def create_message(
        self,
        conversation_id: UUID,
        agent_id: UUID,
        content: str,
        role: str = "assistant",
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
        processing_time_ms: Optional[int] = None,
    ) -> AgentConversationMessage:
        """Create a new conversation message."""

        try:
            # Get next message order for this conversation
            max_order = (
                self.db.query(func.max(AgentConversationMessage.message_order))
                .filter(AgentConversationMessage.conversation_id == conversation_id)
                .scalar()
            ) or 0

            # Get current turn number
            conversation = (
                self.db.query(AgentConversationSession)
                .filter(AgentConversationSession.id == conversation_id)
                .first()
            )

            if not conversation:
                raise Exception("Conversation not found")

            message = AgentConversationMessage(
                conversation_id=conversation_id,
                agent_id=agent_id,
                content=content,
                message_order=max_order + 1,
                turn_number=conversation.current_turn + 1,
                role=role,
                message_type=message_type,
                message_metadata=metadata or {},
                processing_time_ms=processing_time_ms,
                is_processed=True,
            )

            self.db.add(message)

            # Update conversation message count and current turn
            conversation.message_count += 1
            conversation.current_turn = message.turn_number

            # Update agent message count in association table
            self.db.execute(
                agent_conversation_association.update()
                .where(
                    and_(
                        agent_conversation_association.c.agent_id == agent_id,
                        agent_conversation_association.c.conversation_id == conversation_id,
                    )
                )
                .values(message_count=agent_conversation_association.c.message_count + 1)
            )

            self.db.commit()
            self.db.refresh(message)
            return message
        except SQLAlchemyError as e:
            self.db.rollback()
            raise Exception(f"Failed to create message: {str(e)}")

    async def get_message_by_id(self, message_id: UUID) -> Optional[AgentConversationMessage]:
        """Get message by ID."""

        return (
            self.db.query(AgentConversationMessage)
            .filter(AgentConversationMessage.id == message_id)
            .first()
        )

    async def get_conversation_messages(
        self,
        conversation_id: UUID,
        query_params: MessageQueryParams,
    ) -> tuple[List[AgentConversationMessage], int]:
        """Get messages for a conversation with filtering and pagination."""

        query = self.db.query(AgentConversationMessage).filter(
            AgentConversationMessage.conversation_id == conversation_id
        )

        # Apply filters
        if query_params.agent_id:
            query = query.filter(AgentConversationMessage.agent_id == query_params.agent_id)

        if query_params.message_type:
            query = query.filter(
                AgentConversationMessage.message_type == query_params.message_type.value
            )

        if query_params.role:
            query = query.filter(AgentConversationMessage.role == query_params.role.value)

        if query_params.turn_number:
            query = query.filter(AgentConversationMessage.turn_number == query_params.turn_number)

        # Get total count
        total_count = query.count()

        # Apply pagination and ordering
        messages = (
            query.order_by(AgentConversationMessage.message_order)
            .offset((query_params.page - 1) * query_params.page_size)
            .limit(query_params.page_size)
            .all()
        )

        return messages, total_count

    async def update_message_processing_status(
        self,
        message_id: UUID,
        is_processed: bool,
        processing_time_ms: Optional[int] = None,
    ) -> Optional[AgentConversationMessage]:
        """Update message processing status."""

        message = await self.get_message_by_id(message_id)
        if not message:
            return None

        try:
            message.is_processed = is_processed
            if processing_time_ms is not None:
                message.processing_time_ms = processing_time_ms

            self.db.commit()
            self.db.refresh(message)
            return message
        except SQLAlchemyError as e:
            self.db.rollback()
            raise Exception(f"Failed to update message processing status: {str(e)}")

    async def delete_message(self, message_id: UUID) -> bool:
        """Delete a message."""

        message = await self.get_message_by_id(message_id)
        if not message:
            return False

        try:
            # Update conversation message count
            conversation = (
                self.db.query(AgentConversationSession)
                .filter(AgentConversationSession.id == message.conversation_id)
                .first()
            )

            if conversation:
                conversation.message_count = max(0, conversation.message_count - 1)

            # Update agent message count in association table
            self.db.execute(
                agent_conversation_association.update()
                .where(
                    and_(
                        agent_conversation_association.c.agent_id == message.agent_id,
                        agent_conversation_association.c.conversation_id == message.conversation_id,
                    )
                )
                .values(
                    message_count=func.greatest(
                        0, agent_conversation_association.c.message_count - 1
                    )
                )
            )

            self.db.delete(message)
            self.db.commit()
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            raise Exception(f"Failed to delete message: {str(e)}")


class AgentConversationAnalyticsRepository:
    """Repository for conversation analytics and metrics."""

    def __init__(self, db: Session):
        self.db = db

    async def get_conversation_metrics(self, conversation_id: UUID) -> Optional[Dict[str, Any]]:
        """Get analytics metrics for a conversation."""

        conversation = (
            self.db.query(AgentConversationSession)
            .filter(AgentConversationSession.id == conversation_id)
            .first()
        )

        if not conversation:
            return None

        # Get agent participation data
        agent_participation = (
            self.db.query(
                agent_conversation_association.c.agent_id,
                Agent.name.label("agent_name"),
                agent_conversation_association.c.message_count,
                agent_conversation_association.c.role,
            )
            .join(Agent, Agent.id == agent_conversation_association.c.agent_id)
            .filter(agent_conversation_association.c.conversation_id == conversation_id)
            .all()
        )

        # Get processing time statistics
        processing_stats = (
            self.db.query(
                func.avg(AgentConversationMessage.processing_time_ms).label("avg_processing_time"),
                func.min(AgentConversationMessage.processing_time_ms).label("min_processing_time"),
                func.max(AgentConversationMessage.processing_time_ms).label("max_processing_time"),
            )
            .filter(
                and_(
                    AgentConversationMessage.conversation_id == conversation_id,
                    AgentConversationMessage.processing_time_ms.isnot(None),
                )
            )
            .first()
        )

        # Calculate conversation duration
        duration_seconds = None
        if conversation.started_at and conversation.completed_at:
            duration_seconds = int(
                (conversation.completed_at - conversation.started_at).total_seconds()
            )

        # Build turn distribution
        turn_distribution = dict(
            self.db.query(
                AgentConversationMessage.turn_number,
                func.count(AgentConversationMessage.id),
            )
            .filter(AgentConversationMessage.conversation_id == conversation_id)
            .group_by(AgentConversationMessage.turn_number)
            .all()
        )

        return {
            "conversation_id": str(conversation_id),
            "total_messages": conversation.message_count,
            "agent_participation": {
                str(ap.agent_id): {
                    "name": ap.agent_name,
                    "message_count": ap.message_count,
                    "role": ap.role,
                }
                for ap in agent_participation
            },
            "average_processing_time_ms": float(processing_stats.avg_processing_time)
            if processing_stats.avg_processing_time
            else None,
            "min_processing_time_ms": processing_stats.min_processing_time,
            "max_processing_time_ms": processing_stats.max_processing_time,
            "conversation_duration_seconds": duration_seconds,
            "turn_distribution": turn_distribution,
            "status": conversation.status.value
            if hasattr(conversation.status, "value")
            else str(conversation.status),
            "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
            "started_at": conversation.started_at.isoformat() if conversation.started_at else None,
            "completed_at": conversation.completed_at.isoformat()
            if conversation.completed_at
            else None,
        }

    async def get_user_conversation_summary(self, user_id: str) -> Dict[str, Any]:
        """Get conversation summary statistics for a user."""

        total_conversations = (
            self.db.query(func.count(AgentConversationSession.id))
            .filter(AgentConversationSession.user_id == user_id)
            .scalar()
        )

        status_breakdown = dict(
            self.db.query(
                AgentConversationSession.status,
                func.count(AgentConversationSession.id),
            )
            .filter(AgentConversationSession.user_id == user_id)
            .group_by(AgentConversationSession.status)
            .all()
        )

        total_messages = (
            self.db.query(func.sum(AgentConversationSession.message_count))
            .filter(AgentConversationSession.user_id == user_id)
            .scalar()
        ) or 0

        return {
            "user_id": user_id,
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "status_breakdown": {
                status.value if hasattr(status, "value") else str(status): count
                for status, count in status_breakdown.items()
            },
        }
